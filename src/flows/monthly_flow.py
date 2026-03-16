"""
Flow mensual de Prefect — Evaluación + Reentrenamiento condicional

Corre el día 1 de cada mes a las 06:00 hora Santiago.

Qué hace:
    1. Evalúa las métricas de los últimos 30 días comparando
       predicciones vs datos reales en la DB
    2. Detecta si hay degradación usando los umbrales del metadata.json
    3. Si hay degradación → reentrenar con datos completos desde Open-Meteo
    4. Si el nuevo modelo mejora → subirlo a HF Hub y actualizarlo en memoria
    5. Loggear todo en la tabla retraining_log de SQLite

Criterio de degradación (AMBAS condiciones deben cumplirse):
    - RMSE actual > baseline_rmse * (1 + rmse_max_pct_increase)
    - Recall lluvia actual < recall_min

Criterio para aceptar el nuevo modelo:
    - RMSE del nuevo modelo <= RMSE actual del período evaluado
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple

#from prefect import flow, task, get_run_logger
import logging

from sklearn.metrics import (
    mean_squared_error, recall_score
)

from src.data.ingestion import fetch_weather_data
from src.data.preprocessing import build_features
from src.utils.config import (
    LATITUDE, LONGITUDE, TIMEZONE,
    DAILY_VARIABLES, HOURLY_VARIABLES, HOURLY_AGGREGATIONS,
    CLF_RAIN_THRESHOLD,
)
from src.models.two_stage_model import TwoStagePredictor
from src.storage.database import (
    init_db, get_actuals_for_evaluation, count_actuals, log_evaluation
)
from src.storage.hf_model import ModelRegistry
from src.flows.daily_flow import get_registry

logger = logging.getLogger(__name__)

# Mínimo de días con datos reales para hacer una evaluación confiable
MIN_DAYS_FOR_EVALUATION = 15


# ──────────────────────────────────────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────────────────────────────────────

@task(name="evaluate-recent-performance")
def evaluate_recent_performance(
    period_start: date,
    period_end: date,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Calcula RMSE y Recall de lluvia de los últimos 30 días
    comparando predicciones vs reales en la DB.

    Returns:
        Tupla (rmse, recall_lluvia, n_days_with_data)
        rmse y recall son None si no hay suficientes datos.
    """
    # log = get_run_logger()
    log = logging.getLogger(__name__)


    df = get_actuals_for_evaluation(period_start, period_end)

    if len(df) == 0:
        log.warning("⚠️ No hay datos reales en el período evaluado")
        return None, None, 0

    # Solo evaluar días que tienen tanto actual como predicción
    df_eval = df.dropna(subset=["mm_actual", "mm_predicted"])
    n_days  = len(df_eval)

    log.info(f"Días con datos completos: {n_days} / {len(df)}")

    if n_days < MIN_DAYS_FOR_EVALUATION:
        log.warning(
            f"⚠️ Solo {n_days} días con datos — mínimo requerido: {MIN_DAYS_FOR_EVALUATION}. "
            f"Evaluación omitida."
        )
        return None, None, n_days

    # RMSE global
    rmse = float(np.sqrt(mean_squared_error(
        df_eval["mm_actual"], df_eval["mm_predicted"]
    )))

    # Recall de lluvia — días donde realmente llovió y el modelo lo detectó
    y_true_bin = (df_eval["mm_actual"]    > CLF_RAIN_THRESHOLD).astype(int)
    y_pred_bin = (df_eval["mm_predicted"] > CLF_RAIN_THRESHOLD).astype(int)

    n_rainy = y_true_bin.sum()
    if n_rainy < 3:
        # Pocos días lluviosos — recall no es confiable (típico en verano)
        recall = None
        log.info(f"Solo {n_rainy} días lluviosos en el período — recall no evaluado")
    else:
        recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))

    log.info(f"📊 Métricas período {period_start} → {period_end}:")
    log.info(f"   RMSE:   {rmse:.3f}")
    log.info(f"   Recall: {recall:.3f}" if recall is not None else "   Recall: N/A")

    return rmse, recall, n_days


@task(name="check-degradation")
def check_degradation(
    rmse_current: Optional[float],
    recall_current: Optional[float],
) -> bool:
    """
    Decide si hay degradación comparando métricas actuales
    contra los umbrales del metadata.json.

    Degradación = AMBAS condiciones se cumplen:
        1. RMSE creció más del rmse_max_pct_increase respecto al baseline
        2. Recall bajó de recall_min (solo si hay suficientes días lluviosos)

    Si recall_current es None (verano sin lluvia), solo usa el criterio de RMSE.
    """
    # log = get_run_logger()
    log = logging.getLogger(__name__)

    registry = get_registry()

    baseline   = registry.baseline_metrics
    thresholds = registry.degradation_thresholds

    if rmse_current is None:
        log.info("Sin datos suficientes para evaluar degradación")
        return False

    rmse_baseline  = baseline.get("rmse", 4.676)
    recall_baseline = baseline.get("recall", 0.625)
    rmse_max_pct   = thresholds.get("rmse_max_pct_increase", 0.20)
    recall_min     = thresholds.get("recall_min", 0.50)

    rmse_limit     = rmse_baseline * (1 + rmse_max_pct)
    rmse_degraded  = rmse_current > rmse_limit

    if recall_current is not None:
        recall_degraded = recall_current < recall_min
    else:
        # Sin días lluviosos en el período (verano) — no podemos evaluar recall
        # Solo degradamos si el RMSE está muy mal
        recall_degraded = False
        log.info("Recall no evaluable en este período — solo se usa criterio RMSE")

    degraded = rmse_degraded and recall_degraded

    log.info(f"📊 Evaluación de degradación:")
    log.info(f"   RMSE:   {rmse_current:.3f} vs límite {rmse_limit:.3f} → {'❌ degradado' if rmse_degraded else '✅ ok'}")
    if recall_current is not None:
        log.info(f"   Recall: {recall_current:.3f} vs mínimo {recall_min:.3f} → {'❌ degradado' if recall_degraded else '✅ ok'}")
    log.info(f"   Decisión: {'⚠️ REENTRENAR' if degraded else '✅ no reentrenar'}")

    return degraded


@task(
    name="fetch-full-history",
    retries=3,
    retry_delay_seconds=60,
)
def fetch_full_history(end_date: date) -> pd.DataFrame:
    """
    Descarga el historial completo desde 2016 hasta end_date.
    Usado para reentrenar el modelo con todos los datos disponibles.
    """
    # log = get_run_logger()
    log = logging.getLogger(__name__)

    log.info(f"📡 Descargando historial completo 2016-01-01 → {end_date}...")

    df = fetch_weather_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date="2016-01-01",
        end_date=str(end_date),
        daily_variables=DAILY_VARIABLES,
        hourly_variables=HOURLY_VARIABLES,
        hourly_aggregations=HOURLY_AGGREGATIONS,
        timezone=TIMEZONE,
        add_derived=True,
    )

    log.info(f"✅ Historial descargado: {len(df)} días")
    return df


@task(name="retrain-model")
def retrain_model(df_raw: pd.DataFrame) -> Tuple[TwoStagePredictor, dict, float]:
    """
    Reentrenar el modelo con todos los datos disponibles.

    Usa los mismos parámetros del modelo actual (leídos desde metadata.json).
    No re-optimiza hiperparámetros — eso se hace manualmente en Databricks.

    Returns:
        Tupla (nuevo_modelo, nueva_metadata, rmse_validacion)
    """
    # log = get_run_logger()
    log = logging.getLogger(__name__)
    
    registry = get_registry()
    metadata = registry.metadata

    from datetime import datetime, timezone

    # Pipeline de features
    df_full = build_features(df_raw).dropna()
    log.info(f"Dataset para reentrenamiento: {len(df_full)} días")

    # Validación rápida: entrenar en todo menos los últimos 90 días
    df_val   = df_full.iloc[-90:]
    df_train = df_full.iloc[:-90]

    log.info(f"Validación: train hasta {df_train.index.max().date()}, val {df_val.index.min().date()} → {df_val.index.max().date()}")

    # Entrenar modelo de validación
    model_val = TwoStagePredictor(
        clf_params=metadata["clf_params"],
        reg_params=metadata["reg_params"],
        threshold=metadata["threshold"],
        log_target=metadata["log_target"],
        clf_rain_threshold=metadata.get("clf_rain_threshold", 0.5),
        reg_rain_threshold=metadata.get("reg_rain_threshold", 0.5),
    )
    model_val.fit(df_train)
    preds_val = model_val.predict(df_val)
    rmse_val  = float(np.sqrt(mean_squared_error(df_val["precip_t1"], preds_val)))
    log.info(f"RMSE validación: {rmse_val:.3f}")

    # Entrenar modelo de producción sobre todos los datos
    log.info("Entrenando modelo final sobre todos los datos...")
    model_prod = TwoStagePredictor(
        clf_params=metadata["clf_params"],
        reg_params=metadata["reg_params"],
        threshold=metadata["threshold"],
        log_target=metadata["log_target"],
        clf_rain_threshold=metadata.get("clf_rain_threshold", 0.5),
        reg_rain_threshold=metadata.get("reg_rain_threshold", 0.5),
    )
    model_prod.fit(df_full)

    # Nueva metadata — incrementa versión
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    new_metadata = {
        **metadata,
        "version":      version,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "train_start":  str(df_full.index.min().date()),
        "train_end":    str(df_full.index.max().date()),
        "n_samples":    len(df_full),
        "n_features":   len(model_prod.feature_names),
        "feature_names": model_prod.feature_names,
        "retrained_from_version": metadata.get("version"),
    }

    log.info(f"✅ Modelo reentrenado | versión {version} | {len(df_full)} días")
    return model_prod, new_metadata, rmse_val


@task(name="upload-new-model")
def upload_new_model(
    model: TwoStagePredictor,
    new_metadata: dict,
    rmse_new: float,
    rmse_current: float,
) -> Optional[str]:
    """
    Sube el nuevo modelo a HF Hub solo si mejora al actual.

    Returns:
        version del nuevo modelo si se subió, None si se rechazó
    """
    # log = get_run_logger()
    log = logging.getLogger(__name__)

    if rmse_new >= rmse_current:
        log.warning(
            f"⚠️ El nuevo modelo (RMSE={rmse_new:.3f}) no mejora al actual "
            f"(RMSE={rmse_current:.3f}). No se sube."
        )
        return None

    registry = get_registry()
    version  = registry.upload(model, new_metadata)

    log.info(
        f"✅ Nuevo modelo subido | versión={version} | "
        f"RMSE: {rmse_current:.3f} → {rmse_new:.3f}"
    )
    return version


# ──────────────────────────────────────────────────────────────────────────────
# FLOW PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

@flow(
    name="monthly-evaluation-flow",
    description="Evaluación mensual de degradación + reentrenamiento condicional",
    log_prints=True,
)
def monthly_flow(reference_date: Optional[date] = None) -> dict:
    """
    Flow mensual de evaluación y reentrenamiento.

    Args:
        reference_date: Fecha de referencia (default: hoy).
                        El período evaluado es los 30 días anteriores.

    Returns:
        dict con resumen de la evaluación
    """
    if reference_date is None:
        reference_date = date.today()

    # log = get_run_logger()
    log = logging.getLogger(__name__)
    
    log.info(f"🚀 Iniciando monthly_flow para {reference_date}")

    init_db()

    # Período de evaluación: últimos 30 días
    period_end   = reference_date - timedelta(days=1)
    period_start = period_end - timedelta(days=29)

    log.info(f"📅 Período de evaluación: {period_start} → {period_end}")

    # 1. Evaluar métricas recientes
    rmse_current, recall_current, n_days = evaluate_recent_performance(
        period_start, period_end
    )

    # 2. Decidir si hay degradación
    degraded = check_degradation(rmse_current, recall_current)

    retrained        = False
    new_model_version = None
    notes            = None

    # 3. Si hay degradación → reentrenar
    if degraded:
        log.info("⚠️ Degradación detectada — iniciando reentrenamiento...")

        try:
            # Descargar historial completo
            df_raw = fetch_full_history(reference_date)

            # Reentrenar
            new_model, new_metadata, rmse_new = retrain_model(df_raw)

            # Subir si mejora
            new_version = upload_new_model(
                new_model, new_metadata,
                rmse_new=rmse_new,
                rmse_current=rmse_current,
            )

            if new_version:
                retrained         = True
                new_model_version = new_version
                log.info(f"✅ Reentrenamiento exitoso | nueva versión: {new_version}")
            else:
                notes = f"Reentrenado pero RMSE no mejoró ({rmse_new:.3f} >= {rmse_current:.3f})"
                log.warning(notes)

        except Exception as e:
            notes = f"Error en reentrenamiento: {str(e)}"
            log.error(notes)

    # 4. Loggear resultado en DB
    registry = get_registry()
    baseline = registry.baseline_metrics

    log_evaluation(
        period_start=period_start,
        period_end=period_end,
        n_days=n_days,
        rmse_current=rmse_current,
        recall_current=recall_current,
        rmse_baseline=baseline.get("rmse", float("nan")),
        recall_baseline=baseline.get("recall", float("nan")),
        degraded=degraded,
        retrained=retrained,
        new_model_version=new_model_version,
        notes=notes,
    )

    result = {
        "reference_date":    str(reference_date),
        "period":            f"{period_start} → {period_end}",
        "n_days_evaluated":  n_days,
        "rmse_current":      rmse_current,
        "recall_current":    recall_current,
        "degraded":          degraded,
        "retrained":         retrained,
        "new_model_version": new_model_version,
        "notes":             notes,
    }

    log.info(f"✅ monthly_flow completado: {result}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = monthly_flow()
    print(f"\nResultado: {result}")