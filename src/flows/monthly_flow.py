"""
Flow mensual — Evaluación de degradación + Reentrenamiento condicional

Triggerado por GitHub Actions el día 1 de cada mes a las 06:00 Santiago.

Qué hace:
    1. Evalúa RMSE y Recall de los últimos 30 días (predicciones vs reales en SQLite)
    2. Compara contra umbrales del metadata.json
    3. Si hay degradación → reentrenar con historial completo desde Open-Meteo
    4. Si el nuevo modelo mejora → subirlo a HF Hub
    5. Loggear resultado en SQLite (retraining_log)

Criterio de degradación (AMBAS condiciones):
    - RMSE > baseline_rmse * (1 + rmse_max_pct_increase)
    - Recall lluvia < recall_min  (si hay suficientes días lluviosos)
"""

import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple

from sklearn.metrics import mean_squared_error, recall_score

from src.data.ingestion import fetch_weather_data
from src.data.preprocessing import build_features
from src.utils.config import (
    LATITUDE, LONGITUDE, TIMEZONE,
    DAILY_VARIABLES, HOURLY_VARIABLES, HOURLY_AGGREGATIONS,
    CLF_RAIN_THRESHOLD,
)
from src.models.two_stage_model import TwoStagePredictor
from src.storage.database import (
    init_db, get_actuals_for_evaluation, log_evaluation, backup_db_to_hub
)

from src.flows.daily_flow import get_registry

logger = logging.getLogger(__name__)

MIN_DAYS_FOR_EVALUATION = 15


# ──────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_recent_performance(
    period_start: date,
    period_end: date,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Calcula RMSE y Recall de los últimos 30 días.

    Returns:
        (rmse, recall_lluvia, n_days) — rmse y recall son None si no hay datos.
    """
    df = get_actuals_for_evaluation(period_start, period_end)

    if len(df) == 0:
        logger.warning("No hay datos reales en el período evaluado")
        return None, None, 0

    df_eval = df.dropna(subset=["mm_actual", "mm_predicted"])
    n_days  = len(df_eval)
    logger.info(f"Días con datos completos: {n_days} / {len(df)}")

    if n_days < MIN_DAYS_FOR_EVALUATION:
        logger.warning(f"Solo {n_days} días — mínimo requerido: {MIN_DAYS_FOR_EVALUATION}")
        return None, None, n_days

    rmse = float(np.sqrt(mean_squared_error(
        df_eval["mm_actual"], df_eval["mm_predicted"]
    )))

    y_true_bin = (df_eval["mm_actual"]    > CLF_RAIN_THRESHOLD).astype(int)
    y_pred_bin = (df_eval["mm_predicted"] > CLF_RAIN_THRESHOLD).astype(int)
    n_rainy    = y_true_bin.sum()

    if n_rainy < 3:
        recall = None
        logger.info(f"Solo {n_rainy} días lluviosos — recall no evaluado")
    else:
        recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))

    logger.info(f"RMSE: {rmse:.3f} | Recall: {recall:.3f if recall else 'N/A'}")
    return rmse, recall, n_days


def _check_degradation(
    rmse_current: Optional[float],
    recall_current: Optional[float],
) -> bool:
    """Decide si hay degradación comparando contra umbrales del metadata."""
    registry   = get_registry()
    baseline   = registry.baseline_metrics
    thresholds = registry.degradation_thresholds

    if rmse_current is None:
        logger.info("Sin datos suficientes para evaluar degradación")
        return False

    rmse_baseline = baseline.get("rmse", 4.676)
    recall_min    = thresholds.get("recall_min", 0.50)
    rmse_max_pct  = thresholds.get("rmse_max_pct_increase", 0.20)
    rmse_limit    = rmse_baseline * (1 + rmse_max_pct)

    rmse_degraded   = rmse_current > rmse_limit
    recall_degraded = (recall_current < recall_min) if recall_current is not None else False

    degraded = rmse_degraded and recall_degraded

    logger.info(f"RMSE: {rmse_current:.3f} vs límite {rmse_limit:.3f} → {'❌' if rmse_degraded else '✅'}")
    if recall_current is not None:
        logger.info(f"Recall: {recall_current:.3f} vs mínimo {recall_min} → {'❌' if recall_degraded else '✅'}")
    logger.info(f"Decisión: {'⚠️ REENTRENAR' if degraded else '✅ no reentrenar'}")

    return degraded


def _fetch_full_history(end_date: date) -> pd.DataFrame:
    """Descarga el historial completo desde 2016 para reentrenar."""
    logger.info(f"Descargando historial 2016-01-01 → {end_date}...")
    df = fetch_weather_data(
        latitude=LATITUDE, longitude=LONGITUDE,
        start_date="2016-01-01", end_date=str(end_date),
        daily_variables=DAILY_VARIABLES,
        hourly_variables=HOURLY_VARIABLES,
        hourly_aggregations=HOURLY_AGGREGATIONS,
        timezone=TIMEZONE, add_derived=True,
    )
    logger.info(f"Historial descargado: {len(df)} días")
    return df


def _retrain_model(df_raw: pd.DataFrame) -> Tuple[TwoStagePredictor, dict, float]:
    """
    Reentrenar con todos los datos disponibles.
    Usa los mismos parámetros del modelo actual desde metadata.json.

    Returns:
        (nuevo_modelo, nueva_metadata, rmse_validacion)
    """
    from datetime import datetime, timezone as tz

    registry = get_registry()
    metadata = registry.metadata

    df_full = build_features(df_raw).dropna()
    logger.info(f"Dataset reentrenamiento: {len(df_full)} días")

    # Validación rápida en últimos 90 días
    df_val   = df_full.iloc[-90:]
    df_train = df_full.iloc[:-90]

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
    logger.info(f"RMSE validación: {rmse_val:.3f}")

    # Modelo final sobre todos los datos
    logger.info("Entrenando modelo final sobre todos los datos...")
    model_prod = TwoStagePredictor(
        clf_params=metadata["clf_params"],
        reg_params=metadata["reg_params"],
        threshold=metadata["threshold"],
        log_target=metadata["log_target"],
        clf_rain_threshold=metadata.get("clf_rain_threshold", 0.5),
        reg_rain_threshold=metadata.get("reg_rain_threshold", 0.5),
    )
    model_prod.fit(df_full)

    version      = datetime.now(tz.utc).strftime("%Y%m%d_%H%M")
    new_metadata = {
        **metadata,
        "version":    version,
        "trained_at": datetime.now(tz.utc).isoformat(),
        "train_start": str(df_full.index.min().date()),
        "train_end":   str(df_full.index.max().date()),
        "n_samples":   len(df_full),
        "n_features":  len(model_prod.feature_names),
        "feature_names": model_prod.feature_names,
        "retrained_from_version": metadata.get("version"),
    }

    logger.info(f"Modelo reentrenado | versión {version}")
    return model_prod, new_metadata, rmse_val


def _upload_if_better(
    model: TwoStagePredictor,
    new_metadata: dict,
    rmse_new: float,
    rmse_current: float,
) -> Optional[str]:
    """Sube el nuevo modelo a HF Hub solo si mejora al actual."""
    if rmse_new >= rmse_current:
        logger.warning(
            f"Nuevo modelo (RMSE={rmse_new:.3f}) no mejora al actual "
            f"(RMSE={rmse_current:.3f}). No se sube."
        )
        return None

    registry = get_registry()
    version  = registry.upload(model, new_metadata)
    logger.info(f"Nuevo modelo subido | versión={version} | RMSE: {rmse_current:.3f} → {rmse_new:.3f}")
    return version


# ──────────────────────────────────────────────────────────────────────────────
# FLOW PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def monthly_flow(reference_date: Optional[date] = None) -> dict:
    """
    Flow mensual: evaluación + reentrenamiento condicional.

    Args:
        reference_date: Fecha de referencia (default: hoy).

    Returns:
        dict con resumen de la evaluación.
    """
    if reference_date is None:
        reference_date = date.today()

    logger.info(f"🚀 Iniciando monthly_flow para {reference_date}")
    init_db()

    period_end   = reference_date - timedelta(days=1)
    period_start = period_end - timedelta(days=29)
    logger.info(f"Período evaluado: {period_start} → {period_end}")

    # 1. Evaluar métricas recientes
    rmse_current, recall_current, n_days = _evaluate_recent_performance(
        period_start, period_end
    )

    # 2. Detectar degradación
    degraded = _check_degradation(rmse_current, recall_current)

    retrained         = False
    new_model_version = None
    notes             = None

    # 3. Reentrenar si hay degradación
    if degraded:
        logger.info("Degradación detectada — iniciando reentrenamiento...")
        try:
            df_raw = _fetch_full_history(reference_date)
            new_model, new_metadata, rmse_new = _retrain_model(df_raw)
            new_version = _upload_if_better(
                new_model, new_metadata,
                rmse_new=rmse_new,
                rmse_current=rmse_current,
            )
            if new_version:
                retrained         = True
                new_model_version = new_version
                logger.info(f"Reentrenamiento exitoso | nueva versión: {new_version}")
            else:
                notes = f"Reentrenado pero RMSE no mejoró ({rmse_new:.3f} >= {rmse_current:.3f})"
        except Exception as e:
            notes = f"Error en reentrenamiento: {str(e)}"
            logger.error(notes)

    # 4. Loggear en DB
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

    # 5. Backup DB
    backup_db_to_hub()

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

    logger.info(f"✅ monthly_flow completado")
    return result


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(monthly_flow(), indent=2, default=str))