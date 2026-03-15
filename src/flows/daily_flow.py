"""
Flow diario de Prefect — Ingesta + Predicción

Corre todos los días a las 08:00 hora Santiago (America/Santiago).

Qué hace:
    1. Descarga los datos meteorológicos de ayer desde Open-Meteo
    2. Carga el historial reciente desde la DB para construir los lags
    3. Construye features con el pipeline existente
    4. Genera la predicción para mañana
    5. Guarda la predicción en SQLite
    6. Guarda el dato real de ayer en SQLite (para evaluación futura)

Por qué necesitamos historial para los lags:
    El modelo usa lags de 1, 7 y 30 días + rolling de 30 días.
    Un solo día de datos no alcanza — necesitamos al menos 60 días
    de contexto para construir todas las features correctamente.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta as td

from src.data.ingestion import fetch_weather_data
from src.data.preprocessing import build_features, get_feature_names
from src.utils.config import (
    LATITUDE, LONGITUDE, TIMEZONE,
    DAILY_VARIABLES, HOURLY_VARIABLES, HOURLY_AGGREGATIONS,
    CLF_RAIN_THRESHOLD,
)
from src.storage.database import (
    init_db, save_prediction, save_actual, get_recent_predictions
)
from src.storage.hf_model import ModelRegistry

logger = logging.getLogger(__name__)

# Registry global — se carga una vez al iniciar el proceso
# La FastAPI también usa esta instancia
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Retorna el registry global, cargándolo si no está inicializado."""
    global _registry
    if _registry is None or not _registry.is_loaded():
        _registry = ModelRegistry()
        _registry.load()
    return _registry


# ──────────────────────────────────────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────────────────────────────────────

@task(
    name="fetch-weather-data",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=td(hours=12),
)
def fetch_data(target_date: date) -> pd.DataFrame:
    """
    Descarga datos meteorológicos desde Open-Meteo.

    Descarga los últimos 60 días hasta target_date para tener
    suficiente contexto para construir lags y rolling features.

    Args:
        target_date: El día cuyas features queremos construir (hoy).
                     Predeciremos target_date + 1 (mañana).
    """
    log = get_run_logger()

    # Necesitamos 60 días de contexto para lags y rolling de 30 días
    start = target_date - timedelta(days=60)
    end   = target_date

    log.info(f"Descargando datos {start} → {end}")

    df = fetch_weather_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=str(start),
        end_date=str(end),
        daily_variables=DAILY_VARIABLES,
        hourly_variables=HOURLY_VARIABLES,
        hourly_aggregations=HOURLY_AGGREGATIONS,
        timezone=TIMEZONE,
        add_derived=True,
    )

    log.info(f"✅ Datos descargados: {len(df)} días")
    return df


@task(name="build-features")
def build_features_task(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de feature engineering.
    Retorna solo la última fila (el día más reciente) con todas sus features.
    """
    log = get_run_logger()

    df_features = build_features(df_raw)

    # Eliminar NaN — los primeros días no tienen lags completos
    df_clean = df_features.dropna(subset=get_feature_names(df_features))

    if len(df_clean) == 0:
        raise ValueError("No hay filas sin NaN después de build_features.")

    # Solo necesitamos la última fila para predecir
    last_row = df_clean.iloc[[-1]]
    log.info(f"✅ Features construidas para {last_row.index[0].date()}")
    return last_row


@task(name="generate-prediction")
def generate_prediction(df_features: pd.DataFrame) -> dict:
    """
    Genera la predicción para mañana usando el modelo en memoria.

    Returns:
        dict con fecha_prediccion, prob_rain, mm_predicted, will_rain,
        model_version, threshold_used
    """
    log = get_run_logger()
    registry = get_registry()

    model    = registry.model
    metadata = registry.metadata
    threshold = registry.threshold

    # La feature date es el día de hoy (t)
    # La predicción es para mañana (t+1)
    today      = df_features.index[0].date()
    tomorrow   = today + timedelta(days=1)

    prob_rain    = float(model.predict_proba(df_features)[0])
    mm_predicted = float(model.predict(df_features).iloc[0])
    will_rain    = prob_rain > threshold

    result = {
        "fecha_features":  str(today),      # día que usamos como input
        "fecha_prediccion": str(tomorrow),  # día que estamos prediciendo
        "prob_rain":       round(prob_rain, 4),
        "mm_predicted":    round(max(mm_predicted, 0.0), 3),
        "will_rain":       will_rain,
        "model_version":   metadata.get("version", "unknown"),
        "threshold_used":  threshold,
    }

    emoji = "🌧️" if will_rain else "☀️"
    log.info(
        f"{emoji} Predicción para {tomorrow}: "
        f"P(llueve)={prob_rain:.3f} | mm={mm_predicted:.2f} | "
        f"llueve={'sí' if will_rain else 'no'}"
    )
    return result


@task(name="save-prediction")
def save_prediction_task(prediction: dict) -> None:
    """Persiste la predicción en SQLite."""
    log = get_run_logger()

    save_prediction(
        fecha=date.fromisoformat(prediction["fecha_prediccion"]),
        prob_rain=prediction["prob_rain"],
        mm_predicted=prediction["mm_predicted"],
        will_rain=prediction["will_rain"],
        model_version=prediction["model_version"],
        threshold_used=prediction["threshold_used"],
    )
    log.info(f"✅ Predicción guardada para {prediction['fecha_prediccion']}")


@task(name="save-actual")
def save_actual_task(df_raw: pd.DataFrame, target_date: date) -> None:
    """
    Guarda el dato real de precipitación de ayer en la DB.

    Open-Meteo Archive tiene datos con ~1 día de delay, así que
    cuando corremos hoy podemos guardar el real de ayer.

    Args:
        df_raw:      DataFrame con datos crudos descargados
        target_date: Fecha de hoy — el real de ayer es target_date - 1
    """
    log = get_run_logger()

    yesterday = target_date - timedelta(days=1)

    # Buscar precipitación real de ayer en los datos descargados
    df_raw["fecha"] = pd.to_datetime(df_raw["fecha"])
    mask = df_raw["fecha"].dt.date == yesterday

    if not mask.any():
        log.warning(f"⚠️ No se encontró dato real para {yesterday}")
        return

    # La columna puede llamarse precipitation_sum o precip según el estado del df
    precip_col = "precipitation_sum" if "precipitation_sum" in df_raw.columns else "precip"
    mm_real = float(df_raw.loc[mask, precip_col].iloc[0])

    save_actual(
        fecha=yesterday,
        mm_actual=mm_real,
        rain_threshold=CLF_RAIN_THRESHOLD,
    )
    log.info(f"✅ Real guardado para {yesterday}: {mm_real:.2f} mm")


# ──────────────────────────────────────────────────────────────────────────────
# FLOW PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

@flow(
    name="daily-prediction-flow",
    description="Ingesta diaria desde Open-Meteo + predicción de precipitación para mañana",
    log_prints=True,
)
def daily_flow(target_date: Optional[date] = None) -> dict:
    """
    Flow principal diario.

    Args:
        target_date: Día de referencia (default: hoy).
                     Útil para correr en modo backfill.

    Returns:
        dict con el resultado de la predicción
    """
    if target_date is None:
        target_date = date.today()

    log = get_run_logger()
    log.info(f"🚀 Iniciando daily_flow para {target_date}")

    # Asegurar que la DB existe
    init_db()

    # 1. Descargar datos
    df_raw = fetch_data(target_date)

    # 2. Construir features
    df_features = build_features_task(df_raw)

    # 3. Predecir
    prediction = generate_prediction(df_features)

    # 4. Guardar predicción
    save_prediction_task(prediction)

    # 5. Guardar real de ayer
    save_actual_task(df_raw, target_date)

    log.info(f"✅ daily_flow completado para {target_date}")
    return prediction


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT (para correr manualmente o desde Render cron)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = daily_flow()
    print(f"\nResultado: {result}")