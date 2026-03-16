"""
Flow diario — Ingesta + Predicción

Triggerado por GitHub Actions todos los días a las 08:00 hora Santiago.

Qué hace:
    1. Descarga los últimos 60 días desde Open-Meteo
    2. Construye features con el pipeline existente
    3. Genera la predicción para mañana
    4. Guarda la predicción en SQLite
    5. Guarda el dato real de ayer en SQLite
"""

import logging
import pandas as pd
from datetime import date, timedelta
from typing import Optional

from src.data.ingestion import fetch_weather_data
from src.data.preprocessing import build_features, get_feature_names
from src.utils.config import (
    LATITUDE, LONGITUDE, TIMEZONE,
    DAILY_VARIABLES, HOURLY_VARIABLES, HOURLY_AGGREGATIONS,
    CLF_RAIN_THRESHOLD,
)
from src.storage.database import init_db, save_prediction, save_actual
from src.storage.hf_model import ModelRegistry

logger = logging.getLogger(__name__)

# Registry global — compartido con FastAPI y monthly_flow
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Retorna el registry global, cargándolo si no está inicializado."""
    global _registry
    if _registry is None or not _registry.is_loaded():
        _registry = ModelRegistry()
        _registry.load()
    return _registry


def _fetch_data(target_date: date) -> pd.DataFrame:
    """Descarga los últimos 60 días — suficiente para construir lags de 30 días."""
    start = target_date - timedelta(days=60)
    logger.info(f"Descargando datos {start} → {target_date}")
    df = fetch_weather_data(
        latitude=LATITUDE, longitude=LONGITUDE,
        start_date=str(start), end_date=str(target_date),
        daily_variables=DAILY_VARIABLES,
        hourly_variables=HOURLY_VARIABLES,
        hourly_aggregations=HOURLY_AGGREGATIONS,
        timezone=TIMEZONE, add_derived=True,
    )
    logger.info(f"Datos descargados: {len(df)} días")
    return df


def _build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline de features y retorna la última fila."""
    df_features = build_features(df_raw)
    df_clean    = df_features.dropna(subset=get_feature_names(df_features))
    if len(df_clean) == 0:
        raise ValueError("No hay filas sin NaN después de build_features.")
    last_row = df_clean.iloc[[-1]]
    logger.info(f"Features construidas para {last_row.index[0].date()}")
    return last_row


def _generate_prediction(df_features: pd.DataFrame) -> dict:
    """Genera la predicción para mañana usando el modelo en memoria."""
    registry  = get_registry()
    threshold = registry.threshold
    today     = df_features.index[0].date()
    tomorrow  = today + timedelta(days=1)

    prob_rain    = float(registry.model.predict_proba(df_features)[0])
    mm_predicted = float(registry.model.predict(df_features).iloc[0])
    will_rain    = prob_rain > threshold

    result = {
        "fecha_features":   str(today),
        "fecha_prediccion": str(tomorrow),
        "prob_rain":        round(prob_rain, 4),
        "mm_predicted":     round(max(mm_predicted, 0.0), 3),
        "will_rain":        will_rain,
        "model_version":    registry.metadata.get("version", "unknown"),
        "threshold_used":   threshold,
    }
    emoji = "🌧️" if will_rain else "☀️"
    logger.info(f"{emoji} {tomorrow}: P(llueve)={prob_rain:.3f} | mm={mm_predicted:.2f}")
    return result


def _save_prediction(prediction: dict) -> None:
    """Persiste la predicción en SQLite."""
    save_prediction(
        fecha=date.fromisoformat(prediction["fecha_prediccion"]),
        prob_rain=prediction["prob_rain"],
        mm_predicted=prediction["mm_predicted"],
        will_rain=prediction["will_rain"],
        model_version=prediction["model_version"],
        threshold_used=prediction["threshold_used"],
    )
    logger.info(f"Predicción guardada para {prediction['fecha_prediccion']}")


def _save_actual(df_raw: pd.DataFrame, target_date: date) -> None:
    """Guarda el dato real de precipitación de ayer."""
    yesterday = target_date - timedelta(days=1)
    df_raw["fecha"] = pd.to_datetime(df_raw["fecha"])
    mask = df_raw["fecha"].dt.date == yesterday
    if not mask.any():
        logger.warning(f"No se encontró dato real para {yesterday}")
        return
    precip_col = "precipitation_sum" if "precipitation_sum" in df_raw.columns else "precip"
    mm_real    = float(df_raw.loc[mask, precip_col].iloc[0])
    save_actual(fecha=yesterday, mm_actual=mm_real, rain_threshold=CLF_RAIN_THRESHOLD)
    logger.info(f"Real guardado para {yesterday}: {mm_real:.2f} mm")


def daily_flow(target_date: Optional[date] = None) -> dict:
    """
    Flow diario: ingesta + predicción.

    Args:
        target_date: Día de referencia (default: hoy).

    Returns:
        dict con el resultado de la predicción.
    """
    if target_date is None:
        target_date = date.today()

    logger.info(f"🚀 Iniciando daily_flow para {target_date}")
    init_db()

    df_raw      = _fetch_data(target_date)
    df_features = _build_features(df_raw)
    prediction  = _generate_prediction(df_features)
    _save_prediction(prediction)
    _save_actual(df_raw, target_date)

    logger.info(f"✅ daily_flow completado")
    return prediction


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(daily_flow(), indent=2))