"""Módulo para descarga de datos desde Open-Meteo API"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# DESCARGA
# ─────────────────────────────────────────────

def _get(url: str, params: dict) -> dict:
    """Ejecuta GET con manejo de errores."""
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_daily_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    daily_variables: List[str],
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Descarga variables diarias desde Open-Meteo Archive API.

    Returns:
        DataFrame indexado por fecha con una columna por variable.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   latitude,
        "longitude":  longitude,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(daily_variables),
        "timezone":   timezone,
    }

    print(f"📡 [daily] Descargando {len(daily_variables)} variables ({start_date} → {end_date})...")
    data = _get(url, params)

    df = pd.DataFrame(data["daily"])
    df = df.rename(columns={"time": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])

    print(f"✅ [daily] {len(df)} registros | {len(df.columns)-1} variables")
    return df


def fetch_hourly_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: List[str],
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Descarga variables horarias de superficie desde Open-Meteo Archive API.
    Cubre: presión MSL, humedad, nubosidad y viento en 10m.
    NOTA: Variables de niveles de presión (500/700/850 hPa) no disponibles
    en Open-Meteo Archive para Sudamérica.

    Returns:
        DataFrame con columna 'fecha' (datetime horario) y una columna por variable.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   latitude,
        "longitude":  longitude,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ",".join(hourly_variables),
        "timezone":   timezone,
    }

    print(f"📡 [hourly] Descargando {len(hourly_variables)} variables ({start_date} → {end_date})...")
    data = _get(url, params)

    df = pd.DataFrame(data["hourly"])
    df = df.rename(columns={"time": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])

    print(f"✅ [hourly] {len(df)} registros horarios | {len(df.columns)-1} variables")
    return df


# ─────────────────────────────────────────────
# AGREGACIÓN HORARIA → DIARIA
# ─────────────────────────────────────────────

def aggregate_hourly_to_daily(
    df_hourly: pd.DataFrame,
    aggregations: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Agrega DataFrame horario a diario aplicando las funciones definidas en aggregations.

    Args:
        df_hourly:    DataFrame horario con columna 'fecha'
        aggregations: Dict {variable: ['mean', 'min', 'max', ...]}
                      Solo se procesan variables presentes en el DataFrame.

    Returns:
        DataFrame diario con columnas nombradas como {variable}_{agg}.
        Columna 'fecha' contiene la fecha (sin hora).
    """
    df = df_hourly.copy()
    df["fecha"] = df["fecha"].dt.date  # truncar a día

    # Solo agregar variables que existen en el DataFrame
    agg_dict = {}
    for var, funcs in aggregations.items():
        if var in df.columns:
            agg_dict[var] = funcs
        else:
            print(f"  ⚠️  Variable horaria no encontrada, se omite: {var}")

    df_agg = df.groupby("fecha").agg(agg_dict)

    # Aplanar MultiIndex de columnas → variable_agg
    df_agg.columns = ["_".join(col) for col in df_agg.columns]
    df_agg = df_agg.reset_index()
    df_agg["fecha"] = pd.to_datetime(df_agg["fecha"])

    print(f"✅ [aggregate] {len(df_agg)} días | {len(df_agg.columns)-1} columnas agregadas")
    return df_agg


# ─────────────────────────────────────────────
# FEATURES DERIVADAS
# ─────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features derivadas sinópticas a partir de las variables agregadas de superficie.

    Features generadas (si las columnas base existen):
        pressure_trend_24h      caída de presión MSL en 24h (frentes en aproximación)
        pressure_trend_48h      caída de presión MSL en 48h
        pressure_range          amplitud intradía de presión (pressure_msl_max - min)
        temp_range              amplitud térmica diaria (días secos = rango mayor)
        wind_west_component     componente zonal del viento 10m (+ = del oeste → lluvia)
        wind_north_component    componente meridional del viento 10m (+ = del norte)
        rh_vpd_interaction      humedad relativa * VPD (señal de saturación atmosférica)
    """
    df = df.copy()

    def _has(*cols):
        return all(c in df.columns for c in cols)

    # Tendencias de presión superficial
    if _has("pressure_msl_mean"):
        df["pressure_trend_24h"] = df["pressure_msl_mean"].diff(1)
        df["pressure_trend_48h"] = df["pressure_msl_mean"].diff(2)

    # Amplitud intradía de presión
    if _has("pressure_msl_max", "pressure_msl_min"):
        df["pressure_range"] = df["pressure_msl_max"] - df["pressure_msl_min"]

    # Amplitud térmica
    if _has("temp_max", "temp_min"):
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    # Componentes del viento en superficie (10m)
    # wind_direction: 0° = Norte, 90° = Este, 180° = Sur, 270° = Oeste
    # Para viento del OESTE (lluvia en Santiago): dirección ~270°, cos(270°) ≈ 0, sin(270°) = -1
    # Usamos convención meteorológica: componente_u = -speed * sin(dir), componente_v = -speed * cos(dir)
    if _has("wind_speed_10m_mean", "wind_direction_10m_mean"):
        angle = np.radians(df["wind_direction_10m_mean"].to_numpy(dtype=float))
        speed = df["wind_speed_10m_mean"].to_numpy(dtype=float)
        df["wind_west_component"]  = -speed * np.sin(angle)  # u: positivo = del oeste
        df["wind_north_component"] = -speed * np.cos(angle)  # v: positivo = del norte

    # Interacción humedad × VPD (atmósfera saturada = RH alta y VPD bajo → producto pequeño)
    if _has("relative_humidity_2m_mean", "vapour_pressure_deficit_mean"):
        df["rh_vpd_interaction"] = (
            df["relative_humidity_2m_mean"] * df["vapour_pressure_deficit_mean"]
        )

    derived = [c for c in [
        "pressure_trend_24h", "pressure_trend_48h", "pressure_range",
        "temp_range",
        "wind_west_component", "wind_north_component",
        "rh_vpd_interaction",
    ] if c in df.columns]

    print(f"✅ [derived] {len(derived)} features derivadas: {derived}")
    return df


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────

def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    daily_variables: List[str],
    hourly_variables: Optional[List[str]] = None,
    hourly_aggregations: Optional[Dict[str, List[str]]] = None,
    timezone: str = "UTC",
    add_derived: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo: descarga diaria + horaria (opcional), agrega y calcula features derivadas.

    Args:
        latitude, longitude:   Coordenadas
        start_date, end_date:  Rango de fechas (YYYY-MM-DD)
        daily_variables:       Variables diarias de la API
        hourly_variables:      Variables horarias de superficie (presión, humedad, nubosidad, viento). None = no descarga.
        hourly_aggregations:   Dict {variable: [funciones]}. None = no agrega.
        timezone:              Zona horaria
        add_derived:           Si calcular features derivadas sinópticas

    Returns:
        DataFrame diario con todas las variables fusionadas.
    """
    # 1. Datos diarios
    df = fetch_daily_data(latitude, longitude, start_date, end_date, daily_variables, timezone)

    # 2. Datos horarios (si se especifican)
    if hourly_variables and hourly_aggregations:
        df_hourly = fetch_hourly_data(
            latitude, longitude, start_date, end_date, hourly_variables, timezone
        )
        df_daily_from_hourly = aggregate_hourly_to_daily(df_hourly, hourly_aggregations)

        # Merge por fecha
        df = df.merge(df_daily_from_hourly, on="fecha", how="left")
        print(f"✅ [merge] Shape tras merge: {df.shape}")

    # 3. Features derivadas
    if add_derived:
        df = add_derived_features(df)

    print(f"\n✅ Dataset final: {len(df)} días | {len(df.columns)} columnas")
    return df


# ─────────────────────────────────────────────
# DELTA TABLES
# ─────────────────────────────────────────────

def save_to_delta_table(df: pd.DataFrame, table_name: str, spark) -> None:
    """Guarda DataFrame en tabla Delta de Databricks."""
    spark_df = spark.createDataFrame(df)
    (spark_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name))
    print(f"✅ Guardado en tabla Delta: {table_name} ({len(df)} filas, {len(df.columns)} columnas)")


def load_from_delta_table(table_name: str, spark) -> pd.DataFrame:
    """Carga datos desde tabla Delta."""
    df = spark.table(table_name).toPandas()
    print(f"✅ Cargados {len(df)} registros desde {table_name}")
    return df