"""Módulo para descarga de datos desde Open-Meteo API"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    daily_variables: List[str],
    timezone: str = "UTC"
) -> pd.DataFrame:
    """
    Descarga datos meteorológicos históricos desde Open-Meteo API.
    
    Args:
        latitude: Latitud del lugar
        longitude: Longitud del lugar
        start_date: Fecha inicio (formato 'YYYY-MM-DD')
        end_date: Fecha fin (formato 'YYYY-MM-DD')
        daily_variables: Lista de variables a descargar
        timezone: Zona horaria
        
    Returns:
        DataFrame con datos meteorológicos
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_variables),
        "timezone": timezone
    }
    
    print(f"📡 Descargando datos desde {start_date} hasta {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Construir DataFrame
    df = pd.DataFrame({
        "fecha": pd.to_datetime(data["daily"]["time"]),
        "precipitacion": data["daily"]["precipitation_sum"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "viento_max": data["daily"]["windspeed_10m_max"],
        "weather_code": data["daily"]["weathercode"]
    })
    
    print(f"✅ Descargados {len(df)} registros")
    return df


def save_to_delta_table(df: pd.DataFrame, table_name: str, spark) -> None:
    """
    Guarda DataFrame en tabla Delta de Databricks.
    
    Args:
        df: DataFrame pandas a guardar
        table_name: Nombre de la tabla Delta
        spark: Spark session
    """
    spark_df = spark.createDataFrame(df)
    spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"✅ Guardado en tabla Delta: {table_name}")


def load_from_delta_table(table_name: str, spark) -> pd.DataFrame:
    """
    Carga datos desde tabla Delta.
    
    Args:
        table_name: Nombre de la tabla Delta
        spark: Spark session
        
    Returns:
        DataFrame pandas
    """
    df = spark.table(table_name).toPandas()
    print(f"✅ Cargados {len(df)} registros desde {table_name}")
    return df