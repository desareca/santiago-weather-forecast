"""Configuración centralizada del proyecto"""

# Coordenadas Santiago (Quinta Normal)
LATITUDE = -33.4447
LONGITUDE = -70.6828
TIMEZONE = "America/Santiago"

# Datos históricos
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

# Open-Meteo API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
    "precipitation_hours",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "weathercode",
]

# Nombres de columnas en el DataFrame (mapeados desde la API)
RENAME_MAP = {
    "precipitation_sum":          "precip",
    "temperature_2m_max":         "temp_max",
    "temperature_2m_min":         "temp_min",
    "temperature_2m_mean":        "temp_mean",
    "windspeed_10m_max":          "viento_max",
    "windgusts_10m_max":          "rafagas_max",
    "winddirection_10m_dominant": "viento_dir",
    "precipitation_hours":        "precip_horas",
    "shortwave_radiation_sum":    "radiacion",
    "et0_fao_evapotranspiration": "evapotransp",
    "weathercode":                "weather_code",
}

# Columna target (precip del día siguiente)
TARGET_COL = "precip"
TARGET = "precip_t1"  # nombre de la columna target en el DataFrame de features

# Columna de fecha
DATE_COL = "fecha"

# Features de lags — variables sobre las que se calculan lags
LAG_VARS = [
    "precip", "temp_max", "temp_min", "temp_mean",
    "viento_max", "rafagas_max", "radiacion",
    "evapotransp", "weather_code", "precip_horas",
]

# Features de rolling windows
ROLLING_VARS = ["precip", "temp_mean", "radiacion", "evapotransp"]
ROLLING_WINDOWS = [7, 14, 30]

# Modelo
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# CV
N_SPLITS = 5
TEST_SIZE = 365
MIN_TRAIN_SIZE = 365 * 3

# MLflow
EXPERIMENT_NAME = "/Users/carlos.saquel@gmail.com/santiago_weather_forecast"
MODEL_NAME = "santiago_lgbm_precip"