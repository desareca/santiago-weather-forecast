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
    "windspeed_10m_max",
    "weathercode"
]

# Modelo
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# CV
N_SPLITS = 5
TEST_SIZE = 365
MIN_TRAIN_SIZE = 365 * 3

# MLflow
EXPERIMENT_NAME = "/Users/carlos.saquel@gmail.com/santiago_weather_forecast"