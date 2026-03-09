"""Configuración centralizada del proyecto"""

# Coordenadas Santiago (Quinta Normal)
LATITUDE = -33.4447
LONGITUDE = -70.6828
TIMEZONE = "America/Santiago"

# Datos históricos
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

# Open-Meteo API (Historical Archive)
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables diarias — superficie
DAILY_VARIABLES = [
    # Precipitación
    "precipitation_sum",
    "precipitation_hours",

    # Temperatura
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",

    # Viento (superficie)
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",

    # Radiación y evapotranspiración
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",

    # Código de tiempo WMO
    "weathercode",
]

# Variables horarias — presión, humedad, nubosidad y viento de superficie
# NOTA: Variables de niveles de presión (500/700/850 hPa) no disponibles
# en Open-Meteo Archive para Sudamérica en ningún modelo.
HOURLY_VARIABLES = [
    # Presión atmosférica — clave para detectar frentes
    "pressure_msl",             # Presión a nivel del mar (hPa)
    "surface_pressure",         # Presión superficial (hPa)

    # Humedad y punto de rocío
    "relative_humidity_2m",     # Humedad relativa (%)
    "dew_point_2m",             # Punto de rocío (°C)
    "vapour_pressure_deficit",  # VPD — bajo = atmósfera saturada

    # Nubosidad por capas — cirros altos preceden frentes 12-24h
    "cloud_cover",              # Total (%)
    "cloud_cover_low",          # < 3km (lluvia frontal)
    "cloud_cover_mid",          # 3-8 km
    "cloud_cover_high",         # > 8 km (cirros precursores)

    # Viento en superficie
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

# Nombres de columnas en el DataFrame (mapeados desde API diaria)
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

# Agregaciones diarias de variables horarias
# Formato: {variable_horaria: [lista de agregaciones]}
HOURLY_AGGREGATIONS = {
    # Presión: mean/min/max para detectar ciclos y caídas de presión
    "pressure_msl":              ["mean", "min", "max"],
    "surface_pressure":          ["mean"],

    # Humedad
    "relative_humidity_2m":      ["mean", "max"],
    "dew_point_2m":              ["mean", "min"],
    "vapour_pressure_deficit":   ["mean", "min"],

    # Nubosidad
    "cloud_cover":               ["mean", "max"],
    "cloud_cover_low":           ["mean", "max"],
    "cloud_cover_mid":           ["mean"],
    "cloud_cover_high":          ["mean"],

    # Viento superficie
    "wind_speed_10m":            ["mean", "max"],
    "wind_direction_10m":        ["mean"],
    "wind_gusts_10m":            ["max"],
}

# Features derivadas — calculadas en preprocessing a partir de variables agregadas
DERIVED_FEATURES = [
    "pressure_trend_24h",   # pressure_msl_mean(t) - pressure_msl_mean(t-1)
    "pressure_trend_48h",   # pressure_msl_mean(t) - pressure_msl_mean(t-2)
    "pressure_range",       # pressure_msl_max - pressure_msl_min (amplitud intradía)
    "temp_range",           # temp_max - temp_min (días secos tienen mayor rango)
    "wind_west_component",  # wind_speed_10m_mean * cos(wind_direction_10m_mean) — viento del oeste
    "wind_north_component", # wind_speed_10m_mean * sin(wind_direction_10m_mean) — advección norte
    "rh_vpd_interaction",   # relative_humidity_2m_mean * vapour_pressure_deficit_mean
]

# Columna target
TARGET_COL = "precip"
TARGET = "precip_t1"

# Columna de fecha
DATE_COL = "fecha"

# Features de lags — variables sobre las que se calculan lags diarios
LAG_VARS = [
    # Precipitación
    "precip", "precip_horas",

    # Temperatura
    "temp_max", "temp_min", "temp_mean", "temp_range",

    # Viento
    "viento_max", "rafagas_max", "viento_dir",
    "wind_speed_10m_mean", "wind_gusts_10m_max",
    "wind_west_component", "wind_north_component",

    # Radiación y ET
    "radiacion", "evapotransp",

    # Código WMO
    "weather_code",

    # Presión — las más informativas para frentes
    "pressure_msl_mean", "pressure_msl_min", "pressure_msl_max",
    "pressure_trend_24h", "pressure_trend_48h", "pressure_range",

    # Humedad
    "relative_humidity_2m_mean", "relative_humidity_2m_max",
    "dew_point_2m_mean",
    "vapour_pressure_deficit_mean",
    "rh_vpd_interaction",

    # Nubosidad
    "cloud_cover_mean", "cloud_cover_max",
    "cloud_cover_low_mean", "cloud_cover_low_max",
    "cloud_cover_mid_mean",
    "cloud_cover_high_mean",
]

# Features de rolling windows — ventanas temporales para capturar régimen climático
ROLLING_VARS = [
    "precip",
    "pressure_msl_mean",
    "pressure_trend_24h",
    "relative_humidity_2m_mean",
    "cloud_cover_low_mean",
    "dew_point_2m_mean",
    "wind_west_component",
    "temp_mean",
    "radiacion",
    "evapotransp",
]
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

# Two-stage model
CLF_RAIN_THRESHOLD = 0.5                    # mm mínimos para definir "llueve" en el clasificador
                                            # (0.5mm maximiza F1 y minimiza varianza entre folds)
REG_RAIN_THRESHOLD = 0.5                    # mm mínimos para entrenar el regresor
                                            # (filtra lloviznas ambiguas; el reg solo aprende en lluvia real)
RAIN_THRESHOLD = REG_RAIN_THRESHOLD         # alias de compatibilidad — usar los anteriores en código nuevo
CLF_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]      # umbrales de probabilidad a explorar en grid del clasificador
F_BETA = 1.3                                # beta para fbeta_score (2 = recall vale el doble que precision)




























