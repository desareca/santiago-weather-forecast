# 🌧️ Santiago Weather Forecast

Predicción diaria de precipitación para Santiago de Chile usando un modelo Two-Stage (LightGBM), con ingesta automatizada desde Open-Meteo, API REST en Render y orquestación via GitHub Actions.

**API en producción:** `https://santiago-weather-api.onrender.com`

---

## 📋 Descripción del Problema

Santiago tiene un patrón de precipitación extremadamente asimétrico: **~89% de los días son secos** y las lluvias se concentran en invierno (junio-agosto). Esto convierte la predicción de precipitación en un problema de desbalanceo severo donde los modelos de regresión clásicos simplemente predicen "cero" casi siempre.

El proyecto aborda esto con una arquitectura **Two-Stage**:
1. **Clasificador** — ¿Lloverá mañana? (`P(llueve) > threshold`)
2. **Regresor** — ¿Cuánto lloverá? (entrenado solo en días lluviosos)

---

## 🏗️ Arquitectura del Modelo

```
features(t)  ←  variables meteorológicas del día actual
     │
     ▼
LGBMClassifier  →  P(llueve mañana)  [0, 1]
     │
     ├── prob ≤ threshold  →  pred = 0 mm
     │
     └── prob > threshold  →  LGBMRegressor
                                    └──  pred = prob × mm_predichos
```

### Variables de entrada (features)
| Categoría | Variables |
|-----------|-----------|
| Precipitación | lags (t-1, t-7, t-30), rolling mean/std/max (7d, 30d), días con lluvia en ventana |
| Temperatura | temp_max, temp_min, temp_range |
| Viento | windspeed_max, componentes N/O |
| Presión | surface_pressure, pressure_trend_24h, pressure_trend_48h, pressure_range |
| Humedad | relative_humidity, vapor pressure deficit |
| Nubosidad | cloud_cover_low, cloud_cover_mid, cloud_cover_high |
| Tiempo | weather_code, estación (hemisferio sur), mes_sin/cos, dia_año_sin/cos |

### Target
`precip_t1` — precipitación del día siguiente (mm), construido con `shift(-1)`.

---

## 🚀 Arquitectura de Producción

```
Databricks Community (experimentación — manual)
    └── notebooks 01-05: EDA, grid search, entrenamiento
    └── MLflow tracking de experimentos
    └── modelo entrenado → sube a Hugging Face Hub

GitHub Actions (scheduler — automático)
    ├── daily.yml   → POST /flows/daily   (08:00 Santiago, todos los días)
    └── monthly.yml → POST /flows/monthly (06:00 Santiago, día 1 de cada mes)

Render Free Tier (producción)
    └── Web Service — FastAPI
         ├── GET  /health
         ├── GET  /predict/today
         ├── GET  /predict/{fecha}
         ├── GET  /history
         ├── GET  /model-info
         ├── POST /flows/daily    ← daily_flow()
         └── POST /flows/monthly  ← monthly_flow()

Hugging Face Hub (storage)
    ├── two_stage_model.pkl   — modelo serializado
    ├── metadata.json         — versión, parámetros, métricas baseline
    └── santiago_weather.db   — backup diario de la DB SQLite
```

### Flow diario (`daily_flow`)
1. Descarga últimos 60 días desde Open-Meteo (contexto para lags)
2. Construye features con el pipeline existente
3. Genera predicción para mañana
4. Guarda predicción en SQLite
5. Guarda precipitación real de ayer en SQLite
6. Sube backup de la DB a HF Hub

### Flow mensual (`monthly_flow`)
1. Evalúa RMSE y Recall de los últimos 30 días
2. Compara contra umbrales de degradación del `metadata.json`
3. Si hay degradación → reentrenar con historial completo (2016→hoy)
4. Si el nuevo modelo mejora → sube a HF Hub y actualiza en memoria
5. Registra resultado en `retraining_log`

---

## 📁 Estructura del Proyecto

```
santiago-weather-forecast/
│
├── src/
│   ├── api/
│   │   └── main.py               # FastAPI — endpoints REST
│   │
│   ├── data/
│   │   ├── ingestion.py          # Descarga Open-Meteo API
│   │   └── preprocessing.py      # Feature engineering, lags, splits
│   │
│   ├── flows/
│   │   ├── daily_flow.py         # Ingesta + predicción diaria
│   │   └── monthly_flow.py       # Evaluación + reentrenamiento mensual
│   │
│   ├── models/
│   │   ├── base_model.py         # Clase abstracta BasePredictor
│   │   ├── lightgbm_model.py     # LightGBMPredictor (single stage)
│   │   └── two_stage_model.py    # TwoStagePredictor (clf + reg)
│   │
│   ├── storage/
│   │   ├── database.py           # SQLite + backup en HF Hub
│   │   └── hf_model.py           # Wrapper Hugging Face Hub
│   │
│   └── utils/
│       ├── config.py             # Constantes centralizadas
│       └── mlflow_utils.py       # Helpers para logging MLflow
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Descarga y carga a Delta
│   ├── 02_eda.ipynb              # EDA + selección de CLF_RAIN_THRESHOLD
│   ├── 03_experiment_test.ipynb  # Pruebas rápidas del modelo two-stage
│   ├── 04_grid_search.ipynb      # Grid search clf + reg + evaluación final
│   └── 05_train_production.ipynb # Entrenamiento producción → HF Hub
│
├── .github/
│   └── workflows/
│       ├── daily.yml             # Cron diario 08:00 Santiago
│       └── monthly.yml           # Cron mensual día 1
│
├── Dockerfile
├── render.yaml
├── requirements.txt              # Dependencias Databricks
├── requirements-prod.txt         # Dependencias Render (producción)
└── README.md
```

---

## ⚙️ Configuración (`src/utils/config.py`)

```python
# Ubicación: Quinta Normal, Santiago
LATITUDE  = -33.4447
LONGITUDE = -70.6828
TIMEZONE  = "America/Santiago"

# Período histórico
START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

# Split
TRAIN_SPLIT = 0.8    # 80% entrenamiento, 20% test holdout

# Cross-validation
N_SPLITS      = 5
TEST_SIZE     = 365  # días por fold
MIN_TRAIN_SIZE = 365 * 3

# Two-stage
CLF_RAIN_THRESHOLD = 0.5   # mm — umbral para etiquetar "lluvia" en clasificador
REG_RAIN_THRESHOLD = 0.5   # mm — filtro de muestras para el regresor
CLF_THRESHOLDS     = [0.2, 0.3, 0.4, 0.5]  # umbrales de probabilidad a evaluar
F_BETA             = 2.0   # recall vale el doble que precision

# MLflow
EXPERIMENT_NAME = "/Users/carlos.saquel@gmail.com/santiago_weather_forecast"
```

---

## 🧪 Cross-Validation

Se usa `TimeSeriesSplit` que preserva el orden cronológico estrictamente (sin leakage de futuro). Cada fold tiene un mínimo de 3 años de entrenamiento y 1 año de test.

```
Fold 1: [2016–2018] → test 2019
Fold 2: [2016–2019] → test 2020
Fold 3: [2016–2020] → test 2021
Fold 4: [2016–2021] → test 2022
Fold 5: [2016–2022] → test 2023
```

⚠️ El fold 5 (2020-2023) cubre el período La Niña, que es atípicamente seco — es el fold con peores métricas en todos los modelos.

---

## 📊 Resultados

### Comparativa de modelos (test set holdout 2024–2025)

| Modelo | RMSE | R² | Recall lluvia | Recall picos |
|--------|------|----|---------------|--------------|
| Baseline regression_l1 | 4.983 | 0.105 | 39.3% | 50.0% |
| Two-Stage (default) | 4.780 | 0.176 | 50.0% | 50.0% |
| Two-Stage + Fbeta opt. | 4.777 | 0.178 | 69.4% | 78.6% |
| **Two-Stage (best clf + reg)** | **4.676** | **0.212** | 62.5% | 71.4% |

### Mejor configuración

**Clasificador:** `clf_high_reg` — LGBMClassifier con `reg_alpha=1.0`, `reg_lambda=5.0`
- AUC-ROC: 0.829 | Fbeta (β=2): 0.457 | Precision: 31.5% | Recall: 62.5%

**Regresor:** `reg_combo_q85_deep_reg` — objective quantile 85 + profundidad + regularización
- MAE días lluvia: 8.76 mm | RMSE días lluvia: 13.54 mm

### Decisiones clave

- **CLF_RAIN_THRESHOLD = 0.5mm** — maximiza Fbeta y minimiza varianza entre folds (std_Fbeta=0.029, mínimo entre umbrales 0.1–3.0mm evaluados).
- **F_BETA = 2.0** — recall pesa el doble que precision. En predicción de lluvia, los falsos negativos (no detectar lluvia real) son más costosos que los falsos positivos.
- **log_target=True en regresor** — transforma el target con `log1p` para estabilizar la distribución de cola larga.

---

## 🔧 Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Experimentación | Databricks Community Edition |
| Modelos | LightGBM (clasificador + regresor) |
| MLOps tracking | MLflow (Databricks) |
| Model registry | Hugging Face Hub |
| Datos históricos | Open-Meteo Archive API |
| Orquestación | GitHub Actions (cron) |
| API | FastAPI + Uvicorn |
| Deploy | Render (free tier) |
| Base de datos | SQLite + backup en HF Hub |
| Lenguaje | Python 3.11 |

---

## 🗓️ Roadmap

- [x] Ingesta y almacenamiento en Delta Lake
- [x] EDA y selección de umbral de lluvia
- [x] Modelo baseline LightGBM single-stage
- [x] Modelo Two-Stage con optimización Fbeta
- [x] Grid search clasificador y regresor
- [x] Evaluación en test set holdout
- [x] Entrenamiento de producción → Hugging Face Hub
- [x] API REST con FastAPI (`/predict/today`, `/history`, `/health`)
- [x] Flow diario automatizado (GitHub Actions → Render)
- [x] Evaluación mensual y reentrenamiento condicional
- [x] Persistencia de DB con backup en HF Hub
- [ ] Dashboard de monitoreo de métricas

---

## 🌐 API Reference

Base URL: `https://santiago-weather-api.onrender.com`

> ⚠️ El servicio usa el free tier de Render — el primer request puede tardar ~30s si el servicio está dormido.

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio y modelo |
| `/predict/today` | GET | Predicción para mañana |
| `/predict/{fecha}` | GET | Predicción guardada para una fecha (YYYY-MM-DD) |
| `/history?n=30` | GET | Últimas N predicciones |
| `/model-info` | GET | Versión, parámetros y métricas baseline del modelo |
| `/flows/daily` | POST | Triggerear daily_flow manualmente |
| `/flows/monthly` | POST | Triggerear monthly_flow manualmente |
| `/docs` | GET | Documentación interactiva (Swagger UI) |

---

## 📍 Datos

**Fuente:** [Open-Meteo Archive API](https://open-meteo.com/) — datos históricos gratuitos sin clave de API.

**Estación de referencia:** Quinta Normal, Santiago (-33.4447, -70.6828) — estación meteorológica oficial WMO para Santiago.

**Período:** 2016–2025 (~3,650 días)

---

## 🔄 Reentrenamiento Automático

El `monthly_flow` evalúa degradación del modelo comparando las métricas de los últimos 30 días contra los umbrales definidos en `metadata.json`:

```python
degradation_thresholds = {
    "rmse_max_pct_increase": 0.20,  # RMSE no puede subir más del 20%
    "recall_min": 0.50,             # Recall lluvia no puede bajar de 50%
}
```

Si **ambas** condiciones se cumplen simultáneamente, el sistema reentrenar automáticamente con el historial completo (2016→fecha actual) usando los mismos hiperparámetros del modelo actual, y sube el nuevo modelo a HF Hub si mejora las métricas.