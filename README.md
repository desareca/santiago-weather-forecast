# 🌧️ Santiago Weather Forecast

Predicción diaria de precipitación para Santiago de Chile usando un modelo Two-Stage (LightGBM), con ingesta automatizada desde Open-Meteo, tracking en MLflow y orquestación en Databricks.

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

## 📁 Estructura del Proyecto

```
santiago-weather-forecast/
│
├── src/
│   ├── data/
│   │   ├── ingestion.py          # Descarga Open-Meteo API → Delta table
│   │   └── preprocessing.py      # Feature engineering, lags, splits
│   │
│   ├── models/
│   │   ├── base_model.py         # Clase abstracta BasePredictor
│   │   ├── lightgbm_model.py     # LightGBMPredictor (single stage)
│   │   └── two_stage_model.py    # TwoStagePredictor (clf + reg)
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # MAE, RMSE, R², Fbeta, recall lluvia
│   │   ├── cross_validation.py   # TimeSeriesSplit sin leakage
│   │   └── two_stage_cv.py       # CV, grid search y threshold optimization
│   │
│   └── utils/
│       ├── config.py             # Constantes centralizadas
│       └── mlflow_utils.py       # Helpers para logging MLflow
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Descarga y carga a Delta
│   ├── 02_eda.ipynb              # EDA + selección de CLF_RAIN_THRESHOLD
│   ├── 03_experiment_test.ipynb  # Pruebas rápidas del modelo two-stage
│   └── 04_grid_search.ipynb      # Grid search clf + reg + evaluación final
│
├── experiments/
│   ├── grids/
│   │   └── grid_tweedie.json     # Configuraciones LightGBM baseline
│   └── results/
│       └── results_cv_tweedie.csv
│
├── requirements.txt
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

## 🚀 Cómo Ejecutar

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

En Databricks:
```python
%pip install prophet lightgbm prefect holidays tqdm --no-deps --quiet
%pip install -U opentelemetry-api --quiet
```

### 2. Ingesta de datos

Ejecutar `notebooks/01_data_ingestion.ipynb` — descarga desde Open-Meteo y guarda en Delta table `weather_raw`.

### 3. EDA y selección de threshold

Ejecutar `notebooks/02_eda.ipynb` — incluye análisis de distribución de precipitación y selección automática de `CLF_RAIN_THRESHOLD` mediante CV.

### 4. Experimentos

Ejecutar `notebooks/04_grid_search.ipynb`:
- Sección 3: Grid search del clasificador (18 configuraciones)
- Sección 4: Grid search del regresor (20 configuraciones)
- Sección 5: Evaluación final en test set + registro en MLflow

---

## 🔧 Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Plataforma | Databricks (Unity Catalog) |
| Modelos | LightGBM (clasificador + regresor) |
| MLOps | MLflow (tracking + Model Registry) |
| Datos | Open-Meteo Archive API → Delta Lake |
| Orquestación | Prefect (pendiente) |
| API | FastAPI (pendiente) |
| Lenguaje | Python 3.10+ |

---

## 🗓️ Roadmap

- [x] Ingesta y almacenamiento en Delta Lake
- [x] EDA y selección de umbral de lluvia
- [x] Modelo baseline LightGBM single-stage
- [x] Modelo Two-Stage con optimización Fbeta
- [x] Grid search clasificador y regresor
- [x] Evaluación en test set holdout
- [ ] Registro en MLflow Model Registry
- [ ] API REST con FastAPI (`/predict`)
- [ ] Job Prefect diario (ingesta automática + predicción)
- [ ] Dashboard de monitoreo de drift

---

## 📍 Datos

**Fuente:** [Open-Meteo Archive API](https://open-meteo.com/) — datos históricos gratuitos sin clave de API.

**Estación de referencia:** Quinta Normal, Santiago (-33.4447, -70.6828) — estación meteorológica oficial WMO para Santiago.

**Período:** 2016–2025 (~3,650 días)