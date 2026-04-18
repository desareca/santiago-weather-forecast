"""
API REST — Santiago Weather Forecast

Endpoints:
    GET /health          — estado del servicio y modelo
    GET /predict/today   — predicción para mañana (usa datos de hoy)
    GET /predict/{fecha} — predicción guardada para una fecha específica
    GET /history         — últimas N predicciones
    GET /model-info      — info del modelo en producción
    POST /flows/daily    — triggerear el daily_flow manualmente
    POST /flows/monthly  — triggerear el monthly_flow manualmente

El modelo se carga desde HF Hub al arrancar (startup event).
Las predicciones se guardan en SQLite.
"""

import os
import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi.responses import HTMLResponse
from pathlib import Path
import numpy as np
from src.storage.database import (
    init_db, get_prediction, get_recent_predictions,
    get_db_stats, get_actuals_for_evaluation,   # ← nuevo
    get_retraining_history,                      # ← nuevo
)

from src.storage.hf_model import ModelRegistry
from src.flows.daily_flow import daily_flow, get_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y la DB al arrancar el servicio."""
    logger.info("🚀 Iniciando Santiago Weather Forecast API...")

    # Inicializar DB
    init_db()
    logger.info("✅ Base de datos lista")

    # Cargar modelo desde HF Hub
    registry = get_registry()
    logger.info(f"✅ Modelo cargado | versión={registry.version}")

    # Correr daily_flow al arrancar para tener predicción del día
    try:
        result = daily_flow()
        logger.info(f"✅ Predicción inicial generada: {result}")
    except Exception as e:
        logger.warning(f"⚠️ daily_flow inicial falló (no crítico): {e}")

    yield

    logger.info("👋 Apagando servicio...")


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Santiago Weather Forecast",
    description="Predicción diaria de precipitación para Santiago de Chile",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    fecha:          str
    prob_rain:      float
    mm_predicted:   float
    will_rain:      bool
    model_version:  str
    threshold_used: float
    predicted_at:   Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "fecha":          "2026-03-16",
                "prob_rain":      0.342,
                "mm_predicted":   3.7,
                "will_rain":      True,
                "model_version":  "20260315_0800",
                "threshold_used": 0.3,
                "predicted_at":   "2026-03-15T08:00:00+00:00",
            }
        }


class HealthResponse(BaseModel):
    status:          str
    model_version:   Optional[str]
    model_train_end: Optional[str]
    last_prediction: Optional[str]
    db_stats:        dict


class ModelInfoResponse(BaseModel):
    version:          str
    train_end:        str
    n_features:       int
    clf_name:         str
    reg_name:         str
    threshold:        float
    baseline_metrics: dict
    degradation_thresholds: dict


class FlowResult(BaseModel):
    status:  str
    message: str
    result:  Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Estado del servicio: modelo cargado, última predicción, stats de la DB."""
    registry = get_registry()
    db_stats  = get_db_stats()

    return HealthResponse(
        status="ok" if registry.is_loaded() else "degradado",
        model_version=registry.version,
        model_train_end=registry.metadata.get("train_end") if registry.metadata else None,
        last_prediction=db_stats.get("last_prediction_date"),
        db_stats=db_stats,
    )


@app.get("/predict/today", response_model=PredictionResponse, tags=["Predicción"])
async def predict_today():
    """
    Retorna la predicción para mañana.

    Si ya existe en la DB (generada por el daily_flow), la devuelve directamente.
    Si no existe (ej. primer request del día), la genera en el momento.
    """
    tomorrow = date.today() + timedelta(days=1)

    # Buscar en DB primero
    stored = get_prediction(tomorrow)
    if stored:
        return PredictionResponse(**stored)

    # No existe → generar ahora
    logger.info(f"Predicción para {tomorrow} no encontrada en DB, generando...")
    try:
        result = daily_flow()
        stored = get_prediction(tomorrow)
        if stored:
            return PredictionResponse(**stored)
        # Si aún no está (edge case), devolver el resultado del flow
        return PredictionResponse(
            fecha=result["fecha_prediccion"],
            prob_rain=result["prob_rain"],
            mm_predicted=result["mm_predicted"],
            will_rain=result["will_rain"],
            model_version=result["model_version"],
            threshold_used=result["threshold_used"],
        )
    except Exception as e:
        logger.error(f"Error generando predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando predicción: {str(e)}")


@app.get("/predict/{fecha}", response_model=PredictionResponse, tags=["Predicción"])
async def predict_by_date(fecha: str):
    """
    Retorna la predicción guardada para una fecha específica (YYYY-MM-DD).

    Solo retorna predicciones que fueron generadas y guardadas en la DB.
    No genera predicciones para fechas pasadas.
    """
    try:
        fecha_date = date.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Usar YYYY-MM-DD")

    stored = get_prediction(fecha_date)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"No hay predicción guardada para {fecha}. "
                   f"Las predicciones se generan automáticamente cada día."
        )

    return PredictionResponse(**stored)


@app.get("/history", response_model=list[PredictionResponse], tags=["Predicción"])
async def prediction_history(n: int = 30):
    """
    Retorna las últimas N predicciones (default: 30).
    Ordenadas por fecha descendente.
    """
    if n < 1 or n > 365:
        raise HTTPException(status_code=400, detail="n debe estar entre 1 y 365")

    df = get_recent_predictions(n)
    if df.empty:
        return []

    return [
        PredictionResponse(**row)
        for _, row in df.iterrows()
    ]


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def model_info():
    """Info completa del modelo actualmente en producción."""
    registry = get_registry()

    if not registry.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    meta = registry.metadata
    return ModelInfoResponse(
        version=registry.version,
        train_end=meta.get("train_end", ""),
        n_features=len(registry.model.feature_names),
        clf_name=meta.get("clf_name", ""),
        reg_name=meta.get("reg_name", ""),
        threshold=registry.threshold,
        baseline_metrics=registry.baseline_metrics,
        degradation_thresholds=registry.degradation_thresholds,
    )


@app.post("/flows/daily", response_model=FlowResult, tags=["Flows"])
async def trigger_daily_flow(background_tasks: BackgroundTasks):
    """
    Triggerear el daily_flow manualmente.

    Útil para forzar una actualización de la predicción o
    para testear el pipeline sin esperar al cron job.
    Corre en background para no bloquear el request.
    """
    def run_flow():
        try:
            result = daily_flow()
            logger.info(f"daily_flow manual completado: {result}")
        except Exception as e:
            logger.error(f"daily_flow manual falló: {e}")

    background_tasks.add_task(run_flow)

    return FlowResult(
        status="accepted",
        message="daily_flow iniciado en background. Consultar /predict/today en unos segundos.",
    )


@app.post("/flows/monthly", response_model=FlowResult, tags=["Flows"])
async def trigger_monthly_flow(background_tasks: BackgroundTasks):
    """
    Triggerear el monthly_flow manualmente.

    Evalúa degradación y reentrenar si es necesario.
    Corre en background — puede tardar varios minutos si reentrenar.
    """
    from src.flows.monthly_flow import monthly_flow

    def run_flow():
        try:
            result = monthly_flow()
            logger.info(f"monthly_flow manual completado: {result}")
        except Exception as e:
            logger.error(f"monthly_flow manual falló: {e}")

    background_tasks.add_task(run_flow)

    return FlowResult(
        status="accepted",
        message="monthly_flow iniciado en background. Puede tardar varios minutos si reentrenar.",
    )


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
async def dashboard():
    """Sirve el dashboard HTML del proyecto."""
    html_path = Path(__file__).resolve().parent / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/dashboard-data", tags=["Dashboard"])
async def dashboard_data():
    """
    Retorna todos los datos necesarios para el dashboard en una sola llamada.

    Incluye:
    - model: info del modelo activo
    - series: últimos 30 días con predicted + observed + error
    - metrics: MAE, RMSE, recall calculados sobre los días con observación real
    """
    registry = get_registry()
    meta     = registry.metadata or {}

    # Período: últimos 30 días con datos
    end_date   = date.today() - timedelta(days=1)   # ayer (último con observed)
    start_date = end_date - timedelta(days=29)       # 30 días

    df = get_actuals_for_evaluation(start_date, end_date)

    # ── Series ────────────────────────────────────────────────────────────────
    series = []
    for _, row in df.iterrows():
        pred = row.get("mm_predicted")
        obs  = row.get("mm_actual")
        err  = abs(float(pred) - float(obs)) if (pred is not None and obs is not None) else None

        series.append({
            "fecha":      row["fecha"],
            "predicted":  round(float(pred), 3) if pred is not None else None,
            "observed":   round(float(obs),  3) if obs  is not None else None,
            "prob_rain":  round(float(row["prob_rain"]), 4) if row.get("prob_rain") is not None else None,
            "will_rain":  bool(row["will_rain"]) if row.get("will_rain") is not None else None,
            "did_rain":   bool(row["did_rain"])  if row.get("did_rain")  is not None else None,
            "error":      round(err, 3) if err is not None else None,
            "model_version": row.get("model_version"),
        })

    # Ordenar por fecha ascendente para los gráficos
    series.sort(key=lambda x: x["fecha"])

    # ── Métricas (solo días con ambos valores) ────────────────────────────────
    paired = [(s["predicted"], s["observed"], s["will_rain"], s["did_rain"])
              for s in series if s["predicted"] is not None and s["observed"] is not None]

    if paired:
        preds   = np.array([p[0] for p in paired])
        obs_arr = np.array([p[1] for p in paired])
        errors  = np.abs(preds - obs_arr)

        mae  = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Recall: días lluviosos reales detectados
        rain_days = [(p[2], p[3]) for p in paired if p[3] is True]  # (will_rain, did_rain)
        recall = float(sum(1 for w, d in rain_days if w is True) / len(rain_days)) if len(rain_days) >= 3 else None

        n_rain_observed = len(rain_days)
    else:
        mae = rmse = recall = None
        n_rain_observed = 0

    baseline_rmse = registry.baseline_metrics.get("rmse") if registry.baseline_metrics else None
    deg_threshold = registry.degradation_thresholds.get("rmse_max_pct_increase") if registry.degradation_thresholds else None

    metrics = {
        "mae":               round(mae,  3) if mae  is not None else None,
        "rmse":              round(rmse, 3) if rmse is not None else None,
        "recall":            round(recall, 3) if recall is not None else None,
        "n_days":            len(paired),
        "n_rain_days_observed": n_rain_observed,
        "rmse_baseline":     baseline_rmse,
    }

    # ── Model info ────────────────────────────────────────────────────────────
    model_info = {
        "version":                    registry.version,
        "train_end":                  meta.get("train_end"),
        "clf_name":                   meta.get("clf_name"),
        "reg_name":                   meta.get("reg_name"),
        "threshold":                  registry.threshold,
        "baseline_rmse":              baseline_rmse,
        "degradation_threshold_rmse": round(baseline_rmse * (1 + deg_threshold), 3) if (baseline_rmse and deg_threshold) else None,
    }

    return {
        "model":   model_info,
        "series":  series,
        "metrics": metrics,
    }


@app.get("/api/retraining-log", tags=["Dashboard"])
async def retraining_log():
    """
    Historial de evaluaciones mensuales y eventos de reentrenamiento.
    Usado por el gráfico de Model Monitoring del dashboard.
    """
    df = get_retraining_history()
    if df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        records.append({
            "evaluated_at":      row.get("evaluated_at"),
            "period_start":      row.get("period_start"),
            "period_end":        row.get("period_end"),
            "n_days":            int(row["n_days"]) if row.get("n_days") is not None else None,
            "rmse_current":      round(float(row["rmse_current"]), 4) if row.get("rmse_current") is not None else None,
            "recall_current":    round(float(row["recall_current"]), 4) if row.get("recall_current") is not None else None,
            "rmse_baseline":     round(float(row["rmse_baseline"]), 4) if row.get("rmse_baseline") is not None else None,
            "recall_baseline":   round(float(row["recall_baseline"]), 4) if row.get("recall_baseline") is not None else None,
            "degraded":          bool(row["degraded"]),
            "retrained":         bool(row["retrained"]),
            "new_model_version": row.get("new_model_version"),
            "notes":             row.get("notes"),
        })

    return records

# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT LOCAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )