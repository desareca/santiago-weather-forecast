"""
Módulo de persistencia SQLite para predicciones y datos reales.

Tablas:
    predictions  — predicción diaria generada por el modelo
    actuals      — precipitación real observada (llega con 1 día de delay)
    retraining_log — registro de evaluaciones y reentrenamientos mensuales

Diseño:
    - SQLite es suficiente para este volumen (~365 filas/año)
    - No requiere servidor externo — funciona en el free tier de Render
    - El archivo .db se pierde al reiniciar Render (ephemeral storage)
      → las predicciones importantes deben consultarse antes del reinicio
      → para persistencia real, usar Render Persistent Disk ($7/mes) o
        guardar un CSV en HF Hub periódicamente
"""

import sqlite3
import os
import pandas as pd
from datetime import date, datetime, timezone
from typing import Optional
from contextlib import contextmanager


# ── Ruta de la base de datos ────────────────────────────────────────
# En Render: /tmp es ephemeral. Para persistencia, montar disco en /data.
DB_PATH = os.getenv("DB_PATH", "/tmp/santiago_weather.db")


# ──────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """Context manager para conexiones SQLite con cierre automático."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # acceso por nombre de columna
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Crea las tablas si no existen.
    Idempotente — se puede llamar en cada arranque del servicio.
    """
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                fecha           TEXT PRIMARY KEY,   -- YYYY-MM-DD (día al que refiere la predicción)
                predicted_at    TEXT NOT NULL,       -- ISO timestamp de cuándo se generó
                prob_rain       REAL NOT NULL,       -- P(llueve) [0, 1]
                mm_predicted    REAL NOT NULL,       -- mm predichos
                will_rain       INTEGER NOT NULL,    -- 1 si prob_rain > threshold
                model_version   TEXT,                -- versión del modelo usado
                threshold_used  REAL                 -- threshold aplicado
            );

            CREATE TABLE IF NOT EXISTS actuals (
                fecha           TEXT PRIMARY KEY,   -- YYYY-MM-DD
                mm_actual       REAL NOT NULL,       -- precipitación real observada
                did_rain        INTEGER NOT NULL,    -- 1 si mm_actual > 0.5mm
                recorded_at     TEXT NOT NULL        -- cuándo se registró este dato
            );

            CREATE TABLE IF NOT EXISTS retraining_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluated_at    TEXT NOT NULL,       -- cuándo se corrió la evaluación
                period_start    TEXT NOT NULL,       -- inicio del período evaluado
                period_end      TEXT NOT NULL,       -- fin del período evaluado
                n_days          INTEGER NOT NULL,    -- días con datos reales en el período
                rmse_current    REAL,                -- RMSE del período
                recall_current  REAL,                -- Recall lluvia del período
                rmse_baseline   REAL,                -- RMSE baseline del metadata.json
                recall_baseline REAL,                -- Recall baseline del metadata.json
                degraded        INTEGER NOT NULL,    -- 1 si se detectó degradación
                retrained       INTEGER NOT NULL,    -- 1 si se reentrenó
                new_model_version TEXT,              -- versión del nuevo modelo (si reentrenó)
                notes           TEXT                 -- info adicional o errores
            );
        """)
    print(f"✅ Base de datos inicializada: {DB_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCIONES
# ──────────────────────────────────────────────────────────────────────────────

def save_prediction(
    fecha: date,
    prob_rain: float,
    mm_predicted: float,
    will_rain: bool,
    model_version: str,
    threshold_used: float,
) -> None:
    """
    Guarda o reemplaza la predicción de un día.

    Args:
        fecha:          Día al que refiere la predicción (mañana)
        prob_rain:      Probabilidad de lluvia [0, 1]
        mm_predicted:   mm predichos
        will_rain:      True si prob_rain > threshold
        model_version:  Versión del modelo (desde metadata.json)
        threshold_used: Threshold aplicado
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (fecha, predicted_at, prob_rain, mm_predicted, will_rain,
                 model_version, threshold_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(fecha),
            datetime.now(timezone.utc).isoformat(),
            round(float(prob_rain), 4),
            round(float(mm_predicted), 3),
            int(will_rain),
            model_version,
            round(float(threshold_used), 3),
        ))


def get_prediction(fecha: date) -> Optional[dict]:
    """Retorna la predicción de un día específico, o None si no existe."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE fecha = ?", (str(fecha),)
        ).fetchone()
    return dict(row) if row else None


def get_recent_predictions(n: int = 30) -> pd.DataFrame:
    """Retorna las últimas n predicciones ordenadas por fecha DESC."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY fecha DESC LIMIT ?",
            conn, params=(n,)
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# DATOS REALES
# ──────────────────────────────────────────────────────────────────────────────

def save_actual(
    fecha: date,
    mm_actual: float,
    rain_threshold: float = 0.5,
) -> None:
    """
    Guarda o reemplaza el dato real de precipitación de un día.

    Args:
        fecha:          Día observado
        mm_actual:      mm reales registrados
        rain_threshold: Umbral para clasificar como lluvia (default 0.5mm)
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO actuals
                (fecha, mm_actual, did_rain, recorded_at)
            VALUES (?, ?, ?, ?)
        """, (
            str(fecha),
            round(float(mm_actual), 3),
            int(mm_actual > rain_threshold),
            datetime.now(timezone.utc).isoformat(),
        ))


def get_actuals_for_evaluation(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Retorna datos reales en un rango de fechas junto con las predicciones
    correspondientes (join). Usado por el monthly_flow para evaluar degradación.

    Returns:
        DataFrame con columnas: fecha, mm_actual, did_rain,
                                prob_rain, mm_predicted, will_rain
    """
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT
                a.fecha,
                a.mm_actual,
                a.did_rain,
                p.prob_rain,
                p.mm_predicted,
                p.will_rain,
                p.model_version
            FROM actuals a
            LEFT JOIN predictions p ON a.fecha = p.fecha
            WHERE a.fecha >= ? AND a.fecha <= ?
            ORDER BY a.fecha ASC
        """, conn, params=(str(start_date), str(end_date)))
    return df


def count_actuals(start_date: date, end_date: date) -> int:
    """Cuenta cuántos días reales hay en un rango."""
    with get_connection() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM actuals WHERE fecha >= ? AND fecha <= ?",
            (str(start_date), str(end_date))
        ).fetchone()
    return result[0]


# ──────────────────────────────────────────────────────────────────────────────
# RETRAINING LOG
# ──────────────────────────────────────────────────────────────────────────────

def log_evaluation(
    period_start: date,
    period_end: date,
    n_days: int,
    rmse_current: Optional[float],
    recall_current: Optional[float],
    rmse_baseline: float,
    recall_baseline: float,
    degraded: bool,
    retrained: bool,
    new_model_version: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Registra el resultado de una evaluación mensual."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO retraining_log
                (evaluated_at, period_start, period_end, n_days,
                 rmse_current, recall_current, rmse_baseline, recall_baseline,
                 degraded, retrained, new_model_version, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            str(period_start),
            str(period_end),
            n_days,
            round(float(rmse_current), 4) if rmse_current is not None else None,
            round(float(recall_current), 4) if recall_current is not None else None,
            round(float(rmse_baseline), 4),
            round(float(recall_baseline), 4),
            int(degraded),
            int(retrained),
            new_model_version,
            notes,
        ))


def get_retraining_history() -> pd.DataFrame:
    """Retorna el historial completo de evaluaciones y reentrenamientos."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM retraining_log ORDER BY evaluated_at DESC",
            conn
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────

def get_db_stats() -> dict:
    """Retorna estadísticas básicas de la base de datos."""
    with get_connection() as conn:
        n_preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_actuals = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
        n_retrain = conn.execute("SELECT COUNT(*) FROM retraining_log").fetchone()[0]

        last_pred = conn.execute(
            "SELECT fecha FROM predictions ORDER BY fecha DESC LIMIT 1"
        ).fetchone()
        last_actual = conn.execute(
            "SELECT fecha FROM actuals ORDER BY fecha DESC LIMIT 1"
        ).fetchone()

    return {
        "db_path":      DB_PATH,
        "n_predictions": n_preds,
        "n_actuals":    n_actuals,
        "n_retraining_events": n_retrain,
        "last_prediction_date":  last_pred[0] if last_pred else None,
        "last_actual_date":      last_actual[0] if last_actual else None,
    }