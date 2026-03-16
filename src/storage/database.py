"""
Módulo de persistencia — Turso (SQLite en la nube)

Variables de entorno requeridas:
    TURSO_URL   — libsql://santiago-weather-xxxxx.turso.io
    TURSO_TOKEN — token de autenticación de Turso

Fallback a SQLite local si las variables no están configuradas (desarrollo).

Tablas:
    predictions     — predicción diaria generada por el modelo
    actuals         — precipitación real observada (1 día de delay)
    retraining_log  — historial de evaluaciones y reentrenamientos mensuales
"""

import os
import libsql_client
import pandas as pd
from datetime import date, datetime, timezone
from typing import Optional


TURSO_URL   = os.getenv("TURSO_URL")
TURSO_TOKEN = os.getenv("TURSO_TOKEN")
DB_PATH     = os.getenv("DB_PATH", "/tmp/santiago_weather.db")


def _get_client():
    """Cliente Turso si hay credenciales, SQLite local como fallback."""
    if TURSO_URL and TURSO_TOKEN:
        return libsql_client.create_client_sync(
            url=TURSO_URL,
            auth_token=TURSO_TOKEN,
        )
    return libsql_client.create_client_sync(url=f"file:{DB_PATH}")


def init_db() -> None:
    """Crea las tablas si no existen. Idempotente."""
    statements = [
        """CREATE TABLE IF NOT EXISTS predictions (
            fecha           TEXT PRIMARY KEY,
            predicted_at    TEXT NOT NULL,
            prob_rain       REAL NOT NULL,
            mm_predicted    REAL NOT NULL,
            will_rain       INTEGER NOT NULL,
            model_version   TEXT,
            threshold_used  REAL
        )""",
        """CREATE TABLE IF NOT EXISTS actuals (
            fecha       TEXT PRIMARY KEY,
            mm_actual   REAL NOT NULL,
            did_rain    INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS retraining_log (
            id                INTEGER PRIMARY KEY,
            evaluated_at      TEXT NOT NULL,
            period_start      TEXT NOT NULL,
            period_end        TEXT NOT NULL,
            n_days            INTEGER NOT NULL,
            rmse_current      REAL,
            recall_current    REAL,
            rmse_baseline     REAL,
            recall_baseline   REAL,
            degraded          INTEGER NOT NULL,
            retrained         INTEGER NOT NULL,
            new_model_version TEXT,
            notes             TEXT
        )""",
    ]
    with _get_client() as c:
        for stmt in statements:
            c.execute(stmt)
    print(f"✅ DB inicializada ({'Turso' if TURSO_URL else 'SQLite local'})")


def save_prediction(
    fecha: date, prob_rain: float, mm_predicted: float,
    will_rain: bool, model_version: str, threshold_used: float,
) -> None:
    with _get_client() as c:
        c.execute(
            """INSERT OR REPLACE INTO predictions
               (fecha, predicted_at, prob_rain, mm_predicted,
                will_rain, model_version, threshold_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [str(fecha), datetime.now(timezone.utc).isoformat(),
             round(float(prob_rain), 4), round(float(mm_predicted), 3),
             int(will_rain), model_version, round(float(threshold_used), 3)]
        )


def get_prediction(fecha: date) -> Optional[dict]:
    with _get_client() as c:
        result = c.execute(
            "SELECT * FROM predictions WHERE fecha = ?", [str(fecha)]
        )
    if not result.rows:
        return None
    cols = [col.name for col in result.columns]
    return dict(zip(cols, result.rows[0]))


def get_recent_predictions(n: int = 30) -> pd.DataFrame:
    with _get_client() as c:
        result = c.execute(
            "SELECT * FROM predictions ORDER BY fecha DESC LIMIT ?", [n]
        )
    if not result.rows:
        return pd.DataFrame()
    cols = [col.name for col in result.columns]
    return pd.DataFrame(result.rows, columns=cols)


def save_actual(fecha: date, mm_actual: float, rain_threshold: float = 0.5) -> None:
    with _get_client() as c:
        c.execute(
            """INSERT OR REPLACE INTO actuals
               (fecha, mm_actual, did_rain, recorded_at)
               VALUES (?, ?, ?, ?)""",
            [str(fecha), round(float(mm_actual), 3),
             int(mm_actual > rain_threshold), datetime.now(timezone.utc).isoformat()]
        )


def get_actuals_for_evaluation(start_date: date, end_date: date) -> pd.DataFrame:
    with _get_client() as c:
        result = c.execute(
            """SELECT a.fecha, a.mm_actual, a.did_rain,
                      p.prob_rain, p.mm_predicted, p.will_rain, p.model_version
               FROM actuals a
               LEFT JOIN predictions p ON a.fecha = p.fecha
               WHERE a.fecha >= ? AND a.fecha <= ?
               ORDER BY a.fecha ASC""",
            [str(start_date), str(end_date)]
        )
    if not result.rows:
        return pd.DataFrame()
    cols = [col.name for col in result.columns]
    return pd.DataFrame(result.rows, columns=cols)


def count_actuals(start_date: date, end_date: date) -> int:
    with _get_client() as c:
        result = c.execute(
            "SELECT COUNT(*) FROM actuals WHERE fecha >= ? AND fecha <= ?",
            [str(start_date), str(end_date)]
        )
    return result.rows[0][0] if result.rows else 0


def log_evaluation(
    period_start: date, period_end: date, n_days: int,
    rmse_current: Optional[float], recall_current: Optional[float],
    rmse_baseline: float, recall_baseline: float,
    degraded: bool, retrained: bool,
    new_model_version: Optional[str] = None, notes: Optional[str] = None,
) -> None:
    with _get_client() as c:
        c.execute(
            """INSERT INTO retraining_log
               (evaluated_at, period_start, period_end, n_days,
                rmse_current, recall_current, rmse_baseline, recall_baseline,
                degraded, retrained, new_model_version, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [datetime.now(timezone.utc).isoformat(),
             str(period_start), str(period_end), n_days,
             round(float(rmse_current), 4) if rmse_current is not None else None,
             round(float(recall_current), 4) if recall_current is not None else None,
             round(float(rmse_baseline), 4), round(float(recall_baseline), 4),
             int(degraded), int(retrained), new_model_version, notes]
        )


def get_retraining_history() -> pd.DataFrame:
    with _get_client() as c:
        result = c.execute(
            "SELECT * FROM retraining_log ORDER BY evaluated_at DESC"
        )
    if not result.rows:
        return pd.DataFrame()
    cols = [col.name for col in result.columns]
    return pd.DataFrame(result.rows, columns=cols)


def get_db_stats() -> dict:
    with _get_client() as c:
        n_preds   = c.execute("SELECT COUNT(*) FROM predictions").rows[0][0]
        n_actuals = c.execute("SELECT COUNT(*) FROM actuals").rows[0][0]
        n_retrain = c.execute("SELECT COUNT(*) FROM retraining_log").rows[0][0]
        last_pred   = c.execute("SELECT fecha FROM predictions ORDER BY fecha DESC LIMIT 1").rows
        last_actual = c.execute("SELECT fecha FROM actuals ORDER BY fecha DESC LIMIT 1").rows
    return {
        "backend":               "turso" if TURSO_URL else "sqlite_local",
        "n_predictions":         n_preds,
        "n_actuals":             n_actuals,
        "n_retraining_events":   n_retrain,
        "last_prediction_date":  last_pred[0][0] if last_pred else None,
        "last_actual_date":      last_actual[0][0] if last_actual else None,
    }