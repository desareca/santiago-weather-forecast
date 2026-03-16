"""
Módulo de persistencia — SQLite local con backup en HF Hub.

El archivo .db vive en /tmp durante la sesión de Render.
Al arrancar se restaura desde HF Hub (si existe).
Al final de cada daily_flow se sube a HF Hub como backup.

Variables de entorno:
    DB_PATH      — ruta local del archivo SQLite (default: /tmp/santiago_weather.db)
    HF_REPO_ID   — repo de HF Hub donde se guarda el backup
    HF_TOKEN     — token de HF Hub con permisos de escritura
"""

import os
import sqlite3
import pandas as pd
from datetime import date, datetime, timezone
from typing import Optional
from contextlib import contextmanager

DB_PATH   = os.getenv("DB_PATH", "/tmp/santiago_weather.db")
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN   = os.getenv("HF_TOKEN")
DB_FILENAME = "santiago_weather.db"


# ──────────────────────────────────────────────────────────────────────────────
# BACKUP / RESTORE desde HF Hub
# ──────────────────────────────────────────────────────────────────────────────

def restore_db_from_hub() -> bool:
    """
    Descarga el backup de la DB desde HF Hub al arrancar.
    Retorna True si se restauró, False si no había backup.
    """
    if not HF_REPO_ID or not HF_TOKEN:
        return False
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=DB_FILENAME,
            cache_dir="/tmp/db_cache",
            token=HF_TOKEN,
            force_download=True,
        )
        import shutil
        shutil.copy(path, DB_PATH)
        print(f"✅ DB restaurada desde HF Hub")
        return True
    except Exception as e:
        print(f"ℹ️ Sin backup previo en HF Hub ({e})")
        return False


def backup_db_to_hub() -> bool:
    """
    Sube el archivo .db a HF Hub como backup.
    Se llama al final de cada daily_flow.
    Retorna True si se subió correctamente.
    """
    if not HF_REPO_ID or not HF_TOKEN:
        return False
    if not os.path.exists(DB_PATH):
        return False
    try:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=DB_PATH,
            path_in_repo=DB_FILENAME,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="DB backup",
        )
        print(f"✅ DB backup subido a HF Hub")
        return True
    except Exception as e:
        print(f"⚠️ Error subiendo backup DB: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
    """Crea las tablas si no existen. Idempotente."""
    # Intentar restaurar backup al inicializar
    restore_db_from_hub()

    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                fecha           TEXT PRIMARY KEY,
                predicted_at    TEXT NOT NULL,
                prob_rain       REAL NOT NULL,
                mm_predicted    REAL NOT NULL,
                will_rain       INTEGER NOT NULL,
                model_version   TEXT,
                threshold_used  REAL
            );
            CREATE TABLE IF NOT EXISTS actuals (
                fecha       TEXT PRIMARY KEY,
                mm_actual   REAL NOT NULL,
                did_rain    INTEGER NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retraining_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
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
            );
        """)
    print(f"✅ DB inicializada: {DB_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCIONES
# ──────────────────────────────────────────────────────────────────────────────

def save_prediction(
    fecha: date, prob_rain: float, mm_predicted: float,
    will_rain: bool, model_version: str, threshold_used: float,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO predictions
               (fecha, predicted_at, prob_rain, mm_predicted,
                will_rain, model_version, threshold_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(fecha), datetime.now(timezone.utc).isoformat(),
             round(float(prob_rain), 4), round(float(mm_predicted), 3),
             int(will_rain), model_version, round(float(threshold_used), 3))
        )


def get_prediction(fecha: date) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE fecha = ?", (str(fecha),)
        ).fetchone()
    return dict(row) if row else None


def get_recent_predictions(n: int = 30) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY fecha DESC LIMIT ?",
            conn, params=(n,)
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# DATOS REALES
# ──────────────────────────────────────────────────────────────────────────────

def save_actual(fecha: date, mm_actual: float, rain_threshold: float = 0.5) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO actuals
               (fecha, mm_actual, did_rain, recorded_at)
               VALUES (?, ?, ?, ?)""",
            (str(fecha), round(float(mm_actual), 3),
             int(mm_actual > rain_threshold), datetime.now(timezone.utc).isoformat())
        )


def get_actuals_for_evaluation(start_date: date, end_date: date) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """SELECT a.fecha, a.mm_actual, a.did_rain,
                      p.prob_rain, p.mm_predicted, p.will_rain, p.model_version
               FROM actuals a
               LEFT JOIN predictions p ON a.fecha = p.fecha
               WHERE a.fecha >= ? AND a.fecha <= ?
               ORDER BY a.fecha ASC""",
            conn, params=(str(start_date), str(end_date))
        )
    return df


def count_actuals(start_date: date, end_date: date) -> int:
    with get_connection() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM actuals WHERE fecha >= ? AND fecha <= ?",
            (str(start_date), str(end_date))
        ).fetchone()
    return result[0] if result else 0


# ──────────────────────────────────────────────────────────────────────────────
# RETRAINING LOG
# ──────────────────────────────────────────────────────────────────────────────

def log_evaluation(
    period_start: date, period_end: date, n_days: int,
    rmse_current: Optional[float], recall_current: Optional[float],
    rmse_baseline: float, recall_baseline: float,
    degraded: bool, retrained: bool,
    new_model_version: Optional[str] = None, notes: Optional[str] = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO retraining_log
               (evaluated_at, period_start, period_end, n_days,
                rmse_current, recall_current, rmse_baseline, recall_baseline,
                degraded, retrained, new_model_version, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(),
             str(period_start), str(period_end), n_days,
             round(float(rmse_current), 4) if rmse_current is not None else None,
             round(float(recall_current), 4) if recall_current is not None else None,
             round(float(rmse_baseline), 4), round(float(recall_baseline), 4),
             int(degraded), int(retrained), new_model_version, notes)
        )


def get_retraining_history() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM retraining_log ORDER BY evaluated_at DESC", conn
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────

def get_db_stats() -> dict:
    with get_connection() as conn:
        n_preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_actuals = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
        n_retrain = conn.execute("SELECT COUNT(*) FROM retraining_log").fetchone()[0]
        last_pred   = conn.execute("SELECT fecha FROM predictions ORDER BY fecha DESC LIMIT 1").fetchone()
        last_actual = conn.execute("SELECT fecha FROM actuals ORDER BY fecha DESC LIMIT 1").fetchone()
    return {
        "db_path":               DB_PATH,
        "n_predictions":         n_preds,
        "n_actuals":             n_actuals,
        "n_retraining_events":   n_retrain,
        "last_prediction_date":  last_pred[0] if last_pred else None,
        "last_actual_date":      last_actual[0] if last_actual else None,
    }