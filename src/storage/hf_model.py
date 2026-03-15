"""
Wrapper para gestión del modelo en Hugging Face Hub.

Responsabilidades:
    - Descargar modelo y metadata desde HF Hub al arrancar
    - Subir nuevo modelo cuando se reentrenar en producción
    - Cachear el modelo en memoria para no descargarlo en cada request

Uso típico:
    # Al arrancar el servicio (FastAPI startup)
    registry = ModelRegistry()
    registry.load()

    # En cada predicción
    model = registry.model
    metadata = registry.metadata

    # Después de reentrenar
    registry.upload(new_model, new_metadata)
"""

import os
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, upload_file

logger = logging.getLogger(__name__)


# ── Variables de entorno requeridas en Render ───────────────────────
HF_REPO_ID  = os.getenv("HF_REPO_ID")   # ej. "carlos/santiago-weather"
HF_TOKEN    = os.getenv("HF_TOKEN")     # token con permisos de lectura (y escritura para reentrenar)
CACHE_DIR   = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")


class ModelRegistry:
    """
    Gestiona el ciclo de vida del modelo en Hugging Face Hub.

    Attributes:
        model:    TwoStagePredictor cargado en memoria
        metadata: dict con versión, parámetros y métricas baseline
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.repo_id   = repo_id or HF_REPO_ID
        self.token     = token or HF_TOKEN
        self.cache_dir = cache_dir or CACHE_DIR

        if not self.repo_id:
            raise ValueError(
                "HF_REPO_ID no configurado. "
                "Setear variable de entorno HF_REPO_ID o pasar repo_id al constructor."
            )

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model    = None
        self.metadata = None
        self._api     = HfApi(token=self.token) if self.token else HfApi()

    # ──────────────────────────────────────────────────────────────
    # CARGA
    # ──────────────────────────────────────────────────────────────

    def load(self, force_download: bool = False) -> None:
        """
        Descarga modelo y metadata desde HF Hub y los carga en memoria.

        Args:
            force_download: Si True, ignora el caché local y descarga siempre.
                            Útil después de un reentrenamiento.
        """
        logger.info(f"Cargando modelo desde HF Hub: {self.repo_id}")

        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="two_stage_model.pkl",
                cache_dir=self.cache_dir,
                token=self.token,
                force_download=force_download,
            )
            meta_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="metadata.json",
                cache_dir=self.cache_dir,
                token=self.token,
                force_download=force_download,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error descargando modelo desde {self.repo_id}: {e}"
            ) from e

        self.model = joblib.load(model_path)

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(
            f"✅ Modelo cargado | versión={self.metadata.get('version')} "
            f"| features={len(self.model.feature_names)} "
            f"| train_end={self.metadata.get('train_end')}"
        )

    # ──────────────────────────────────────────────────────────────
    # PROPIEDADES DE CONVENIENCIA
    # ──────────────────────────────────────────────────────────────

    @property
    def version(self) -> Optional[str]:
        return self.metadata.get("version") if self.metadata else None

    @property
    def threshold(self) -> float:
        if self.metadata:
            return float(self.metadata.get("threshold", 0.3))
        return 0.3

    @property
    def baseline_metrics(self) -> dict:
        if self.metadata:
            return self.metadata.get("baseline_metrics", {})
        return {}

    @property
    def degradation_thresholds(self) -> dict:
        if self.metadata:
            return self.metadata.get("degradation_thresholds", {})
        return {
            "rmse_max_pct_increase": 0.20,
            "recall_min": 0.50,
        }

    def is_loaded(self) -> bool:
        return self.model is not None and self.metadata is not None

    # ──────────────────────────────────────────────────────────────
    # UPLOAD (usado por monthly_flow después de reentrenar)
    # ──────────────────────────────────────────────────────────────

    def upload(self, model, metadata: dict) -> str:
        """
        Serializa y sube un nuevo modelo a HF Hub.
        Actualiza el modelo en memoria sin necesitar restart.

        Args:
            model:    TwoStagePredictor ya entrenado
            metadata: dict con versión, parámetros y métricas

        Returns:
            version del modelo subido
        """
        if not self.token:
            raise ValueError(
                "HF_TOKEN no configurado. Se necesita token con permisos de escritura."
            )

        version    = metadata.get("version", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
        model_path = os.path.join(self.cache_dir, "two_stage_model.pkl")
        meta_path  = os.path.join(self.cache_dir, "metadata.json")

        # Serializar localmente
        joblib.dump(model, model_path, compress=3)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Subiendo modelo v{version} a HF Hub ({model_size_mb:.2f} MB)...")

        commit_msg = (
            f"Retrain {version} | "
            f"clf={metadata.get('clf_name')} "
            f"reg={metadata.get('reg_name')} "
            f"thr={metadata.get('threshold')}"
        )

        # Subir ambos archivos
        for local_path, remote_name in [
            (model_path, "two_stage_model.pkl"),
            (meta_path,  "metadata.json"),
        ]:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                commit_message=commit_msg,
            )
            logger.info(f"  ✅ {remote_name} subido")

        # Actualizar en memoria sin re-descargar
        self.model    = model
        self.metadata = metadata

        logger.info(f"✅ Modelo v{version} en producción")
        return version

    # ──────────────────────────────────────────────────────────────
    # INFO
    # ──────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Retorna info del modelo actualmente cargado."""
        if not self.is_loaded():
            return {"status": "no cargado"}
        return {
            "status":        "cargado",
            "version":       self.version,
            "train_end":     self.metadata.get("train_end"),
            "n_features":    len(self.model.feature_names),
            "threshold":     self.threshold,
            "clf_name":      self.metadata.get("clf_name"),
            "reg_name":      self.metadata.get("reg_name"),
            "hf_repo":       self.repo_id,
            "baseline_metrics": self.baseline_metrics,
        }