"""Two-Stage model para predicción de precipitación.

Etapa 1 — Clasificador: ¿Llueve mañana? (binario)
Etapa 2 — Regresor:     ¿Cuánto llueve? (solo días lluviosos)

Predice: P(llueve) > threshold → reg.predict(X) | 0
"""

import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import lightgbm as lgb
from typing import Dict, Any, Optional

from src.utils.config import TARGET, CLF_RAIN_THRESHOLD, REG_RAIN_THRESHOLD


class TwoStagePredictor:
    """
    Predictor two-stage para precipitación diaria.

    Parámetros
    ----------
    clf_params : dict
        Hiperparámetros para el LGBMClassifier (etapa 1).
    reg_params : dict
        Hiperparámetros para el LGBMRegressor (etapa 2).
    threshold : float
        Umbral de probabilidad del clasificador para predecir lluvia.
    log_target : bool
        Si True, transforma el target del regresor con log1p / expm1.
        Comprime la cola larga y mejora el ajuste en magnitudes bajas.
    clf_rain_threshold : float
        mm mínimos para etiquetar positivos en el clasificador (default 0.5mm).
    reg_rain_threshold : float
        mm mínimos para entrenar el regresor — filtra lloviznas (default 1.0mm).
    """

    def __init__(
        self,
        clf_params:         Optional[Dict[str, Any]] = None,
        reg_params:         Optional[Dict[str, Any]] = None,
        threshold:          float = 0.3,
        log_target:         bool  = True,
        clf_rain_threshold: float = CLF_RAIN_THRESHOLD,
        reg_rain_threshold: float = REG_RAIN_THRESHOLD,
    ):
        self.clf_params         = clf_params or {}
        self.reg_params         = reg_params or {}
        self.threshold          = threshold
        self.log_target         = log_target
        self.clf_rain_threshold = clf_rain_threshold
        self.reg_rain_threshold = reg_rain_threshold

        self.clf = None
        self.reg = None
        self.feature_names: list = []
        self.model_name = "TwoStage"

    # ──────────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Entrena clf y reg.

        Parameters
        ----------
        df : DataFrame con features X y columna TARGET (precip_t1).
        """
        from src.data.preprocessing import get_feature_names

        self.feature_names = get_feature_names(df)
        X = df[self.feature_names]
        y = df[TARGET]

        # ── Etapa 1: clasificador binario ──────────────────────────
        y_clf = (y > self.clf_rain_threshold).astype(int)

        clf_p = {
            "objective":   "binary",
            "is_unbalance": True,       # compensar desbalance (0.5mm: ~20% positivos)
            "verbose":     -1,
            "random_state": 42,
        }
        clf_p.update(self.clf_params)

        self.clf = lgb.LGBMClassifier(**clf_p)
        self.clf.fit(X, y_clf)

        n_lluvia = y_clf.sum()
        print(f"✅ Clasificador entrenado — {len(X)} muestras "
              f"({n_lluvia} lluviosas >{self.clf_rain_threshold}mm, {len(X)-n_lluvia} secas)")

        # ── Etapa 2: regresor de magnitud ─────────────────────────
        # reg_rain_threshold (1.0mm): entrena solo en lluvia real, filtra lloviznas
        mask_lluvia = y > self.reg_rain_threshold
        X_rain = X[mask_lluvia]
        y_rain = y[mask_lluvia]

        if self.log_target:
            y_rain_fit = np.log1p(y_rain)
        else:
            y_rain_fit = y_rain

        reg_p = {
            "objective":    "regression",
            "verbose":      -1,
            "random_state": 42,
        }
        reg_p.update(self.reg_params)

        self.reg = lgb.LGBMRegressor(**reg_p)
        self.reg.fit(X_rain, y_rain_fit)

        print(f"✅ Regresor entrenado   — {len(X_rain)} días lluviosos "
              f"({'log1p' if self.log_target else 'lineal'})")

        # Log MLflow si hay run activo
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_param("threshold",      self.threshold)
            mlflow.log_param("log_target",     self.log_target)
            mlflow.log_param("clf_rain_threshold", self.clf_rain_threshold)
            mlflow.log_param("reg_rain_threshold", self.reg_rain_threshold)
            mlflow.log_param("n_train",        len(X))
            mlflow.log_param("n_train_rain",   int(n_lluvia))
            for k, v in self.clf_params.items():
                mlflow.log_param(f"clf_{k}", v)
            for k, v in self.reg_params.items():
                mlflow.log_param(f"reg_{k}", v)

    # ──────────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones combinando clf y reg.

        Lógica:
            pred = P(llueve) * mm_predichos  si P(llueve) > threshold
            pred = 0.0                        si P(llueve) <= threshold

        Returns
        -------
        pd.Series con las predicciones, mismo índice que df.
        """
        self._check_fitted()
        X = df[self.feature_names]

        # Probabilidades del clasificador
        prob_llueve = self.clf.predict_proba(X)[:, 1]

        # Magnitud del regresor
        mm_raw = self.reg.predict(X)
        if self.log_target:
            mm_pred = np.expm1(mm_raw)
        else:
            mm_pred = mm_raw
        mm_pred = np.maximum(mm_pred, 0.0)   # no puede llover negativo

        # Combinar: escalar mm por probabilidad, cero si bajo el umbral
        pred = np.where(prob_llueve > self.threshold, prob_llueve * mm_pred, 0.0)

        return pd.Series(pred, index=df.index, name="pred")

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Devuelve la probabilidad de lluvia del clasificador."""
        self._check_fitted()
        return self.clf.predict_proba(df[self.feature_names])[:, 1]

    def predict_magnitude(self, df: pd.DataFrame) -> pd.Series:
        """Devuelve la magnitud predicha por el regresor (sin filtro de umbral)."""
        self._check_fitted()
        X = df[self.feature_names]
        mm_raw = self.reg.predict(X)
        mm = np.expm1(mm_raw) if self.log_target else mm_raw
        return pd.Series(np.maximum(mm, 0.0), index=df.index)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Devuelve importancia de features para clf y reg por separado.

        Returns
        -------
        dict con claves 'clf' y 'reg', cada una un DataFrame
        con columnas ['feature', 'importance', 'importance_pct'].
        """
        self._check_fitted()
        result = {}
        for name, model in [("clf", self.clf), ("reg", self.reg)]:
            imp = pd.DataFrame({
                "feature":    self.feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(top_n)
            imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
            result[name] = imp.reset_index(drop=True)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "threshold":      self.threshold,
            "log_target":     self.log_target,
            "clf_rain_threshold": self.clf_rain_threshold,
            "reg_rain_threshold": self.reg_rain_threshold,
            **{f"clf_{k}": v for k, v in self.clf_params.items()},
            **{f"reg_{k}": v for k, v in self.reg_params.items()},
        }

    def print_summary(self) -> None:
        self._check_fitted()
        print("\n" + "=" * 50)
        print("TwoStagePredictor — Resumen")
        print("=" * 50)
        print(f"  Umbral:        {self.threshold}")
        print(f"  Log target:    {self.log_target}")
        print(f"  CLF rain threshold: {self.clf_rain_threshold} mm")
        print(f"  REG rain threshold: {self.reg_rain_threshold} mm")
        print(f"\n  Clasificador:")
        for k, v in self.clf_params.items():
            print(f"    {k:25s}: {v}")
        print(f"\n  Regresor:")
        for k, v in self.reg_params.items():
            print(f"    {k:25s}: {v}")
        print("=" * 50)

    def _check_fitted(self):
        if self.clf is None or self.reg is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")