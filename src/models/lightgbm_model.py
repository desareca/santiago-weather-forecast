"""Modelo LightGBM para predicción de precipitación next-day"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional
from src.utils.config import RANDOM_SEED, TARGET
from src.data.preprocessing import get_X_y, get_feature_names
from src.models.base_model import BasePredictor

class LightGBMPredictor(BasePredictor):
    """
    Predictor next-day de precipitación usando LightGBM.

    Responsabilidades:
        - Entrenar con X, y
        - Predecir dado un DataFrame con features
        - Retornar métricas de importancia de features
        - NO sabe nada de MLflow, CV ni preprocessing
    """

    # Parámetros por defecto — buen punto de partida para precipitación
    DEFAULT_PARAMS = {
        "objective":               "tweedie",
        "tweedie_variance_power":  1.5,
        "n_estimators":            500,
        "learning_rate":           0.05,
        "max_depth":               6,
        "num_leaves":              31,
        "min_child_samples":       20,
        "subsample":               0.8,
        "colsample_bytree":        0.8,
        "reg_alpha":               0.1,
        "reg_lambda":              1.0,
        "random_state":            RANDOM_SEED,
        "verbose":                 -1,
    }

    def __init__(self, **params):
        """
        Args:
            **params: Hiperparámetros de LightGBM.
                      Sobreescriben los DEFAULT_PARAMS.
        """
        super().__init__(model_name="LightGBM")
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: Optional[list] = None
        self.is_fitted: bool = False

    # ──────────────────────────────────────────
    # Entrenamiento
    # ──────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "LightGBMPredictor":
        """
        Entrena el modelo con un DataFrame de train ya procesado.

        Args:
            train: DataFrame con features y target (sin NaN)

        Returns:
            self — permite encadenar: model.fit(train).predict(test)
        """
        X_train, y_train = get_X_y(train)
        self.feature_names = X_train.columns.tolist()

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        print(f"✅ LightGBM entrenado — {len(self.feature_names)} features, "
              f"{len(X_train)} muestras")
        return self

    # ──────────────────────────────────────────
    # Predicción
    # ──────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones para un DataFrame con features.

        Args:
            df: DataFrame con las mismas features que train (puede incluir target)

        Returns:
            Serie de predicciones indexada por fecha, clippeada a >= 0
        """
        self._check_fitted()
        X, _ = get_X_y(df)
        X = self._align_features(X)

        preds = self.model.predict(X)
        preds = np.clip(preds, 0, None)  # precipitación no puede ser negativa

        return pd.Series(preds, index=df.index, name="precip_pred")

    def predict_one(self, row: pd.DataFrame) -> float:
        """
        Predice un solo día. Útil en producción.

        Args:
            row: DataFrame de una sola fila con features

        Returns:
            Predicción en mm (float)
        """
        self._check_fitted()
        X, _ = get_X_y(row)
        X = self._align_features(X)
        pred = self.model.predict(X)[0]
        return float(np.clip(pred, 0, None))

    # ──────────────────────────────────────────
    # Información del modelo
    # ──────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna importancia de features ordenada.

        Args:
            top_n: Número de features a retornar

        Returns:
            DataFrame con columnas ['feature', 'importance', 'importance_pct']
        """
        self._check_fitted()
        importance = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance["importance_pct"] = (
            importance["importance"] / importance["importance"].sum() * 100
        ).round(2)

        return importance.head(top_n)

    def get_params(self) -> Dict[str, Any]:
        """Retorna hiperparámetros del modelo"""
        return self.params.copy()

    def print_summary(self) -> None:
        """Imprime resumen del modelo"""
        self._check_fitted()
        print(f"\n{'='*50}")
        print(f"LightGBM — Resumen")
        print(f"{'='*50}")
        print(f"  Objetivo:      {self.params['objective']}")
        print(f"  N estimators:  {self.params['n_estimators']}")
        print(f"  Learning rate: {self.params['learning_rate']}")
        print(f"  Max depth:     {self.params['max_depth']}")
        print(f"  Num leaves:    {self.params['num_leaves']}")
        print(f"  N features:    {len(self.feature_names)}")
        print(f"\n  Top 10 features:")
        top10 = self.get_feature_importance(top_n=10)
        for _, row in top10.iterrows():
            bar = "█" * int(row["importance_pct"] / 2)
            print(f"    {row['feature']:35s} {row['importance_pct']:5.1f}%  {bar}")

    # ──────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura que X tenga exactamente las mismas columnas que en train,
        en el mismo orden. Agrega columnas faltantes con 0.
        Necesario para que train y test/producción sean compatibles.
        """
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            print(f"⚠️  Features faltantes en predict, se rellenan con 0: {missing}")
            for col in missing:
                X[col] = 0
        return X[self.feature_names]