"""Modelo LightGBM para predicción de precipitación con features"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
from src.models.base_model import BasePredictor
from typing import List, Dict, Any, Tuple

class LightGBMPredictor(BasePredictor):
    """Predictor usando LightGBM con features de lag y estacionales"""
    
    def __init__(self, lags: List[int] = None, rolling_windows: List[int] = None, horizon: int = 1, **kwargs):
        super().__init__(model_name="LightGBM")
        self.lags = lags if lags is not None else [1, 7, 30]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [7, 30]
        self.feature_names = None
        self.train_history = None
        self.horizon = horizon
        self.model_params = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def _create_features(self, serie: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'precipitacion': serie})
        umbral_lluvia_bajo = 1.0  # mm
        umbral_lluvia_alto = 10.0  # mm
        for lag in self.lags:
            df[f'lag_{lag}'] = df['precipitacion'].shift(lag)
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df['precipitacion'].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df['precipitacion'].shift(1).rolling(window).std()
            df[f'rolling_max_{window}'] = df['precipitacion'].shift(1).rolling(window).max()
            df[f'rolling_min_{window}'] = df['precipitacion'].shift(1).rolling(window).min()
            df[f'rolling_umbral_lluvia_bajo_{window}'] = df['precipitacion'].shift(1).rolling(window).apply(lambda x: (x > umbral_lluvia_bajo).sum())
            df[f'rolling_umbral_lluvia_alto_{window}'] = df['precipitacion'].shift(1).rolling(window).apply(lambda x: (x > umbral_lluvia_alto).sum())
        # Estacionalidad hemisferio sur
        df['estacion_verano'] = df.index.month.isin([12, 1, 2]).astype(int)
        df['estacion_otoño'] = df.index.month.isin([3, 4, 5]).astype(int)
        df['estacion_invierno'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['estacion_primavera'] = df.index.month.isin([9, 10, 11]).astype(int)
        df['mes_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['dia_año_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['dia_año_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        df['target'] = df['precipitacion'].shift(-self.horizon)
        return df
    
    def fit(self, train: pd.Series, **kwargs):
        """Entrenar LightGBM para predecir h días adelante (horizon)."""
        self.train_history = train.copy()
        df_train = self._create_features(train)
        df_train = df_train.dropna()
        X_train = df_train.drop(['target'], axis=1)
        y_train = df_train['target']
        self.feature_names = X_train.columns.tolist()
        model_params = self.model_params.copy()
        model_params.update(kwargs)
        model_params.setdefault('random_state', 42)
        model_params.setdefault('verbose', -1)
        self.model = lgb.LGBMRegressor(**model_params)
        self.model.fit(X_train, y_train)
        if mlflow.active_run():
            for k, v in model_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("lags", str(self.lags))
            mlflow.log_param("rolling_windows", str(self.rolling_windows))
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("horizon", self.horizon)
        print(f"✅ LightGBM entrenado con {len(self.feature_names)} features para horizon={self.horizon}")
    
    def predict(self, steps: int = 1, history: pd.Series = None, autoregressive: bool = True) -> pd.Series:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        if history is None:
            raise ValueError("No hay histórico para predecir.")
        # Para predicción clásica (no autoregresiva): usar el histórico real pasado
        if history is not None and not autoregressive:
            df_features = self._create_features(history)
            X_next = df_features.drop('target', axis=1)
            preds = self.model.predict(X_next)
            return pd.Series(self.model.predict(X_next), index=X_next.index)
        # Para rolling forecast autoregresivo por defecto
        if self.train_history is None:
            raise ValueError("No hay histórico de entrenamiento guardado.")
        predictions = []
        current_history = history.copy() if history is not None else self.train_history.copy()
        for i in range(steps):
            df_features = self._create_features(current_history)
            X_next = df_features.drop('target', axis=1).iloc[[-1]]
            pred = self.model.predict(X_next)[0]
            pred = max(0, pred)
            predictions.append(pred)
            next_date = current_history.index[-1] + pd.Timedelta(days=1)
            current_history[next_date] = pred
        return pd.Series(predictions)
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        from src.evaluation.metrics import evaluate_model
        metrics = evaluate_model(y_true, y_pred, self.model_name)
        if self.model and self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n📊 Top 10 Features más importantes:")
            print(importance.head(10).to_string(index=False))
            if mlflow.active_run():
                for idx, row in importance.head(10).iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        params = self.model_params.copy()
        params['lags'] = str(self.lags)
        params['rolling_windows'] = str(self.rolling_windows)
        return params
