"""Modelo LightGBM para predicción de precipitación con features"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
from src.models.base_model import BasePredictor
from typing import List
from typing import Dict, Any, List, Tuple

class LightGBMPredictor(BasePredictor):
    """Predictor usando LightGBM con features de lag y estacionales"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        lags: List[int] = None,
        rolling_windows: List[int] = None
    ):
        super().__init__(model_name="LightGBM")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lags = lags if lags is not None else [1, 7, 30]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [7, 30]
        self.feature_names = None
        self.train_history = None
        
    def _create_features(self, serie: pd.Series) -> pd.DataFrame:
        """
        Crea features de lag, rolling y temporales.
        
        Args:
            serie: Serie temporal de precipitación
            
        Returns:
            DataFrame con features
        """
        df = pd.DataFrame({'precipitacion': serie})
        
        # Features de lag
        for lag in self.lags:
            df[f'lag_{lag}'] = df['precipitacion'].shift(lag)
        
        # Rolling windows
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df['precipitacion'].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df['precipitacion'].shift(1).rolling(window).std()
            df[f'rolling_max_{window}'] = df['precipitacion'].shift(1).rolling(window).max()
            df[f'rolling_min_{window}'] = df['precipitacion'].shift(1).rolling(window).min()
        
        # Features temporales
        df['mes'] = df.index.month
        df['dia_año'] = df.index.dayofyear
        df['dia_mes'] = df.index.day
        df['trimestre'] = df.index.quarter
        
        # Estación del año (hemisferio sur)
        df['estacion_verano'] = df['mes'].isin([12, 1, 2]).astype(int)
        df['estacion_otoño'] = df['mes'].isin([3, 4, 5]).astype(int)
        df['estacion_invierno'] = df['mes'].isin([6, 7, 8]).astype(int)
        df['estacion_primavera'] = df['mes'].isin([9, 10, 11]).astype(int)
        
        # Indicador de mes lluvioso (Jun-Ago en Santiago)
        df['mes_lluvioso'] = df['mes'].isin([6, 7, 8]).astype(int)
        
        # Features cíclicas (sin, cos para capturar ciclicidad)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_año_sin'] = np.sin(2 * np.pi * df['dia_año'] / 365)
        df['dia_año_cos'] = np.cos(2 * np.pi * df['dia_año'] / 365)
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        return df
    
    def fit(self, train: pd.Series, **kwargs):
        """Entrenar LightGBM"""
        
        # Guardar histórico completo para predicción
        self.train_history = train.copy()
        
        # Crear features
        df_train = self._create_features(train)
        
        # Separar X e y
        X_train = df_train.drop('precipitacion', axis=1)
        y_train = df_train['precipitacion']
        
        # Guardar nombres de features
        self.feature_names = X_train.columns.tolist()
        
        # Crear modelo
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            verbose=-1
        )
        
        # Entrenar
        self.model.fit(X_train, y_train)
        
        # Log parámetros en MLflow
        if mlflow.active_run():
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("num_leaves", self.num_leaves)
            mlflow.log_param("lags", str(self.lags))
            mlflow.log_param("rolling_windows", str(self.rolling_windows))
            mlflow.log_param("n_features", len(self.feature_names))
        
        print(f"✅ LightGBM entrenado con {len(self.feature_names)} features")
    
    def predict(self, steps: int) -> pd.Series:
        """
        Predicción iterativa usando el histórico de entrenamiento.
        
        Args:
            steps: Número de pasos a predecir
            
        Returns:
            Serie con predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        if self.train_history is None:
            raise ValueError("No hay histórico de entrenamiento guardado.")
        
        predictions = []
        current_history = self.train_history.copy()
        
        for i in range(steps):
            # Crear features con histórico actual
            df_features = self._create_features(current_history)
            
            # Tomar última fila (la más reciente)
            X_next = df_features.drop('precipitacion', axis=1).iloc[[-1]]
            
            # Predecir
            pred = self.model.predict(X_next)[0]
            
            # No puede llover negativo
            pred = max(0, pred)
            
            predictions.append(pred)
            
            # Agregar predicción al histórico para siguiente iteración
            next_date = current_history.index[-1] + pd.Timedelta(days=1)
            current_history[next_date] = pred
        
        return pd.Series(predictions)
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """Override para mostrar feature importance"""
        from src.evaluation.metrics import evaluate_model
        
        metrics = evaluate_model(y_true, y_pred, self.model_name)
        
        # Feature importance
        if self.model and self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top 10 Features más importantes:")
            print(importance.head(10).to_string(index=False))
            
            # Log en MLflow
            if mlflow.active_run():
                for idx, row in importance.head(10).iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """Retornar hiperparámetros del modelo"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'lags': str(self.lags),
            'rolling_windows': str(self.rolling_windows)
        }