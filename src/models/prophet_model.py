"""Modelo Prophet para series temporales con estacionalidad"""

import pandas as pd
from prophet import Prophet
import mlflow
from src.models.base_model import BasePredictor
from typing import Dict, Any, List, Tuple

class ProphetPredictor(BasePredictor):
    """Predictor usando Prophet de Meta"""
    
    def __init__(
        self, 
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        super().__init__(model_name="Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        
    def fit(self, train: pd.Series, **kwargs):
        """Entrenar Prophet"""
        
        # Prophet requiere formato específico: ds (fecha), y (valor)
        df_prophet = pd.DataFrame({
            'ds': train.index,
            'y': train.values
        })
        
        # Crear modelo
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )
        
        # Entrenar (silenciar warnings)
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        self.model.fit(df_prophet)
        
        # Log parámetros
        if mlflow.active_run():
            mlflow.log_param("yearly_seasonality", self.yearly_seasonality)
            mlflow.log_param("weekly_seasonality", self.weekly_seasonality)
            mlflow.log_param("changepoint_prior_scale", self.changepoint_prior_scale)
            mlflow.log_param("seasonality_prior_scale", self.seasonality_prior_scale)
        
        print(f"✅ Prophet entrenado")
        
    def predict(self, steps: int) -> pd.Series:
        """Generar predicciones"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        # Crear dataframe futuro
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        
        # Predecir
        forecast = self.model.predict(future)
        
        # Retornar solo las predicciones futuras
        predictions = forecast['yhat'].tail(steps)
        
        # Valores negativos a 0 (no puede llover negativo)
        predictions = predictions.clip(lower=0)
        
        return pd.Series(predictions.values)
    
    def get_params(self) -> Dict[str, Any]:
        """Retornar hiperparámetros del modelo"""
        return {
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale
        }