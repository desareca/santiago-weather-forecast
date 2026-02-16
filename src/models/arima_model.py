"""Modelo ARIMA para series temporales"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import mlflow
from src.models.base_model import BasePredictor


class ARIMAPredictor(BasePredictor):
    """Predictor usando ARIMA"""
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        super().__init__(model_name=f"ARIMA({p},{d},{q})")
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        
    def fit(self, train: pd.Series, **kwargs):
        """Entrenar ARIMA"""
        model = ARIMA(train, order=self.order)
        self.model = model.fit()
        
        # Log parámetros en MLflow si hay run activo
        if mlflow.active_run():
            mlflow.log_param("p", self.p)
            mlflow.log_param("d", self.d)
            mlflow.log_param("q", self.q)
        
        print(f"✅ ARIMA entrenado con orden {self.order}")
        
    def predict(self, steps: int) -> pd.Series:
        """Generar predicciones"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        predictions = self.model.forecast(steps=steps)
        return pd.Series(predictions)