"""Clase base abstracta para modelos de predicción"""

from abc import ABC, abstractmethod
import pandas as pd
import mlflow
from typing import Dict, Any
from src.evaluation.metrics import evaluate_model, plot_predictions


class BasePredictor(ABC):
    """Clase base para todos los modelos predictivos"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.metrics = None
        
    @abstractmethod
    def fit(self, train: pd.Series, **kwargs):
        """Entrenar el modelo"""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones"""
        pass
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Evaluar modelo"""
        self.metrics = evaluate_model(y_true, y_pred, self.model_name)
        return self.metrics
    
    def train_and_evaluate(
        self, 
        train: pd.Series, 
        test: pd.Series,
        log_mlflow: bool = True,
        **fit_kwargs
    ) -> Dict[str, float]:
        """Pipeline completo: entrenar, predecir, evaluar"""
        
        if log_mlflow:
            with mlflow.start_run(run_name=self.model_name):
                return self._train_eval_pipeline(train, test, **fit_kwargs)
        else:
            return self._train_eval_pipeline(train, test, **fit_kwargs)
    
    def _train_eval_pipeline(self, train: pd.Series, test: pd.Series, **fit_kwargs):
        """Pipeline interno"""
        
        # Entrenar
        print(f"\n🚀 Entrenando {self.model_name}...")
        self.fit(train, **fit_kwargs)
        
        # Predecir
        print(f"🔮 Generando predicciones...")
        predictions = self.predict(steps=len(test))
        predictions.index = test.index
        
        # Evaluar
        metrics = self.evaluate(test, predictions)
        
        # Log MLflow
        if mlflow.active_run():
            # Parámetros
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("train_size", len(train))
            mlflow.log_param("test_size", len(test))
            
            # Métricas
            mlflow.log_metrics(metrics)
            
            # Plot
            fig = plot_predictions(test, predictions, self.model_name)
            mlflow.log_figure(fig, "predictions.png")
            
            # Modelo
            mlflow.sklearn.log_model(self.model, "model")
        
        return metrics