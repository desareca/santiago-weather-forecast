"""Clase base abstracta para modelos de predicción"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BasePredictor(ABC):
    """
    Interfaz que debe cumplir cualquier modelo del proyecto.

    Responsabilidades:
        - Definir contrato de fit/predict/get_params
        - NO contiene lógica de MLflow, CV ni preprocessing
        - NO contiene lógica de evaluación

    Para agregar un modelo nuevo:
        1. Heredar de BasePredictor
        2. Implementar fit, predict y get_params
        3. El resto del pipeline (CV, MLflow) funciona sin cambios
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name:  str  = model_name
        self.model            = None
        self.feature_names    = None
        self.is_fitted: bool  = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BasePredictor":
        """
        Entrena el modelo.

        Args:
            train: DataFrame con features y target (sin NaN)

        Returns:
            self — permite encadenar model.fit(train).predict(test)
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones.

        Args:
            df: DataFrame con las mismas features que train

        Returns:
            Serie de predicciones indexada por fecha
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Retorna hiperparámetros del modelo como diccionario"""
        pass

    def _check_fitted(self) -> None:
        """Valida que el modelo esté entrenado antes de predecir"""
        if not self.is_fitted:
            raise ValueError(
                f"{self.model_name} no está entrenado. "
                f"Llama a fit() primero."
            )