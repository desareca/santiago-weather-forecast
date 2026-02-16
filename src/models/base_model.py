"""Clase base abstracta para modelos de predicción"""

from abc import ABC, abstractmethod
import pandas as pd
import mlflow
from typing import Dict, Any, List, Tuple
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
    
    def train_and_evaluate_cv(
        self,
        data: pd.Series,
        n_splits: int = 5,
        test_size: int = 30,
        log_mlflow: bool = True,
        **fit_kwargs
    ) -> pd.DataFrame:
        """
        Entrenamiento y evaluación con Cross-Validation temporal.
        
        Args:
            data: Serie temporal completa
            n_splits: Número de folds
            test_size: Días en cada test set
            log_mlflow: Si registrar en MLflow
            **fit_kwargs: Parámetros adicionales para fit()
            
        Returns:
            DataFrame con métricas por fold
        """
        from src.evaluation.cross_validation import TimeSeriesSplit
        
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        folds = cv.split(data)
        
        results = []
        
        # MLflow parent run
        parent_run_name = f"{self.model_name}_CV_{n_splits}folds"
        
        if log_mlflow:
            with mlflow.start_run(run_name=parent_run_name):
                mlflow.log_param("cv_strategy", "TimeSeriesSplit")
                mlflow.log_param("n_splits", n_splits)
                mlflow.log_param("test_size", test_size)
                
                for fold_idx, (train, test) in enumerate(folds):
                    metrics = self._run_single_fold(fold_idx, train, test, log_mlflow=True, **fit_kwargs)
                    results.append(metrics)
                
                # Log métricas promedio
                results_df = pd.DataFrame(results)
                avg_metrics = results_df[['mae', 'rmse', 'r2', 'accuracy', 'f1_score']].mean()
                
                for metric_name, metric_value in avg_metrics.items():
                    mlflow.log_metric(f"cv_avg_{metric_name}", metric_value)
        else:
            for fold_idx, (train, test) in enumerate(folds):
                metrics = self._run_single_fold(fold_idx, train, test, log_mlflow=False, **fit_kwargs)
                results.append(metrics)
            
            results_df = pd.DataFrame(results)
        
        # Mostrar resumen
        self._print_cv_summary(results_df)
        
        return results_df
    
    def _run_single_fold(
        self,
        fold_idx: int,
        train: pd.Series,
        test: pd.Series,
        log_mlflow: bool = True,
        **fit_kwargs
    ) -> Dict[str, float]:
        """Ejecuta un único fold de CV"""
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'='*60}")
        
        if log_mlflow:
            with mlflow.start_run(run_name=f"fold_{fold_idx+1}", nested=True):
                return self._train_eval_pipeline(train, test, fold_idx=fold_idx, **fit_kwargs)
        else:
            return self._train_eval_pipeline(train, test, fold_idx=fold_idx, **fit_kwargs)
    
    def _print_cv_summary(self, results_df: pd.DataFrame):
        """Imprime resumen de CV"""
        print(f"\n{'='*60}")
        print("RESUMEN CROSS-VALIDATION")
        print(f"{'='*60}")
        print(results_df[['fold', 'mae', 'rmse', 'r2', 'accuracy', 'f1_score']].to_string(index=False))
        print(f"\n📊 Promedios:")
        print(results_df[['mae', 'rmse', 'r2', 'accuracy', 'f1_score']].mean().to_string())
    
    def _train_eval_pipeline(self, train: pd.Series, test: pd.Series, fold_idx: int = None, **fit_kwargs):
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
        
        if fold_idx is not None:
            metrics['fold'] = fold_idx + 1
        
        # Log MLflow
        if mlflow.active_run():
            # Parámetros
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("train_size", len(train))
            mlflow.log_param("test_size", len(test))
            if fold_idx is not None:
                mlflow.log_param("fold", fold_idx + 1)
            
            # Métricas
            mlflow.log_metrics(metrics)
            
            # Plot (solo si no es fold para evitar muchos gráficos)
            if fold_idx is None:
                fig = plot_predictions(test, predictions, self.model_name)
                mlflow.log_figure(fig, "predictions.png")
            
            # Modelo (solo si no es fold)
            if fold_idx is None:
                mlflow.sklearn.log_model(self.model, "model")
        
        return metrics