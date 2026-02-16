"""Cross-validation para series temporales"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from src.evaluation.metrics import evaluate_model


class TimeSeriesSplit:
    """
    Split temporal para series temporales.
    Preserva orden cronológico, sin leakage de futuro.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 30):
        """
        Args:
            n_splits: Número de folds
            test_size: Días en cada test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, data: pd.Series) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Genera folds preservando orden temporal.
        
        Returns:
            Lista de tuplas (train, test)
        """
        total_size = len(data)
        min_train_size = 100  # Mínimo para entrenar
        
        # Calcular tamaño de cada incremento
        max_train_size = total_size - (self.n_splits * self.test_size)
        step_size = (max_train_size - min_train_size) // (self.n_splits - 1)
        
        folds = []
        
        for i in range(self.n_splits):
            # Train crece progresivamente
            train_end = min_train_size + (i * step_size)
            test_start = train_end
            test_end = test_start + self.test_size
            
            if test_end > total_size:
                break
                
            train = data[:train_end]
            test = data[test_start:test_end]
            
            folds.append((train, test))
        
        print(f"\n📊 {len(folds)} folds creados:")
        for i, (train, test) in enumerate(folds):
            print(f"  Fold {i+1}: Train={len(train)} días, Test={len(test)} días")
        
        return folds
    
    def visualize_splits(self, data: pd.Series):
        """Visualiza los splits"""
        folds = self.split(data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, (train, test) in enumerate(folds):
            # Train en azul
            ax.barh(i, len(train), left=0, color='steelblue', alpha=0.6)
            # Test en naranja
            ax.barh(i, len(test), left=len(train), color='coral', alpha=0.8)
        
        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(len(folds))])
        ax.set_xlabel('Días desde inicio')
        ax.set_title('Time Series Cross-Validation Splits', fontsize=14, fontweight='bold')
        ax.legend(['Train', 'Test'])
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()


def evaluate_with_cv(
    model_class,
    data: pd.Series,
    n_splits: int = 5,
    register_best: bool = False,  
    model_name: str = None, 
    **model_params
) -> pd.DataFrame:
    """
    Evalúa un modelo con cross-validation temporal.
    
    Args:
        model_class: Clase del modelo (ARIMAPredictor, ProphetPredictor, etc.)
        data: Serie temporal completa
        n_splits: Número de folds
        **model_params: Parámetros para inicializar el modelo
    
    Returns:
        DataFrame con métricas por fold
    """
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=30)
    folds = cv.split(data)
    
    results = []
    best_f1 = -1
    best_model = None

    for fold_idx, (train, test) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")
        
        # Crear instancia del modelo
        model = model_class(**model_params)
        
        # Entrenar
        model.fit(train)
        
        # Predecir
        predictions = model.predict(steps=len(test))
        predictions.index = test.index
        
        # Evaluar
        metrics = model.evaluate(test, predictions)
        metrics['fold'] = fold_idx + 1
        
        results.append(metrics)

        # Guardar mejor modelo
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model.model
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print("RESUMEN CROSS-VALIDATION")
    print(f"{'='*60}")
    print(results_df[['fold', 'mae', 'rmse', 'r2', 'accuracy', 'f1_score']].to_string(index=False))
    print(f"\n📊 Promedios:")
    print(results_df[['mae', 'rmse', 'r2', 'accuracy', 'f1_score']].mean().to_string())


        # Registrar mejor modelo del CV
    if register_best and model_name and best_model:
        with mlflow.start_run(run_name=f"{model_name}_CV_best"):
            avg_metrics = results_df[['mae', 'rmse', 'r2', 'accuracy', 'f1_score']].mean()
            
            # Log métricas promedio
            mlflow.log_metrics(avg_metrics.to_dict())
            mlflow.log_param("cv_folds", n_splits)
            mlflow.log_param("model_type", model_name)
            
            # Log mejor modelo
            mlflow.sklearn.log_model(best_model, "model")
            
            # Registrar en Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "santiago_weather_predictor")
            
            print(f"\n✅ Mejor modelo del CV registrado (F1={best_f1:.3f})")
    
    return results_df