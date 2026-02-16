"""Métricas de evaluación para modelos de predicción de precipitación"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Dict con métricas
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (evitando división por 0)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series, threshold: float = 1.0) -> Dict[str, float]:
    """
    Evalúa como clasificación binaria (¿lloverá o no?).
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        threshold: Umbral para considerar lluvia
        
    Returns:
        Dict con métricas de clasificación
    """
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Métricas básicas
    accuracy = (y_true_binary == y_pred_binary).mean()
    
    # True Positives, False Positives, etc.
    tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluación completa (regresión + clasificación).
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        model_name: Nombre del modelo para display
        
    Returns:
        Dict con todas las métricas
    """
    # Métricas de regresión
    reg_metrics = evaluate_regression(y_true, y_pred)
    
    # Métricas de clasificación
    clf_metrics = evaluate_classification(y_true, y_pred)
    
    # Combinar
    all_metrics = {**reg_metrics, **clf_metrics}
    
    # Display
    print(f"\n{'='*60}")
    print(f"{model_name} - Resultados de Evaluación")
    print(f"{'='*60}")
    print("\n📊 REGRESIÓN:")
    print(f"  MAE:  {reg_metrics['mae']:.3f} mm")
    print(f"  RMSE: {reg_metrics['rmse']:.3f} mm")
    print(f"  R²:   {reg_metrics['r2']:.3f}")
    print(f"  MAPE: {reg_metrics['mape']:.1f}%")
    
    print("\n🎯 CLASIFICACIÓN (lluvia sí/no):")
    print(f"  Accuracy:  {clf_metrics['accuracy']*100:.1f}%")
    print(f"  Precision: {clf_metrics['precision']*100:.1f}%")
    print(f"  Recall:    {clf_metrics['recall']*100:.1f}%")
    print(f"  F1-Score:  {clf_metrics['f1_score']:.3f}")
    
    return all_metrics


def plot_predictions(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualiza predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
        figsize: Tamaño de la figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(y_true.index, y_true.values, label='Real', alpha=0.7, linewidth=2, color='blue')
    ax.plot(y_pred.index, y_pred.values, label='Predicción', alpha=0.7, linewidth=2, color='orange')
    
    ax.set_title(f'{model_name} - Predicciones vs Real', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Precipitación (mm)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig