"""Transformaciones y feature engineering para datos meteorológicos"""

import pandas as pd
import numpy as np
from typing import Tuple


def prepare_time_series(df: pd.DataFrame, target_col: str = "precipitacion") -> pd.Series:
    """
    Prepara serie temporal ordenada por fecha.
    
    Args:
        df: DataFrame con columna 'fecha'
        target_col: Columna objetivo
        
    Returns:
        Serie temporal indexada por fecha
    """
    df = df.sort_values('fecha').copy()
    df.set_index('fecha', inplace=True)
    serie = df[target_col]
    return serie


def train_test_split_temporal(
    serie: pd.Series, 
    train_ratio: float = 0.8
) -> Tuple[pd.Series, pd.Series]:
    """
    Split temporal (preserva orden cronológico).
    
    Args:
        serie: Serie temporal
        train_ratio: Proporción de datos para entrenamiento
        
    Returns:
        Tupla (train, test)
    """
    split_idx = int(len(serie) * train_ratio)
    train = serie[:split_idx]
    test = serie[split_idx:]
    
    print(f"📊 Train: {len(train)} días ({train.index.min()} a {train.index.max()})")
    print(f"📊 Test: {len(test)} días ({test.index.min()} a {test.index.max()})")
    
    return train, test


def create_lag_features(df: pd.DataFrame, target_col: str = "precipitacion", lags: list = [1, 7, 30]) -> pd.DataFrame:
    """
    Crea features de lag y temporales para LightGBM incluyendo codificación cíclica.
    
    Args:
        df: DataFrame con datos
        target_col: Columna objetivo
        lags: Lista de lags a crear
        
    Returns:
        DataFrame con features de lag y temporales
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling windows
    df[f'{target_col}_rolling_mean_7'] = df[target_col].shift(1).rolling(7).mean()
    df[f'{target_col}_rolling_mean_30'] = df[target_col].shift(1).rolling(30).mean()
    
    # Features temporales
    #df['mes'] = df.index.month
    #df['dia_año'] = df.index.dayofyear
    df['estacion'] = df['mes'].apply(lambda x: 
        'verano' if x in [12, 1, 2] else
        'otoño' if x in [3, 4, 5] else
        'invierno' if x in [6, 7, 8] else 'primavera'
    )
    
    # Codificación cíclica
    df['mes_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['dia_año_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['dia_año_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    # Eliminar filas con NaN (por los lags y windows)
    df = df.dropna()
    
    print(f"✅ Features creadas. Shape final: {df.shape}")
    return df
