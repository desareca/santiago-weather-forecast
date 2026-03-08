"""Preprocessing y feature engineering para predicción de precipitación next-day"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from src.utils.config import (
    RENAME_MAP, TARGET_COL, TARGET, DATE_COL,
    LAG_VARS, ROLLING_VARS, ROLLING_WINDOWS,
    TRAIN_SPLIT, RANDOM_SEED
)


# ──────────────────────────────────────────────
# 1. Preparación base
# ──────────────────────────────────────────────

def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas, ordena por fecha y setea índice.

    Args:
        df_raw: DataFrame crudo desde Delta (nombres API)

    Returns:
        DataFrame limpio con nombres cortos, indexado por fecha
    """
    rename = {k: v for k, v in RENAME_MAP.items() if k in df_raw.columns}
    df = df_raw.rename(columns=rename).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    return df


# ──────────────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega lag=1 de todas las variables en LAG_VARS.
    Convención: var_lag1 = valor del día anterior.
    Sin data leakage: solo usamos t-1 para predecir t+1.
    """
    df = df.copy()
    for var in LAG_VARS:
        if var in df.columns:
            df[f"{var}_lag1"] = df[var].shift(1)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rolling mean, std y max para variables en ROLLING_VARS.
    Shift(1) antes del rolling para evitar data leakage.
    """
    df = df.copy()
    for var in ROLLING_VARS:
        if var not in df.columns:
            continue
        base = df[var].shift(1)
        for w in ROLLING_WINDOWS:
            df[f"{var}_rmean_{w}"] = base.rolling(w).mean()
            df[f"{var}_rstd_{w}"]  = base.rolling(w).std()
            df[f"{var}_rmax_{w}"]  = base.rolling(w).max()
    return df


def add_streak_features(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Agrega racha de días secos consecutivos hasta el día anterior.

    Args:
        threshold: mm mínimos para considerar día lluvioso
    """
    df = df.copy()
    # Serie binaria: 1=seco, 0=lluvioso (shifteada para no usar info del día actual)
    seco = (df[TARGET_COL].shift(1) <= threshold).astype(int)

    rachas = np.zeros(len(df), dtype=int)
    contador = 0
    for i, val in enumerate(seco):
        if val == 1:
            contador += 1
        else:
            contador = 0
        rachas[i] = contador

    df["racha_seca"] = rachas
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features cíclicas de estacionalidad.
    No requieren shift porque son deterministas (solo dependen de la fecha).
    """
    df = df.copy()
    df["mes_sin"]      = np.sin(2 * np.pi * df.index.month / 12)
    df["mes_cos"]      = np.cos(2 * np.pi * df.index.month / 12)
    df["dia_año_sin"]  = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["dia_año_cos"]  = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df["estacion_verano"]    = df.index.month.isin([12, 1, 2]).astype(int)
    df["estacion_otoño"]     = df.index.month.isin([3, 4, 5]).astype(int)
    df["estacion_invierno"]  = df.index.month.isin([6, 7, 8]).astype(int)
    df["estacion_primavera"] = df.index.month.isin([9, 10, 11]).astype(int)
    return df


def add_weather_code_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding de weather_code_lag1.
    Agrupa códigos WMO en categorías climáticas para reducir dimensionalidad.

    Grupos:
        0-1: despejado
        2-3: nublado
        45-48: niebla
        51-67: llovizna/lluvia leve
        71-77: nieve
        80-99: tormenta/lluvia intensa
    """
    df = df.copy()

    if "weather_code_lag1" not in df.columns:
        return df

    def agrupar_wmo(code):
        if pd.isna(code):
            return "desconocido"
        code = int(code)
        if code <= 1:
            return "despejado"
        elif code <= 3:
            return "nublado"
        elif code <= 48:
            return "niebla"
        elif code <= 67:
            return "lluvia_leve"
        elif code <= 77:
            return "nieve"
        else:
            return "tormenta"

    grupos = df["weather_code_lag1"].apply(agrupar_wmo)
    dummies = pd.get_dummies(grupos, prefix="wmo").astype(int)

    # Asegurar que todas las categorías existan aunque no aparezcan en el split
    categorias_esperadas = [
        "wmo_despejado", "wmo_nublado", "wmo_niebla",
        "wmo_lluvia_leve", "wmo_nieve", "wmo_tormenta", "wmo_desconocido"
    ]
    for cat in categorias_esperadas:
        if cat not in dummies.columns:
            dummies[cat] = 0

    df = pd.concat([df, dummies[categorias_esperadas]], axis=1)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columna target: precipitación del día siguiente.
    precip_t1(t) = precip(t+1)
    El último día queda con NaN y se elimina en el split.
    """
    df = df.copy()
    df[TARGET] = df[TARGET_COL].shift(-1)
    return df


# ──────────────────────────────────────────────
# 3. Pipeline completo
# ──────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.

    Args:
        df_raw: DataFrame crudo desde Delta

    Returns:
        DataFrame con todas las features y target.
        NaN no se eliminan aquí — responsabilidad del modelo/CV.
    """
    df = prepare_dataframe(df_raw)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_streak_features(df)
    df = add_seasonal_features(df)
    df = add_weather_code_dummies(df)
    df = add_target(df)

    print(f"✅ Features construidas: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"   Features X: {len(get_feature_names(df))}")
    print(f"   Target:     {TARGET}")
    print(f"   NaN en target: {df[TARGET].isna().sum()} (último día, esperado)")
    return df


# ──────────────────────────────────────────────
# 4. Helpers para modelo y CV
# ──────────────────────────────────────────────

def get_feature_names(df: pd.DataFrame) -> List[str]:
    """
    Retorna lista de columnas X (excluye target y variables crudas originales).
    Las variables crudas no se usan como features directas — solo sus lags.
    """
    # Columnas que NO son features
    raw_cols = list(RENAME_MAP.values())  # variables crudas originales
    excluir = raw_cols + [TARGET]

    return [c for c in df.columns if c not in excluir]


def train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal preservando orden cronológico.
    Elimina filas con NaN antes de splitear.

    Returns:
        Tupla (train, test) — DataFrames completos con X y target
    """
    df_clean = df.dropna().copy()
    split_idx = int(len(df_clean) * TRAIN_SPLIT)

    train = df_clean.iloc[:split_idx]
    test  = df_clean.iloc[split_idx:]

    print(f"📊 Train: {len(train)} días  ({train.index.min().date()} → {train.index.max().date()})")
    print(f"📊 Test:  {len(test)} días   ({test.index.min().date()} → {test.index.max().date()})")
    return train, test


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extrae X e y de un DataFrame ya procesado.

    Args:
        df: DataFrame con features y target (sin NaN)

    Returns:
        Tupla (X, y)
    """
    feature_cols = get_feature_names(df)
    X = df[feature_cols]
    y = df[TARGET]
    return X, y