"""Cross-validation temporal para series meteorológicas"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from src.utils.config import N_SPLITS, TEST_SIZE, MIN_TRAIN_SIZE, TARGET
from src.evaluation.metrics import evaluate_model


class TimeSeriesSplit:
    """
    Split temporal para series de tiempo.
    Preserva orden cronológico — sin leakage de futuro.

    Estrategias:
        'expanding': train crece en cada fold (más datos históricos)
        'sliding':   train se desplaza manteniendo tamaño fijo
    """

    def __init__(
        self,
        n_splits:       int = N_SPLITS,
        test_size:      int = TEST_SIZE,
        min_train_size: int = MIN_TRAIN_SIZE,
        strategy:       str = 'expanding',  # 'expanding' | 'sliding'
    ):
        if strategy not in ('expanding', 'sliding'):
            raise ValueError("strategy debe ser 'expanding' o 'sliding'")

        self.n_splits       = n_splits
        self.test_size      = test_size
        self.min_train_size = min_train_size
        self.strategy       = strategy

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Genera folds preservando orden temporal.

        Args:
            df: DataFrame completo con features y target (sin NaN)

        Returns:
            Lista de tuplas (train_df, test_df)
        """
        n = len(df)

        if n < self.min_train_size + self.test_size:
            raise ValueError(
                f"Dataset insuficiente: {n} filas. "
                f"Necesitas al menos {self.min_train_size + self.test_size}."
            )

        if self.strategy == 'expanding':
            folds = self._expanding_split(df, n)
        else:
            folds = self._sliding_split(df, n)

        print(f"\n📊 {len(folds)} folds generados ({self.strategy} window):")
        for i, (train, test) in enumerate(folds):
            print(f"  Fold {i+1}: "
                  f"train={len(train)}d "
                  f"({train.index.min().date()}→{train.index.max().date()})  "
                  f"test={len(test)}d  "
                  f"({test.index.min().date()}→{test.index.max().date()})")
        return folds

    def _expanding_split(self, df: pd.DataFrame, n: int):
        """
        Expanding window — train crece en cada fold.

        Fold 1: train=[0..min_train],         test=[min_train..min_train+test]
        Fold 2: train=[0..min_train+step],    test=[...]
        ...
        Fold N: train=[0..n-test],            test=[n-test..n]
        """
        last_train_end = n - self.test_size
        step = (last_train_end - self.min_train_size) // (self.n_splits - 1) \
               if self.n_splits > 1 else 0

        folds = []
        for i in range(self.n_splits):
            if i == self.n_splits - 1:
                # Último fold usa exactamente el final del dataset
                train_end  = last_train_end
            else:
                train_end  = self.min_train_size + i * step

            test_start = train_end
            test_end   = test_start + self.test_size

            if test_end > n:
                break

            train = df.iloc[:train_end].copy()
            test  = df.iloc[test_start:test_end].copy()
            folds.append((train, test))

        return folds

    def _sliding_split(self, df: pd.DataFrame, n: int):
        """
        Sliding window — train se desplaza manteniendo tamaño fijo.
        Todos los folds tienen exactamente min_train_size días de train.

        Fold 1: train=[0..min_train],              test=[min_train..min_train+test]
        Fold 2: train=[step..min_train+step],      test=[...]
        ...
        Fold N: train=[n-test-min_train..n-test],  test=[n-test..n]
        """
        last_test_end   = n
        last_test_start = n - self.test_size
        last_train_end  = last_test_start
        last_train_start = last_train_end - self.min_train_size

        if last_train_start < 0:
            raise ValueError(
                f"Dataset insuficiente para sliding window con "
                f"min_train_size={self.min_train_size} y "
                f"n_splits={self.n_splits}."
            )

        # Calcular step hacia atrás desde el último fold
        first_train_start = 0
        step = (last_train_start - first_train_start) // (self.n_splits - 1) \
               if self.n_splits > 1 else 0

        folds = []
        for i in range(self.n_splits):
            if i == self.n_splits - 1:
                train_start = last_train_start
            else:
                train_start = first_train_start + i * step

            train_end  = train_start + self.min_train_size
            test_start = train_end
            test_end   = test_start + self.test_size

            if test_end > n:
                break

            train = df.iloc[train_start:train_end].copy()
            test  = df.iloc[test_start:test_end].copy()
            folds.append((train, test))

        return folds

    def visualize(self, df: pd.DataFrame) -> None:
        """
        Visualiza los splits como diagrama de barras horizontales.
        Expanding: el train crece visualmente.
        Sliding: el train se desplaza manteniendo el mismo ancho.
        """
        folds = self.split(df)
        n     = len(df)

        fig, ax = plt.subplots(figsize=(14, 5))

        for i, (train, test) in enumerate(folds):
            train_start = df.index.get_loc(train.index[0])  \
                          if train.index[0] in df.index else 0
            train_len   = len(train)
            test_len    = len(test)

            # Zona no usada antes del train (solo en sliding)
            if train_start > 0:
                ax.barh(i, train_start, left=0,
                        color='#e0e0e0', alpha=0.4)

            # Train
            ax.barh(i, train_len, left=train_start,
                    color='#457b9d', alpha=0.7)

            # Test
            ax.barh(i, test_len, left=train_start + train_len,
                    color='#e63946', alpha=0.85)

            # Zona no usada después del test
            used = train_start + train_len + test_len
            if used < n:
                ax.barh(i, n - used, left=used,
                        color='#e0e0e0', alpha=0.4)

        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(len(folds))])
        ax.set_xlabel('Días desde inicio del dataset')
        ax.set_title(
            f'Time Series Cross-Validation — {self.strategy.capitalize()} Window',
            fontsize=13, fontweight='bold'
        )

        from matplotlib.patches import Patch
        legend = [
            Patch(color='#457b9d', alpha=0.7,  label='Train'),
            Patch(color='#e63946', alpha=0.85, label='Test'),
            Patch(color='#e0e0e0', alpha=0.5,  label='No usado'),
        ]
        ax.legend(handles=legend, loc='lower right')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()


def run_cv(
    model_class,
    df:             pd.DataFrame,
    n_splits:       int = N_SPLITS,
    test_size:      int = TEST_SIZE,
    min_train_size: int = MIN_TRAIN_SIZE,
    strategy:       str = 'expanding',
    **model_params,
) -> pd.DataFrame:
    """
    Ejecuta cross-validation temporal para un modelo.

    Args:
        model_class:    Clase del modelo (LightGBMPredictor)
        df:             DataFrame completo con features y target (sin NaN)
        n_splits:       Número de folds
        test_size:      Días en cada test set
        min_train_size: Mínimo de días para entrenar
        strategy:       'expanding' o 'sliding'
        **model_params: Hiperparámetros para inicializar el modelo

    Returns:
        DataFrame con métricas por fold + fila de promedios
    """
    cv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        strategy=strategy,
    )
    folds = cv.split(df)
    results = []

    for fold_idx, (train, test) in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"FOLD {fold_idx + 1} / {len(folds)}")
        print(f"{'='*55}")

        model = model_class(**model_params)
        model.fit(train)
        preds = model.predict(test)

        y_true  = test[TARGET]
        metrics = evaluate_model(y_true, preds, model_name=f"Fold {fold_idx + 1}")
        metrics['fold']       = fold_idx + 1
        metrics['train_size'] = len(train)
        metrics['test_size']  = len(test)
        metrics['train_start'] = str(train.index.min().date())
        metrics['train_end']  = str(train.index.max().date())
        metrics['test_start'] = str(test.index.min().date())
        metrics['test_end']   = str(test.index.max().date())

        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Fila de promedios
    numeric_cols = ['mae', 'rmse', 'r2', 'mape', 'accuracy', 'f1_score']
    numeric_cols = [c for c in numeric_cols if c in results_df.columns]
    avg_row = results_df[numeric_cols].mean().to_dict()
    avg_row['fold'] = 'promedio'
    results_df = pd.concat(
        [results_df, pd.DataFrame([avg_row])],
        ignore_index=True
    )
    results_df['fold'] = results_df['fold'].astype(str)

    _print_cv_summary(results_df, numeric_cols)
    return results_df


def _print_cv_summary(results_df: pd.DataFrame, numeric_cols: list) -> None:
    """Imprime tabla resumen del CV"""
    print(f"\n{'='*55}")
    print("RESUMEN CROSS-VALIDATION")
    print(f"{'='*55}")

    cols_display = ['fold'] + numeric_cols
    cols_display = [c for c in cols_display if c in results_df.columns]
    print(results_df[cols_display].to_string(index=False))

    avg = results_df[results_df['fold'] == 'promedio'].iloc[0]
    print(f"\n🏆 Promedios:")
    print(f"   MAE:      {avg.get('mae',      float('nan')):.3f} mm")
    print(f"   RMSE:     {avg.get('rmse',     float('nan')):.3f} mm")
    print(f"   R²:       {avg.get('r2',       float('nan')):.3f}")
    print(f"   Accuracy: {avg.get('accuracy', float('nan'))*100:.1f}%")
    print(f"   F1:       {avg.get('f1_score', float('nan')):.3f}")