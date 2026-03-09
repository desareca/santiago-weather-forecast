"""
Cross-validation y métricas para TwoStagePredictor.

Funciones exportadas:
    evaluate_two_stage(y_true, y_pred, prob, threshold, ...)  →  dict
    find_best_threshold(y_true, prob, thresholds)             →  (float, float)
    run_cv_two_stage(clf_params, reg_params, df, ...)         →  DataFrame
    run_cv_clf_grid(clf_grid, df, ...)                        →  DataFrame
    run_cv_reg_grid(reg_grid, best_clf_params, df, ...)       →  DataFrame
"""

import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from typing import Dict, List, Tuple

from src.utils.config import (
    CLF_RAIN_THRESHOLD, REG_RAIN_THRESHOLD, CLF_THRESHOLDS, F_BETA,
    N_SPLITS, TEST_SIZE, MIN_TRAIN_SIZE,
    TARGET,
)


# ──────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_two_stage(
    y_true:             pd.Series,
    y_pred:             pd.Series,
    prob_llueve:        np.ndarray,
    threshold:          float,
    clf_rain_threshold: float = CLF_RAIN_THRESHOLD,
    reg_rain_threshold: float = REG_RAIN_THRESHOLD,
    label:              str   = "",
    verbose:            bool  = True,
) -> Dict[str, float]:
    """
    Evalúa el two-stage model reportando métricas por etapa.

    Etapa 1 (clasificador): AUC-ROC, F1, Precision, Recall
    Etapa 2 (regresor):     MAE, RMSE, R² solo en días lluviosos reales
    Combinado:              MAE, RMSE, R² sobre todos los días
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    prob   = np.array(prob_llueve)

    y_bin_true = (y_true > clf_rain_threshold).astype(int)
    y_bin_pred = (prob   > threshold     ).astype(int)

    # ── Etapa 1: clasificador ─────────────────────────────────────
    try:
        auc = roc_auc_score(y_bin_true, prob)
    except Exception:
        auc = np.nan

    f1_clf  = f1_score(y_bin_true, y_bin_pred, zero_division=0)
    fbeta   = fbeta_score(y_bin_true, y_bin_pred, beta=F_BETA, zero_division=0)
    prec    = precision_score(y_bin_true, y_bin_pred, zero_division=0)
    rec     = recall_score(y_bin_true, y_bin_pred, zero_division=0)

    # ── Etapa 2: regresor (solo días lluviosos reales) ────────────
    mask_rain = y_true > reg_rain_threshold
    if mask_rain.sum() > 1:
        mae_rain  = mean_absolute_error(y_true[mask_rain], y_pred[mask_rain])
        rmse_rain = np.sqrt(mean_squared_error(y_true[mask_rain], y_pred[mask_rain]))
        r2_rain   = r2_score(y_true[mask_rain], y_pred[mask_rain])
    else:
        mae_rain = rmse_rain = r2_rain = np.nan

    # ── Recall en picos (>20mm) ───────────────────────────────────
    mask_picos = y_true > 20.0
    if mask_picos.sum() > 0:
        recall_picos = float((y_pred[mask_picos] > reg_rain_threshold).sum() / mask_picos.sum())
    else:
        recall_picos = np.nan

    # ── Combinado (todos los días) ────────────────────────────────
    mae_all  = mean_absolute_error(y_true, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_all   = r2_score(y_true, y_pred)

    metrics = {
        "auc":          float(auc),
        "f1_clf":       float(f1_clf),
        "fbeta":        float(fbeta),
        "precision":    float(prec),
        "recall":       float(rec),
        "mae_rain":     float(mae_rain)  if not np.isnan(mae_rain)  else np.nan,
        "rmse_rain":    float(rmse_rain) if not np.isnan(rmse_rain) else np.nan,
        "r2_rain":      float(r2_rain)   if not np.isnan(r2_rain)   else np.nan,
        "recall_picos": float(recall_picos) if not np.isnan(recall_picos) else np.nan,
        "mae":          float(mae_all),
        "rmse":         float(rmse_all),
        "r2":           float(r2_all),
        "threshold":    float(threshold),
        "n_lluvia":     int(mask_rain.sum()),
        "n_picos":      int(mask_picos.sum()),
    }

    if verbose:
        title = f" — {label}" if label else ""
        print(f"\n{'='*60}")
        print(f"TwoStage{title} — Evaluación")
        print(f"{'='*60}")
        print(f"\n🎯 ETAPA 1 — Clasificador (umbral={threshold}):")
        print(f"  AUC-ROC:   {auc:.3f}")
        print(f"  F1:        {f1_clf:.3f}")
        print(f"  F{F_BETA:.0f}:        {fbeta:.3f}  (recall x{F_BETA:.0f})")
        print(f"  Precision: {prec*100:.1f}%")
        print(f"  Recall:    {rec*100:.1f}%")
        print(f"\n📏 ETAPA 2 — Regresor (solo días lluviosos, n={mask_rain.sum()}):")
        print(f"  MAE:   {mae_rain:.2f} mm")
        print(f"  RMSE:  {rmse_rain:.2f} mm")
        print(f"  R²:    {r2_rain:.3f}")
        print(f"\n⛈️  PICOS (>20mm, n={mask_picos.sum()}):")
        if not np.isnan(recall_picos):
            print(f"  Recall: {recall_picos*100:.1f}%")
        else:
            print(f"  Recall: — (sin picos en este período)")
        print(f"\n📊 COMBINADO (todos los días):")
        print(f"  MAE:   {mae_all:.3f} mm")
        print(f"  RMSE:  {rmse_all:.3f} mm")
        print(f"  R²:    {r2_all:.3f}")

    return metrics


def find_best_threshold(
    y_true:         np.ndarray,
    prob_llueve:    np.ndarray,
    thresholds:     List[float] = CLF_THRESHOLDS,
    clf_rain_threshold: float       = CLF_RAIN_THRESHOLD,
) -> Tuple[float, float]:
    """
    Busca el umbral que maximiza Fbeta en el conjunto dado.
    beta=F_BETA (config): recall pesa más que precision.
    Usar SOLO sobre datos de validación interna, nunca sobre test.

    Returns
    -------
    (best_threshold, best_fbeta)
    """
    y_bin = (np.array(y_true) > clf_rain_threshold).astype(int)
    best_score, best_thr = -np.inf, thresholds[0]

    for thr in thresholds:
        y_pred_bin = (np.array(prob_llueve) > thr).astype(int)
        val = fbeta_score(y_bin, y_pred_bin, beta=F_BETA, zero_division=0)
        if val > best_score:
            best_score, best_thr = val, thr

    return best_thr, best_score


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ──────────────────────────────────────────────────────────────────────────────

def _make_folds(df, n_splits, test_size, min_train_size):
    from src.evaluation.cross_validation import TimeSeriesSplit
    cv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        strategy="sliding",
    )
    return cv.split(df)


def _print_cv_summary(results_df: pd.DataFrame, title: str = ""):
    print(f"\n{'='*55}")
    print(f"RESUMEN CROSS-VALIDATION — {title}")
    print(f"{'='*55}")
    cols = ["fold", "auc", "f1_clf", "fbeta", "recall", "precision"],
    cols += ["recall_picos", "mae_rain", "rmse_rain", "r2_rain",
             "rmse", "r2", "threshold"]
    cols = [c for c in cols if c in results_df.columns]
    print(results_df[cols].to_string(index=False))
    print(f"\n📊 Promedios:")
    num_cols = [c for c in cols if c != "fold"]
    print(results_df[num_cols].mean().round(3).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# CV COMPLETO (clf + reg juntos) — para verificar pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_two_stage(
    clf_params:     Dict,
    reg_params:     Dict,
    df:             pd.DataFrame,
    thresholds:     List[float] = CLF_THRESHOLDS,
    log_target:     bool        = True,
    clf_rain_threshold: float       = CLF_RAIN_THRESHOLD,
    reg_rain_threshold: float       = REG_RAIN_THRESHOLD,
    n_splits:           int         = N_SPLITS,
    test_size:          int         = TEST_SIZE,
    min_train_size:     int         = MIN_TRAIN_SIZE,
    val_ratio:          float       = 0.2,
    verbose:            bool        = True,
) -> pd.DataFrame:
    """
    CV temporal para TwoStagePredictor con búsqueda de umbral interna.

    En cada fold:
      1. Divide train_fold en train_interno + val_interno (val_ratio)
      2. Entrena clf en train_interno → busca umbral óptimo en val_interno
      3. Entrena modelo completo (clf + reg) en train_fold completo
      4. Evalúa en test_fold con el umbral encontrado
    """
    from src.models.two_stage_model import TwoStagePredictor
    from src.data.preprocessing import get_feature_names

    folds = _make_folds(df, n_splits, test_size, min_train_size)
    feat  = get_feature_names(df)
    rows  = []

    for fold_idx, (df_train_fold, df_test_fold) in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"FOLD {fold_idx+1} / {len(folds)}")
        print(f"  Train: {df_train_fold.index.min().date()} → "
              f"{df_train_fold.index.max().date()} ({len(df_train_fold)}d)")
        print(f"  Test:  {df_test_fold.index.min().date()} → "
              f"{df_test_fold.index.max().date()} ({len(df_test_fold)}d)")
        print(f"{'='*55}")

        split_idx  = int(len(df_train_fold) * (1 - val_ratio))
        df_tr_int  = df_train_fold.iloc[:split_idx]
        df_val_int = df_train_fold.iloc[split_idx:]

        clf_p = {"objective": "binary", "is_unbalance": True,
                 "verbose": -1, "random_state": 42}
        clf_p.update(clf_params)
        clf_search = lgb.LGBMClassifier(**clf_p)
        clf_search.fit(df_tr_int[feat],
                       (df_tr_int[TARGET] > clf_rain_threshold).astype(int))

        prob_val = clf_search.predict_proba(df_val_int[feat])[:, 1]
        best_thr, _ = find_best_threshold(df_val_int[TARGET], prob_val,
                                          thresholds, clf_rain_threshold)
        print(f"  🎯 Umbral óptimo (val interno): {best_thr}")

        model = TwoStagePredictor(
            clf_params=clf_params,
            reg_params=reg_params,
            threshold=best_thr,
            log_target=log_target,
            clf_rain_threshold=clf_rain_threshold,
            reg_rain_threshold=reg_rain_threshold,
        )
        model.fit(df_train_fold)

        y_true    = df_test_fold[TARGET]
        y_pred    = model.predict(df_test_fold)
        prob_test = model.predict_proba(df_test_fold)

        metrics = evaluate_two_stage(
            y_true=y_true, y_pred=y_pred, prob_llueve=prob_test,
            threshold=best_thr, clf_rain_threshold=clf_rain_threshold,
            reg_rain_threshold=reg_rain_threshold,
            label=f"Fold {fold_idx+1}", verbose=verbose,
        )
        metrics["fold"] = fold_idx + 1
        rows.append(metrics)

    results_df = pd.DataFrame(rows)
    _print_cv_summary(results_df, title="TwoStage")
    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# GRID SEARCH — ETAPA 1: CLASIFICADOR
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_clf_grid(
    clf_grid:       List[Dict],
    df:             pd.DataFrame,
    thresholds:     List[float] = CLF_THRESHOLDS,
    clf_rain_threshold: float       = CLF_RAIN_THRESHOLD,
    reg_rain_threshold: float       = REG_RAIN_THRESHOLD,
    n_splits:           int         = N_SPLITS,
    test_size:          int         = TEST_SIZE,
    min_train_size:     int         = MIN_TRAIN_SIZE,
    val_ratio:          float       = 0.2,
    log_mlflow:         bool        = True,
) -> pd.DataFrame:
    """
    Grid search sobre configuraciones de clasificador.

    Para cada config en clf_grid:
      - CV temporal: en cada fold busca umbral en val interno, evalúa en test
      - Métricas: AUC, F1, Precision, Recall (promedio ± std de folds)
      - Loggea en MLflow con tags.stage = "clf_grid"

    Parameters
    ----------
    clf_grid : lista de dicts con hiperparámetros. Cada dict debe tener 'name'.

    Returns
    -------
    DataFrame ordenado por avg_f1 descendente.
    Columna 'best_threshold': umbral más frecuente entre folds.
    """
    from src.data.preprocessing import get_feature_names

    folds = _make_folds(df, n_splits, test_size, min_train_size)
    feat  = get_feature_names(df)
    summary_rows = []

    print(f"\n{'='*65}")
    print(f"GRID SEARCH — CLASIFICADOR ({len(clf_grid)} configs × {len(folds)} folds)")
    print(f"{'='*65}")

    for cfg_idx, cfg in enumerate(clf_grid):
        name = cfg.get("name", f"clf_{cfg_idx}")
        params = {k: v for k, v in cfg.items() if k != "name"}

        clf_p = {"objective": "binary", "is_unbalance": True,
                 "verbose": -1, "random_state": 42}
        clf_p.update(params)

        print(f"\n[{cfg_idx+1}/{len(clf_grid)}] {name}")
        fold_rows = []

        for fold_idx, (df_train_fold, df_test_fold) in enumerate(folds):
            # Buscar umbral en val interno
            split_idx  = int(len(df_train_fold) * (1 - val_ratio))
            df_tr_int  = df_train_fold.iloc[:split_idx]
            df_val_int = df_train_fold.iloc[split_idx:]

            clf_search = lgb.LGBMClassifier(**clf_p)
            clf_search.fit(df_tr_int[feat],
                           (df_tr_int[TARGET] > clf_rain_threshold).astype(int))
            prob_val = clf_search.predict_proba(df_val_int[feat])[:, 1]
            best_thr, _ = find_best_threshold(df_val_int[TARGET], prob_val,
                                                              thresholds, clf_rain_threshold)

            # Reentrenar con fold completo
            clf_full = lgb.LGBMClassifier(**clf_p)
            clf_full.fit(df_train_fold[feat],
                           (df_train_fold[TARGET] > clf_rain_threshold).astype(int))

            # Evaluar en test
            prob_test  = clf_full.predict_proba(df_test_fold[feat])[:, 1]
            y_bin_true = (df_test_fold[TARGET].values > clf_rain_threshold).astype(int)
            y_bin_pred = (prob_test > best_thr).astype(int)

            try:
                auc = roc_auc_score(y_bin_true, prob_test)
            except Exception:
                auc = np.nan

            f1_v    = f1_score   (y_bin_true, y_bin_pred, zero_division=0)
            fbeta_v = fbeta_score(y_bin_true, y_bin_pred, beta=F_BETA, zero_division=0)
            prec_v  = precision_score(y_bin_true, y_bin_pred, zero_division=0)
            rec_v   = recall_score   (y_bin_true, y_bin_pred, zero_division=0)

            fold_rows.append({
                "fold":      fold_idx + 1,
                "threshold": best_thr,
                "auc":       auc,
                "f1":        f1_v,
                "fbeta":     fbeta_v,
                "precision": prec_v,
                "recall":    rec_v,
            })
            print(f"  Fold {fold_idx+1}: thr={best_thr:.2f}  "
                  f"AUC={auc:.3f}  F1={f1_v:.3f}  "
                  f"Prec={prec_v:.3f}  Rec={rec_v:.3f}")

        fold_df = pd.DataFrame(fold_rows)
        avg     = fold_df[["auc", "f1", "fbeta", "precision", "recall"]].mean()
        std     = fold_df[["auc", "f1", "fbeta", "precision", "recall"]].std()
        best_thr_global = fold_df["threshold"].mode()[0]

        row = {
            "name":           name,
            "best_threshold": best_thr_global,
            "avg_auc":        avg["auc"],
            "avg_f1":         avg["f1"],        "std_f1":        std["f1"],
            "avg_fbeta":      avg["fbeta"],     "std_fbeta":     std["fbeta"],
            "avg_precision":  avg["precision"], "std_precision": std["precision"],
            "avg_recall":     avg["recall"],    "std_recall":    std["recall"],
            **{f"params_{k}": v for k, v in params.items()},
        }
        summary_rows.append(row)

        print(f"  → AVG: AUC={avg['auc']:.3f}  F1={avg['f1']:.3f}  "
              f"F{F_BETA:.0f}={avg['fbeta']:.3f}  "
              f"Prec={avg['precision']:.3f}  Rec={avg['recall']:.3f}  "
              f"thr={best_thr_global}")

        if log_mlflow:
            with mlflow.start_run(run_name=f"clf_{name}"):
                mlflow.set_tag("stage", "clf_grid")
                mlflow.set_tag("config_name", name)
                for k, v in clf_p.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("best_threshold", best_thr_global)
                mlflow.log_param("val_ratio", val_ratio)
                mlflow.log_metric("avg_auc",       float(avg["auc"]))
                mlflow.log_metric("avg_f1",        float(avg["f1"]))
                mlflow.log_metric("avg_fbeta",     float(avg["fbeta"]))
                mlflow.log_metric("avg_precision", float(avg["precision"]))
                mlflow.log_metric("avg_recall",    float(avg["recall"]))
                mlflow.log_metric("std_f1",        float(std["f1"]))
                mlflow.log_metric("std_fbeta",     float(std["fbeta"]))

    results_df = pd.DataFrame(summary_rows).sort_values("avg_fbeta", ascending=False)

    print(f"\n{'='*65}")
    print("RESULTADOS GRID SEARCH — CLASIFICADOR (ordenado por avg_F2)")
    print(f"{'='*65}")
    cols = ["name", "best_threshold", "avg_auc", "avg_f1", "std_f1",
            "avg_precision", "avg_recall"]
    print(results_df[cols].to_string(index=False))
    print(f"\n🏆 Mejor clf: {results_df.iloc[0]['name']}  "
          f"(F1={results_df.iloc[0]['avg_f1']:.3f}, "
          f"thr={results_df.iloc[0]['best_threshold']})")

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# GRID SEARCH — ETAPA 2: REGRESOR
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_reg_grid(
    reg_grid:        List[Dict],
    best_clf_params: Dict,
    best_threshold:  float,
    df:              pd.DataFrame,
    log_target:      bool        = True,
    clf_rain_threshold: float       = CLF_RAIN_THRESHOLD,
    reg_rain_threshold: float       = REG_RAIN_THRESHOLD,
    n_splits:        int         = N_SPLITS,
    test_size:       int         = TEST_SIZE,
    min_train_size:  int         = MIN_TRAIN_SIZE,
    log_mlflow:      bool        = True,
) -> pd.DataFrame:
    """
    Grid search sobre configuraciones de regresor, fijando el mejor clf.

    El clf se re-entrena en cada fold (sin leakage).
    Se evalúa principalmente sobre días lluviosos reales.

    Parameters
    ----------
    reg_grid         : lista de dicts con hiperparámetros. Cada dict debe tener 'name'.
    best_clf_params  : parámetros del clasificador ganador del clf_grid.
    best_threshold   : umbral del clasificador ganador.
    log_target       : si aplicar log1p al target del regresor.

    Returns
    -------
    DataFrame ordenado por avg_rmse_rain ascendente.
    """
    from src.models.two_stage_model import TwoStagePredictor

    folds = _make_folds(df, n_splits, test_size, min_train_size)
    summary_rows = []

    clf_name = best_clf_params.get("name", "best_clf")

    print(f"\n{'='*65}")
    print(f"GRID SEARCH — REGRESOR ({len(reg_grid)} configs × {len(folds)} folds)")
    print(f"  Clf fijo: {clf_name}  |  Umbral fijo: {best_threshold}")
    print(f"{'='*65}")

    for cfg_idx, cfg in enumerate(reg_grid):
        name = cfg.get("name", f"reg_{cfg_idx}")
        reg_params = {k: v for k, v in cfg.items() if k != "name"}

        print(f"\n[{cfg_idx+1}/{len(reg_grid)}] {name}")
        fold_rows = []

        for fold_idx, (df_train_fold, df_test_fold) in enumerate(folds):
            model = TwoStagePredictor(
                clf_params=best_clf_params,
                reg_params=reg_params,
                threshold=best_threshold,
                log_target=log_target,
            clf_rain_threshold=clf_rain_threshold,
            reg_rain_threshold=reg_rain_threshold,
            )
            model.fit(df_train_fold)

            y_true    = df_test_fold[TARGET].values
            y_pred    = model.predict(df_test_fold)

            mask_rain  = y_true > reg_rain_threshold
            mask_picos = y_true > 20.0

            if mask_rain.sum() > 1:
                mae_rain  = mean_absolute_error(y_true[mask_rain], y_pred[mask_rain])
                rmse_rain = np.sqrt(mean_squared_error(y_true[mask_rain], y_pred[mask_rain]))
                r2_rain   = r2_score(y_true[mask_rain], y_pred[mask_rain])
            else:
                mae_rain = rmse_rain = r2_rain = np.nan

            recall_picos = (
        float((y_pred[mask_picos] > reg_rain_threshold).sum() / mask_picos.sum())
                if mask_picos.sum() > 0 else np.nan
            )

            rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
            r2_all   = r2_score(y_true, y_pred)

            fold_rows.append({
                "fold":         fold_idx + 1,
                "mae_rain":     mae_rain,
                "rmse_rain":    rmse_rain,
                "r2_rain":      r2_rain,
                "recall_picos": recall_picos,
                "rmse":         rmse_all,
                "r2":           r2_all,
            })
            rp_str = f"{recall_picos*100:.0f}%" if not np.isnan(recall_picos) else "—"
            print(f"  Fold {fold_idx+1}: "
                  f"RMSE_rain={rmse_rain:.2f}  R²_rain={r2_rain:.3f}  "
                  f"Recall_picos={rp_str}")

        fold_df = pd.DataFrame(fold_rows)
        avg     = fold_df[["mae_rain", "rmse_rain", "r2_rain",
                            "recall_picos", "rmse", "r2"]].mean()
        std     = fold_df[["rmse_rain", "r2_rain"]].std()

        row = {
            "name":             name,
            "log_target":       log_target,
            "avg_mae_rain":     avg["mae_rain"],
            "avg_rmse_rain":    avg["rmse_rain"],  "std_rmse_rain": std["rmse_rain"],
            "avg_r2_rain":      avg["r2_rain"],    "std_r2_rain":   std["r2_rain"],
            "avg_recall_picos": avg["recall_picos"],
            "avg_rmse":         avg["rmse"],
            "avg_r2":           avg["r2"],
            **{f"params_{k}": v for k, v in reg_params.items()},
        }
        summary_rows.append(row)

        rp_avg = avg["recall_picos"]
        rp_str = f"{rp_avg*100:.0f}%" if not np.isnan(rp_avg) else "—"
        print(f"  → AVG: RMSE_rain={avg['rmse_rain']:.2f}  "
              f"R²_rain={avg['r2_rain']:.3f}  Recall_picos={rp_str}")

        if log_mlflow:
            with mlflow.start_run(run_name=f"reg_{name}"):
                mlflow.set_tag("stage", "reg_grid")
                mlflow.set_tag("config_name", name)
                mlflow.set_tag("clf_fixed", clf_name)
                mlflow.log_param("threshold", best_threshold)
                mlflow.log_param("log_target", log_target)
                for k, v in reg_params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("avg_mae_rain",     float(avg["mae_rain"]))
                mlflow.log_metric("avg_rmse_rain",    float(avg["rmse_rain"]))
                mlflow.log_metric("avg_r2_rain",      float(avg["r2_rain"]))
                mlflow.log_metric("avg_recall_picos", float(rp_avg) if not np.isnan(rp_avg) else 0.0)
                mlflow.log_metric("avg_rmse",         float(avg["rmse"]))
                mlflow.log_metric("std_rmse_rain",    float(std["rmse_rain"]))

    results_df = pd.DataFrame(summary_rows).sort_values("avg_rmse_rain", ascending=True)

    print(f"\n{'='*65}")
    print("RESULTADOS GRID SEARCH — REGRESOR (ordenado por avg_RMSE_rain)")
    print(f"{'='*65}")
    cols = ["name", "avg_mae_rain", "avg_rmse_rain", "std_rmse_rain",
            "avg_r2_rain", "avg_recall_picos", "avg_rmse"]
    print(results_df[cols].to_string(index=False))
    print(f"\n🏆 Mejor reg: {results_df.iloc[0]['name']}  "
          f"(RMSE_rain={results_df.iloc[0]['avg_rmse_rain']:.2f}, "
          f"R²_rain={results_df.iloc[0]['avg_r2_rain']:.3f})")

    return results_df