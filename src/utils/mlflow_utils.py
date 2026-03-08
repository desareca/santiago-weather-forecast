"""Utilidades centralizadas de MLflow para tracking de experimentos"""

import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Dict, Any, Optional
from src.utils.config import EXPERIMENT_NAME, MODEL_NAME, TARGET


def setup_experiment() -> str:
    """
    Configura el experimento de MLflow.

    Returns:
        experiment_id
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"✅ MLflow experiment: {EXPERIMENT_NAME}")
    print(f"   ID: {experiment.experiment_id}")
    return experiment.experiment_id


def log_cv_run(
    model,
    results_df:     pd.DataFrame,
    model_params:   Dict[str, Any],
    run_name:       str,
    description:    Optional[str] = None,
    register_model: bool = False,
) -> str:
    """
    Loggea un experimento de CV completo en MLflow.

    Estructura:
        Parent run  → métricas promedio, parámetros, modelo final
            Child run fold_1 → métricas fold 1
            Child run fold_2 → métricas fold 2
            ...

    Args:
        model:          Instancia de LightGBMPredictor ya entrenada (modelo final)
        results_df:     DataFrame retornado por run_cv (incluye fila promedio)
        model_params:   Hiperparámetros del modelo
        run_name:       Nombre del run en MLflow
        description:    Descripción opcional del experimento
        register_model: Si registrar el modelo en MLflow Model Registry

    Returns:
        run_id del parent run
    """
    folds_df = results_df[results_df["fold"] != "promedio"].copy()
    avg_row  = results_df[results_df["fold"] == "promedio"].iloc[0]

    numeric_cols = ["mae", "rmse", "r2", "mape", "accuracy", "f1_score"]
    numeric_cols = [c for c in numeric_cols if c in results_df.columns]

    with mlflow.start_run(run_name=run_name) as parent_run:

        # Tags
        mlflow.set_tag("model_family",    "LightGBM")
        mlflow.set_tag("evaluation_type", "cross_validation")
        mlflow.set_tag("run_type",        "parent")
        mlflow.set_tag("run_stage",       "experimentation")
        mlflow.set_tag("target",          TARGET)
        if description:
            mlflow.set_tag("description", description)

        # Parámetros
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("n_splits",  len(folds_df))
        mlflow.log_param("n_features", len(model.feature_names) if model.feature_names else None)
        if "test_size" in folds_df.columns:
            mlflow.log_param("test_size", int(folds_df["test_size"].iloc[0]))

        # Métricas promedio
        for col in numeric_cols:
            if col in avg_row.index and pd.notna(avg_row[col]):
                mlflow.log_metric(f"cv_avg_{col}", float(avg_row[col]))

        # Child runs por fold
        for _, fold_row in folds_df.iterrows():
            fold_num = fold_row["fold"]
            with mlflow.start_run(run_name=f"fold_{fold_num}", nested=True):
                mlflow.set_tag("run_type",    "fold")
                mlflow.set_tag("fold_number", fold_num)

                for tag in ["train_end", "test_start", "test_end"]:
                    if tag in fold_row.index:
                        mlflow.set_tag(tag, fold_row[tag])

                for col in numeric_cols:
                    if col in fold_row.index and pd.notna(fold_row[col]):
                        mlflow.log_metric(col, float(fold_row[col]))

                for param in ["train_size", "test_size"]:
                    if param in fold_row.index:
                        mlflow.log_param(param, int(fold_row[param]))

        # Modelo final
        if model.is_fitted:
            mlflow.sklearn.log_model(model.model, "model")

            importance_df = model.get_feature_importance(top_n=30)
            importance_path = "/tmp/feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, "feature_importance")

        # Registro en Model Registry
        run_id = parent_run.info.run_id
        if register_model and model.is_fitted:
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"✅ Modelo registrado en Registry: {MODEL_NAME}")

        print(f"\n✅ Run loggeado: {run_name}")
        print(f"   Run ID: {run_id}")

    return run_id


def log_test_evaluation(
    run_id:       str,
    test_metrics: Dict[str, float],
    description:  Optional[str] = None,
) -> None:
    """
    Agrega métricas de test set a un run existente.

    Args:
        run_id:       Run ID del parent run de CV
        test_metrics: Dict con métricas del test set
        description:  Descripción opcional
    """
    with mlflow.start_run(run_id=run_id):
        for metric_name, metric_value in test_metrics.items():
            if pd.notna(metric_value):
                mlflow.log_metric(f"test_{metric_name}", float(metric_value))

        mlflow.set_tag("has_test_evaluation", "true")
        if description:
            mlflow.set_tag("test_description", description)

    print(f"✅ Métricas de test loggeadas en run {run_id[:8]}...")


def get_best_run(
    metric:       str  = "cv_avg_rmse",
    ascending:    bool = True,
    model_family: str  = "LightGBM",
) -> Optional[pd.Series]:
    """
    Busca el mejor run en el experimento según una métrica.

    Args:
        metric:       Métrica a optimizar
        ascending:    True=minimizar, False=maximizar
        model_family: Filtrar por familia de modelo

    Returns:
        Serie con info del mejor run, o None si no hay runs
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("⚠️  Experimento no encontrado.")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.run_type = 'parent' "
            f"and tags.evaluation_type = 'cross_validation' "
            f"and tags.model_family = '{model_family}'"
        ),
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
    )

    if len(runs) == 0:
        print(f"⚠️  No se encontraron runs con métrica '{metric}'.")
        return None

    best = runs.iloc[0]
    print(f"\n🏆 Mejor run encontrado:")
    print(f"   Run ID:      {best['run_id'][:12]}...")
    print(f"   {metric}:   {best.get(f'metrics.{metric}', 'N/A'):.4f}")
    print(f"   Descripción: {best.get('tags.description', 'N/A')}")
    return best


def load_model_from_run(run_id: str) -> Any:
    """
    Carga un modelo sklearn desde un run de MLflow.

    Args:
        run_id: Run ID donde está el modelo guardado

    Returns:
        Modelo sklearn cargado
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modelo cargado desde run {run_id[:8]}...")
    return model