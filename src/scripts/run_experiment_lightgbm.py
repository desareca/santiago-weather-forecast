from src.data.ingestion import load_from_delta_table
from src.data.preprocessing import prepare_time_series
from src.models.lightgbm_model import LightGBMPredictor
from src.evaluation.cross_validation import TimeSeriesSplit
from src.utils.config import *
import mlflow
import pandas as pd
import json
import importlib
import warnings

def run_lightgbm_batch(grid_config, output_csv, spark):
    # Cargar el grid
    with open(grid_config, "r") as f:
        lightgbm_grid = json.load(f)

    # Cargar y preparar datos
    df = load_from_delta_table("weather_raw", spark)
    serie = prepare_time_series(df, target_col="precipitacion")
    serie = serie[:int(len(serie)*TRAIN_SPLIT)]

    print(f"\n📊 Datos preparados:")
    print(f"  Serie completa: {len(serie)} días")
    print(f"  Fecha inicio: {serie.index.min().date()}")
    print(f"  Fecha fin: {serie.index.max().date()}")


    print("\n" + "="*70)
    print("CROSS-VALIDATION: LightGBM - Grid Search")
    print("="*70)

    # Entrenar todos
    results_lgbm_grid = []
    results_cv_all = pd.DataFrame()

    for i, params in enumerate(lightgbm_grid):
        print(f"\n[{i+1}/{len(lightgbm_grid)}] Probando LightGBM - {params['name']}...")

        # Separar params para el modelo y para features
        lags = params.get('lags', [1, 7, 30])
        rolling_windows = params.get('rolling_windows', [7, 30])
        params_ = {k: v for k, v in params.items() if k not in ['name', 'lags', 'rolling_windows']}

        lgbm = LightGBMPredictor(lags=lags, rolling_windows=rolling_windows, **params_)
        
        try:
            results_cv = lgbm.train_and_evaluate_cv(
                data=serie,
                n_splits=N_SPLITS, 
                test_size=TEST_SIZE, 
                min_train_size=MIN_TRAIN_SIZE,
                log_mlflow=True,
                run_description=f"{params['name']}"
            )

            results_cv['model'] = params['name']
            results_cv_all = pd.concat([results_cv_all, results_cv], ignore_index=True)

            results_cv.drop(['model'], axis=1, inplace=True)
            
            avg_metrics = results_cv[['mae', 'rmse', 'r2', 'f1_score']].mean()
            
            results_lgbm_grid.append({
                'model': params['name'],
                'n_estimators': params.get('n_estimators', None),
                'learning_rate': params.get('learning_rate', None),
                'max_depth': params.get('max_depth', None),
                'n_lags': len(lags),
                'mae': avg_metrics['mae'],
                'rmse': avg_metrics['rmse'],
                'r2': avg_metrics['r2'],
                'f1_score': avg_metrics['f1_score']
            })
            
        except Exception as e:
            print(f"  ⚠️  Error: {str(e)}")
            continue

    # Resumen
    df_lgbm_grid = pd.DataFrame(results_lgbm_grid)
    df_lgbm_grid = df_lgbm_grid.sort_values('f1_score', ascending=False)

    print("\n" + "="*70)
    print("RESULTADOS GRID SEARCH LIGHTGBM")
    print("="*70)
    print(df_lgbm_grid[['model', 'mae', 'rmse', 'r2', 'f1_score']].to_string(index=False))
    print(f"\n🏆 Mejor LightGBM: {df_lgbm_grid.iloc[0]['model']} (F1={df_lgbm_grid.iloc[0]['f1_score']:.3f})")

    results_cv_all.to_csv(output_csv, index=False)
    print(f"\n✅ Resultados guardados en {output_csv}. Revisa MLflow y CSV.")
