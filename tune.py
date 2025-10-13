"""
Hyperparameter Tuning

Systematic hyperparameter optimization using Optuna.

Components (in dependency order):
1. Return prediction (no dependencies)
2. Volatility prediction (depends on returns)
3. Meta-labeling (depends on returns)

After optimizing each component, the script retrains it with best parameters.

Usage:
    python tune.py --stage all              # Tune everything
    python tune.py --stage returns          # Tune return models only
    python tune.py --stage volatility       # Tune volatility models
    python tune.py --stage meta             # Tune meta-labeling

    # Advanced options:
    python tune.py --stage returns --n-trials 100 --folds 1,2
    python tune.py --stage all --timeout 3600  # 1 hour per stage
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import pickle
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.preprocessing import MarketDataPreprocessor
from src.features import FeatureEngineer, select_features_by_importance
from src.returns import ReturnPredictor
from src.volatility import VolatilityPredictor
from src.meta_labeling import MetaLabelPipeline

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)


# ============================================================================
# Hyperparameter Search Spaces
# ============================================================================


def get_return_search_space(trial):
    """Define search space for return prediction model."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "verbose": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
    }


def get_volatility_search_space(trial):
    """Define search space for volatility prediction model."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "verbose": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
    }


def get_meta_search_space(trial):
    """Define search space for meta-labeling classifier."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "verbose": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
        "is_unbalance": True,
    }


# ============================================================================
# Tuning Functions
# ============================================================================


def tune_return_models(n_trials: int = 50, timeout: int = None, folds: list = None):
    """
    Tune return prediction model hyperparameters.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds (per fold)
    folds : list, optional
        List of fold numbers to tune (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TUNING RETURN MODELS")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))
    results_dir = MODELS_DIR / "tuning_results"
    results_dir.mkdir(exist_ok=True)

    all_fold_results = []

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Load processed data
        print("\n1. Loading data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        print(f"   Train: {train_df.shape}")
        print(f"   Val:   {val_df.shape}")

        # Engineer features
        print("\n2. Engineering features...")
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        feature_cols = [col for col in train_df.columns if col not in special_cols]

        engineer = FeatureEngineer(continuous_features=feature_cols)
        train_features = engineer.fit_transform(train_df)
        val_features = engineer.transform(val_df)

        print(f"   Created {train_features.shape[1]} features")

        # Prepare data
        exclude_cols = ["date_id", "forward_returns", "risk_free_rate", TARGET_COL]
        X_cols = [col for col in train_features.columns if col not in exclude_cols]

        # Remove high-NaN features
        nan_ratio = train_features[X_cols].isna().mean()
        valid_features = nan_ratio[nan_ratio < 0.5].index.tolist()

        X_train = train_features[valid_features].fillna(0)
        y_train = train_features[TARGET_COL]
        X_val = val_features[valid_features].fillna(0)
        y_val = val_features[TARGET_COL]

        print(f"   Using {len(valid_features)} features")

        # Define objective function
        def objective(trial):
            params = get_return_search_space(trial)

            # Train model with suggested parameters
            model = ReturnPredictor(
                params=params,
                num_boost_round=LGBM_RETURN_ROUNDS,
                early_stopping_rounds=LGBM_RETURN_EARLY_STOP,
            )
            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            mse = np.mean((y_val - val_pred) ** 2)
            rmse = np.sqrt(mse)

            # Return negative RMSE (optuna minimizes)
            return rmse

        # Create study
        print(f"\n3. Running {n_trials} optimization trials...")
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=RANDOM_SEED),
            study_name=f"return_fold{fold}",
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,  # LightGBM already uses multiple threads
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"\n✓ Best RMSE: {best_value:.6f}")
        print(f"\n   Best parameters:")
        for param, value in best_params.items():
            print(f"     {param:20s}: {value}")

        # Retrain with best parameters
        print("\n4. Retraining with best parameters...")
        best_lgbm_params = get_return_search_space(study.best_trial)

        model_final = ReturnPredictor(
            params=best_lgbm_params,
            num_boost_round=LGBM_RETURN_ROUNDS,
            early_stopping_rounds=LGBM_RETURN_EARLY_STOP,
        )
        model_final.fit(X_train, y_train, X_val, y_val)

        # Get feature importance and select top features
        importances = model_final.get_feature_importance()
        selected_features = select_features_by_importance(importances, MAX_FEATURES)

        print(f"   Selected {len(selected_features)} top features")

        # Retrain with selected features
        print("\n5. Final training with selected features...")
        model_final = ReturnPredictor(
            params=best_lgbm_params,
            num_boost_round=LGBM_RETURN_ROUNDS,
            early_stopping_rounds=LGBM_RETURN_EARLY_STOP,
        )
        model_final.fit(
            X_train[selected_features], y_train, X_val[selected_features], y_val
        )

        # Save model and parameters
        model_final.save(MODELS_DIR / "returns" / f"fold_{fold}.pkl")
        engineer.save(MODELS_DIR / "returns" / f"engineer_fold_{fold}.pkl")
        pd.DataFrame({"feature": selected_features}).to_csv(
            MODELS_DIR / "returns" / f"features_fold_{fold}.csv", index=False
        )

        # Save tuning results
        results = {
            "fold": fold,
            "best_rmse": best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
        }
        all_fold_results.append(results)

        with open(results_dir / f"return_fold{fold}_study.pkl", "wb") as f:
            pickle.dump(study, f)

        print(f"\n✅ Fold {fold} complete!")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame(all_fold_results)
    summary_df.to_csv(
        results_dir / f"return_tuning_summary_{timestamp}.csv", index=False
    )

    print("\n" + "=" * 80)
    print("RETURN MODEL TUNING COMPLETE")
    print("=" * 80)
    print(f"\nAverage RMSE: {summary_df['best_rmse'].mean():.6f}")
    print(f"Results saved to: {results_dir}")


def tune_volatility_models(n_trials: int = 50, timeout: int = None, folds: list = None):
    """
    Tune volatility prediction model hyperparameters.

    Requires return models to be trained first.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds (per fold)
    folds : list, optional
        List of fold numbers to tune (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TUNING VOLATILITY MODELS")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))
    results_dir = MODELS_DIR / "tuning_results"
    results_dir.mkdir(exist_ok=True)

    all_fold_results = []

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Check if return model exists
        return_model_path = MODELS_DIR / "returns" / f"fold_{fold}.pkl"
        if not return_model_path.exists():
            print(f"⚠️  Return model not found for fold {fold}. Skipping...")
            continue

        # Load data
        print("\n1. Loading data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        # Load return model
        return_model = ReturnPredictor.load(return_model_path)
        engineer = FeatureEngineer.load(
            MODELS_DIR / "returns" / f"engineer_fold_{fold}.pkl"
        )
        selected_features = pd.read_csv(
            MODELS_DIR / "returns" / f"features_fold_{fold}.csv"
        )["feature"].tolist()

        # Engineer features
        print("\n2. Engineering features...")
        train_features = engineer.transform(train_df)
        val_features = engineer.transform(val_df)

        X_train = train_features[selected_features].fillna(0)
        X_val = val_features[selected_features].fillna(0)

        # Get return predictions
        print("\n3. Getting return predictions...")
        mu_train = return_model.predict(X_train)
        mu_val = return_model.predict(X_val)

        y_train = train_df[TARGET_COL]
        y_val = val_df[TARGET_COL]

        # Define objective function
        def objective(trial):
            params = get_volatility_search_space(trial)

            # Also tune calibration parameters
            bias_delta = trial.suggest_float("bias_delta", -0.5, 0.5)
            clip_low = trial.suggest_int("clip_low", 5, 15)
            clip_high = trial.suggest_int("clip_high", 85, 95)
            ewma_lambda = trial.suggest_float("ewma_lambda", 0.7, 0.99)

            # Train model
            model = VolatilityPredictor(
                params=params,
                num_boost_round=LGBM_VOL_ROUNDS,
                early_stopping_rounds=LGBM_VOL_EARLY_STOP,
                bias_delta=bias_delta,
                clip_percentiles=(clip_low, clip_high),
                ewma_lambda=ewma_lambda,
            )
            model.fit(train_df, y_train, mu_train, val_df, y_val, mu_val)

            # Evaluate on validation set
            sigma_val = model.predict(val_df, mu_val)

            # Compute QLIKE (lower is better)
            residuals = y_val - mu_val
            squared_resid = residuals**2
            qlike = np.mean(np.log(sigma_val**2) + squared_resid / (sigma_val**2))

            return qlike

        # Create study
        print(f"\n4. Running {n_trials} optimization trials...")
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=RANDOM_SEED),
            study_name=f"volatility_fold{fold}",
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"\n✓ Best QLIKE: {best_value:.6f}")
        print(f"\n   Best parameters:")
        for param, value in best_params.items():
            print(f"     {param:20s}: {value}")

        # Extract calibration parameters
        bias_delta = best_params.pop("bias_delta")
        clip_low = best_params.pop("clip_low")
        clip_high = best_params.pop("clip_high")
        ewma_lambda = best_params.pop("ewma_lambda")

        # Retrain with best parameters
        print("\n5. Retraining with best parameters...")
        best_lgbm_params = get_volatility_search_space(study.best_trial)

        model_final = VolatilityPredictor(
            params=best_lgbm_params,
            num_boost_round=LGBM_VOL_ROUNDS,
            early_stopping_rounds=LGBM_VOL_EARLY_STOP,
            bias_delta=bias_delta,
            clip_percentiles=(clip_low, clip_high),
            ewma_lambda=ewma_lambda,
        )
        model_final.fit(train_df, y_train, mu_train, val_df, y_val, mu_val)

        # Save
        model_final.save(MODELS_DIR / "volatility" / f"fold_{fold}.pkl")

        # Save tuning results
        results = {
            "fold": fold,
            "best_qlike": best_value,
            "best_params": best_params,
            "bias_delta": bias_delta,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "ewma_lambda": ewma_lambda,
            "n_trials": len(study.trials),
        }
        all_fold_results.append(results)

        with open(results_dir / f"volatility_fold{fold}_study.pkl", "wb") as f:
            pickle.dump(study, f)

        print(f"\n✅ Fold {fold} complete!")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame(all_fold_results)
    summary_df.to_csv(
        results_dir / f"volatility_tuning_summary_{timestamp}.csv", index=False
    )

    print("\n" + "=" * 80)
    print("VOLATILITY MODEL TUNING COMPLETE")
    print("=" * 80)
    print(f"\nAverage QLIKE: {summary_df['best_qlike'].mean():.6f}")
    print(f"Results saved to: {results_dir}")


def tune_meta_labeling(n_trials: int = 50, timeout: int = None, folds: list = None):
    """
    Tune meta-labeling classifier hyperparameters.

    Requires return models to be trained first.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds (per fold)
    folds : list, optional
        List of fold numbers to tune (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TUNING META-LABELING")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))
    results_dir = MODELS_DIR / "tuning_results"
    results_dir.mkdir(exist_ok=True)

    all_fold_results = []

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Check if return model exists
        return_model_path = MODELS_DIR / "returns" / f"fold_{fold}.pkl"
        if not return_model_path.exists():
            print(f"⚠️  Return model not found for fold {fold}. Skipping...")
            continue

        # Load data
        print("\n1. Loading data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        # Load models
        return_model = ReturnPredictor.load(return_model_path)
        engineer = FeatureEngineer.load(
            MODELS_DIR / "returns" / f"engineer_fold_{fold}.pkl"
        )
        selected_features = pd.read_csv(
            MODELS_DIR / "returns" / f"features_fold_{fold}.csv"
        )["feature"].tolist()

        # Engineer features
        print("\n2. Engineering features...")
        train_features = engineer.transform(train_df)
        val_features = engineer.transform(val_df)

        X_train = train_features[selected_features].fillna(0)
        X_val = val_features[selected_features].fillna(0)

        # Get predictions
        print("\n3. Getting return predictions...")
        mu_train = return_model.predict(X_train)
        mu_val = return_model.predict(X_val)

        y_train = train_df[TARGET_COL].values
        y_val = val_df[TARGET_COL].values

        # Define objective function
        def objective(trial):
            params = get_meta_search_space(trial)

            # Train meta-labeling
            model = MetaLabelPipeline(
                params=params,
                num_boost_round=LGBM_META_ROUNDS,
                early_stopping_rounds=LGBM_META_EARLY_STOP,
            )
            model.fit(X_train, mu_train, y_train, X_val, mu_val, y_val)

            # Evaluate on validation set
            meta_probs = model.predict_proba(X_val, mu_val)

            # Compute ROC AUC
            from sklearn.metrics import roc_auc_score

            meta_labels_val = (np.sign(mu_val) == np.sign(y_val)).astype(int)
            auc = roc_auc_score(meta_labels_val, meta_probs)

            # Return negative AUC (optuna minimizes, we want to maximize)
            return -auc

        # Create study
        print(f"\n4. Running {n_trials} optimization trials...")
        study = optuna.create_study(
            direction="minimize",  # Minimizing negative AUC = maximizing AUC
            sampler=TPESampler(seed=RANDOM_SEED),
            study_name=f"meta_fold{fold}",
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,
        )

        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive AUC

        print(f"\n✓ Best AUC: {best_value:.6f}")
        print(f"\n   Best parameters:")
        for param, value in best_params.items():
            print(f"     {param:20s}: {value}")

        # Retrain with best parameters
        print("\n5. Retraining with best parameters...")
        best_lgbm_params = get_meta_search_space(study.best_trial)

        model_final = MetaLabelPipeline(
            params=best_lgbm_params,
            num_boost_round=LGBM_META_ROUNDS,
            early_stopping_rounds=LGBM_META_EARLY_STOP,
        )
        model_final.fit(X_train, mu_train, y_train, X_val, mu_val, y_val)

        # Save
        model_final.save(MODELS_DIR / "meta" / f"fold_{fold}.pkl")

        # Save tuning results
        results = {
            "fold": fold,
            "best_auc": best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
        }
        all_fold_results.append(results)

        with open(results_dir / f"meta_fold{fold}_study.pkl", "wb") as f:
            pickle.dump(study, f)

        print(f"\n✅ Fold {fold} complete!")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame(all_fold_results)
    summary_df.to_csv(results_dir / f"meta_tuning_summary_{timestamp}.csv", index=False)

    print("\n" + "=" * 80)
    print("META-LABELING TUNING COMPLETE")
    print("=" * 80)
    print(f"\nAverage AUC: {summary_df['best_auc'].mean():.6f}")
    print(f"Results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for predictive models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune all models with default settings
  python tune.py --stage all
  
  # Tune only return models with 100 trials
  python tune.py --stage returns --n-trials 100
  
  # Tune volatility models on specific folds
  python tune.py --stage volatility --folds 1,2,3
  
  # Tune with timeout (1 hour per fold)
  python tune.py --stage meta --timeout 3600
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "returns", "volatility", "meta"],
        default="all",
        help="Which component to tune (respects dependencies)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials per fold (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per fold (default: None)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold numbers (e.g., '1,2,3'). Default: all folds",
    )

    args = parser.parse_args()

    # Parse folds
    folds = None
    if args.folds:
        folds = [int(f) for f in args.folds.split(",")]

    # Check if processed data exists
    fold1_path = PROCESSED_DIR / "train_fold1.csv"
    if not fold1_path.exists():
        print("\n" + "=" * 80)
        print("ERROR: Processed data not found!")
        print("=" * 80)
        print("\nPlease run preprocessing first:")
        print("  python train.py --stage preprocess")
        sys.exit(1)

    # Print configuration
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING CONFIGURATION")
    print("=" * 80)
    print(f"\nStage:     {args.stage}")
    print(f"Trials:    {args.n_trials} per fold")
    print(f"Timeout:   {args.timeout if args.timeout else 'None'}")
    print(f"Folds:     {folds if folds else 'All'}")
    print("\nOptuna sampler: TPE (Tree-structured Parzen Estimator)")

    # Execute tuning stages (respecting dependencies)
    start_time = datetime.now()

    if args.stage in ["all", "returns"]:
        tune_return_models(args.n_trials, args.timeout, folds)

    if args.stage in ["all", "volatility"]:
        tune_volatility_models(args.n_trials, args.timeout, folds)

    if args.stage in ["all", "meta"]:
        tune_meta_labeling(args.n_trials, args.timeout, folds)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("✅ TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nTotal duration: {duration}")
    print(f"\nTuning results saved to: {MODELS_DIR / 'tuning_results'}")
    print("\nModels have been retrained with optimized hyperparameters.")
    print("Run evaluation to see improvements:")
    print("  python evaluate.py")


if __name__ == "__main__":
    main()
