"""
Training Orchestration

Handles training of all model components with dependency management.

Components:
1. Preprocessing + CV splits
2. Feature engineering
3. Return prediction (no dependencies)
4. Volatility prediction (depends on returns)
5. Meta-labeling (depends on returns)

Usage:
    python train.py --stage all              # Train everything
    python train.py --stage preprocess       # Just preprocess data
    python train.py --stage returns          # Train return models only
    python train.py --stage volatility       # Train volatility models
    python train.py --stage meta             # Train meta-labeling
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.preprocessing import (
    PurgedKFold,
    MarketDataPreprocessor,
    load_and_prepare_data,
)
from src.features import FeatureEngineer
from src.returns import ReturnPredictor
from src.volatility import VolatilityPredictor
from src.meta_labeling import MetaLabelPipeline


def preprocess_data(force: bool = False):
    """
    Preprocess data and create CV folds.

    Parameters
    ----------
    force : bool
        Force reprocessing even if processed files exist
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)

    # Check if already processed
    fold1_path = PROCESSED_DIR / "train_fold1.csv"
    if fold1_path.exists() and not force:
        print("✓ Processed data already exists. Use --force to reprocess.")
        return

    # Load raw data
    print("\n1. Loading raw data...")
    data_path = DATA_DIR / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = load_and_prepare_data(data_path)
    print(f"   Loaded {len(data):,} rows, {len(data.columns)} columns")

    # Create purged CV splits
    print("\n2. Creating purged K-fold splits...")
    cv = PurgedKFold(
        n_splits=CV_N_SPLITS,
        min_train_pct=CV_MIN_TRAIN_PCT,
        purge_gap=CV_PURGE_GAP,
        embargo_pct=CV_EMBARGO_PCT,
        verbose=True,
    )

    splits = cv.split(data)

    # Get feature columns
    special_cols = ["date_id", "risk_free_rate", TARGET_COL]
    feature_cols = [col for col in data.columns if col not in special_cols]
    feature_cols = [col for col in feature_cols if col not in FEATURES_TO_DROP]

    # Separate continuous and binary features
    continuous_features = [f for f in feature_cols if not f.startswith("D")]
    binary_features = [f for f in feature_cols if f.startswith("D")]

    # Process each fold
    print("\n3. Preprocessing each fold...")
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n   Fold {fold_idx}:")

        # Split data
        train_data = data.iloc[train_idx].copy()
        val_data = data.iloc[val_idx].copy()

        print(f"     Train: {len(train_data):,} rows")
        print(f"     Val:   {len(val_data):,} rows")

        # Fit preprocessor on training fold
        preprocessor = MarketDataPreprocessor(
            continuous_features=continuous_features,
            winsorize_config=WINSORIZE_CONFIG,
            late_starter_features=LATE_STARTER_FEATURES,
        )

        # Fit and transform
        train_processed, train_masks = preprocessor.fit_transform(train_data)
        val_processed, val_masks = preprocessor.transform(val_data)

        # Combine with masks
        train_final = pd.concat([train_processed, train_masks], axis=1)
        val_final = pd.concat([val_processed, val_masks], axis=1)

        # Save
        train_final.to_csv(PROCESSED_DIR / f"train_fold{fold_idx}.csv", index=False)
        val_final.to_csv(PROCESSED_DIR / f"val_fold{fold_idx}.csv", index=False)
        preprocessor.save(PROCESSED_DIR / f"preprocessor_fold{fold_idx}.pkl")

        print(f"     ✓ Saved processed fold {fold_idx}")

    print("\n✅ Preprocessing complete!")
    print(f"   Processed data saved to: {PROCESSED_DIR}")


def train_return_models(folds: list = None):
    """
    Train return prediction models.

    Parameters
    ----------
    folds : list, optional
        List of fold numbers to train (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TRAINING RETURN MODELS")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Load processed data
        print("\n1. Loading processed data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        print(f"   Train: {train_df.shape}")
        print(f"   Val:   {val_df.shape}")

        # Engineer features
        print("\n2. Engineering features...")
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        feature_cols = [col for col in train_df.columns if col not in special_cols]

        engineer = FeatureEngineer(continuous_features=feature_cols)

        # Fit on train, transform both
        train_features = engineer.fit_transform(train_df)
        val_features = engineer.transform(val_df)

        print(f"   Created {train_features.shape[1]} features")

        # Prepare training data
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

        # Train model
        print("\n3. Training LightGBM...")
        model = ReturnPredictor()
        model.fit(X_train, y_train, X_val, y_val)

        # Get feature importance
        importances = model.get_feature_importance()

        # Feature selection
        from src.features import select_features_by_importance

        selected_features = select_features_by_importance(importances, MAX_FEATURES)

        print(f"   Selected {len(selected_features)} top features")
        print("\n   Top 10 features:")
        for i, (feat, imp) in enumerate(list(importances.items())[:10], 1):
            print(f"     {i:2d}. {feat:45s}  {imp:>10.1f}")

        # Retrain with selected features
        print("\n4. Retraining with selected features...")
        model_final = ReturnPredictor()
        model_final.fit(
            X_train[selected_features], y_train, X_val[selected_features], y_val
        )

        # Save
        model_final.save(MODELS_DIR / "returns" / f"fold_{fold}.pkl")
        engineer.save(MODELS_DIR / "returns" / f"engineer_fold_{fold}.pkl")

        # Save feature list
        pd.DataFrame({"feature": selected_features}).to_csv(
            MODELS_DIR / "returns" / f"features_fold_{fold}.csv", index=False
        )

        print(f"\n✅ Fold {fold} complete!")


def train_volatility_models(folds: list = None):
    """
    Train volatility prediction models.

    Requires return models to be trained first.

    Parameters
    ----------
    folds : list, optional
        List of fold numbers to train (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TRAINING VOLATILITY MODELS")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Load processed data
        print("\n1. Loading data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        # Load return model
        return_model = ReturnPredictor.load(MODELS_DIR / "returns" / f"fold_{fold}.pkl")
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

        # Train volatility model
        print("\n4. Training volatility model...")
        vol_model = VolatilityPredictor()
        vol_model.fit(train_df, y_train, mu_train, val_df, y_val, mu_val)

        # Save
        vol_model.save(MODELS_DIR / "volatility" / f"fold_{fold}.pkl")

        print(f"\n✅ Fold {fold} volatility model complete!")


def train_meta_labeling(folds: list = None):
    """
    Train meta-labeling classifiers.

    Requires return models to be trained first.

    Parameters
    ----------
    folds : list, optional
        List of fold numbers to train (default: all folds)
    """
    print("\n" + "=" * 80)
    print("TRAINING META-LABELING")
    print("=" * 80)

    folds = folds or list(range(1, CV_N_SPLITS + 1))

    for fold in folds:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

        # Load data
        print("\n1. Loading data...")
        train_df = pd.read_csv(PROCESSED_DIR / f"train_fold{fold}.csv")
        val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

        # Load models
        return_model = ReturnPredictor.load(MODELS_DIR / "returns" / f"fold_{fold}.pkl")
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

        # Train meta-labeling
        print("\n4. Training meta-labeling...")
        meta_model = MetaLabelPipeline()
        meta_model.fit(X_train, mu_train, y_train, X_val, mu_val, y_val)

        # Save
        meta_model.save(MODELS_DIR / "meta" / f"fold_{fold}.pkl")

        # Train calibrated version
        print("\n5. Training calibrated meta-labeling...")
        meta_model_calibrated = MetaLabelPipeline(
            use_calibration=True, calibration_method="isotonic"
        )
        meta_model_calibrated.fit(X_train, mu_train, y_train, X_val, mu_val, y_val)

        # Save calibrated version
        meta_model_calibrated.save(MODELS_DIR / "meta" / f"fold_{fold}_calibrated.pkl")

        print(f"\n✅ Fold {fold} meta-labeling complete!")


def main():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "preprocess", "returns", "volatility", "meta"],
        default="all",
        help="Which stage to train",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold numbers (e.g., '1,2,3'). Default: all folds",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing/retraining"
    )

    args = parser.parse_args()

    # Parse folds
    folds = None
    if args.folds:
        folds = [int(f) for f in args.folds.split(",")]

    # Execute stages
    if args.stage in ["all", "preprocess"]:
        preprocess_data(force=args.force)

    if args.stage in ["all", "returns"]:
        train_return_models(folds)

    if args.stage in ["all", "volatility"]:
        train_volatility_models(folds)

    if args.stage in ["all", "meta"]:
        train_meta_labeling(folds)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
