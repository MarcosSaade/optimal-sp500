"""
Example Usage

This file demonstrates how to use the package components.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import components
from src.preprocessing import PurgedKFold, MarketDataPreprocessor, load_and_prepare_data
from src.features import FeatureEngineer
from src.returns import ReturnPredictor
from src.volatility import VolatilityPredictor
from src.meta_labeling import MetaLabelPipeline
from src.allocation import RegimeDependentAllocator
from src.config import *


def example_preprocessing():
    """Example: Preprocess data and create CV splits."""
    print("Example 1: Preprocessing\n" + "=" * 50)

    # Load data
    data_path = DATA_DIR / "train.csv"
    data = load_and_prepare_data(data_path)
    print(f"Loaded {len(data):,} rows")

    # Create purged CV splits
    cv = PurgedKFold(n_splits=5, purge_gap=1, embargo_pct=0.01)
    splits = cv.split(data)
    print(f"Created {len(splits)} CV folds")

    # Preprocess first fold
    train_idx, val_idx = splits[0]
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    # Get features
    special_cols = ["date_id", "risk_free_rate", TARGET_COL]
    feature_cols = [col for col in data.columns if col not in special_cols]
    continuous_features = [f for f in feature_cols if not f.startswith("D")]

    # Fit preprocessor
    preprocessor = MarketDataPreprocessor(
        continuous_features=continuous_features,
        winsorize_config=WINSORIZE_CONFIG,
        late_starter_features=LATE_STARTER_FEATURES,
    )

    train_processed, train_masks = preprocessor.fit_transform(train_data)
    val_processed, val_masks = preprocessor.transform(val_data)

    print(f"Train: {train_processed.shape}, Val: {val_processed.shape}")


def example_feature_engineering():
    """Example: Engineer features."""
    print("\n\nExample 2: Feature Engineering\n" + "=" * 50)

    # Load processed data (assuming preprocessing was done)
    train_df = pd.read_csv(PROCESSED_DIR / "train_fold1.csv")

    # Get features
    special_cols = ["date_id", "risk_free_rate", TARGET_COL]
    feature_cols = [col for col in train_df.columns if col not in special_cols]

    # Fit feature engineer
    engineer = FeatureEngineer(continuous_features=feature_cols)
    train_features = engineer.fit_transform(train_df)

    print(f"Original features: {len(feature_cols)}")
    print(f"Engineered features: {train_features.shape[1]}")


def example_return_prediction():
    """Example: Train return prediction model."""
    print("\n\nExample 3: Return Prediction\n" + "=" * 50)

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train_fold1.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_fold1.csv")

    # Engineer features
    special_cols = ["date_id", "risk_free_rate", TARGET_COL]
    feature_cols = [col for col in train_df.columns if col not in special_cols]

    engineer = FeatureEngineer(continuous_features=feature_cols)
    train_features = engineer.fit_transform(train_df)
    val_features = engineer.transform(val_df)

    # Prepare data
    X_cols = [col for col in train_features.columns if col not in special_cols]
    X_cols = [
        col
        for col in X_cols
        if train_features[col].notna().sum() > len(train_features) * 0.5
    ]

    X_train = train_features[X_cols].fillna(0)
    y_train = train_features[TARGET_COL]

    X_val = val_features[X_cols].fillna(0)
    y_val = val_features[TARGET_COL]

    # Train model
    model = ReturnPredictor()
    model.fit(X_train, y_train, X_val, y_val)

    # Predict
    predictions = model.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Mean prediction: {predictions.mean():.6f}")

    # Get top features
    top_features = model.get_feature_importance(top_n=10)
    print("\nTop 10 features:")
    for i, (feat, imp) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feat:40s} {imp:>10.1f}")


def example_volatility_prediction():
    """Example: Train volatility prediction model."""
    print("\n\nExample 4: Volatility Prediction\n" + "=" * 50)

    # Load data and return model (assuming they exist)
    train_df = pd.read_csv(PROCESSED_DIR / "train_fold1.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_fold1.csv")

    return_model = ReturnPredictor.load(MODELS_DIR / "returns" / "fold_1.pkl")
    engineer = FeatureEngineer.load(MODELS_DIR / "returns" / "engineer_fold_1.pkl")

    # Get return predictions
    train_features = engineer.transform(train_df)
    val_features = engineer.transform(val_df)

    selected_features = pd.read_csv(MODELS_DIR / "returns" / "features_fold_1.csv")[
        "feature"
    ].tolist()

    X_train = train_features[selected_features].fillna(0)
    X_val = val_features[selected_features].fillna(0)

    mu_train = return_model.predict(X_train)
    mu_val = return_model.predict(X_val)

    # Train volatility model
    vol_model = VolatilityPredictor()
    vol_model.fit(
        train_df, train_df[TARGET_COL], mu_train, val_df, val_df[TARGET_COL], mu_val
    )

    # Predict
    sigma_val = vol_model.predict(val_df)
    print(f"Volatility predictions shape: {sigma_val.shape}")
    print(f"Mean volatility: {sigma_val.mean():.6f}")
    print(f"Volatility range: [{sigma_val.min():.6f}, {sigma_val.max():.6f}]")


def example_meta_labeling():
    """Example: Train meta-labeling classifier."""
    print("\n\nExample 5: Meta-Labeling\n" + "=" * 50)

    # Load data and models (assuming they exist)
    train_df = pd.read_csv(PROCESSED_DIR / "train_fold1.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_fold1.csv")

    return_model = ReturnPredictor.load(MODELS_DIR / "returns" / "fold_1.pkl")
    engineer = FeatureEngineer.load(MODELS_DIR / "returns" / "engineer_fold_1.pkl")

    # Get features and predictions
    train_features = engineer.transform(train_df)
    val_features = engineer.transform(val_df)

    selected_features = pd.read_csv(MODELS_DIR / "returns" / "features_fold_1.csv")[
        "feature"
    ].tolist()

    X_train = train_features[selected_features].fillna(0)
    X_val = val_features[selected_features].fillna(0)

    mu_train = return_model.predict(X_train)
    mu_val = return_model.predict(X_val)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    # Train meta-labeling
    meta_model = MetaLabelPipeline()
    meta_model.fit(X_train, mu_train, y_train, X_val, mu_val, y_val)

    # Predict meta-probabilities
    meta_proba = meta_model.predict_proba(X_val, mu_val)
    print(f"Meta-probabilities shape: {meta_proba.shape}")
    print(f"Mean meta-probability: {meta_proba.mean():.3f}")
    print(f"Meta-prob range: [{meta_proba.min():.3f}, {meta_proba.max():.3f}]")


def example_allocation():
    """Example: Allocate positions."""
    print("\n\nExample 6: Position Allocation\n" + "=" * 50)

    # Load models and data (assuming they exist)
    val_df = pd.read_csv(PROCESSED_DIR / "val_fold1.csv")

    return_model = ReturnPredictor.load(MODELS_DIR / "returns" / "fold_1.pkl")
    vol_model = VolatilityPredictor.load(MODELS_DIR / "volatility" / "fold_1.pkl")
    engineer = FeatureEngineer.load(MODELS_DIR / "returns" / "engineer_fold_1.pkl")

    # Get predictions
    val_features = engineer.transform(val_df)
    selected_features = pd.read_csv(MODELS_DIR / "returns" / "features_fold_1.csv")[
        "feature"
    ].tolist()

    X_val = val_features[selected_features].fillna(0)
    mu = return_model.predict(X_val)
    sigma = vol_model.predict(val_df)

    # Allocate
    allocator = RegimeDependentAllocator()
    allocations = allocator.allocate(mu, sigma, data=val_features)

    print(f"Allocations shape: {allocations.shape}")
    print(f"Mean allocation: {allocations.mean():.3f}")
    print(f"Allocation range: [{allocations.min():.3f}, {allocations.max():.3f}]")

    # Apply meta-labeling (if available)
    meta_path = MODELS_DIR / "meta" / "fold_1.pkl"
    if meta_path.exists():
        meta_model = MetaLabelPipeline.load(meta_path)
        meta_proba = meta_model.predict_proba(X_val, mu)
        allocations_meta = meta_model.scale_positions(allocations, meta_proba)

        print(f"\nWith meta-labeling:")
        print(f"  Mean allocation: {allocations_meta.mean():.3f}")
        print(
            f"  Allocation range: [{allocations_meta.min():.3f}, {allocations_meta.max():.3f}]"
        )


def example_complete_pipeline():
    """Example: Complete pipeline from data to allocations."""
    print("\n\nExample 7: Complete Pipeline\n" + "=" * 50)

    # This assumes models are already trained
    # Load validation data
    val_df = pd.read_csv(PROCESSED_DIR / "val_fold1.csv")

    # Load all models
    return_model = ReturnPredictor.load(MODELS_DIR / "returns" / "fold_1.pkl")
    vol_model = VolatilityPredictor.load(MODELS_DIR / "volatility" / "fold_1.pkl")
    meta_model = MetaLabelPipeline.load(MODELS_DIR / "meta" / "fold_1.pkl")
    engineer = FeatureEngineer.load(MODELS_DIR / "returns" / "engineer_fold_1.pkl")
    allocator = RegimeDependentAllocator()

    # Feature engineering
    val_features = engineer.transform(val_df)
    selected_features = pd.read_csv(MODELS_DIR / "returns" / "features_fold_1.csv")[
        "feature"
    ].tolist()
    X_val = val_features[selected_features].fillna(0)

    # Predict returns
    mu = return_model.predict(X_val)

    # Predict volatility
    sigma = vol_model.predict(val_df)

    # Allocate positions
    allocations_base = allocator.allocate(mu, sigma, data=val_features)

    # Apply meta-labeling
    meta_proba = meta_model.predict_proba(X_val, mu)
    allocations_final = meta_model.scale_positions(allocations_base, meta_proba)

    # Compute returns
    forward_returns = val_df["forward_returns"].values[: len(allocations_final)]
    risk_free = val_df["risk_free_rate"].values[: len(allocations_final)]

    excess_returns = forward_returns - risk_free
    portfolio_returns = allocations_final * excess_returns

    # Compute Sharpe
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

    print(f"Number of samples: {len(allocations_final)}")
    print(f"Mean allocation: {allocations_final.mean():.3f}")
    print(f"Mean portfolio return: {portfolio_returns.mean():.6f}")
    print(f"Portfolio volatility: {portfolio_returns.std():.6f}")
    print(f"Sharpe ratio: {sharpe:.3f}")


if __name__ == "__main__":
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("\nNote: These examples assume data and models exist.")
    print("Run 'python train.py --stage all' first.\n")

    try:
        example_preprocessing()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_feature_engineering()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_return_prediction()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_volatility_prediction()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_meta_labeling()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_allocation()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    try:
        example_complete_pipeline()
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("=" * 80)
