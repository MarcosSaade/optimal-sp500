# Feature Engineering Pipeline Update

## Summary

The feature engineering pipeline has been enhanced with **fractional differencing** and **polynomial/interaction features** based on LGBM importance-guided feature selection.

## New Pipeline Architecture

### Stage 1: Base Feature Generation
- Temporal features (rolling stats, lags, EMA)
- Volatility features (historical vol, EWMA vol, vol-of-vol)
- Regime features (volatility-based regimes)
- PCA features (dimensionality reduction)

**Output:** ~2,100 features

### Stage 2: Fractional Differencing
- Applied to all continuous features
- Uses fractional order d=0.5 for semi-stationarity
- Creates features like `feature_fd` for each continuous feature

**Output:** ~2,100 features (105 new fractional diff features)

### Stage 3: First Feature Selection (Top-K)
- Train LightGBM on all features including fractional differencing
- Select top 80 features by importance
- Ensures only the most predictive features move forward

**Output:** 80 features

### Stage 4: Polynomial & Interaction Features
- Squared terms: `feature_sq` for each top-80 feature
- Pairwise interactions: `feature1_x_feature2` for top 20 features
- Creates ~270 new polynomial/interaction features

**Output:** ~350 features (80 base + 270 polynomial)

### Stage 5: Final Feature Selection (Top 150)
- Train another LightGBM on full feature set
- Select top 150 features by importance
- Final feature set for model training

**Output:** 150 features

## Configuration Parameters

Added to `src/config.py`:

```python
# Fractional differencing parameters
FRAC_DIFF_D = 0.5  # Fractional difference order
FRAC_DIFF_THRESHOLD = 0.01  # Minimum weight threshold

# Feature selection stages
TOP_K_AFTER_FRAC_DIFF = 80  # Top K features after fractional differencing
POLY_DEGREE = 2  # Polynomial degree for interactions
```

## Results

The final 150 features consist of:
- **42 fractional differencing features** (28%)
- **104 polynomial/interaction features** (69%)
- **4 other base features** (3%)

### Top Features by Importance

1. `forward_returns_fd` - Fractionally differenced forward returns
2. `forward_returns_fd_sq` - Squared fractional diff returns
3. `forward_returns_fd_x_E3_fd` - Interaction between FD returns and FD earnings
4. `forward_returns_fd_x_V3_ema50` - FD returns × volatility EMA
5. `forward_returns_fd_x_max_drawdown_63` - FD returns × max drawdown

## Benefits

1. **Stationarity**: Fractional differencing helps achieve stationarity while preserving memory
2. **Non-linear Relationships**: Polynomial features capture quadratic relationships
3. **Feature Interactions**: Interaction terms capture synergies between features
4. **Automated Selection**: LGBM importance ensures only predictive features are retained
5. **Controlled Complexity**: Two-stage selection prevents feature explosion

## Usage

The pipeline is fully integrated into the existing training workflow:

```python
from src.features import FeatureEngineer

# Initialize (uses config defaults)
engineer = FeatureEngineer(continuous_features=feature_list)

# Fit on training data
engineer.fit(train_df)

# Transform train and validation
train_features = engineer.transform(train_df)
val_features = engineer.transform(val_df)

# Save for later use
engineer.save('path/to/engineer.pkl')
```

## Next Steps

Run the full training pipeline to evaluate the impact on model performance:

```bash
python tune.py
```

This will retrain all models with the new feature engineering pipeline and report the Sharpe ratio improvement.
