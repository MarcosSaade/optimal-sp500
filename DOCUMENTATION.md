# Project Documentation

## Overview

This project implements a sophisticated machine learning pipeline for predicting S&P 500 excess returns using ensemble methods, meta-labeling, and regime-dependent position sizing.

**Performance**: See evaluation results (per-fold and aggregate statistics are recorded by the evaluation pipeline)

## Architecture

### Pipeline Flow

```
Raw Data
    ↓
[1] Preprocessing & Purged CV
    ↓
[2] Feature Engineering (2000+ features → 150 selected)
    ↓
[3] Return Prediction (LightGBM)
    ↓
[4] Volatility Prediction (LightGBM)
    ↓
[5] Meta-Labeling (Confidence Scoring)
    ↓
[6] Position Allocation (Regime-Dependent Kelly)
    ↓
Strategy Returns
```

### Key Components

#### 1. Purged K-Fold Cross-Validation (`src/preprocessing.py`)

**Purpose**: Prevent label leakage in overlapping time series

**Implementation**:
- 5 expanding window folds
- Purges training samples whose labels overlap with validation period
- 1% embargo period to prevent serial correlation


**Why Critical**: Standard CV can inflate validation scores due to leakage in time series data

#### 2. Feature Engineering (`src/features.py`)

**Temporal Features**:
- Rolling statistics (mean, std, min, max) over multiple windows
- Momentum indicators (EMA, lag features)
- Volatility-of-volatility

**Volatility Features**:
- Historical volatility (multiple windows)
- EWMA volatility
- Range-based volatility
- Absolute and squared returns

**Regime Features**:
- Volatility regime classification (low, medium, high)
- Regime-conditional features
- Transition indicators

**Dimensionality Reduction**:
- Correlation clustering (groups features with r > 0.95)
- PCA per cluster (retains 85% variance)
- Feature selection by importance (top 150 features)

#### 3. Return Prediction (`src/returns.py`)

**Model**: LightGBM Regressor

**Features**: ~150 engineered features selected by importance

**Training**:
- Early stopping on validation set
- Hyperparameters optimized for financial time series
- Feature importance analysis for interpretability

**Output**: Predicted 1-day excess returns (μ)

#### 4. Volatility Prediction (`src/volatility.py`)

**Model**: LightGBM Regressor

**Target**: Log-variance of residuals from return predictions
```python
target = log((actual - predicted)² + ε)
```

**Features**:
- Historical volatility measures
- GARCH-like features (squared returns)
- Volatility trends and percentiles

**Calibration**:
- Bias correction: √(σ² + δ²)
- Clipping to training distribution (10th-90th percentile)
- EWMA smoothing (λ = 0.9)

**Output**: Calibrated conditional volatility (σ)

#### 5. Meta-Labeling (`src/meta_labeling.py`)

**Purpose**: Separate prediction from position sizing

**Concept** (from textbook Chapter 3):
- Primary model predicts direction and magnitude
- Meta-model predicts if primary model is correct
- Scale positions by confidence

**Meta-Label Generation**:
```python
meta_label = 1 if sign(prediction) == sign(actual) else 0
```

**Meta-Features**:
- Original features (market conditions, regimes)
- Prediction magnitude (|μ|)
- Prediction sign
- Prediction volatility (rolling std of past predictions)

**Meta-Classifier**: LightGBM binary classifier

**Output**: P(primary prediction is correct)

**Position Scaling**:
```python
final_allocation = base_allocation × meta_probability
```

**Impact**: measurable average improvement in risk-adjusted returns (see evaluation_results.csv)

#### 6. Position Allocation (`src/allocation.py`)

**Method**: Regime-Dependent Kelly Criterion

**Formula**:
```python
allocation = k × tanh(μ / (σ² × s))
```

**Regime-Specific Parameters**:
- Low volatility: k = 0.98 (aggressive)
- Medium volatility: k = 0.71 (moderate)
- High volatility: k = 0.50 (conservative)
- Scale: s = 1.15

**Why Regime-Dependent**:
- Stable markets tolerate larger positions
- Volatile markets require caution
- Reduces drawdowns while maintaining returns

**Constraints**: Allocations ∈ [0, 2] (0% to 200% of capital)

## Configuration (`src/config.py`)

All hyperparameters and settings in one place:

### Data Preprocessing
- `CV_N_SPLITS = 5`: Number of CV folds
- `CV_PURGE_GAP = 1`: Samples to purge (1-day forward returns)
- `CV_EMBARGO_PCT = 0.01`: Embargo percentage
- `WINSORIZE_CONFIG`: Feature-specific outlier thresholds

### Feature Engineering
- `TEMPORAL_WINDOWS = [5, 21, 63]`: Rolling window sizes
- `EXTENDED_WINDOWS = [10, 42, 126]`: Extended horizons
- `MAX_FEATURES = 150`: Maximum features for final model
- `PCA_VARIANCE_THRESHOLD = 0.85`: PCA variance retention

### Models
- `LGBM_RETURN_PARAMS`: Return model hyperparameters
- `LGBM_VOL_PARAMS`: Volatility model hyperparameters
- `LGBM_META_PARAMS`: Meta-labeling hyperparameters

### Allocation
- `KELLY_K_LOW/MEDIUM/HIGH`: Regime-specific Kelly fractions
- `KELLY_SCALE`: Soft-capping scale parameter
- `ALLOCATION_MIN/MAX`: Position size constraints

## Training Script (`train.py`)

### Dependency Management

Models have dependencies:
1. **Return Model**: No dependencies (train first)
2. **Volatility Model**: Depends on return predictions
3. **Meta-Labeling**: Depends on return predictions

The training script handles this automatically:

```bash
# This trains in correct order
python train.py --stage all
```

### Stages

**Preprocess**: Create purged CV folds
```bash
python train.py --stage preprocess
```

**Returns**: Train return prediction models
```bash
python train.py --stage returns
```

**Volatility**: Train volatility models (requires returns)
```bash
python train.py --stage volatility
```

**Meta**: Train meta-labeling (requires returns)
```bash
python train.py --stage meta
```

### Advanced Usage

**Train specific folds**:
```bash
python train.py --stage returns --folds 1,2,3
```

**Force reprocessing**:
```bash
python train.py --stage preprocess --force
```

## Evaluation Script (`evaluate.py`)

### Metrics

**Primary**: Sharpe Ratio (annualized)
```python
sharpe = √252 × E[alloc × (return - rf)] / σ[alloc × (return - rf)]
```

**Secondary**:
- Directional accuracy
- Mean portfolio return
- Portfolio volatility
- Meta-probability statistics

### Usage

**Evaluate all folds**:
```bash
python evaluate.py
```

**Without meta-labeling**:
```bash
python evaluate.py --no-meta
```

**Single fold**:
```bash
python evaluate.py --fold 3
```

### Output

Results saved to `evaluation_results.csv`:
- Per-fold metrics
- Aggregate statistics
- Meta-labeling impact

## Performance Results

Performance numbers have been removed from the public documentation. Detailed per-fold and aggregate evaluation metrics are saved by the evaluation pipeline to `evaluation_results.csv`. Consult that file for numerical results and exportable reports.

## Theoretical Foundation

Based on established financial machine learning literature and best practices:

- Purged K-Fold prevents label leakage
- Embargo periods reduce serial correlation issues
- Meta-labeling separates prediction quality from position sizing

The documentation focuses on methods and reproducible procedures. Implementation details and citations may be added to a references section as required for publication.

## Code Quality

### Organization
- Modular design (each component in separate file)
- Clear separation of concerns
- Minimal dependencies between modules

### Documentation
- Comprehensive docstrings
- Type hints for function signatures
- Inline comments for complex logic

### Configuration
- All hyperparameters in `config.py`
- Easy to modify and experiment
- Clear parameter descriptions

### Error Handling
- Validation of inputs
- Informative error messages
- Graceful fallbacks

## Enhancements and Roadmap

Potential enhancements are tracked in the project issue tracker. The documentation intentionally omits speculative numeric impact estimates; implementation and empirical validation are required before reporting measurable changes.

## File Structure

```
build/
├── README.md                   # Project overview and methodology
├── QUICKSTART.md               # Getting started guide
├── DOCUMENTATION.md            # This file - comprehensive documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── data/                       # Data directory
│   ├── README.md              # Data format and placement
│   └── processed/             # Auto-generated CV folds
│
├── models/                     # Trained models (auto-generated)
│   ├── returns/               # Return prediction models
│   ├── volatility/            # Volatility prediction models
│   └── meta/                  # Meta-labeling classifiers
│
├── notebooks/                  # Jupyter notebooks
│   └── eda.ipynb              # Exploratory data analysis
│
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration and hyperparameters
│   ├── preprocessing.py       # Data preprocessing and purged CV
│   ├── features.py            # Feature engineering
│   ├── returns.py             # Return prediction model
│   ├── volatility.py          # Volatility prediction model
│   ├── meta_labeling.py       # Meta-labeling pipeline
│   └── allocation.py          # Position allocation strategies
│
├── train.py                    # Training orchestration
└── evaluate.py                 # Model evaluation
```

## Dependencies

### Core
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: ML utilities, preprocessing
- `lightgbm`: Gradient boosting models

### Visualization
- `matplotlib`, `seaborn`: Plotting

### Statistics
- `scipy`, `statsmodels`: Statistical analysis

### Development
- `jupyter`: Interactive analysis
- `tqdm`: Progress bars

See `requirements.txt` for version details.

## Reproducibility

### Random Seeds
All random operations use `RANDOM_SEED = 42` from config.

### Deterministic Operations
- Fixed CV splits (purged K-fold)
- Consistent preprocessing (fit on train only)
- Deterministic model training (seeded)

### Version Control
Track changes to hyperparameters in `src/config.py`.

## Testing

### Validation Strategy
- 5-fold purged cross-validation
- Honest out-of-sample evaluation
- No data leakage

### Sanity Checks
- Temporal ordering preserved
- No NaNs in model inputs
- Allocations within bounds
- Positive Sharpe ratios

## Maintenance

### Adding New Features
1. Implement in `src/features.py`
2. Add to `FeatureEngineer.transform()`
3. Retrain models
4. Evaluate impact

### Tuning Hyperparameters
1. Modify `src/config.py`
2. Retrain affected models
3. Compare evaluation results

### Debugging
1. Check preprocessing output
2. Verify feature engineering
3. Inspect model predictions
4. Review allocation logic

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
