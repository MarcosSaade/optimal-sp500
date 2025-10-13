# Hull Tactical Market Prediction

**S&P 500 Excess Return Forecasting with Meta-Labeling**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io/)


## 📊 Overview

This project implements a machine learning pipeline for predicting S&P 500 excess returns using a sophisticated multi-stage approach:

1. **Return Prediction**: LightGBM model predicts expected excess returns
2. **Volatility Prediction**: LightGBM model predicts conditional volatility  
3. **Meta-Labeling**: Confidence-based position sizing via meta-classifier
4. **Position Allocation**: Regime-dependent Kelly criterion for optimal sizing

**Performance**: See evaluation outputs for detailed metrics (recorded by the evaluation pipeline)

---

## 🎯 Key Features

### Robust Cross-Validation
- **Purged K-Fold CV**: Prevents label leakage in overlapping time series
- **Embargo Period**: Accounts for serial correlation
- Based on established financial machine learning literature

### Meta-Labeling Framework
- Separates prediction (primary model) from confidence (meta-model)
- Scales positions by prediction confidence
- Demonstrated to improve risk-adjusted returns in empirical evaluations (see evaluation pipeline)

### Regime-Dependent Allocation
- Adapts position sizing to market volatility
- Conservative in high volatility, aggressive in low volatility
- Reduces drawdowns while maintaining returns

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

Place your data file in the `data/` directory:
- `train.csv`: Training data with features and target

### Run Pipeline

```bash
# 1. Preprocess data and create CV folds
python src/train.py --stage preprocess

# 2. Train all models (returns, volatility, meta-labeling)
python src/train.py --stage all

# 3. Evaluate performance
python src/evaluate.py

# Or run everything at once:
python src/train.py --stage all --evaluate
```

### Train Individual Components

```bash
# Train only return model
python src/train.py --stage returns

# Train only volatility model
python src/train.py --stage volatility

# Train only meta-labeling classifier
python src/train.py --stage meta
```

---

## 📁 Project Structure

```
build/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── train.csv            # Raw training data (user provided)
│   └── processed/           # Processed CV folds (auto-generated)
├── models/                   # Trained models (auto-generated)
│   ├── returns/             # Return prediction models
│   ├── volatility/          # Volatility prediction models
│   └── meta/                # Meta-labeling classifiers
├── notebooks/               # Exploratory analysis
│   └── eda.ipynb           # Exploratory data analysis
└── src/                     # Source code
    ├── config.py           # Configuration and constants
    ├── preprocessing.py    # Data preprocessing and CV
    ├── features.py         # Feature engineering
    ├── returns.py          # Return prediction model
    ├── volatility.py       # Volatility prediction model
    ├── meta_labeling.py    # Meta-labeling framework
    ├── allocation.py       # Position allocation
    ├── train.py            # Training orchestration
    └── evaluate.py         # Model evaluation
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Purged K-Fold Cross-Validation**:
- 5 expanding window folds
- Purges samples with overlapping labels
- 1% embargo period to prevent serial correlation leakage

**Preprocessing Steps**:
- Drop features with >10% missing values
- Impute remaining missing values with training fold medians
- Winsorize outliers using Median Absolute Deviation (MAD)
- Create availability masks for features with late start dates

### 2. Feature Engineering

**Temporal Features**:
- Rolling statistics (mean, std, min, max) over multiple windows (5, 21, 63 days)
- Momentum indicators
- Lag features

**Volatility Features**:
- Historical volatility (multiple windows)
- EWMA volatility
- Volatility-of-volatility
- Range-based volatility

**Regime Features**:
- Volatility regime classification (low, medium, high)
- Regime interaction features
- Regime transition indicators

**Feature Selection**:
- Correlation clustering to remove redundant features
- PCA for dimensionality reduction per cluster
- Importance-based selection from initial model training

### 3. Return Prediction

**Model**: LightGBM Regressor

**Features**:
- ~150 engineered features (selected from 2000+ candidates)
- Volatility regime indicators
- Technical and statistical features

**Training**:
- Trained on each CV fold independently
- Early stopping to prevent overfitting
- Feature importance analysis for interpretability

### 4. Volatility Prediction

**Model**: LightGBM Regressor

**Target**: Log-variance of residuals from return predictions

**Features**:
- Historical volatility measures
- Squared returns (GARCH-like features)
- Volatility trends and percentiles
- Recent extreme returns

**Calibration**:
- Bias correction
- Clipping to training distribution percentiles
- EWMA smoothing for stability

### 5. Meta-Labeling

**Concept**: Separate prediction accuracy from position sizing

**Primary Model**: Predicts return direction and magnitude

**Meta-Model**: Predicts whether primary model is correct

**Meta-Labels**: Binary labels indicating if primary prediction has correct sign

**Meta-Classifier**: LightGBM classifier trained on:
- Market features
- Primary model predictions
- Regime indicators

**Position Sizing**: Scale allocations by meta-probability:
```
final_allocation = base_allocation × meta_probability
```

**Benefits**:
- Reduces positions when model is uncertain
- Increases positions when model is confident
- Improves risk-adjusted returns significantly

### 6. Position Allocation

**Method**: Regime-Dependent Kelly Criterion

**Kelly Formula**: 
```
allocation = k × tanh(μ / (σ² × s))
```

Where:
- μ = predicted return
- σ = predicted volatility
- k = Kelly fraction (regime-dependent)
- s = scaling parameter

**Regime-Specific Parameters**:
- Low volatility: k = 0.98 (aggressive)
- Medium volatility: k = 0.71 (moderate)
- High volatility: k = 0.50 (conservative)

**Constraints**: Allocations clipped to [0, 2] (0% to 200% of capital)

---

## 📈 Performance Metrics

### Cross-Validation Results (5-Fold Purged CV)

| Metric | Value |
|--------|-------|
Numerical performance results are recorded by the evaluation pipeline and exported to `evaluation_results.csv`. The public README omits explicit performance figures; consult the evaluation output for full numeric detail.

### Performance by Fold

Per-fold numerical results are produced by the evaluation pipeline and saved to `evaluation_results.csv`. See that output for dataset-specific fold metrics.
---

## 🛠️ Technical Details

### Dependencies

Core libraries:
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `lightgbm`: Gradient boosting models
- `matplotlib`, `seaborn`: Visualization

See `requirements.txt` for complete list.

### Model Persistence

Trained models are saved in `models/` directory:
- Return models: `models/returns/fold_{1-5}.pkl`
- Volatility models: `models/volatility/fold_{1-5}.pkl`
- Meta-classifiers: `models/meta/fold_{1-5}.pkl`

### Configuration

All hyperparameters and settings are in `src/config.py`:
- Feature lists
- Winsorization thresholds
- Model hyperparameters
- CV parameters
- Allocation parameters

---

## 📚 Theoretical Foundation

This implementation is based on established financial machine learning techniques and best practices, including purged cross-validation and meta-labeling for position sizing. Additional methodological extensions are tracked in the project issue tracker and will be validated before being reported.

---

## 🔄 Model Retraining

The pipeline handles circular dependencies between models:

**Dependency Chain**:
1. Return model (no dependencies)
2. Volatility model (depends on return predictions)
3. Meta-labeling (depends on return predictions and actual returns)

**Retraining Options**:

```bash
# Retrain everything (handles dependencies automatically)
python src/train.py --stage all

# Retrain returns only
python src/train.py --stage returns

# Retrain volatility (will use existing return predictions)
python src/train.py --stage volatility

# Retrain meta-labeling (will use existing predictions)
python src/train.py --stage meta
```

**Note**: If you retrain the return model, you should also retrain volatility and meta-labeling to maintain consistency.

---

## Enhancements

Potential enhancements are tracked in the project issue tracker. The README intentionally omits speculative numeric impact estimates; implementation and empirical validation are required before reporting measurable changes.

---

## 📊 Evaluation Metrics

The pipeline evaluates models using multiple metrics:

**Return Prediction**:
- R² score
- RMSE
- Directional accuracy

**Volatility Prediction**:
- QLIKE (Quasi-Likelihood)
- Correlation with absolute returns
- Calibration slope

**Meta-Labeling**:
- ROC AUC
- Precision, Recall, F1
- Lift (return improvement when meta-label = 1)

**Overall Strategy**:
- Sharpe Ratio (primary metric)
- Maximum drawdown
- Win rate
- Profit factor

---

## 💡 Usage Tips

### Memory Optimization
- The feature engineering creates 2000+ features
- Feature selection reduces this to ~150 most important
- Adjust `max_features` in `config.py` to balance performance vs memory

### Hyperparameter Tuning
- Default parameters are optimized for this dataset
- Modify `LGBM_PARAMS` in `config.py` for experimentation
- Use `optuna` for systematic hyperparameter search (see training script)

### Cross-Validation
- 5 folds balance computation time and robustness
- Increase `n_splits` in `config.py` for more robust estimates
- Decrease for faster iteration during development

---

## 📝 References

See the project's bibliography or issue tracker for methodology references and suggested citations.

---

## 📞 Support

For questions or issues:
1. Check the code documentation in each module
2. Review the methodology section above
3. Examine the EDA notebook for data insights

---

## ⚖️ License

This project is for educational and research purposes.

---

**Last Updated**: October 2025  
**Status**: Production Ready ✅
