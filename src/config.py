"""
Configuration and Constants

All hyperparameters, feature definitions, and pipeline settings.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "returns").mkdir(exist_ok=True)
(MODELS_DIR / "volatility").mkdir(exist_ok=True)
(MODELS_DIR / "meta").mkdir(exist_ok=True)

# ============================================================================
# Global Settings
# ============================================================================

RANDOM_SEED = 42
TARGET_COL = "market_forward_excess_returns"

# ============================================================================
# Data Preprocessing
# ============================================================================

# Features to exclude from model training
FEATURES_TO_DROP = ["E7", "I9"]  # E7: >70% missing, I9: duplicate of I5

# Features that start late in the dataset (need availability masks)
LATE_STARTER_FEATURES = [
    "V10",  # 51.5% available
    "S3",  # 57.1% available
    "M1",  # 60.3% available
    "M13",  # 60.5% available
    "M14",  # 60.5% available
    "M6",  # 69.2% available
    "V9",  # 78.0% available
    "S12",  # 95.5% available
]

# Winsorization thresholds (MAD multipliers) from EDA
WINSORIZE_CONFIG = {
    # Conservative (2.5x MAD): High outlier frequency
    "E11": 2.5,
    "E12": 2.5,
    "E13": 2.5,
    "E4": 2.5,
    "V12": 2.5,
    "P12": 2.5,
    "E6": 2.5,
    "S5": 2.5,
    "V8": 2.5,
    # Moderate (3.0x MAD): Medium outlier frequency
    "E14": 3.0,
    "E19": 3.0,
    "M3": 3.0,
    "M4": 3.0,
    "M6": 3.0,
    "M7": 3.0,
    "M8": 3.0,
    "M9": 3.0,
    "P5": 3.0,
    "P8": 3.0,
    "P10": 3.0,
    "P13": 3.0,
    "S8": 3.0,
    "V10": 3.0,
    "V13": 3.0,
    # Relaxed (4.0x MAD): Low outlier frequency
    "E1": 4.0,
    "E16": 4.0,
    "E17": 4.0,
    "E20": 4.0,
    "I1": 4.0,
    "I2": 4.0,
    "M1": 4.0,
    "M11": 4.0,
    "M13": 4.0,
    "M14": 4.0,
    "P2": 4.0,
    "P9": 4.0,
    "P11": 4.0,
    "S1": 4.0,
    "S12": 4.0,
    "V1": 4.0,
    "V6": 4.0,
    "V7": 4.0,
}

# NaN handling
MAX_MISSING_PCT = 10.0  # Drop rows with >10% missing values before training

# ============================================================================
# Cross-Validation
# ============================================================================

CV_N_SPLITS = 5
CV_MIN_TRAIN_PCT = 0.3  # Start with 30% of data in first fold
CV_PURGE_GAP = 1  # Purge 1 sample (1-day forward returns)
CV_EMBARGO_PCT = 0.01  # Embargo 1% of validation samples

# ============================================================================
# Feature Engineering
# ============================================================================

# Rolling window sizes for temporal features
TEMPORAL_WINDOWS = [5, 21, 63]  # ~1 week, 1 month, 3 months
EXTENDED_WINDOWS = [10, 42, 126]  # ~2 weeks, 2 months, 6 months

# Feature selection
PCA_VARIANCE_THRESHOLD = 0.85  # Retain 85% variance in PCA
CORRELATION_THRESHOLD = 0.95  # Cluster features with r > 0.95
MAX_FEATURES = 150  # Maximum features for final model

# Fractional differencing parameters
FRAC_DIFF_D = 0.5  # Fractional difference order (0.5 = semi-stationary)
FRAC_DIFF_THRESHOLD = 0.01  # Minimum weight threshold

# Feature selection stages
TOP_K_AFTER_FRAC_DIFF = 80  # Top K features to select after fractional differencing
POLY_DEGREE = 2  # Polynomial degree for interactions

# Volatility regime thresholds (percentiles)
VOL_REGIME_LOW = 33  # Below 33rd percentile = low volatility
VOL_REGIME_HIGH = 66  # Above 66th percentile = high volatility

# ============================================================================
# Return Prediction Model (LightGBM)
# ============================================================================

LGBM_RETURN_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 4,
}

LGBM_RETURN_ROUNDS = 500
LGBM_RETURN_EARLY_STOP = 50

# ============================================================================
# Volatility Prediction Model (LightGBM)
# ============================================================================

LGBM_VOL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 4,
}

LGBM_VOL_ROUNDS = 500
LGBM_VOL_EARLY_STOP = 50

# Volatility calibration parameters
VOL_BIAS_DELTA = 0.0  # Bias correction (0 = no correction)
VOL_CLIP_PERCENTILES = (10, 90)  # Clip to 10th-90th percentile of training
VOL_EWMA_LAMBDA = 0.9  # EWMA smoothing parameter

# ============================================================================
# Meta-Labeling
# ============================================================================

META_LABEL_METHOD = (
    "sign"  # How to generate meta-labels ("sign", "barrier", "symmetric")
)
META_MIN_CONFIDENCE = 0.0  # Minimum probability to trade (0 = always trade)

LGBM_META_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,  # Smaller tree for meta-labeling
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 4,  # Shallower than return model
    "min_data_in_leaf": 30,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 4,
    "is_unbalance": True,  # Handle class imbalance
}

LGBM_META_ROUNDS = 300
LGBM_META_EARLY_STOP = 30

# ============================================================================
# Position Allocation (Regime-Dependent Kelly)
# ============================================================================

# Kelly fractions by volatility regime (optimized values)
KELLY_K_LOW = 0.9854  # Low volatility: aggressive
KELLY_K_MEDIUM = 0.7116  # Medium volatility: moderate
KELLY_K_HIGH = 0.5001  # High volatility: conservative

# Kelly scaling parameter
KELLY_SCALE = 1.1543

# Allocation constraints
ALLOCATION_MIN = 0.0  # Minimum allocation (0%)
ALLOCATION_MAX = 2.0  # Maximum allocation (200%)

# Small constant to prevent division by zero
EPSILON = 1e-6

# ============================================================================
# Evaluation
# ============================================================================

# Minimum samples required for meta-labeling deployment
META_MIN_SAMPLES = 3000

# Sharpe ratio calculation
ANNUALIZATION_FACTOR = 252  # Trading days per year
