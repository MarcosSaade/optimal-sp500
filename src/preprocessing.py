"""
Data Preprocessing and Cross-Validation

Implements:
1. Purged K-Fold cross-validation (prevents label leakage)
2. Data preprocessing (imputation, winsorization)
3. Feature availability masks

Based on "Advances in Financial Machine Learning" Chapter 7.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle

from .config import *


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time series with overlapping labels.

    Prevents label leakage by purging training samples whose prediction
    horizons overlap with the validation period.

    For 1-day forward returns:
    - Label at time t depends on price at t+1
    - Purge training sample at t-1 from validation starting at t
    - Add embargo period at end of validation to prevent serial correlation

    Parameters
    ----------
    n_splits : int
        Number of validation folds
    min_train_pct : float
        Minimum proportion of data for initial training (e.g., 0.3 = 30%)
    purge_gap : int
        Number of samples to purge before validation (1 for 1-day forward returns)
    embargo_pct : float
        Proportion of validation samples to embargo at end (e.g., 0.01 = 1%)
    verbose : bool
        Print detailed split information
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.3,
        purge_gap: int = 1,
        embargo_pct: float = 0.01,
        verbose: bool = True,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 < min_train_pct < 1:
            raise ValueError("min_train_pct must be between 0 and 1")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be between 0 and 0.5")

        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.verbose = verbose

        self.splits_ = []
        self.purge_stats_ = []

    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged expanding window splits.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe with temporal ordering

        Returns
        -------
        splits : List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, val_indices) for each fold
        """
        n = len(data)

        # Calculate fold boundaries
        min_train_size = int(n * self.min_train_pct)
        remaining = n - min_train_size
        fold_size = remaining // self.n_splits

        if self.verbose:
            print("=" * 80)
            print("PURGED K-FOLD CROSS-VALIDATION")
            print("=" * 80)
            print(f"Total samples: {n:,}")
            print(f"Number of folds: {self.n_splits}")
            print(f"Min train size: {min_train_size:,} ({self.min_train_pct:.1%})")
            print(f"Fold size: ~{fold_size:,}")
            print(f"Purge gap: {self.purge_gap} samples")
            print(f"Embargo: {self.embargo_pct:.1%} of validation")
            print()

        # Generate splits
        self.splits_ = []
        self.purge_stats_ = []

        for i in range(self.n_splits):
            # Validation boundaries
            val_start = min_train_size + (i * fold_size)
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                continue

            # Calculate embargo
            val_size = val_end - val_start
            embargo_size = int(val_size * self.embargo_pct)

            # Apply purging: remove samples overlapping with validation
            train_end = val_start - self.purge_gap

            if train_end <= 0:
                continue

            # Generate indices
            train_indices = np.arange(0, train_end)

            if embargo_size > 0:
                val_indices = np.arange(val_start, val_end - embargo_size)
            else:
                val_indices = np.arange(val_start, val_end)

            if len(val_indices) == 0:
                continue

            # Store split
            self.splits_.append((train_indices, val_indices))

            # Statistics
            purged_samples = self.purge_gap
            embargoed_samples = embargo_size

            stats = {
                "fold": i + 1,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "purged_samples": purged_samples,
                "embargoed_samples": embargoed_samples,
            }
            self.purge_stats_.append(stats)

            if self.verbose:
                print(f"Fold {i + 1}:")
                print(f"  Train: {len(train_indices):,} samples (0 to {train_end - 1})")
                print(
                    f"  Val:   {len(val_indices):,} samples ({val_start} to {val_indices[-1]})"
                )
                print(f"  Purged: {purged_samples} samples")
                if embargoed_samples > 0:
                    print(f"  Embargoed: {embargoed_samples} samples")
                print()

        if self.verbose:
            print(f"Generated {len(self.splits_)} valid folds\n")
            print("=" * 80)

        return self.splits_

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return len(self.splits_)


class MarketDataPreprocessor:
    """
    Preprocessing pipeline with leak-safe operations.

    Operations:
    1. Create availability masks for features with late start dates
    2. Impute missing values with training fold medians
    3. Winsorize outliers using Median Absolute Deviation (MAD)

    All parameters are fitted on training fold only.
    """

    def __init__(
        self,
        continuous_features: List[str],
        winsorize_config: Dict[str, float],
        late_starter_features: List[str],
        random_seed: int = RANDOM_SEED,
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        continuous_features : List[str]
            Names of continuous features to preprocess
        winsorize_config : Dict[str, float]
            Mapping of feature -> MAD multiplier for winsorization
        late_starter_features : List[str]
            Features that start late (need availability masks)
        random_seed : int
            Random seed for reproducibility
        """
        self.continuous_features = continuous_features
        self.winsorize_config = winsorize_config
        self.late_starter_features = late_starter_features
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # Fitted parameters (from training fold only)
        self.train_medians: Dict[str, float] = {}
        self.winsorize_bounds: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def _create_availability_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary availability masks for late-starter features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        masks : pd.DataFrame
            Binary availability indicators (1 = available, 0 = not available)
        """
        masks = pd.DataFrame(index=df.index)

        for feature in self.late_starter_features:
            if feature in df.columns:
                masks[f"available_{feature}"] = (~df[feature].isna()).astype(int)

        return masks

    def fit(self, train_data: pd.DataFrame) -> "MarketDataPreprocessor":
        """
        Fit preprocessing parameters on training fold.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training fold data

        Returns
        -------
        self : MarketDataPreprocessor
            Fitted preprocessor
        """
        # Calculate medians for imputation
        for feature in self.continuous_features:
            if feature in train_data.columns:
                median_val = train_data[feature].median()
                if pd.isna(median_val):
                    median_val = 0.0
                self.train_medians[feature] = float(median_val)

        # Impute training data for winsorization calculation
        train_imputed = train_data.copy()
        for feature in self.continuous_features:
            if feature in train_imputed.columns:
                train_imputed[feature] = train_imputed[feature].fillna(
                    self.train_medians[feature]
                )

        # Calculate winsorization bounds
        for feature, mad_mult in self.winsorize_config.items():
            if feature in self.continuous_features and feature in train_imputed.columns:
                median = float(train_imputed[feature].median())
                mad = float((train_imputed[feature] - median).abs().median())

                self.winsorize_bounds[feature] = {
                    "lower": median - mad_mult * mad,
                    "upper": median + mad_mult * mad,
                }

        self.is_fitted = True
        return self

    def transform(
        self,
        data: pd.DataFrame,
        create_masks: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply fitted transformations.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform
        create_masks : bool
            Whether to create availability masks

        Returns
        -------
        transformed_data : pd.DataFrame
            Preprocessed data
        masks : pd.DataFrame
            Availability masks
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Create availability masks BEFORE imputation
        masks = (
            self._create_availability_masks(data)
            if create_masks
            else pd.DataFrame(index=data.index)
        )

        # Work on copy
        result = data.copy()

        # Impute missing values
        for feature in self.continuous_features:
            if feature in result.columns:
                result[feature] = result[feature].fillna(self.train_medians[feature])

        # Winsorize outliers
        for feature, bounds in self.winsorize_bounds.items():
            if feature in result.columns:
                result[feature] = result[feature].clip(
                    lower=bounds["lower"], upper=bounds["upper"]
                )

        return result, masks

    def fit_transform(
        self,
        train_data: pd.DataFrame,
        create_masks: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit and transform training data.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        create_masks : bool
            Whether to create availability masks

        Returns
        -------
        transformed_data : pd.DataFrame
            Preprocessed training data
        masks : pd.DataFrame
            Availability masks
        """
        self.fit(train_data)
        return self.transform(train_data, create_masks=create_masks)

    def save(self, filepath: Path):
        """Save fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "MarketDataPreprocessor":
        """Load fitted preprocessor."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def find_trim_point(
    data: pd.DataFrame,
    feature_columns: List[str],
    max_missing_pct: float = MAX_MISSING_PCT,
) -> int:
    """
    Find first index where aggregate missingness < threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    feature_columns : List[str]
        Feature columns to check
    max_missing_pct : float
        Maximum allowed missing percentage

    Returns
    -------
    trim_index : int
        Index to trim from
    """
    missing_by_row = data[feature_columns].isna().mean(axis=1) * 100
    valid_rows = missing_by_row <= max_missing_pct

    if valid_rows.any():
        return valid_rows.idxmax()
    return 0


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """
    Load raw data and perform initial preparation.

    Parameters
    ----------
    data_path : Path
        Path to raw CSV file

    Returns
    -------
    data : pd.DataFrame
        Prepared data
    """
    # Load data
    data = pd.read_csv(data_path)

    # Get feature columns
    special_cols = ["date_id", "risk_free_rate", TARGET_COL]
    feature_cols = [col for col in data.columns if col not in special_cols]

    # Remove features to drop
    feature_cols = [col for col in feature_cols if col not in FEATURES_TO_DROP]

    # Find and apply trim point
    trim_idx = find_trim_point(data, feature_cols)
    data = data.iloc[trim_idx:].reset_index(drop=True)

    print(f"Trimmed to {len(data):,} rows (removed first {trim_idx:,})")

    # Verify temporal order
    assert data["date_id"].is_monotonic_increasing, "date_id must be monotonic"

    return data
