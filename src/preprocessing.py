"""
Data Preprocessing and Cross-Validation

Implements:
1. Purged walk-forward cross-validation (prevents label leakage)
2. Data preprocessing (imputation, winsorization)
3. Feature availability masks
4. Fold construction, including the held-out lockbox
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import pickle

from .config import *


class PurgedKFold:
    """
    Purged walk-forward (expanding window) cross-validation.

    Prevents label leakage by purging training samples whose prediction
    horizons overlap with the validation period.

    For 1-day forward returns:
    - Label at time t depends on price at t+1
    - Purge training sample at t-1 from validation starting at t

    Training data always precedes the validation window, so no embargo
    is needed after it.

    Parameters
    ----------
    n_splits : int
        Number of validation folds
    min_train_pct : float
        Minimum proportion of data for initial training (e.g., 0.3 = 30%)
    purge_gap : int
        Number of samples to purge before validation (1 for 1-day forward returns)
    verbose : bool
        Print detailed split information
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.3,
        purge_gap: int = 1,
        verbose: bool = True,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 < min_train_pct < 1:
            raise ValueError("min_train_pct must be between 0 and 1")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")

        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.purge_gap = purge_gap
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

            # Apply purging: remove samples overlapping with validation
            train_end = val_start - self.purge_gap

            if train_end <= 0:
                continue

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            self.splits_.append((train_indices, val_indices))
            self.purge_stats_.append(
                {
                    "fold": i + 1,
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "purged_samples": self.purge_gap,
                }
            )

            if self.verbose:
                print(f"Fold {i + 1}:")
                print(f"  Train: {len(train_indices):,} samples (0 to {train_end - 1})")
                print(
                    f"  Val:   {len(val_indices):,} samples ({val_start} to {val_end - 1})"
                )
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

    feature_cols = get_feature_columns(data)

    # Drop the features excluded in config
    data = data.drop(columns=[c for c in FEATURES_TO_DROP if c in data.columns])

    # Find and apply trim point
    trim_idx = find_trim_point(data, feature_cols)
    data = data.iloc[trim_idx:].reset_index(drop=True)

    print(f"Trimmed to {len(data):,} rows (removed first {trim_idx:,})")

    # Verify temporal order
    assert data["date_id"].is_monotonic_increasing, "date_id must be monotonic"

    return data


def get_feature_columns(data: pd.DataFrame) -> List[str]:
    """Raw input feature columns (everything except labels and bookkeeping)."""
    return [
        col
        for col in data.columns
        if col not in SPECIAL_COLS and col not in FEATURES_TO_DROP
    ]


@dataclass
class Fold:
    """
    One evaluation window with its training window.

    frame holds rows 0 .. eval_end - 1 of the data, preprocessed with
    parameters fitted on the training rows only. Keeping the rows contiguous
    lets rolling features in the evaluation window see their real history.
    """

    name: str
    frame: pd.DataFrame
    train_idx: np.ndarray  # Positions in frame used for fitting
    eval_idx: np.ndarray  # Positions in frame that are scored


def _build_fold(
    data: pd.DataFrame,
    name: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> Fold:
    """Fit the preprocessor on train_idx and preprocess rows up to the eval end."""
    feature_cols = get_feature_columns(data)
    continuous_features = [f for f in feature_cols if not f.startswith("D")]

    preprocessor = MarketDataPreprocessor(
        continuous_features=continuous_features,
        winsorize_config=WINSORIZE_CONFIG,
        late_starter_features=LATE_STARTER_FEATURES,
    )
    preprocessor.fit(data.iloc[train_idx])

    end = eval_idx[-1] + 1 if len(eval_idx) else len(data)
    window = data.iloc[:end]
    processed, masks = preprocessor.transform(window)
    frame = pd.concat([processed, masks], axis=1).reset_index(drop=True)

    return Fold(name=name, frame=frame, train_idx=train_idx, eval_idx=eval_idx)


def make_folds(
    data: pd.DataFrame,
    lockbox: bool = False,
    verbose: bool = False,
) -> List[Fold]:
    """
    Build the evaluation folds.

    The last LOCKBOX_DAYS rows are never part of a development fold.

    Parameters
    ----------
    data : pd.DataFrame
        Output of load_and_prepare_data
    lockbox : bool or "all"
        If True, return only the lockbox fold: train on all development
        data, score the held-out rows. If "all", return one fold that trains
        on every row (for the deployable model). Otherwise return the
        walk-forward development folds.
    verbose : bool
        Print split information

    Returns
    -------
    folds : List[Fold]
    """
    n_dev = len(data) - LOCKBOX_DAYS

    if lockbox == "all":
        # Deployment: fit on every row, nothing to score
        return [_build_fold(data, "all", np.arange(len(data)), np.array([], int))]

    if lockbox:
        train_idx = np.arange(0, n_dev - CV_PURGE_GAP)
        eval_idx = np.arange(n_dev, len(data))
        return [_build_fold(data, "lockbox", train_idx, eval_idx)]

    cv = PurgedKFold(
        n_splits=CV_N_SPLITS,
        min_train_pct=CV_MIN_TRAIN_PCT,
        purge_gap=CV_PURGE_GAP,
        verbose=verbose,
    )
    splits = cv.split(data.iloc[:n_dev])

    return [
        _build_fold(data, str(i), train_idx, val_idx)
        for i, (train_idx, val_idx) in enumerate(splits, 1)
    ]
