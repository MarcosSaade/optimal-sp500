"""
Feature Engineering

Creates temporal, volatility, and regime features for market prediction.
All operations are leak-safe (no future information used).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import warnings
import pickle
from pathlib import Path
from scipy.special import comb

from .config import *

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Feature engineering with fractional differencing and polynomial interactions.

    Pipeline:
    1. Add temporal, volatility, and regime features
    2. Add fractional differencing features
    3. Select top-k features by LGBM importance
    4. Add polynomial/interaction features on top-k
    5. Select final top 150 features by LGBM importance
    """

    def __init__(
        self,
        continuous_features: List[str],
        temporal_windows: List[int] = TEMPORAL_WINDOWS,
        extended_windows: List[int] = EXTENDED_WINDOWS,
        pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        random_seed: int = RANDOM_SEED,
        frac_diff_d_values: List[float] = None,
        frac_diff_threshold: float = FRAC_DIFF_THRESHOLD,
        top_k_after_frac_diff: int = TOP_K_AFTER_FRAC_DIFF,
        poly_degree: int = POLY_DEGREE,
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        continuous_features : List[str]
            Base continuous feature names
        temporal_windows : List[int]
            Rolling window sizes
        extended_windows : List[int]
            Extended window sizes
        pca_variance_threshold : float
            Variance to retain in PCA
        correlation_threshold : float
            Threshold for correlation clustering
        random_seed : int
            Random seed for reproducibility
        frac_diff_d_values : List[float]
            List of fractional differencing orders to try
        frac_diff_threshold : float
            Minimum weight threshold for fractional differencing
        top_k_after_frac_diff : int
            Number of top features to select after fractional differencing
        poly_degree : int
            Polynomial degree for interactions
        """
        self.continuous_features = continuous_features
        # Safety: ensure target-like columns aren't used as inputs (prevent leakage)
        forbidden = {TARGET_COL, "forward_returns"}
        removed = [c for c in self.continuous_features if c in forbidden]
        if removed:
            print(f"⚠️  Removed forbidden features from continuous_features to avoid leakage: {removed}")
        self.continuous_features = [c for c in self.continuous_features if c not in forbidden]
        self.temporal_windows = temporal_windows
        self.extended_windows = extended_windows
        self.pca_variance_threshold = pca_variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_seed = random_seed
        self.frac_diff_d_values = frac_diff_d_values if frac_diff_d_values is not None else [0.2, 0.3, 0.4, 0.5]
        self.frac_diff_threshold = frac_diff_threshold
        self.top_k_after_frac_diff = top_k_after_frac_diff
        self.poly_degree = poly_degree

        # Fitted components
        self.feature_clusters: Dict[int, List[str]] = {}
        self.pca_models: Dict[int, PCA] = {}
        self.vol_regime_thresholds = {}
        self.frac_diff_weights = {}
        self.top_k_features_after_fd = []
        self.poly_feature_names = []
        self.final_selected_features = []  # Final top 150 features
        self.fd_selected_features = {}  # Track which fd value was selected for each feature
        self.is_fitted = False

        np.random.seed(random_seed)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit feature engineering on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe

        Returns
        -------
        self : FeatureEngineer
            Fitted feature engineer
        """
        print("🔧 Fitting feature engineer...")

        # Calculate volatility regime thresholds from training data
        self._fit_volatility_regimes(df)

        # Fit PCA and clustering
        self._fit_cross_sectional(df)

        # Calculate fractional differencing weights
        self._fit_fractional_differencing()

        # Phase 1: Create base features (temporal, volatility, regime, PCA)
        print("⏱  Creating base features...")
        df_base = self._add_base_features(df)

        # Phase 2: Add fractional differencing features
        print("📈 Adding fractional differencing features...")
        df_with_fd = self._add_fractional_diff_features(df_base)

        # Phase 3: Select top-k features using LGBM
        print(f"🔍 Selecting top {self.top_k_after_frac_diff} features after fractional differencing...")
        self.top_k_features_after_fd = self._select_top_k_features(
            df_with_fd, self.top_k_after_frac_diff
        )
        print(f"   Selected {len(self.top_k_features_after_fd)} features")
        
        # Track which fd values were selected
        self._track_fd_selection()

        # Phase 4: Determine polynomial feature names
        print("🔗 Preparing polynomial/interaction features...")
        self._fit_polynomial_features()

        # Phase 5: Keep only top-k features before adding polynomials
        df_top_k = self._keep_top_k_features(df_with_fd)
        
        # Phase 6: Create full feature set with polynomials on top-k
        print("🏗  Creating full feature set with polynomials...")
        df_with_poly = self._add_polynomial_features(df_top_k)

        # Phase 7: Select final top 150 features
        print(f"🎯 Selecting final top {MAX_FEATURES} features...")
        self.final_selected_features = self._select_top_k_features(
            df_with_poly, MAX_FEATURES
        )
        print(f"   Selected {len(self.final_selected_features)} final features")
        
        # Track final fd selection
        self._track_final_fd_selection()

        self.is_fitted = True
        print("✅ Feature engineer fitted")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with all feature engineering.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        df_out : pd.DataFrame
            Dataframe with engineered features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        # Phase 1: Base features
        print("⏱  Phase 1: Base features...")
        df_out = self._add_base_features(df)

        # Phase 2: Fractional differencing
        print("� Phase 2: Fractional differencing...")
        df_out = self._add_fractional_diff_features(df_out)

        # Phase 3: Keep only top-k features + special columns
        print(f"🔍 Phase 3: Selecting top {self.top_k_after_frac_diff} features...")
        df_out = self._keep_top_k_features(df_out)

        # Phase 4: Add polynomial/interaction features
        print("🔗 Phase 4: Polynomial/interaction features...")
        df_out = self._add_polynomial_features(df_out)

        # Phase 5: Select final top features
        print(f"🎯 Phase 5: Selecting final top {MAX_FEATURES} features...")
        df_out = self._keep_final_features(df_out)

        print(f"✅ Feature engineering complete: {df_out.shape[1]} features")
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def save(self, path):
        """Save the fitted feature engineer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load a fitted feature engineer from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ============================================================================
    # Base Features (Temporal + Volatility + Regime + PCA)
    # ============================================================================

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all base features (temporal, volatility, regime, PCA)."""
        df_out = df.copy()
        
        # Add temporal features
        df_out = self._add_temporal_features(df_out)
        
        # Add volatility features
        df_out = self._add_volatility_features(df_out)
        
        # Add regime features
        df_out = self._add_regime_features(df_out)
        
        # Add PCA features
        df_out = self._add_pca_features(df_out)
        
        return df_out

    # ============================================================================
    # Phase 1: Temporal Features
    # ============================================================================

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics, lags, and momentum features."""
        df_out = df.copy()

        for feature in self.continuous_features:
            if feature not in df.columns or df[feature].isna().all():
                continue

            # Lag features
            df_out[f"{feature}_lag1"] = df[feature].shift(1)
            df_out[f"{feature}_lag5"] = df[feature].shift(5)

            # Rolling statistics
            for window in self.temporal_windows:
                # Rolling mean (trend)
                df_out[f"{feature}_ma{window}"] = (
                    df[feature].shift(1).rolling(window, min_periods=window // 2).mean()
                )

                # Rolling std (volatility)
                df_out[f"{feature}_std{window}"] = (
                    df[feature].shift(1).rolling(window, min_periods=window // 2).std()
                )

                # Rolling min/max
                if window <= 21:
                    df_out[f"{feature}_min{window}"] = (
                        df[feature]
                        .shift(1)
                        .rolling(window, min_periods=window // 2)
                        .min()
                    )
                    df_out[f"{feature}_max{window}"] = (
                        df[feature]
                        .shift(1)
                        .rolling(window, min_periods=window // 2)
                        .max()
                    )

            # EMA (exponential moving average)
            for span in [10, 20, 50]:
                df_out[f"{feature}_ema{span}"] = (
                    df[feature].ewm(span=span, adjust=False).mean().shift(1)
                )

            # Extended windows
            for window in self.extended_windows:
                df_out[f"{feature}_ma{window}"] = (
                    df[feature].shift(1).rolling(window, min_periods=window // 2).mean()
                )

        return df_out

    # ============================================================================
    # Phase 2: Volatility Features
    # ============================================================================

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-specific features."""
        df_out = df.copy()

        # Use excess returns as the main volatility proxy
        if TARGET_COL in df.columns:
            returns = df[TARGET_COL].shift(1)

            # Historical volatility (multiple windows)
            for window in [5, 10, 21, 42, 63, 126]:
                df_out[f"hist_vol_{window}"] = returns.rolling(window).std()

            # EWMA volatility
            for span in [10, 20, 50]:
                df_out[f"vol_ema_{span}"] = (
                    returns.rolling(21).std().ewm(span=span).mean()
                )

            # Volatility of volatility
            for window in [21, 63]:
                vol = returns.rolling(window).std()
                df_out[f"vol_of_vol_{window}"] = vol.rolling(window).std()

            # Absolute returns
            df_out["abs_return"] = np.abs(returns)
            for window in [5, 10, 21]:
                df_out[f"abs_return_ma_{window}"] = (
                    df_out["abs_return"].rolling(window).mean()
                )

            # Squared returns (GARCH-like)
            df_out["squared_return"] = returns**2
            for window in [5, 10, 21]:
                df_out[f"squared_return_ma_{window}"] = (
                    df_out["squared_return"].rolling(window).mean()
                )

            # Volatility trends
            short_vol = returns.rolling(21).std()
            long_vol = returns.rolling(63).std()
            df_out["vol_trend_21_63"] = short_vol / (long_vol + 1e-8) - 1

            # Maximum drawdown
            rolling_max = returns.shift(1).rolling(63).max()
            df_out["max_drawdown_63"] = rolling_max - returns

        return df_out

    # ============================================================================
    # Phase 3: Regime Features
    # ============================================================================

    def _fit_volatility_regimes(self, df: pd.DataFrame):
        """Fit volatility regime thresholds on training data."""
        if TARGET_COL not in df.columns:
            return

        returns = df[TARGET_COL].shift(1)
        vol_21 = returns.rolling(21).std()

        # Calculate percentile thresholds
        self.vol_regime_thresholds = {
            "low": np.nanpercentile(vol_21, VOL_REGIME_LOW),
            "high": np.nanpercentile(vol_21, VOL_REGIME_HIGH),
        }

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime indicators."""
        df_out = df.copy()

        if TARGET_COL not in df.columns:
            return df_out

        returns = df[TARGET_COL].shift(1)
        vol_21 = returns.rolling(21).std()

        # Create regime indicators
        df_out["regime_low_vol"] = (vol_21 <= self.vol_regime_thresholds["low"]).astype(
            int
        )
        df_out["regime_medium_vol"] = (
            (vol_21 > self.vol_regime_thresholds["low"])
            & (vol_21 <= self.vol_regime_thresholds["high"])
        ).astype(int)
        df_out["regime_high_vol"] = (
            vol_21 > self.vol_regime_thresholds["high"]
        ).astype(int)

        # Numeric regime (0=low, 1=medium, 2=high)
        regime = np.zeros(len(df))
        regime[vol_21 > self.vol_regime_thresholds["low"]] = 1
        regime[vol_21 > self.vol_regime_thresholds["high"]] = 2
        df_out["vol_regime"] = regime

        return df_out

    # ============================================================================
    # Phase 4: PCA Features
    # ============================================================================

    def _fit_cross_sectional(self, df: pd.DataFrame):
        """Fit PCA and correlation clustering on training data."""
        available_features = [f for f in self.continuous_features if f in df.columns]

        if len(available_features) < 3:
            return

        # Prepare data
        X = df[available_features].fillna(df[available_features].median())

        # Compute correlation matrix
        corr_matrix = X.corr()

        # Simple clustering by correlation
        cluster_labels = np.arange(len(available_features))
        current_cluster = 0

        for i in range(len(available_features)):
            if cluster_labels[i] == i:
                cluster_labels[i] = current_cluster
                for j in range(i + 1, len(available_features)):
                    if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                        cluster_labels[j] = current_cluster
                current_cluster += 1

        # Group features by cluster
        for i, feature in enumerate(available_features):
            cluster_id = cluster_labels[i]
            if cluster_id not in self.feature_clusters:
                self.feature_clusters[cluster_id] = []
            self.feature_clusters[cluster_id].append(feature)

        # Fit PCA for each cluster with 2+ features
        for cluster_id, features in self.feature_clusters.items():
            if len(features) < 2:
                continue

            X_cluster = df[features].fillna(df[features].median())

            pca = PCA(
                n_components=self.pca_variance_threshold, random_state=self.random_seed
            )
            pca.fit(X_cluster)

            self.pca_models[cluster_id] = pca

        print(f"  Found {len(self.feature_clusters)} feature clusters")
        print(f"  Created {len(self.pca_models)} PCA models")

    def _add_pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add PCA-transformed features."""
        df_out = df.copy()

        for cluster_id, pca in self.pca_models.items():
            features = self.feature_clusters[cluster_id]

            if not all(f in df.columns for f in features):
                continue

            X_cluster = df[features].fillna(df[features].median())
            X_pca = pca.transform(X_cluster)

            # Add PCA components
            for i in range(X_pca.shape[1]):
                df_out[f"pca_cluster{cluster_id}_comp{i}"] = X_pca[:, i]

        return df_out

    # ============================================================================
    # Fractional Differencing
    # ============================================================================

    def _fit_fractional_differencing(self):
        """Calculate fractional differencing weights for all d values."""
        # Calculate binomial weights for each fractional differencing order
        k_max = 100  # Maximum lag to consider
        
        for d in self.frac_diff_d_values:
            weights = [1.0]
            
            for k in range(1, k_max + 1):
                weight = -weights[-1] * (d - k + 1) / k
                if abs(weight) < self.frac_diff_threshold:
                    break
                weights.append(weight)
            
            self.frac_diff_weights[d] = np.array(weights)
            print(f"   Fractional diff weights for d={d}: {len(self.frac_diff_weights[d])} terms")

    def _add_fractional_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fractional differencing features with correlation-based selection."""
        df_out = df.copy()
        
        # Get all numeric columns except special columns
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        
        # Apply fractional differencing to selected continuous features
        # Create all fd features for all d values
        fd_features_by_base = {}  # Store all fd versions for each base feature
        
        for feature in self.continuous_features:
            if feature not in df.columns or df[feature].isna().all():
                continue
            
            series = df[feature].fillna(method='ffill').fillna(0).values
            fd_features_by_base[feature] = {}
            
            # Create fd features for all d values
            for d in self.frac_diff_d_values:
                fd_series = self._fractional_diff_series(series, d)
                col_name = f"{feature}_fd{d}"
                df_out[col_name] = fd_series
                fd_features_by_base[feature][d] = col_name
        
        # Perform correlation-based selection within each feature's fd versions
        # Keep the one with highest variance if correlation > 0.95
        features_to_drop = []
        
        for feature, fd_cols_dict in fd_features_by_base.items():
            if len(fd_cols_dict) <= 1:
                continue
                
            # Get the fd column names for this feature
            fd_cols = list(fd_cols_dict.values())
            
            # Calculate correlation matrix
            fd_data = df_out[fd_cols].dropna()
            if len(fd_data) < 10:  # Skip if too few valid rows
                continue
                
            corr_matrix = fd_data.corr()
            variances = fd_data.var()
            
            # Track which features to keep
            to_keep = set(fd_cols)
            
            # Check each pair for high correlation
            for i, col1 in enumerate(fd_cols):
                for j in range(i + 1, len(fd_cols)):
                    col2 = fd_cols[j]
                    
                    if col1 not in to_keep or col2 not in to_keep:
                        continue
                        
                    # If correlation > 0.95, drop the one with lower variance
                    if abs(corr_matrix.loc[col1, col2]) > 0.95:
                        if variances[col1] >= variances[col2]:
                            to_keep.discard(col2)
                        else:
                            to_keep.discard(col1)
            
            # Mark features not in to_keep for removal
            for col in fd_cols:
                if col not in to_keep:
                    features_to_drop.append(col)
        
        # Drop highly correlated fd features
        if features_to_drop:
            print(f"   Dropping {len(features_to_drop)} fd features due to high correlation (>0.95)")
            df_out = df_out.drop(columns=features_to_drop)
        
        return df_out

    def _fractional_diff_series(self, series: np.ndarray, d: float) -> np.ndarray:
        """Apply fractional differencing to a series with specified d value."""
        weights = self.frac_diff_weights[d]
        n = len(series)
        result = np.zeros(n)
        
        for i in range(len(weights), n):
            result[i] = np.dot(weights, series[i - len(weights) + 1:i + 1][::-1])
        
        # Set early values to NaN
        result[:len(weights)] = np.nan
        
        return result

    # ============================================================================
    # Top-K Feature Selection
    # ============================================================================

    def _track_fd_selection(self):
        """Track which fd values were selected for features and print statistics."""
        # Count how many features were selected for each fd value
        fd_counts = {d: 0 for d in self.frac_diff_d_values}
        
        for feature in self.top_k_features_after_fd:
            # Check if this is an fd feature
            for d in self.frac_diff_d_values:
                if f"_fd{d}" in feature:
                    fd_counts[d] += 1
                    break
        
        # Print statistics
        print("\n📊 Fractional Differencing Selection Statistics (Top-K):")
        print("=" * 60)
        total_fd_features = sum(fd_counts.values())
        for d in sorted(self.frac_diff_d_values):
            count = fd_counts[d]
            percentage = (count / total_fd_features * 100) if total_fd_features > 0 else 0
            print(f"   fd={d}: {count} features ({percentage:.1f}%)")
        print(f"   Total FD features: {total_fd_features}")
        print(f"   Non-FD features: {len(self.top_k_features_after_fd) - total_fd_features}")
        print("=" * 60 + "\n")

    def _track_final_fd_selection(self):
        """Track which fd values were selected in final features and print statistics."""
        # Count how many features were selected for each fd value
        fd_counts = {d: 0 for d in self.frac_diff_d_values}
        
        for feature in self.final_selected_features:
            # Check if this is an fd feature (could be base fd or polynomial with fd)
            for d in self.frac_diff_d_values:
                if f"_fd{d}" in feature:
                    fd_counts[d] += 1
                    break
        
        # Print statistics
        print("\n📊 Fractional Differencing Selection Statistics (Final):")
        print("=" * 60)
        total_fd_features = sum(fd_counts.values())
        for d in sorted(self.frac_diff_d_values):
            count = fd_counts[d]
            percentage = (count / total_fd_features * 100) if total_fd_features > 0 else 0
            print(f"   fd={d}: {count} features ({percentage:.1f}%)")
        print(f"   Total FD features: {total_fd_features}")
        print(f"   Non-FD features: {len(self.final_selected_features) - total_fd_features}")
        print("=" * 60 + "\n")

    def _select_top_k_features(self, df: pd.DataFrame, top_k: int) -> List[str]:
        """Select top-k features using LightGBM importance."""
        # Prepare data
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        feature_cols = [col for col in df.columns if col not in special_cols]
        
        # Remove high-NaN features
        nan_ratio = df[feature_cols].isna().mean()
        valid_features = nan_ratio[nan_ratio < 0.5].index.tolist()
        
        if len(valid_features) == 0:
            return []
        
        X = df[valid_features].fillna(0)
        y = df[TARGET_COL]
        
        # Remove NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(y) == 0:
            return []
        
        # Train a simple LightGBM model
        print(f"   Training LGBM on {len(valid_features)} features...")
        
        import lightgbm as lgb
        
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': self.random_seed,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(valid_features, importance))
        
        # Sort and select top-k
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in sorted_features[:top_k]]
        
        print(f"   Top 10 features: {selected[:10]}")
        
        return selected

    def _keep_top_k_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only top-k features selected during fit."""
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        cols_to_keep = special_cols + self.top_k_features_after_fd
        
        # Keep only columns that exist
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        
        return df[cols_to_keep]

    def _keep_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only final selected features."""
        special_cols = ["date_id", "risk_free_rate", TARGET_COL, "forward_returns"]
        cols_to_keep = special_cols + self.final_selected_features
        
        # Keep only columns that exist
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        
        return df[cols_to_keep]

    # ============================================================================
    # Polynomial/Interaction Features
    # ============================================================================

    def _fit_polynomial_features(self):
        """Determine polynomial feature names based on top-k features."""
        # Generate interaction pairs and squared terms
        self.poly_feature_names = []
        
        # Squared terms
        for feat in self.top_k_features_after_fd:
            self.poly_feature_names.append(f"{feat}_sq")
        
        # Pairwise interactions (limit to avoid explosion)
        # Only create interactions for top features
        max_interactions = min(20, len(self.top_k_features_after_fd))
        top_for_interactions = self.top_k_features_after_fd[:max_interactions]
        
        for i in range(len(top_for_interactions)):
            for j in range(i + 1, len(top_for_interactions)):
                feat1 = top_for_interactions[i]
                feat2 = top_for_interactions[j]
                self.poly_feature_names.append(f"{feat1}_x_{feat2}")
        
        print(f"   Will create {len(self.poly_feature_names)} polynomial features")

    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial and interaction features."""
        df_out = df.copy()
        
        # Add squared terms
        for feat in self.top_k_features_after_fd:
            if feat in df.columns:
                df_out[f"{feat}_sq"] = df[feat] ** 2
        
        # Add pairwise interactions (limit to avoid explosion)
        max_interactions = min(20, len(self.top_k_features_after_fd))
        top_for_interactions = self.top_k_features_after_fd[:max_interactions]
        
        for i in range(len(top_for_interactions)):
            for j in range(i + 1, len(top_for_interactions)):
                feat1 = top_for_interactions[i]
                feat2 = top_for_interactions[j]
                
                if feat1 in df.columns and feat2 in df.columns:
                    df_out[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
        
        return df_out

    # ============================================================================
    # Phase 5: Interaction Features (OLD - kept for compatibility)
    # ============================================================================

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime interaction features."""
        df_out = df.copy()

        # Check if we have regime and volatility features
        if "vol_regime" not in df_out.columns:
            return df_out

        # Interact key features with regime
        key_features = []

        # Find historical volatility features
        for col in df_out.columns:
            if col.startswith("hist_vol_21"):
                key_features.append(col)
                break

        # Create regime-conditional features
        for feature in key_features:
            if feature not in df_out.columns:
                continue

            for regime in [0, 1, 2]:
                mask = (df_out["vol_regime"] == regime).astype(int)
                df_out[f"{feature}_regime{regime}"] = df_out[feature] * mask

        return df_out


def select_features_by_importance(
    feature_importances: Dict[str, float],
    max_features: int = MAX_FEATURES,
) -> List[str]:
    """
    Select top features by importance.

    Parameters
    ----------
    feature_importances : Dict[str, float]
        Mapping of feature name to importance score
    max_features : int
        Maximum number of features to select

    Returns
    -------
    selected_features : List[str]
        List of selected feature names
    """
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )

    return [f for f, _ in sorted_features[:max_features]]
