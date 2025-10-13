"""
Feature Engineering

Creates temporal, volatility, and regime features for market prediction.
All operations are leak-safe (no future information used).
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.decomposition import PCA
import warnings

from .config import *

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Feature engineering with temporal dynamics, volatility features, and regime indicators.

    Features created:
    1. Temporal: rolling statistics, lags, momentum
    2. Volatility: historical vol, EWMA vol, vol-of-vol
    3. Regimes: volatility-based regimes and interactions
    4. PCA: dimensionality reduction per feature cluster
    """

    def __init__(
        self,
        continuous_features: List[str],
        temporal_windows: List[int] = TEMPORAL_WINDOWS,
        extended_windows: List[int] = EXTENDED_WINDOWS,
        pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        continuous_features : List[str]
            Base continuous feature names
        temporal_windows : List[int]
            Rolling window sizes (e.g., [5, 21, 63] = 1 week, 1 month, 3 months)
        extended_windows : List[int]
            Extended window sizes for multi-scale analysis
        pca_variance_threshold : float
            Variance to retain in PCA
        correlation_threshold : float
            Threshold for correlation clustering
        random_seed : int
            Random seed for reproducibility
        """
        self.continuous_features = continuous_features
        self.temporal_windows = temporal_windows
        self.extended_windows = extended_windows
        self.pca_variance_threshold = pca_variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_seed = random_seed

        # Fitted components
        self.feature_clusters: Dict[int, List[str]] = {}
        self.pca_models: Dict[int, PCA] = {}
        self.vol_regime_thresholds = {}
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

        df_out = df.copy()

        # Phase 1: Temporal dynamics
        print("⏱  Phase 1: Temporal dynamics...")
        df_out = self._add_temporal_features(df_out)

        # Phase 2: Volatility features
        print("📊 Phase 2: Volatility features...")
        df_out = self._add_volatility_features(df_out)

        # Phase 3: Regime features
        print("🌤  Phase 3: Regime features...")
        df_out = self._add_regime_features(df_out)

        # Phase 4: PCA features
        print("🔬 Phase 4: PCA features...")
        df_out = self._add_pca_features(df_out)

        # Phase 5: Interaction features
        print("🔗 Phase 5: Interaction features...")
        df_out = self._add_interaction_features(df_out)

        print(f"✅ Feature engineering complete: {df_out.shape[1]} features")
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

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
    # Phase 5: Interaction Features
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
