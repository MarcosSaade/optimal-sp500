"""
Position Allocation

Regime-dependent Kelly criterion for position sizing.

Adapts position sizing to volatility regimes:
- Low volatility: Aggressive (larger positions)
- Medium volatility: Moderate
- High volatility: Conservative (smaller positions)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from .config import *


class RegimeDependentAllocator:
    """
    Allocator with regime-dependent Kelly fractions.

    Uses volatility regimes to adapt position sizing:
    - Low volatility (stable markets): Aggressive kelly fraction
    - Medium volatility (normal): Moderate kelly fraction
    - High volatility (chaotic): Conservative kelly fraction

    Allocation formula:
        allocation = k * tanh(μ / (σ² * s))

    Where:
        - μ = predicted return
        - σ = predicted volatility
        - k = Kelly fraction (regime-dependent)
        - s = scaling parameter
    """

    def __init__(
        self,
        k_low: float = KELLY_K_LOW,
        k_medium: float = KELLY_K_MEDIUM,
        k_high: float = KELLY_K_HIGH,
        scale: float = KELLY_SCALE,
        epsilon: float = EPSILON,
    ):
        """
        Initialize regime-dependent allocator.

        Parameters
        ----------
        k_low : float
            Kelly fraction for low volatility regime
        k_medium : float
            Kelly fraction for medium volatility regime
        k_high : float
            Kelly fraction for high volatility regime
        scale : float
            Scaling parameter for tanh soft-capping
        epsilon : float
            Small constant to prevent division by zero
        """
        self.k_low = k_low
        self.k_medium = k_medium
        self.k_high = k_high
        self.scale = scale
        self.epsilon = epsilon

    def _get_regime_from_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract volatility regime from data.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing regime information

        Returns
        -------
        regime : np.ndarray
            Array of regime labels (0=low, 1=medium, 2=high)
        """
        # Check for pre-computed regime features
        if "vol_regime" in data.columns:
            return data["vol_regime"].values

        # Fallback: compute simple regime based on rolling volatility
        if TARGET_COL in data.columns:
            returns = data[TARGET_COL].values
            rolling_vol = pd.Series(returns).rolling(21).std().values

            # Terciles: low, medium, high volatility
            q33 = np.nanpercentile(rolling_vol, 33)
            q66 = np.nanpercentile(rolling_vol, 66)

            regime = np.where(rolling_vol <= q33, 0, np.where(rolling_vol <= q66, 1, 2))
            return regime

        # Default to medium regime if no information available
        return np.ones(len(data), dtype=int)

    def allocate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        regime: np.ndarray = None,
        data: pd.DataFrame = None,
    ) -> np.ndarray:
        """
        Allocate positions based on regime.

        Parameters
        ----------
        mu : np.ndarray
            Predicted returns
        sigma : np.ndarray
            Predicted volatility
        regime : np.ndarray, optional
            Regime labels (0=low, 1=medium, 2=high)
            If None, will try to extract from data
        data : pd.DataFrame, optional
            Data for regime extraction

        Returns
        -------
        allocations : np.ndarray
            Position sizes clipped to [ALLOCATION_MIN, ALLOCATION_MAX]
        """
        # Get regime if not provided
        if regime is None:
            if data is not None:
                regime = self._get_regime_from_data(data)
            else:
                # Default to medium regime
                regime = np.ones(len(mu), dtype=int)

        # Ensure arrays are same length
        min_len = min(len(mu), len(sigma), len(regime))
        mu = mu[:min_len]
        sigma = sigma[:min_len]
        regime = regime[:min_len]

        # Select Kelly fraction based on regime (vectorized)
        k = np.where(
            regime == 0, self.k_low, np.where(regime == 1, self.k_medium, self.k_high)
        )

        # Kelly allocation: k * (μ / σ²)
        raw_allocation = mu / (sigma**2 + self.epsilon)

        # Soft cap with tanh
        allocation = k * np.tanh(raw_allocation / self.scale)

        # Shift from [-k, k] to approximately [0, 2k]
        k_avg = (self.k_low + self.k_medium + self.k_high) / 3
        allocation = allocation + k_avg

        # Clip to allowed range
        allocation = np.clip(allocation, ALLOCATION_MIN, ALLOCATION_MAX)

        return allocation

    def save(self, filepath: Path):
        """Save allocator configuration."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "RegimeDependentAllocator":
        """Load allocator configuration."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class SimpleAllocator:
    """
    Simple fixed Kelly fraction allocator (for comparison).

    Simpler alternative that doesn't adapt to regimes.
    """

    def __init__(
        self,
        k: float = 0.7,
        scale: float = KELLY_SCALE,
        epsilon: float = EPSILON,
    ):
        """
        Initialize simple allocator.

        Parameters
        ----------
        k : float
            Fixed Kelly fraction
        scale : float
            Scaling parameter
        epsilon : float
            Small constant to prevent division by zero
        """
        self.k = k
        self.scale = scale
        self.epsilon = epsilon

    def allocate(self, mu: np.ndarray, sigma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Allocate positions with fixed Kelly fraction.

        Parameters
        ----------
        mu : np.ndarray
            Predicted returns
        sigma : np.ndarray
            Predicted volatility
        **kwargs
            Ignored (for compatibility)

        Returns
        -------
        allocations : np.ndarray
            Position sizes
        """
        # Kelly allocation
        raw_allocation = mu / (sigma**2 + self.epsilon)

        # Soft cap with tanh
        allocation = self.k * np.tanh(raw_allocation / self.scale)

        # Shift and clip
        allocation = allocation + self.k
        allocation = np.clip(allocation, ALLOCATION_MIN, ALLOCATION_MAX)

        return allocation
