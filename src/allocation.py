"""
Position Allocation

Fractional Kelly sizing around a neutral position:

    allocation = neutral + k * tanh(mu / (sigma^2 * s)) * m

Where:
    - mu = predicted excess return
    - sigma = predicted volatility (mu / sigma^2 is the Kelly fraction)
    - k = tilt size, optionally different per volatility regime
    - s = tanh soft-cap scale
    - m = meta-labeling multiplier (1 without meta-labeling)

The result is clipped to [ALLOCATION_MIN, ALLOCATION_MAX]. With k = 0 the
allocation is constant, i.e. buy-and-hold.
"""

import itertools
import numpy as np
from pathlib import Path
import pickle

from .config import *
from .metrics import sharpe_ratio


class RegimeDependentAllocator:
    """
    Allocator with (optionally) regime-dependent Kelly tilts.

    Regimes come from trailing realized volatility (0=low, 1=medium, 2=high).
    k and s are chosen by grid search on out-of-sample predictions from
    inside the training window, maximizing the Sharpe ratio.
    """

    def __init__(
        self,
        regime_dependent: bool = True,
        neutral: float = ALLOCATION_NEUTRAL,
        k: float = KELLY_K,
        scale: float = KELLY_SCALE,
        epsilon: float = EPSILON,
    ):
        """
        Initialize allocator.

        Parameters
        ----------
        regime_dependent : bool
            Use a separate tilt size per volatility regime
        neutral : float
            Position when the model has no view
        k : float
            Initial tilt size (all regimes)
        scale : float
            Initial tanh soft-cap scale
        epsilon : float
            Small constant to prevent division by zero
        """
        self.regime_dependent = regime_dependent
        self.neutral = neutral
        self.k = np.array([k, k, k])  # low, medium, high
        self.scale = scale
        self.epsilon = epsilon

    def _allocate(self, kelly, regime, multiplier, k, scale):
        tilt = k[regime] * np.tanh(kelly / scale) * multiplier
        return np.clip(self.neutral + tilt, ALLOCATION_MIN, ALLOCATION_MAX)

    def fit(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        regime: np.ndarray,
        excess_returns: np.ndarray,
    ) -> "RegimeDependentAllocator":
        """
        Choose tilt sizes and scale by grid search.

        Parameters
        ----------
        mu, sigma : np.ndarray
            Out-of-sample predictions
        regime : np.ndarray
            Volatility regime per row
        excess_returns : np.ndarray
            Realized excess returns for the same rows
        """
        kelly = mu / (sigma**2 + self.epsilon)
        multiplier = np.ones(len(mu))

        if self.regime_dependent:
            k_grid = itertools.product(ALLOCATION_K_GRID, repeat=3)
        else:
            k_grid = ((k, k, k) for k in ALLOCATION_K_GRID)

        best = (-np.inf, self.k, self.scale)
        for k in k_grid:
            k = np.array(k)
            for scale in ALLOCATION_SCALE_GRID:
                alloc = self._allocate(kelly, regime, multiplier, k, scale)
                score = sharpe_ratio(alloc * excess_returns)
                if score > best[0]:
                    best = (score, k, scale)

        _, self.k, self.scale = best
        return self

    def allocate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        regime: np.ndarray,
        multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """
        Allocate positions.

        Parameters
        ----------
        mu : np.ndarray
            Predicted returns
        sigma : np.ndarray
            Predicted volatility
        regime : np.ndarray
            Regime labels (0=low, 1=medium, 2=high), from past data only
        multiplier : np.ndarray, optional
            Meta-labeling tilt multiplier

        Returns
        -------
        allocations : np.ndarray
            Position sizes clipped to [ALLOCATION_MIN, ALLOCATION_MAX]
        """
        if multiplier is None:
            multiplier = np.ones(len(mu))
        kelly = mu / (sigma**2 + self.epsilon)
        return self._allocate(kelly, regime, multiplier, self.k, self.scale)

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
