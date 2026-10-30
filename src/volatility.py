"""
Volatility Prediction Models

Forecast the conditional 1-day volatility of excess returns for position
sizing. All models share the same interface:

    fit(frame, train_idx, mu_train)   fit on the training rows of a fold frame
    predict(frame) -> np.ndarray       one forecast per row of the frame

The forecast for row t only uses returns realized before t (the label of row
t - 1 is known when row t is traded).

- VolatilityPredictor: LightGBM on log squared residuals of the return model
- EWMAVolatility: RiskMetrics exponentially weighted variance
- GARCHVolatility: GARCH(1,1), GJR-GARCH(1,1,1) or EGARCH(1,1,1), Student-t errors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle

from .boosting import train_booster
from .config import *


class _Persistable:
    """save/load via pickle."""

    def save(self, filepath: Path):
        """Save fitted model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path):
        """Load fitted model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class VolatilityPredictor(_Persistable):
    """
    Volatility prediction using LightGBM.

    Target: log-variance of residuals from return predictions.

    exp(E[log e^2]) underestimates the variance (by a factor of about 3.5 for
    Gaussian residuals), so predictions are rescaled to match the mean squared
    residual on the training rows, then clipped to the training range and
    EWMA-smoothed.
    """

    def __init__(
        self,
        params: Dict = None,
        num_boost_round: int = LGBM_VOL_ROUNDS,
        early_stopping_rounds: int = LGBM_VOL_EARLY_STOP,
        clip_percentiles: tuple = VOL_CLIP_PERCENTILES,
        ewma_lambda: float = VOL_EWMA_LAMBDA,
    ):
        """
        Initialize volatility predictor.

        Parameters
        ----------
        params : Dict, optional
            LightGBM parameters
        num_boost_round : int
            Maximum boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        clip_percentiles : tuple
            (low, high) percentiles for clipping
        ewma_lambda : float
            EWMA smoothing parameter
        """
        self.params = params if params is not None else LGBM_VOL_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.clip_percentiles = clip_percentiles
        self.ewma_lambda = ewma_lambda

        self.model = None
        self.feature_names = None
        self.variance_scale = None
        self.train_vol_range = None

    def _add_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features from past returns only."""
        out = pd.DataFrame(index=df.index)
        past_returns = df[TARGET_COL].shift(1)

        # Historical volatility
        for window in [5, 10, 21, 42, 63]:
            out[f"hist_vol_{window}"] = past_returns.rolling(window).std()

        # Absolute returns
        out["abs_return"] = np.abs(past_returns)
        for window in [5, 10, 21]:
            out[f"abs_return_ma_{window}"] = out["abs_return"].rolling(window).mean()

        # Squared returns
        out["squared_return"] = past_returns**2
        for window in [5, 10, 21]:
            out[f"squared_return_ma_{window}"] = (
                out["squared_return"].rolling(window).mean()
            )

        # EWMA volatility
        for span in [10, 20]:
            out[f"vol_ema_{span}"] = past_returns.rolling(21).std().ewm(span=span).mean()

        return out.fillna(0)

    def fit(
        self,
        frame: pd.DataFrame,
        train_idx: np.ndarray,
        mu_train: np.ndarray,
    ) -> "VolatilityPredictor":
        """
        Train on the training rows of a fold frame.

        Parameters
        ----------
        frame : pd.DataFrame
            Contiguous fold frame (training rows first)
        train_idx : np.ndarray
            Positions of the training rows
        mu_train : np.ndarray
            Return predictions for the training rows
        """
        X = self._add_vol_features(frame).iloc[train_idx]
        self.feature_names = list(X.columns)

        # Target: log-variance of the return model's residuals
        residuals = frame[TARGET_COL].values[train_idx] - mu_train
        log_variance = np.log(residuals**2 + 1e-8)

        self.model = train_booster(
            self.params,
            X,
            log_variance,
            self.num_boost_round,
            self.early_stopping_rounds,
        )

        # Variance bias correction, then the training range for clipping
        raw_variance = np.exp(self.model.predict(X))
        self.variance_scale = float(np.mean(residuals**2) / np.mean(raw_variance))
        train_vol = np.sqrt(raw_variance * self.variance_scale)
        self.train_vol_range = (
            np.percentile(train_vol, self.clip_percentiles[0]),
            np.percentile(train_vol, self.clip_percentiles[1]),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated volatility forecast for every row of the frame."""
        X = self._add_vol_features(frame)[self.feature_names]
        vol_pred = np.sqrt(np.exp(self.model.predict(X)) * self.variance_scale)
        return self._calibrate_volatility(vol_pred)

    def _calibrate_volatility(self, vol_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate volatility predictions.

        Steps:
        1. Clip to training distribution
        2. EWMA smoothing (causal)
        """
        vol_calibrated = np.clip(vol_pred, *self.train_vol_range)

        vol_smoothed = np.zeros_like(vol_calibrated)
        vol_smoothed[0] = vol_calibrated[0]
        for i in range(1, len(vol_calibrated)):
            vol_smoothed[i] = (
                self.ewma_lambda * vol_smoothed[i - 1]
                + (1 - self.ewma_lambda) * vol_calibrated[i]
            )
        return vol_smoothed


class EWMAVolatility(_Persistable):
    """RiskMetrics: sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2."""

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self.initial_variance = None

    def fit(self, frame, train_idx, mu_train=None) -> "EWMAVolatility":
        returns = frame[TARGET_COL].values[train_idx]
        self.initial_variance = float(np.var(returns))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        returns = np.nan_to_num(frame[TARGET_COL].values)
        variance = np.empty(len(returns))
        variance[0] = self.initial_variance
        for t in range(1, len(returns)):
            variance[t] = self.lam * variance[t - 1] + (1 - self.lam) * returns[t - 1] ** 2
        return np.sqrt(variance)


class GARCHVolatility(_Persistable):
    """
    GARCH-family model with Student-t errors (arch package).

    Parameters are estimated on the training rows, then held fixed while the
    variance recursion runs over the whole frame.
    """

    SPECS = {
        "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
        "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
        "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    }

    # Returns are scaled to percent for numerically stable estimation
    SCALE = 100.0

    def __init__(self, kind: str = "garch"):
        if kind not in self.SPECS:
            raise ValueError(f"Unknown GARCH kind: {kind}")
        self.kind = kind
        self.params = None

    def _model(self, returns: np.ndarray):
        from arch import arch_model

        return arch_model(
            returns * self.SCALE, mean="Constant", dist="t", **self.SPECS[self.kind]
        )

    def fit(self, frame, train_idx, mu_train=None) -> "GARCHVolatility":
        returns = frame[TARGET_COL].values[train_idx]
        result = self._model(returns).fit(disp="off", options={"maxiter": 1000})
        self.params = result.params
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        returns = np.nan_to_num(frame[TARGET_COL].values)
        result = self._model(returns).fix(self.params)
        return np.asarray(result.conditional_volatility) / self.SCALE


VOLATILITY_MODELS = {
    "lgbm": VolatilityPredictor,
    "ewma": EWMAVolatility,
    "garch": lambda: GARCHVolatility("garch"),
    "gjr": lambda: GARCHVolatility("gjr"),
    "egarch": lambda: GARCHVolatility("egarch"),
}
