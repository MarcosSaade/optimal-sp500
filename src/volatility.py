"""
Volatility Prediction Model

LightGBM model for predicting conditional 1-day volatility.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict
import pickle
from scipy import stats

from .config import *


class VolatilityPredictor:
    """
    Volatility prediction using LightGBM.

    Predicts conditional 1-day volatility for position sizing.
    Target: log-variance of residuals from return predictions.
    """

    def __init__(
        self,
        params: Dict = None,
        num_boost_round: int = LGBM_VOL_ROUNDS,
        early_stopping_rounds: int = LGBM_VOL_EARLY_STOP,
        bias_delta: float = VOL_BIAS_DELTA,
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
        bias_delta : float
            Bias correction parameter
        clip_percentiles : tuple
            (low, high) percentiles for clipping
        ewma_lambda : float
            EWMA smoothing parameter
        """
        self.params = params if params is not None else LGBM_VOL_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.bias_delta = bias_delta
        self.clip_percentiles = clip_percentiles
        self.ewma_lambda = ewma_lambda

        self.model = None
        self.feature_names = None
        self.train_vol_range = None
        self.is_fitted = False

    def _create_vol_target(
        self,
        returns: np.ndarray,
        mu_predictions: np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Create log-variance target from residuals."""
        residuals = returns - mu_predictions
        log_variance = np.log(residuals**2 + epsilon)
        return log_variance

    def _add_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-specific features."""
        df = df.copy()

        # Get past returns
        past_returns = df[TARGET_COL].shift(1)

        # Historical volatility
        for window in [5, 10, 21, 42, 63]:
            df[f"hist_vol_{window}"] = past_returns.rolling(window).std()

        # Absolute returns
        df["abs_return"] = np.abs(past_returns)
        for window in [5, 10, 21]:
            df[f"abs_return_ma_{window}"] = df["abs_return"].rolling(window).mean()

        # Squared returns
        df["squared_return"] = past_returns**2
        for window in [5, 10, 21]:
            df[f"squared_return_ma_{window}"] = (
                df["squared_return"].rolling(window).mean()
            )

        # EWMA volatility
        for span in [10, 20]:
            df[f"vol_ema_{span}"] = past_returns.rolling(21).std().ewm(span=span).mean()

        return df

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        mu_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        mu_val: np.ndarray = None,
    ) -> "VolatilityPredictor":
        """
        Train volatility prediction model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training actual returns
        mu_train : np.ndarray
            Training return predictions (from return model)
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation actual returns
        mu_val : np.ndarray, optional
            Validation return predictions

        Returns
        -------
        self : VolatilityPredictor
            Fitted predictor
        """
        # Add volatility features
        X_train_vol = self._add_vol_features(X_train)

        # Create volatility target (log-variance of residuals)
        vol_target = self._create_vol_target(y_train.values, mu_train)

        # Get volatility feature columns
        vol_features = [
            col for col in X_train_vol.columns if col not in X_train.columns
        ]
        vol_features = [
            col for col in vol_features if not X_train_vol[col].isna().all()
        ]

        self.feature_names = vol_features

        # Fill NaNs
        X_train_clean = X_train_vol[vol_features].fillna(0)

        # Create datasets
        train_data = lgb.Dataset(X_train_clean, label=vol_target)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and mu_val is not None:
            X_val_vol = self._add_vol_features(X_val)
            vol_target_val = self._create_vol_target(y_val.values, mu_val)
            X_val_clean = X_val_vol[vol_features].fillna(0)

            val_data = lgb.Dataset(
                X_val_clean, label=vol_target_val, reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append("val")

        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                )
            ],
        )

        # Store training volatility range for calibration
        train_vol = np.exp(self.model.predict(X_train_clean) / 2)
        self.train_vol_range = (
            np.percentile(train_vol, self.clip_percentiles[0]),
            np.percentile(train_vol, self.clip_percentiles[1]),
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated volatility.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        volatility : np.ndarray
            Predicted calibrated volatility
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Add volatility features
        X_vol = self._add_vol_features(X)
        X_clean = X_vol[self.feature_names].fillna(0)

        # Predict log-variance and convert to volatility
        log_var_pred = self.model.predict(X_clean)
        vol_pred = np.exp(log_var_pred / 2)

        # Calibrate
        vol_calibrated = self._calibrate_volatility(vol_pred)

        return vol_calibrated

    def _calibrate_volatility(self, vol_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate volatility predictions.

        Steps:
        1. Bias correction
        2. Clip to training distribution
        3. EWMA smoothing
        """
        # Bias correction
        vol_calibrated = np.sqrt(vol_pred**2 + self.bias_delta**2)

        # Clip to training distribution
        if self.train_vol_range is not None:
            vol_calibrated = np.clip(
                vol_calibrated, self.train_vol_range[0], self.train_vol_range[1]
            )

        # EWMA smoothing
        if len(vol_calibrated) > 1:
            vol_smoothed = np.zeros_like(vol_calibrated)
            vol_smoothed[0] = vol_calibrated[0]
            for i in range(1, len(vol_calibrated)):
                vol_smoothed[i] = (
                    self.ewma_lambda * vol_smoothed[i - 1]
                    + (1 - self.ewma_lambda) * vol_calibrated[i]
                )
            return vol_smoothed

        return vol_calibrated

    def save(self, filepath: Path):
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "VolatilityPredictor":
        """Load fitted model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
