"""
Meta-Labeling

Confidence-based position sizing using a meta-classifier.

- Primary model predicts returns; its sign sets the direction of the tilt
  away from the neutral position
- Meta-classifier predicts whether that sign will be right
- The tilt is scaled by the meta-probability relative to its base rate

The meta-classifier must be trained on out-of-sample primary predictions
(see pipeline.py): on in-sample predictions the primary model looks far
more accurate than it is.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle

from .boosting import train_booster
from .config import *


class MetaLabelPipeline:
    """
    Complete meta-labeling pipeline.

    Workflow:
    1. Generate meta-labels from primary predictions
    2. Train meta-classifier to predict correctness
    3. Use meta-probabilities to scale the tilt
    """

    def __init__(
        self,
        params: Dict = None,
        num_boost_round: int = LGBM_META_ROUNDS,
        early_stopping_rounds: int = LGBM_META_EARLY_STOP,
    ):
        """
        Initialize meta-labeling pipeline.

        Parameters
        ----------
        params : Dict, optional
            LightGBM parameters
        num_boost_round : int
            Maximum boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        """
        self.params = params if params is not None else LGBM_META_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.base_rate = None

    @staticmethod
    def generate_meta_labels(
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> np.ndarray:
        """Meta-labels: 1 if sign(prediction) == sign(actual), else 0."""
        return (np.sign(predictions) == np.sign(actuals)).astype(int)

    def _prepare_meta_features(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Prepare features for meta-classifier.

        Includes:
        - Original features (market conditions, regimes)
        - Prediction magnitude (confidence proxy)
        - Prediction sign
        - Rolling volatility of past predictions
        """
        meta_features = features.reset_index(drop=True).copy()

        meta_features["pred_magnitude"] = np.abs(predictions)
        meta_features["pred_sign"] = np.sign(predictions)

        # Prediction volatility (only using past predictions)
        pred_series = pd.Series(predictions)
        for window in [5, 10, 21]:
            meta_features[f"pred_vol_{window}"] = (
                pred_series.rolling(window, min_periods=1).std().shift(1)
            )

        return meta_features.fillna(0)

    def fit(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "MetaLabelPipeline":
        """
        Train the meta-classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Features, in time order
        predictions : np.ndarray
            Out-of-sample primary model predictions for the same rows
        actuals : np.ndarray
            Realized excess returns
        """
        meta_labels = self.generate_meta_labels(predictions, actuals)
        self.base_rate = float(meta_labels.mean())

        X_meta = self._prepare_meta_features(features, predictions)
        self.model = train_booster(
            self.params,
            X_meta,
            meta_labels,
            self.num_boost_round,
            self.early_stopping_rounds,
        )
        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Probability that the primary prediction has the right sign."""
        X_meta = self._prepare_meta_features(features, predictions)
        return self.model.predict(X_meta)

    def tilt_multiplier(self, meta_proba: np.ndarray) -> np.ndarray:
        """
        Scale factor for the tilt: confidence relative to the base rate.

        Averages about 1, so the meta-model redistributes tilt size across
        days instead of shrinking every position.
        """
        return np.clip(meta_proba / self.base_rate, 0.0, 2.0)

    def save(self, filepath: Path):
        """Save fitted pipeline."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "MetaLabelPipeline":
        """Load fitted pipeline."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
