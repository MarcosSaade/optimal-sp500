"""
Meta-Labeling

Confidence-based position sizing using a meta-classifier.

From "Advances in Financial Machine Learning" Chapter 3:
- Primary model predicts returns (direction and magnitude)
- Meta-classifier predicts if primary model is correct
- Scale positions by meta-probability
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict
import pickle

from .config import *


class MetaLabelPipeline:
    """
    Complete meta-labeling pipeline.

    Workflow:
    1. Generate meta-labels from primary predictions
    2. Train meta-classifier to predict correctness
    3. Use meta-probabilities to scale positions
    """

    def __init__(
        self,
        meta_label_method: str = META_LABEL_METHOD,
        min_confidence: float = META_MIN_CONFIDENCE,
        params: Dict = None,
        num_boost_round: int = LGBM_META_ROUNDS,
        early_stopping_rounds: int = LGBM_META_EARLY_STOP,
    ):
        """
        Initialize meta-labeling pipeline.

        Parameters
        ----------
        meta_label_method : str
            How to generate meta-labels ("sign", "barrier")
        min_confidence : float
            Minimum meta-probability to trade (0 = always trade)
        params : Dict, optional
            LightGBM parameters
        num_boost_round : int
            Maximum boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        """
        self.meta_label_method = meta_label_method
        self.min_confidence = min_confidence
        self.params = params if params is not None else LGBM_META_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def _generate_meta_labels(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> np.ndarray:
        """
        Generate meta-labels: 1 if prediction correct, 0 otherwise.

        For "sign" method: 1 if sign(prediction) == sign(actual)
        """
        if self.meta_label_method == "sign":
            pred_sign = np.sign(predictions)
            actual_sign = np.sign(actuals)
            return (pred_sign == actual_sign).astype(int)
        else:
            raise ValueError(f"Unknown method: {self.meta_label_method}")

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
        - Volatility features
        """
        meta_features = features.copy()

        # Add prediction-based features
        meta_features["pred_magnitude"] = np.abs(predictions)
        meta_features["pred_sign"] = np.sign(predictions)

        # Prediction volatility (only using past predictions)
        pred_series = pd.Series(predictions)
        for window in [5, 10, 21]:
            rolling_std = (
                pred_series.rolling(window, min_periods=1).std().shift(1).fillna(0.0)
            )
            meta_features[f"pred_vol_{window}"] = rolling_std

        return meta_features

    def fit(
        self,
        train_features: pd.DataFrame,
        train_predictions: np.ndarray,
        train_actuals: np.ndarray,
        val_features: pd.DataFrame = None,
        val_predictions: np.ndarray = None,
        val_actuals: np.ndarray = None,
    ) -> "MetaLabelPipeline":
        """
        Train meta-labeling pipeline.

        Parameters
        ----------
        train_features : pd.DataFrame
            Training features
        train_predictions : np.ndarray
            Primary model predictions on training set
        train_actuals : np.ndarray
            Actual returns on training set
        val_features : pd.DataFrame, optional
            Validation features for early stopping
        val_predictions : np.ndarray, optional
            Validation predictions
        val_actuals : np.ndarray, optional
            Validation actuals

        Returns
        -------
        self : MetaLabelPipeline
            Fitted pipeline
        """
        # Generate meta-labels
        meta_labels = self._generate_meta_labels(train_predictions, train_actuals)

        # Prepare meta-features
        X_meta = self._prepare_meta_features(train_features, train_predictions)

        # Remove NaNs
        X_meta = X_meta.fillna(0)
        self.feature_names = list(X_meta.columns)

        # Create datasets
        train_data = lgb.Dataset(X_meta, label=meta_labels)

        valid_sets = [train_data]
        valid_names = ["train"]

        if (
            val_features is not None
            and val_predictions is not None
            and val_actuals is not None
        ):
            val_meta_labels = self._generate_meta_labels(val_predictions, val_actuals)
            X_val_meta = self._prepare_meta_features(val_features, val_predictions)
            X_val_meta = X_val_meta.fillna(0)

            val_data = lgb.Dataset(
                X_val_meta, label=val_meta_labels, reference=train_data
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

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Predict meta-probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Features
        predictions : np.ndarray
            Primary model predictions

        Returns
        -------
        meta_proba : np.ndarray
            Probability that primary prediction is correct
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare meta-features
        X_meta = self._prepare_meta_features(features, predictions)
        X_meta = X_meta.fillna(0)

        # Predict probabilities
        meta_proba = self.model.predict(X_meta)

        return meta_proba

    def scale_positions(
        self,
        allocations: np.ndarray,
        meta_proba: np.ndarray,
    ) -> np.ndarray:
        """
        Scale positions by meta-probability.

        Parameters
        ----------
        allocations : np.ndarray
            Base allocations from primary model
        meta_proba : np.ndarray
            Meta-probabilities (confidence)

        Returns
        -------
        scaled_allocations : np.ndarray
            Allocations scaled by confidence
        """
        # Only trade if confidence > threshold
        trade_mask = meta_proba >= self.min_confidence

        # Scale by probability
        scaled_allocations = allocations * meta_proba * trade_mask

        return scaled_allocations

    def save(self, filepath: Path):
        """Save fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "MetaLabelPipeline":
        """Load fitted pipeline."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
