"""
Return Prediction Model

LightGBM model for predicting excess returns.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Tuple, Dict
import pickle

from .config import *


class ReturnPredictor:
    """
    Return prediction using LightGBM.

    Predicts 1-day forward excess returns given market features.
    """

    def __init__(
        self,
        params: Dict = None,
        num_boost_round: int = LGBM_RETURN_ROUNDS,
        early_stopping_rounds: int = LGBM_RETURN_EARLY_STOP,
    ):
        """
        Initialize return predictor.

        Parameters
        ----------
        params : Dict, optional
            LightGBM parameters (uses default if None)
        num_boost_round : int
            Maximum boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        """
        self.params = params if params is not None else LGBM_RETURN_PARAMS.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> "ReturnPredictor":
        """
        Train return prediction model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets (excess returns)
        X_val : pd.DataFrame, optional
            Validation features for early stopping
        y_val : pd.Series, optional
            Validation targets

        Returns
        -------
        self : ReturnPredictor
            Fitted predictor
        """
        # Store feature names
        self.feature_names = list(X_train.columns)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
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

        # Store feature importances
        self.feature_importances = dict(
            zip(
                self.feature_names,
                self.model.feature_importance(importance_type="gain"),
            )
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict excess returns.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        predictions : np.ndarray
            Predicted excess returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self, top_n: int = None) -> Dict[str, float]:
        """
        Get feature importances.

        Parameters
        ----------
        top_n : int, optional
            Return only top N features

        Returns
        -------
        importances : Dict[str, float]
            Feature importances (sorted if top_n specified)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if top_n is None:
            return self.feature_importances

        sorted_features = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_features[:top_n])

    def save(self, filepath: Path):
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "ReturnPredictor":
        """Load fitted model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
