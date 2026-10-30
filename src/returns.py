"""
Return Prediction Models

Predict 1-day forward excess returns. All models share fit(X, y) / predict(X).

- ReturnPredictor: LightGBM (default)
- RidgeReturnPredictor: standardized ridge regression
- EnsembleReturnPredictor: average of LightGBM and ridge
- AutoMLReturnPredictor: FLAML search over model families
- ConstantReturnPredictor: training mean (no return view; isolates volatility timing)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import pickle

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


class ReturnPredictor(_Persistable):
    """
    Return prediction using LightGBM.

    Early stopping uses the end of the training window (see boosting.py).
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
        self.feature_importances = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ReturnPredictor":
        """Train on time-ordered features X and excess returns y."""
        self.model = train_booster(
            self.params, X, y, self.num_boost_round, self.early_stopping_rounds
        )
        self.feature_importances = dict(
            sorted(
                zip(X.columns, self.model.feature_importance(importance_type="gain")),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict excess returns."""
        return self.model.predict(X)


class RegularizedReturnPredictor(ReturnPredictor):
    """LightGBM with small trees, strong shrinkage and large leaves."""

    def __init__(self):
        params = LGBM_RETURN_PARAMS.copy()
        params.update(
            {
                "num_leaves": 7,
                "max_depth": 3,
                "learning_rate": 0.02,
                "min_data_in_leaf": 200,
                "lambda_l2": 10.0,
                "feature_fraction": 0.5,
            }
        )
        super().__init__(params=params)


class RidgeReturnPredictor(_Persistable):
    """Ridge regression on standardized features, penalty chosen by CV."""

    def __init__(self, alphas: np.ndarray = None):
        self.alphas = alphas if alphas is not None else np.logspace(0, 6, 25)
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RidgeReturnPredictor":
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class EnsembleReturnPredictor(_Persistable):
    """Equal-weight average of LightGBM and ridge predictions."""

    def __init__(self):
        self.members = [ReturnPredictor(), RidgeReturnPredictor()]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnsembleReturnPredictor":
        for member in self.members:
            member.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([member.predict(X) for member in self.members], axis=0)


class AutoMLReturnPredictor(_Persistable):
    """
    FLAML AutoML regression.

    Searches LightGBM, XGBoost, random forest, extra trees and linear models,
    validating on the last INNER_VAL_PCT of the training window.
    """

    def __init__(self, time_budget: int = 30):
        self.time_budget = time_budget
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AutoMLReturnPredictor":
        from flaml import AutoML

        self.model = AutoML()
        self.model.fit(
            X_train=X,
            y_train=np.asarray(y),
            task="regression",
            metric="mse",
            time_budget=self.time_budget,
            eval_method="holdout",
            split_type="time",
            split_ratio=INNER_VAL_PCT,
            estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "enet"],
            seed=RANDOM_SEED,
            verbose=0,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class ConstantReturnPredictor(_Persistable):
    """Predict the training mean every day: no view on returns."""

    def __init__(self):
        self.mean = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConstantReturnPredictor":
        self.mean = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean)


RETURN_MODELS = {
    "lgbm": ReturnPredictor,
    "lgbm_reg": RegularizedReturnPredictor,
    "ridge": RidgeReturnPredictor,
    "ensemble": EnsembleReturnPredictor,
    "automl": AutoMLReturnPredictor,
    "constant": ConstantReturnPredictor,
}
