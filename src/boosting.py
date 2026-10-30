"""
LightGBM Training Helper

Early stopping without touching the data being scored:
1. Hold out the last INNER_VAL_PCT of the training window (time-ordered)
2. Train with early stopping on that slice to pick the number of rounds
3. Refit on the full training window with that number of rounds
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict

from .config import *


def train_booster(
    params: Dict,
    X: pd.DataFrame,
    y: np.ndarray,
    num_boost_round: int,
    early_stopping_rounds: int,
    inner_val_pct: float = INNER_VAL_PCT,
) -> lgb.Booster:
    """
    Train a LightGBM model, early-stopping on the end of the training window.

    Parameters
    ----------
    params : Dict
        LightGBM parameters
    X : pd.DataFrame
        Training features, in time order
    y : np.ndarray
        Training labels
    num_boost_round : int
        Maximum boosting rounds
    early_stopping_rounds : int
        Early stopping patience
    inner_val_pct : float
        Fraction of the training window used to pick the number of rounds

    Returns
    -------
    model : lgb.Booster
        Model refitted on all of X with the selected number of rounds
    """
    y = np.asarray(y)
    n_inner = int(len(X) * inner_val_pct)
    split = len(X) - n_inner

    # The row just before the inner slice has a label that overlaps it
    fit_end = split - CV_PURGE_GAP

    inner_train = lgb.Dataset(X.iloc[:fit_end], label=y[:fit_end])
    inner_val = lgb.Dataset(X.iloc[split:], label=y[split:], reference=inner_train)

    probe = lgb.train(
        params,
        inner_train,
        num_boost_round=num_boost_round,
        valid_sets=[inner_val],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_rounds = max(probe.best_iteration, 1)

    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=best_rounds)
