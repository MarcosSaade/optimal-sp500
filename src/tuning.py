"""
Nested Hyperparameter Tuning

Tunes the LightGBM return model for one fold with Optuna, scoring each trial
on out-of-fold predictions inside the fold's training window (the same
expanding windows used to fit the allocator). The fold's scored rows are
never seen by the search.

A configuration with tune_trials > 0 is resolved per fold by
prepare_config before it is run.
"""

from dataclasses import replace

import optuna
from optuna.samplers import TPESampler

from .allocation import RegimeDependentAllocator
from .config import *
from .metrics import sharpe_ratio
from .pipeline import PipelineConfig, excess_returns, oof_predictions


def get_return_search_space(trial) -> dict:
    """Search space for the LightGBM return model."""
    params = LGBM_RETURN_PARAMS.copy()
    params.update(
        {
            "num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.95),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
        }
    )
    return params


def inner_score(cfg: PipelineConfig, fold, feats) -> float:
    """Sharpe ratio of sized positions on out-of-fold training-window predictions."""
    oof_idx, mu, sigma = oof_predictions(cfg, fold, feats)
    excess = excess_returns(fold.frame)[oof_idx]
    regime = feats.regime[oof_idx]

    allocator = RegimeDependentAllocator(regime_dependent=cfg.regime_dependent)
    allocator.fit(mu, sigma, regime, excess)
    return sharpe_ratio(allocator.allocate(mu, sigma, regime) * excess)


def prepare_config(cfg: PipelineConfig, fold, feats) -> PipelineConfig:
    """
    Resolve a configuration for one fold.

    If cfg.tune_trials > 0, search the return model's parameters on the
    fold's training window and return a copy with the best parameters.
    """
    if cfg.tune_trials <= 0:
        return cfg

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        trial_cfg = replace(
            cfg, tune_trials=0, return_kwargs={"params": get_return_search_space(trial)}
        )
        return inner_score(trial_cfg, fold, feats)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=cfg.tune_trials)

    return replace(
        cfg,
        tune_trials=0,
        return_kwargs={"params": get_return_search_space(study.best_trial)},
    )
