"""
Pipeline

Runs one configuration on one fold:

1. Feature engineering (fit on the training rows, applied to the whole
   contiguous fold frame so rolling features have real history)
2. Out-of-fold return and volatility predictions for the last OOF_PCT of the
   training window, from expanding windows
3. Allocation sizing (and meta-labeling, if enabled) fitted on those
   out-of-fold predictions
4. Return and volatility models refitted on the whole training window,
   then used to allocate on the evaluation rows

Nothing in steps 1 to 4 sees the evaluation rows' labels.
"""

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .allocation import RegimeDependentAllocator
from .config import *
from .features import FeatureEngineer
from .meta_labeling import MetaLabelPipeline
from .metrics import competition_score, max_drawdown, sharpe_ratio
from .preprocessing import Fold, get_feature_columns
from .returns import RETURN_MODELS
from .volatility import VOLATILITY_MODELS


@dataclass
class PipelineConfig:
    """One experiment configuration."""

    name: str = "baseline"
    return_model: str = "lgbm"  # Key in returns.RETURN_MODELS
    vol_model: str = "lgbm"  # Key in volatility.VOLATILITY_MODELS
    meta: bool = False  # Scale the tilt with meta-labeling
    regime_dependent: bool = True  # Separate tilt size per volatility regime
    return_kwargs: Dict = field(default_factory=dict)  # Extra model arguments
    tune_trials: int = 0  # > 0: tune the LightGBM return model per fold (tuning.py)


@dataclass
class FoldFeatures:
    """Engineered features for every row of a fold frame (config-independent)."""

    X: pd.DataFrame
    regime: np.ndarray
    engineer: FeatureEngineer


@dataclass
class FoldResult:
    """Allocations and predictions on the evaluation rows of one fold."""

    fold: str
    allocations: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    meta_proba: Optional[np.ndarray]
    excess: np.ndarray  # Realized excess return of the market
    forward_returns: np.ndarray
    risk_free_rate: np.ndarray
    date_id: np.ndarray
    models: Dict


def excess_returns(frame: pd.DataFrame) -> np.ndarray:
    """What a 100% position earns over the risk-free rate."""
    return (frame["forward_returns"] - frame["risk_free_rate"]).values


def engineer_fold(fold: Fold, verbose: bool = False) -> FoldFeatures:
    """Fit feature engineering on the training rows, transform the whole frame."""
    engineer = FeatureEngineer(
        continuous_features=get_feature_columns(fold.frame), verbose=verbose
    )
    engineer.fit(fold.frame.iloc[fold.train_idx])
    features = engineer.transform(fold.frame)

    X = features[engineer.final_selected_features].fillna(0)
    return FoldFeatures(X=X, regime=engineer.regime(fold.frame), engineer=engineer)


def _fit_stack(cfg: PipelineConfig, fold: Fold, feats: FoldFeatures, fit_idx):
    """Fit the return and volatility models on rows fit_idx."""
    y = excess_returns(fold.frame)[fit_idx]
    return_model = RETURN_MODELS[cfg.return_model](**cfg.return_kwargs)
    return_model.fit(feats.X.iloc[fit_idx], y)

    mu_fit = return_model.predict(feats.X.iloc[fit_idx])
    vol_model = VOLATILITY_MODELS[cfg.vol_model]()
    vol_model.fit(fold.frame, fit_idx, mu_fit)

    return return_model, vol_model


def _predict_stack(return_model, vol_model, fold: Fold, feats: FoldFeatures, idx):
    """Return and volatility forecasts for rows idx."""
    mu = return_model.predict(feats.X.iloc[idx])
    sigma = vol_model.predict(fold.frame)[idx]
    return mu, sigma


def oof_predictions(cfg: PipelineConfig, fold: Fold, feats: FoldFeatures):
    """
    Out-of-fold predictions for the last OOF_PCT of the training window.

    The region is cut into OOF_SPLITS chunks; each chunk is predicted by
    models fitted on all training rows before it (minus the purge gap).

    Returns
    -------
    oof_idx, mu, sigma : np.ndarray
    """
    train_idx = fold.train_idx
    n = len(train_idx)
    bounds = np.linspace(int(n * (1 - OOF_PCT)), n, OOF_SPLITS + 1).astype(int)

    oof_idx, mus, sigmas = [], [], []
    for start, end in zip(bounds[:-1], bounds[1:]):
        fit_idx = train_idx[: start - CV_PURGE_GAP]
        chunk = train_idx[start:end]
        return_model, vol_model = _fit_stack(cfg, fold, feats, fit_idx)
        mu, sigma = _predict_stack(return_model, vol_model, fold, feats, chunk)
        oof_idx.append(chunk)
        mus.append(mu)
        sigmas.append(sigma)

    return np.concatenate(oof_idx), np.concatenate(mus), np.concatenate(sigmas)


def fit_fold(cfg: PipelineConfig, fold: Fold, feats: FoldFeatures) -> Dict:
    """
    Fit every component on the training rows of a fold.

    Returns
    -------
    models : Dict
        engineer, returns, volatility, allocator, meta (None if disabled)
    """
    excess = excess_returns(fold.frame)

    # Sizing (and meta-labeling) are fitted on out-of-fold predictions
    oof_idx, mu_oof, sigma_oof = oof_predictions(cfg, fold, feats)
    allocator = RegimeDependentAllocator(regime_dependent=cfg.regime_dependent)
    allocator.fit(mu_oof, sigma_oof, feats.regime[oof_idx], excess[oof_idx])

    meta_model = None
    if cfg.meta:
        meta_model = MetaLabelPipeline()
        meta_model.fit(feats.X.iloc[oof_idx], mu_oof, excess[oof_idx])

    # Final models on the whole training window
    return_model, vol_model = _fit_stack(cfg, fold, feats, fold.train_idx)

    return {
        "engineer": feats.engineer,
        "returns": return_model,
        "volatility": vol_model,
        "allocator": allocator,
        "meta": meta_model,
    }


def run_fold(cfg: PipelineConfig, fold: Fold, feats: FoldFeatures) -> FoldResult:
    """Fit everything on the training rows and allocate on the evaluation rows."""
    models = fit_fold(cfg, fold, feats)
    eval_idx = fold.eval_idx
    mu, sigma = _predict_stack(
        models["returns"], models["volatility"], fold, feats, eval_idx
    )

    meta_proba, multiplier = None, None
    if models["meta"] is not None:
        meta_proba = models["meta"].predict_proba(feats.X.iloc[eval_idx], mu)
        multiplier = models["meta"].tilt_multiplier(meta_proba)

    allocations = models["allocator"].allocate(
        mu, sigma, feats.regime[eval_idx], multiplier
    )

    frame = fold.frame.iloc[eval_idx]
    return FoldResult(
        fold=fold.name,
        allocations=allocations,
        mu=mu,
        sigma=sigma,
        meta_proba=meta_proba,
        excess=excess_returns(frame),
        forward_returns=frame["forward_returns"].values,
        risk_free_rate=frame["risk_free_rate"].values,
        date_id=frame["date_id"].values,
        models=models,
    )


def fold_metrics(result: FoldResult) -> dict:
    """Strategy and buy-and-hold metrics for one fold."""
    strategy = result.allocations * result.excess
    buy_hold = result.excess
    resid = result.excess - result.mu

    return {
        "fold": result.fold,
        "n_days": len(strategy),
        "sharpe": sharpe_ratio(strategy),
        "sharpe_buy_hold": sharpe_ratio(buy_hold),
        "score": competition_score(
            result.allocations, result.forward_returns, result.risk_free_rate
        ),
        "score_buy_hold": competition_score(
            np.ones(len(strategy)), result.forward_returns, result.risk_free_rate
        ),
        "max_drawdown": max_drawdown(strategy),
        "max_drawdown_buy_hold": max_drawdown(buy_hold),
        "mean_allocation": float(np.mean(result.allocations)),
        "dir_accuracy": float(np.mean(np.sign(result.mu) == np.sign(result.excess))),
        "vol_qlike": float(
            np.mean(np.log(result.sigma**2) + resid**2 / result.sigma**2)
        ),
    }


def daily_returns(results: List[FoldResult]) -> pd.DataFrame:
    """Concatenated out-of-sample daily returns of the strategy and the market."""
    return pd.DataFrame(
        {
            "fold": np.concatenate([[r.fold] * len(r.excess) for r in results]),
            "date_id": np.concatenate([r.date_id for r in results]),
            "allocation": np.concatenate([r.allocations for r in results]),
            "strategy": np.concatenate([r.allocations * r.excess for r in results]),
            "buy_hold": np.concatenate([r.excess for r in results]),
        }
    )


def load_fold_features(fold: Fold, cache_dir=None, verbose: bool = False) -> FoldFeatures:
    """engineer_fold with an on-disk cache (features do not depend on the config)."""
    if cache_dir is None:
        return engineer_fold(fold, verbose=verbose)

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"fold_{fold.name}_{fold.train_idx[-1]}_{fold.eval_idx[-1]}.pkl"
    path = cache_dir / key
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    feats = engineer_fold(fold, verbose=verbose)
    with open(path, "wb") as f:
        pickle.dump(feats, f)
    return feats
