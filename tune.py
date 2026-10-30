"""
Hyperparameter Tuning (nested)

Tunes the LightGBM return model with Optuna without looking at the folds
being scored (see src/tuning.py):

1. For each development fold, every trial is scored on out-of-fold
   predictions inside that fold's training window: Sharpe ratio of the
   sized positions.
2. The best parameters are refitted on the whole training window and the
   fold's evaluation rows are scored once.

The tuned configuration is logged to the experiments ledger like any other
trial, so it counts towards the Deflated Sharpe Ratio.

Usage:
    python tune.py --vol-model lgbm --n-trials 50
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from experiments import CACHE_DIR, RETURNS_DIR, log_to_ledger
from src.config import *
from src.pipeline import (
    PipelineConfig,
    daily_returns,
    fold_metrics,
    load_fold_features,
    run_fold,
)
from src.preprocessing import load_and_prepare_data, make_folds
from src.tuning import prepare_config


def main():
    parser = argparse.ArgumentParser(description="Nested hyperparameter tuning")
    parser.add_argument("--vol-model", default="lgbm", help="Volatility model")
    parser.add_argument("--n-trials", type=int, default=50, help="Trials per fold")
    args = parser.parse_args()

    cfg = PipelineConfig(
        name=f"tuned_lgbm+{args.vol_model}",
        return_model="lgbm",
        vol_model=args.vol_model,
        tune_trials=args.n_trials,
    )
    start = time.time()

    data = load_and_prepare_data(DATA_DIR / "train.csv")
    results = []
    for fold in make_folds(data):
        feats = load_fold_features(fold, CACHE_DIR)
        result = run_fold(prepare_config(cfg, fold, feats), fold, feats)
        results.append(result)
        print(f"Fold {fold.name}: Sharpe {fold_metrics(result)['sharpe']:.3f}")

    RETURNS_DIR.mkdir(parents=True, exist_ok=True)
    daily_returns(results).to_csv(RETURNS_DIR / f"{cfg.name}.csv", index=False)
    metrics = pd.DataFrame([fold_metrics(r) for r in results])
    log_to_ledger(cfg, metrics, time.time() - start)

    print(f"\nMean fold Sharpe: {metrics['sharpe'].mean():.3f}")
    print(f"Mean buy-and-hold: {metrics['sharpe_buy_hold'].mean():.3f}")


if __name__ == "__main__":
    main()
