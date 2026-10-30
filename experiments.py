"""
Experiments

Runs pipeline configurations on the development folds, logs every one to a
ledger (results/experiments.csv), and reports the best configuration with a
Deflated Sharpe Ratio that accounts for all configurations tried.

The lockbox (the last LOCKBOX_DAYS rows) is scored once, with one
configuration, via --lockbox.

Usage:
    python experiments.py                       # all experiment groups
    python experiments.py --group vol           # one group
    python experiments.py --summary             # ledger summary and DSR
    python experiments.py --lockbox NAME        # score NAME on the lockbox (once)
"""

import argparse
import ast
import itertools
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_difference_ci,
    sharpe_ratio,
)
from src.pipeline import (
    PipelineConfig,
    daily_returns,
    fold_metrics,
    load_fold_features,
    run_fold,
)
from src.preprocessing import load_and_prepare_data, make_folds
from src.tuning import prepare_config

LEDGER_PATH = RESULTS_DIR / "experiments.csv"
RETURNS_DIR = RESULTS_DIR / "returns"
CACHE_DIR = RESULTS_DIR / "cache"
LOCKBOX_PATH = RESULTS_DIR / "lockbox.csv"

RETURN_CHOICES = ["constant", "lgbm", "lgbm_reg", "ridge", "ensemble"]
VOL_CHOICES = ["lgbm", "ewma", "garch", "gjr", "egarch"]


def experiment_groups() -> dict:
    """Named groups of configurations."""
    grid = [
        PipelineConfig(name=f"{ret}+{vol}", return_model=ret, vol_model=vol)
        for ret, vol in itertools.product(RETURN_CHOICES, VOL_CHOICES)
    ]
    flat = [
        PipelineConfig(
            name=f"{ret}+{vol}+flat",
            return_model=ret,
            vol_model=vol,
            regime_dependent=False,
        )
        for ret, vol in itertools.product(RETURN_CHOICES, VOL_CHOICES)
    ]
    meta = [
        PipelineConfig(
            name=f"{ret}+{vol}+meta", return_model=ret, vol_model=vol, meta=True
        )
        for ret, vol in itertools.product(["lgbm", "ensemble"], VOL_CHOICES)
    ]
    automl = [
        PipelineConfig(
            name=f"automl+{vol}",
            return_model="automl",
            vol_model=vol,
            return_kwargs={"time_budget": 30},
        )
        for vol in ["lgbm", "garch"]
    ]
    return {"grid": grid, "flat": flat, "meta": meta, "automl": automl}


def run_config(cfg: PipelineConfig, folds, features) -> pd.DataFrame:
    """Run one configuration on all folds; return per-fold metrics."""
    results = [run_fold(cfg, fold, features[fold.name]) for fold in folds]
    metrics = pd.DataFrame([fold_metrics(r) for r in results])

    RETURNS_DIR.mkdir(parents=True, exist_ok=True)
    daily_returns(results).to_csv(RETURNS_DIR / f"{cfg.name}.csv", index=False)
    return metrics


def log_to_ledger(cfg: PipelineConfig, metrics: pd.DataFrame, seconds: float):
    """Append (or replace) the configuration's row in the ledger."""
    daily = pd.read_csv(RETURNS_DIR / f"{cfg.name}.csv")
    row = {
        "name": cfg.name,
        **{k: v for k, v in asdict(cfg).items() if k != "name"},
        "mean_fold_sharpe": metrics["sharpe"].mean(),
        "pooled_sharpe": sharpe_ratio(daily["strategy"].values),
        "pooled_sharpe_buy_hold": sharpe_ratio(daily["buy_hold"].values),
        "pooled_sr_daily": sharpe_ratio(daily["strategy"].values, annualize=False),
        "mean_fold_score": metrics["score"].mean(),
        "mean_dir_accuracy": metrics["dir_accuracy"].mean(),
        "mean_vol_qlike": metrics["vol_qlike"].mean(),
        "mean_allocation": metrics["mean_allocation"].mean(),
        **{f"sharpe_fold{m.fold}": m.sharpe for m in metrics.itertuples()},
        "seconds": round(seconds, 1),
    }

    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    if len(ledger):
        ledger = ledger[ledger["name"] != cfg.name]
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(LEDGER_PATH, index=False)


def summarize():
    """Print the ledger and the Deflated Sharpe Ratio of the best configuration."""
    ledger = pd.read_csv(LEDGER_PATH).sort_values("pooled_sharpe", ascending=False)
    cols = [
        "name",
        "pooled_sharpe",
        "pooled_sharpe_buy_hold",
        "mean_fold_sharpe",
        "mean_fold_score",
        "mean_dir_accuracy",
        "mean_vol_qlike",
    ]
    print(ledger[cols].to_string(index=False, float_format="%.3f"))

    best = ledger.iloc[0]
    daily = pd.read_csv(RETURNS_DIR / f"{best['name']}.csv")
    strategy, buy_hold = daily["strategy"].values, daily["buy_hold"].values

    trials = ledger["pooled_sr_daily"].values
    dsr = deflated_sharpe_ratio(strategy, trials)
    dsr_bh = deflated_sharpe_ratio(strategy, trials, benchmark_excess=buy_hold)
    low, high = sharpe_difference_ci(strategy, buy_hold)

    print("\n" + "=" * 80)
    print(f"Best configuration: {best['name']}")
    print("=" * 80)
    print(f"  Pooled Sharpe (dev folds):     {best['pooled_sharpe']:.3f}")
    print(f"  Buy-and-hold, same days:       {best['pooled_sharpe_buy_hold']:.3f}")
    print(f"  Sharpe difference, 95% CI:     [{low:+.3f}, {high:+.3f}]")
    print(f"  Configurations tried (N):      {dsr['n_trials']}")
    print(f"  Expected max Sharpe, no skill: {dsr['sr0_annual']:.3f}")
    print(f"  Deflated Sharpe Ratio (> 0):   {dsr['dsr']:.3f}")
    print(f"  DSR vs buy-and-hold:           {dsr_bh['dsr']:.3f}  (benchmark {dsr_bh['sr0_annual']:.3f})")
    print(f"  Probabilistic SR (> 0):        {probabilistic_sharpe_ratio(strategy):.3f}")


def score_lockbox(name: str, data: pd.DataFrame, force: bool = False):
    """Score one configuration on the lockbox. Meant to be run once."""
    if LOCKBOX_PATH.exists() and not force:
        print(f"The lockbox was already scored ({LOCKBOX_PATH}).")
        print(pd.read_csv(LOCKBOX_PATH).T.to_string(header=False))
        return

    ledger = pd.read_csv(LEDGER_PATH).set_index("name")
    row = ledger.loc[name]
    cfg = PipelineConfig(
        name=name,
        return_model=row["return_model"],
        vol_model=row["vol_model"],
        meta=bool(row["meta"]),
        regime_dependent=bool(row["regime_dependent"]),
        return_kwargs=ast.literal_eval(row["return_kwargs"]),
        tune_trials=int(row["tune_trials"]),
    )

    fold = make_folds(data, lockbox=True)[0]
    feats = load_fold_features(fold)
    result = run_fold(prepare_config(cfg, fold, feats), fold, feats)
    metrics = fold_metrics(result)

    strategy = result.allocations * result.excess
    low, high = sharpe_difference_ci(strategy, result.excess)
    metrics.update(
        {
            "config": name,
            "date_id_start": int(result.date_id[0]),
            "date_id_end": int(result.date_id[-1]),
            "sharpe_diff_ci_low": low,
            "sharpe_diff_ci_high": high,
            "psr": probabilistic_sharpe_ratio(strategy),
        }
    )
    pd.DataFrame([metrics]).to_csv(LOCKBOX_PATH, index=False)
    daily_returns([result]).to_csv(RESULTS_DIR / "lockbox_daily.csv", index=False)

    print(pd.DataFrame([metrics]).T.to_string(header=False))


def main():
    groups = experiment_groups()

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--group", choices=list(groups), action="append")
    parser.add_argument("--summary", action="store_true", help="Only summarize")
    parser.add_argument("--lockbox", metavar="NAME", help="Score NAME on the lockbox")
    parser.add_argument("--force", action="store_true", help="Re-score the lockbox")
    args = parser.parse_args()

    if args.summary:
        summarize()
        return

    data = load_and_prepare_data(DATA_DIR / "train.csv")

    if args.lockbox:
        score_lockbox(args.lockbox, data, force=args.force)
        return

    folds = make_folds(data)
    print("Engineering features per fold...")
    features = {fold.name: load_fold_features(fold, CACHE_DIR) for fold in folds}

    for group in args.group or list(groups):
        for cfg in groups[group]:
            start = time.time()
            metrics = run_config(cfg, folds, features)
            log_to_ledger(cfg, metrics, time.time() - start)
            print(
                f"{cfg.name:28s} Sharpe {metrics['sharpe'].mean():6.3f}  "
                f"(buy-and-hold {metrics['sharpe_buy_hold'].mean():.3f})  "
                f"{time.time() - start:5.1f}s"
            )

    summarize()


if __name__ == "__main__":
    main()
