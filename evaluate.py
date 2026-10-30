"""
Model Evaluation

Evaluates the final configuration (FINAL_CONFIG in src/config.py) on the
walk-forward development folds and compares it with buy-and-hold over the
same days.

Metrics per fold:
- Sharpe ratio (annualized), strategy and buy-and-hold
- Competition score (penalty-adjusted Sharpe ratio)
- Maximum drawdown
- Directional accuracy and volatility forecast loss (QLIKE)

The lockbox is scored separately, once: python experiments.py --lockbox NAME

Usage:
    python evaluate.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.metrics import sharpe_ratio
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
    cfg = PipelineConfig(**FINAL_CONFIG)
    print(f"Evaluating {cfg.name} on the development folds...")

    data = load_and_prepare_data(DATA_DIR / "train.csv")
    results = []
    for fold in make_folds(data, verbose=True):
        feats = load_fold_features(fold, RESULTS_DIR / "cache")
        results.append(run_fold(prepare_config(cfg, fold, feats), fold, feats))
        print(f"  Fold {fold.name} done")

    df = pd.DataFrame([fold_metrics(r) for r in results])
    daily = daily_returns(results)

    print("\nPer-Fold Results:")
    print(
        df[
            [
                "fold",
                "n_days",
                "sharpe",
                "sharpe_buy_hold",
                "score",
                "score_buy_hold",
                "max_drawdown",
                "max_drawdown_buy_hold",
                "dir_accuracy",
            ]
        ].to_string(index=False, float_format="%.3f")
    )

    print("\n" + "-" * 80)
    print(f"  Mean Sharpe:               {df['sharpe'].mean():.3f}")
    print(f"  Mean Sharpe, buy-and-hold: {df['sharpe_buy_hold'].mean():.3f}")
    print(f"  Pooled Sharpe:             {sharpe_ratio(daily['strategy'].values):.3f}")
    print(f"  Pooled, buy-and-hold:      {sharpe_ratio(daily['buy_hold'].values):.3f}")
    print(f"  Mean Dir Accuracy:         {df['dir_accuracy'].mean():.1%}")

    results_path = PROJECT_ROOT / "evaluation_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
