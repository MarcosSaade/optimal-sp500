"""
Model Evaluation

Evaluates the complete pipeline across all CV folds.

Metrics:
- Sharpe Ratio (primary)
- Directional accuracy
- Return statistics
- Meta-labeling improvement
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.returns import ReturnPredictor
from src.volatility import VolatilityPredictor
from src.meta_labeling import MetaLabelPipeline
from src.allocation import RegimeDependentAllocator
from src.features import FeatureEngineer


def compute_sharpe_ratio(
    forward_returns: np.ndarray,
    allocations: np.ndarray,
    risk_free_rate: np.ndarray,
) -> float:
    """
    Compute Sharpe ratio (annualized).

    Formula: sqrt(252) * E[alloc * (returns - rf)] / std[alloc * (returns - rf)]
    """
    excess_returns = forward_returns - risk_free_rate
    portfolio_returns = allocations * excess_returns

    if len(portfolio_returns) == 0 or np.std(portfolio_returns) == 0:
        return 0.0

    sharpe = (
        np.mean(portfolio_returns)
        / np.std(portfolio_returns)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )
    return sharpe


def evaluate_fold(
    fold: int,
    use_meta_labeling: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a single fold.

    Parameters
    ----------
    fold : int
        Fold number
    use_meta_labeling : bool
        Whether to use meta-labeling
    verbose : bool
        Print detailed results

    Returns
    -------
    results : dict
        Evaluation metrics
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}")
        print(f"{'=' * 80}")

    # Load data
    if verbose:
        print("\n1. Loading data...")
    val_df = pd.read_csv(PROCESSED_DIR / f"val_fold{fold}.csv")

    # Load models
    if verbose:
        print("2. Loading models...")
    return_model = ReturnPredictor.load(MODELS_DIR / "returns" / f"fold_{fold}.pkl")
    vol_model = VolatilityPredictor.load(MODELS_DIR / "volatility" / f"fold_{fold}.pkl")
    engineer = FeatureEngineer.load(
        MODELS_DIR / "returns" / f"engineer_fold_{fold}.pkl"
    )

    selected_features = pd.read_csv(
        MODELS_DIR / "returns" / f"features_fold_{fold}.csv"
    )["feature"].tolist()

    # Load meta-labeling if requested
    meta_model = None
    if use_meta_labeling:
        meta_path = MODELS_DIR / "meta" / f"fold_{fold}.pkl"
        if meta_path.exists():
            meta_model = MetaLabelPipeline.load(meta_path)
        else:
            if verbose:
                print("   ⚠ Meta-labeling model not found, skipping")
            use_meta_labeling = False

    # Engineer features
    if verbose:
        print("3. Engineering features...")
    val_features = engineer.transform(val_df)
    X_val = val_features[selected_features].fillna(0)

    # Predict returns
    if verbose:
        print("4. Predicting returns...")
    mu = return_model.predict(X_val)

    # Predict volatility
    if verbose:
        print("5. Predicting volatility...")
    sigma = vol_model.predict(val_df)

    # Allocate positions
    if verbose:
        print("6. Allocating positions...")
    allocator = RegimeDependentAllocator()
    allocations = allocator.allocate(mu, sigma, data=val_features)

    # Apply meta-labeling if available
    if use_meta_labeling and meta_model is not None:
        if verbose:
            print("7. Applying meta-labeling...")
        meta_proba = meta_model.predict_proba(X_val, mu)
        allocations_meta = meta_model.scale_positions(allocations, meta_proba)
    else:
        allocations_meta = allocations
        meta_proba = np.ones_like(allocations)

    # Get actual values
    forward_returns = val_df["forward_returns"].values[: len(allocations)]
    risk_free = val_df["risk_free_rate"].values[: len(allocations)]
    actual_excess = val_df[TARGET_COL].values[: len(allocations)]

    # Compute metrics
    if verbose:
        print("8. Computing metrics...")

    # Baseline Sharpe (no meta-labeling)
    sharpe_base = compute_sharpe_ratio(forward_returns, allocations, risk_free)

    # Meta-labeling Sharpe
    sharpe_meta = compute_sharpe_ratio(forward_returns, allocations_meta, risk_free)

    # Directional accuracy
    pred_direction = np.sign(mu)
    actual_direction = np.sign(actual_excess)
    dir_accuracy = np.mean(pred_direction == actual_direction)

    # Portfolio statistics
    excess_returns = forward_returns - risk_free
    portfolio_returns_base = allocations * excess_returns
    portfolio_returns_meta = allocations_meta * excess_returns

    results = {
        "fold": fold,
        "n_samples": len(allocations),
        "sharpe_base": float(sharpe_base),
        "sharpe_meta": float(sharpe_meta) if use_meta_labeling else None,
        "improvement_pct": float((sharpe_meta / sharpe_base - 1) * 100)
        if use_meta_labeling and sharpe_base != 0
        else None,
        "dir_accuracy": float(dir_accuracy),
        "mean_return_base": float(np.mean(portfolio_returns_base)),
        "std_return_base": float(np.std(portfolio_returns_base)),
        "mean_return_meta": float(np.mean(portfolio_returns_meta))
        if use_meta_labeling
        else None,
        "std_return_meta": float(np.std(portfolio_returns_meta))
        if use_meta_labeling
        else None,
        "mean_meta_proba": float(np.mean(meta_proba)) if use_meta_labeling else None,
    }

    if verbose:
        print(f"\n📊 Results:")
        print(f"   Samples: {results['n_samples']}")
        print(f"   Directional Accuracy: {results['dir_accuracy']:.1%}")
        print(f"\n   Base Sharpe: {results['sharpe_base']:.3f}")
        if use_meta_labeling:
            print(f"   Meta Sharpe: {results['sharpe_meta']:.3f}")
            print(f"   Improvement: {results['improvement_pct']:+.1f}%")
            print(f"   Mean Meta Probability: {results['mean_meta_proba']:.3f}")

    return results


def evaluate_all_folds(use_meta_labeling: bool = True):
    """
    Evaluate all folds and print summary.

    Parameters
    ----------
    use_meta_labeling : bool
        Whether to use meta-labeling
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Meta-labeling: {'ENABLED' if use_meta_labeling else 'DISABLED'}")

    all_results = []

    for fold in range(1, CV_N_SPLITS + 1):
        results = evaluate_fold(fold, use_meta_labeling=use_meta_labeling, verbose=True)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    print("\nPer-Fold Results:")
    print(
        df[
            [
                "fold",
                "n_samples",
                "sharpe_base",
                "sharpe_meta",
                "improvement_pct",
                "dir_accuracy",
            ]
        ].to_string(index=False)
    )

    # Aggregate statistics
    print("\n" + "-" * 80)
    print("Aggregate Statistics:")
    print(
        f"  Mean Base Sharpe:   {df['sharpe_base'].mean():.3f} ± {df['sharpe_base'].std():.3f}"
    )

    if use_meta_labeling:
        print(
            f"  Mean Meta Sharpe:   {df['sharpe_meta'].mean():.3f} ± {df['sharpe_meta'].std():.3f}"
        )
        print(f"  Mean Improvement:   {df['improvement_pct'].mean():+.1f}%")
        print(
            f"  Best Fold:          Fold {df.loc[df['sharpe_meta'].idxmax(), 'fold']} (Sharpe {df['sharpe_meta'].max():.3f})"
        )
        print(
            f"  Worst Fold:         Fold {df.loc[df['sharpe_meta'].idxmin(), 'fold']} (Sharpe {df['sharpe_meta'].min():.3f})"
        )
    else:
        print(
            f"  Best Fold:          Fold {df.loc[df['sharpe_base'].idxmax(), 'fold']} (Sharpe {df['sharpe_base'].max():.3f})"
        )
        print(
            f"  Worst Fold:         Fold {df.loc[df['sharpe_base'].idxmin(), 'fold']} (Sharpe {df['sharpe_base'].min():.3f})"
        )

    print(f"  Mean Dir Accuracy:  {df['dir_accuracy'].mean():.1%}")

    # Save results
    results_path = PROJECT_ROOT / "evaluation_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--no-meta", action="store_true", help="Disable meta-labeling")
    parser.add_argument(
        "--fold", type=int, default=None, help="Evaluate single fold only"
    )

    args = parser.parse_args()

    use_meta = not args.no_meta

    if args.fold is not None:
        # Evaluate single fold
        evaluate_fold(args.fold, use_meta_labeling=use_meta, verbose=True)
    else:
        # Evaluate all folds
        evaluate_all_folds(use_meta_labeling=use_meta)


if __name__ == "__main__":
    main()
