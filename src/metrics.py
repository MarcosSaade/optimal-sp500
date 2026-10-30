"""
Evaluation Metrics

- Annualized Sharpe ratio and maximum drawdown
- Competition score: Sharpe ratio with volatility and return-shortfall
  penalties, reimplemented from the Hull Tactical metric description
- Probabilistic and Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2012/2014)
- Block-bootstrap confidence interval for a Sharpe ratio difference
"""

import numpy as np
from scipy import stats

from .config import *

EULER_GAMMA = 0.5772156649


def sharpe_ratio(portfolio_excess: np.ndarray, annualize: bool = True) -> float:
    """Sharpe ratio of daily excess returns (annualized by default)."""
    std = np.std(portfolio_excess)
    if len(portfolio_excess) == 0 or std == 0:
        return 0.0
    sr = np.mean(portfolio_excess) / std
    return float(sr * np.sqrt(ANNUALIZATION_FACTOR)) if annualize else float(sr)


def max_drawdown(portfolio_excess: np.ndarray) -> float:
    """Largest peak-to-trough fall of the compounded excess-return curve."""
    wealth = np.cumprod(1 + portfolio_excess)
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1 - wealth / peak))


def competition_score(
    allocations: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
) -> float:
    """
    Hull Tactical competition score (penalty-adjusted Sharpe ratio).

    The Sharpe ratio uses the geometric mean excess return. It is divided by
    a volatility penalty when the strategy's volatility exceeds 1.2x the
    market's, and by a return penalty when its mean excess return falls
    below the market's.
    """
    n = len(allocations)
    strategy_returns = risk_free_rate * (1 - allocations) + allocations * forward_returns
    strategy_excess = strategy_returns - risk_free_rate
    strategy_mean = np.prod(1 + strategy_excess) ** (1 / n) - 1
    strategy_std = np.std(strategy_returns, ddof=1)
    sharpe = strategy_mean / strategy_std * np.sqrt(ANNUALIZATION_FACTOR)

    market_excess = forward_returns - risk_free_rate
    market_mean = np.prod(1 + market_excess) ** (1 / n) - 1
    market_std = np.std(forward_returns, ddof=1)

    excess_vol = max(0.0, strategy_std / market_std - 1.2)
    return_gap = max(0.0, (market_mean - strategy_mean) * 100 * ANNUALIZATION_FACTOR)
    vol_penalty = 1 + excess_vol
    return_penalty = 1 + return_gap**2 / 100

    return float(sharpe / (vol_penalty * return_penalty))


def probabilistic_sharpe_ratio(
    portfolio_excess: np.ndarray,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Probability that the true (per-period) Sharpe ratio exceeds benchmark_sr,
    accounting for sample length, skewness and kurtosis.
    """
    sr = sharpe_ratio(portfolio_excess, annualize=False)
    t = len(portfolio_excess)
    skew = stats.skew(portfolio_excess)
    kurt = stats.kurtosis(portfolio_excess, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(t - 1) / denom))


def deflated_sharpe_ratio(
    portfolio_excess: np.ndarray,
    trial_sharpes: np.ndarray,
    benchmark_excess: np.ndarray = None,
) -> dict:
    """
    Deflated Sharpe Ratio.

    The benchmark is the Sharpe ratio expected from the best of N
    unskilled trials, given the spread of the trials' Sharpe ratios.

    Any long position in a rising market beats a Sharpe ratio of zero, so
    if benchmark_excess is given, that market's Sharpe ratio is added to the
    benchmark: the result is the probability that the strategy beats
    buy-and-hold after accounting for the trials.

    Parameters
    ----------
    portfolio_excess : np.ndarray
        Daily excess returns of the selected strategy
    trial_sharpes : np.ndarray
        Per-period (not annualized) Sharpe ratios of every trial run,
        including the selected one
    benchmark_excess : np.ndarray, optional
        Daily excess returns of buy-and-hold over the same days

    Returns
    -------
    result : dict
        dsr (probability), sr0_annual (annualized benchmark Sharpe ratio),
        n_trials
    """
    n = len(trial_sharpes)
    sd = np.std(trial_sharpes, ddof=1)
    sr0 = sd * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    )
    if benchmark_excess is not None:
        sr0 = sr0 + sharpe_ratio(benchmark_excess, annualize=False)

    return {
        "dsr": probabilistic_sharpe_ratio(portfolio_excess, benchmark_sr=sr0),
        "sr0_annual": float(sr0 * np.sqrt(ANNUALIZATION_FACTOR)),
        "n_trials": n,
    }


def sharpe_difference_ci(
    strategy_excess: np.ndarray,
    benchmark_excess: np.ndarray,
    block: int = 20,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Circular block bootstrap of the annualized Sharpe ratio difference
    (strategy minus benchmark), resampling the same days for both.

    Returns
    -------
    (low, high) : tuple
        (1 - alpha) confidence interval
    """
    rng = np.random.default_rng(seed)
    t = len(strategy_excess)
    n_blocks = int(np.ceil(t / block))
    diffs = np.empty(n_boot)

    for b in range(n_boot):
        starts = rng.integers(0, t, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:t] % t
        diffs[b] = sharpe_ratio(strategy_excess[idx]) - sharpe_ratio(
            benchmark_excess[idx]
        )

    return (
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
    )
