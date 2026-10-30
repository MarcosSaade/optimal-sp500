"""
Result Figures

Reads results/ (written by experiments.py) and saves figures to
results/figures/:

1. sharpe_by_fold.png    strategy vs buy-and-hold Sharpe ratio per fold
2. growth.png            growth of $1, development folds then lockbox
3. experiments.png       every configuration tried vs buy-and-hold

Usage:
    python plots.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.metrics import deflated_sharpe_ratio, sharpe_ratio

FIGURES_DIR = RESULTS_DIR / "figures"

# Reference data-viz palette (categorical slots 1 and 2, light mode)
STRATEGY = "#2a78d6"
BUY_HOLD = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#a3a29b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "font.size": 10.5,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_daily(name: str) -> pd.DataFrame:
    """Development-fold daily returns of a configuration, plus the lockbox."""
    dev = pd.read_csv(RESULTS_DIR / "returns" / f"{name}.csv")
    dev["fold"] = dev["fold"].astype(str)
    lockbox_path = RESULTS_DIR / "lockbox_daily.csv"
    if lockbox_path.exists():
        lockbox = pd.read_csv(lockbox_path)
        lockbox["fold"] = "lockbox"
        dev = pd.concat([dev, lockbox], ignore_index=True)
    return dev


def sharpe_by_fold(daily: pd.DataFrame, path: Path):
    folds = list(dict.fromkeys(daily["fold"]))
    strat = [sharpe_ratio(daily.loc[daily.fold == f, "strategy"].values) for f in folds]
    bh = [sharpe_ratio(daily.loc[daily.fold == f, "buy_hold"].values) for f in folds]
    labels = [f"Fold {f}" if f != "lockbox" else "Lockbox" for f in folds]

    x = np.arange(len(folds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2 - 0.01, strat, width, color=STRATEGY, label="Strategy")
    ax.bar(x + width / 2 + 0.01, bh, width, color=BUY_HOLD, label="Buy-and-hold")
    for xi, s, b in zip(x, strat, bh):
        ax.text(xi - width / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)
        ax.text(xi + width / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    if "lockbox" in folds:
        ax.axvline(len(folds) - 1.5, color=MUTED, linewidth=1, linestyle="--")

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_ylim(min(0, *strat, *bh) * 1.2, max(*strat, *bh) * 1.25)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("Sharpe ratio per evaluation window")
    ax.legend(loc="upper left", ncols=2)
    fig.savefig(path)
    plt.close(fig)


def growth(daily: pd.DataFrame, path: Path):
    t = np.arange(len(daily))
    strat = np.cumprod(1 + daily["strategy"].values)
    bh = np.cumprod(1 + daily["buy_hold"].values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, bh, color=BUY_HOLD, linewidth=2, label="Buy-and-hold")
    ax.plot(t, strat, color=STRATEGY, linewidth=2, label="Strategy")

    # Fold boundaries
    starts = daily.index[daily["fold"] != daily["fold"].shift()].tolist()
    for start in starts[1:]:
        ax.axvline(start, color=GRID, linewidth=1)
    if "lockbox" in set(daily["fold"]):
        lock_start = daily.index[daily["fold"] == "lockbox"][0]
        ax.axvspan(lock_start, len(daily), color=GRID, alpha=0.5, linewidth=0)
        ax.text(
            lock_start + 15, 0.03, "Lockbox", va="bottom",
            color=TEXT_SECONDARY, transform=ax.get_xaxis_transform(),
        )

    ax.set_yscale("log")
    ticks = [t for t in [0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50, 100] if min(strat.min(), bh.min()) * 0.9 <= t <= max(strat.max(), bh.max()) * 1.1]
    ax.set_yticks(ticks, [f"${t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Trading days (out-of-sample, in order)")
    ax.set_ylabel("Growth of $1 above cash (log scale)")
    ax.set_title("Growth of $1, out-of-sample")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def experiments(final_name: str, path: Path):
    ledger = pd.read_csv(RESULTS_DIR / "experiments.csv").sort_values("pooled_sharpe")
    bh = ledger["pooled_sharpe_buy_hold"].iloc[0]

    daily = pd.read_csv(RESULTS_DIR / "returns" / f"{final_name}.csv")
    dsr = deflated_sharpe_ratio(
        daily["strategy"].values,
        ledger["pooled_sr_daily"].values,
        benchmark_excess=daily["buy_hold"].values,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(ledger))
    colors = [STRATEGY if n == final_name else MUTED for n in ledger["name"]]
    ax.scatter(ledger["pooled_sharpe"], y, s=36, color=colors, zorder=3)
    ax.axvline(bh, color=BUY_HOLD, linewidth=2)
    ax.axvline(dsr["sr0_annual"], color=TEXT_SECONDARY, linewidth=1, linestyle="--")

    ax.text(bh, 0, "Buy-and-hold ", color=TEXT, va="bottom", ha="right", fontsize=9)
    ax.text(
        dsr["sr0_annual"], 0, "Needed to beat buy-and-hold\nafter deflation ",
        color=TEXT, va="bottom", ha="right", fontsize=9,
    )
    final_row = ledger.reset_index(drop=True).index[ledger["name"].values == final_name][0]
    ax.annotate(
        "Selected",
        (ledger["pooled_sharpe"].values[final_row], final_row),
        xytext=(6, -20),
        textcoords="offset points",
        va="center",
        ha="right",
        fontsize=9,
        color=TEXT,
    )

    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pooled Sharpe ratio on the development folds")
    ax.set_title(f"All {len(ledger)} configurations tried")
    fig.savefig(path)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    name = FINAL_CONFIG["name"]
    daily = load_daily(name)

    sharpe_by_fold(daily, FIGURES_DIR / "sharpe_by_fold.png")
    growth(daily, FIGURES_DIR / "growth.png")
    experiments(name, FIGURES_DIR / "experiments.png")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
