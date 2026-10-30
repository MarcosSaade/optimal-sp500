# Quick Start Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `train.csv` from the [competition data page](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data) and place it in `data/`. The competition's `test.csv` is 10 rows that also appear in `train.csv`, so it is not used.

## Reproduce the results

```bash
python experiments.py            # all configurations on the development folds (~30 min, mostly AutoML)
python tune.py --vol-model lgbm  # nested Optuna search (~6 min)
python experiments.py --summary  # ledger, best configuration, Deflated Sharpe Ratio
python plots.py                  # figures in results/figures/
```

Faster subsets:

```bash
python experiments.py --group grid   # return models x volatility models (~3 min)
python experiments.py --group flat   # same, one tilt size for all regimes
python experiments.py --group meta   # with meta-labeling
python experiments.py --group automl # FLAML return model
```

## Evaluate and train the final configuration

The final configuration is `FINAL_CONFIG` in `src/config.py`.

```bash
python evaluate.py   # per-fold Sharpe ratio, competition score and drawdown vs buy-and-hold
python train.py      # fit on all data, save to models/
```

## The lockbox

The last 500 trading days are held out from all development work. They are scored once:

```bash
python experiments.py --lockbox tuned_lgbm+lgbm
```

The result is saved to `results/lockbox.csv`, and the script refuses to score the lockbox again unless you pass `--force`. Re-scoring after looking at the result would turn the lockbox into another development fold.

## Outputs

```
results/
├── experiments.csv      # one row per configuration tried
├── lockbox.csv          # the one lockbox score
├── lockbox_daily.csv    # lockbox daily allocations and returns
├── figures/             # plots.py output
├── returns/             # daily out-of-sample returns per configuration (generated)
└── cache/               # engineered features per fold (generated)
evaluation_results.csv   # evaluate.py output
models/                  # train.py output
```

## Exploratory data analysis

`notebooks/eda.ipynb` covers missingness, outliers and the late-starting features that motivated the preprocessing choices.
