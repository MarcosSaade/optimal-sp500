# Data Directory

The competition rules don't allow redistributing the data. Download `train.csv` from the [competition data page](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data) and place it here:

```
data/
└── train.csv
```

## Format

- `date_id`: trading-day index (dates are hidden by the competition)
- `forward_returns`: next-day S&P 500 return
- `risk_free_rate`: risk-free rate for the day
- `market_forward_excess_returns`: competition target, the forward excess return minus its rolling 5-year mean
- Feature columns, anonymized by family: `D*` (binary), `E*`, `I*`, `M*`, `P*`, `S*`, `V*`

`test.csv` from the competition is 10 rows that also appear in `train.csv`, so it is not needed.

Nothing is written to this directory: folds are built in memory, and engineered features are cached in `results/cache/`.
