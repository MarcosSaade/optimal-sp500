# Data Directory

Place your training data file here:

```
data/
└── train.csv
```

## Expected Format

The `train.csv` file should contain:
- `date_id`: Monotonically increasing date identifier
- `risk_free_rate`: Risk-free rate for each period
- `market_forward_excess_returns`: Target variable (1-day forward excess returns)
- Feature columns: Various market indicators (E*, I*, M*, P*, S*, V*, D*)

## Processed Data

After running preprocessing, this directory will contain:

```
data/
├── train.csv              # Your raw data (you provide this)
└── processed/             # Auto-generated
    ├── train_fold1.csv
    ├── val_fold1.csv
    ├── train_fold2.csv
    ├── val_fold2.csv
    ...
    └── preprocessor_fold*.pkl
```

## Getting Started

1. Place your `train.csv` file in this directory
2. Run: `python train.py --stage preprocess`
3. Processed folds will be created automatically
