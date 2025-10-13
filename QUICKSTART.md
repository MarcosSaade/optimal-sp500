# Quick Start Guide

## Setup (5 minutes)

### 1. Create Virtual Environment

```bash
cd build
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Data

Place your `train.csv` file in the `data/` directory.

## Training (2-4 hours)

### Option A: Train Everything

```bash
python train.py --stage all
```

This will:
1. Preprocess data (create 5-fold purged CV splits)
2. Train return prediction models (5 folds)
3. Train volatility prediction models (5 folds)
4. Train meta-labeling classifiers (5 folds)

### Option B: Train Step-by-Step

```bash
# Step 1: Preprocess data
python train.py --stage preprocess

# Step 2: Train return models
python train.py --stage returns

# Step 3: Train volatility models  
python train.py --stage volatility

# Step 4: Train meta-labeling
python train.py --stage meta
```

### Option C: Train Specific Folds

```bash
# Train only folds 1 and 2
python train.py --stage returns --folds 1,2
```

## Evaluation

### Evaluate All Folds

```bash
python evaluate.py
```

### Evaluate Without Meta-Labeling

```bash
python evaluate.py --no-meta
```

### Evaluate Single Fold

```bash
python evaluate.py --fold 3
```

## Expected Output

After training and evaluation, the pipeline will write per-fold and aggregate evaluation metrics to `evaluation_results.csv`. The Quick Start intentionally omits fixed numeric examples; consult the generated CSV or the console output for dataset-specific results.

## Directory Structure After Training

```
build/
├── data/
│   ├── train.csv              # Your data
│   └── processed/             # Generated
│       ├── train_fold*.csv
│       └── val_fold*.csv
├── models/                    # Generated
│   ├── returns/
│   │   ├── fold_*.pkl
│   │   ├── engineer_fold_*.pkl
│   │   └── features_fold_*.csv
│   ├── volatility/
│   │   └── fold_*.pkl
│   └── meta/
│       └── fold_*.pkl
├── evaluation_results.csv     # Generated
└── ...
```

## Retraining Models

### Retrain Everything

```bash
python train.py --stage all --force
```

### Retrain Only Returns (Lightweight)

```bash
python train.py --stage returns
```

**Note**: If you retrain returns, you should also retrain volatility and meta-labeling since they depend on return predictions.

### Proper Retraining After Return Model Changes

```bash
# Retrain return models
python train.py --stage returns

# Retrain dependent models
python train.py --stage volatility
python train.py --stage meta

# Evaluate
python evaluate.py
```

## Exploratory Data Analysis

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Troubleshooting

### "Data file not found"

Make sure `data/train.csv` exists. Check `data/README.md` for format details.

### "Model must be fitted before prediction"

You need to train models first. Run `python train.py --stage all`.

### Memory Issues

- Reduce `MAX_FEATURES` in `src/config.py`
- Train one fold at a time: `python train.py --stage returns --folds 1`

### Poor Performance

- Check data quality (missing values, outliers)
- Verify preprocessing completed successfully
- Review `notebooks/eda.ipynb` for data insights

## Next Steps

1. **Analyze Results**: Review `evaluation_results.csv`
2. **Tune Hyperparameters**: Modify `src/config.py`
3. **Add Features**: Extend `src/features.py`
4. **Try Different Allocators**: Modify `src/allocation.py`

## Getting Help

- Check main `README.md` for detailed methodology
- Review code documentation in each module
- Examine `notebooks/eda.ipynb` for data insights
