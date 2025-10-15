# Quick Start Guide

## Setup (5 minutes)

### 1. Create Virtual Environment

```bash
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

After training and evaluation, you will see:

- Per-fold performance metrics printed to console
- Aggregate statistics across all folds
- Detailed results saved to `evaluation_results.csv`

The specific metrics will vary based on your dataset and hyperparameters.

## Directory Structure After Training

```
.
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
