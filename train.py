"""
Training

Fits the final configuration (FINAL_CONFIG in src/config.py) on every row of
train.csv and saves the fitted components to models/, ready to produce
allocations for new days.

Model selection and evaluation happen in experiments.py and evaluate.py;
this script does not score anything.

Usage:
    python train.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config import *
from src.pipeline import PipelineConfig, engineer_fold, fit_fold
from src.preprocessing import load_and_prepare_data, make_folds
from src.tuning import prepare_config


def main():
    cfg = PipelineConfig(**FINAL_CONFIG)
    print(f"Training {cfg.name} on all data...")

    data = load_and_prepare_data(DATA_DIR / "train.csv")
    fold = make_folds(data, lockbox="all")[0]
    feats = engineer_fold(fold, verbose=True)
    models = fit_fold(prepare_config(cfg, fold, feats), fold, feats)

    for name, model in models.items():
        if model is not None:
            model.save(MODELS_DIR / f"{name}.pkl")
            print(f"  Saved models/{name}.pkl")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
