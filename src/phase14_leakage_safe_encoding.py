import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold

print("Loading Phase 12 datasets...")

train = pd.read_csv(
    "outputs/processed/train_phase12.csv"
)

test = pd.read_csv(
    "outputs/processed/test_phase12.csv"
)

TARGET = "demand"

print("Datasets loaded!")

# =========================
# SETTINGS
# =========================

N_SPLITS = 5

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

# =========================
# TARGET ENCODING COLS
# =========================

te_cols = [
    "geohash",
    "geohash_prefix_4",
    "RoadType",
    "Weather"
]

# =========================
# GLOBAL MEAN
# =========================

global_mean = train[TARGET].mean()

print("\nCreating leakage-safe target encoding...")

# =========================
# TRAIN TARGET ENCODING
# =========================

for col in te_cols:

    print(f"\nEncoding: {col}")

    train[f"{col}_te"] = np.nan

    for train_idx, val_idx in kf.split(train):

        X_train = train.iloc[train_idx]
        X_val = train.iloc[val_idx]

        means = (
            X_train
            .groupby(col)[TARGET]
            .mean()
        )

        train.loc[
            val_idx,
            f"{col}_te"
        ] = (
            X_val[col]
            .map(means)
        )

    # Fill missing
    train[f"{col}_te"] = (
        train[f"{col}_te"]
        .fillna(global_mean)
    )

    # =========================
    # TEST TARGET ENCODING
    # =========================

    full_means = (
        train
        .groupby(col)[TARGET]
        .mean()
    )

    test[f"{col}_te"] = (
        test[col]
        .map(full_means)
    )

    test[f"{col}_te"] = (
        test[f"{col}_te"]
        .fillna(global_mean)
    )

# =========================
# SAVE
# =========================

os.makedirs(
    "outputs/processed",
    exist_ok=True
)

train.to_csv(
    "outputs/processed/train_phase14.csv",
    index=False
)

test.to_csv(
    "outputs/processed/test_phase14.csv",
    index=False
)

print("\n========== PHASE 14 COMPLETED ==========")

print("Saved:")
print("- train_phase14.csv")
print("- test_phase14.csv")