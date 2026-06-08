import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold

# =========================
# CREATE OUTPUT DIRECTORY
# =========================

os.makedirs("outputs/processed", exist_ok=True)

# =========================
# LOAD PHASE 3 DATA
# =========================

print("Loading Phase 3 datasets...")

train_df = pd.read_csv(
    "outputs/processed/train_phase3.csv"
)

test_df = pd.read_csv(
    "outputs/processed/test_phase3.csv"
)

print("Datasets loaded successfully!")

# =========================
# SETTINGS
# =========================

TARGET = "demand"

N_SPLITS = 5

# =========================
# INITIALIZE KFOLD
# =========================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

# =========================
# FEATURES TO CREATE
# =========================

aggregate_configs = [
    ["geohash"],
    ["geohash", "hour"],
    ["RoadType"],
    ["Weather"],
    ["RoadType", "hour"]
]

# =========================
# CREATE LEAKAGE-FREE FEATURES
# =========================

print("\nCreating leakage-free aggregate features...")

for cols in aggregate_configs:

    feature_name = (
        "agg_"
        + "_".join(cols)
    )

    print(f"\nProcessing: {feature_name}")

    # Initialize column
    train_df[feature_name] = np.nan

    # =========================
    # OOF FEATURE CREATION
    # =========================

    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(train_df)
    ):

        fold_train = train_df.iloc[train_idx]
        fold_valid = train_df.iloc[valid_idx]

        # Create mapping ONLY from fold train
        agg_map = fold_train.groupby(cols)[
            TARGET
        ].mean()

        # Apply to validation fold
        train_df.loc[
            valid_idx,
            feature_name
        ] = (
            fold_valid
            .set_index(cols)
            .index
            .map(agg_map)
        )

    # =========================
    # TEST FEATURE CREATION
    # =========================

    full_agg_map = train_df.groupby(cols)[
        TARGET
    ].mean()

    test_df[feature_name] = (
        test_df
        .set_index(cols)
        .index
        .map(full_agg_map)
    )

    # =========================
    # FILL NULLS
    # =========================

    global_mean = train_df[TARGET].mean()

    train_df[feature_name] = (
        train_df[feature_name]
        .fillna(global_mean)
    )

    test_df[feature_name] = (
        test_df[feature_name]
        .fillna(global_mean)
    )

print("\nLeakage-free features created!")

# =========================
# FINAL OVERVIEW
# =========================

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

print("\nGenerated Features:")

generated_features = [
    "agg_" + "_".join(cols)
    for cols in aggregate_configs
]

print(generated_features)

# =========================
# SAVE DATASETS
# =========================

print("\nSaving leakage-free datasets...")

train_df.to_csv(
    "outputs/processed/train_phase6.csv",
    index=False
)

test_df.to_csv(
    "outputs/processed/test_phase6.csv",
    index=False
)

print("Datasets saved successfully!")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== PHASE 6 FEATURE ENGINEERING COMPLETED ==========")

print("""
Successfully completed:

1. KFold leakage prevention
2. Out-of-fold aggregate features
3. Safe target encoding
4. Leaderboard-safe feature engineering

Generated Files:
- train_phase6.csv
- test_phase6.csv

Next Step:
-> Cross-validation model training
""")