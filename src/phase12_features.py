import pandas as pd
import numpy as np
import os

print("Loading Phase 6 datasets...")

train = pd.read_csv("outputs/processed/train_phase6.csv")
test = pd.read_csv("outputs/processed/test_phase6.csv")

TARGET = "demand"

print("Datasets loaded!")

# =========================
# COMBINE TRAIN + TEST
# =========================

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

full = pd.concat([train, test], axis=0).reset_index(drop=True)

# =========================
# TIMESTAMP FEATURES
# =========================

print("\nCreating advanced time features...")

full["timestamp"] = pd.to_datetime(full["timestamp"])

full["hour"] = full["timestamp"].dt.hour
full["day"] = full["timestamp"].dt.day
full["month"] = full["timestamp"].dt.month
full["weekday"] = full["timestamp"].dt.weekday

# Rush hour
full["is_rush_hour"] = (
    ((full["hour"] >= 7) & (full["hour"] <= 10)) |
    ((full["hour"] >= 17) & (full["hour"] <= 20))
).astype(int)

# Night traffic
full["is_night"] = (
    (full["hour"] >= 22) |
    (full["hour"] <= 5)
).astype(int)

# Weekend
full["is_weekend"] = (
    full["weekday"] >= 5
).astype(int)

# =========================
# INTERACTION FEATURES
# =========================

print("Creating interaction features...")

# Temperature interaction
if "Temperature" in full.columns:

    full["temp_hour_interaction"] = (
        full["Temperature"] * full["hour"]
    )

# Humidity interaction
if "Humidity" in full.columns:

    full["humidity_hour_interaction"] = (
        full["Humidity"] * full["hour"]
    )
# =========================
# GROUP STAT FEATURES
# =========================

print("Creating grouped statistical features...")

group_features = [
    ["geohash", "hour"],
    ["geohash_prefix_4", "hour"],
    ["RoadType", "hour"],
    ["Weather", "hour"],
    ["weekday", "hour"]
]

train_only = full[full["is_train"] == 1]

for cols in group_features:

    feature_name = "_".join(cols)

    stats = (
        train_only
        .groupby(cols)[TARGET]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    stats.columns = (
        cols +
        [
            f"{feature_name}_mean",
            f"{feature_name}_median",
            f"{feature_name}_std"
        ]
    )

    full = full.merge(
        stats,
        on=cols,
        how="left"
    )

# =========================
# COUNT ENCODING
# =========================

print("Creating count encoding features...")

count_cols = [
    "geohash",
    "geohash_prefix_4",
    "RoadType",
    "Weather"
]

for col in count_cols:

    counts = full[col].value_counts()

    full[f"{col}_count"] = (
        full[col]
        .map(counts)
    )

# =========================
# TARGET ENCODING
# =========================

print("Creating target encoding features...")

target_encode_cols = [
    "geohash",
    "geohash_prefix_4",
    "RoadType",
    "Weather"
]

for col in target_encode_cols:

    means = (
        train_only
        .groupby(col)[TARGET]
        .mean()
    )

    full[f"{col}_target_mean"] = (
        full[col]
        .map(means)
    )

# =========================
# FILL MISSING VALUES
# =========================

print("Filling missing values...")

for col in full.columns:

    if full[col].dtype != "object":

        full[col] = full[col].fillna(
            full[col].median()
        )

# =========================
# SPLIT BACK
# =========================

train_final = full[
    full["is_train"] == 1
].drop(columns=["is_train"])

test_final = full[
    full["is_train"] == 0
].drop(columns=["is_train", TARGET])

# =========================
# SAVE
# =========================

os.makedirs("outputs/processed", exist_ok=True)

train_final.to_csv(
    "outputs/processed/train_phase12.csv",
    index=False
)

test_final.to_csv(
    "outputs/processed/test_phase12.csv",
    index=False
)

print("\n========== PHASE 12 COMPLETED ==========")

print("Saved:")
print("- train_phase12.csv")
print("- test_phase12.csv")