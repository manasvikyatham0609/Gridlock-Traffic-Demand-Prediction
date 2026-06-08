import pandas as pd
import numpy as np
import os

# =========================
# CREATE OUTPUT DIRECTORY
# =========================

os.makedirs("outputs/processed", exist_ok=True)

# =========================
# LOAD PHASE 2 DATA
# =========================

print("Loading Phase 2 processed datasets...")

train_df = pd.read_csv("outputs/processed/train_phase2.csv")
test_df = pd.read_csv("outputs/processed/test_phase2.csv")

print("Datasets loaded successfully!")

# =========================
# BASIC INFO
# =========================

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =========================
# HANDLE MISSING VALUES
# =========================

print("\nHandling missing values...")

# Numerical columns
numerical_cols = train_df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Remove target from numerical filling
if "demand" in numerical_cols:
    numerical_cols.remove("demand")

# Fill numerical columns with median
for col in numerical_cols:

    median_value = train_df[col].median()

    train_df[col] = train_df[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

# Categorical columns
categorical_cols = train_df.select_dtypes(
    include=["object"]
).columns.tolist()

for col in categorical_cols:

    mode_value = train_df[col].mode()[0]

    train_df[col] = train_df[col].fillna(mode_value)
    test_df[col] = test_df[col].fillna(mode_value)

print("Missing values handled successfully!")

# =========================
# REMOVE USELESS FEATURES
# =========================

print("\nRemoving unnecessary columns...")

columns_to_remove = []

# These were artificial due to time-only timestamp parsing
possible_bad_cols = [
    "month",
    "weekofyear",
    "dayofmonth",
    "dayofweek"
]

for col in possible_bad_cols:
    if col in train_df.columns:
        columns_to_remove.append(col)

train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

print("Removed columns:", columns_to_remove)

# =========================
# LOG TRANSFORM TARGET
# =========================

print("\nApplying log1p transform to demand...")

train_df["demand"] = np.log1p(train_df["demand"])

print("Log transform completed!")

# =========================
# CYCLIC TIME FEATURES
# =========================

print("\nCreating cyclic time features...")

for df in [train_df, test_df]:

    # Hour cyclic encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Minute cyclic encoding
    df["minute_sin"] = np.sin(
        2 * np.pi * df["minute"] / 60
    )

    df["minute_cos"] = np.cos(
        2 * np.pi * df["minute"] / 60
    )

print("Cyclic features created!")

# =========================
# GEOHASH FEATURE ENGINEERING
# =========================

print("\nCreating geohash prefix features...")

for prefix_length in [2, 3, 4]:

    feature_name = f"geohash_prefix_{prefix_length}"

    train_df[feature_name] = (
        train_df["geohash"]
        .astype(str)
        .str[:prefix_length]
    )

    test_df[feature_name] = (
        test_df["geohash"]
        .astype(str)
        .str[:prefix_length]
    )

print("Geohash prefixes created!")

# =========================
# INTERACTION FEATURES
# =========================

print("\nCreating interaction features...")

for df in [train_df, test_df]:

    # Road capacity feature
    df["lane_vehicle_interaction"] = (
        df["NumberofLanes"]
        *
        (df["LargeVehicles"] == "Allowed").astype(int)
    )

    # Temperature x lanes
    df["temp_lane_interaction"] = (
        df["Temperature"]
        *
        df["NumberofLanes"]
    )

print("Interaction features created!")

# =========================
# AGGREGATED STATISTICAL FEATURES
# =========================

print("\nCreating aggregate statistical features...")

# Average demand by hour
hour_demand_map = train_df.groupby("hour")[
    "demand"
].mean()

train_df["avg_demand_by_hour"] = (
    train_df["hour"].map(hour_demand_map)
)

test_df["avg_demand_by_hour"] = (
    test_df["hour"].map(hour_demand_map)
)

# Average demand by RoadType
road_demand_map = train_df.groupby("RoadType")[
    "demand"
].mean()

train_df["avg_demand_by_road"] = (
    train_df["RoadType"].map(road_demand_map)
)

test_df["avg_demand_by_road"] = (
    test_df["RoadType"].map(road_demand_map)
)

# Average demand by Weather
weather_demand_map = train_df.groupby("Weather")[
    "demand"
].mean()

train_df["avg_demand_by_weather"] = (
    train_df["Weather"].map(weather_demand_map)
)

test_df["avg_demand_by_weather"] = (
    test_df["Weather"].map(weather_demand_map)
)

print("Aggregate features created!")

# =========================
# FEATURE TYPE CHECK
# =========================

print("\nFinal feature overview:")

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

# =========================
# SAVE PHASE 3 DATA
# =========================

print("\nSaving Phase 3 datasets...")

train_df.to_csv(
    "outputs/processed/train_phase3.csv",
    index=False
)

test_df.to_csv(
    "outputs/processed/test_phase3.csv",
    index=False
)

print("Phase 3 datasets saved successfully!")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== PHASE 3 COMPLETED ==========")

print("""
Successfully completed:

1. Missing value handling
2. Feature cleanup
3. Log target transformation
4. Cyclic time encoding
5. Geohash feature engineering
6. Interaction feature creation
7. Aggregate statistical features
8. ML-ready dataset generation

Generated Files:
- train_phase3.csv
- test_phase3.csv

Next Phase:
-> Baseline Model Training
""")