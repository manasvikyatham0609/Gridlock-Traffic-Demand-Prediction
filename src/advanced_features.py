import pandas as pd
import numpy as np
import pygeohash as pgh
import os

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
# GEOHASH DECODING
# =========================

print("\nDecoding geohash coordinates...")

def decode_geohash(gh):

    try:
        lat, lon = pgh.decode(gh)
        return pd.Series([lat, lon])

    except:
        return pd.Series([np.nan, np.nan])

# Decode train
train_df[["latitude", "longitude"]] = (
    train_df["geohash"]
    .apply(decode_geohash)
)

# Decode test
test_df[["latitude", "longitude"]] = (
    test_df["geohash"]
    .apply(decode_geohash)
)

print("Geohash decoding completed!")

# =========================
# TRAFFIC PEAK FEATURES
# =========================

print("\nCreating traffic peak features...")

def get_time_period(hour):

    if 6 <= hour < 10:
        return "morning_peak"

    elif 10 <= hour < 16:
        return "midday"

    elif 16 <= hour < 21:
        return "evening_peak"

    else:
        return "night"

for df in [train_df, test_df]:

    df["time_period"] = (
        df["hour"]
        .apply(get_time_period)
    )

print("Traffic peak features created!")

# =========================
# AGGREGATE FEATURES
# =========================

print("\nCreating advanced aggregate features...")

# =========================
# GEOHASH AVG DEMAND
# =========================

geo_demand_map = train_df.groupby(
    "geohash"
)["demand"].mean()

train_df["avg_demand_by_geohash"] = (
    train_df["geohash"]
    .map(geo_demand_map)
)

test_df["avg_demand_by_geohash"] = (
    test_df["geohash"]
    .map(geo_demand_map)
)

# =========================
# GEOHASH + HOUR DEMAND
# =========================

geo_hour_map = train_df.groupby(
    ["geohash", "hour"]
)["demand"].mean()

train_df["avg_demand_geo_hour"] = (
    train_df.set_index(
        ["geohash", "hour"]
    ).index.map(geo_hour_map)
)

test_df["avg_demand_geo_hour"] = (
    test_df.set_index(
        ["geohash", "hour"]
    ).index.map(geo_hour_map)
)

# =========================
# WEATHER + HOUR DEMAND
# =========================

weather_hour_map = train_df.groupby(
    ["Weather", "hour"]
)["demand"].mean()

train_df["avg_demand_weather_hour"] = (
    train_df.set_index(
        ["Weather", "hour"]
    ).index.map(weather_hour_map)
)

test_df["avg_demand_weather_hour"] = (
    test_df.set_index(
        ["Weather", "hour"]
    ).index.map(weather_hour_map)
)

# =========================
# FILL ANY NEW NULLS
# =========================

print("\nHandling newly created null values...")

for col in train_df.columns:

    # Skip target column
    if col == "demand":
        continue

    # Only numerical columns
    if train_df[col].dtype != "object":

        median_value = train_df[col].median()

        train_df[col] = train_df[col].fillna(
            median_value
        )

        # Fill only if column exists in test
        if col in test_df.columns:

            test_df[col] = test_df[col].fillna(
                median_value
            )

print("Null handling completed!")

# =========================
# FINAL OVERVIEW
# =========================

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

print("\nNew Features Added:")
print([
    "latitude",
    "longitude",
    "time_period",
    "avg_demand_by_geohash",
    "avg_demand_geo_hour",
    "avg_demand_weather_hour"
])

# =========================
# SAVE DATASETS
# =========================

print("\nSaving advanced datasets...")

train_df.to_csv(
    "outputs/processed/train_phase5.csv",
    index=False
)

test_df.to_csv(
    "outputs/processed/test_phase5.csv",
    index=False
)

print("Advanced datasets saved successfully!")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== ADVANCED FEATURE ENGINEERING COMPLETED ==========")

print("""
Successfully completed:

1. Geohash decoding
2. Latitude/Longitude extraction
3. Traffic peak features
4. Advanced aggregate features
5. Null handling

Generated Files:
- train_phase5.csv
- test_phase5.csv

Next Step:
-> Advanced model training
""")