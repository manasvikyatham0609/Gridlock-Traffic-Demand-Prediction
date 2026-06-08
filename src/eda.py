import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CREATE OUTPUT DIRECTORIES
# =========================

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

# =========================
# LOAD DATASETS
# =========================

print("Loading datasets...")

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

print("\nDatasets Loaded Successfully!")

# =========================
# BASIC SHAPE
# =========================

print("\n========== DATASET SHAPES ==========")
print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =========================
# DISPLAY FIRST FEW ROWS
# =========================

print("\n========== TRAIN HEAD ==========")
print(train_df.head())

# =========================
# DATASET INFO
# =========================

print("\n========== TRAIN INFO ==========")
print(train_df.info())

# =========================
# MISSING VALUES
# =========================

print("\n========== MISSING VALUES ==========")

missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]

print(missing_values)

print("\n========== MISSING VALUE PERCENTAGE ==========")

missing_percentage = (
    train_df.isnull().sum() / len(train_df)
) * 100

missing_percentage = missing_percentage[
    missing_percentage > 0
]

print(missing_percentage)

# =========================
# STATISTICAL SUMMARY
# =========================

print("\n========== NUMERICAL SUMMARY ==========")
print(train_df.describe())

# =========================
# TARGET DISTRIBUTION
# =========================

print("\n========== DEMAND DISTRIBUTION ==========")

plt.figure(figsize=(10, 5))

train_df["demand"].hist(bins=50)

plt.title("Demand Distribution")
plt.xlabel("Demand")
plt.ylabel("Frequency")

plt.savefig("outputs/plots/demand_distribution.png")
plt.close()

print("Saved: demand_distribution.png")

# =========================
# DEMAND SKEWNESS
# =========================

print("\n========== DEMAND SKEWNESS ==========")

skewness = train_df["demand"].skew()

print("Skewness:", skewness)

if skewness > 1:
    print("Demand is highly skewed.")
    print("You may later apply log1p transformation.")
else:
    print("Demand distribution is reasonably normal.")

# =========================
# IDENTIFY CATEGORICAL COLUMNS
# =========================

print("\n========== CATEGORICAL COLUMNS ==========")

categorical_cols = train_df.select_dtypes(
    include="object"
).columns.tolist()

print(categorical_cols)

# =========================
# UNIQUE VALUES IN CATEGORICALS
# =========================

print("\n========== CATEGORICAL VALUE COUNTS ==========")

for col in categorical_cols:

    print(f"\n--- {col} ---")

    print(train_df[col].value_counts())

# =========================
# IDENTIFY NUMERICAL COLUMNS
# =========================

print("\n========== NUMERICAL COLUMNS ==========")

numerical_cols = train_df.select_dtypes(
    exclude="object"
).columns.tolist()

print(numerical_cols)

# =========================
# CORRELATION MATRIX
# =========================

print("\n========== CORRELATION MATRIX ==========")

corr_matrix = train_df[numerical_cols].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.savefig("outputs/plots/correlation_matrix.png")
plt.close()

print("Saved: correlation_matrix.png")

# =========================
# CONVERT TIMESTAMP
# =========================

print("\n========== TIMESTAMP CONVERSION ==========")

train_df["timestamp"] = pd.to_datetime(
    train_df["timestamp"],
    format="%H:%M"
)

test_df["timestamp"] = pd.to_datetime(
    test_df["timestamp"],
    format="%H:%M"
)

print("Timestamp conversion completed.")

# =========================
# EXTRACT TIME FEATURES
# =========================

print("\n========== EXTRACTING TIME FEATURES ==========")

for df in [train_df, test_df]:

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofmonth"] = df["timestamp"].dt.day
    df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

print("Time feature extraction completed.")

# =========================
# HOURLY DEMAND ANALYSIS
# =========================

print("\n========== HOURLY DEMAND ANALYSIS ==========")

hourly_demand = train_df.groupby("hour")[
    "demand"
].mean()

plt.figure(figsize=(12, 5))

hourly_demand.plot(marker="o")

plt.title("Average Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Demand")

plt.grid(True)

plt.savefig("outputs/plots/hourly_demand.png")
plt.close()

print("Saved: hourly_demand.png")

# =========================
# WEATHER VS DEMAND
# =========================

print("\n========== WEATHER VS DEMAND ==========")

weather_demand = train_df.groupby("Weather")[
    "demand"
].mean()

print(weather_demand)

plt.figure(figsize=(10, 5))

weather_demand.plot(kind="bar")

plt.title("Average Demand by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Demand")

plt.savefig("outputs/plots/weather_vs_demand.png")
plt.close()

print("Saved: weather_vs_demand.png")

# =========================
# ROADTYPE VS DEMAND
# =========================

if "RoadType" in train_df.columns:

    print("\n========== ROADTYPE VS DEMAND ==========")

    road_demand = train_df.groupby("RoadType")[
        "demand"
    ].mean()

    print(road_demand)

    plt.figure(figsize=(10, 5))

    road_demand.plot(kind="bar")

    plt.title("Average Demand by Road Type")
    plt.xlabel("Road Type")
    plt.ylabel("Average Demand")

    plt.savefig("outputs/plots/roadtype_vs_demand.png")
    plt.close()

    print("Saved: roadtype_vs_demand.png")

# =========================
# GEOHASH ANALYSIS
# =========================

print("\n========== GEOHASH ANALYSIS ==========")

unique_geohash = train_df["geohash"].nunique()

print("Unique geohash count:", unique_geohash)

# =========================
# CHECK DUPLICATES
# =========================

print("\n========== DUPLICATE ROWS ==========")

duplicates = train_df.duplicated().sum()

print("Duplicate Rows:", duplicates)

# =========================
# SAVE PROCESSED FILES
# =========================

print("\n========== SAVING PROCESSED FILES ==========")

train_df.to_csv(
    "outputs/processed/train_phase2.csv",
    index=False
)

test_df.to_csv(
    "outputs/processed/test_phase2.csv",
    index=False
)

print("Processed files saved successfully!")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== PHASE 2 COMPLETED ==========")

print("""
You have successfully completed:

1. Dataset Loading
2. Missing Value Analysis
3. Statistical Analysis
4. Demand Distribution Analysis
5. Categorical Feature Analysis
6. Correlation Analysis
7. Timestamp Processing
8. Time Feature Engineering
9. Hourly Demand Analysis
10. Weather Impact Analysis
11. Geohash Exploration
12. Processed File Saving

Next Phase:
-> Data Cleaning + Feature Engineering
""")