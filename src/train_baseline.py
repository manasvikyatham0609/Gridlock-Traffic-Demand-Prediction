import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor

# =========================
# CREATE OUTPUT DIRECTORIES
# =========================

os.makedirs("outputs/submissions", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# =========================
# LOAD DATA
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
# DEFINE TARGET
# =========================

TARGET = "demand"

# =========================
# DROP UNUSED COLUMNS
# =========================

drop_cols = []

# Timestamp raw string not needed
if "timestamp" in train_df.columns:
    drop_cols.append("timestamp")

# Index should not help prediction
if "Index" in train_df.columns:
    drop_cols.append("Index")

print("\nDropping columns:", drop_cols)

# =========================
# PREPARE FEATURES
# =========================

X = train_df.drop(columns=[TARGET] + drop_cols)

y = train_df[TARGET]

X_test = test_df.drop(columns=drop_cols)

# =========================
# IDENTIFY CATEGORICAL FEATURES
# =========================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

# =========================
# TRAIN VALIDATION SPLIT
# =========================

print("\nCreating train-validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Shape:", X_train.shape)
print("Validation Shape:", X_valid.shape)

# =========================
# INITIALIZE MODEL
# =========================

print("\nInitializing CatBoost model...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=100
)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

print("Model training completed!")

# =========================
# VALIDATION PREDICTIONS
# =========================

print("\nGenerating validation predictions...")

valid_preds_log = model.predict(X_valid)

# Reverse log transform
valid_preds = np.expm1(valid_preds_log)
y_valid_actual = np.expm1(y_valid)

# =========================
# EVALUATE MODEL
# =========================

print("\nEvaluating model...")

r2 = r2_score(
    y_valid_actual,
    valid_preds
)

competition_score = max(0, 100 * r2)

print(f"\nValidation R2 Score: {r2:.6f}")
print(f"Competition Score : {competition_score:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(15))

# =========================
# TEST PREDICTIONS
# =========================

print("\nGenerating test predictions...")

test_preds_log = model.predict(X_test)

# Reverse log transform
test_preds = np.expm1(test_preds_log)

# =========================
# CREATE SUBMISSION
# =========================

print("\nCreating submission file...")

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "demand": test_preds
})

submission_path = (
    "outputs/submissions/"
    "baseline_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print(f"Submission saved to: {submission_path}")

# =========================
# SAVE MODEL
# =========================

model_path = (
    "outputs/models/"
    "catboost_baseline_model.cbm"
)

model.save_model(model_path)

print(f"Model saved to: {model_path}")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== PHASE 4 COMPLETED ==========")

print("""
Successfully completed:

1. Baseline model training
2. Validation evaluation
3. Feature importance analysis
4. Test prediction generation
5. Submission file creation
6. Model saving

Generated Files:
- baseline_submission.csv
- catboost_baseline_model.cbm

Next Phase:
-> Model Improvement + Hyperparameter Tuning
""")