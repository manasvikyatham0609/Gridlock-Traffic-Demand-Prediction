import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("Loading OOF predictions...")

# =========================
# LOAD OOF FILES
# =========================

lgb_oof = pd.read_csv(
    "outputs/oof/lgb_oof.csv"
)

xgb_oof = pd.read_csv(
    "outputs/oof/xgb_oof.csv"
)

cat_oof = pd.read_csv(
    "outputs/oof/catboost_oof.csv"
)

# =========================
# TARGET
# =========================

# =========================
# LOAD TARGET
# =========================

train_df = pd.read_csv(
    "outputs/processed/train_phase12.csv"
)

TARGET = "demand"

# target is log transformed
y = np.expm1(
    train_df[TARGET].values
)
# =========================
# STACK FEATURES
# =========================

X_stack = pd.DataFrame({
    "lgb": lgb_oof["pred"],
    "xgb": xgb_oof["pred"],
    "cat": cat_oof["pred"]
})

# =========================
# TRAIN META MODEL
# =========================

print("\nTraining Ridge meta model...")

meta_model = Ridge(
    alpha=1.0
)

meta_model.fit(X_stack, y)

# =========================
# OOF PREDICTIONS
# =========================

stack_oof = meta_model.predict(X_stack)

stack_score = r2_score(
    y,
    stack_oof
)

print("\n========== STACKING RESULTS ==========")

print(f"Stacking R2 : {stack_score:.6f}")
print(f"Competition Score : {stack_score*100:.4f}")

# =========================
# LOAD TEST PREDICTIONS
# =========================

lgb_sub = pd.read_csv(
    "outputs/submissions/lightgbm_submission.csv"
)

xgb_sub = pd.read_csv(
    "outputs/submissions/xgb_submission.csv"
)

cat_sub = pd.read_csv(
    "outputs/submissions/catboost_submission.csv"
)

# =========================
# TEST STACK FEATURES
# =========================

X_test_stack = pd.DataFrame({
    "lgb": lgb_sub["demand"],
    "xgb": xgb_sub["demand"],
    "cat": cat_sub["demand"]
})

# =========================
# FINAL PREDICTIONS
# =========================

final_preds = meta_model.predict(
    X_test_stack
)

final_preds = np.clip(
    final_preds,
    0,
    None
)

# =========================
# FINAL SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": lgb_sub["Index"],
    "demand": final_preds
})

os.makedirs(
    "outputs/submissions",
    exist_ok=True
)

submission_path = (
    "outputs/submissions/"
    "stacking_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print("\nSaved stacking submission!")
print(f"File: {submission_path}")

# =========================
# META MODEL WEIGHTS
# =========================

print("\nMeta Model Weights:")

for feature, coef in zip(
    X_stack.columns,
    meta_model.coef_
):

    print(f"{feature}: {coef:.6f}")