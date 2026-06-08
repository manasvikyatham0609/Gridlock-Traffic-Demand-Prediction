import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

print("Loading predictions...")

# =========================
# LOAD OOF FILES
# =========================

lgb_oof = pd.read_csv(
    "outputs/oof/lgb_oof.csv"
)

cat_oof = pd.read_csv(
    "outputs/oof/catboost_oof.csv"
)

# =========================
# LOAD SUBMISSIONS
# =========================

lgb_sub = pd.read_csv(
    "outputs/submissions/lightgbm_submission.csv"
)

cat_sub = pd.read_csv(
    "outputs/submissions/catboost_submission.csv"
)

# =========================
# LOAD TRUE TARGET
# =========================

train_df = pd.read_csv(
    "outputs/processed/train_phase6.csv"
)

y_true = np.expm1(
    train_df["demand"]
)

# =========================
# GET OOF PREDICTIONS
# =========================

lgb_pred = lgb_oof["pred"].values
cat_pred = cat_oof["pred"].values

# =========================
# SEARCH BEST WEIGHT
# =========================

print("\n========== SEARCHING BEST ENSEMBLE ==========")

best_score = -1
best_weight = 0

for w in np.linspace(0, 1, 101):

    blended = (
        w * lgb_pred +
        (1 - w) * cat_pred
    )

    score = r2_score(
        y_true,
        blended
    )

    if score > best_score:

        best_score = score
        best_weight = w

print("\nBest Ensemble Found:")
print(f"LightGBM Weight : {best_weight:.2f}")
print(f"CatBoost Weight : {1-best_weight:.2f}")

print(f"\nEnsemble R2 : {best_score:.6f}")
print(f"Competition Score : {best_score * 100:.4f}")

# =========================
# TEST PREDICTIONS
# =========================

lgb_test = lgb_sub["demand"].values

# CatBoost submission column may differ
if "demand" in cat_sub.columns:
    cat_test = cat_sub["demand"].values
else:
    cat_test = cat_sub["prediction"].values

final_preds = (
    best_weight * lgb_test +
    (1 - best_weight) * cat_test
)

# =========================
# SAVE FINAL SUBMISSION
# =========================

test_df = pd.read_csv(
    "outputs/processed/test_phase6.csv"
)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "demand": final_preds
})

submission.to_csv(
    "outputs/submissions/final_ensemble.csv",
    index=False
)

print("\nFinal ensemble saved!")
print("File: outputs/submissions/final_ensemble.csv")

print("\n========== PHASE 8 COMPLETED ==========")