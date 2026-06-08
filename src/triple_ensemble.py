import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

print("Loading OOF predictions...")

# =========================
# LOAD OOF FILES
# =========================

lgb_oof = pd.read_csv(
    "outputs/oof/lgb_oof.csv"
)["pred"].values

cat_oof = pd.read_csv(
    "outputs/oof/catboost_oof.csv"
)["pred"].values

xgb_oof = pd.read_csv(
    "outputs/oof/xgb_oof.csv"
)["pred"].values

# =========================
# TRUE TARGET
# =========================

train_df = pd.read_csv(
    "outputs/processed/train_phase6.csv"
)

y_true = np.expm1(
    train_df["demand"]
)

# =========================
# LOAD TEST SUBMISSIONS
# =========================

lgb_sub = pd.read_csv(
    "outputs/submissions/lightgbm_submission.csv"
)

cat_sub = pd.read_csv(
    "outputs/submissions/catboost_submission.csv"
)

xgb_sub = pd.read_csv(
    "outputs/submissions/xgb_submission.csv"
)

# =========================
# GET TEST PREDICTIONS
# =========================

lgb_test = lgb_sub["demand"].values

if "demand" in cat_sub.columns:
    cat_test = cat_sub["demand"].values
else:
    cat_test = cat_sub["prediction"].values

xgb_test = xgb_sub["demand"].values

# =========================
# WEIGHT SEARCH
# =========================

print("\nSearching best weights...")

best_score = -1

best_weights = None

for w1 in np.arange(0, 1.01, 0.05):

    for w2 in np.arange(0, 1.01 - w1, 0.05):

        w3 = 1 - w1 - w2

        blended = (
            w1 * lgb_oof +
            w2 * cat_oof +
            w3 * xgb_oof
        )

        score = r2_score(
            y_true,
            blended
        )

        if score > best_score:

            best_score = score

            best_weights = (w1, w2, w3)

# =========================
# BEST RESULT
# =========================

w1, w2, w3 = best_weights

print("\n========== BEST ENSEMBLE ==========")

print(f"LightGBM Weight : {w1:.2f}")
print(f"CatBoost Weight : {w2:.2f}")
print(f"XGBoost Weight  : {w3:.2f}")

print(f"\nBest R2 : {best_score:.6f}")
print(f"Competition Score : {best_score * 100:.4f}")

# =========================
# FINAL TEST PREDS
# =========================

final_preds = (
    w1 * lgb_test +
    w2 * cat_test +
    w3 * xgb_test
)

# =========================
# SAVE FINAL SUBMISSION
# =========================

submission = pd.DataFrame({

    "Index": pd.read_csv(
        "outputs/processed/test_phase6.csv"
    )["Index"],

    "demand": final_preds
})

submission.to_csv(
    "outputs/submissions/final_triple_ensemble.csv",
    index=False
)

print("\nSaved final submission!")
print(
    "File: outputs/submissions/final_triple_ensemble.csv"
)