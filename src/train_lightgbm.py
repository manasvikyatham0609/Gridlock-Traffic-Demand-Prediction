import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import lightgbm as lgb

# =========================
# CREATE OUTPUT DIRECTORIES
# =========================

os.makedirs("outputs/submissions", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/oof", exist_ok=True)

# =========================
# LOAD DATA
# =========================

print("Loading Phase 14 datasets...")

train_df = pd.read_csv(
    "outputs/processed/train_phase14.csv"
)

test_df = pd.read_csv(
    "outputs/processed/test_phase14.csv"
)

print("Datasets loaded successfully!")

# =========================
# SETTINGS
# =========================

TARGET = "demand"

N_SPLITS = 5

RANDOM_STATE = 42

# =========================
# DROP UNUSED COLUMNS
# =========================

drop_cols = []

for col in ["timestamp", "Index"]:

    if col in train_df.columns:
        drop_cols.append(col)

print("\nDropping columns:", drop_cols)

# =========================
# PREPARE FEATURES
# =========================

X = train_df.drop(
    columns=[TARGET] + drop_cols
)

y = train_df[TARGET]

X_test = test_df.drop(
    columns=drop_cols
)

# =========================
# HANDLE CATEGORICALS
# =========================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

# Convert categorical columns
for col in categorical_features:

    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# =========================
# KFOLD
# =========================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =========================
# STORAGE
# =========================

oof_predictions = np.zeros(len(train_df))

test_predictions = np.zeros(len(test_df))

fold_scores = []

# =========================
# TRAINING
# =========================

print("\n========== STARTING LIGHTGBM CV ==========")

for fold, (train_idx, valid_idx) in enumerate(
    kf.split(X)
):

    print(f"\n========== FOLD {fold+1} ==========")

    # =========================
    # SPLIT DATA
    # =========================

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    # =========================
    # MODEL
    # =========================

    model = lgb.LGBMRegressor(

    objective="regression",

    n_estimators=3000,

    learning_rate=0.03710597898827531,

    num_leaves=202,

    max_depth=7,

    min_child_samples=35,

    subsample=0.9759673159038151,

    colsample_bytree=0.787406013748241,

    reg_alpha=0.08399746570292999,

    reg_lambda=4.828601311610841,

    random_state=RANDOM_STATE
)

    # =========================
    # TRAIN
    # =========================

    model.fit(
        X_train,
        y_train,

        eval_set=[(X_valid, y_valid)],

        eval_metric="rmse",

        callbacks=[
            lgb.early_stopping(300),
            lgb.log_evaluation(200)
        ]
    )

    # =========================
    # VALIDATION PREDICTIONS
    # =========================

    valid_preds_log = model.predict(
        X_valid
    )

    valid_preds = np.expm1(
        valid_preds_log
    )

    y_valid_actual = np.expm1(
        y_valid
    )

    # Store OOF
    oof_predictions[valid_idx] = valid_preds

    # =========================
    # SCORE
    # =========================

    fold_r2 = r2_score(
        y_valid_actual,
        valid_preds
    )

    fold_scores.append(fold_r2)

    print(f"\nFold {fold+1} R2: {fold_r2:.6f}")

    # =========================
    # TEST PREDICTIONS
    # =========================

    fold_test_preds_log = model.predict(
        X_test
    )

    fold_test_preds = np.expm1(
        fold_test_preds_log
    )

    fold_test_preds = np.clip(
        fold_test_preds,
        0,
        None
    )

    test_predictions += (
        fold_test_preds / N_SPLITS
    )

# =========================
# FINAL RESULTS
# =========================

print("\n========== FINAL LIGHTGBM RESULTS ==========")

y_actual = np.expm1(y)

overall_r2 = r2_score(
    y_actual,
    oof_predictions
)

competition_score = max(
    0,
    100 * overall_r2
)

print(f"\nOverall CV R2 : {overall_r2:.6f}")
print(f"Competition Score: {competition_score:.4f}")

print("\nFold Scores:")

for i, score in enumerate(fold_scores):

    print(f"Fold {i+1}: {score:.6f}")

print(
    f"\nMean Fold Score: "
    f"{np.mean(fold_scores):.6f}"
)

# =========================
# SAVE SUBMISSION
# =========================

print("\nCreating LightGBM submission...")

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "demand": test_predictions
})

submission_path = (
    "outputs/submissions/"
    "lightgbm_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print(f"Submission saved to: {submission_path}")

# =========================
# SAVE OOF PREDICTIONS
# =========================

print("\nSaving OOF predictions...")

oof_df = pd.DataFrame({
    "pred": oof_predictions
})

oof_df.to_csv("outputs/oof/lgb_oof.csv", index=False)

print("Saved: outputs/oof/lgb_oof.csv")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== LIGHTGBM TRAINING COMPLETED ==========")

print("""
Successfully completed:

1. LightGBM CV training
2. OOF prediction generation
3. Ensemble-ready predictions
4. LightGBM submission generation

Generated Files:
- lightgbm_submission.csv

Next Step:
-> Ensemble CatBoost + LightGBM
""")