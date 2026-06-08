import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
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

print("Loading Phase 6 datasets...")

train_df = pd.read_csv(
    "outputs/processed/train_phase6.csv"
)

test_df = pd.read_csv(
    "outputs/processed/test_phase6.csv"
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
# CATEGORICAL FEATURES
# =========================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

# =========================
# KFOLD SETUP
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
# CROSS VALIDATION
# =========================

print("\n========== STARTING CROSS VALIDATION ==========")

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

    print("Train Shape:", X_train.shape)
    print("Validation Shape:", X_valid.shape)

    # =========================
    # MODEL
    # =========================

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,

        loss_function="RMSE",
        eval_metric="R2",

        bootstrap_type="Bernoulli",
        subsample=0.8,

        random_seed=RANDOM_STATE,

        verbose=200
    )

    # =========================
    # TRAIN
    # =========================

    model.fit(
        X_train,
        y_train,

        cat_features=categorical_features,

        eval_set=(X_valid, y_valid),

        early_stopping_rounds=300,

        use_best_model=True
    )

    # =========================
    # VALIDATION PREDICTIONS
    # =========================

    valid_preds_log = model.predict(X_valid)

    valid_preds = np.expm1(
        valid_preds_log
    )

    y_valid_actual = np.expm1(
        y_valid
    )

    # Store OOF predictions
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
    # SAVE MODEL
    # =========================

    model_path = (
        f"outputs/models/"
        f"catboost_fold_{fold+1}.cbm"
    )

    model.save_model(model_path)

    print(f"Saved: {model_path}")

# =========================
# FINAL CV SCORE
# =========================

print("\n========== FINAL CV RESULTS ==========")

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
# CREATE SUBMISSION
# =========================

print("\nCreating CV submission file...")

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "demand": test_predictions
})

submission_path = (
    "outputs/submissions/"
    "cv_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print(f"Submission saved to: {submission_path}")

# =========================
# FINAL SUMMARY
# =========================

print("\n========== PHASE 6 COMPLETED ==========")

print("""
Successfully completed:

1. 5-Fold Cross Validation
2. Leakage-safe evaluation
3. OOF prediction generation
4. Fold model saving
5. Ensemble test prediction
6. Final submission creation

Generated Files:
- cv_submission.csv
- catboost_fold_1.cbm
- catboost_fold_2.cbm
- catboost_fold_3.cbm
- catboost_fold_4.cbm
- catboost_fold_5.cbm

Pipeline Status:
-> Competition-grade solution ready
""")