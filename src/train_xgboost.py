import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# =========================
# CREATE DIRS
# =========================

os.makedirs("outputs/submissions", exist_ok=True)
os.makedirs("outputs/oof", exist_ok=True)

# =========================
# LOAD DATA
# =========================

print("Loading datasets...")

train_df = pd.read_csv(
    "outputs/processed/train_phase14.csv"
)

test_df = pd.read_csv(
    "outputs/processed/test_phase14.csv"
)

TARGET = "demand"

# =========================
# DROP COLS
# =========================

drop_cols = []

for col in ["timestamp", "Index"]:

    if col in train_df.columns:
        drop_cols.append(col)

# =========================
# FEATURES
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

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

# label encode
for col in categorical_cols:

    full_data = pd.concat([
        X[col],
        X_test[col]
    ]).astype(str)

    mapping = {
        v: i for i, v in enumerate(
            full_data.unique()
        )
    }

    X[col] = X[col].astype(str).map(mapping)
    X_test[col] = X_test[col].astype(str).map(mapping)

# =========================
# KFOLD
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================
# STORAGE
# =========================

oof_preds = np.zeros(len(X))

test_preds = np.zeros(len(X_test))

scores = []

# =========================
# TRAINING
# =========================

print("\n========== STARTING XGBOOST CV ==========")

for fold, (train_idx, valid_idx) in enumerate(
    kf.split(X)
):

    print(f"\n========== FOLD {fold+1} ==========")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = XGBRegressor(

        n_estimators=3000,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42,

        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[(X_valid, y_valid)],

        verbose=200
    )

    preds_log = model.predict(X_valid)

    preds = np.expm1(preds_log)

    y_actual = np.expm1(y_valid)

    oof_preds[valid_idx] = preds

    score = r2_score(
        y_actual,
        preds
    )

    scores.append(score)

    print(f"\nFold {fold+1} R2: {score:.6f}")

    test_fold = np.expm1(
        model.predict(X_test)
    )

    test_fold = np.clip(
        test_fold,
        0,
        None
    )

    test_preds += (
        test_fold / 5
    )

# =========================
# FINAL SCORE
# =========================

overall = r2_score(
    np.expm1(y),
    oof_preds
)

print("\n========== FINAL XGBOOST RESULTS ==========")

print(f"Overall R2: {overall:.6f}")
print(f"Competition Score: {overall * 100:.4f}")

# =========================
# SAVE OOF
# =========================

pd.DataFrame({
    "pred": oof_preds
}).to_csv(
    "outputs/oof/xgb_oof.csv",
    index=False
)

# =========================
# SAVE SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "demand": test_preds
})

submission.to_csv(
    "outputs/submissions/xgb_submission.csv",
    index=False
)

print("\nSaved XGBoost submission!")