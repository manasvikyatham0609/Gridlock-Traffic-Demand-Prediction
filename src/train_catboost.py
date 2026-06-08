import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# =========================
# CREATE OUTPUT DIRS
# =========================
os.makedirs("outputs/submissions", exist_ok=True)
os.makedirs("outputs/oof", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading Phase 14 datasets...")

train = pd.read_csv("outputs/processed/train_phase14.csv")
test = pd.read_csv("outputs/processed/test_phase14.csv")

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
drop_cols = [c for c in ["timestamp", "Index"] if c in train.columns]

train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# =========================
# SPLIT FEATURES / TARGET
# =========================
X = train.drop(columns=[TARGET])
y = train[TARGET]

X_test = test.copy()

# =========================
# CATEGORICAL FEATURES
# =========================
cat_features = [
    'geohash', 'RoadType', 'LargeVehicles',
    'Landmarks', 'Weather',
    'geohash_prefix_2', 'geohash_prefix_3', 'geohash_prefix_4'
]

# keep only existing columns
cat_features = [c for c in cat_features if c in X.columns]

print("\nCategorical Features:")
print(cat_features)

# =========================
# K-FOLD
# =========================
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

fold_scores = []

print("\n========== STARTING CATBOOST CV ==========")

# =========================
# TRAIN LOOP
# =========================
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print(f"\n========== FOLD {fold+1} ==========")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=RANDOM_STATE,
        verbose=200,
        early_stopping_rounds=200
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    # =========================
    # VALIDATION
    # =========================
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    fold_r2 = r2_score(y_val, val_pred)
    fold_scores.append(fold_r2)

    print(f"Fold {fold+1} R2: {fold_r2:.6f}")

    # =========================
    # TEST PREDICTIONS
    # =========================
    test_preds += model.predict(X_test) / N_SPLITS

# =========================
# FINAL SCORE
# =========================
overall_r2 = r2_score(y, oof_preds)

print("\n========== FINAL CATBOOST RESULTS ==========")
print(f"Overall CV R2 : {overall_r2:.6f}")
print(f"Competition Score: {overall_r2 * 100:.4f}")

print("\nFold Scores:")
for i, s in enumerate(fold_scores):
    print(f"Fold {i+1}: {s:.6f}")

print(f"\nMean Fold Score: {np.mean(fold_scores):.6f}")

# =========================
# SAVE OOF
# =========================
oof_df = pd.DataFrame({
    "actual": y,
    "pred": oof_preds
})

oof_df.to_csv("outputs/oof/catboost_oof.csv", index=False)

# =========================
# SAVE SUBMISSION
# =========================
submission = pd.DataFrame({
    "Index": test["Index"] if "Index" in test.columns else np.arange(len(test)),
    "demand": test_preds
})

submission.to_csv(
    "outputs/submissions/catboost_submission.csv",
    index=False
)

print("\nCatBoost submission saved!")
print("File: outputs/submissions/catboost_submission.csv")

print("\n========== CATBOOST TRAINING COMPLETED ==========")