import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

print("Loading datasets...")

train_df = pd.read_csv(
    "outputs/processed/train_phase6.csv"
)

TARGET = "demand"

# =========================
# DROP UNUSED
# =========================

drop_cols = []

for col in ["timestamp", "Index"]:

    if col in train_df.columns:
        drop_cols.append(col)

X = train_df.drop(
    columns=[TARGET] + drop_cols
)

y = train_df[TARGET]

# =========================
# CATEGORICALS
# =========================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

for col in categorical_features:

    X[col] = X[col].astype("category")

# =========================
# CV
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================
# OBJECTIVE
# =========================

def objective(trial):

    params = {

        "objective": "regression",

        "metric": "rmse",

        "boosting_type": "gbdt",

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.1
            ),

        "num_leaves":
            trial.suggest_int(
                "num_leaves",
                31,
                255
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                4,
                12
            ),

        "min_child_samples":
            trial.suggest_int(
                "min_child_samples",
                5,
                100
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

        "reg_alpha":
            trial.suggest_float(
                "reg_alpha",
                0.0,
                5.0
            ),

        "reg_lambda":
            trial.suggest_float(
                "reg_lambda",
                0.0,
                5.0
            ),

        "n_estimators": 3000,

        "random_state": 42
    }

    oof_preds = np.zeros(len(X))

    for train_idx, valid_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = lgb.LGBMRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train,

            eval_set=[(X_valid, y_valid)],

            eval_metric="rmse",

            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(0)
            ]
        )

        preds_log = model.predict(X_valid)

        preds = np.expm1(preds_log)

        oof_preds[valid_idx] = preds

    y_actual = np.expm1(y)

    score = r2_score(
        y_actual,
        oof_preds
    )

    return score

# =========================
# RUN OPTUNA
# =========================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=20
)

# =========================
# RESULTS
# =========================

print("\n========== BEST RESULTS ==========")

print("\nBest Score:")
print(study.best_value)

print("\nBest Params:")

for k, v in study.best_params.items():

    print(f"{k}: {v}")