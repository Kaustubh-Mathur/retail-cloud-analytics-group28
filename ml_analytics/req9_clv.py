import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from db_and_features import build_household_rfm, get_engine, load_raw_tables


def run_req9_clv_forecasting():
    engine = get_engine()
    tx, hh, _ = load_raw_tables(engine)
    df = build_household_rfm(tx, hh)

    active_months = np.maximum(df["tenure_days"] / 30.4375, 1.0)
    df["annualized_run_rate"] = (df["total_spend"] / active_months) * 12.0

    numeric_cols = [c for c in ["n_baskets", "total_units", "avg_basket_value", "tenure_days", "recency_days"] if c in df.columns]
    cat_cols = [c for c in ["INCOME_RANGE", "AGE_RANGE", "HH_SIZE", "CHILDREN", "MARITAL_STATUS"] if c in df.columns]

    X = df[numeric_cols + cat_cols].copy()
    y_log = np.log1p(df["total_spend"].clip(lower=0))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    reg = Pipeline([
        ("prep", preprocessor),
        ("gbr", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.25, random_state=42)
    reg.fit(X_train, y_train)

    pred_test_spend = np.expm1(np.clip(reg.predict(X_test), a_min=0, a_max=None))
    actual_test_spend = np.expm1(y_test)

    print("\n=== CLV Model Evaluation (Historical Spend Target) ===")
    print(f"MAE ($): {mean_absolute_error(actual_test_spend, pred_test_spend):,.2f}")
    print(f"R²:      {r2_score(actual_test_spend, pred_test_spend):.4f}")

    ml_est_spend = np.expm1(np.clip(reg.predict(X), a_min=0, a_max=None))
    recency_factor = np.exp(-df["recency_days"] / 180.0)
    df["projected_12m_clv"] = (0.5 * ml_est_spend + 0.5 * df["annualized_run_rate"]) * np.clip(recency_factor, 0.25, 1.0)

    top_clv = df.sort_values("projected_12m_clv", ascending=False)[["HSHD_NUM", "total_spend", "n_baskets", "annualized_run_rate", "projected_12m_clv"]].head(10)
    print("\n=== Top 10 Households by Projected 12-Month CLV ===")
    print(top_clv.round(2).to_string(index=False))
    return df


if __name__ == "__main__":
    run_req9_clv_forecasting()