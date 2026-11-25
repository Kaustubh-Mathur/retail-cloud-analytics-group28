import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from db_and_features import build_household_rfm, get_engine, load_raw_tables


def run_req8_churn_prediction():
    engine = get_engine()
    tx, hh, _ = load_raw_tables(engine)
    df = build_household_rfm(tx, hh)

    max_recency = df["recency_days"].max()
    threshold = 90 if max_recency > 120 else np.percentile(df["recency_days"], 75)
    df["churned"] = (df["recency_days"] >= threshold).astype(int)

    numeric_cols = [c for c in ["total_spend", "avg_spend", "total_units", "n_baskets", "tenure_days", "avg_basket_value"] if c in df.columns]
    cat_cols = [c for c in ["INCOME_RANGE", "AGE_RANGE", "HH_SIZE", "CHILDREN", "MARITAL_STATUS"] if c in df.columns]

    X = df[numeric_cols + cat_cols].copy()
    y = df["churned"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== Churn Prediction (Threshold: {threshold:.1f} days | Churn Rate: {y.mean():.1%}) ===")
    if y_test.nunique() > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    df["churn_probability"] = model.predict_proba(X)[:, 1]
    top_at_risk = df.sort_values(["churn_probability", "total_spend"], ascending=False)[["HSHD_NUM", "total_spend", "n_baskets", "recency_days", "churn_probability"]].head(10)
    print("\n=== Top 10 At-Risk Households by Value ===")
    print(top_at_risk.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    run_req8_churn_prediction()