import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from db_and_features import build_household_rfm, get_engine, load_raw_tables


def run_req1_demographics_engagement():
    engine = get_engine()
    tx, hh, _ = load_raw_tables(engine)
    df = build_household_rfm(tx, hh)

    demo_cols = [
        c for c in ["INCOME_RANGE", "AGE_RANGE", "HH_SIZE", "CHILDREN", "MARITAL_STATUS", "HOMEOWNER_DESC"]
        if c in df.columns
    ]

    for col in demo_cols:
        summary = (
            df.groupby(col, dropna=False)
            .agg(
                households=("HSHD_NUM", "nunique"),
                avg_total_spend=("total_spend", "mean"),
                median_total_spend=("total_spend", "median"),
                avg_baskets=("n_baskets", "mean"),
                avg_recency_days=("recency_days", "mean"),
            )
            .sort_values("avg_total_spend", ascending=False)
        )
        print(f"\n=== Engagement by {col} ===")
        print(summary.round(2).to_string())

    cluster_features = ["total_spend", "n_baskets", "total_units", "recency_days"]
    X = df[cluster_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    df["engagement_cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)

    cluster_summary = (
        df.groupby("engagement_cluster")
        .agg(
            households=("HSHD_NUM", "nunique"),
            mean_spend=("total_spend", "mean"),
            mean_baskets=("n_baskets", "mean"),
            mean_recency=("recency_days", "mean"),
        )
        .sort_values("mean_spend", ascending=False)
    )
    print("\n=== K-Means Engagement Segments ===")
    print(cluster_summary.round(2).to_string())
    return df


if __name__ == "__main__":
    run_req1_demographics_engagement()