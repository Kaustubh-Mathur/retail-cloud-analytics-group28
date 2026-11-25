import io
import os
import sys
from contextlib import redirect_stdout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work whether run from GroupProject/ or ml_analytics/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from db_and_features import get_engine, load_raw_tables
from req1_demographics import run_req1_demographics_engagement
from req7_basket import run_req7_basket_analysis
from req8_churn import run_req8_churn_prediction
from req9_clv import run_req9_clv_forecasting

RESULTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(filename):
    path = os.path.join(RESULTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart -> {path}")


def plot_req1_demographics(df_req1):
    # Chart 1A: Average Spend by Income Range
    if "INCOME_RANGE" in df_req1.columns:
        inc = (
            df_req1.dropna(subset=["INCOME_RANGE"])
            .groupby("INCOME_RANGE")["total_spend"]
            .mean()
            .sort_values(ascending=True)
        )
        plt.figure(figsize=(8, 4.5))
        bars = plt.barh(inc.index.astype(str), inc.values, color="#2563eb")
        plt.title("Requirement 1: Average Household Spend by Income Range", fontsize=12, fontweight="bold")
        plt.xlabel("Average Total Spend ($)")
        plt.ylabel("Income Range")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 30, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9)
        save_fig("req1_spend_by_income.png")

    # Chart 1B: K-Means Engagement Clusters (Spend vs Baskets)
    if "engagement_cluster" in df_req1.columns:
        plt.figure(figsize=(8, 5))
        palette = {0: "#0ea5e9", 1: "#16a34a", 2: "#ef4444"}
        labels = {
            0: "Cluster 0: Core Active",
            1: "Cluster 1: VIP High-Spend",
            2: "Cluster 2: Lapsed / Low-Engagement",
        }
        for cid, grp in df_req1.groupby("engagement_cluster"):
            plt.scatter(
                grp["n_baskets"],
                grp["total_spend"],
                label=labels.get(cid, f"Cluster {cid}"),
                color=palette.get(cid, "#64748b"),
                alpha=0.75,
                edgecolors="white",
                s=55,
            )
        plt.title("Requirement 1: K-Means Household Engagement Segments", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Shopping Baskets")
        plt.ylabel("Total Household Spend ($)")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.35)
        save_fig("req1_kmeans_segments.png")


def compute_commodity_rules(min_support=0.015, min_confidence=0.20, min_lift=1.15, top_n=12):
    """Also compute commodity-level rules so Req 7 has rich multi-pair visuals even if department only yields 1 pair."""
    from collections import Counter
    from itertools import combinations

    engine = get_engine()
    tx, _, prod = load_raw_tables(engine)
    if "COMMODITY" not in prod.columns:
        return pd.DataFrame()

    basket_df = pd.merge(
        tx[["BASKET_NUM", "PRODUCT_NUM"]].dropna(),
        prod[["PRODUCT_NUM", "COMMODITY"]].drop_duplicates(),
        on="PRODUCT_NUM",
        how="inner",
    )
    basket_df["COMMODITY"] = basket_df["COMMODITY"].astype(str).str.strip()
    basket_df = basket_df[basket_df["COMMODITY"] != ""]

    baskets = basket_df.groupby("BASKET_NUM")["COMMODITY"].apply(lambda s: sorted(set(s)))
    total_baskets = len(baskets)
    if total_baskets == 0:
        return pd.DataFrame()

    item_counts = Counter()
    pair_counts = Counter()
    for items in baskets:
        for item in items:
            item_counts[item] += 1
        for a, b in combinations(items, 2):
            pair_counts[(a, b)] += 1

    rules = []
    for (a, b), ab_cnt in pair_counts.items():
        sup_ab = ab_cnt / total_baskets
        if sup_ab < min_support:
            continue
        sup_a = item_counts[a] / total_baskets
        sup_b = item_counts[b] / total_baskets
        for ant, con, sup_con in [(a, b, sup_b), (b, a, sup_a)]:
            conf = ab_cnt / item_counts[ant]
            lift = conf / sup_con if sup_con > 0 else 0.0
            if conf >= min_confidence and lift >= min_lift:
                rules.append({
                    "antecedent": ant,
                    "consequent": con,
                    "support": round(sup_ab, 4),
                    "confidence": round(conf, 4),
                    "lift": round(lift, 3),
                    "basket_count": ab_cnt,
                })

    return pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).head(top_n)


def plot_req7_basket(dept_rules_df, comm_rules_df):
    plot_df = comm_rules_df if not comm_rules_df.empty else dept_rules_df
    if plot_df.empty:
        return

    plot_df = plot_df.copy().iloc[::-1]
    rule_labels = plot_df["antecedent"] + " → " + plot_df["consequent"]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(rule_labels, plot_df["lift"], color="#7c3aed")
    plt.axvline(1.0, color="#94a3b8", linestyle="--", linewidth=1, label="Baseline Lift (1.0)")
    plt.title("Requirement 7: Top Market Basket Association Rules by Lift", fontsize=12, fontweight="bold")
    plt.xlabel("Lift Score")
    plt.ylabel("Association Rule (Antecedent → Consequent)")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, conf in zip(bars, plot_df["confidence"]):
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"Lift {w:.2f} (Conf {conf:.0%})", va="center", fontsize=8.5)
    save_fig("req7_market_basket_lift.png")


def plot_req8_churn(df_req8):
    # Chart 8A: Churn Risk Distribution
    plt.figure(figsize=(8, 4.5))
    plt.hist(df_req8["churn_probability"], bins=20, color="#f97316", edgecolor="white")
    plt.axvline(0.5, color="#1e293b", linestyle="--", label="Decision Threshold (0.50)")
    plt.title("Requirement 8: Household Churn Probability Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Churn Probability")
    plt.ylabel("Number of Households")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    save_fig("req8_churn_distribution.png")

    # Chart 8B: Top 10 High-Value At-Risk Households
    at_risk = (
        df_req8[df_req8["churn_probability"] >= 0.5]
        .sort_values("total_spend", ascending=False)
        .head(10)
        .iloc[::-1]
    )
    if not at_risk.empty:
        plt.figure(figsize=(8.5, 4.8))
        labels = "HSHD " + at_risk["HSHD_NUM"].astype(str)
        bars = plt.barh(labels, at_risk["total_spend"], color="#dc2626")
        plt.title("Requirement 8: Top 10 At-Risk Households by Historical Spend", fontsize=12, fontweight="bold")
        plt.xlabel("Historical Spend at Risk ($)")
        plt.ylabel("Household ID")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        for bar, prob in zip(bars, at_risk["churn_probability"]):
            w = bar.get_width()
            plt.text(w + 15, bar.get_y() + bar.get_height() / 2, f"${w:,.0f} ({prob:.0%} risk)", va="center", fontsize=8.5)
        save_fig("req8_top_at_risk_households.png")


def plot_req9_clv(df_req9):
    # Chart 9A: Top 10 Households by Projected 12-Month CLV
    top10 = df_req9.sort_values("projected_12m_clv", ascending=False).head(10).iloc[::-1]
    plt.figure(figsize=(8.5, 4.8))
    labels = "HSHD " + top10["HSHD_NUM"].astype(str)
    bars = plt.barh(labels, top10["projected_12m_clv"], color="#059669")
    plt.title("Requirement 9: Top 10 Households by Projected 12-Month CLV", fontsize=12, fontweight="bold")
    plt.xlabel("Projected 12-Month CLV ($)")
    plt.ylabel("Household ID")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 60, bar.get_y() + bar.get_height() / 2, f"${w:,.2f}", va="center", fontsize=9)
    save_fig("req9_top10_clv.png")

    # Chart 9B: Historical Spend vs Projected 12M CLV
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df_req9["total_spend"],
        df_req9["projected_12m_clv"],
        alpha=0.65,
        color="#0284c7",
        edgecolors="white",
        s=50,
    )
    plt.title("Requirement 9: Historical Spend vs. Projected 12-Month CLV", fontsize=12, fontweight="bold")
    plt.xlabel("Historical Total Spend ($)")
    plt.ylabel("Projected 12-Month CLV ($)")
    plt.grid(True, linestyle="--", alpha=0.35)
    save_fig("req9_spend_vs_clv.png")


def main():
    log_buffer = io.StringIO()

    print("Running Requirement 1: Demographics & Engagement...")
    with redirect_stdout(log_buffer):
        df_req1 = run_req1_demographics_engagement()
    df_req1.to_csv(os.path.join(RESULTS_DIR, "req1_household_segments.csv"), index=False)
    plot_req1_demographics(df_req1)

    print("Running Requirement 7: Market Basket Analysis...")
    with redirect_stdout(log_buffer):
        dept_rules = run_req7_basket_analysis()
    comm_rules = compute_commodity_rules()
    dept_rules.to_csv(os.path.join(RESULTS_DIR, "req7_department_rules.csv"), index=False)
    if not comm_rules.empty:
        comm_rules.to_csv(os.path.join(RESULTS_DIR, "req7_commodity_rules.csv"), index=False)
    plot_req7_basket(dept_rules, comm_rules)

    print("Running Requirement 8: Customer Churn Prediction...")
    with redirect_stdout(log_buffer):
        df_req8 = run_req8_churn_prediction()
    df_req8[["HSHD_NUM", "total_spend", "n_baskets", "recency_days", "churned", "churn_probability"]].sort_values(
        "churn_probability", ascending=False
    ).to_csv(os.path.join(RESULTS_DIR, "req8_churn_predictions.csv"), index=False)
    plot_req8_churn(df_req8)

    print("Running Requirement 9: Customer Lifetime Value (CLV) Forecasting...")
    with redirect_stdout(log_buffer):
        df_req9 = run_req9_clv_forecasting()
    df_req9[["HSHD_NUM", "total_spend", "n_baskets", "annualized_run_rate", "projected_12m_clv"]].sort_values(
        "projected_12m_clv", ascending=False
    ).to_csv(os.path.join(RESULTS_DIR, "req9_clv_forecasts.csv"), index=False)
    plot_req9_clv(df_req9)

    summary_path = os.path.join(RESULTS_DIR, "ml_metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(log_buffer.getvalue())
    print(f"Saved full text metrics log -> {summary_path}")
    print(f"\nAll charts, CSVs, and metrics saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()