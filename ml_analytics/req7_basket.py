from collections import Counter
from itertools import combinations
import pandas as pd
from db_and_features import get_engine, load_raw_tables


def run_req7_basket_analysis(min_support=0.01, min_confidence=0.15, min_lift=1.1, top_n=20):
    engine = get_engine()
    tx, _, prod = load_raw_tables(engine)

    label_col = "COMMODITY_DESC" if "COMMODITY_DESC" in prod.columns else (
        "DEPARTMENT" if "DEPARTMENT" in prod.columns else prod.columns[-1]
    )
    basket_df = pd.merge(
        tx[["BASKET_NUM", "PRODUCT_NUM"]].dropna(),
        prod[["PRODUCT_NUM", label_col]].drop_duplicates(),
        on="PRODUCT_NUM",
        how="inner",
    )
    basket_df[label_col] = basket_df[label_col].astype(str).str.strip()
    basket_df = basket_df[basket_df[label_col] != ""]

    baskets = basket_df.groupby("BASKET_NUM")[label_col].apply(lambda s: sorted(set(s)))
    total_baskets = len(baskets)
    if total_baskets == 0:
        print("No baskets found for association rule mining.")
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

    rules_df = pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).head(top_n)
    print(f"\n=== Top {top_n} Market Basket Rules (Total Baskets: {total_baskets:,}) ===")
    print(rules_df.to_string(index=False))
    return rules_df


if __name__ == "__main__":
    run_req7_basket_analysis()