import os
import certifi
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def get_engine():
    server = os.getenv("DB_SERVER", "sql-retail-group28.database.windows.net")
    database = os.getenv("DB_NAME", "retail_db")
    username = os.getenv("DB_USER", "sqladmin")
    password = os.getenv("DB_PASS", "RetailCloud2026!")

    return create_engine(
        f"mssql+pytds://{username}:{password}@{server}:1433/{database}",
        connect_args={
            "use_mars": True,
            "cafile": certifi.where(),
            "validate_host": False,
        },
    )


def load_raw_tables(engine):
    tx = pd.read_sql("SELECT * FROM transactions", engine)
    hh = pd.read_sql("SELECT * FROM households", engine)
    prod = pd.read_sql("SELECT * FROM products", engine)

    tx.columns = [c.upper() for c in tx.columns]
    hh.columns = [c.upper() for c in hh.columns]
    prod.columns = [c.upper() for c in prod.columns]

    if "PURCHASE_DATE" in tx.columns:
        tx["PURCHASE_DT"] = pd.to_datetime(tx["PURCHASE_DATE"], errors="coerce")
    elif "DATE" in tx.columns:
        tx["PURCHASE_DT"] = pd.to_datetime(tx["DATE"], errors="coerce")
    else:
        tx["PURCHASE_DT"] = pd.NaT

    for col in ["SPEND", "UNITS", "QUANTITY", "PRICE"]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    if "SPEND" not in tx.columns:
        if "PRICE" in tx.columns and "UNITS" in tx.columns:
            tx["SPEND"] = tx["PRICE"].fillna(0) * tx["UNITS"].fillna(1)
        elif "PRICE" in tx.columns:
            tx["SPEND"] = tx["PRICE"].fillna(0)
        else:
            tx["SPEND"] = 0.0

    if "UNITS" not in tx.columns:
        tx["UNITS"] = tx["QUANTITY"].fillna(1) if "QUANTITY" in tx.columns else 1.0

    if "BASKET_NUM" not in tx.columns:
        tx["BASKET_NUM"] = (
            tx["HSHD_NUM"].astype(str) + "_" + tx["PURCHASE_DT"].dt.strftime("%Y%m%d").fillna("NA")
        )

    return tx, hh, prod


def build_household_rfm(tx, hh):
    ref_date = tx["PURCHASE_DT"].max()
    if pd.isna(ref_date):
        ref_date = pd.Timestamp.today().normalize()

    rfm = (
        tx.groupby("HSHD_NUM", dropna=False)
        .agg(
            total_spend=("SPEND", "sum"),
            avg_spend=("SPEND", "mean"),
            total_units=("UNITS", "sum"),
            total_rows=("SPEND", "count"),
            n_baskets=("BASKET_NUM", "nunique"),
            first_purchase=("PURCHASE_DT", "min"),
            last_purchase=("PURCHASE_DT", "max"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (ref_date - rfm["last_purchase"]).dt.days.fillna(999).clip(lower=0)
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days.fillna(0).clip(lower=0)
    rfm["avg_basket_value"] = np.where(rfm["n_baskets"] > 0, rfm["total_spend"] / rfm["n_baskets"], rfm["total_spend"])

    merged = pd.merge(rfm, hh, on="HSHD_NUM", how="left")
    return merged