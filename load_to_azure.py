import os
import urllib
import pandas as pd
from sqlalchemy import create_engine, text


BASE_DIR = "/Users/kaus/Desktop/UC/Senior/CC/GroupProject"

# Azure SQL Server details
SERVER = "sql-retail-group28.database.windows.net"
DATABASE = "retail_db"
USERNAME = "sqladmin"
PASSWORD = "RetailCloud2026!"  # <-- Replace with your sqladmin password
DRIVER = "{ODBC Driver 18 for SQL Server}"

conn_str = (
    f"DRIVER={DRIVER};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    f"Encrypt=yes;TrustServerCertificate=no;"
)
params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df.replace({"null": None, "NULL": None, "": None})


print("1. Loading households...")
h_df = clean_nulls(pd.read_csv(os.path.join(BASE_DIR, "clean_households.csv")))
h_df.to_sql("households", engine, if_exists="append", index=False, chunksize=200)
print(f"   Inserted {len(h_df):,} households.")

print("2. Loading products...")
p_df = clean_nulls(pd.read_csv(os.path.join(BASE_DIR, "clean_products.csv")))
p_df.to_sql("products", engine, if_exists="append", index=False, chunksize=1000)
print(f"   Inserted {len(p_df):,} products.")

print("3. Loading transactions...")
t_df = clean_nulls(pd.read_csv(os.path.join(BASE_DIR, "clean_transactions.csv")))

# Align date column name to match SQL schema (PURCHASE_DATE)
if "TX_DATE" in t_df.columns and "PURCHASE_DATE" not in t_df.columns:
    t_df = t_df.rename(columns={"TX_DATE": "PURCHASE_DATE"})
elif "PURCHASE_" in t_df.columns and "PURCHASE_DATE" not in t_df.columns:
    t_df = t_df.rename(columns={"PURCHASE_": "PURCHASE_DATE"})
elif "DATE" in t_df.columns and "PURCHASE_DATE" not in t_df.columns:
    t_df = t_df.rename(columns={"DATE": "PURCHASE_DATE"})

# Keep only columns present in the SQL transactions table
expected_tx_cols = [
    "BASKET_NUM", "HSHD_NUM", "PURCHASE_DATE", "PRODUCT_NUM",
    "SPEND", "UNITS", "STORE_R", "WEEK_NUM", "YEAR"
]
t_df = t_df[[c for c in expected_tx_cols if c in t_df.columns]]

t_df.to_sql("transactions", engine, if_exists="append", index=False, chunksize=1000)
print(f"   Inserted {len(t_df):,} transactions.")

print("\n4. Verifying row counts in Azure SQL:")
with engine.connect() as conn:
    for tbl in ["households", "products", "transactions"]:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        print(f"   {tbl}: {cnt:,} rows")

    # Verify Requirement #3 default household (HSHD_NUM = 10)
    h10_cnt = conn.execute(
        text("SELECT COUNT(*) FROM transactions WHERE HSHD_NUM = 10")
    ).scalar()
    print(f"   Transactions for HSHD_NUM = 10: {h10_cnt:,} rows")