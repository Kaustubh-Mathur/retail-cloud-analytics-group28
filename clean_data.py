import pandas as pd
ignore_warnings = True # Set to False to see warnings 
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")
# 1. Load datasets
households = pd.read_csv('400_households.csv')
transactions = pd.read_csv('400_transactions.csv')
products = pd.read_csv('400_products.csv')

# 2. Strip whitespace from column names and string values
for df in [households, transactions, products]:
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

# 3. Parse transaction date (verify exact column name, usually PURCHASE_ or DATE)
date_col = 'PURCHASE_' if 'PURCHASE_' in transactions.columns else 'DATE'
transactions['TX_DATE'] = pd.to_datetime(transactions[date_col])

# 4. Filter for Aug (8), Sep (9), Oct (10), Nov (11), Dec (12)
target_months = [8, 9, 10, 11, 12]
tx_sub = transactions[transactions['TX_DATE'].dt.month.isin(target_months)].copy()

# 5. Keep ONLY the last 7 days of each month (per year present in dataset)
tx_sub['YEAR'] = tx_sub['TX_DATE'].dt.year
tx_sub['MONTH'] = tx_sub['TX_DATE'].dt.month

# Find max date per (YEAR, MONTH) and keep dates within 6 days of max date (7 days total)
max_dates = tx_sub.groupby(['YEAR', 'MONTH'])['TX_DATE'].transform('max')
filtered_tx = tx_sub[(max_dates - tx_sub['TX_DATE']).dt.days <= 6].copy()

# Drop helper columns and standardize date format
filtered_tx.drop(columns=['YEAR', 'MONTH'], inplace=True)
filtered_tx['TX_DATE'] = filtered_tx['TX_DATE'].dt.strftime('%Y-%m-%d')

# 6. Save cleaned files for Azure upload
households.to_csv('clean_households.csv', index=False)
filtered_tx.to_csv('clean_transactions.csv', index=False)
products.to_csv('clean_products.csv', index=False)

print(f"Original transactions: {len(transactions)} -> Filtered transactions: {len(filtered_tx)}")