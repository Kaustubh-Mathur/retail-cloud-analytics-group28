# Retail Cloud Analytics Platform (Group 28)

End-to-end cloud data engineering, full-stack web application, and machine learning analytics pipeline built on **Microsoft Azure** using retail household, product, and transaction datasets.

## Live Deployment & Cloud Architecture
* **Web Application (Azure App Service - Linux):** https://retail-analytics-group28-gbbje0dyatgghmgc.swedencentral-01.azurewebsites.net/
* **Cloud Relational Database (Azure SQL Database):** `sql-retail-group28.database.windows.net` (`retail_db`)
* **CI/CD Pipeline:** Automated GitHub Actions deployment (`.github/workflows`) triggered on pushes to `main`.

## Repository Structure
* `app.py` — Flask web application (Auth, Household Lookup, CSV Upload, Dashboard)
* `templates/` — Responsive HTML/Jinja2 UI templates
* `requirements.txt` — Production Python dependencies for Azure App Service
* `clean_data.py` — Data cleaning and preprocessing script for raw CSVs
* `load_to_azure.py` — ETL ingestion script loading cleaned tables into Azure SQL
* `ml_analytics/` — Machine Learning & Data Science pipelines (`db_and_features.py`, `req1_demographics.py`, `req7_basket.py`, `req8_churn.py`, `req9_clv.py`, `run_all_and_chart.py`)
* `results/` — Generated visualizations (`.png`), predictions (`.csv`), and metrics log

## Key Features & ML Results

1. **Interactive Web Application (`app.py`)**
   * **User Authentication:** Secure registration and login flow.
   * **Household Transaction Lookup:** Real-time SQL `JOIN` across `transactions`, `households`, and `products` (`HSHD_NUM = 10` returns 362 sorted records).
   * **Dynamic CSV Upload:** Append new household, product, or transaction batches directly to Azure SQL.
   * **Executive Analytics Dashboard:** Interactive spend, regional, and category breakdowns.

2. **Machine Learning & Data Science (`ml_analytics/`)**
   * **Demographics & K-Means Segmentation (`req1_demographics.py`):** Identified three distinct household segments across 347 active households—VIP High-Spend ($3,962.85 mean spend), Core Active ($1,570.72), and Lapsed ($887.75)—and confirmed higher engagement among larger households (5+ members) and $150K+ income brackets.
   * **Market Basket Analysis (`req7_basket.py`):** Evaluated 8,921 baskets, identifying `PHARMA -> NON-FOOD` (Support: `0.0512`, Confidence: `62.6%`, Lift: `1.426`) along with top commodity-level cross-sell pairs.
   * **Churn Prediction (`req8_churn.py`):** Trained
