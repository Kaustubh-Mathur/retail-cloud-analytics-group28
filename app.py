import os
import urllib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "group28-retail-secret-2026")

# Azure SQL Connection Settings
SERVER = os.environ.get("DB_SERVER", "sql-retail-group28.database.windows.net")
DATABASE = os.environ.get("DB_NAME", "retail_db")
USERNAME = os.environ.get("DB_USER", "sqladmin")
PASSWORD = os.environ.get("DB_PASS", "")
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


# --- REQUIREMENT 2: Registration & Login (Username, Password, Email) ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form["username"].strip()
        email = request.form["email"].strip()
        pwd_hash = generate_password_hash(request.form["password"], method="pbkdf2:sha256")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO app_users (username, email, password_hash) VALUES (:u, :e, :p)"),
                    {"u": uname, "e": email, "p": pwd_hash},
                )
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration error: {str(e)}", "danger")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["username"].strip()
        pwd = request.form["password"]
        with engine.connect() as conn:
            user = conn.execute(
                text("SELECT * FROM app_users WHERE username = :u"), {"u": uname}
            ).mappings().first()
        if user and check_password_hash(user["password_hash"], pwd):
            session["user"] = user["username"]
            session["email"] = user["email"]
            return redirect(url_for("hshd_search"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --- REQUIREMENTS 3 & 4: Default HSHD_NUM=10 & Interactive Search ---
@app.route("/", methods=["GET", "POST"])
def hshd_search():
    if "user" not in session:
        return redirect(url_for("login"))

    raw_hshd = request.form.get("hshd_num", "10").strip()
    try:
        hshd_num = int(raw_hshd)
    except ValueError:
        hshd_num = 10
        flash("Invalid household number entered; showing HSHD_NUM = 10.", "warning")

    query = text("""
        SELECT
            t.HSHD_NUM,
            t.BASKET_NUM,
            t.PURCHASE_DATE AS DATE,
            t.PRODUCT_NUM,
            p.DEPARTMENT,
            p.COMMODITY,
            t.SPEND,
            t.UNITS,
            t.STORE_R,
            t.WEEK_NUM,
            t.YEAR,
            h.L,
            h.AGE_RANGE,
            h.MARITAL,
            h.INCOME_RANGE,
            h.HOMEOWNER,
            h.HSHD_COMPOSITION,
            h.HH_SIZE,
            h.CHILDREN,
            p.BRAND_TY,
            p.NATURAL_ORGANIC_FLAG
        FROM transactions t
        INNER JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE t.HSHD_NUM = :hshd
        ORDER BY
            t.HSHD_NUM ASC,
            t.BASKET_NUM ASC,
            t.PURCHASE_DATE ASC,
            t.PRODUCT_NUM ASC,
            p.DEPARTMENT ASC,
            p.COMMODITY ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"hshd": hshd_num})

    records = df.to_dict(orient="records")
    return render_template("hshd_search.html", records=records, hshd_num=hshd_num)


# --- REQUIREMENT 5: CSV Data Upload Application ---
@app.route("/upload", methods=["GET", "POST"])
def upload_data():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        table_target = request.form.get("table_type")
        file = request.files.get("csv_file")

        if table_target not in ["households", "transactions", "products"]:
            flash("Invalid target table selected.", "danger")
            return redirect(url_for("upload_data"))

        if not file or not file.filename.endswith(".csv"):
            flash("Please upload a valid .csv file.", "danger")
            return redirect(url_for("upload_data"))

        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            df = df.replace({"null": None, "NULL": None, "": None})

            if table_target == "transactions":
                if "PURCHASE_DATE" not in df.columns:
                    if "TX_DATE" in df.columns:
                        df = df.rename(columns={"TX_DATE": "PURCHASE_DATE"})
                    elif "PURCHASE_" in df.columns:
                        df = df.rename(columns={"PURCHASE_": "PURCHASE_DATE"})
                    elif "DATE" in df.columns:
                        df = df.rename(columns={"DATE": "PURCHASE_DATE"})
                expected_cols = [
                    "BASKET_NUM", "HSHD_NUM", "PURCHASE_DATE", "PRODUCT_NUM",
                    "SPEND", "UNITS", "STORE_R", "WEEK_NUM", "YEAR"
                ]
                df = df[[c for c in expected_cols if c in df.columns]]
            elif table_target == "households":
                expected_cols = [
                    "HSHD_NUM", "L", "AGE_RANGE", "MARITAL", "INCOME_RANGE",
                    "HOMEOWNER", "HSHD_COMPOSITION", "HH_SIZE", "CHILDREN"
                ]
                df = df[[c for c in expected_cols if c in df.columns]]
            elif table_target == "products":
                expected_cols = [
                    "PRODUCT_NUM", "DEPARTMENT", "COMMODITY",
                    "BRAND_TY", "NATURAL_ORGANIC_FLAG"
                ]
                df = df[[c for c in expected_cols if c in df.columns]]

            # Upsert primary keys for households/products so re-uploading doesn't fail on PK conflicts
            with engine.begin() as conn:
                if table_target == "households" and "HSHD_NUM" in df.columns:
                    for val in df["HSHD_NUM"].dropna().unique():
                        conn.execute(text("DELETE FROM households WHERE HSHD_NUM = :id"), {"id": int(val)})
                elif table_target == "products" and "PRODUCT_NUM" in df.columns:
                    for val in df["PRODUCT_NUM"].dropna().unique():
                        conn.execute(text("DELETE FROM products WHERE PRODUCT_NUM = :id"), {"id": int(val)})

            df.to_sql(table_target, engine, if_exists="append", index=False, chunksize=500)
            flash(f"Successfully loaded {len(df):,} rows into '{table_target}'! You can now search the updated records.", "success")
        except Exception as e:
            flash(f"Error loading CSV: {str(e)}", "danger")

    return render_template("upload.html")


# --- REQUIREMENT 6: Interactive Analytics Dashboard ---
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    selected_region = request.args.get("region", "ALL")
    region_filter = "" if selected_region == "ALL" else "WHERE t.STORE_R = :region"
    params = {} if selected_region == "ALL" else {"region": selected_region}

    with engine.connect() as conn:
        regions_df = pd.read_sql("SELECT DISTINCT STORE_R FROM transactions WHERE STORE_R IS NOT NULL ORDER BY STORE_R", conn)
        regions = regions_df["STORE_R"].tolist()

        kpi_query = text(f"""
            SELECT
                COALESCE(SUM(t.SPEND), 0) AS total_sales,
                COUNT(DISTINCT t.BASKET_NUM) AS total_baskets,
                COALESCE(SUM(t.UNITS), 0) AS total_units,
                COUNT(DISTINCT t.HSHD_NUM) AS active_households
            FROM transactions t
            {region_filter}
        """)
        kpis = conn.execute(kpi_query, params).mappings().first()

        dept_df = pd.read_sql(text(f"""
            SELECT TOP 10 p.DEPARTMENT, SUM(t.SPEND) AS total_spend
            FROM transactions t
            JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            {region_filter}
            GROUP BY p.DEPARTMENT
            ORDER BY total_spend DESC
        """), conn, params=params)

        month_df = pd.read_sql(text(f"""
            SELECT MONTH(t.PURCHASE_DATE) AS month_num, SUM(t.SPEND) AS total_spend
            FROM transactions t
            {region_filter}
            GROUP BY MONTH(t.PURCHASE_DATE)
            ORDER BY month_num
        """), conn, params=params)

        income_df = pd.read_sql(text(f"""
            SELECT COALESCE(h.INCOME_RANGE, 'Unknown') AS income_range, SUM(t.SPEND) AS total_spend
            FROM transactions t
            JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
            {region_filter}
            GROUP BY COALESCE(h.INCOME_RANGE, 'Unknown')
            ORDER BY total_spend DESC
        """), conn, params=params)

        brand_df = pd.read_sql(text(f"""
            SELECT COALESCE(p.BRAND_TY, 'Unknown') AS brand_ty, SUM(t.SPEND) AS total_spend
            FROM transactions t
            JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            {region_filter}
            GROUP BY COALESCE(p.BRAND_TY, 'Unknown')
        """), conn, params=params)

    total_sales = float(kpis["total_sales"])
    total_baskets = int(kpis["total_baskets"])
    total_units = int(kpis["total_units"])
    spend_per_visit = round(total_sales / total_baskets, 2) if total_baskets > 0 else 0
    units_per_visit = round(total_units / total_baskets, 2) if total_baskets > 0 else 0

    month_names = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_labels = [month_names.get(m, str(m)) for m in month_df["month_num"].tolist()]

    return render_template(
        "dashboard.html",
        regions=regions,
        selected_region=selected_region,
        total_sales=round(total_sales, 2),
        total_baskets=total_baskets,
        spend_per_visit=spend_per_visit,
        units_per_visit=units_per_visit,
        active_households=int(kpis["active_households"]),
        dept_labels=dept_df["DEPARTMENT"].tolist(),
        dept_values=dept_df["total_spend"].round(2).tolist(),
        month_labels=month_labels,
        month_values=month_df["total_spend"].round(2).tolist(),
        income_labels=income_df["income_range"].tolist(),
        income_values=income_df["total_spend"].round(2).tolist(),
        brand_labels=brand_df["brand_ty"].tolist(),
        brand_values=brand_df["total_spend"].round(2).tolist(),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)