import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="SmartStock | Upload & Forecast", page_icon="📦", layout="wide")
st.title("📦 SmartStock")
st.caption("Upload monthly sales data to forecast demand and assess inventory risk.")

REQUIRED = {"month", "units_sold", "current_stock", "lead_time_days"}
# Common names from retail, warehouse, ERP, POS and e-commerce exports.
ALIASES = {
    "date": "month", "sales_date": "month", "order_date": "month", "transaction_date": "month",
    "period": "month", "sales_month": "month", "month_year": "month", "year_month": "month",
    "product": "product_name", "item": "product_name", "item_name": "product_name", "description": "product_name",
    "sku": "product_id", "item_id": "product_id", "item_code": "product_id", "product_code": "product_id",
    "productid": "product_id", "sku_id": "product_id", "material_number": "product_id",
    "sales": "units_sold", "quantity_sold": "units_sold", "qty_sold": "units_sold",
    "qty": "units_sold", "quantity": "units_sold", "units": "units_sold", "total_sales": "units_sold",
    "sales_quantity": "units_sold", "sold_quantity": "units_sold", "unitsales": "units_sold",
    "stock": "current_stock", "inventory": "current_stock", "available_stock": "current_stock",
    "available_quantity": "current_stock", "warehouse_qty": "current_stock", "on_hand": "current_stock",
    "stock_on_hand": "current_stock", "closing_stock": "current_stock", "inventory_qty": "current_stock",
    "supplier_lead_time": "lead_time_days", "lead_time": "lead_time_days", "delivery_days": "lead_time_days",
    "supplier_days": "lead_time_days", "leadtime": "lead_time_days",
    "stockout_flag": "stockout", "out_of_stock": "stockout", "is_stockout": "stockout",
    "product_category": "category", "type": "category",
}
FIELD_LABELS = {
    "month": "Date / Month", "product_id": "Product ID / SKU (or map Product Name instead)",
    "product_name": "Product Name (or map Product ID instead)", "units_sold": "Units Sold / Sales Quantity",
    "current_stock": "Current Stock / Inventory", "lead_time_days": "Lead Time (days)",
    "category": "Category (optional)", "stockout": "Historical Stockout Flag (optional)",
}


def make_sample():
    rng = np.random.default_rng(12)
    rows = []
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    products = [
        ("P001", "Wireless Mouse", "Electronics", 115, 5, 85),
        ("P002", "Keyboard", "Electronics", 70, 7, 65),
        ("P003", "Water Bottle", "Lifestyle", 90, 4, 110),
    ]
    for pid, name, category, base, lead, stock in products:
        for i, month in enumerate(months):
            seasonal = 1 + 0.18 * np.sin(2 * np.pi * i / 12)
            trend = 1 + i * 0.008
            sold = max(0, int(rng.normal(base * seasonal * trend, base * 0.09)))
            rows.append({
                "month": month.strftime("%Y-%m"),
                "product_id": pid,
                "product_name": name,
                "category": category,
                "units_sold": sold,
                "current_stock": stock,
                "lead_time_days": lead,
            })
    return pd.DataFrame(rows)

def load_file(uploaded):
    suffix = uploaded.name.lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(uploaded)
    if suffix.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Upload a CSV or Excel file.")

def clean_header(value):
    text = str(value).strip().lower()
    text = "_".join(text.replace("-", " ").replace("/", " ").split())
    return text

def suggest_column(columns, canonical):
    cleaned = {col: clean_header(col) for col in columns}
    for col, name in cleaned.items():
        if name == canonical or ALIASES.get(name) == canonical:
            return col
    # Conservative partial matches for common compound headers.
    hints = {
        "month": ("date", "month", "period"),
        "product_id": ("sku", "product_code", "item_code", "product_id"),
        "product_name": ("product_name", "item_name", "product", "item"),
        "units_sold": ("sales", "sold", "quantity", "units", "qty"),
        "current_stock": ("stock", "inventory", "on_hand", "warehouse"),
        "lead_time_days": ("lead", "delivery"),
        "category": ("category", "type"),
        "stockout": ("stockout", "out_of_stock"),
    }
    for col, name in cleaned.items():
        if any(hint in name for hint in hints.get(canonical, ())):
            return col
    return None

def prepare_data(raw, mapping):
    df = raw.copy()
    selected_sources = [source for source in mapping.values() if source and source != "(Not available)"]
    # Do not allow the same source column to be assigned to multiple fields.
    if len(set(selected_sources)) != len(selected_sources):
        raise ValueError("A source column has been mapped more than once. Choose a different column for each field.")
    rename = {source: target for target, source in mapping.items() if source and source != "(Not available)"}
    df = df.rename(columns={source: target for source, target in rename.items()})
    required = {"month", "units_sold", "current_stock", "lead_time_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Please map these required fields: " + ", ".join(sorted(missing)))
    if "product_id" not in df.columns and "product_name" not in df.columns:
        raise ValueError("Map either Product ID / SKU or Product Name.")
    if "product_id" not in df.columns:
        df["product_id"] = df["product_name"].astype(str)
    if "product_name" not in df.columns:
        df["product_name"] = df["product_id"].astype(str)
    if "category" not in df.columns:
        df["category"] = "Unspecified"

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    for col in ["units_sold", "current_stock", "lead_time_days"]:
        # Handles values such as "1,200" from spreadsheet exports.
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "stockout" in df.columns:
        df["stockout"] = pd.to_numeric(df["stockout"], errors="coerce")

    df = df.dropna(subset=["month", "units_sold", "current_stock", "lead_time_days", "product_id"])
    if df.empty:
        raise ValueError("No valid rows remain. Check the selected date and numeric columns.")
    if (df[["units_sold", "current_stock", "lead_time_days"]] < 0).any().any():
        raise ValueError("Sales, stock and lead time cannot be negative.")
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["category"] = df["category"].astype(str)
    aggregations = {
        "units_sold": ("units_sold", "sum"),
        "current_stock": ("current_stock", "last"),
        "lead_time_days": ("lead_time_days", "last"),
    }
    if "stockout" in df.columns:
        aggregations["stockout"] = ("stockout", "max")
    df = df.groupby(["product_id", "product_name", "category", "month"], as_index=False).agg(**aggregations)
    return df.sort_values(["product_id", "month"]).reset_index(drop=True)

def build_supervised(df):
    rows = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("month").reset_index(drop=True)
        sales = g["units_sold"].to_numpy(dtype=float)
        for i in range(3, len(g)):
            rows.append({
                "product_id": pid,
                "month": g.loc[i, "month"],
                "lag_1": sales[i-1],
                "lag_2": sales[i-2],
                "lag_3": sales[i-3],
                "rolling_mean_3": float(np.mean(sales[i-3:i])),
                "month_num": int(g.loc[i, "month"].month),
                "current_stock": float(g.loc[i, "current_stock"]),
                "lead_time_days": float(g.loc[i, "lead_time_days"]),
                "target_demand": float(sales[i]),
                "stockout": int(g.loc[i, "stockout"]) if "stockout" in g.columns and pd.notna(g.loc[i, "stockout"]) else np.nan,
            })
    return pd.DataFrame(rows)

FEATURES = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month_num", "current_stock", "lead_time_days"]

def forecast_products(df, reg):
    results = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("month").reset_index(drop=True)
        if len(g) < 3:
            continue
        sales = list(g["units_sold"].astype(float))
        last = g.iloc[-1]
        next_month = (last["month"] + pd.offsets.MonthBegin(1))
        x = pd.DataFrame([{
            "lag_1": sales[-1],
            "lag_2": sales[-2],
            "lag_3": sales[-3],
            "rolling_mean_3": float(np.mean(sales[-3:])),
            "month_num": int(next_month.month),
            "current_stock": float(last["current_stock"]),
            "lead_time_days": float(last["lead_time_days"]),
        }])
        predicted = max(0.0, float(reg.predict(x[FEATURES])[0]))
        lead_days = max(1.0, float(last["lead_time_days"]))
        # Monthly demand converted to a daily rate for lead-time coverage.
        lead_demand = predicted * min(lead_days, 30.0) / 30.0
        stock = float(last["current_stock"])
        buffer_days = 7
        target_stock = predicted * min(lead_days + buffer_days, 30.0) / 30.0
        reorder = max(0, int(np.ceil(target_stock - stock)))
        risk = "CRITICAL" if stock < lead_demand * 0.5 else "HIGH" if stock < lead_demand else "MEDIUM" if stock < target_stock else "LOW"
        results.append({
            "product_id": pid,
            "product_name": last["product_name"],
            "category": last["category"],
            "last_month": last["month"].strftime("%Y-%m"),
            "predicted_next_month_sales": round(predicted, 1),
            "current_stock": int(round(stock)),
            "lead_time_days": int(round(lead_days)),
            "estimated_lead_time_demand": round(lead_demand, 1),
            "stockout_risk": risk,
            "suggested_reorder_qty": reorder,
        })
    return pd.DataFrame(results)

st.sidebar.header("Upload sales history")
uploaded = st.sidebar.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])
sample = make_sample()
st.sidebar.download_button(
    "Download sample CSV template",
    sample.to_csv(index=False).encode("utf-8"),
    file_name="smartstock_sample_monthly_sales.csv",
    mime="text/csv"
)
st.sidebar.caption("Column names and order can be different. Use the mapping panel to tell SmartStock what each column means.")

if uploaded is None:
    st.info("Upload your monthly product sales file, or download the sample CSV template to try the app.")
    st.subheader("Example input data")
    st.dataframe(sample.head(8), use_container_width=True, hide_index=True)
    st.stop()

try:
    raw = load_file(uploaded)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

if raw.empty or len(raw.columns) == 0:
    st.error("The uploaded file has no readable rows or columns.")
    st.stop()

st.subheader("Step 1 — Preview your uploaded file")
st.caption(f"Detected {len(raw):,} rows and {len(raw.columns)} columns. Column order does not matter.")
st.dataframe(raw.head(10), use_container_width=True, hide_index=True)

st.subheader("Step 2 — Map your columns")
st.write("Select which uploaded column represents each SmartStock field. Suggestions are filled automatically when recognizable names are found.")
source_options = ["(Not available)"] + list(raw.columns)
fields = ["month", "product_id", "product_name", "units_sold", "current_stock", "lead_time_days", "category", "stockout"]
required_fields = {"month", "units_sold", "current_stock", "lead_time_days"}
mapping = {}
cols = st.columns(2)
for i, field in enumerate(fields):
    suggested = suggest_column(raw.columns, field)
    default_index = source_options.index(suggested) if suggested in source_options else 0
    with cols[i % 2]:
        selected = st.selectbox(
            FIELD_LABELS[field], source_options, index=default_index,
            key=f"map_{field}",
            help="Required field" if field in required_fields else "Optional field. Map either Product ID or Product Name."
        )
    if selected != "(Not available)":
        mapping[field] = selected

if not (mapping.get("product_id") or mapping.get("product_name")):
    st.warning("Map either Product ID / SKU or Product Name to identify products.")
if st.button("Validate columns and analyze data", type="primary"):
    st.session_state["smartstock_mapping"] = mapping
    st.session_state["smartstock_file_name"] = uploaded.name

if "smartstock_mapping" not in st.session_state or st.session_state.get("smartstock_file_name") != uploaded.name:
    st.info("Confirm the column mapping above to continue.")
    st.stop()

try:
    data = prepare_data(raw, st.session_state["smartstock_mapping"])
except Exception as exc:
    st.error(f"Could not process file: {exc}")
    st.stop()

st.success(f"Validated {len(data):,} product-month records across {data['product_id'].nunique()} products.")
with st.expander("Preview cleaned and standardized data"):
    st.dataframe(data, use_container_width=True, hide_index=True)

supervised = build_supervised(data)
if len(supervised) < 6:
    st.error("Not enough history to train a useful demo model. Provide at least 4 monthly records per product; 12–24 months is preferable.")
    st.stop()

# A simple chronological holdout by row order. For production, use a date-based panel split
# that holds out the latest month(s) for every product.
supervised = supervised.sort_values(["month", "product_id"]).reset_index(drop=True)
split = max(1, int(len(supervised) * 0.8))
train = supervised.iloc[:split]
test = supervised.iloc[split:]
if len(test) == 0:
    train = supervised.iloc[:-1]
    test = supervised.iloc[-1:]

reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42)
reg.fit(train[FEATURES], train["target_demand"])
if len(test):
    predictions = reg.predict(test[FEATURES])
    mae = mean_absolute_error(test["target_demand"], predictions)
    rmse = np.sqrt(mean_squared_error(test["target_demand"], predictions))
else:
    mae, rmse = np.nan, np.nan

forecasts = forecast_products(data, reg)
if forecasts.empty:
    st.error("No product has enough history for forecasting.")
    st.stop()

# Train a classifier only if user supplies historical stockout labels.
classifier = None
classifier_score = None
if "stockout" in data.columns:
    labelled = supervised.dropna(subset=["stockout"])
    if len(labelled) >= 10 and labelled["stockout"].nunique() == 2:
        classifier = RandomForestClassifier(
            n_estimators=150, max_depth=7, class_weight="balanced", random_state=42
        )
        classifier.fit(labelled[FEATURES], labelled["stockout"].astype(int))
        latest_rows = []
        for _, row in forecasts.iterrows():
            g = data[data["product_id"] == row["product_id"]].sort_values("month")
            sales = g["units_sold"].to_numpy(dtype=float)
            latest_rows.append({
                "lag_1": sales[-1], "lag_2": sales[-2], "lag_3": sales[-3],
                "rolling_mean_3": float(np.mean(sales[-3:])),
                "month_num": int((g.iloc[-1]["month"] + pd.offsets.MonthBegin(1)).month),
                "current_stock": float(row["current_stock"]),
                "lead_time_days": float(row["lead_time_days"]),
            })
        probs = classifier.predict_proba(pd.DataFrame(latest_rows)[FEATURES])[:, 1]
        forecasts["classifier_stockout_score"] = np.round(probs, 3)

st.header("Inventory intelligence")
a, b, c, d = st.columns(4)
a.metric("Products analyzed", f"{len(forecasts)}")
b.metric("Total current stock", f"{forecasts['current_stock'].sum():,} units")
c.metric("High / critical risk", int(forecasts["stockout_risk"].isin(["HIGH", "CRITICAL"]).sum()))
d.metric("Average predicted monthly sales", f"{forecasts['predicted_next_month_sales'].mean():.1f}")

st.subheader("Demand forecast and reorder recommendations")
st.dataframe(forecasts, use_container_width=True, hide_index=True)

csv_bytes = forecasts.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast results (CSV)", csv_bytes, "smartstock_forecast_results.csv", "text/csv")

st.subheader("Predicted demand by product")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(forecasts["product_name"], forecasts["predicted_next_month_sales"])
ax.set_ylabel("Predicted units next month")
ax.set_xlabel("Product")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Historical sales trend")
chosen = st.selectbox("Choose a product", data["product_name"].drop_duplicates().tolist())
history = data[data["product_name"] == chosen].sort_values("month")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["month"], history["units_sold"], marker="o")
ax.set_ylabel("Units sold")
ax.set_xlabel("Month")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("Model evaluation and interpretation"):
    st.write(f"Demand model holdout MAE: **{mae:.2f} units**")
    st.write(f"Demand model holdout RMSE: **{rmse:.2f} units**")
    st.write("The demand model is a Random Forest Regressor trained on lagged monthly sales and calendar/inventory features.")
    if classifier is not None:
        st.write("A Random Forest Classifier was trained because the uploaded file contained both historical stockout classes (0 and 1).")
        st.write("Classifier scores are available in the results table. Validate with a chronological holdout before operational use.")
    else:
        st.info("No usable historical stockout labels were supplied. The app therefore uses a transparent inventory coverage rule for stockout risk instead of pretending to train a classifier. To train the classifier, add a `stockout` column with 1 for an observed stockout and 0 otherwise.")
    st.caption("Forecasts from short histories are uncertain. 12–24+ months per product is preferable. The sample data is synthetic; test with real historical sales before business use.")
