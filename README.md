# SmartStock — Flexible Inventory Analytics & Demand Forecasting

SmartStock accepts CSV and Excel (`.xlsx`, `.xls`) files even when column names and column order differ. It previews the uploaded data, suggests column mappings, and lets the user manually map fields before analysis.

## Run on Windows PowerShell

From this project folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Input columns

Required concepts (column names can be anything if mapped correctly):
- Month/date
- Product ID/SKU **or** product name
- Units sold / sales quantity
- Current stock / inventory
- Lead time in days

Optional concepts:
- Product name or product ID (whichever was not used above)
- Category
- Historical stockout label (`1` = stockout, `0` = no stockout)

The app automatically suggests mappings for common headers such as `Date`, `Order_Date`, `SKU`, `Item_Code`, `Sales`, `Qty`, `Warehouse_Qty`, and `Delivery_Days`. If headers are generic (for example, A, B, C), select their meaning manually in the mapping panel.

## What it does

- Validates and standardizes uploaded monthly product data.
- Aggregates duplicate product-month rows.
- Trains a Random Forest regression model on historical monthly sales.
- Forecasts next-month demand per product.
- Estimates inventory risk and suggested reorder quantities.
- Displays demand forecasts, historical sales trends, and model evaluation metrics.
- Exports forecast results to CSV.

## Data guidance and limitations

Use one row per product per month where possible. If there are multiple transactions per product-month, the app sums units sold and takes the last stock and lead-time values in the uploaded row order. Ideally, current stock should represent a clearly defined inventory snapshot (for example, month-end stock). At least four monthly records per product are needed for a basic forecast; 12–24+ months is preferable. Forecasts are educational estimates, not production purchasing advice.
