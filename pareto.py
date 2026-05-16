import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise ImportError(
        "LightGBM is not installed. Run: pip install lightgbm"
    )


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "/home/hoangb/BOS/forecasting_pipeline_refactor_package/pre POS Data.xlsx"

OUTPUT_DIR = "forecasting_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARETO_THRESHOLD = 0.8
FORECAST_HORIZON_WEEKS = 12

DATE_COL = "Week"
SKU_COL = "Code"
DIVISION_COL = "Division"
SALES_COL = "Cases Sold - Final"
PRODUCT_COL = "Product name"

OPEN_STOCK_COL = "Open Stock Quantity"
TOTAL_DIVISION_STOCK_COL = "total division stock (cases)"


# =====================================================
# 1. LOAD DATA
# =====================================================

print("Loading data...")

dfr = pd.read_excel(INPUT_FILE)

required_cols = [DATE_COL, SKU_COL, DIVISION_COL, SALES_COL]

missing_cols = [c for c in required_cols if c not in dfr.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

dfr[DATE_COL] = pd.to_datetime(dfr[DATE_COL], errors="coerce")
dfr = dfr.dropna(subset=[DATE_COL, SKU_COL, DIVISION_COL])

dfr = dfr.sort_values([SKU_COL, DIVISION_COL, DATE_COL]).reset_index(drop=True)

print(f"Raw rows: {len(dfr)}")


# =====================================================
# 2. CLEAN SALES COLUMN
# =====================================================

dfr[SALES_COL] = pd.to_numeric(dfr[SALES_COL], errors="coerce")
dfr.loc[dfr[SALES_COL] < 0, SALES_COL] = np.nan


# =====================================================
# 3. INTERPOLATION
# =====================================================

print("Running interpolation...")

def interpolate_group(group):
    group = group.sort_values(DATE_COL).copy()
    group[SALES_COL] = (
        group[SALES_COL]
        .interpolate(method="linear", limit_direction="both")
    )
    return group

dfr = (
    dfr.groupby([SKU_COL, DIVISION_COL], group_keys=False)
       .apply(interpolate_group)
       .reset_index(drop=True)
)


# =====================================================
# 4. KNN IMPUTATION
# =====================================================

print("Running KNN imputation...")

numeric_cols = dfr.select_dtypes(include=[np.number]).columns.tolist()

if SALES_COL not in numeric_cols:
    numeric_cols.append(SALES_COL)

knn_cols = list(dict.fromkeys(numeric_cols))

imputer = KNNImputer(n_neighbors=3)

dfr[knn_cols] = imputer.fit_transform(dfr[knn_cols])

dfr[SALES_COL] = dfr[SALES_COL].clip(lower=0)


# =====================================================
# 5. SET OPEN STOCK QUANTITY
# open stock quantity tuần i =
# total division stock cases tuần i-1
# =====================================================

if TOTAL_DIVISION_STOCK_COL in dfr.columns:
    print("Calculating Open Stock Quantity...")

    dfr[TOTAL_DIVISION_STOCK_COL] = pd.to_numeric(
        dfr[TOTAL_DIVISION_STOCK_COL],
        errors="coerce"
    )

    dfr[OPEN_STOCK_COL] = (
        dfr.groupby([SKU_COL, DIVISION_COL])[TOTAL_DIVISION_STOCK_COL]
           .shift(1)
    )

    dfr[OPEN_STOCK_COL] = dfr[OPEN_STOCK_COL].fillna(0)
else:
    print(
        f"Warning: '{TOTAL_DIVISION_STOCK_COL}' not found. "
        f"Skipping '{OPEN_STOCK_COL}' calculation."
    )


# =====================================================
# 6. PARETO FUNCTIONS
# =====================================================

def pareto_by_sku(df_pareto, sku_code, threshold=0.8):
    temp = (
        df_pareto[df_pareto[SKU_COL] == sku_code]
        .groupby(DIVISION_COL, as_index=False)[SALES_COL]
        .sum()
    )

    temp = temp.sort_values(SALES_COL, ascending=False).reset_index(drop=True)

    total_sales = temp[SALES_COL].sum()

    if total_sales == 0:
        temp["Sales_pct"] = 0
        temp["Cum_pct"] = 0
        return temp, []

    temp["Sales_pct"] = temp[SALES_COL] / total_sales
    temp["Cum_pct"] = temp["Sales_pct"].cumsum()

    key_divisions = temp.loc[
        temp["Cum_pct"].shift(1, fill_value=0) < threshold,
        DIVISION_COL
    ].tolist()

    return temp, key_divisions


def build_pareto_summary(df_pareto, threshold=0.8):
    records = []

    for sku_code in df_pareto[SKU_COL].unique():
        temp, top_divs = pareto_by_sku(
            df_pareto,
            sku_code,
            threshold=threshold
        )

        records.append({
            SKU_COL: sku_code,
            "Num_Divisions": len(temp),
            "Num_Divisions_80pct": len(top_divs),
            "Top_80pct_Divisions": ", ".join(map(str, top_divs))
        })

    return (
        pd.DataFrame(records)
        .sort_values(SKU_COL)
        .reset_index(drop=True)
    )


def filter_by_pareto(dfr, summary_df):
    pairs = (
        summary_df[[SKU_COL, "Top_80pct_Divisions"]]
        .assign(
            Division=lambda x: x["Top_80pct_Divisions"].str.split(", ")
        )
        .explode("Division")
        [[SKU_COL, "Division"]]
    )

    pairs = pairs.dropna()
    pairs = pairs[pairs["Division"] != ""]

    return (
        dfr.merge(
            pairs,
            left_on=[SKU_COL, DIVISION_COL],
            right_on=[SKU_COL, "Division"],
            how="inner"
        )
        .drop(columns=["Division_y"], errors="ignore")
        .rename(columns={"Division_x": DIVISION_COL})
        .reset_index(drop=True)
    )


def plot_pareto_all(df_pareto, filepath):
    with PdfPages(filepath) as pdf:
        for sku_code in df_pareto[SKU_COL].unique():
            pareto_table, _ = pareto_by_sku(
                df_pareto,
                sku_code,
                threshold=PARETO_THRESHOLD
            )

            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.bar(
                pareto_table[DIVISION_COL].astype(str),
                pareto_table[SALES_COL]
            )
            ax1.set_xlabel("Division")
            ax1.set_ylabel(SALES_COL)

            ax2 = ax1.twinx()
            ax2.plot(
                pareto_table[DIVISION_COL].astype(str),
                pareto_table["Cum_pct"],
                marker="o",
                color="red"
            )
            ax2.axhline(PARETO_THRESHOLD, linestyle="--", color="grey")
            ax2.set_ylabel("Cumulative %")
            ax2.set_ylim(0, 1.05)

            plt.title(f"Pareto Chart - SKU {sku_code}")
            plt.xticks(rotation=45)
            plt.tight_layout()

            pdf.savefig(fig)
            plt.close()

    print(f"Saved: {filepath}")


def export_pareto_summary_to_pdf(summary_df, filepath):
    doc = SimpleDocTemplate(filepath, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "Pareto Summary - Top 80% Divisions by SKU",
            styles["Title"]
        )
    )
    elements.append(Spacer(1, 12))

    data = [list(summary_df.columns)] + summary_df.values.tolist()

    table = Table(data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(table)
    doc.build(elements)

    print(f"Saved: {filepath}")


# =====================================================
# 7. RUN PARETO AFTER INTERPOLATION + KNN
# =====================================================

print("Running Pareto filtering...")

df_pareto = (
    dfr.groupby([SKU_COL, DIVISION_COL], as_index=False)
       .agg({SALES_COL: "sum"})
)

pareto_summary = build_pareto_summary(
    df_pareto,
    threshold=PARETO_THRESHOLD
)

pareto_summary_path = os.path.join(OUTPUT_DIR, "pareto_summary.xlsx")
pareto_summary_pdf = os.path.join(OUTPUT_DIR, "pareto_summary.pdf")
pareto_chart_pdf = os.path.join(OUTPUT_DIR, "pareto_all_skus.pdf")

pareto_summary.to_excel(pareto_summary_path, index=False)

export_pareto_summary_to_pdf(
    pareto_summary,
    filepath=pareto_summary_pdf
)

plot_pareto_all(
    df_pareto,
    filepath=pareto_chart_pdf
)

dfr_prt = filter_by_pareto(dfr, pareto_summary)

dfr_prt_path = os.path.join(OUTPUT_DIR, "dfr_prt.xlsx")
dfr_prt.to_excel(dfr_prt_path, index=False)

print(f"Original rows: {len(dfr)}")
print(f"Pareto rows: {len(dfr_prt)}")
print(f"SKUs retained: {dfr_prt[SKU_COL].nunique()}")
print(
    "SKU-Division pairs retained:",
    dfr_prt[[SKU_COL, DIVISION_COL]].drop_duplicates().shape[0]
)


# =====================================================
# 8. FEATURE ENGINEERING FOR FORECASTING
# =====================================================

print("Creating forecasting features...")

forecast_input_df = dfr_prt.copy()

forecast_input_df = forecast_input_df.sort_values(
    [SKU_COL, DIVISION_COL, DATE_COL]
).reset_index(drop=True)

forecast_input_df["weekofyear"] = forecast_input_df[DATE_COL].dt.isocalendar().week.astype(int)
forecast_input_df["month"] = forecast_input_df[DATE_COL].dt.month
forecast_input_df["quarter"] = forecast_input_df[DATE_COL].dt.quarter
forecast_input_df["year"] = forecast_input_df[DATE_COL].dt.year

for lag in [1, 2, 4, 8, 12]:
    forecast_input_df[f"lag_{lag}"] = (
        forecast_input_df
        .groupby([SKU_COL, DIVISION_COL])[SALES_COL]
        .shift(lag)
    )

for window in [4, 8, 12]:
    forecast_input_df[f"rolling_mean_{window}"] = (
        forecast_input_df
        .groupby([SKU_COL, DIVISION_COL])[SALES_COL]
        .shift(1)
        .rolling(window)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

forecast_input_df = forecast_input_df.dropna().reset_index(drop=True)


# =====================================================
# 9. TRAIN + FORECAST
# =====================================================

print("Running forecasting using Pareto output...")

feature_cols = [
    "weekofyear",
    "month",
    "quarter",
    "year",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_12",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_12",
]

forecast_records = []
metrics_records = []

for (sku, division), group in forecast_input_df.groupby([SKU_COL, DIVISION_COL]):
    group = group.sort_values(DATE_COL).copy()

    if len(group) < 20:
        print(f"Skipping {sku} - {division}: not enough data")
        continue

    train_size = int(len(group) * 0.8)

    train_df = group.iloc[:train_size]
    test_df = group.iloc[train_size:]

    X_train = train_df[feature_cols]
    y_train = train_df[SALES_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[SALES_COL]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics_records.append({
        SKU_COL: sku,
        DIVISION_COL: division,
        "MAE": mae,
        "RMSE": rmse,
        "Train_Rows": len(train_df),
        "Test_Rows": len(test_df)
    })

    history = group.copy()

    future_rows = []

    last_date = history[DATE_COL].max()

    for step in range(1, FORECAST_HORIZON_WEEKS + 1):
        future_date = last_date + pd.DateOffset(weeks=step)

        temp = history.sort_values(DATE_COL).copy()

        row = {
            SKU_COL: sku,
            DIVISION_COL: division,
            DATE_COL: future_date,
            "weekofyear": int(future_date.isocalendar().week),
            "month": future_date.month,
            "quarter": future_date.quarter,
            "year": future_date.year,
        }

        for lag in [1, 2, 4, 8, 12]:
            row[f"lag_{lag}"] = temp[SALES_COL].iloc[-lag] if len(temp) >= lag else temp[SALES_COL].mean()

        for window in [4, 8, 12]:
            row[f"rolling_mean_{window}"] = temp[SALES_COL].tail(window).mean()

        X_future = pd.DataFrame([row])[feature_cols]

        pred = model.predict(X_future)[0]
        pred = max(pred, 0)

        row["Forecast_Cases_Sold"] = pred

        future_rows.append(row)

        new_history_row = row.copy()
        new_history_row[SALES_COL] = pred
        history = pd.concat(
            [history, pd.DataFrame([new_history_row])],
            ignore_index=True
        )

    forecast_records.extend(future_rows)


forecast_df = pd.DataFrame(forecast_records)
metrics_df = pd.DataFrame(metrics_records)


# =====================================================
# 10. SAFETY STOCK, REORDER POINT, REQUIRED STOCK
# =====================================================

print("Calculating safety stock, reorder point, and required stock...")

if not forecast_df.empty:
    demand_stats = (
        forecast_input_df
        .groupby([SKU_COL, DIVISION_COL])[SALES_COL]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "Avg_Weekly_Demand",
            "std": "Std_Weekly_Demand"
        })
    )

    forecast_df = forecast_df.merge(
        demand_stats,
        on=[SKU_COL, DIVISION_COL],
        how="left"
    )

    forecast_df["Std_Weekly_Demand"] = forecast_df["Std_Weekly_Demand"].fillna(0)

    Z_SCORE = 1.65
    LEAD_TIME_WEEKS = 2

    forecast_df["Safety_Stock"] = (
        Z_SCORE
        * forecast_df["Std_Weekly_Demand"]
        * np.sqrt(LEAD_TIME_WEEKS)
    )

    forecast_df["Reorder_Point"] = (
        forecast_df["Avg_Weekly_Demand"] * LEAD_TIME_WEEKS
        + forecast_df["Safety_Stock"]
    )

    if OPEN_STOCK_COL in forecast_input_df.columns:
        latest_stock = (
            forecast_input_df
            .sort_values(DATE_COL)
            .groupby([SKU_COL, DIVISION_COL])
            .tail(1)
            [[SKU_COL, DIVISION_COL, OPEN_STOCK_COL]]
        )

        forecast_df = forecast_df.merge(
            latest_stock,
            on=[SKU_COL, DIVISION_COL],
            how="left"
        )

        forecast_df[OPEN_STOCK_COL] = forecast_df[OPEN_STOCK_COL].fillna(0)
    else:
        forecast_df[OPEN_STOCK_COL] = 0

    forecast_df["Required_Stock"] = (
        forecast_df["Forecast_Cases_Sold"]
        + forecast_df["Safety_Stock"]
        - forecast_df[OPEN_STOCK_COL]
    ).clip(lower=0)

    forecast_df["Forecast_Month"] = forecast_df[DATE_COL].dt.to_period("M").astype(str)

    monthly_required_stock = (
        forecast_df
        .groupby([SKU_COL, DIVISION_COL, "Forecast_Month"], as_index=False)
        .agg({
            "Forecast_Cases_Sold": "sum",
            "Safety_Stock": "mean",
            "Reorder_Point": "mean",
            OPEN_STOCK_COL: "last",
            "Required_Stock": "sum"
        })
    )
else:
    monthly_required_stock = pd.DataFrame()


# =====================================================
# 11. EXPORT RESULTS
# =====================================================

forecast_path = os.path.join(OUTPUT_DIR, "forecast_result.xlsx")
metrics_path = os.path.join(OUTPUT_DIR, "forecast_metrics.xlsx")
monthly_stock_path = os.path.join(OUTPUT_DIR, "monthly_required_stock.xlsx")

forecast_df.to_excel(forecast_path, index=False)
metrics_df.to_excel(metrics_path, index=False)
monthly_required_stock.to_excel(monthly_stock_path, index=False)

print("Done.")
print(f"Saved: {dfr_prt_path}")
print(f"Saved: {pareto_summary_path}")
print(f"Saved: {forecast_path}")
print(f"Saved: {metrics_path}")
print(f"Saved: {monthly_stock_path}")
