#!/usr/bin/env python3
"""
End-to-end Ubuntu script: build dfr from pre POS data, run forecasting,
and run inventory optimization/simulation.

No Jupyter needed.

Input:
    /home/hoangb/BOS/forecasting_pipeline_refactor_package/pre POS Data.xlsx

Outputs:
    dfr_prt.xlsx
    forecast_output/summary.xlsx
    forecast_output/backtest_detail.xlsx
    forecast_output/future_forecast.xlsx
    forecast_output/model_metrics.xlsx
    forecast_output/selected_model_summary.xlsx
    forecast_output/forecast_charts.pdf
    forecast_output/inventory_plan.xlsx
    forecast_output/inventory_kpi.xlsx
    forecast_output/required_stock_by_sku_month.xlsx
    forecast_output/inventory_charts.pdf

Run:
    cd /home/hoangb/BOS/forecasting_pipeline_refactor_package
    python run_from_pre_pos_with_inventory.py
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from forecasting_pipeline_lightgbm_arima_hybrid import ForecastConfig, run_pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths and column names
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/hoangb/BOS/forecasting_pipeline_refactor_package")
INPUT_FILE = BASE_DIR / "pre POS Data.xlsx"
PREPARED_FILE = BASE_DIR / "dfr_prt.xlsx"
OUTPUT_DIR = BASE_DIR / "forecast_output"
SHEET_NAME = "Weekly_POS"

DATE_COL = "Week"
CODE_COL = "Code"
PRODUCT_COL = "Product name"
DIVISION_COL = "Division"
TARGET_RAW_COL = "Cases Sold"
TARGET_FINAL_COL = "Cases Sold - Final"
STOCK_COL = "Total Division Stock (Cases)"

# ---------------------------------------------------------------------
# Inventory policy settings
# ---------------------------------------------------------------------
FORECAST_WEEKS = 52
BACKTEST_HORIZON = 8
N_BACKTEST_ORIGINS = 5

SERVICE_LEVEL = 0.95
Z_VALUE = 1.65                 # Approx. z-score for 95% cycle service level
LEAD_TIME_WEEKS = 4             # Change if your real lead time differs
REVIEW_PERIOD_WEEKS = 1         # Weekly planning/replenishment review
MIN_ORDER_QTY = 0.0             # Set >0 if you have minimum order quantity
ORDER_MULTIPLE = 1.0            # Set to pallet/container multiple if needed
SIMULATE_INVENTORY_IF_MISSING = True
INITIAL_STOCK_WEEKS_COVER = 6.0 # Mirrors current policy mentioned in thesis

RANDOM_STATE = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def interpolate_zeros_by_series(group: pd.DataFrame) -> pd.Series:
    """Replace 0 sales with NaN, then linearly interpolate inside each Code-Division series."""
    s = pd.to_numeric(group[TARGET_RAW_COL], errors="coerce").replace(0, np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=group.index)
    out = s.interpolate(method="linear", limit_direction="both")
    return out.fillna(0).clip(lower=0).round(0)


def build_dfr_from_pre_pos(input_file: Path, output_file: Path) -> pd.DataFrame:
    """Load original POS Excel and create the forecasting input dfr_prt.xlsx."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info("Loading original POS data: %s", input_file)
    df = pd.read_excel(input_file, sheet_name=SHEET_NAME, engine="openpyxl")

    required_cols = {DATE_COL, CODE_COL, PRODUCT_COL, DIVISION_COL, TARGET_RAW_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_file.name}: {sorted(missing)}")

    keep_cols = [DATE_COL, CODE_COL, PRODUCT_COL, DIVISION_COL, TARGET_RAW_COL]
    if STOCK_COL in df.columns:
        keep_cols.append(STOCK_COL)

    dfr = df.loc[:, keep_cols].copy()
    dfr[DATE_COL] = pd.to_datetime(dfr[DATE_COL], errors="coerce")
    dfr[TARGET_RAW_COL] = pd.to_numeric(dfr[TARGET_RAW_COL], errors="coerce")
    if STOCK_COL in dfr.columns:
        dfr[STOCK_COL] = pd.to_numeric(dfr[STOCK_COL], errors="coerce")

    dfr = dfr.dropna(subset=[DATE_COL, CODE_COL, DIVISION_COL, TARGET_RAW_COL])
    dfr = dfr[dfr[TARGET_RAW_COL] >= 0]
    dfr = dfr.sort_values([CODE_COL, DIVISION_COL, DATE_COL]).reset_index(drop=True)

    logging.info("Rows after cleaning: %s", f"{len(dfr):,}")
    logging.info("Zero sales rows before interpolation: %s", f"{int((dfr[TARGET_RAW_COL] == 0).sum()):,}")

    dfr["Cases Sold - Interpolated"] = (
        dfr.groupby([CODE_COL, DIVISION_COL], group_keys=False)
        .apply(interpolate_zeros_by_series)
        .astype(float)
    )
    dfr[TARGET_FINAL_COL] = dfr["Cases Sold - Interpolated"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dfr.to_excel(output_file, index=False)
    logging.info("Prepared dfr saved: %s", output_file)
    return dfr


def run_forecast(prepared_file: Path) -> None:
    """Run the existing ARIMA + LightGBM + Hybrid forecasting pipeline."""
    config = ForecastConfig(
        date_col=DATE_COL,
        code_col=CODE_COL,
        division_col=DIVISION_COL,
        target_col=TARGET_FINAL_COL,
        product_col=PRODUCT_COL,
        output_dir=str(OUTPUT_DIR),
        forecast_weeks=FORECAST_WEEKS,
        backtest_horizon=BACKTEST_HORIZON,
        n_backtest_origins=N_BACKTEST_ORIGINS,
        min_train_weeks=24,
        min_total_weeks=30,
        seasonal_arima=False,
        seasonal_period=52,
        zero_as_missing=False,
        missing_week_strategy="interpolate",
        n_jobs=-1,
        use_pmdarima=True,
    )
    logging.info("Starting forecast pipeline...")
    run_pipeline(input_path=prepared_file, config=config, sheet_name=None)
    logging.info("Forecast outputs saved in: %s", OUTPUT_DIR)


def ceil_to_multiple(value: float, multiple: float) -> float:
    if multiple <= 0:
        return max(value, 0.0)
    return math.ceil(max(value, 0.0) / multiple) * multiple


def get_last_stock_or_simulated(group: pd.DataFrame) -> Tuple[float, str]:
    """Use the latest real stock if available; otherwise simulate 6 weeks cover."""
    if STOCK_COL in group.columns:
        stock_series = pd.to_numeric(group[STOCK_COL], errors="coerce").dropna()
        stock_series = stock_series[stock_series >= 0]
        if len(stock_series) > 0:
            return float(stock_series.iloc[-1]), "actual_last_stock"

    if not SIMULATE_INVENTORY_IF_MISSING:
        raise ValueError(
            f"{STOCK_COL} is missing or empty. Enable simulation or provide inventory data."
        )

    demand = pd.to_numeric(group[TARGET_FINAL_COL], errors="coerce").dropna()
    avg_weekly = float(demand.tail(12).mean()) if len(demand) else 0.0
    return avg_weekly * INITIAL_STOCK_WEEKS_COVER, "simulated_6_weeks_cover"


def historical_demand_stats(dfr: pd.DataFrame) -> pd.DataFrame:
    """Compute demand statistics used by the inventory policy."""
    rows = []
    for (code, division), g in dfr.groupby([CODE_COL, DIVISION_COL]):
        s = pd.to_numeric(g[TARGET_FINAL_COL], errors="coerce").dropna().clip(lower=0)
        if len(s) == 0:
            continue
        init_stock, stock_source = get_last_stock_or_simulated(g)
        rows.append(
            {
                "Code": code,
                "Division": division,
                "Product_Name": g[PRODUCT_COL].dropna().iloc[-1] if PRODUCT_COL in g.columns and g[PRODUCT_COL].notna().any() else None,
                "Hist_Avg_Weekly_Demand": float(s.mean()),
                "Hist_Last12_Avg_Weekly_Demand": float(s.tail(12).mean()),
                "Hist_Std_Weekly_Demand": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "Initial_Inventory": float(init_stock),
                "Initial_Inventory_Source": stock_source,
                "Historical_Weeks": int(len(s)),
            }
        )
    return pd.DataFrame(rows)


def normalize_future_forecast(future_df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected inventory columns exist."""
    required = {"Code", "Division", "Week"}
    missing = required - set(future_df.columns)
    if missing:
        raise ValueError(f"future_forecast.xlsx is missing columns: {sorted(missing)}")

    if "Selected_Forecast" in future_df.columns:
        forecast_col = "Selected_Forecast"
    elif "Forecast" in future_df.columns:
        forecast_col = "Forecast"
    else:
        possible = [c for c in future_df.columns if "Forecast" in c]
        if not possible:
            raise ValueError("Cannot find forecast column in future_forecast.xlsx")
        forecast_col = possible[-1]

    out = future_df.copy()
    out["Week"] = pd.to_datetime(out["Week"], errors="coerce")
    out["Forecast_Demand"] = pd.to_numeric(out[forecast_col], errors="coerce").fillna(0).clip(lower=0)
    return out.sort_values(["Code", "Division", "Week"]).reset_index(drop=True)


def simulate_inventory_for_pair(pair_fc: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    """Weekly periodic-review inventory simulation with lead-time order arrivals.

    Important stock convention requested by user:
    - Open_Stock_Qty for week i equals Total Division Stock (Cases) from week i-1.
    - For the first forecast week, Open_Stock_Qty equals the latest historical
      Total Division Stock (Cases). If no real stock exists, it is simulated.
    - In the forecast horizon, Total Division Stock (Cases) is the simulated
      closing/ending inventory after demand consumption for that week.

    Inventory policy correction:
    - Reorder Point (ROP) = forecast demand during lead time + safety stock during lead time.
    - Safety stock for ROP uses lead-time uncertainty: z * weekly_std * sqrt(L).
    - Because this is weekly periodic review, order-up-to target also uses the
      protection period L + R: forecast demand over (lead time + review period)
      + safety stock over (lead time + review period).
    """
    pair_fc = pair_fc.sort_values("Week").reset_index(drop=True)

    demand_std = float(stats.get("Hist_Std_Weekly_Demand", 0.0) or 0.0)
    initial_inventory = float(stats.get("Initial_Inventory", 0.0) or 0.0)

    lead_time_weeks = max(int(LEAD_TIME_WEEKS), 1)
    review_period_weeks = max(int(REVIEW_PERIOD_WEEKS), 1)
    protection_weeks = lead_time_weeks + review_period_weeks

    # Correct formulas.
    safety_stock_lt = Z_VALUE * demand_std * math.sqrt(lead_time_weeks)
    safety_stock_protection = Z_VALUE * demand_std * math.sqrt(protection_weeks)

    # This variable represents beginning/opening stock before receipts and demand.
    # For week 1, it comes from the latest actual Total Division Stock (Cases)
    # or simulated six-week stock if no real stock exists.
    previous_total_division_stock = initial_inventory

    pipeline_orders = []  # list of tuples: (arrival_week_index, order_qty)
    records = []

    forecasts = pair_fc["Forecast_Demand"].astype(float).to_numpy()

    for t, row in pair_fc.iterrows():
        # User-requested convention:
        # Open stock of week i = Total Division Stock (Cases) of week i-1.
        open_stock_qty = previous_total_division_stock

        # Receive any orders due at the beginning of this week.
        arrivals = sum(qty for arrival_t, qty in pipeline_orders if arrival_t == t)
        pipeline_orders = [(arrival_t, qty) for arrival_t, qty in pipeline_orders if arrival_t > t]

        available_stock = open_stock_qty + arrivals
        forecast_demand = float(forecasts[t])

        # Inventory position = available stock + already ordered but not yet arrived.
        pipeline_qty = sum(qty for _, qty in pipeline_orders)
        inventory_position = available_stock + pipeline_qty

        # Forecast demand during lead time for ROP.
        lead_end = min(len(forecasts), t + lead_time_weeks)
        lead_time_demand = float(forecasts[t:lead_end].sum())
        if lead_end < t + lead_time_weeks:
            lead_time_demand += forecast_demand * (t + lead_time_weeks - lead_end)

        # ROP for a continuous-review interpretation.
        reorder_point = lead_time_demand + safety_stock_lt

        # Weekly periodic-review target/order-up-to level uses protection period L + R.
        protection_end = min(len(forecasts), t + protection_weeks)
        protection_period_demand = float(forecasts[t:protection_end].sum())
        if protection_end < t + protection_weeks:
            protection_period_demand += forecast_demand * (t + protection_weeks - protection_end)

        target_inventory_position = protection_period_demand + safety_stock_protection

        # Order-up-to policy. For weekly review, order only enough to bring
        # inventory position to the target level.
        raw_order_qty = max(target_inventory_position - inventory_position, 0.0)
        if raw_order_qty > 0 and raw_order_qty < MIN_ORDER_QTY:
            raw_order_qty = MIN_ORDER_QTY
        order_qty = ceil_to_multiple(raw_order_qty, ORDER_MULTIPLE)

        if order_qty > 0:
            pipeline_orders.append((t + lead_time_weeks, order_qty))

        # Consume forecast demand at the end of the week.
        fulfilled_demand = min(available_stock, forecast_demand)
        lost_sales = max(forecast_demand - available_stock, 0.0)
        ending_inventory = max(available_stock - forecast_demand, 0.0)

        # Current week's Total Division Stock (Cases). This becomes next week's open stock.
        total_division_stock_cases = ending_inventory
        previous_total_division_stock = total_division_stock_cases

        records.append(
            {
                "Code": row["Code"],
                "Division": row["Division"],
                "Product_Name": row.get("Product_Name", None),
                "Week": row["Week"],
                "Selected_Model": row.get("Selected_Model", None),
                "Forecast_Demand": forecast_demand,
                "Open_Stock_Qty": open_stock_qty,
                "Beginning_Inventory": open_stock_qty,  # backward-compatible alias
                "Arrivals": arrivals,
                "Available_Stock_After_Arrivals": available_stock,
                "Pipeline_Before_Order": pipeline_qty,
                "Inventory_Position": inventory_position,
                "Lead_Time_Demand": lead_time_demand,
                "Protection_Period_Demand": protection_period_demand,
                "Safety_Stock_LT": safety_stock_lt,
                "Safety_Stock_Protection": safety_stock_protection,
                "Safety_Stock": safety_stock_lt,  # ROP safety stock
                "Reorder_Point": reorder_point,
                "Target_Inventory_Position": target_inventory_position,
                "Order_Qty": order_qty,
                "Fulfilled_Demand": fulfilled_demand,
                "Lost_Sales": lost_sales,
                "Ending_Inventory": ending_inventory,
                STOCK_COL: total_division_stock_cases,
                "Stockout_Flag": int(lost_sales > 0),
                "Inventory_Cover_Weeks": ending_inventory / max(forecast_demand, 1e-9),
                "Service_Level_Target": SERVICE_LEVEL,
                "Lead_Time_Weeks": lead_time_weeks,
                "Review_Period_Weeks": review_period_weeks,
                "Protection_Period_Weeks": protection_weeks,
                "Initial_Inventory_Source": stats.get("Initial_Inventory_Source", None),
            }
        )

    return pd.DataFrame(records)

def calculate_required_stock_by_sku_month(inventory_plan: pd.DataFrame) -> pd.DataFrame:
    """Calculate required stock per SKU per month across all divisions.

    Definition used for monthly SKU planning:
    - Monthly_Forecast_Demand = sum of weekly forecast demand in the month.
    - Required_Stock_Cases = monthly forecast demand + monthly safety stock buffer.
    - Monthly safety stock buffer is summed across SKU-Division because each
      division needs its own buffer under decentralized replenishment.
    - Monthly_Order_Qty is the planned replenishment quantity generated by the
      weekly inventory simulation.

    This output is useful for monthly supply/production planning at SKU level.
    """
    plan = inventory_plan.copy()
    plan["Week"] = pd.to_datetime(plan["Week"], errors="coerce")
    plan["Month"] = plan["Week"].dt.to_period("M").astype(str)

    # One safety-stock value per SKU-Division per month. Using max avoids
    # counting the same weekly buffer repeatedly inside the same month.
    div_month = (
        plan.groupby(["Code", "Product_Name", "Division", "Month"], dropna=False)
        .agg(
            Monthly_Forecast_Demand=("Forecast_Demand", "sum"),
            Monthly_Order_Qty=("Order_Qty", "sum"),
            Monthly_Arrivals=("Arrivals", "sum"),
            Monthly_Lost_Sales=("Lost_Sales", "sum"),
            Avg_Open_Stock_Qty=("Open_Stock_Qty", "mean"),
            Month_Ending_Inventory=("Ending_Inventory", "last"),
            Safety_Stock_LT=("Safety_Stock_LT", "max"),
            Safety_Stock_Protection=("Safety_Stock_Protection", "max"),
            Avg_Reorder_Point=("Reorder_Point", "mean"),
            Avg_Target_Inventory_Position=("Target_Inventory_Position", "mean"),
        )
        .reset_index()
    )

    sku_month = (
        div_month.groupby(["Code", "Product_Name", "Month"], dropna=False)
        .agg(
            Num_Divisions=("Division", "nunique"),
            Monthly_Forecast_Demand=("Monthly_Forecast_Demand", "sum"),
            Monthly_Safety_Stock_LT=("Safety_Stock_LT", "sum"),
            Monthly_Safety_Stock_Protection=("Safety_Stock_Protection", "sum"),
            Monthly_Order_Qty=("Monthly_Order_Qty", "sum"),
            Monthly_Arrivals=("Monthly_Arrivals", "sum"),
            Monthly_Lost_Sales=("Monthly_Lost_Sales", "sum"),
            Avg_Open_Stock_Qty=("Avg_Open_Stock_Qty", "sum"),
            Month_Ending_Inventory=("Month_Ending_Inventory", "sum"),
            Avg_Reorder_Point=("Avg_Reorder_Point", "sum"),
            Avg_Target_Inventory_Position=("Avg_Target_Inventory_Position", "sum"),
        )
        .reset_index()
    )

    sku_month["Required_Stock_Cases"] = (
        sku_month["Monthly_Forecast_Demand"]
        + sku_month["Monthly_Safety_Stock_Protection"]
    )
    sku_month["Required_Stock_LT_Cases"] = (
        sku_month["Monthly_Forecast_Demand"]
        + sku_month["Monthly_Safety_Stock_LT"]
    )
    sku_month["Net_Required_Order_Qty"] = (
        sku_month["Required_Stock_Cases"]
        - sku_month["Avg_Open_Stock_Qty"]
        - sku_month["Monthly_Arrivals"]
    ).clip(lower=0)

    ordered_cols = [
        "Code",
        "Product_Name",
        "Month",
        "Num_Divisions",
        "Monthly_Forecast_Demand",
        "Monthly_Safety_Stock_LT",
        "Monthly_Safety_Stock_Protection",
        "Required_Stock_LT_Cases",
        "Required_Stock_Cases",
        "Net_Required_Order_Qty",
        "Monthly_Order_Qty",
        "Monthly_Arrivals",
        "Avg_Open_Stock_Qty",
        "Month_Ending_Inventory",
        "Monthly_Lost_Sales",
        "Avg_Reorder_Point",
        "Avg_Target_Inventory_Position",
    ]
    return sku_month[ordered_cols].sort_values(["Code", "Month"]).reset_index(drop=True)


def run_inventory_optimization(dfr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read forecasts, simulate inventory, export inventory plan, KPIs, and monthly required stock."""
    future_path = OUTPUT_DIR / "future_forecast.xlsx"
    if not future_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {future_path}")

    logging.info("Loading future forecast: %s", future_path)
    future_df = pd.read_excel(future_path, engine="openpyxl")
    future_df = normalize_future_forecast(future_df)

    stats_df = historical_demand_stats(dfr)
    stats_lookup = {
        (row["Code"], row["Division"]): row.to_dict()
        for _, row in stats_df.iterrows()
    }

    plans = []
    for (code, division), pair_fc in future_df.groupby(["Code", "Division"]):
        stats = stats_lookup.get((code, division))
        if stats is None:
            logging.warning("No historical stats found for %s | %s. Skipping inventory simulation.", code, division)
            continue
        plans.append(simulate_inventory_for_pair(pair_fc, stats))

    if not plans:
        raise RuntimeError("No inventory plans were generated.")

    inventory_plan = pd.concat(plans, ignore_index=True)

    kpi_rows = []
    for (code, division), g in inventory_plan.groupby(["Code", "Division"]):
        total_demand = float(g["Forecast_Demand"].sum())
        total_lost = float(g["Lost_Sales"].sum())
        total_order = float(g["Order_Qty"].sum())
        avg_inventory = float(g["Ending_Inventory"].mean())
        avg_forecast = float(g["Forecast_Demand"].mean())
        service_level = 1.0 - (total_lost / total_demand if total_demand > 0 else 0.0)
        stockout_rate = float(g["Stockout_Flag"].mean())
        turnover = total_demand / avg_inventory if avg_inventory > 0 else np.nan
        weeks_cover = avg_inventory / avg_forecast if avg_forecast > 0 else np.nan

        kpi_rows.append(
            {
                "Code": code,
                "Division": division,
                "Product_Name": g["Product_Name"].dropna().iloc[0] if g["Product_Name"].notna().any() else None,
                "Selected_Model": g["Selected_Model"].dropna().iloc[0] if g["Selected_Model"].notna().any() else None,
                "Total_Forecast_Demand": total_demand,
                "Total_Order_Qty": total_order,
                "Avg_Ending_Inventory": avg_inventory,
                "Avg_Inventory_Cover_Weeks": weeks_cover,
                "Total_Lost_Sales": total_lost,
                "Stockout_Rate": stockout_rate,
                "Simulated_Service_Level": service_level,
                "Target_Service_Level": SERVICE_LEVEL,
                "Inventory_Turnover_Approx": turnover,
                "Safety_Stock_LT": float(g["Safety_Stock_LT"].iloc[0]),
                "Safety_Stock_Protection": float(g["Safety_Stock_Protection"].iloc[0]),
                "Reorder_Point_Avg": float(g["Reorder_Point"].mean()),
                "Initial_Inventory_Source": g["Initial_Inventory_Source"].iloc[0],
                "Lead_Time_Weeks": LEAD_TIME_WEEKS,
                "Review_Period_Weeks": REVIEW_PERIOD_WEEKS,
            }
        )

    inventory_kpi = pd.DataFrame(kpi_rows).sort_values(["Code", "Division"])

    required_stock_monthly = calculate_required_stock_by_sku_month(inventory_plan)

    plan_path = OUTPUT_DIR / "inventory_plan.xlsx"
    kpi_path = OUTPUT_DIR / "inventory_kpi.xlsx"
    monthly_required_path = OUTPUT_DIR / "required_stock_by_sku_month.xlsx"

    inventory_plan.to_excel(plan_path, index=False)
    inventory_kpi.to_excel(kpi_path, index=False)
    required_stock_monthly.to_excel(monthly_required_path, index=False)

    logging.info("Inventory plan saved: %s", plan_path)
    logging.info("Inventory KPI saved: %s", kpi_path)
    logging.info("Required stock by SKU-month saved: %s", monthly_required_path)

    create_inventory_charts(inventory_plan, inventory_kpi, OUTPUT_DIR / "inventory_charts.pdf")
    return inventory_plan, inventory_kpi


def create_inventory_charts(plan: pd.DataFrame, kpi: pd.DataFrame, pdf_path: Path) -> None:
    """Export inventory visualizations to PDF."""
    logging.info("Creating inventory charts: %s", pdf_path)
    with PdfPages(pdf_path) as pdf:
        # KPI: average inventory cover by pair
        top = kpi.copy()
        top["Pair"] = top["Code"].astype(str) + "-" + top["Division"].astype(str)
        top = top.sort_values("Avg_Inventory_Cover_Weeks", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(top["Pair"], top["Avg_Inventory_Cover_Weeks"])
        ax.axhline(INITIAL_STOCK_WEEKS_COVER, linestyle="--", linewidth=1, label="Current policy reference: 6 weeks")
        ax.set_title("Average Inventory Cover by SKU-Division")
        ax.set_ylabel("Weeks of Cover")
        ax.set_xlabel("SKU-Division")
        ax.tick_params(axis="x", rotation=75)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # KPI: service level by pair
        svc = kpi.copy()
        svc["Pair"] = svc["Code"].astype(str) + "-" + svc["Division"].astype(str)
        svc = svc.sort_values("Simulated_Service_Level").head(30)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(svc["Pair"], svc["Simulated_Service_Level"])
        ax.axhline(SERVICE_LEVEL, linestyle="--", linewidth=1, label="Target service level")
        ax.set_title("Simulated Service Level by SKU-Division")
        ax.set_ylabel("Service Level")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=75)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Detailed charts for each pair
        for (code, division), g in plan.groupby(["Code", "Division"]):
            g = g.sort_values("Week")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(g["Week"], g["Forecast_Demand"], label="Forecast Demand", linewidth=2)
            ax1.plot(g["Week"], g["Ending_Inventory"], label="Ending Inventory", linewidth=2)
            ax1.plot(g["Week"], g["Safety_Stock"], label="Safety Stock", linestyle="--")
            ax1.plot(g["Week"], g["Reorder_Point"], label="Reorder Point", linestyle=":")
            ax1.set_title(f"Inventory Simulation | SKU {code} | Division {division}")
            ax1.set_xlabel("Week")
            ax1.set_ylabel("Cases")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dfr = build_dfr_from_pre_pos(INPUT_FILE, PREPARED_FILE)
    run_forecast(PREPARED_FILE)
    run_inventory_optimization(dfr)

    logging.info("ALL DONE")
    logging.info("Main outputs:")
    logging.info("  %s", OUTPUT_DIR / "future_forecast.xlsx")
    logging.info("  %s", OUTPUT_DIR / "inventory_plan.xlsx")
    logging.info("  %s", OUTPUT_DIR / "inventory_kpi.xlsx")
    logging.info("  %s", OUTPUT_DIR / "required_stock_by_sku_month.xlsx")
    logging.info("  %s", OUTPUT_DIR / "inventory_charts.pdf")


if __name__ == "__main__":
    main()
