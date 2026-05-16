"""
Forecasting + Inventory Optimization Framework

Input/Output flow:
1) Load raw data from: Pre POS data.xlsx  -> df
2) Process df and export: dfr.xlsx
3) Process dfr.xlsx and export: dfr_prt.xlsx
4) Use dfr_prt.xlsx to train/test forecast models, forecast to 2026-08-30,
   calculate weekly/monthly inventory/order quantity, and generate heatmap.

Expected minimum columns after normalization:
- date
- sku
- division
- demand
- current_inventory OR total_division_stock_cases

Optional columns:
- lead_time_weeks
- service_level_z

Install dependencies:
pip install pandas numpy openpyxl scikit-learn lightgbm matplotlib seaborn statsmodels

Run:
python forecasting_inventory_framework.py --input "Pre POS data.xlsx" --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.impute import KNNImputer
except ImportError as exc:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from exc

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Please install lightgbm: pip install lightgbm") from exc

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

@dataclass
class Config:
    input_file: str = "Pre POS data.xlsx"
    output_dir: str = "outputs"
    dfr_file: str = "dfr.xlsx"
    dfr_prt_file: str = "dfr_prt.xlsx"
    forecast_end_date: str = "2026-08-30"
    date_col: str = "date"
    sku_col: str = "sku"
    division_col: str = "division"
    target_col: str = "demand"
    current_inventory_col: str = "current_inventory"
    stock_col_fallback: str = "total_division_stock_cases"
    max_consecutive_missing: int = 10
    max_missing_ratio: float = 0.10
    forecast_freq: str = "W-SUN"
    selected_heatmap_skus: Optional[List[str]] = None
    default_lead_time_weeks: int = 2
    default_service_level_z: float = 1.65
    safety_stock_window_weeks: int = 8
    lgbm_lags: Tuple[int, ...] = (1, 2, 4, 8, 12)
    lgbm_rolling_windows: Tuple[int, ...] = (4, 8, 12)


# =========================
# Utility Functions
# =========================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names into the expected schema."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    rename_map = {
        "week": "date",
        "week_date": "date",
        "date_week": "date",
        "ds": "date",
        "sku_code": "sku",
        "item": "sku",
        "product": "sku",
        "division_name": "division",
        "region": "division",
        "sales": "demand",
        "qty": "demand",
        "quantity": "demand",
        "actual": "demand",
        "y": "demand",
        "total_division_stock_(cases)": "total_division_stock_cases",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def validate_required_columns(df: pd.DataFrame, cfg: Config) -> None:
    required = [cfg.date_col, cfg.sku_col, cfg.division_col, cfg.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Current columns: {list(df.columns)}"
        )


def get_inventory_col(df: pd.DataFrame, cfg: Config) -> str:
    if cfg.current_inventory_col in df.columns:
        return cfg.current_inventory_col
    if cfg.stock_col_fallback in df.columns:
        return cfg.stock_col_fallback
    df[cfg.current_inventory_col] = 0.0
    return cfg.current_inventory_col


def max_consecutive_na(s: pd.Series) -> int:
    is_na = s.isna().astype(int)
    groups = (is_na != is_na.shift()).cumsum()
    return int(is_na.groupby(groups).sum().max()) if len(s) else 0


def should_skip_imputation(s: pd.Series, cfg: Config) -> bool:
    missing_ratio = float(s.isna().mean())
    max_consec = max_consecutive_na(s)
    return max_consec > cfg.max_consecutive_missing or missing_ratio > cfg.max_missing_ratio


# =========================
# Step 1 -> Step 2: dfr.xlsx
# =========================

def load_raw_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.input_file)
    df = normalize_columns(df)
    validate_required_columns(df, cfg)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.sku_col, cfg.division_col])
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    return df


def process_to_dfr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Basic cleaning and weekly aggregation by SKU-Division-Date.
    """
    df = df.copy()
    inv_col = get_inventory_col(df, cfg)
    df[inv_col] = pd.to_numeric(df[inv_col], errors="coerce")

    grouped = (
        df.groupby([cfg.sku_col, cfg.division_col, pd.Grouper(key=cfg.date_col, freq=cfg.forecast_freq)], as_index=False)
        .agg(
            demand=(cfg.target_col, "sum"),
            current_inventory=(inv_col, "last"),
        )
    )
    grouped[cfg.target_col] = grouped[cfg.target_col].clip(lower=0)
    return grouped.sort_values([cfg.sku_col, cfg.division_col, cfg.date_col])


# =========================
# Step 2 -> Step 3: dfr_prt.xlsx
# Imputation: KNN + Linear Interpolation
# =========================

def complete_weekly_index(dfr: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for (sku, div), g in dfr.groupby([cfg.sku_col, cfg.division_col]):
        g = g.sort_values(cfg.date_col)
        full_dates = pd.date_range(g[cfg.date_col].min(), g[cfg.date_col].max(), freq=cfg.forecast_freq)
        base = pd.DataFrame({cfg.date_col: full_dates})
        base[cfg.sku_col] = sku
        base[cfg.division_col] = div
        merged = base.merge(g, on=[cfg.date_col, cfg.sku_col, cfg.division_col], how="left")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def add_imputed_columns(dfr: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    For each SKU-Division series:
    - If max consecutive missing > 10 OR total missing ratio > 10%, skip imputation.
    - Otherwise generate:
      demand_imputed_linear
      demand_imputed_knn
      demand_final = average of both methods where possible, else original demand.
    """
    dfr = complete_weekly_index(dfr, cfg)
    result = []

    for (sku, div), g in dfr.groupby([cfg.sku_col, cfg.division_col]):
        g = g.sort_values(cfg.date_col).copy()
        s = g[cfg.target_col]
        skip = should_skip_imputation(s, cfg)
        g["skip_imputation"] = skip
        g["missing_ratio"] = s.isna().mean()
        g["max_consecutive_missing"] = max_consecutive_na(s)

        if skip:
            g["demand_imputed_linear"] = s
            g["demand_imputed_knn"] = s
            g["demand_final"] = s
        else:
            # Method A: Linear interpolation
            g["demand_imputed_linear"] = (
                s.interpolate(method="linear", limit_direction="both").clip(lower=0)
            )

            # Method B: KNN imputation using time index, month, week number, inventory, and demand
            tmp = g.copy()
            tmp["time_idx"] = np.arange(len(tmp))
            tmp["month"] = tmp[cfg.date_col].dt.month
            tmp["weekofyear"] = tmp[cfg.date_col].dt.isocalendar().week.astype(int)
            feature_cols = ["time_idx", "month", "weekofyear", "current_inventory", cfg.target_col]
            for c in feature_cols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(tmp) - 1)))
            imputed = imputer.fit_transform(tmp[feature_cols])
            demand_knn = imputed[:, feature_cols.index(cfg.target_col)]
            g["demand_imputed_knn"] = np.clip(demand_knn, 0, None)

            # Final demand: average KNN and linear when original is missing; original otherwise
            avg_imputed = (g["demand_imputed_linear"] + g["demand_imputed_knn"]) / 2
            g["demand_final"] = np.where(s.isna(), avg_imputed, s)
            g["demand_final"] = pd.Series(g["demand_final"]).clip(lower=0)

        result.append(g)

    return pd.concat(result, ignore_index=True)


# =========================
# Forecasting Features + Models
# =========================

def create_ts_features(df: pd.DataFrame, cfg: Config, target_col: str = "demand_final") -> pd.DataFrame:
    df = df.sort_values([cfg.sku_col, cfg.division_col, cfg.date_col]).copy()
    df["weekofyear"] = df[cfg.date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[cfg.date_col].dt.month
    df["quarter"] = df[cfg.date_col].dt.quarter
    df["year"] = df[cfg.date_col].dt.year

    group_cols = [cfg.sku_col, cfg.division_col]
    for lag in cfg.lgbm_lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    for win in cfg.lgbm_rolling_windows:
        df[f"roll_mean_{win}"] = (
            df.groupby(group_cols)[target_col]
            .shift(1)
            .rolling(win)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        df[f"roll_std_{win}"] = (
            df.groupby(group_cols)[target_col]
            .shift(1)
            .rolling(win)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )
    return df


def fit_lightgbm(train_df: pd.DataFrame, cfg: Config) -> Tuple[lgb.LGBMRegressor, List[str]]:
    feature_cols = [
        "weekofyear", "month", "quarter", "year",
        *[f"lag_{lag}" for lag in cfg.lgbm_lags],
        *[f"roll_mean_{w}" for w in cfg.lgbm_rolling_windows],
        *[f"roll_std_{w}" for w in cfg.lgbm_rolling_windows],
    ]
    model_df = train_df.dropna(subset=feature_cols + ["demand_final"]).copy()
    if model_df.empty:
        raise ValueError("Not enough data to train LightGBM after lag/rolling feature creation.")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(model_df[feature_cols], model_df["demand_final"])
    return model, feature_cols


def recursive_lgbm_forecast(history: pd.DataFrame, model: lgb.LGBMRegressor, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    history = history.sort_values([cfg.sku_col, cfg.division_col, cfg.date_col]).copy()
    last_date = history[cfg.date_col].max()
    end_date = pd.to_datetime(cfg.forecast_end_date)
    future_dates = pd.date_range(last_date + pd.offsets.Week(weekday=6), end_date, freq=cfg.forecast_freq)

    all_rows = history.copy()
    forecasts = []

    for dt in future_dates:
        new_rows = []
        for (sku, div), g in all_rows.groupby([cfg.sku_col, cfg.division_col]):
            g = g.sort_values(cfg.date_col)
            row = {
                cfg.date_col: dt,
                cfg.sku_col: sku,
                cfg.division_col: div,
                "current_inventory": g["current_inventory"].dropna().iloc[-1] if g["current_inventory"].notna().any() else 0,
                "demand_final": np.nan,
            }
            new_rows.append(row)

        step = pd.DataFrame(new_rows)
        combined = pd.concat([all_rows, step], ignore_index=True)
        combined = create_ts_features(combined, cfg, target_col="demand_final")
        pred_rows = combined[combined[cfg.date_col] == dt].copy()
        pred_rows[feature_cols] = pred_rows[feature_cols].fillna(0)
        pred_rows["forecast_demand"] = np.clip(model.predict(pred_rows[feature_cols]), 0, None)
        pred_rows["demand_final"] = pred_rows["forecast_demand"]
        forecasts.append(pred_rows[[cfg.date_col, cfg.sku_col, cfg.division_col, "forecast_demand", "current_inventory"]])

        all_rows = pd.concat([
            all_rows,
            pred_rows[[cfg.date_col, cfg.sku_col, cfg.division_col, "current_inventory", "demand_final"]]
        ], ignore_index=True)

    return pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()


# =========================
# Inventory Optimization
# =========================

def calculate_inventory_outputs(history: pd.DataFrame, forecast: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Target Inventory logic:
    - Forecast demand per SKU-Division-Week
    - Safety Stock = z * std(recent weekly demand) * sqrt(lead_time_weeks)
    - Reorder Point = lead_time_weeks * avg weekly forecast + safety_stock
    - Target Inventory = demand during lead time + safety stock
    - Order Quantity = max(Target Inventory - Current Inventory, 0)
    """
    hist_stats = []
    for (sku, div), g in history.groupby([cfg.sku_col, cfg.division_col]):
        recent = g.sort_values(cfg.date_col).tail(cfg.safety_stock_window_weeks)["demand_final"]
        hist_stats.append({
            cfg.sku_col: sku,
            cfg.division_col: div,
            "recent_demand_std": recent.std(ddof=0) if len(recent) else 0,
            "recent_demand_avg": recent.mean() if len(recent) else 0,
        })
    stats = pd.DataFrame(hist_stats)

    out = forecast.merge(stats, on=[cfg.sku_col, cfg.division_col], how="left")
    out["lead_time_weeks"] = cfg.default_lead_time_weeks
    out["service_level_z"] = cfg.default_service_level_z
    out["safety_stock"] = (
        out["service_level_z"] * out["recent_demand_std"].fillna(0) * np.sqrt(out["lead_time_weeks"])
    )
    out["reorder_point"] = out["forecast_demand"] * out["lead_time_weeks"] + out["safety_stock"]
    out["target_inventory"] = out["reorder_point"]
    out["order_quantity"] = (out["target_inventory"] - out["current_inventory"].fillna(0)).clip(lower=0)

    weekly_target_by_sku_div = out[[
        cfg.date_col, cfg.sku_col, cfg.division_col,
        "forecast_demand", "current_inventory", "safety_stock",
        "reorder_point", "target_inventory", "order_quantity"
    ]].copy()

    weekly_order_by_sku = (
        out.groupby([cfg.date_col, cfg.sku_col], as_index=False)["order_quantity"]
        .sum()
        .sort_values([cfg.sku_col, cfg.date_col])
    )

    monthly_order_by_sku = weekly_order_by_sku.copy()
    monthly_order_by_sku["month"] = monthly_order_by_sku[cfg.date_col].dt.to_period("M").astype(str)
    monthly_order_by_sku = (
        monthly_order_by_sku.groupby(["month", cfg.sku_col], as_index=False)["order_quantity"]
        .sum()
        .sort_values([cfg.sku_col, "month"])
    )

    return weekly_target_by_sku_div, weekly_order_by_sku, monthly_order_by_sku


def plot_weekly_heatmap(weekly_order_by_sku: pd.DataFrame, cfg: Config, output_dir: Path) -> Path:
    """Plot weekly heatmap for exactly 3 SKUs if provided, otherwise first 3 SKUs."""
    skus = cfg.selected_heatmap_skus or weekly_order_by_sku[cfg.sku_col].dropna().astype(str).unique().tolist()[:3]
    data = weekly_order_by_sku[weekly_order_by_sku[cfg.sku_col].astype(str).isin([str(x) for x in skus])].copy()
    data["week"] = data[cfg.date_col].dt.strftime("%Y-%m-%d")
    pivot = data.pivot_table(index=cfg.sku_col, columns="week", values="order_quantity", aggfunc="sum", fill_value=0)

    fig_width = max(12, len(pivot.columns) * 0.45)
    plt.figure(figsize=(fig_width, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Order Quantity")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Weekly Order Quantity Heatmap")
    plt.xlabel("Week")
    plt.ylabel("SKU")
    plt.tight_layout()

    path = output_dir / "weekly_order_quantity_heatmap_3_skus.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# =========================
# End-to-End Runner
# =========================

def run_pipeline(cfg: Config) -> Dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(cfg)

    dfr = process_to_dfr(df, cfg)
    dfr_path = output_dir / cfg.dfr_file
    dfr.to_excel(dfr_path, index=False)

    dfr_loaded = pd.read_excel(dfr_path)
    dfr_loaded[cfg.date_col] = pd.to_datetime(dfr_loaded[cfg.date_col])
    dfr_prt = add_imputed_columns(dfr_loaded, cfg)
    dfr_prt_path = output_dir / cfg.dfr_prt_file
    dfr_prt.to_excel(dfr_prt_path, index=False)

    train_df = create_ts_features(dfr_prt, cfg, target_col="demand_final")
    model, feature_cols = fit_lightgbm(train_df, cfg)
    forecast = recursive_lgbm_forecast(dfr_prt, model, feature_cols, cfg)
    forecast_path = output_dir / "forecast_weekly_sku_division.xlsx"
    forecast.to_excel(forecast_path, index=False)

    weekly_target, weekly_order, monthly_order = calculate_inventory_outputs(dfr_prt, forecast, cfg)

    weekly_target_path = output_dir / "weekly_target_inventory_by_sku_division.xlsx"
    weekly_order_path = output_dir / "weekly_order_quantity_by_sku.xlsx"
    monthly_order_path = output_dir / "monthly_order_quantity_by_sku.xlsx"

    weekly_target.to_excel(weekly_target_path, index=False)
    weekly_order.to_excel(weekly_order_path, index=False)
    monthly_order.to_excel(monthly_order_path, index=False)

    heatmap_path = plot_weekly_heatmap(weekly_order, cfg, output_dir)

    return {
        "dfr": dfr_path,
        "dfr_prt": dfr_prt_path,
        "forecast": forecast_path,
        "weekly_target_inventory": weekly_target_path,
        "weekly_order_quantity": weekly_order_path,
        "monthly_order_quantity": monthly_order_path,
        "heatmap": heatmap_path,
    }


# =========================
# Pseudo-code Notes
# =========================

PSEUDO_CODE = r'''
1. ARIMA baseline
   FOR each SKU-Division time series:
       Sort data by week
       Use demand_final as target
       Split historical data into train/test
       Check stationarity if needed
       Select ARIMA(p,d,q), e.g. by AIC or fixed baseline order
       Fit ARIMA on train demand
       Forecast test horizon and future horizon to 2026-08-30
       Clip negative forecasts to zero
       Save forecast per SKU-Division-Week

2. LightGBM model
   Load dfr_prt.xlsx
   Create calendar features: weekofyear, month, quarter, year
   Create lag features: lag_1, lag_2, lag_4, lag_8, lag_12
   Create rolling features: rolling mean/std for 4, 8, 12 weeks
   Drop rows where features or target are missing
   Train LightGBMRegressor with time-series-safe split
   Forecast recursively week by week until 2026-08-30:
       For next week, build features from history + prior predictions
       Predict demand
       Append prediction back to history
   Output forecast_weekly_sku_division.xlsx

3. Hybrid model
   FOR each SKU-Division:
       Fit ARIMA on demand_final
       Generate ARIMA fitted values and residuals
       Train LightGBM to predict residuals using calendar/lag/rolling features
       Future forecast = ARIMA forecast + LightGBM residual forecast
       Clip negative forecasts to zero
   Hybrid output combines statistical trend/seasonality with ML correction.

4. Stock Quantity Optimization
   FOR each SKU-Division-Week forecast:
       avg_forecast = forecast demand for target week
       recent_std = std of recent historical demand
       safety_stock = service_level_z * recent_std * sqrt(lead_time_weeks)
       reorder_point = avg_forecast * lead_time_weeks + safety_stock
       target_inventory = reorder_point
       order_quantity = max(target_inventory - current_inventory, 0)
   Export:
       Weekly target inventory by SKU-Division
       Weekly order quantity aggregated by SKU
       Monthly order quantity aggregated by SKU
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Pre POS data.xlsx")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--forecast-end-date", default="2026-08-30")
    parser.add_argument("--heatmap-skus", nargs="*", default=None, help="Provide exactly 3 SKU codes if desired")
    args = parser.parse_args()

    cfg = Config(
        input_file=args.input,
        output_dir=args.output_dir,
        forecast_end_date=args.forecast_end_date,
        selected_heatmap_skus=args.heatmap_skus,
    )
    outputs = run_pipeline(cfg)

    print("Pipeline completed. Outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("\nPseudo-code summary:")
    print(PSEUDO_CODE)


if __name__ == "__main__":
    main()
