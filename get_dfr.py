#!/usr/bin/env python3
"""
Run the full forecasting pipeline directly from the original POS Excel file.

No Jupyter needed.

This script:
1. Reads:  /home/hoangb/BOS/forecasting_pipeline_refactor_package/pre POS Data.xlsx
2. Loads sheet: Weekly_POS
3. Creates Cases Sold - Final using zero-to-NaN linear interpolation by Code-Division
4. Saves the prepared dfr file as: dfr_prt.xlsx
5. Runs forecasting_pipeline_lightgbm_arima_hybrid.py outputs to forecast_output/

Run:
    python run_from_pre_pos.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_pipeline_lightgbm_arima_hybrid import ForecastConfig, run_pipeline

# ---------------------------------------------------------------------
# Fixed paths based on your Ubuntu folder
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def interpolate_zeros_by_series(group: pd.DataFrame) -> pd.Series:
    """Replace 0 sales with NaN, then linearly interpolate inside each Code-Division series."""
    s = pd.to_numeric(group[TARGET_RAW_COL], errors="coerce").replace(0, np.nan)

    # If a whole group is zero/missing, keep zeros instead of crashing.
    if s.notna().sum() == 0:
        return pd.Series(0, index=group.index, dtype="float64")

    out = s.interpolate(method="linear", limit_direction="both")
    out = out.fillna(0).clip(lower=0).round(0)
    out.index = group.index
    return out


def build_dfr_from_pre_pos(input_file: Path, output_file: Path) -> pd.DataFrame:
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

    # The thesis selected linear interpolation as the final cleaned target.
    dfr[TARGET_FINAL_COL] = dfr["Cases Sold - Interpolated"]

    # Save prepared file for audit/reuse.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dfr.to_excel(output_file, index=False)
    logging.info("Prepared dfr saved: %s", output_file)

    return dfr


def main() -> None:
    setup_logging()

    build_dfr_from_pre_pos(INPUT_FILE, PREPARED_FILE)

    config = ForecastConfig(
        date_col=DATE_COL,
        code_col=CODE_COL,
        division_col=DIVISION_COL,
        target_col=TARGET_FINAL_COL,
        product_col=PRODUCT_COL,
        output_dir=str(OUTPUT_DIR),
        forecast_weeks=52,
        backtest_horizon=8,
        n_backtest_origins=5,
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
    run_pipeline(input_path=PREPARED_FILE, config=config, sheet_name=None)
    logging.info("Done. Forecast outputs saved in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
