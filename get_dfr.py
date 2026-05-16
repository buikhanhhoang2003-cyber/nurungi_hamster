#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_pipeline_lightgbm_arima_hybrid import ForecastConfig, run_pipeline

BASE_DIR = Path("/home/hoangb/BOS/forecasting_pipeline_refactor_package")
INPUT_FILE = BASE_DIR / "pre POS Data.xlsx"

FULL_DFR_FILE = BASE_DIR / "dfr_full.xlsx"
PREPARED_FILE = BASE_DIR / "dfr_prt.xlsx"
PARETO_SUMMARY_FILE = BASE_DIR / "pareto_summary.xlsx"

OUTPUT_DIR = BASE_DIR / "forecast_output"
SHEET_NAME = "Weekly_POS"

DATE_COL = "Week"
CODE_COL = "Code"
PRODUCT_COL = "Product name"
DIVISION_COL = "Division"
TARGET_RAW_COL = "Cases Sold"
TARGET_FINAL_COL = "Cases Sold - Final"
STOCK_COL = "Total Division Stock (Cases)"
OPEN_STOCK_COL = "Open Stock Quantity"

PARETO_THRESHOLD = 0.8


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def interpolate_zeros_by_series(group: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(group[TARGET_RAW_COL], errors="coerce").replace(0, np.nan)

    if s.notna().sum() == 0:
        return pd.Series(0, index=group.index, dtype="float64")

    out = s.interpolate(method="linear", limit_direction="both")
    out = out.fillna(0).clip(lower=0).round(0)
    out.index = group.index
    return out


def pareto_by_sku(df_pareto: pd.DataFrame, sku_code, threshold: float = 0.8):
    temp = (
        df_pareto[df_pareto[CODE_COL] == sku_code]
        .groupby(DIVISION_COL, as_index=False)[TARGET_FINAL_COL]
        .sum()
    )

    temp = temp.sort_values(
        TARGET_FINAL_COL,
        ascending=False
    ).reset_index(drop=True)

    total_sales = temp[TARGET_FINAL_COL].sum()

    if total_sales == 0:
        temp["Sales_pct"] = 0
        temp["Cum_pct"] = 0
        return temp, []

    temp["Sales_pct"] = temp[TARGET_FINAL_COL] / total_sales
    temp["Cum_pct"] = temp["Sales_pct"].cumsum()

    # Include the division that crosses 80%
    key_divisions = temp.loc[
        temp["Cum_pct"].shift(1, fill_value=0) < threshold,
        DIVISION_COL
    ].tolist()

    return temp, key_divisions


def build_pareto_summary(df_pareto: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    records = []

    for sku_code in df_pareto[CODE_COL].unique():
        temp, top_divs = pareto_by_sku(
            df_pareto=df_pareto,
            sku_code=sku_code,
            threshold=threshold,
        )

        records.append({
            CODE_COL: sku_code,
            "Num_Divisions": len(temp),
            "Num_Divisions_80pct": len(top_divs),
            "Top_80pct_Divisions": ", ".join(map(str, top_divs)),
        })

    return (
        pd.DataFrame(records)
        .sort_values(CODE_COL)
        .reset_index(drop=True)
    )


def filter_by_pareto(dfr: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        summary_df[[CODE_COL, "Top_80pct_Divisions"]]
        .assign(
            **{
                DIVISION_COL: lambda x: x["Top_80pct_Divisions"].str.split(", ")
            }
        )
        .explode(DIVISION_COL)
        [[CODE_COL, DIVISION_COL]]
    )

    pairs = pairs.dropna()
    pairs = pairs[pairs[DIVISION_COL] != ""]

    return (
        dfr.merge(
            pairs,
            on=[CODE_COL, DIVISION_COL],
            how="inner",
        )
        .reset_index(drop=True)
    )


def build_dfr_from_pre_pos(input_file: Path, output_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info("Loading original POS data: %s", input_file)
    df = pd.read_excel(input_file, sheet_name=SHEET_NAME, engine="openpyxl")

    required_cols = {DATE_COL, CODE_COL, PRODUCT_COL, DIVISION_COL, TARGET_RAW_COL}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {input_file.name}: {sorted(missing)}"
        )

    keep_cols = [
        DATE_COL,
        CODE_COL,
        PRODUCT_COL,
        DIVISION_COL,
        TARGET_RAW_COL,
    ]

    if STOCK_COL in df.columns:
        keep_cols.append(STOCK_COL)

    dfr = df.loc[:, keep_cols].copy()

    dfr[DATE_COL] = pd.to_datetime(dfr[DATE_COL], errors="coerce")
    dfr[TARGET_RAW_COL] = pd.to_numeric(dfr[TARGET_RAW_COL], errors="coerce")

    if STOCK_COL in dfr.columns:
        dfr[STOCK_COL] = pd.to_numeric(dfr[STOCK_COL], errors="coerce")

    dfr = dfr.dropna(
        subset=[DATE_COL, CODE_COL, DIVISION_COL, TARGET_RAW_COL]
    )

    dfr = dfr[dfr[TARGET_RAW_COL] >= 0]

    dfr = dfr.sort_values(
        [CODE_COL, DIVISION_COL, DATE_COL]
    ).reset_index(drop=True)

    logging.info("Rows after cleaning: %s", f"{len(dfr):,}")
    logging.info(
        "Zero sales rows before interpolation: %s",
        f"{int((dfr[TARGET_RAW_COL] == 0).sum()):,}",
    )

    # -------------------------------------------------
    # 1. Interpolation
    # -------------------------------------------------
    dfr["Cases Sold - Interpolated"] = (
        dfr.groupby([CODE_COL, DIVISION_COL], group_keys=False)
        .apply(interpolate_zeros_by_series)
        .astype(float)
    )

    dfr[TARGET_FINAL_COL] = dfr["Cases Sold - Interpolated"]

    # -------------------------------------------------
    # 2. Open stock quantity
    # Week i open stock = week i-1 total division stock
    # -------------------------------------------------
    if STOCK_COL in dfr.columns:
        dfr[OPEN_STOCK_COL] = (
            dfr.groupby([CODE_COL, DIVISION_COL])[STOCK_COL]
            .shift(1)
            .fillna(0)
        )

        logging.info("Created column: %s", OPEN_STOCK_COL)
    else:
        logging.warning("Stock column not found, skipped open stock calculation.")

    # -------------------------------------------------
    # 3. Save full cleaned dfr before Pareto
    # -------------------------------------------------
    FULL_DFR_FILE.parent.mkdir(parents=True, exist_ok=True)
    dfr.to_excel(FULL_DFR_FILE, index=False)
    logging.info("Full cleaned dfr saved: %s", FULL_DFR_FILE)

    # -------------------------------------------------
    # 4. Pareto after interpolation
    # -------------------------------------------------
    logging.info("Running Pareto filtering after interpolation...")

    df_pareto = (
        dfr.groupby([CODE_COL, DIVISION_COL], as_index=False)
        .agg({TARGET_FINAL_COL: "sum"})
    )

    pareto_summary = build_pareto_summary(
        df_pareto=df_pareto,
        threshold=PARETO_THRESHOLD,
    )

    pareto_summary.to_excel(PARETO_SUMMARY_FILE, index=False)
    logging.info("Pareto summary saved: %s", PARETO_SUMMARY_FILE)

    dfr_prt = filter_by_pareto(
        dfr=dfr,
        summary_df=pareto_summary,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dfr_prt.to_excel(output_file, index=False)

    logging.info("Pareto filtered dfr saved: %s", output_file)
    logging.info("Original rows: %s", f"{len(dfr):,}")
    logging.info("Pareto rows: %s", f"{len(dfr_prt):,}")
    logging.info("SKUs retained: %s", dfr_prt[CODE_COL].nunique())
    logging.info(
        "SKU-Division pairs retained: %s",
        dfr_prt[[CODE_COL, DIVISION_COL]].drop_duplicates().shape[0],
    )

    return dfr_prt


def main() -> None:
    setup_logging()

    # This now creates dfr_prt.xlsx using Pareto output.
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

    logging.info("Starting forecast pipeline using Pareto output...")
    run_pipeline(
        input_path=PREPARED_FILE,
        config=config,
        sheet_name=None,
    )

    logging.info("Done. Forecast outputs saved in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
