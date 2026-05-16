"""
Production-quality weekly demand forecasting pipeline.

Models:
    1) Auto-ARIMA / SARIMAX selected by AIC
    2) LightGBMRegressor
    3) Optimized hybrid forecast: w_arima * ARIMA + w_lgbm * LightGBM

Main outputs:
    - summary.xlsx
    - backtest_detail.xlsx
    - future_forecast.xlsx
    - model_metrics.xlsx
    - selected_model_summary.xlsx
    - forecast_charts.pdf

Designed for SKU-Division weekly retail demand forecasting with walk-forward
validation and no future leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Matplotlib is imported lazily in chart generation to keep non-chart runs lighter.

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - joblib is optional but recommended
    Parallel = None
    delayed = None

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "LightGBM is required. Install it with: pip install lightgbm"
    ) from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


LOGGER = logging.getLogger("forecasting_pipeline")

MODEL_NAMES = ("ARIMA", "LightGBM", "Hybrid")

COLOR_MAP = {
    "History": "#BDBDBD",
    "Actual": "#111111",
    "ARIMA": "#1F77B4",
    "LightGBM": "#FF7F0E",
    "Hybrid": "#2CA02C",
    "Selected": "#D62728",
}


@dataclass
class ForecastConfig:
    """Configuration for the complete forecasting pipeline."""

    # Input columns
    date_col: str = "Week"
    code_col: str = "Code"
    division_col: str = "Division"
    target_col: str = "Cases Sold - Final"
    product_col: Optional[str] = "Product name"

    # Time-series settings
    weekly_frequency: str = "W-MON"
    missing_week_strategy: str = "interpolate"  # interpolate, ffill, zero
    zero_as_missing: bool = False
    clamp_negative_forecasts: bool = True

    # Backtesting settings
    min_train_weeks: int = 24
    min_total_weeks: int = 30
    backtest_horizon: int = 8
    n_backtest_origins: int = 5

    # Future forecasting settings
    forecast_weeks: int = 52
    forecast_end_date: Optional[str] = None

    # ARIMA settings
    use_pmdarima: bool = True
    seasonal_arima: bool = False
    seasonal_period: int = 52
    min_seasonal_weeks: int = 110
    max_p: int = 2
    max_d: int = 2
    max_q: int = 2
    max_P: int = 1
    max_D: int = 1
    max_Q: int = 1

    # LightGBM feature settings
    lags: Tuple[int, ...] = (1, 2, 4, 8, 12, 24, 52)
    rolling_windows: Tuple[int, ...] = (4, 8, 12, 24)
    use_log_target: bool = True
    min_lgbm_train_rows: int = 12

    # LightGBM model settings: reasonable defaults for weekly demand series
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 10,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": -1,
        }
    )

    # Parallelization
    n_jobs: int = -1

    # Output
    output_dir: str = "forecast_output"
    chart_last_history_weeks: int = 60


@dataclass
class ARIMAFitResult:
    model: Any
    order: Optional[Tuple[int, int, int]]
    seasonal_order: Optional[Tuple[int, int, int, int]]
    aic: Optional[float]
    method: str
    error: Optional[str] = None


@dataclass
class LGBMFitResult:
    model: Optional[LGBMRegressor]
    feature_columns: List[str]
    active_lags: List[int]
    active_windows: List[int]
    fallback_value: float
    error: Optional[str] = None


@dataclass
class PairResult:
    code: Any
    division: Any
    product_name: Optional[str]
    n_weeks: int
    first_week: pd.Timestamp
    last_week: pd.Timestamp
    selected_model: str
    arima_order: Optional[Tuple[int, int, int]]
    arima_seasonal_order: Optional[Tuple[int, int, int, int]]
    arima_aic: Optional[float]
    arima_method: str
    lightgbm_params: Dict[str, Any]
    hybrid_w_arima: float
    hybrid_w_lightgbm: float
    metrics: Dict[str, Dict[str, float]]
    backtest_detail: pd.DataFrame
    future_forecast: pd.DataFrame
    series: pd.Series
    warnings: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    if math.isfinite(value):
        return value
    return np.nan


def safe_round(value: Any, digits: int = 4) -> Optional[float]:
    value = _safe_float(value)
    if np.isnan(value):
        return None
    return round(value, digits)


def clamp_forecast(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def wape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    denom = np.sum(y_true)
    bias_pct = np.nan if denom == 0 else float(np.sum(err) / denom * 100.0)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "Bias": float(np.mean(err)),
        "Bias_%": bias_pct,
    }


def adf_pvalue(values: Sequence[float]) -> Optional[float]:
    arr = pd.Series(values).dropna().astype(float)
    if len(arr) < 8 or arr.nunique() <= 1:
        return None
    try:
        return float(adfuller(arr, autolag="AIC")[1])
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------


def load_and_preprocess_data(
    input_path: str | Path,
    config: ForecastConfig,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load Excel/CSV data, normalize columns, and aggregate to weekly level."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Loading input data: %s", input_path)
    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=0 if sheet_name is None else sheet_name)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be an Excel or CSV file.")

    required = {config.date_col, config.code_col, config.division_col, config.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    keep_cols = [config.date_col, config.code_col, config.division_col, config.target_col]
    if config.product_col and config.product_col in df.columns:
        keep_cols.append(config.product_col)

    df = df.loc[:, keep_cols].copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df[config.target_col] = pd.to_numeric(df[config.target_col], errors="coerce")
    df = df.dropna(subset=[config.date_col, config.code_col, config.division_col, config.target_col])

    # Align dates to Monday week starts for consistent W-MON indexing.
    df[config.date_col] = df[config.date_col] - pd.to_timedelta(df[config.date_col].dt.weekday, unit="d")
    df[config.target_col] = df[config.target_col].clip(lower=0)

    group_cols = [config.code_col, config.division_col, config.date_col]
    agg_dict: Dict[str, Any] = {config.target_col: "sum"}
    if config.product_col and config.product_col in df.columns:
        agg_dict[config.product_col] = "first"

    df_weekly = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_dict)
        .sort_values([config.code_col, config.division_col, config.date_col])
        .reset_index(drop=True)
    )

    LOGGER.info(
        "Loaded %s rows after weekly aggregation | %s SKUs | %s divisions",
        f"{len(df_weekly):,}",
        df_weekly[config.code_col].nunique(),
        df_weekly[config.division_col].nunique(),
    )
    return df_weekly


def build_weekly_series(group: pd.DataFrame, config: ForecastConfig) -> pd.Series:
    """Build a complete weekly series for one SKU-Division pair."""

    s = (
        group.groupby(config.date_col)[config.target_col]
        .sum()
        .sort_index()
        .astype(float)
    )

    if s.empty:
        return s

    full_index = pd.date_range(start=s.index.min(), end=s.index.max(), freq=config.weekly_frequency)
    s = s.reindex(full_index)

    if config.zero_as_missing:
        s = s.replace(0.0, np.nan)

    if config.missing_week_strategy == "interpolate":
        s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
    elif config.missing_week_strategy == "ffill":
        s = s.ffill().bfill()
    elif config.missing_week_strategy == "zero":
        s = s.fillna(0.0)
    else:
        raise ValueError(
            "missing_week_strategy must be one of: interpolate, ffill, zero"
        )

    return s.fillna(0.0).clip(lower=0.0)


def get_product_name(group: pd.DataFrame, config: ForecastConfig) -> Optional[str]:
    if config.product_col and config.product_col in group.columns:
        values = group[config.product_col].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return None


# -----------------------------------------------------------------------------
# LightGBM feature engineering
# -----------------------------------------------------------------------------


def active_lags_and_windows(n_obs: int, config: ForecastConfig) -> Tuple[List[int], List[int]]:
    """Select only lag/rolling features that leave enough training rows."""

    max_allowed_lag = max(1, n_obs - config.min_lgbm_train_rows)
    active_lags = [lag for lag in config.lags if lag <= max_allowed_lag]
    active_windows = [w for w in config.rolling_windows if w <= max_allowed_lag]

    # Always keep short-term lags if possible.
    if not active_lags and n_obs >= 3:
        active_lags = [1]
    if not active_windows and n_obs >= 5:
        active_windows = [4]

    return active_lags, active_windows


def make_lgbm_training_frame(
    ts: pd.Series,
    config: ForecastConfig,
    active_lags: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str], List[int], List[int]]:
    """Create leakage-safe supervised features using shifted lags and rolling stats."""

    ts = ts.astype(float).sort_index()
    if active_lags is None or active_windows is None:
        active_lags, active_windows = active_lags_and_windows(len(ts), config)

    dm = pd.DataFrame({"y_original": ts.values}, index=ts.index)

    for lag in active_lags:
        dm[f"lag_{lag}"] = dm["y_original"].shift(lag)

    shifted = dm["y_original"].shift(1)
    for window in active_windows:
        roll = shifted.rolling(window=window, min_periods=window)
        dm[f"roll_{window}_mean"] = roll.mean()
        dm[f"roll_{window}_std"] = roll.std().fillna(0.0)
        dm[f"roll_{window}_min"] = roll.min()
        dm[f"roll_{window}_max"] = roll.max()

    dm["diff_1"] = dm["y_original"].shift(1) - dm["y_original"].shift(2)
    if 4 in active_lags or len(ts) >= 5:
        dm["momentum_4"] = dm["y_original"].shift(1) - dm["y_original"].shift(4)
    else:
        dm["momentum_4"] = 0.0

    iso_week = dm.index.isocalendar().week.astype(int)
    dm["month"] = dm.index.month.astype(int)
    dm["weekofyear"] = iso_week
    dm["quarter"] = dm.index.quarter.astype(int)
    dm["time_idx"] = np.arange(len(dm), dtype=float)

    dm["month_sin"] = np.sin(2.0 * np.pi * dm["month"] / 12.0)
    dm["month_cos"] = np.cos(2.0 * np.pi * dm["month"] / 12.0)
    dm["week_sin"] = np.sin(2.0 * np.pi * dm["weekofyear"] / 52.0)
    dm["week_cos"] = np.cos(2.0 * np.pi * dm["weekofyear"] / 52.0)

    target_col = "y_log" if config.use_log_target else "y_original"
    if config.use_log_target:
        dm[target_col] = np.log1p(dm["y_original"].clip(lower=0.0))

    feature_columns = [c for c in dm.columns if c not in {"y_original", "y_log"}]
    dm = dm.dropna(subset=feature_columns + [target_col]).copy()
    dm[feature_columns] = dm[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return dm, feature_columns, active_lags, active_windows


def make_future_feature_row(
    history: List[float],
    future_date: pd.Timestamp,
    time_idx: int,
    feature_columns: List[str],
    active_lags: List[int],
    active_windows: List[int],
) -> pd.DataFrame:
    """Build one recursive LightGBM feature row from known/predicted history only."""

    hist = np.asarray(history, dtype=float)
    last_value = float(hist[-1]) if len(hist) else 0.0
    row: Dict[str, float] = {}

    for lag in active_lags:
        row[f"lag_{lag}"] = float(hist[-lag]) if len(hist) >= lag else last_value

    for window in active_windows:
        tail = hist[-window:] if len(hist) >= window else hist
        if len(tail) == 0:
            tail = np.asarray([0.0])
        row[f"roll_{window}_mean"] = float(np.mean(tail))
        row[f"roll_{window}_std"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
        row[f"roll_{window}_min"] = float(np.min(tail))
        row[f"roll_{window}_max"] = float(np.max(tail))

    row["diff_1"] = float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
    row["momentum_4"] = float(hist[-1] - hist[-4]) if len(hist) >= 4 else 0.0

    week = int(future_date.isocalendar().week)
    month = int(future_date.month)
    row["month"] = month
    row["weekofyear"] = week
    row["quarter"] = int(future_date.quarter)
    row["time_idx"] = float(time_idx)
    row["month_sin"] = float(np.sin(2.0 * np.pi * month / 12.0))
    row["month_cos"] = float(np.cos(2.0 * np.pi * month / 12.0))
    row["week_sin"] = float(np.sin(2.0 * np.pi * week / 52.0))
    row["week_cos"] = float(np.cos(2.0 * np.pi * week / 52.0))

    # Reindex ensures stable training/prediction column order.
    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0.0)


def fit_lgbm_model(ts: pd.Series, config: ForecastConfig) -> LGBMFitResult:
    """Fit a LightGBM model for one weekly series."""

    fallback = float(ts.iloc[-1]) if len(ts) else 0.0
    try:
        dm, feature_columns, active_lags, active_windows = make_lgbm_training_frame(ts, config)
        target_col = "y_log" if config.use_log_target else "y_original"

        if len(dm) < config.min_lgbm_train_rows:
            return LGBMFitResult(
                model=None,
                feature_columns=feature_columns,
                active_lags=active_lags,
                active_windows=active_windows,
                fallback_value=fallback,
                error=f"Not enough LightGBM rows after feature engineering: {len(dm)}",
            )

        model = LGBMRegressor(**config.lgbm_params)
        model.fit(dm[feature_columns], dm[target_col])
        return LGBMFitResult(
            model=model,
            feature_columns=feature_columns,
            active_lags=active_lags,
            active_windows=active_windows,
            fallback_value=fallback,
        )
    except Exception as exc:
        return LGBMFitResult(
            model=None,
            feature_columns=[],
            active_lags=[],
            active_windows=[],
            fallback_value=fallback,
            error=str(exc),
        )


def forecast_lgbm_recursive(
    ts: pd.Series,
    future_index: pd.DatetimeIndex,
    config: ForecastConfig,
    fitted: Optional[LGBMFitResult] = None,
) -> np.ndarray:
    """Forecast LightGBM recursively over the requested future dates."""

    if fitted is None:
        fitted = fit_lgbm_model(ts, config)

    if fitted.model is None or not fitted.feature_columns:
        return np.repeat(max(fitted.fallback_value, 0.0), len(future_index))

    history = list(ts.astype(float).values)
    forecasts: List[float] = []
    time_idx = len(history)

    for future_date in future_index:
        x_future = make_future_feature_row(
            history=history,
            future_date=pd.Timestamp(future_date),
            time_idx=time_idx,
            feature_columns=fitted.feature_columns,
            active_lags=fitted.active_lags,
            active_windows=fitted.active_windows,
        )
        pred = float(fitted.model.predict(x_future)[0])
        if config.use_log_target:
            pred = float(np.expm1(pred))
        pred = max(pred, 0.0)
        forecasts.append(pred)
        history.append(pred)
        time_idx += 1

    return clamp_forecast(forecasts)


# -----------------------------------------------------------------------------
# ARIMA selection and forecasting
# -----------------------------------------------------------------------------


def _seasonal_allowed(ts: pd.Series, config: ForecastConfig) -> bool:
    return bool(
        config.seasonal_arima
        and config.seasonal_period > 1
        and len(ts) >= max(config.min_seasonal_weeks, 2 * config.seasonal_period)
    )


def fit_arima_pmdarima(ts: pd.Series, config: ForecastConfig) -> ARIMAFitResult:
    """Fit pmdarima.auto_arima using AIC; falls through on failure."""

    try:
        from pmdarima import auto_arima
    except Exception as exc:
        raise RuntimeError(f"pmdarima unavailable: {exc}") from exc

    seasonal = _seasonal_allowed(ts, config)
    m = config.seasonal_period if seasonal else 1

    model = auto_arima(
        ts.astype(float).values,
        start_p=0,
        start_q=0,
        max_p=config.max_p,
        max_d=config.max_d,
        max_q=config.max_q,
        seasonal=seasonal,
        m=m,
        start_P=0,
        start_Q=0,
        max_P=config.max_P if seasonal else 0,
        max_D=config.max_D if seasonal else 0,
        max_Q=config.max_Q if seasonal else 0,
        d=None,
        D=None,
        test="adf",
        seasonal_test="ocsb",
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        with_intercept="auto",
        random_state=config.lgbm_params.get("random_state", 42),
        max_order=5,
        maxiter=50,
        n_jobs=1,
    )

    seasonal_order = tuple(model.seasonal_order) if seasonal else (0, 0, 0, 0)
    return ARIMAFitResult(
        model=model,
        order=tuple(model.order),
        seasonal_order=seasonal_order,
        aic=float(model.aic()),
        method="pmdarima.auto_arima",
    )


def fit_arima_custom_aic(ts: pd.Series, config: ForecastConfig) -> ARIMAFitResult:
    """Custom SARIMAX AIC grid search fallback."""

    values = ts.astype(float)
    seasonal = _seasonal_allowed(ts, config)

    best_result: Optional[Any] = None
    best_order: Optional[Tuple[int, int, int]] = None
    best_seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    best_aic = np.inf
    last_error: Optional[str] = None

    seasonal_candidates: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0)]
    if seasonal:
        seasonal_candidates = []
        for P in range(config.max_P + 1):
            for D in range(config.max_D + 1):
                for Q in range(config.max_Q + 1):
                    seasonal_candidates.append((P, D, Q, config.seasonal_period))

    # Try AIC over p, d, q. d is still bounded to protect runtime.
    for p in range(config.max_p + 1):
        for d in range(config.max_d + 1):
            for q in range(config.max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    # A pure mean model is sometimes valid but rarely helpful for demand.
                    # Keep it available only if the series is very short.
                    if len(values) >= 2 * config.min_train_weeks:
                        continue
                order = (p, d, q)
                for seasonal_order in seasonal_candidates:
                    try:
                        model = SARIMAX(
                            values,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            trend="c",
                        )
                        result = model.fit(disp=False, maxiter=100)
                        aic = float(result.aic)
                        if math.isfinite(aic) and aic < best_aic:
                            best_result = result
                            best_order = order
                            best_seasonal_order = seasonal_order
                            best_aic = aic
                    except Exception as exc:
                        last_error = str(exc)
                        continue

    if best_result is None or best_order is None:
        raise RuntimeError(f"No ARIMA model converged. Last error: {last_error}")

    return ARIMAFitResult(
        model=best_result,
        order=best_order,
        seasonal_order=best_seasonal_order,
        aic=best_aic,
        method="custom_sarimax_aic_grid",
    )


def fit_arima_model(ts: pd.Series, config: ForecastConfig) -> ARIMAFitResult:
    """Fit ARIMA with automatic AIC-based selection and robust fallback."""

    ts = ts.astype(float).clip(lower=0.0)
    fallback_error: Optional[str] = None

    if len(ts) < 8 or ts.nunique() <= 1:
        return ARIMAFitResult(
            model=None,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            aic=None,
            method="naive_last_value",
            error="Series too short or constant for ARIMA.",
        )

    if config.use_pmdarima:
        try:
            return fit_arima_pmdarima(ts, config)
        except Exception as exc:
            fallback_error = str(exc)
            LOGGER.debug("pmdarima failed; falling back to custom AIC grid: %s", exc)

    try:
        result = fit_arima_custom_aic(ts, config)
        if fallback_error:
            result.error = f"pmdarima fallback reason: {fallback_error}"
        return result
    except Exception as exc:
        return ARIMAFitResult(
            model=None,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            aic=None,
            method="naive_last_value",
            error=str(exc),
        )


def forecast_arima(fit: ARIMAFitResult, ts: pd.Series, steps: int) -> np.ndarray:
    """Forecast from a fitted ARIMA result with naive fallback."""

    if steps <= 0:
        return np.asarray([], dtype=float)

    fallback = float(ts.iloc[-1]) if len(ts) else 0.0
    if fit.model is None:
        return np.repeat(max(fallback, 0.0), steps)

    try:
        if fit.method == "pmdarima.auto_arima":
            pred = fit.model.predict(n_periods=steps)
        else:
            pred = fit.model.forecast(steps=steps)
        return clamp_forecast(pred)
    except Exception:
        return np.repeat(max(fallback, 0.0), steps)


# -----------------------------------------------------------------------------
# Backtesting, hybrid optimization, future forecast
# -----------------------------------------------------------------------------


def make_backtest_origins(n_obs: int, config: ForecastConfig) -> List[int]:
    """Return train-end positions for walk-forward validation."""

    if n_obs < config.min_total_weeks:
        return []
    min_origin = max(config.min_train_weeks, max(config.lags, default=1) + config.min_lgbm_train_rows)
    max_origin = n_obs - config.backtest_horizon
    if max_origin < min_origin:
        # Relax LightGBM lag-driven origin when series is short.
        min_origin = config.min_train_weeks
    if max_origin < min_origin:
        return []

    origins = np.linspace(min_origin, max_origin, num=config.n_backtest_origins, dtype=int)
    return sorted(set(int(x) for x in origins if x + config.backtest_horizon <= n_obs))


def optimize_hybrid_weights(
    y_true: Sequence[float],
    arima_pred: Sequence[float],
    lgbm_pred: Sequence[float],
) -> Tuple[float, float]:
    """Constrained least-squares weight search: w_a + w_l = 1, each in [0, 1]."""

    y = np.asarray(y_true, dtype=float)
    a = np.asarray(arima_pred, dtype=float)
    l = np.asarray(lgbm_pred, dtype=float)

    if len(y) == 0:
        return 0.5, 0.5

    d = a - l
    denom = float(np.dot(d, d))
    if denom <= 1e-12:
        return 0.5, 0.5

    w_arima = float(np.dot(y - l, d) / denom)
    w_arima = min(1.0, max(0.0, w_arima))
    w_lgbm = 1.0 - w_arima
    return w_arima, w_lgbm


def run_walk_forward_backtest(ts: pd.Series, config: ForecastConfig) -> pd.DataFrame:
    """Run rolling-origin validation for ARIMA and LightGBM."""

    rows: List[Dict[str, Any]] = []
    origins = make_backtest_origins(len(ts), config)
    if not origins:
        return pd.DataFrame()

    for origin_id, origin in enumerate(origins, start=1):
        train = ts.iloc[:origin]
        test = ts.iloc[origin : origin + config.backtest_horizon]
        future_idx = test.index

        arima_fit = fit_arima_model(train, config)
        arima_fc = forecast_arima(arima_fit, train, steps=len(test))

        lgbm_fit = fit_lgbm_model(train, config)
        lgbm_fc = forecast_lgbm_recursive(train, future_idx, config, fitted=lgbm_fit)

        for step, (week, actual, arima_value, lgbm_value) in enumerate(
            zip(future_idx, test.values, arima_fc, lgbm_fc), start=1
        ):
            rows.append(
                {
                    "Origin_ID": origin_id,
                    "Train_End_Week": train.index[-1],
                    "Week": week,
                    "Horizon_Step": step,
                    "Actual": float(actual),
                    "ARIMA_Forecast": float(arima_value),
                    "LightGBM_Forecast": float(lgbm_value),
                    "ARIMA_Order_Fold": str(arima_fit.order),
                    "ARIMA_Seasonal_Order_Fold": str(arima_fit.seasonal_order),
                    "ARIMA_AIC_Fold": safe_round(arima_fit.aic, 4),
                }
            )

    return pd.DataFrame(rows)


def make_future_index(ts: pd.Series, config: ForecastConfig) -> pd.DatetimeIndex:
    future_start = ts.index[-1] + pd.Timedelta(weeks=1)
    if config.forecast_end_date:
        end_date = pd.Timestamp(config.forecast_end_date)
        end_date = end_date - pd.to_timedelta(end_date.weekday(), unit="d")
        if end_date < future_start:
            return pd.DatetimeIndex([])
        return pd.date_range(start=future_start, end=end_date, freq=config.weekly_frequency)
    return pd.date_range(start=future_start, periods=config.forecast_weeks, freq=config.weekly_frequency)


def evaluate_pair(
    code: Any,
    division: Any,
    group: pd.DataFrame,
    config: ForecastConfig,
) -> Optional[PairResult]:
    """Evaluate one SKU-Division pair end-to-end."""

    warnings_list: List[str] = []
    product_name = get_product_name(group, config)
    ts = build_weekly_series(group, config)

    if len(ts) < config.min_total_weeks:
        LOGGER.warning("Skipping %s | %s: only %s weeks", code, division, len(ts))
        return None

    try:
        backtest = run_walk_forward_backtest(ts, config)
        if backtest.empty:
            LOGGER.warning("Skipping %s | %s: no valid backtest origins", code, division)
            return None

        w_arima, w_lgbm = optimize_hybrid_weights(
            backtest["Actual"],
            backtest["ARIMA_Forecast"],
            backtest["LightGBM_Forecast"],
        )
        backtest["Hybrid_Forecast"] = (
            w_arima * backtest["ARIMA_Forecast"]
            + w_lgbm * backtest["LightGBM_Forecast"]
        )
        backtest["Hybrid_Forecast"] = clamp_forecast(backtest["Hybrid_Forecast"])

        metrics = {
            "ARIMA": compute_metrics(backtest["Actual"], backtest["ARIMA_Forecast"]),
            "LightGBM": compute_metrics(backtest["Actual"], backtest["LightGBM_Forecast"]),
            "Hybrid": compute_metrics(backtest["Actual"], backtest["Hybrid_Forecast"]),
        }
        selected_model = min(MODEL_NAMES, key=lambda m: metrics[m].get("RMSE", np.inf))
        selected_col = f"{selected_model}_Forecast" if selected_model != "LightGBM" else "LightGBM_Forecast"
        backtest["Selected_Model"] = selected_model
        backtest["Selected_Forecast"] = backtest[selected_col]

        final_arima = fit_arima_model(ts, config)
        if final_arima.error:
            warnings_list.append(f"ARIMA: {final_arima.error}")

        final_lgbm = fit_lgbm_model(ts, config)
        if final_lgbm.error:
            warnings_list.append(f"LightGBM: {final_lgbm.error}")

        future_idx = make_future_index(ts, config)
        arima_future = forecast_arima(final_arima, ts, len(future_idx))
        lgbm_future = forecast_lgbm_recursive(ts, future_idx, config, fitted=final_lgbm)
        hybrid_future = clamp_forecast(w_arima * arima_future + w_lgbm * lgbm_future)

        future = pd.DataFrame(
            {
                "Week": future_idx,
                "ARIMA_Forecast": arima_future,
                "LightGBM_Forecast": lgbm_future,
                "Hybrid_Forecast": hybrid_future,
            }
        )
        if not future.empty:
            future["Selected_Model"] = selected_model
            future["Selected_Forecast"] = future[
                f"{selected_model}_Forecast" if selected_model != "LightGBM" else "LightGBM_Forecast"
            ]
            future["Hybrid_w_ARIMA"] = w_arima
            future["Hybrid_w_LightGBM"] = w_lgbm

        return PairResult(
            code=code,
            division=division,
            product_name=product_name,
            n_weeks=len(ts),
            first_week=ts.index[0],
            last_week=ts.index[-1],
            selected_model=selected_model,
            arima_order=final_arima.order,
            arima_seasonal_order=final_arima.seasonal_order,
            arima_aic=final_arima.aic,
            arima_method=final_arima.method,
            lightgbm_params=config.lgbm_params,
            hybrid_w_arima=w_arima,
            hybrid_w_lightgbm=w_lgbm,
            metrics=metrics,
            backtest_detail=backtest,
            future_forecast=future,
            series=ts,
            warnings=warnings_list,
        )
    except Exception as exc:
        LOGGER.exception("Failed pair %s | %s", code, division)
        return PairResult(
            code=code,
            division=division,
            product_name=product_name,
            n_weeks=len(ts),
            first_week=ts.index[0] if len(ts) else pd.NaT,
            last_week=ts.index[-1] if len(ts) else pd.NaT,
            selected_model="ERROR",
            arima_order=None,
            arima_seasonal_order=None,
            arima_aic=None,
            arima_method="none",
            lightgbm_params=config.lgbm_params,
            hybrid_w_arima=0.5,
            hybrid_w_lightgbm=0.5,
            metrics={},
            backtest_detail=pd.DataFrame(),
            future_forecast=pd.DataFrame(),
            series=ts,
            warnings=[str(exc)],
        )


# -----------------------------------------------------------------------------
# Export and visualization
# -----------------------------------------------------------------------------


def pair_summary_record(result: PairResult) -> Dict[str, Any]:
    record = {
        "Code": result.code,
        "Division": result.division,
        "Product_Name": result.product_name,
        "N_Weeks": result.n_weeks,
        "First_Week": result.first_week,
        "Last_Week": result.last_week,
        "Selected_Model": result.selected_model,
        "ARIMA_Order": str(result.arima_order),
        "ARIMA_Seasonal_Order": str(result.arima_seasonal_order),
        "ARIMA_AIC": safe_round(result.arima_aic, 4),
        "ARIMA_Method": result.arima_method,
        "Hybrid_w_ARIMA": safe_round(result.hybrid_w_arima, 6),
        "Hybrid_w_LightGBM": safe_round(result.hybrid_w_lightgbm, 6),
        "LightGBM_Params": json.dumps(result.lightgbm_params, sort_keys=True),
        "Warnings": " | ".join(result.warnings),
    }
    for model_name in MODEL_NAMES:
        model_metrics = result.metrics.get(model_name, {})
        for metric_name in ["RMSE", "MAE", "SMAPE", "WAPE", "Bias", "Bias_%"]:
            record[f"{model_name}_{metric_name}"] = safe_round(model_metrics.get(metric_name), 4)
    return record


def build_output_tables(results: List[PairResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_results = [r for r in results if r is not None]

    summary_df = pd.DataFrame([pair_summary_record(r) for r in valid_results])

    backtest_frames = []
    future_frames = []
    metric_rows = []

    for r in valid_results:
        if not r.backtest_detail.empty:
            tmp = r.backtest_detail.copy()
            tmp.insert(0, "Division", r.division)
            tmp.insert(0, "Code", r.code)
            tmp.insert(2, "Product_Name", r.product_name)
            tmp["Hybrid_w_ARIMA"] = r.hybrid_w_arima
            tmp["Hybrid_w_LightGBM"] = r.hybrid_w_lightgbm
            backtest_frames.append(tmp)

        if not r.future_forecast.empty:
            tmp = r.future_forecast.copy()
            tmp.insert(0, "Division", r.division)
            tmp.insert(0, "Code", r.code)
            tmp.insert(2, "Product_Name", r.product_name)
            future_frames.append(tmp)

        for model_name, metrics in r.metrics.items():
            row = {
                "Code": r.code,
                "Division": r.division,
                "Product_Name": r.product_name,
                "Model": model_name,
                "Selected_Model": r.selected_model,
            }
            row.update({k: safe_round(v, 4) for k, v in metrics.items()})
            metric_rows.append(row)

    backtest_detail_df = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    future_forecast_df = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    selected_cols = [
        "Code",
        "Division",
        "Product_Name",
        "N_Weeks",
        "Selected_Model",
        "ARIMA_Order",
        "ARIMA_AIC",
        "Hybrid_w_ARIMA",
        "Hybrid_w_LightGBM",
        "ARIMA_RMSE",
        "LightGBM_RMSE",
        "Hybrid_RMSE",
        "Warnings",
    ]
    selected_model_summary_df = summary_df[[c for c in selected_cols if c in summary_df.columns]].copy()

    return summary_df, backtest_detail_df, future_forecast_df, metrics_df, selected_model_summary_df


def export_excel_outputs(results: List[PairResult], config: ForecastConfig) -> Dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, backtest_df, future_df, metrics_df, selected_df = build_output_tables(results)

    paths = {
        "summary": output_dir / "summary.xlsx",
        "backtest_detail": output_dir / "backtest_detail.xlsx",
        "future_forecast": output_dir / "future_forecast.xlsx",
        "model_metrics": output_dir / "model_metrics.xlsx",
        "selected_model_summary": output_dir / "selected_model_summary.xlsx",
    }

    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        config_df = pd.DataFrame(
            [(k, json.dumps(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in asdict(config).items()],
            columns=["Parameter", "Value"],
        )
        config_df.to_excel(writer, sheet_name="Config", index=False)

    backtest_df.to_excel(paths["backtest_detail"], index=False)
    future_df.to_excel(paths["future_forecast"], index=False)
    metrics_df.to_excel(paths["model_metrics"], index=False)
    selected_df.to_excel(paths["selected_model_summary"], index=False)

    return paths


def _aggregate_backtest_for_plot(backtest: pd.DataFrame) -> pd.DataFrame:
    cols = ["Actual", "ARIMA_Forecast", "LightGBM_Forecast", "Hybrid_Forecast", "Selected_Forecast"]
    agg = backtest.groupby("Week", as_index=False)[cols].mean().sort_values("Week")
    return agg


def generate_charts_pdf(results: List[PairResult], config: ForecastConfig) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "forecast_charts.pdf"

    valid_results = [r for r in results if r is not None and r.selected_model != "ERROR"]

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        winner_counts = pd.Series([r.selected_model for r in valid_results]).value_counts().to_dict()
        title_text = (
            "Demand Forecasting Report\n"
            "ARIMA vs LightGBM vs Optimized Hybrid\n\n"
            f"Pairs processed: {len(valid_results)}\n"
            f"Winner counts: {winner_counts}\n"
            f"Backtest horizon: {config.backtest_horizon} weeks | Origins: {config.n_backtest_origins}\n"
            f"Future horizon: {config.forecast_end_date or str(config.forecast_weeks) + ' weeks'}"
        )
        ax.text(0.05, 0.85, title_text, ha="left", va="top", fontsize=16, linespacing=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Global average RMSE by model
        metric_rows = []
        for r in valid_results:
            for model_name in MODEL_NAMES:
                metric_rows.append(
                    {
                        "Pair": f"{r.code}-{r.division}",
                        "Model": model_name,
                        "RMSE": r.metrics.get(model_name, {}).get("RMSE", np.nan),
                        "Winner": r.selected_model,
                    }
                )
        metrics_plot_df = pd.DataFrame(metric_rows)
        if not metrics_plot_df.empty:
            avg_rmse = metrics_plot_df.groupby("Model")["RMSE"].mean().reindex(MODEL_NAMES)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(avg_rmse.index, avg_rmse.values, color=[COLOR_MAP[m] for m in avg_rmse.index])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")
            ax.set_title("Average Backtest RMSE by Model", fontsize=14, fontweight="bold")
            ax.set_ylabel("RMSE")
            ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Per-pair charts
        for r in valid_results:
            bt = _aggregate_backtest_for_plot(r.backtest_detail)
            future = r.future_forecast.copy()

            # 1) Actual vs backtest forecasts
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(r.series.index, r.series.values, label="History", color=COLOR_MAP["History"], linewidth=1.5)
            if not bt.empty:
                ax.plot(bt["Week"], bt["Actual"], label="Actual", color=COLOR_MAP["Actual"], linewidth=2.0)
                ax.plot(bt["Week"], bt["ARIMA_Forecast"], label="ARIMA", color=COLOR_MAP["ARIMA"], linestyle="--")
                ax.plot(bt["Week"], bt["LightGBM_Forecast"], label="LightGBM", color=COLOR_MAP["LightGBM"], linestyle=":")
                ax.plot(bt["Week"], bt["Hybrid_Forecast"], label="Hybrid", color=COLOR_MAP["Hybrid"], linestyle="-.")
                ax.plot(
                    bt["Week"],
                    bt["Selected_Forecast"],
                    label=f"Winner: {r.selected_model}",
                    color=COLOR_MAP["Selected"],
                    linewidth=2.8,
                )
            ax.set_title(f"Actual vs Backtest Forecast | SKU {r.code} | Division {r.division}\nWinner: {r.selected_model}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Week")
            ax.set_ylabel("Cases Sold")
            ax.grid(alpha=0.25)
            ax.legend(ncol=3, fontsize=9)
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # 2) Future forecast
            fig, ax = plt.subplots(figsize=(14, 6))
            hist_tail = r.series.iloc[-config.chart_last_history_weeks :]
            ax.plot(hist_tail.index, hist_tail.values, label="Historical Demand", color=COLOR_MAP["Actual"], linewidth=2.0)
            if not future.empty:
                ax.axvline(future["Week"].min(), color="#666666", linestyle="--", alpha=0.7)
                ax.plot(future["Week"], future["ARIMA_Forecast"], label="ARIMA", color=COLOR_MAP["ARIMA"], linestyle="--")
                ax.plot(future["Week"], future["LightGBM_Forecast"], label="LightGBM", color=COLOR_MAP["LightGBM"], linestyle=":")
                ax.plot(future["Week"], future["Hybrid_Forecast"], label="Hybrid", color=COLOR_MAP["Hybrid"], linestyle="-.")
                ax.plot(
                    future["Week"],
                    future["Selected_Forecast"],
                    label=f"Selected Forecast ({r.selected_model})",
                    color=COLOR_MAP["Selected"],
                    linewidth=2.8,
                    marker="o",
                    markersize=3,
                )
            ax.set_title(
                f"Future Weekly Forecast | SKU {r.code} | Division {r.division}\n"
                f"Hybrid weights: ARIMA={r.hybrid_w_arima:.3f}, LightGBM={r.hybrid_w_lightgbm:.3f}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("Week")
            ax.set_ylabel("Cases Sold")
            ax.grid(alpha=0.25)
            ax.legend(ncol=3, fontsize=9)
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # 3) Model comparison RMSE bar chart
            rmse_values = [r.metrics.get(model, {}).get("RMSE", np.nan) for model in MODEL_NAMES]
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = [COLOR_MAP[m] for m in MODEL_NAMES]
            bars = ax.bar(MODEL_NAMES, rmse_values, color=colors)
            for bar, model_name, value in zip(bars, MODEL_NAMES, rmse_values):
                if model_name == r.selected_model:
                    bar.set_edgecolor(COLOR_MAP["Selected"])
                    bar.set_linewidth(3)
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
            ax.set_title(f"RMSE Comparison | SKU {r.code} | Division {r.division}\nWinner: {r.selected_model}", fontsize=13, fontweight="bold")
            ax.set_ylabel("RMSE")
            ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return pdf_path


# -----------------------------------------------------------------------------
# Main pipeline entrypoint
# -----------------------------------------------------------------------------


def run_pipeline(
    input_path: str | Path,
    config: Optional[ForecastConfig] = None,
    sheet_name: Optional[str] = None,
) -> Dict[str, Path]:
    """Run the complete forecasting pipeline and return output file paths."""

    config = config or ForecastConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess_data(input_path, config=config, sheet_name=sheet_name)
    pairs_df = df[[config.code_col, config.division_col]].drop_duplicates().sort_values([config.code_col, config.division_col])
    pairs = list(pairs_df.itertuples(index=False, name=None))
    LOGGER.info("Forecasting %s SKU-Division pairs", len(pairs))

    group_lookup = {
        (code, division): group
        for (code, division), group in df.groupby([config.code_col, config.division_col], sort=False)
    }

    if Parallel is not None and config.n_jobs != 1:
        results = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(evaluate_pair)(code, division, group_lookup[(code, division)], config)
            for code, division in pairs
        )
    else:
        results = [
            evaluate_pair(code, division, group_lookup[(code, division)], config)
            for code, division in pairs
        ]

    results = [r for r in results if r is not None]
    LOGGER.info("Successfully evaluated %s pairs", len(results))

    paths = export_excel_outputs(results, config)
    paths["forecast_charts"] = generate_charts_pdf(results, config)

    for name, path in paths.items():
        LOGGER.info("Saved %s: %s", name, path)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly demand forecasting pipeline: ARIMA + LightGBM + Hybrid")
    parser.add_argument("--input", required=True, help="Input Excel/CSV file, for example dfr_prt.xlsx")
    parser.add_argument("--sheet", default=None, help="Excel sheet name. Leave empty for first sheet.")
    parser.add_argument("--output-dir", default="forecast_output", help="Output folder")
    parser.add_argument("--date-col", default="Week")
    parser.add_argument("--code-col", default="Code")
    parser.add_argument("--division-col", default="Division")
    parser.add_argument("--target-col", default="Cases Sold - Final")
    parser.add_argument("--product-col", default="Product name")
    parser.add_argument("--forecast-weeks", type=int, default=52)
    parser.add_argument("--forecast-end-date", default=None, help="Optional final forecast date, e.g. 2026-12-28")
    parser.add_argument("--backtest-horizon", type=int, default=8)
    parser.add_argument("--n-backtest-origins", type=int, default=5)
    parser.add_argument("--min-train-weeks", type=int, default=24)
    parser.add_argument("--min-total-weeks", type=int, default=30)
    parser.add_argument("--seasonal-arima", action="store_true", help="Enable seasonal ARIMA when enough data is available")
    parser.add_argument("--seasonal-period", type=int, default=52)
    parser.add_argument("--zero-as-missing", action="store_true", help="Treat zero target values as missing before interpolation")
    parser.add_argument("--missing-week-strategy", choices=["interpolate", "ffill", "zero"], default="interpolate")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-pmdarima", action="store_true", help="Force custom SARIMAX AIC grid search instead of pmdarima")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    product_col = args.product_col if args.product_col else None
    config = ForecastConfig(
        date_col=args.date_col,
        code_col=args.code_col,
        division_col=args.division_col,
        target_col=args.target_col,
        product_col=product_col,
        output_dir=args.output_dir,
        forecast_weeks=args.forecast_weeks,
        forecast_end_date=args.forecast_end_date,
        backtest_horizon=args.backtest_horizon,
        n_backtest_origins=args.n_backtest_origins,
        min_train_weeks=args.min_train_weeks,
        min_total_weeks=args.min_total_weeks,
        seasonal_arima=args.seasonal_arima,
        seasonal_period=args.seasonal_period,
        zero_as_missing=args.zero_as_missing,
        missing_week_strategy=args.missing_week_strategy,
        n_jobs=args.n_jobs,
        use_pmdarima=not args.no_pmdarima,
    )

    run_pipeline(input_path=args.input, config=config, sheet_name=args.sheet)


if __name__ == "__main__":
    main()
