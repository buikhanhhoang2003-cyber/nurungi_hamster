# Refactored Demand Forecasting Pipeline

This package replaces the old XGBoost workflow with a production-style weekly demand forecasting pipeline using:

1. Auto-ARIMA / SARIMAX selected by AIC
2. LightGBMRegressor
3. Optimized Hybrid forecast: `w_arima * ARIMA + w_lightgbm * LightGBM`

The pipeline is designed for SKU-Division weekly demand forecasting, rolling / walk-forward validation, and future demand projection.

## Files

- `forecasting_pipeline_lightgbm_arima_hybrid.py` - main production script
- `Forecasting_Pipeline_LightGBM_ARIMA_Hybrid.ipynb` - notebook runner
- `requirements_forecasting_pipeline.txt` - dependencies

## Expected input columns

Default input assumes a preprocessed Excel or CSV file such as `dfr_prt.xlsx` with:

- `Week`
- `Code`
- `Division`
- `Cases Sold - Final`
- optional: `Product name`

If using raw POS data, change `--target-col` to `Cases Sold` and consider using `--zero-as-missing` because the thesis data treats zero sales records as missing values after validation.

## Install dependencies

```bash
pip install -r requirements_forecasting_pipeline.txt
```

XGBoost is not required and is not imported anywhere.

## Run from command line

```bash
python forecasting_pipeline_lightgbm_arima_hybrid.py \
  --input dfr_prt.xlsx \
  --output-dir forecast_output \
  --target-col "Cases Sold - Final" \
  --forecast-weeks 52 \
  --backtest-horizon 8 \
  --n-backtest-origins 5 \
  --n-jobs -1
```

To enable seasonal ARIMA when enough weekly history is available:

```bash
python forecasting_pipeline_lightgbm_arima_hybrid.py \
  --input dfr_prt.xlsx \
  --seasonal-arima \
  --seasonal-period 52
```

## Outputs

The script writes these files into `forecast_output/`:

- `summary.xlsx`
- `backtest_detail.xlsx`
- `future_forecast.xlsx`
- `model_metrics.xlsx`
- `selected_model_summary.xlsx`
- `forecast_charts.pdf`

## Leakage prevention

- All LightGBM lag and rolling features are shifted before target construction.
- Backtesting uses chronological rolling-origin validation only.
- ARIMA and LightGBM are refit using training data available at each origin.
- Future LightGBM forecasts are recursive and use only historical or previously predicted values.
- Negative forecasts are clamped to zero.

## Hybrid weight optimization

Hybrid weights are optimized on backtest predictions by constrained least squares:

- `w_arima + w_lightgbm = 1`
- `0 <= w_arima <= 1`
- `0 <= w_lightgbm <= 1`

The selected model is the model with the lowest backtest RMSE among ARIMA, LightGBM, and Hybrid.
