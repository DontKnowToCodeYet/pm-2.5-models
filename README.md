# PM2.5 Models

Forecasting hourly PM2.5 at **Anand Vihar, Delhi** using CPCB air-quality data. Baseline models are compared against a novel model.

## Data

`data/Delhi_AnandVihar_2025_Aug2026_11pollutants_combined_unimputed.csv`

- Hourly readings from 1 Jan 2025 to 31 Aug 2026 (14,592 rows)
- 11 pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene
- Unimputed: 941 PM2.5 values (6.4%) are missing

## Evaluation protocol (use this for every model)

- **Chronological 80/20 split**, no shuffling
  - Train: 2025-01-01 00:00 to 2026-05-02 08:00
  - Test: 2026-05-02 09:00 to 2026-08-31 23:00
- **Metrics:** MAE, RMSE, MAPE, R²
- **Horizons:** 1, 3, 6, 12 and 24 hours ahead
- **Gaps:** a pair is scored only when both the input value and the target value were observed

> Note: the test window is the monsoon season (lower PM2.5), while training includes winters. Keep the split identical across models so the comparison stays fair.

## Models

| Model | Status | Folder |
|---|---|---|
| Persistence (naive) | ✅ Done | `models/persistence/` |
| SARIMA | ✅ Done | `models/sarima/` |

### Comparison on the test set

| Horizon | Persistence MAE | SARIMA MAE | Persistence RMSE | SARIMA RMSE | Persistence R² | SARIMA R² |
|---|---|---|---|---|---|---|
| 1 h | 14.71 | **14.27** | 21.43 | **20.60** | 0.693 | **0.717** |
| 3 h | 27.59 | **24.24** | 37.33 | **32.71** | 0.081 | **0.295** |
| 6 h | 32.28 | **25.37** | 42.88 | **34.37** | −0.210 | **0.222** |
| 12 h | 31.12 | **24.75** | 42.28 | **34.26** | −0.184 | **0.222** |
| 24 h | 33.21 | **25.60** | 45.71 | **35.77** | −0.374 | **0.158** |

MAE and RMSE are in µg/m³. Best value per horizon in bold.

### 1. Persistence (naive forecast)

The forecast h hours ahead equals the last observed value: ŷ(t+h) = y(t). The model has no learnable parameters.

**Test-set results**

| Horizon | MAE (µg/m³) | RMSE | MAPE | R² |
|---|---|---|---|---|
| 1 h | 14.71 | 21.43 | 27.8% | 0.693 |
| 3 h | 27.59 | 37.33 | 64.4% | 0.081 |
| 6 h | 32.28 | 42.88 | 88.6% | −0.210 |
| 12 h | 31.12 | 42.28 | 65.1% | −0.184 |
| 24 h | 33.21 | 45.71 | 79.5% | −0.374 |

Full metrics, including train-set results and a forward-fill variant, are in `results/persistence/persistence_metrics.csv`. The 1-hour-ahead test predictions are in `results/persistence/persistence_test_predictions_1h.csv`.

### 2. SARIMA

Seasonal ARIMA with a 24-hour daily cycle, fitted to the seasonally differenced series w(t) = y(t) − y(t−24).

- **Selected order:** SARIMA(3,0,2)(1,1,1)₂₄, chosen by lowest AIC from 36 candidates (p ∈ {1,2,3}, q ∈ {0,1,2}, P ∈ {0,1}, Q ∈ {0,1}) fitted on the training data
- **Estimation:** conditional sum of squares (CSS), implemented with numpy and scipy. CSS is a standard Box–Jenkins estimator, but `statsmodels` defaults to exact maximum likelihood, so its coefficients may differ slightly.
- **Forecasting:** parameters are fixed after training. At every test hour, forecasts 1–24 h ahead use only data up to that hour. Missing inputs are forward-filled, so no future values are used.

**Test-set results**

| Horizon | MAE (µg/m³) | RMSE | MAPE | R² |
|---|---|---|---|---|
| 1 h | 14.27 | 20.60 | 27.6% | 0.717 |
| 3 h | 24.24 | 32.71 | 59.2% | 0.295 |
| 6 h | 25.37 | 34.37 | 68.8% | 0.222 |
| 12 h | 24.75 | 34.26 | 59.4% | 0.222 |
| 24 h | 25.60 | 35.77 | 61.1% | 0.158 |

Files in `results/sarima/`:
- `sarima_metrics.csv`: train and test metrics at every horizon
- `sarima_order_selection.csv`: AIC for all 36 candidate orders
- `sarima_params.txt`: fitted coefficients
- `sarima_test_predictions_1h.csv`: 1-hour-ahead test predictions

## Run

```bash
pip install -r requirements.txt
python models/persistence/persistence_model.py
python models/sarima/sarima_model.py
```

Run from the repository root.
