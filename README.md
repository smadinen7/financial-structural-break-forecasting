# Financial Structural Break Detection & Forecasting

Detecting regime changes in financial time series and building break-aware forecasting models that outperform naive baselines.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![MLflow](https://img.shields.io/badge/tracked%20with-MLflow-0194E2)

---

## Motivation

This project is directly inspired by work performed at Amazon Stores Finance, where structural break detection was applied to P&L forecasting across 840+ financial datasets to identify regime changes in business performance. This repository applies the same methodology at a macro level using publicly available FRED data — validating the approach on a reproducible, open dataset.

The core insight: generic forecasting models assume stationarity, but real financial data doesn't. Interest rate hikes, recessions, and policy shifts create structural breaks that silently destroy forecast accuracy. This project detects those breaks, then shows how accounting for them measurably improves forecasts.

---

## Overview

**Pipeline:**
```
FRED API → Raw Series → Break Detection (CUSUM / Bai-Perron / Chow) → Identified Regimes
    → Per-Regime Train/Test Split → Forecast (ARIMA / Prophet / LSTM*)
    → MLflow Metric Logging → Naive vs. Break-Aware Comparison
```

*LSTM is included as an experimental baseline; classical methods are expected to outperform on low-frequency macro series.

---

## What This Project Does

**Break Detection**
- **CUSUM** — cumulative sum control chart for unknown breakpoint discovery
- **Bai-Perron (via `ruptures` Dynp)** — optimal partitioning for multiple structural breaks; computationally equivalent to Bai-Perron (1998) but uses BIC model selection rather than sequential F-tests
- **Chow Test** — parametric validation of a candidate breakpoint at known economic events (e.g., Lehman collapse 2008-09-15, COVID shock 2020-03)

**Forecasting**
- ARIMA (statsmodels SARIMAX)
- Prophet (Meta/community)
- LSTM (PyTorch) — experimental; included to demonstrate limitations of deep learning on low-frequency macro data

**Evaluation**
- RMSE, MAE, MAPE per model per regime
- Naive (full-history) vs. break-aware (per-regime) training comparison
- Diebold-Mariano test for statistical significance of model differences
- All runs logged to MLflow

---

## FRED Series Used

| Series ID | Description | Frequency | Key Breaks |
|-----------|-------------|-----------|------------|
| `UNRATE` | Unemployment Rate | Monthly | 2008-09, 2020-04 |
| `FEDFUNDS` | Federal Funds Rate | Monthly | 1979-80, 2008, 2020, 2022 |
| `CPIAUCSL` | Consumer Price Index | Monthly | 1973-74, 2021-22 |
| `INDPRO` | Industrial Production | Monthly | 1973, 1981, 2008, 2020 |
| `GDP` | Gross Domestic Product | Quarterly | 2008-Q4, 2020-Q1 |

---

## Stack

| Component | Tool |
|-----------|------|
| Data | FRED API (`fredapi`) |
| Break Detection | `ruptures`, `statsmodels` |
| Forecasting | `statsmodels`, `prophet`, `pytorch` |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Plotly |
| Environment | Python 3.11, Docker |

**Dependency note:** `prophet` requires `cmdstanpy`. Pin `prophet==1.1.5` and run `cmdstanpy.install_cmdstan()` on first setup. See `requirements.txt` for full pinned versions.

---

## Quickstart

```bash
git clone https://github.com/smadinen7/financial-structural-break-forecasting
cd financial-structural-break-forecasting
cp .env.example .env          # add your FRED_API_KEY
pip install -r requirements.txt
python src/run_pipeline.py --series UNRATE FEDFUNDS CPIAUCSL
mlflow ui                     # view experiment results at localhost:5000
```

**FRED API Key:** Free and instant at [stlouisfed.org](https://fredaccount.stlouisfed.org/login/secure/). 120 requests/minute, no daily cap.

```bash
# .env.example
FRED_API_KEY=your_key_here
```

---

## Project Structure

```
financial-structural-break-forecasting/
├── data/                       # Raw and cached FRED series (CSV)
├── notebooks/
│   ├── 01_eda.ipynb            # Series exploration, stationarity tests
│   ├── 02_break_detection.ipynb
│   ├── 03_forecasting_baseline.ipynb
│   └── 04_break_aware_forecasting.ipynb
├── src/
│   ├── break_detection.py      # Chow, CUSUM, Bai-Perron via ruptures
│   ├── forecasting.py          # ARIMA, Prophet, LSTM wrappers
│   ├── evaluation.py           # RMSE, MAE, MAPE, Diebold-Mariano
│   └── run_pipeline.py         # End-to-end entry point
├── results/                    # Saved metrics, plots, MLflow runs
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Key Results

*In progress — target metrics below. Results will be updated as experiments complete.*

| Series | Break Year | ARIMA (naive) RMSE | ARIMA (break-aware) RMSE | Prophet RMSE |
|--------|------------|-------------------|--------------------------|-------------|
| UNRATE | 2020 | — | — | — |
| FEDFUNDS | 2008 | — | — | — |
| CPIAUCSL | 2021 | — | — | — |

**Hypothesis:** Break-aware training reduces RMSE by at least 15% on post-break windows vs. naive full-history training.

---

## Limitations

- `ruptures` Dynp does not reproduce the Bai-Perron sequential F-test statistics — breakpoint location estimates are consistent but no asymptotic critical values are computed
- LSTM results are sensitive to preprocessing choices and are included as a negative result baseline, not a performance claim
- FRED data is subject to retroactive revisions; raw series are cached as CSV on first pull for reproducibility
