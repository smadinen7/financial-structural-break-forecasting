# Financial Structural Break Detection & Forecasting

Detecting regime changes in financial time series and building break-aware forecasting models that outperform naive baselines.

---

## Overview

Generic forecasting models assume stationarity — but real financial data doesn't. Interest rate hikes, recessions, earnings shocks, and policy changes create structural breaks that silently destroy forecast accuracy. This project detects those breaks, then shows how accounting for them measurably improves forecasts.

---

## What This Project Does

**Break Detection**
- Chow Test — classic parametric test for a known breakpoint
- CUSUM — cumulative sum control chart for unknown breakpoints
- Bai-Perron — multiple structural break detection

**Forecasting**
- Baseline models: ARIMA, Prophet
- Deep learning: LSTM
- Break-naive vs. break-aware comparison on the same datasets

**Evaluation**
- RMSE, MAE, MAPE across all model/regime combinations
- Visualizations of detected breakpoints overlaid on time series
- Summary report: does break-awareness improve accuracy? By how much?

---

## Dataset

FRED (Federal Reserve Economic Data) — publicly available macroeconomic and financial indicators including GDP, CPI, unemployment, and interest rates.

---

## Stack

| Component | Tool |
|-----------|------|
| Data | FRED API (`fredapi`) |
| Break Detection | `ruptures`, `statsmodels` |
| Forecasting | `statsmodels`, `prophet`, `pytorch` |
| Tracking | MLflow |
| Visualization | Matplotlib, Plotly |
| Environment | Python 3.11, Docker |

---

## Project Structure

```
financial-structural-break-forecasting/
├── data/                   # Raw and processed time series data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_break_detection.ipynb
│   ├── 03_forecasting_baseline.ipynb
│   └── 04_break_aware_forecasting.ipynb
├── src/
│   ├── break_detection.py
│   ├── forecasting.py
│   └── evaluation.py
├── results/                # Saved metrics, plots, reports
├── requirements.txt
└── README.md
```

---

## Key Results

*In progress — results will be updated as experiments complete.*

---

## Background

This project is directly inspired by work done at Amazon Stores Finance, where structural break detection was applied to P&L forecasting across 840+ financial datasets. This repo recreates and extends that methodology on public data.
