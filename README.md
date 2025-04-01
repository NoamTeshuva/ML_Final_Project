# 🧠 ML Final Project: Predicting Small-Cap Stock Movements

## 📈 Goal
Use machine learning models to predict whether a small-cap stock (from the Russell 2000 Index) will go **up or down tomorrow** based on historical technical indicators.

## 🗂️ Dataset
- **Source**: Yahoo Finance via `yfinance`
- **Stocks**: 200 randomly selected small-cap stocks from the Russell 2000
- **Period**: 2015–2025 (daily prices)
- **Target**: Binary — `1` if tomorrow's closing price > today's, else `0`

## 🛠️ Features Used
- SMA_7, SMA_14 — short-term moving averages
- SMA_50, SMA_200 — medium/long-term moving averages
- RSI_14 — Relative Strength Index
- MACD, MACD_Signal, MACD_Hist — momentum indicators
- Volatility — price variation metric

## 🤖 Models Implemented
- **Logistic Regression** – Simple linear baseline
- **AdaBoost** – Boosted weak learners
- **K-Means Clustering** – Used for behavioral grouping (unsupervised)
- **Random Forest** – Ensemble of decision trees
- **Baseline Model** – "Predict same as yesterday" rule

## ⚙️ Accuracy Results
| Model             | Accuracy |
|------------------|----------|
| Logistic          | ~51.5%   |
| AdaBoost          | ~51.7%   |
| Random Forest     | ~51.8%   |
| Baseline (naive)  | ~60.0%   |

> ⚠️ The rule-based baseline surprisingly performed better than ML models — highlighting the noisy, random nature of short-term market predictions.

## 📊 Visualizations
- Accuracy comparison bar chart
- Confusion matrices
- Random Forest feature importance chart
- Stock price + SMA trendline example

All visuals saved under `/visuals/`.
