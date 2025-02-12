# **Outperforming Russell 2000 Using Machine Learning**

## 📌 **Project Overview**
This project aims to build a **Machine Learning (ML) model** to predict small-cap stock performance and construct a **portfolio of 10 stocks per year** that outperforms the **Russell 2000 index**.

## 🚀 **1. Project Goals**
- ✅ Use **Yahoo Finance** to collect stock data (2015-2025).
- ✅ Extract **technical indicators** (SMA, RSI, MACD, Volatility).
- ✅ Train **multiple ML models** to predict stock movements.
- ✅ Select the **top 10 best stocks per year** (2020-2025) and compare performance against Russell 2000.
- ✅ **Backtest the strategy** and analyze risk-adjusted returns.

---
##✅ **Time-Series Classification Sequence**
This sequence explains how time-series classification is applied in this project.

1️⃣ Collect Stock Data (2015-2025)

Download daily stock price data from Yahoo Finance (yfinance).
Each stock’s data is stored chronologically to preserve time dependencies.
2️⃣ Feature Engineering (Extracting Time-Series Indicators)

Compute rolling-window indicators like SMA, RSI, MACD, Volatility to detect trends.
These indicators summarize past price movements over different time periods.
3️⃣ Label Creation (Defining the Prediction Target)

Label each stock as 1 (price increases) or 0 (price decreases).
The next day’s close price determines the label.
4️⃣ Time-Aware Train-Test Splitting

Training: 2016-2020 (past data).
Testing: 2020-2025 (future data).
No random shuffling—models must learn in chronological order.
5️⃣ Training ML Models on Sequential Data

Models detect patterns from past trends to predict stock movement.
Decision Tree, Logistic Regression, AdaBoost, PCA, and K-Means are trained on historical sequences.
6️⃣ Backtesting on Unseen Future Data

The model selects top 10 stocks per year (2020-2025) based on predictions.
Simulated portfolio returns are compared with Russell 2000.
7️⃣ Performance Evaluation

Accuracy of predictions (did the stock move as expected?).
Portfolio performance metrics (CAGR, Sharpe Ratio).
Adjustments based on model results to improve future performance.
📌 Outcome:
By maintaining chronological order, using rolling features, and testing on future unseen data, this project ensures true time-series classification.
---

## 📂 **2. Project Structure**
```
ML_final_project/
│── project/
│   ├── data/                  # Processed stock data
│   │   ├── processed/         # Cleaned & feature-engineered stock data
│   │   ├── training_data.csv  # Dataset for model training (2015-2020)
│   │   ├── testing_data.csv   # Dataset for model testing (2020-2025)
│   │
│   ├── models/                # Trained ML models
│   │   ├── dt_model.joblib    # Decision Tree model
│   │   ├── log_model.joblib   # Logistic Regression model
│   │   ├── adaboost_model.joblib   # Adaboost model
│   │   ├── pca_model.joblib   # PCA model
│   │   ├── kmeans_model.joblib # K-Means Clustering model
│   │
│   ├── src/                   # Source code
│   │   ├── data_acquisition.py     # Fetch stock data from Yahoo Finance
│   │   ├── data_preprocessing.py   # Clean and prepare stock data
│   │   ├── feature_engineering.py  # Generate technical indicators
│   │   ├── model_preparation.py    # Merge and split training/testing datasets
│   │   ├── model_training.py       # Train ML models (Decision Tree, Logistic Regression, AdaBoost, PCA, Clustering)
│   │   ├── model_testing.py        # Test model accuracy on unseen data
│   │   ├── backtesting.py          # Evaluate strategy vs. Russell 2000
│
│── README.md                 # Project documentation
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignore large files (CSV, model files)
```

---

## 🏗 **3. Steps in the Project**

### 📌 **1️⃣ Data Collection (`data_acquisition.py`)**
- **Fetches historical stock data from Yahoo Finance (`yfinance`).**
- **Stores each stock’s data in `data/processed/*.csv`.**

✅ **Example Data:**
```
Date,Open,High,Low,Close,Volume
2015-02-09,26.76,26.79,26.47,26.50,155559200
2015-02-10,27.28,27.30,26.86,26.86,248034000
```

### 📌 **2️⃣ Feature Engineering (`feature_engineering.py`)**
**Adds technical indicators**:
| Feature  | Description |
|----------|------------|
| `SMA_50` | 50-day moving average |
| `SMA_200` | 200-day moving average |
| `RSI_14` | 14-day Relative Strength Index |
| `MACD_12_26_9` | MACD Line |
| `Volatility` | Rolling standard deviation (20 days) |

### 📌 **3️⃣ Model Training (`model_training.py`)**
- **Trained on 2015-2020 stock data.**
- **Models Used:**
  - ❌ **SVM (Did not work due to slow training on large datasets)**
  - ✅ Decision Tree
  - ✅ Logistic Regression
  - ✅ AdaBoost
  - ✅ PCA (for dimensionality reduction)
  - ✅ K-Means Clustering (for exploratory analysis)
- **Saved models in `models/` directory.**



### 📌 **4️⃣ Model Testing (`model_testing.py`)**
- **Evaluates models on 2020-2025 data.**
- ❌ **Initial Accuracy was ~50% (Random Guessing).**
- ✅ **Identified issues:**
  - Dataset imbalance
  - Features not contributing significantly
  - Model not learning effectively
- ✅ **Fixes Applied:**
  - Feature selection improvements
  - Data balancing (SMOTE)
  - Extended training period (2016-2020 instead of 2018-2020)



### 📌 **5️⃣ Portfolio Backtesting (`backtesting.py`)**
- **Selects top 10 predicted stocks per year (2020-2025).**
- **Compares returns vs. Russell 2000.**



---

## 🔥 **4. How to Run the Project**

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run Full Pipeline**
```bash
python project/src/data_acquisition.py      # Fetch stock data
python project/src/data_preprocessing.py    # Clean data
python project/src/feature_engineering.py   # Generate technical indicators
python project/src/model_preparation.py     # Create training/testing datasets
python project/src/model_training.py        # Train Decision Tree, Logistic Regression, AdaBoost, PCA, K-Means
python project/src/model_testing.py         # Test model accuracy
python project/src/backtesting.py           # Compare ML portfolio vs. Russell 2000
```

---

## 📌 **5. Next Steps**
- 🟢 **Optimize stock selection using advanced ML models (XGBoost, Random Forest).**
- 🟢 **Improve feature selection for better accuracy.**
- 🟢 **Enhance dataset balancing techniques.**
- 🟢 **Deploy a dashboard to visualize stock predictions.**

---

## 📌 **6. Contributors**
👨‍💻 **Noam Teshuva**  
📩 **GitHub:** [NoamTeshuva](https://github.com/NoamTeshuva)

---

### 🎯 **Final Goal:** **Beat the Russell 2000 with AI-driven stock selection!** 🚀

