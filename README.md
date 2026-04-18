# Lifestyle Impact of Productivity
### A Machine Learning Perspective on Modern Work Habits
**Machine Learning 1 - Group 11**

| Student ID | Name |
|---|---|
| s16796 | Amasha Fernando |
| s16658 | Saviru Mendis |
| s16943 | Kaumindi Herath |
| s16877 | Dilmith Yahathugoda |

---

## ⚠️ Important Disclaimer

**This is a university learning exercise, not a production ML project.**

The goal was to practice and explore ML techniques - imputation methods, clustering, feature analysis, and a wide range of models - even in cases where simpler approaches would have sufficed. Many steps (e.g. testing 5 imputation strategies, trying multiple clustering algorithms, evaluating 9 model families) were done deliberately for learning purposes rather than necessity.

**The dataset is fully synthetic** - it was generated to simulate realistic lifestyle patterns and does not represent real people or real behavior. None of the findings should be interpreted as real-world conclusions about productivity or human habits.

---

## Overview

This project investigates how daily lifestyle habits - including sleep patterns, stress levels, work breaks, social media usage, and job satisfaction - influence individual productivity. Using a dataset of 30,000 simulated behavioral records from Kaggle, we build and evaluate multiple machine learning models to predict `actual_productivity_score`.

---

## Dataset

- **Source:** [Kaggle - Social Media vs Productivity](https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity)
- **Size:** 30,000 records
- **Target variable:** `actual_productivity_score` (continuous)
- **Key features:** age, sleep hours, stress level, daily social media time, work hours per day, job satisfaction score, burnout days per month, screen time before sleep, job type, and more

The dataset is downloaded automatically via `kagglehub` when you run the notebook.

---

## Project Structure

```
├── ML_SL_Final_Project_Group_11.ipynb   # Main notebook
├── requirements.txt                      # Python dependencies
├── Group_11.pdf                          # Full project report
└── README.md
```

---

## Methodology

**Data Preprocessing**
- Removed records with missing target values
- Capped unrealistic `daily_social_media_time` values (>12 hrs) at 12
- Compared 5 imputation strategies: Median, KNN, MICE, Bayesian-MICE, MissForest, and PMM
- Selected **PMM (Predictive Mean Matching)** as the final imputation method

**Cluster Analysis**
- Tried K-Prototype and DBSCAN clustering - neither provided meaningful structure or improved model performance

**EDA & Feature Analysis**
- Mutual Information revealed `job_satisfaction_score` as the only variable with a meaningful relationship to productivity
- All other lifestyle variables showed near-zero dependency

**Modeling** - Evaluated 9 model families:
- Linear Regression, Ridge, Lasso, ElasticNet
- K-Nearest Neighbors
- Random Forest, Gradient Boosting, XGBoost, LightGBM
- CatBoost, GAM

For full details see [`Project Report.pdf`](./Project Report.pdf).

---

## Results

| Model | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.6447 | 0.8384 | 1.1314 |
| Ridge / Lasso / ElasticNet | ~0.645 | ~0.839 | ~1.130 |
| KNN | 0.4003 | 1.2208 | 1.4698 |
| Random Forest | 0.6494 | 0.8317 | 1.1238 |
| Gradient Boosting | 0.6516 | 0.8302 | 1.1203 |
| XGBoost | 0.6511 | 0.8281 | 1.1211 |
| LightGBM | 0.6479 | 0.8318 | 1.1263 |
| **CatBoost** | **0.6522** | **0.8271** | **1.1194** |
| **GAM (Tuned)** ✅ | **0.6529** | **0.8241** | **1.1181** |

**Best model: GAM (Tuned)** - highest test R2 with interpretable structure.

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set up Kaggle credentials**

The notebook downloads the dataset automatically via `kagglehub`. You need a Kaggle account and API token:
- Go to [kaggle.com](https://www.kaggle.com) → Account → Create New Token
- Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`

**3. Run the notebook**
```bash
jupyter notebook ML_SL_Final_Project_Group_11.ipynb
```

---

## Tech Stack

See [`requirements.txt`](./requirements.txt) for full details. Key libraries:

- **Data:** `pandas`, `numpy`, `scipy`
- **ML:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `pygam`
- **Imputation:** `miceforest`
- **Clustering:** `kmodes`, `prince`
- **Explainability:** `shap`
- **EDA:** `ydata-profiling`, `seaborn`, `matplotlib`
- **Stats:** `statsmodels`
- **Data source:** `kagglehub`
