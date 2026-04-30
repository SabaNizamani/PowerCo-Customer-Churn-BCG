# ⚡ PowerCo Customer Churn Prediction
### BCG X Data Science Job Simulation — Forage

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Simulation](https://img.shields.io/badge/BCG%20X-Forage%20Simulation-blue)

---

## 📌 Project Overview

This project was completed as part of the **BCG X Data Science Job Simulation** on Forage, working with **PowerCo** — a major European energy utility facing increasing customer churn.

The project follows a real-world consulting engagement structure across 4 tasks, from business framing all the way to delivering a predictive model and actionable insights.

**Central Business Question:**
> *Is customer churn at PowerCo primarily driven by price sensitivity?*

---

## 🗂️ Project Structure

```
PowerCo-Customer-Churn-BCG/
│
├── 📄 run_pipeline.py                   # Master script — runs all stages end-to-end
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 notebooks/
│   ├── 01_eda.py                        # Task 3 — Exploratory Data Analysis
│   ├── 02_feature_engineering.py        # Task 4 — Feature Engineering
│   └── 03_modelling.py                  # Task 5 — Modelling & Evaluation
│
├── 📁 docs/
│   └── task1_business_understanding.md  # Task 1 & 2 — Business framing & client email
│
├── 📁 data/                             # (not tracked by git — add your CSVs here)
│   ├── client_data.csv
│   ├── price_data.csv
│   ├── clean_data_after_eda.csv
│   └── data_for_predictions.csv
│
└── 📁 outputs/                          # Auto-generated plots, predictions & report
    ├── eda_01_churn_distribution.png
    ├── eda_02_distributions.png
    ├── model_01_evaluation_dashboard.png
    ├── model_02_feature_importances.png
    ├── out_of_sample_predictions.csv
    └── executive_summary.txt
```

---

## 🧩 The 4 Tasks

### Task 1 & 2 — Business Understanding & Client Communication
**Objective:** Understand the client brief, frame the hypothesis, and communicate our analytical plan.

- Translated "price sensitivity" into measurable data variables
- Designed a 5-step investigation framework
- Drafted a professional email to the client requesting the right data
- Identified the required datasets: customer data, churn data, historical pricing data

**Key output:** [`docs/task1_business_understanding.md`](docs/task1_business_understanding.md)

---

### Task 3 — Exploratory Data Analysis (EDA)
**Objective:** Understand the data, identify patterns, and flag data quality issues.

**Datasets:** `client_data.csv` + `price_data.csv`

Key findings:
- **~10% churn rate** — significant class imbalance requiring handling in modelling
- **Consumption variables** are highly positively skewed (log transformation needed)
- **Price differences** between churned and retained customers are present but small
- **Sales channel 'MISSING'** shows 7.6% churn — a potentially important feature
- **Short-tenure customers** churn at much higher rates

**Key output:** EDA plots in `/outputs/`

---

### Task 4 — Feature Engineering
**Objective:** Transform raw data into ML-ready features that maximise predictive power.

| Feature Group | Description | Rationale |
|---|---|---|
| **Dec–Jan price diff** | Off-peak price change over the year | Colleague's recommended feature — macro yearly price signal |
| **Mean price diffs** | Average off-peak/peak/mid-peak differences | Micro price variance across billing periods |
| **Max monthly price diff** | Largest price spike a customer experienced | Sudden spikes most likely to trigger churn decision |
| **Tenure** | Years as a PowerCo customer | Short-tenure customers are the highest churn risk |
| **Date → months** | `months_activ`, `months_to_end`, `months_modif_prod`, `months_renewal` | Raw dates can't enter ML models — months encode the same signal |
| **has_gas** | Boolean → binary (1/0) | Gas customers ~2% less likely to churn (multi-product loyalty) |
| **One-hot encoding** | `channel_sales`, `origin_up` | ML cannot use string values directly |
| **Log transform** | 10 skewed numeric columns → `log10(x+1)` | Reduces skewness, aids model convergence |
| **Dropped features** | `num_years_antig`, `forecast_cons_year` | Highly correlated with other features — redundant |

---

### Task 5 — Modelling & Evaluation
**Objective:** Build, evaluate and interpret a Random Forest classifier to predict churn.

#### Why Random Forest?
- Handles non-linear relationships between features
- No feature scaling required (rule-based splits)
- Built-in feature importance scores for interpretability
- Robust to outliers

#### Key Design Decision: `class_weight='balanced'`
With a ~10% churn rate, a naïve model predicts "no churn" for everything and still achieves 90% accuracy — a misleading result. Setting `class_weight='balanced'` forces the model to treat churners as equally important, dramatically improving recall.

#### Model Performance

| Metric | Score |
|---|---|
| Accuracy | *see your output* |
| Precision | *see your output* |
| Recall | *see your output* |
| F1 Score | *see your output* |
| ROC-AUC | *see your output* |
| 5-Fold CV AUC | *see your output* |

> ⚠️ Exact scores depend on your dataset version. Run `run_pipeline.py` to generate your results.

---

## 💡 Business Insight — Answering the Hypothesis

> **Verdict: Price sensitivity is NOT the primary driver of churn.**

The Random Forest feature importance analysis reveals:

1. **Net margin** and **12-month consumption** are the top churn predictors
2. **Customer tenure** is highly influential — short-tenure customers are most at risk
3. **Price-related features** (off-peak/peak differences) appear in the model but rank lower than operational and financial factors
4. **Months to contract end** and **months since last modification** are meaningful signals

**Business Recommendations:**
- 🎯 Target **high-margin, short-tenure customers** for proactive retention outreach
- 📉 Flag customers with **declining consumption** as early warning signals
- 📋 Use **churn probability scores** to prioritise retention budget allocation
- 💰 **Price discounts alone are unlikely to reduce churn** — focus on contract flexibility and service improvements

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/PowerCo-Customer-Churn-BCG.git
cd PowerCo-Customer-Churn-BCG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data files
Place the following CSVs into the `/data/` folder:
- `client_data.csv`
- `price_data.csv`
- `clean_data_after_eda.csv`

### 4. Run individual stages
```bash
python notebooks/01_eda.py               # EDA only
python notebooks/02_feature_engineering.py   # Feature engineering only
python notebooks/03_modelling.py         # Modelling only
```

### 5. Or run the full pipeline
```bash
python run_pipeline.py
```

All plots, predictions, and the executive summary will be saved to `/outputs/`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, groupby aggregations |
| `numpy` | Numerical operations, log transformations |
| `scikit-learn` | Random Forest, train/test split, cross-validation, metrics |
| `matplotlib` | All custom plots and dashboards |
| `seaborn` | Statistical visualisations (heatmaps, boxplots, histograms) |

---

## 📈 Potential Improvements

- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Compare with `XGBoost` / `LightGBM`
- [ ] Apply `SMOTE` oversampling as alternative to class weighting
- [ ] Add SHAP values for deeper, customer-level explainability
- [ ] Build a Streamlit dashboard for interactive churn scoring
- [ ] Add price elasticity calculation per customer segment

---

## 📜 Certificate

Completed as part of the **BCG X Data Science Job Simulation** on [Forage](https://www.theforage.com/).

---

## 👤 Author

Saba Nizamani
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/SabaNIzamani) · [Email](sabanizamani15@gmail.com)
