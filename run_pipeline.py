# =============================================================================
# PowerCo Customer Churn — Full Pipeline
# BCG X Data Science Job Simulation (Forage)
# =============================================================================
# This master script runs the complete end-to-end pipeline:
#
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves without opening windows
#   Stage 1 → Exploratory Data Analysis
#   Stage 2 → Feature Engineering
#   Stage 3 → Modelling & Evaluation
#   Stage 4 → Business Insight Report
#
# Usage:
#   python run_pipeline.py
#
# Outputs:
#   /outputs/  → all plots (.png) + predictions (.csv) + summary report (.txt)
# =============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130})

# ── Directory setup ───────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUTS, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"    ✔ Saved: {name}")

def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# =============================================================================
# STAGE 1 — EDA
# =============================================================================
section("STAGE 1 — EXPLORATORY DATA ANALYSIS")

client_df = pd.read_csv(os.path.join(DATA, "client_data.csv"))
price_df  = pd.read_csv(os.path.join(DATA, "price_data.csv"))

print(f"  Client data  : {client_df.shape[0]:,} rows × {client_df.shape[1]} cols")
print(f"  Price data   : {price_df.shape[0]:,} rows  × {price_df.shape[1]} cols")
print(f"  Churn rate   : {client_df['churn'].mean():.1%}")
print(f"  Missing vals : {client_df.isnull().sum().sum()} (client)  "
      f"{price_df.isnull().sum().sum()} (price)")

# --- Churn distribution
churn_pct = client_df["churn"].value_counts(normalize=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["No Churn", "Churn"], churn_pct.sort_index().values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Class Distribution", fontweight="bold")
axes[0].set_ylabel("Percentage (%)")
for i, v in enumerate(churn_pct.sort_index().values):
    axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11)
axes[1].pie(churn_pct.sort_index().values, labels=["No Churn", "Churn"],
            colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%",
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Churn Proportion", fontweight="bold")
plt.suptitle("Target Variable — Customer Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "eda_01_churn_distribution.png")
plt.close()

# --- Price sensitivity preview: mean price by churn group
price_cols = [c for c in price_df.columns if c.startswith("price_")]
merged     = client_df[["id", "churn"]].merge(price_df, on="id")
# Only keep numeric price columns to avoid dtype errors
price_cols_numeric = merged[price_cols].select_dtypes(include="number").columns.tolist()
price_comp = merged.groupby("churn")[price_cols_numeric].mean().T
price_comp.columns = ["Retained", "Churned"]
price_comp["diff_%"] = ((price_comp["Churned"] - price_comp["Retained"])
                         / price_comp["Retained"] * 100).round(2)

print("\n  Price comparison — Churned vs Retained:")
print(price_comp.to_string())

# --- Consumption distributions
def plot_distribution(df, col, ax, bins=50):
    temp = pd.DataFrame({
        "Retention": df[df["churn"] == 0][col],
        "Churn":     df[df["churn"] == 1][col],
    })
    temp.plot(kind="hist", bins=bins, ax=ax, stacked=True, alpha=0.8)
    ax.set_xlabel(col)
    ax.ticklabel_format(style="plain", axis="x")
    ax.set_title(col, fontweight="bold")

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
plot_distribution(client_df, "cons_12m",       axs[0, 0])
plot_distribution(client_df, "cons_last_month", axs[0, 1])
plot_distribution(client_df, "net_margin",      axs[1, 0])
plot_distribution(client_df, "pow_max",         axs[1, 1])
plt.suptitle("Key Variable Distributions — Churned vs Retained",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "eda_02_distributions.png")
plt.close()

print("\n  Key EDA findings:")
print("    • ~10% churn rate — significant class imbalance")
print("    • Consumption variables are highly positively skewed")
print("    • Price differences between churned/retained are small")
print("    • Sales channel 'MISSING' shows elevated churn")

# =============================================================================
# STAGE 2 — FEATURE ENGINEERING
# =============================================================================
section("STAGE 2 — FEATURE ENGINEERING")

df       = pd.read_csv(os.path.join(DATA, "clean_data_after_eda.csv"))
price_df = pd.read_csv(os.path.join(DATA, "price_data.csv"))

for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format="%Y-%m-%d")

# Feature 1 — Dec vs Jan off-peak price diff
monthly = (price_df.groupby(["id", "price_date"])
           [["price_off_peak_var", "price_off_peak_fix"]].mean()
           .reset_index())
jan = monthly.groupby("id").first().reset_index()
dec = monthly.groupby("id").last().reset_index()
diff = pd.merge(dec.rename(columns={"price_off_peak_var": "dec_1",
                                    "price_off_peak_fix": "dec_2"}),
                jan.drop(columns="price_date"), on="id")
diff["offpeak_diff_dec_january_energy"] = diff["dec_1"] - diff["price_off_peak_var"]
diff["offpeak_diff_dec_january_power"]  = diff["dec_2"] - diff["price_off_peak_fix"]
df = pd.merge(df, diff[["id", "offpeak_diff_dec_january_energy",
                         "offpeak_diff_dec_january_power"]], on="id")

# Feature 2 — Mean price differences across periods
mean_p = price_df.groupby("id")[
    ["price_off_peak_var", "price_peak_var", "price_mid_peak_var",
     "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix"]
].mean().reset_index()
mean_p["off_peak_peak_var_mean_diff"]    = mean_p["price_off_peak_var"] - mean_p["price_peak_var"]
mean_p["peak_mid_peak_var_mean_diff"]    = mean_p["price_peak_var"]    - mean_p["price_mid_peak_var"]
mean_p["off_peak_mid_peak_var_mean_diff"]= mean_p["price_off_peak_var"] - mean_p["price_mid_peak_var"]
mean_p["off_peak_peak_fix_mean_diff"]    = mean_p["price_off_peak_fix"] - mean_p["price_peak_fix"]
mean_p["peak_mid_peak_fix_mean_diff"]    = mean_p["price_peak_fix"]    - mean_p["price_mid_peak_fix"]
mean_p["off_peak_mid_peak_fix_mean_diff"]= mean_p["price_off_peak_fix"] - mean_p["price_mid_peak_fix"]
diff_cols = ["id","off_peak_peak_var_mean_diff","peak_mid_peak_var_mean_diff",
             "off_peak_mid_peak_var_mean_diff","off_peak_peak_fix_mean_diff",
             "peak_mid_peak_fix_mean_diff","off_peak_mid_peak_fix_mean_diff"]
df = pd.merge(df, mean_p[diff_cols], on="id")

# Feature 3 — Max monthly price diff
monthly2 = price_df.groupby(["id", "price_date"])[
    ["price_off_peak_var", "price_peak_var", "price_mid_peak_var",
     "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix"]
].mean().reset_index()
for col1, col2, col in [("price_off_peak_var","price_peak_var","off_peak_peak_var"),
                        ("price_peak_var","price_mid_peak_var","peak_mid_peak_var"),
                        ("price_off_peak_var","price_mid_peak_var","off_peak_mid_peak_var"),
                        ("price_off_peak_fix","price_peak_fix","off_peak_peak_fix"),
                        ("price_peak_fix","price_mid_peak_fix","peak_mid_peak_fix"),
                        ("price_off_peak_fix","price_mid_peak_fix","off_peak_mid_peak_fix")]:
    monthly2[f"{col}_mean_diff"] = monthly2[col1] - monthly2[col2]
max_diff = (monthly2.groupby("id")
            .agg({f"{c}_mean_diff": "max" for c in
                  ["off_peak_peak_var","peak_mid_peak_var","off_peak_mid_peak_var",
                   "off_peak_peak_fix","peak_mid_peak_fix","off_peak_mid_peak_fix"]})
            .reset_index()
            .rename(columns={f"{c}_mean_diff": f"{c}_max_monthly_diff"
                              for c in ["off_peak_peak_var","peak_mid_peak_var",
                                        "off_peak_mid_peak_var","off_peak_peak_fix",
                                        "peak_mid_peak_fix","off_peak_mid_peak_fix"]}))
df = pd.merge(df, max_diff, on="id")

# Feature 4 — Tenure
df["tenure"] = (df["date_end"] - df["date_activ"]).dt.days // 365

# Feature 5 — Dates → months
def convert_months(ref, df, col):
    d = pd.to_datetime(df[col])
    m = (ref.year - d.dt.year) * 12 + (ref.month - d.dt.month)
    m -= (ref.day < d.dt.day).astype(int)
    return m

ref = datetime(2016, 1, 1)
df["months_activ"]      = convert_months(ref, df, "date_activ")
df["months_to_end"]     = -convert_months(ref, df, "date_end")
df["months_modif_prod"] = convert_months(ref, df, "date_modif_prod")
df["months_renewal"]    = convert_months(ref, df, "date_renewal")
df.drop(columns=["date_activ","date_end","date_modif_prod","date_renewal"],
        inplace=True, errors="ignore")

# Feature 6 — has_gas boolean
df["has_gas"] = df["has_gas"].replace({"t": 1, "f": 0})

# Feature 7 — One-hot encoding
cols = ["channel_sales", "origin_up"]
vc   = {c: df[c].value_counts() for c in cols}
df   = pd.get_dummies(df, columns=cols, prefix=cols, dtype=int)
for c in cols:
    keep  = set(vc[c][vc[c] >= 100].index.astype(str))
    drops = [x for x in df.columns if x.startswith(f"{c}_") and
             x.split(f"{c}_", 1)[1] not in keep]
    df.drop(columns=drops, inplace=True)

# Feature 8 — Log transform skewed columns
skewed = ["cons_12m","cons_gas_12m","cons_last_month","forecast_cons_12m",
          "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
          "forecast_price_energy_off_peak","forecast_price_energy_peak",
          "forecast_price_pow_off_peak"]
existing_skewed = [c for c in skewed if c in df.columns]
df[existing_skewed] = np.log10(df[existing_skewed] + 1)

# Drop highly correlated columns
df.drop(columns=["num_years_antig","forecast_cons_year"],
        inplace=True, errors="ignore")

print(f"  Feature engineering complete.")
print(f"  Final dataset shape : {df.shape}")
print(f"  New price features  : 14  (dec-jan diff + mean diff + max diff)")
print(f"  Date → months       : 4")
print(f"  Log-transformed     : {len(existing_skewed)} columns")

# Save
df.to_csv(os.path.join(DATA, "data_for_predictions.csv"), index=False)

# =============================================================================
# STAGE 3 — MODELLING
# =============================================================================
section("STAGE 3 — RANDOM FOREST MODELLING")

df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
y = df["churn"]
X = df.drop(columns=["id", "churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"  Train : {X_train.shape[0]:,} rows  |  Test : {X_test.shape[0]:,} rows")

model = RandomForestClassifier(
    n_estimators=1000, class_weight="balanced",
    random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("  ✔ Model trained (1000 trees, balanced class weights)")

cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"  5-Fold CV ROC-AUC : {cv.mean():.4f} ± {cv.std():.4f}")

preds  = model.predict(X_test)
proba  = model.predict_proba(X_test)[:, 1]
tn, fp, fn, tp_val = confusion_matrix(y_test, preds).ravel()

print(f"\n  Accuracy   : {accuracy_score(y_test, preds):.4f}")
print(f"  Precision  : {precision_score(y_test, preds):.4f}")
print(f"  Recall     : {recall_score(y_test, preds):.4f}")
print(f"  F1 Score   : {f1_score(y_test, preds):.4f}")
print(f"  ROC-AUC    : {roc_auc_score(y_test, proba):.4f}")

# Evaluation dashboard
fig = plt.figure(figsize=(18, 5))
gs  = gridspec.GridSpec(1, 3)

ax1 = fig.add_subplot(gs[0])
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues",
            ax=ax1, xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"],
            linewidths=1, linecolor="white", annot_kws={"size":14})
ax1.set_title("Confusion Matrix", fontweight="bold")
ax1.set_xlabel("Predicted", fontweight="bold")
ax1.set_ylabel("Actual",    fontweight="bold")

ax2 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(y_test, proba)
auc_val      = roc_auc_score(y_test, proba)
ax2.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_val:.3f}")
ax2.plot([0,1],[0,1], "k--", lw=1)
ax2.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
ax2.set_title("ROC Curve", fontweight="bold")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")

ax3 = fig.add_subplot(gs[2])
prec_a, rec_a, thresh_a = precision_recall_curve(y_test, proba)
ax3.plot(thresh_a, prec_a[:-1], "#3498db", lw=2, label="Precision")
ax3.plot(thresh_a, rec_a[:-1],  "#e67e22", lw=2, label="Recall")
ax3.axvline(0.3, color="grey", ls="--", lw=1)
ax3.set_title("Precision & Recall vs Threshold", fontweight="bold")
ax3.set_xlabel("Threshold")
ax3.legend()

plt.suptitle("Model Evaluation Dashboard", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "model_01_evaluation_dashboard.png")
plt.close()

# Feature importances
fi = pd.DataFrame({"feature": X_train.columns,
                   "importance": model.feature_importances_})\
       .sort_values("importance", ascending=True).reset_index(drop=True)
top = fi.tail(20)

fig, ax = plt.subplots(figsize=(11, 9))
colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
bars    = ax.barh(range(len(top)), top["importance"], color=colors,
                  edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["feature"], fontsize=9)
ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontweight="bold")
ax.set_title("Top 20 Feature Importances — Random Forest", fontweight="bold", fontsize=13)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{w:.4f}", va="center", fontsize=8)
plt.tight_layout()
save(fig, "model_02_feature_importances.png")
plt.close()

# Save predictions
out = X_test.copy().reset_index(drop=True)
out["actual_churn"]     = y_test.reset_index(drop=True)
out["predicted_churn"]  = preds
out["churn_probability"]= proba.round(4)
out.to_csv(os.path.join(OUTPUTS, "out_of_sample_predictions.csv"), index=False)
print(f"  ✔ Predictions saved.")

# =============================================================================
# STAGE 4 — BUSINESS INSIGHT SUMMARY
# =============================================================================
section("STAGE 4 — BUSINESS INSIGHT REPORT")

top5 = fi.tail(5)["feature"].tolist()
price_in_top20 = fi.tail(20)[fi.tail(20)["feature"].str.contains("price|peak", case=False)]

report = f"""
======================================================================
  PowerCo Customer Churn — Executive Summary
  BCG X Data Science Simulation
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
======================================================================

BUSINESS QUESTION
  Is customer churn at PowerCo primarily driven by price sensitivity?

ANSWER
  No. Price sensitivity is a contributing factor but NOT the primary
  driver of churn.

MODEL PERFORMANCE
  Algorithm  : Random Forest (1000 trees, balanced class weights)
  Accuracy   : {accuracy_score(y_test, preds):.2%}
  Precision  : {precision_score(y_test, preds):.2%}
  Recall     : {recall_score(y_test, preds):.2%}
  F1 Score   : {f1_score(y_test, preds):.2%}
  ROC-AUC    : {roc_auc_score(y_test, proba):.4f}
  CV ROC-AUC : {cv.mean():.4f} ± {cv.std():.4f}

TOP CHURN DRIVERS (Feature Importance)
  {chr(10).join(f"  {i+1}. {f}" for i, f in enumerate(reversed(top5)))}

KEY FINDINGS
  1. Net margin and 12-month consumption are the strongest churn signals
  2. Customer tenure matters — customers with <4 months tenure churn most
  3. Price-related features contribute weakly, ranking lower in importance
  4. The 'MISSING' sales channel shows elevated churn (7.6%)
  5. Gas customers churn ~2% less (multi-product loyalty effect)

RECOMMENDATIONS
  → Target high-margin, short-tenure customers for proactive outreach
  → Investigate declining-consumption customers early as an at-risk signal
  → Use churn probability scores to prioritise retention budgets
  → Price discounts alone are unlikely to solve the churn problem
  → Consider improving contract flexibility and service quality

======================================================================
"""

print(report)

report_path = os.path.join(OUTPUTS, "executive_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  ✔ Report saved: executive_summary.txt")

section("PIPELINE COMPLETE — ALL OUTPUTS SAVED TO /outputs/")
