import matplotlib
matplotlib.use("Agg")

# =============================================================================
# Task 4 - Feature Engineering
# PowerCo Customer Churn - BCG X Data Science Simulation
# =============================================================================
# Framework:
#   1. Remove irrelevant/redundant columns
#   2. Create new price difference features
#   3. Convert dates to months (ML models need numbers, not dates)
#   4. Encode boolean and categorical columns
#   5. Log-transform skewed columns
#   6. Drop highly correlated columns
# =============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("whitegrid")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

df       = pd.read_csv(os.path.join(DATA, "clean_data_after_eda.csv"))
price_df = pd.read_csv(os.path.join(DATA, "price_data.csv"))

for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format="%Y-%m-%d")

print(f"Loaded: {df.shape[0]:,} customers, {price_df.shape[0]:,} price records")

# =============================================================================
# FEATURE 1 - Off-peak price difference December vs January
# Why: Captures the yearly price change a customer experiences
# =============================================================================
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
print("Feature 1 added: Dec-Jan off-peak price differences")

# =============================================================================
# FEATURE 2 - Mean price differences across periods
# Why: Captures variance between peak/off-peak/mid-peak pricing per customer
# =============================================================================
mean_p = price_df.groupby("id")[
    ["price_off_peak_var", "price_peak_var", "price_mid_peak_var",
     "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix"]
].mean().reset_index()
mean_p["off_peak_peak_var_mean_diff"]     = mean_p["price_off_peak_var"] - mean_p["price_peak_var"]
mean_p["peak_mid_peak_var_mean_diff"]     = mean_p["price_peak_var"]    - mean_p["price_mid_peak_var"]
mean_p["off_peak_mid_peak_var_mean_diff"] = mean_p["price_off_peak_var"] - mean_p["price_mid_peak_var"]
mean_p["off_peak_peak_fix_mean_diff"]     = mean_p["price_off_peak_fix"] - mean_p["price_peak_fix"]
mean_p["peak_mid_peak_fix_mean_diff"]     = mean_p["price_peak_fix"]    - mean_p["price_mid_peak_fix"]
mean_p["off_peak_mid_peak_fix_mean_diff"] = mean_p["price_off_peak_fix"] - mean_p["price_mid_peak_fix"]
diff_cols = ["id","off_peak_peak_var_mean_diff","peak_mid_peak_var_mean_diff",
             "off_peak_mid_peak_var_mean_diff","off_peak_peak_fix_mean_diff",
             "peak_mid_peak_fix_mean_diff","off_peak_mid_peak_fix_mean_diff"]
df = pd.merge(df, mean_p[diff_cols], on="id")
print("Feature 2 added: Mean price differences across periods")

# =============================================================================
# FEATURE 3 - Maximum monthly price difference
# Why: Captures the worst price spike a customer experienced in any month
#      Customers react most strongly to sudden price increases
# =============================================================================
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
print("Feature 3 added: Maximum monthly price differences")

# =============================================================================
# FEATURE 4 - Customer Tenure (years)
# Why: Short-tenure customers churn at much higher rates
# =============================================================================
df["tenure"] = (df["date_end"] - df["date_activ"]).dt.days // 365
print(f"Feature 4 added: Tenure | Churn by tenure:")
print(df.groupby("tenure")["churn"].mean().sort_values(ascending=False).head(5))

# =============================================================================
# FEATURE 5 - Date columns to months
# Why: ML models cannot use raw datetime objects - convert to numeric months
# =============================================================================
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
print("Feature 5 added: Dates converted to months")

# =============================================================================
# FEATURE 6 - Boolean encoding: has_gas t/f -> 1/0
# Why: ML models need numbers, not strings
#      Gas customers churn ~2% less (multi-product loyalty effect)
# =============================================================================
df["has_gas"] = df["has_gas"].replace({"t": 1, "f": 0})
print("Feature 6 added: has_gas encoded as binary")
print(df.groupby("has_gas")["churn"].mean())

# =============================================================================
# FEATURE 7 - One-hot encoding: channel_sales and origin_up
# Why: Categorical string columns must be converted to numeric dummy variables
#      Rare categories (<100 occurrences) are dropped to avoid noise
# =============================================================================
cols = ["channel_sales", "origin_up"]
vc   = {c: df[c].value_counts() for c in cols}
df   = pd.get_dummies(df, columns=cols, prefix=cols, dtype=int)
for c in cols:
    keep  = set(vc[c][vc[c] >= 100].index.astype(str))
    drops = [x for x in df.columns if x.startswith(f"{c}_") and
             x.split(f"{c}_", 1)[1] not in keep]
    df.drop(columns=drops, inplace=True)
print("Feature 7 added: One-hot encoding for channel_sales and origin_up")

# =============================================================================
# FEATURE 8 - Log transformation of skewed columns
# Why: Many ML algorithms assume near-normal distributions
#      log10(x+1) reduces skewness (+1 avoids log(0) errors)
# =============================================================================
skewed = ["cons_12m","cons_gas_12m","cons_last_month","forecast_cons_12m",
          "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
          "forecast_price_energy_off_peak","forecast_price_energy_peak",
          "forecast_price_pow_off_peak"]
existing_skewed = [c for c in skewed if c in df.columns]
df[existing_skewed] = np.log10(df[existing_skewed] + 1)
print(f"Feature 8 added: Log10 transform applied to {len(existing_skewed)} skewed columns")

# Plot log-transformed distributions
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axs, ["cons_12m", "cons_last_month", "forecast_price_energy_peak"]):
    if col in df.columns:
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#3498db")
        ax.set_title(f"{col}\n(after log transform)", fontweight="bold")
plt.suptitle("Log-Transformed Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_01_log_transforms.png"), bbox_inches="tight")
plt.close()
print("Saved: fe_01_log_transforms.png")

# =============================================================================
# STEP 9 - Drop highly correlated columns
# Why: Highly correlated columns hold duplicate information
#      Removing them reduces noise and speeds up training
# =============================================================================
df.drop(columns=["num_years_antig","forecast_cons_year"], inplace=True, errors="ignore")
print("Dropped: num_years_antig, forecast_cons_year (high correlation with other features)")

# Save final dataset
out_path = os.path.join(DATA, "data_for_predictions.csv")
df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Final dataset shape : {df.shape}")
print(f"Saved to            : {out_path}")
print("Next step: Run 03_modelling.py")
