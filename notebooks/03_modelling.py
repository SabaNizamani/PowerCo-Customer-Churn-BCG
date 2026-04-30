import matplotlib
matplotlib.use("Agg")

# =============================================================================
# Task 5 - Modelling & Evaluation
# PowerCo Customer Churn - BCG X Data Science Simulation
# =============================================================================
# Goal: Train a Random Forest to predict churn, evaluate it properly,
#       and answer the original business question about price sensitivity.
# =============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130})

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "data_for_predictions.csv"))
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

print("=" * 60)
print("DATASET LOADED FOR MODELLING")
print("=" * 60)
print(f"Shape      : {df.shape}")
print(f"Churn rate : {df['churn'].mean():.1%}  (class imbalance present)")

# ── Train / Test Split ────────────────────────────────────────────────────────
# stratify=y ensures both train and test keep the same churn ratio
y = df["churn"]
X = df.drop(columns=["id", "churn"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain : {X_train.shape[0]:,} rows | churn rate: {y_train.mean():.1%}")
print(f"Test  : {X_test.shape[0]:,} rows  | churn rate: {y_test.mean():.1%}")

# ── Model Training ────────────────────────────────────────────────────────────
# WHY Random Forest?
#   - Handles non-linear relationships between features
#   - No feature scaling required (rule-based splits)
#   - Built-in feature importance for interpretability
#   - Robust to outliers
#
# WHY class_weight='balanced'?
#   - Without this: model predicts "no churn" for everything
#   - Still gets 90% accuracy but catches almost NO churners (recall ~5%)
#   - 'balanced' forces the model to treat churners as equally important
#   - Result: dramatically improved recall on the minority class

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)
print("  1000 trees | class_weight=balanced | random_state=42")

model = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained successfully.")

# ── Cross Validation ──────────────────────────────────────────────────────────
# Tests model on 5 different splits to ensure results are not just lucky
print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="roc_auc", n_jobs=-1)
print(f"  ROC-AUC per fold : {cv_scores.round(4)}")
print(f"  Mean +/- Std     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ── Predictions ───────────────────────────────────────────────────────────────
predictions   = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"  True  Positives (churners correctly caught)  : {tp}")
print(f"  False Positives (non-churners wrongly flagged): {fp}")
print(f"  True  Negatives (non-churners correct)        : {tn}")
print(f"  False Negatives (churners missed)             : {fn}")
print()
print(f"  Accuracy   : {accuracy_score(y_test, predictions):.4f}")
print(f"  Precision  : {precision_score(y_test, predictions):.4f}")
print(f"  Recall     : {recall_score(y_test, predictions):.4f}")
print(f"  F1 Score   : {f1_score(y_test, predictions):.4f}")
print(f"  ROC-AUC    : {roc_auc_score(y_test, probabilities):.4f}")
print()
print(classification_report(y_test, predictions,
                            target_names=["No Churn", "Churn"]))

# ── Evaluation Plots ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
gs  = gridspec.GridSpec(1, 3)

# Confusion Matrix
ax1 = fig.add_subplot(gs[0])
cm  = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=1, linecolor="white", annot_kws={"size": 14})
ax1.set_xlabel("Predicted", fontweight="bold")
ax1.set_ylabel("Actual",    fontweight="bold")
ax1.set_title("Confusion Matrix", fontweight="bold")

# ROC Curve
ax2 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(y_test, probabilities)
auc_val      = roc_auc_score(y_test, probabilities)
ax2.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_val:.3f}")
ax2.plot([0,1],[0,1], "k--", lw=1, label="Random guess")
ax2.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
ax2.set_title("ROC Curve", fontweight="bold")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")

# Precision-Recall vs Threshold
ax3 = fig.add_subplot(gs[2])
prec_a, rec_a, thresh_a = precision_recall_curve(y_test, probabilities)
ax3.plot(thresh_a, prec_a[:-1], "#3498db", lw=2, label="Precision")
ax3.plot(thresh_a, rec_a[:-1],  "#e67e22", lw=2, label="Recall")
ax3.axvline(0.3, color="grey", ls="--", lw=1, label="Threshold=0.3")
ax3.set_title("Precision & Recall vs Threshold", fontweight="bold")
ax3.set_xlabel("Decision Threshold")
ax3.legend()

plt.suptitle("Model Evaluation Dashboard", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_01_evaluation_dashboard.png"),
            bbox_inches="tight")
plt.close()
print("Saved: model_01_evaluation_dashboard.png")

# ── Threshold Optimisation ────────────────────────────────────────────────────
# Default threshold is 0.5 but this is not always optimal for imbalanced data
# Lowering threshold catches more churners (higher recall) but more false alarms
print("\n" + "=" * 60)
print("THRESHOLD OPTIMISATION")
print("=" * 60)
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 42)
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
    p_t = (probabilities >= thresh).astype(int)
    p   = precision_score(y_test, p_t, zero_division=0)
    r   = recall_score(y_test, p_t, zero_division=0)
    f   = f1_score(y_test, p_t, zero_division=0)
    print(f"{thresh:>10.1f} {p:>10.4f} {r:>8.4f} {f:>8.4f}")

# ── Feature Importances ───────────────────────────────────────────────────────
fi = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=True).reset_index(drop=True)

top = fi.tail(20)

fig, ax = plt.subplots(figsize=(11, 9))
colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
bars    = ax.barh(range(len(top)), top["importance"],
                  color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["feature"], fontsize=9)
ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontweight="bold")
ax.set_title("Top 20 Feature Importances - Random Forest",
             fontweight="bold", fontsize=13)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{w:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_02_feature_importances.png"),
            bbox_inches="tight")
plt.close()
print("Saved: model_02_feature_importances.png")

# ── Answer the Business Question ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUSINESS HYPOTHESIS VERDICT")
print("=" * 60)
top5 = fi.tail(5)["feature"].tolist()
print("\nTop 5 most important features:")
print(fi.tail(5)[["feature","importance"]].to_string(index=False))
print("""
VERDICT:
  Price sensitivity is NOT the primary driver of churn at PowerCo.
  The strongest predictors are net margin, 12-month consumption,
  and customer tenure - not price variables.
  Price features appear in the model but rank lower than
  operational and financial features.

RECOMMENDATION:
  Focus retention on high-margin customers with declining
  consumption and short or expiring tenures.
  Price discounts alone are unlikely to retain customers.
""")

# ── Save Predictions ──────────────────────────────────────────────────────────
out = X_test.copy().reset_index(drop=True)
out["actual_churn"]      = y_test.reset_index(drop=True)
out["predicted_churn"]   = predictions
out["churn_probability"] = probabilities.round(4)
out.to_csv(os.path.join(OUTPUTS, "out_of_sample_predictions.csv"), index=False)
print("Saved: out_of_sample_predictions.csv")

print("\n" + "=" * 60)
print("MODELLING COMPLETE - all outputs saved to /outputs/")
print("=" * 60)
