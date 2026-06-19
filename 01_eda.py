"""
01_eda.py
---------
Task 1 — Business framing (printed to console + saved to outputs/task1_business_framing.txt)
Task 2 — EDA & class imbalance analysis (charts saved to outputs/figures/)

Run from the project root:  python 01_eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "data/novapay_transactions.csv"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

# =====================================================================
# TASK 1 — Business Problem Framing
# =====================================================================
task1_text = """
TASK 1 — BUSINESS PROBLEM FRAMING
==================================

False Negative (missed fraud, model says legit but it was fraud):
  Cost = $340 per case (chargeback + investigation + customer compensation).
  This is a DIRECT, GUARANTEED cash loss to NovaPay every single time it happens.

False Positive (wrongly blocked legitimate transaction):
  Cost = $25 per case (customer service handling + some lost revenue/goodwill).
  This is a smaller, mostly recoverable cost (the customer can retry, call
  support, or the txn can be manually approved).

Which metric should we optimise for: PRECISION or RECALL?
-----------------------------------------------------------
RECALL. The cost asymmetry is roughly 13.6x ($340 vs $25) in favor of
catching fraud over avoiding false alarms. A model that misses fraud is far
more expensive than one that occasionally over-flags a legitimate purchase.
Given NovaPay lost $2.1M last quarter to undetected fraud, the priority is
to minimize False Negatives even if that means accepting somewhat more
False Positives along the way -- as long as the threshold is tuned so the
combined dollar cost (FN x $340 + FP x $25) keeps shrinking, not just the
recall number in isolation. So: optimize Recall, but validate every
threshold choice against Total Dollar Cost (Task 5), since recall pushed
too far (e.g. flagging everything) would blow up FP costs and customer
trust without limit.
"""
print(task1_text)
with open("outputs/task1_business_framing.txt", "w") as f:
    f.write(task1_text)

# =====================================================================
# TASK 2 — EDA & Imbalance Analysis
# =====================================================================

# --- Class distribution ---
class_counts = df["is_fraud"].value_counts().sort_index()
fraud_rate = df["is_fraud"].mean() * 100
print(f"\nClass distribution:\n{class_counts}")
print(f"Fraud rate: {fraud_rate:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.countplot(x="is_fraud", data=df, hue="is_fraud", palette=["#4C72B0", "#C44E52"],
              legend=False, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Legit (0)", "Fraud (1)"])
axes[0].set_title(f"Class Distribution (Fraud rate = {fraud_rate:.1f}%)")
axes[0].set_ylabel("Transaction count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                      ha="center", va="bottom", fontsize=10)

axes[1].pie(class_counts, labels=["Legit", "Fraud"], autopct="%1.1f%%",
            colors=["#4C72B0", "#C44E52"], startangle=90)
axes[1].set_title("Class Balance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png")
plt.close()

# --- Fraud rate by country ---
country_fraud = df.groupby("country")["is_fraud"].agg(["mean", "count"]).sort_values("mean", ascending=False)
country_fraud["mean_pct"] = country_fraud["mean"] * 100

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=country_fraud.index, y=country_fraud["mean_pct"], hue=country_fraud.index,
                  palette="Reds_r", legend=False)
plt.title("Fraud Rate by Country")
plt.ylabel("Fraud rate (%)")
plt.xlabel("Country")
for i, v in enumerate(country_fraud["mean_pct"]):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_fraud_by_country.png")
plt.close()

# --- Fraud rate by transaction hour ---
hour_fraud = df.groupby("txn_hour")["is_fraud"].mean() * 100
plt.figure(figsize=(9, 4.5))
ax = sns.lineplot(x=hour_fraud.index, y=hour_fraud.values, marker="o", color="#C44E52")
plt.title("Fraud Rate by Hour of Day")
plt.xlabel("Transaction Hour (0-23)")
plt.ylabel("Fraud rate (%)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_fraud_by_hour.png")
plt.close()

# --- Fraud rate by amount bracket ---
bins = [0, 50, 200, 500, 1000, 2000, 5000, 100000]
labels = ["<$50", "$50-200", "$200-500", "$500-1k", "$1k-2k", "$2k-5k", "$5k+"]
df["amount_bracket"] = pd.cut(df["amount_usd"], bins=bins, labels=labels)
bracket_fraud = df.groupby("amount_bracket", observed=True)["is_fraud"].mean() * 100

plt.figure(figsize=(9, 4.5))
ax = sns.barplot(x=bracket_fraud.index, y=bracket_fraud.values, hue=bracket_fraud.index,
                  palette="Oranges", legend=False)
plt.title("Fraud Rate by Transaction Amount Bracket")
plt.xlabel("Amount bracket")
plt.ylabel("Fraud rate (%)")
for i, v in enumerate(bracket_fraud.values):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_fraud_by_amount_bracket.png")
plt.close()

# --- Fraud rate by prior_fraud_flags and velocity_24h (bonus signal charts) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
pf = df.groupby("prior_fraud_flags")["is_fraud"].mean() * 100
sns.barplot(x=pf.index, y=pf.values, hue=pf.index, palette="Purples", legend=False, ax=axes[0])
axes[0].set_title("Fraud Rate by Prior Fraud Flags")
axes[0].set_xlabel("prior_fraud_flags")
axes[0].set_ylabel("Fraud rate (%)")

vel_bins = [-1, 1, 3, 5, 8, 100]
vel_labels = ["0-1", "2-3", "4-5", "6-8", "9+"]
df["velocity_bracket"] = pd.cut(df["velocity_24h"], bins=vel_bins, labels=vel_labels)
vf = df.groupby("velocity_bracket", observed=True)["is_fraud"].mean() * 100
sns.barplot(x=vf.index, y=vf.values, hue=vf.index, palette="Greens", legend=False, ax=axes[1])
axes[1].set_title("Fraud Rate by 24h Velocity Bracket")
axes[1].set_xlabel("velocity_24h bracket")
axes[1].set_ylabel("Fraud rate (%)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_fraud_by_priorflags_velocity.png")
plt.close()

# --- Correlation heatmap of numeric features ---
numeric_cols = ["amount_usd", "user_age", "acct_age_days", "txn_hour",
                 "prior_fraud_flags", "velocity_24h", "avg_txn_usd",
                 "distance_km", "is_fraud"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_correlation_heatmap.png")
plt.close()

print(f"\nSaved 6 charts to {FIG_DIR}/")
print("\nTop fraud patterns observed:")
print(f"  - Highest-risk country: {country_fraud['mean_pct'].idxmax()} ({country_fraud['mean_pct'].max():.1f}% fraud rate)")
print(f"  - Highest-risk amount bracket: {bracket_fraud.idxmax()} ({bracket_fraud.max():.1f}% fraud rate)")
print(f"  - Accounts with prior_fraud_flags>=1 have fraud rate: {df[df.prior_fraud_flags>=1]['is_fraud'].mean()*100:.1f}% vs {df[df.prior_fraud_flags==0]['is_fraud'].mean()*100:.1f}% for clean accounts")
