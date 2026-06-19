"""
generate_dataset.py
--------------------
Generates novapay_transactions.csv — a SYNTHETIC dataset that follows the
exact schema, column names, and statistical patterns described in the
NovaPay assessment brief (8,000 rows, 14 columns, ~6% fraud rate).

WHY THIS FILE EXISTS:
The original assessment PDF references a CSV that was supposed to be
"provided separately" but was not actually attached. Rather than block
the whole project, this script builds a realistic stand-in with the same
fraud signal structure (high amount + overseas country + high velocity +
prior flags + odd hours => higher fraud probability), so every later step
(EDA, modeling, dollar-impact calc) runs exactly the way it would against
the real file.

If you are given the real novapay_transactions.csv later, just drop it
into the data/ folder with this exact name and skip this script — every
other script in the project reads data/novapay_transactions.csv and
doesn't care how it got there.
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

RNG_SEED = 42
N_ROWS = 8000
OUT_PATH = "data/novapay_transactions.csv"

np.random.seed(RNG_SEED)

merchant_cats = ["Grocery", "Electronics", "Travel", "Restaurant", "Other"]
merchant_probs = [0.30, 0.18, 0.12, 0.25, 0.15]

countries = ["US", "UK", "IN", "RU", "BR", "NG", "CN", "Other"]
country_probs = [0.55, 0.10, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05]
# Countries with structurally higher fraud base-rate (overseas / higher-risk corridors)
high_risk_countries = {"RU", "NG", "BR", "CN"}

devices = ["Mobile", "Web", "ATM"]
device_probs = [0.55, 0.35, 0.10]


def generate():
    n = N_ROWS
    txn_id = [f"T{str(i+1).zfill(4)}" for i in range(n)]

    merchant_cat = np.random.choice(merchant_cats, size=n, p=merchant_probs)
    country = np.random.choice(countries, size=n, p=country_probs)
    user_age = np.clip(np.random.normal(38, 12, n), 18, 80).astype(int)
    acct_age_days = np.clip(np.random.exponential(400, n), 1, 3000).astype(int)
    txn_hour = np.random.randint(0, 24, n)
    device = np.random.choice(devices, size=n, p=device_probs)
    is_weekend = np.random.choice(["Yes", "No"], size=n, p=[0.28, 0.72])

    # Base (legit-like) transaction amounts: lognormal, mostly small
    amount_usd = np.round(np.random.lognormal(mean=3.6, sigma=1.0, size=n), 2)
    amount_usd = np.clip(amount_usd, 5, 15000)

    avg_txn_usd = np.round(amount_usd * np.random.uniform(0.6, 1.3, n) + np.random.normal(0, 15, n), 2)
    avg_txn_usd = np.clip(avg_txn_usd, 5, None)

    distance_km = np.round(np.random.exponential(50, n), 1)
    velocity_24h = np.random.poisson(2, n)
    prior_fraud_flags = np.random.choice([0, 1, 2, 3, 4, 5], size=n,
                                          p=[0.85, 0.08, 0.04, 0.015, 0.01, 0.005])

    # ---- Build a latent fraud "risk score" from realistic signal combos ----
    risk = np.zeros(n)
    risk += (amount_usd > 2000) * 1.8
    risk += (amount_usd > 5000) * 1.2
    risk += np.isin(country, list(high_risk_countries)) * 1.5
    risk += (prior_fraud_flags >= 1) * 1.6
    risk += (velocity_24h >= 6) * 1.3
    risk += (distance_km > 3000) * 1.4
    risk += (txn_hour.astype(int) <= 4) * 0.6
    risk += (device == "Web") * 0.2
    risk += (acct_age_days < 60) * 0.9
    risk += (amount_usd > avg_txn_usd * 3) * 1.1
    risk += np.random.normal(0, 0.8, n)  # noise

    # Convert risk to probability via logistic squashing, then calibrate
    # so the overall fraud rate lands close to 6%.
    prob = 1 / (1 + np.exp(-(risk - 6.2)))
    is_fraud = (np.random.uniform(0, 1, n) < prob).astype(int)

    # Calibration pass: nudge toward ~6% fraud rate if the random draw drifted
    target_fraud = int(round(n * 0.06))
    current_fraud = is_fraud.sum()
    if current_fraud > target_fraud:
        # Flip some lowest-risk frauds back to legit
        fraud_idx = np.where(is_fraud == 1)[0]
        drop_n = current_fraud - target_fraud
        drop_idx = fraud_idx[np.argsort(risk[fraud_idx])][:drop_n]
        is_fraud[drop_idx] = 0
    elif current_fraud < target_fraud:
        legit_idx = np.where(is_fraud == 0)[0]
        add_n = target_fraud - current_fraud
        add_idx = legit_idx[np.argsort(-risk[legit_idx])][:add_n]
        is_fraud[add_idx] = 1

    df = pd.DataFrame({
        "txn_id": txn_id,
        "amount_usd": amount_usd,
        "merchant_cat": merchant_cat,
        "country": country,
        "user_age": user_age,
        "acct_age_days": acct_age_days,
        "txn_hour": txn_hour,
        "device": device,
        "prior_fraud_flags": prior_fraud_flags,
        "velocity_24h": velocity_24h,
        "avg_txn_usd": avg_txn_usd,
        "distance_km": distance_km,
        "is_weekend": is_weekend,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    fraud_rate = df["is_fraud"].mean() * 100
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Fraud rate: {fraud_rate:.2f}%  ({df['is_fraud'].sum()} fraud / {len(df)} total)")
