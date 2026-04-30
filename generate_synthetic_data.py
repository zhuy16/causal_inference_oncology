"""
Generate a synthetic TCGA-like dataset and save it to
data/processed/analysis_dataset.parquet.

Usage (offline / no real data available):
    python generate_synthetic_data.py

This is a FALLBACK option. If real TCGA clinical files are available,
use fetch_lfs_clinical.py + build_real_dataset.py instead to get real data.

The synthetic dataset (n=8,000) has:
- Realistic age/stage/cancer-type distributions
- Indication bias: stage IV patients have higher chemo probability
- Simulated TMB (higher in chemo-treated patients)
- Survival drawn from an exponential model with chemo benefit
Seed is fixed (42) for full reproducibility.
"""
import numpy as np
import pandas as pd
import os

PROC_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

rng = np.random.default_rng(42)
n = 8000

cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'SKCM',
                'BLCA', 'KIRC', 'HNSC', 'LIHC', 'STAD']
ct    = rng.choice(cancer_types, size=n, p=[.18,.12,.11,.10,.09,.09,.08,.08,.08,.07])
age   = rng.normal(60, 12, n).clip(25, 90)
stage = rng.choice([1, 2, 3, 4], size=n, p=[.20, .30, .30, .20]).astype(float)

logit   = -1.5 + 0.55 * stage - 0.015 * (age - 60)
p_chemo = 1 / (1 + np.exp(-logit))
chemo   = (rng.uniform(size=n) < p_chemo).astype(float)

tmb = rng.gamma(shape=2.5 + 0.8 * chemo, scale=1.2).clip(0.1)

base_surv = (36 - 8 * (stage - 1) + 5 * chemo + 3 * (tmb > 5).astype(float)).clip(5)
os_months = rng.exponential(scale=base_surv).clip(0.5, 200)
os_event  = (rng.uniform(size=n) < 0.55).astype(int)

df = pd.DataFrame({
    'AGE':              age,
    'STAGE':            stage,
    'CANCER_TYPE_ABBR': ct,
    'CHEMO':            chemo,
    'TMB':              tmb,
    'OS_MONTHS':        os_months,
    'OS_EVENT':         os_event,
})

out = os.path.join(PROC_DIR, 'analysis_dataset.parquet')
df.to_parquet(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(df.dtypes)
print(df.head(3))
