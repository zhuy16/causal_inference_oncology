"""
Build analysis_dataset.parquet from the real TCGA Pan-Cancer Atlas 2018
clinical files.

Data source
-----------
cBioPortal datahub (GitHub):
  https://github.com/cBioPortal/datahub/tree/master/public

One file is consumed per cancer type:
  datahub/public/<cancer>_tcga_pan_can_atlas_2018/data_clinical_patient.txt

Example (thyroid carcinoma):
  https://github.com/cBioPortal/datahub/tree/master/public/thca_tcga_pan_can_atlas_2018

Files are Git LFS objects; run fetch_lfs_clinical.py first to resolve pointers.

Citation
--------
Hoadley KA et al. (2018). Cell-of-Origin Patterns Dominate the Molecular
Classification of 10,000 Tumors from 33 Types of Cancer.
Cell 173(2):291-304. https://doi.org/10.1016/j.cell.2018.03.022

Conversion steps
----------------
1. Glob all data_clinical_patient.txt files under *pan_can_atlas_2018/
2. Parse each with pd.read_csv(sep='\\t', comment='#') — skips 5-line # header
3. Tag rows with cancer type abbreviation from directory name (e.g. THCA)
4. Concatenate all 33 cancer types (~10,000+ rows)
5. Normalise columns:
     OS_MONTHS                     -> float (errors -> NaN)
     OS_STATUS "1:DECEASED"        -> 1, else 0
     AGE / DIAGNOSIS_AGE           -> float
     AJCC_PATHOLOGIC_TUMOR_STAGE   -> int 1-4 (Roman numeral mapping)
     TMB_NONSYNONYMOUS or
       MUTATION_COUNT / 38         -> TMB in mut/Mb (38 Mb ~= exome size)
6. Derive chemotherapy proxy (TCGA does not uniformly record treatment):
     logit P(Chemo) = -1.5 + 0.55*Stage - 0.015*(Age-60)
     Bernoulli draw with seed 42 creates realistic indication bias.
7. Drop rows missing OS_MONTHS, OS_EVENT, AGE, STAGE, CHEMO or OS_MONTHS <= 0
8. Save to data/processed/analysis_dataset.parquet (snappy compression)

Result: 6,568 patients across 19 cancer types.

Usage (run from repo root):
    python src/build_real_dataset.py
    python src/build_real_dataset.py --datahub /path/to/datahub/public

If --datahub is omitted, the script auto-detects a datahub/public sibling
directory next to this repo.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Build TCGA analysis parquet.')
parser.add_argument('--datahub', default=None,
                    help='Path to datahub/public directory')
args = parser.parse_args()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if args.datahub:
    DATA_DIR = args.datahub
else:
    sibling = os.path.join(os.path.dirname(BASE), 'datahub', 'public')
    if os.path.isdir(sibling):
        DATA_DIR = sibling
        print(f'Auto-detected datahub at: {DATA_DIR}')
    else:
        print('ERROR: datahub/public not found. Run fetch_lfs_clinical.py first or pass --datahub.')
        raise SystemExit(1)

OUT_PATH = os.path.join(BASE, 'data', 'processed', 'analysis_dataset.parquet')

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

files = glob.glob(os.path.join(DATA_DIR, '*pan_can_atlas_2018', 'data_clinical_patient.txt'))
print(f'Found {len(files)} clinical files.')

dfs = []
for fpath in files:
    abbr = os.path.basename(os.path.dirname(fpath)).split('_tcga')[0].upper()
    try:
        tmp = pd.read_csv(fpath, sep='\t', comment='#', low_memory=False)
        tmp.columns = tmp.columns.str.upper()
        tmp['CANCER_TYPE_ABBR'] = abbr
        dfs.append(tmp)
    except Exception as exc:
        print(f'  Skipped {abbr}: {exc}')

raw = pd.concat(dfs, ignore_index=True)
print(f'Raw rows: {len(raw):,}  columns: {len(raw.columns)}')

raw['OS_MONTHS'] = pd.to_numeric(raw.get('OS_MONTHS', pd.Series(dtype=float)), errors='coerce')
raw['OS_EVENT']  = raw.get('OS_STATUS', pd.Series(dtype=str)).apply(
    lambda x: 1 if '1' in str(x).split(':')[0] else 0)

age_col   = next((c for c in ['AGE', 'DIAGNOSIS_AGE'] if c in raw.columns), None)
stage_col = next((c for c in ['AJCC_PATHOLOGIC_TUMOR_STAGE', 'TUMOR_STAGE'] if c in raw.columns), None)
tmb_col   = next((c for c in ['TMB_NONSYNONYMOUS', 'MUTATION_COUNT'] if c in raw.columns), None)

raw['AGE']   = pd.to_numeric(raw[age_col], errors='coerce') if age_col else np.nan
raw['STAGE'] = raw[stage_col].apply(
    lambda s: 4 if 'IV' in str(s).upper() else 3 if 'III' in str(s).upper()
    else 2 if 'II' in str(s).upper() else 1 if 'I' in str(s).upper() else np.nan
) if stage_col else np.nan
raw['TMB'] = (pd.to_numeric(raw[tmb_col], errors='coerce') /
              (38.0 if tmb_col == 'MUTATION_COUNT' else 1.0)) if tmb_col else np.nan

# Derive chemotherapy proxy (stage + age logistic model)
np.random.seed(42)
logit   = -1.5 + 0.55 * raw['STAGE'].fillna(2.5) - 0.015 * (raw['AGE'].fillna(60) - 60)
p_chemo = 1 / (1 + np.exp(-logit))
raw['CHEMO'] = np.where(
    raw['STAGE'].notna() & raw['AGE'].notna(),
    (np.random.uniform(size=len(raw)) < p_chemo).astype(float),
    np.nan
)

keep = ['AGE', 'STAGE', 'CANCER_TYPE_ABBR', 'CHEMO', 'TMB', 'OS_MONTHS', 'OS_EVENT']
df = (raw[[c for c in keep if c in raw.columns]]
      .dropna(subset=['OS_MONTHS', 'OS_EVENT', 'AGE', 'STAGE', 'CHEMO'])
      .query('OS_MONTHS > 0')
      .reset_index(drop=True))

df.to_parquet(OUT_PATH, index=False)
print(f'\nSaved {len(df):,} patients → {OUT_PATH}')
print(f'Cancer types: {df["CANCER_TYPE_ABBR"].nunique()}  ({", ".join(sorted(df["CANCER_TYPE_ABBR"].unique()))})')
print(f'Chemo rate: {df["CHEMO"].mean():.1%}')
print(f'TMB available: {df["TMB"].notna().mean():.1%}')
print(df.head(3).to_string())
