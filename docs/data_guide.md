# Data Guide: What Is Real, What Is Derived, and How to Rebuild

## Current Dataset Status

The analysis dataset (`data/processed/analysis_dataset.parquet`) contains **6,568 patients** across **19 cancer types** built from real TCGA Pan-Cancer Atlas 2018 clinical files.

| Variable | Real or Derived? | Notes |
|----------|-----------------|-------|
| `AGE` | ✅ Real TCGA | Range 14–90, mean ~60 |
| `STAGE` | ✅ Real TCGA | AJCC I–IV, parsed from free-text |
| `CANCER_TYPE_ABBR` | ✅ Real TCGA | 19 types present in clinical files |
| `OS_MONTHS` | ✅ Real TCGA | Overall survival in months |
| `OS_EVENT` | ✅ Real TCGA | 1 = deceased, 0 = alive at last follow-up |
| `CHEMO` | ⚠️ Derived proxy | Logistic model from Stage + Age (see below) |
| `TMB` | ⚠️ Simulated (NB04 only) | TCGA clinical files do not include TMB |

### Cancer Types Present
ACC, BLCA, BRCA, CHOL, COADREAD, ESCA, HNSC, KICH, KIRC, KIRP, LIHC, LUAD, LUSC, MESO, PAAD, SKCM, STAD, TGCT, THCA

Some cancer types present in the full Pan-Cancer Atlas (e.g., GBM, OV, UCEC) do not appear because their AJCC stage or age data was missing at sufficient rates to be excluded after filtering.

---

## The Chemotherapy Proxy

TCGA does not record chemotherapy uniformly across all cancer types. The proxy is derived as:

```
logit(P(Chemo)) = -1.5 + 0.55 × Stage - 0.015 × (Age - 60)
```

This creates **realistic indication bias**: Stage IV patients have ~55% chemo probability vs ~20% for Stage I, and younger patients are slightly more likely to receive chemo. The proxy is seeded with `np.random.seed(42)` for reproducibility.

**What this means for results**: The causal estimates are valid *demonstrations of methodology* with realistic confounding structure. They are not clinical evidence about chemotherapy efficacy.

**How to use real chemotherapy data**: Several TCGA cancer types do include treatment fields:
- `PHARMACEUTICAL_TX_GIVEN` (some studies)
- `CHEMOTHERAPY` (some studies)
Check column availability with: `raw.columns[raw.columns.str.contains('CHEMO|PHARMA|TREATMENT', case=False)]`

---

## Getting Real TMB Data

TMB (mutations per megabase) is stored in the mutation files, not the clinical files. To add real TMB:

```bash
# 1. Download mutation data for specific cancer types of interest
# Edit src/fetch_lfs_clinical.py: change 'data_clinical_patient.txt' to 'data_mutations.txt'

# 2. Count mutations per patient
python3 -c "
import pandas as pd, glob, os
files = glob.glob('/path/to/datahub/public/*pan_can_atlas_2018/data_mutations.txt')
dfs = [pd.read_csv(f, sep='\t', usecols=['Tumor_Sample_Barcode'], comment='#') for f in files[:5]]
mut_counts = pd.concat(dfs).groupby('Tumor_Sample_Barcode').size() / 38.0  # mut/Mb
print(mut_counts.describe())
"
```

Note: mutation files are large (~50–500 MB each). Use sparse download for only the cancer types you need.

---

## How to Rebuild the Dataset

### Prerequisites
- Conda environment active: `conda activate causal_multiomics` (or `multiomics-demo`)
- datahub repo cloned — see README.md Step 2 for the sparse-checkout command

### Full rebuild from real TCGA files:
```bash
# Download real clinical TSV files via LFS
python src/fetch_lfs_clinical.py
# or, if datahub is not a sibling directory:
python src/fetch_lfs_clinical.py --datahub /your/path/to/datahub/public

# Build the analysis parquet
python src/build_real_dataset.py

# Clear derived caches and re-run all notebooks
rm -f data/processed/matched_cohort.parquet data/processed/sensitivity_summary.csv
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/0*.ipynb
```

### Rebuild with synthetic data (no internet required):
```bash
python src/generate_synthetic_data.py
# Generates 8,000 synthetic patients with realistic confounding structure
```

### Selective rebuild (one notebook):
```bash
# Example: rebuild only NB04 (mediation analysis)
rm -f data/processed/analysis_dataset.parquet
python src/build_real_dataset.py
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/04_mediation_analysis.ipynb
```

---

## File Inventory

```
data/
├── README.md                           # This file's companion for variable definitions
├── raw/
│   └── (clinical TSV files downloaded by src/fetch_lfs_clinical.py)
└── processed/
    ├── analysis_dataset.parquet        # 6,568 rows × 7 cols — main analysis input
    ├── matched_cohort.parquet          # PSM output from NB02 (treated + matched controls)
    └── sensitivity_summary.csv         # Spec curve results from NB06
```

---

## Data Limitations for Causal Inference

1. **TCGA is not a treatment study**: Samples were collected for genomics, not treatment effectiveness. Survival follow-up is shorter than in dedicated clinical trials.

2. **Left truncation**: Patients had to survive long enough to be enrolled and have tissue collected. This biases survival estimates upward.

3. **No ECOG performance status**: The single strongest predictor of chemo eligibility and survival is missing. This is the primary residual unmeasured confounder (see Notebook 06).

4. **Version note**: We use the 2018 Pan-Cancer Atlas freeze. Clinical data may differ from earlier TCGA releases for the same patients.
