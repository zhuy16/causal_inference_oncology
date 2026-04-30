# Data Provenance

## Source

**Dataset**: TCGA Pan-Cancer Atlas 2018  
**Provider**: cBioPortal for Cancer Genomics  
**Repository**: https://github.com/cBioPortal/datahub/tree/master/public  
**Example study folder**: https://github.com/cBioPortal/datahub/tree/master/public/thca_tcga_pan_can_atlas_2018  
**File used per study**: `data_clinical_patient.txt` (one per cancer type, 33 total)  
**License**: TCGA clinical data is open-tier (no dbGaP access required). Files are distributed under the cBioPortal datahub license.

### Citation

Hoadley KA, et al. (2018). Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer. *Cell*, 173(2), 291–304. https://doi.org/10.1016/j.cell.2018.03.022

CBioPortal datahub: https://github.com/cBioPortal/datahub

## Download Instructions

The raw files live in the cBioPortal datahub GitHub repository and are stored via Git LFS. The recommended approach uses a sparse checkout to download only the clinical files (~2 MB total) rather than the full multi-GB repo.

```bash
# 1. Sparse-clone the datahub (clinical files only)
git clone --no-checkout --depth 1 --filter=blob:none \
    https://github.com/cBioPortal/datahub.git ../datahub
cd ../datahub
git sparse-checkout init --cone
git sparse-checkout set $(git ls-tree HEAD public/ | grep pan_can_atlas | awk '{print $4}' | tr '\n' ' ')
git checkout
cd ../causal_inference_multiomics

# 2. Resolve LFS pointers → download actual file content
python src/fetch_lfs_clinical.py       # auto-detects ../datahub/public

# 3. Build the analysis parquet
python src/build_real_dataset.py
```

After these steps, `data/processed/analysis_dataset.parquet` contains the analysis-ready dataset.

## Raw File Structure (per cancer type)

Each of the 33 cancer types has its own subdirectory in `datahub/public/`:

```
datahub/public/
├── acc_tcga_pan_can_atlas_2018/
│   ├── data_clinical_patient.txt   <- used by build_real_dataset.py
│   ├── data_clinical_sample.txt
│   ├── data_mutations.txt          <- real TMB (not used by default)
│   └── meta_*.txt
├── blca_tcga_pan_can_atlas_2018/
├── ... (33 cancer types total)
└── thca_tcga_pan_can_atlas_2018/
    └── data_clinical_patient.txt   <- example: https://github.com/cBioPortal/datahub/tree/master/public/thca_tcga_pan_can_atlas_2018
```

Only `data_clinical_patient.txt` is used. Each file has a 5-row comment header (lines starting with `#`) followed by a tab-separated table.

## Key Variables

### Patient-Level Clinical Data (`data_clinical_patient.txt`)

| Variable | Description | Notes |
|----------|-------------|-------|
| `PATIENT_ID` | Unique patient ID (TCGA-XX-XXXX format) | TSS code (XX) used as center proxy for IV analysis |
| `AGE` | Age at diagnosis (years) | |
| `SEX` | Patient sex | |
| `OS_MONTHS` | Overall survival duration (months) | Primary outcome |
| `OS_STATUS` | Survival status (1:DECEASED, 0:LIVING) | Event indicator |
| `DFS_MONTHS` | Disease-free survival (months) | Secondary outcome |
| `DFS_STATUS` | Disease-free survival status | |
| `SUBTYPE` | Molecular subtype | |
| `MUTATION_COUNT` | Total somatic mutation count | Used to derive TMB |
| `TMB_NONSYNONYMOUS` | Tumor Mutation Burden (mut/Mb) | Mediator variable; available in some cancer types |
| `FRACTION_GENOME_ALTERED` | Fraction of genome with copy number alterations | |
| `ANEUPLOIDY_SCORE` | Chromosomal instability score | |

### Sample-Level Data (`data_clinical_sample.txt`)

| Variable | Description |
|----------|-------------|
| `SAMPLE_ID` | Tumor sample ID (TCGA-XX-XXXX-01 format) |
| `PATIENT_ID` | Patient ID (links to clinical) |
| `CANCER_TYPE` | Cancer type name |
| `CANCER_TYPE_DETAILED` | Histological subtype |
| `SAMPLE_TYPE` | Primary vs. Metastatic |
| `AJCC_PATHOLOGIC_TUMOR_STAGE` | TNM staging |

## Conversion Pipeline: Raw Files → `analysis_dataset.parquet`

`build_real_dataset.py` performs the following steps:

1. **Glob** all `data_clinical_patient.txt` files matching `*pan_can_atlas_2018/data_clinical_patient.txt`
2. **Parse** each file with `pd.read_csv(..., sep='\t', comment='#')` — skips the 5-line `#`-prefixed metadata header
3. **Tag** each row with the cancer type abbreviation extracted from the directory name (e.g. `thca_tcga_pan_can_atlas_2018` → `THCA`)
4. **Concatenate** all 33 cancer types into a single dataframe (~10,000+ rows)
5. **Normalise columns:**
   - `OS_MONTHS` → `float` (coerce errors to NaN)
   - `OS_STATUS` → binary event flag: `1` if the string starts with `"1:"` (deceased), else `0`
   - `AGE` / `DIAGNOSIS_AGE` → `float`
   - `AJCC_PATHOLOGIC_TUMOR_STAGE` → integer 1–4 (Roman numeral mapping: I→1, II→2, III→3, IV→4)
   - `TMB_NONSYNONYMOUS` or `MUTATION_COUNT/38` → TMB in mut/Mb (38 Mb ≈ exome size)
6. **Derive chemotherapy proxy** (TCGA does not uniformly record treatment):
   ```
   logit P(Chemo) = -1.5 + 0.55 × Stage - 0.015 × (Age - 60)
   ```
   This creates realistic indication bias: Stage IV patients have ~60% chemo probability; Stage I patients ~20%.
   Binary assignment is drawn from Bernoulli(P) with seed 42.
7. **Filter** to rows with non-missing OS_MONTHS, OS_EVENT, AGE, STAGE, CHEMO and OS_MONTHS > 0
8. **Save** to `data/processed/analysis_dataset.parquet` (pandas parquet, snappy compression)

The resulting dataset has **6,568 patients** across **19 cancer types** (cancer types with missing stage or OS data are dropped at step 7).

---

## Variable Construction for Causal Analysis

### Treatment: Chemotherapy
- **Preferred**: `PHARMACEUTICAL_TX_GIVEN` or explicit chemotherapy field (available in select cancer types)
- **Fallback**: Binary proxy derived from stage and cancer-type-specific treatment guidelines, with explicit documentation

> **Important caveat**: TCGA was not designed as a treatment-outcome study. Chemotherapy records are inconsistent across cancer types and time periods. For real pharma RWE, you would use EHR treatment records, claims data, or registry data with verified treatment information.

### Mediator: Tumor Mutation Burden (TMB)
- Primary: `TMB_NONSYNONYMOUS` (nonsynonymous mutations per megabase)
- Fallback: `MUTATION_COUNT` / 38 (approximate exome size in Mb)
- TMB is a well-validated biomarker for immunotherapy response (FDA-approved companion diagnostic for pembrolizumab)

### Confounder: Cancer Stage
- Derived from `AJCC_PATHOLOGIC_TUMOR_STAGE` (Roman numeral → integer 1–4)
- Stage is the most important confounder: Stage IV patients have worse prognosis AND are more likely to receive systemic therapy

### Instrument: Treatment Center
- Derived from TCGA Patient ID TSS code (characters 6–7: `TCGA-[TSS]-XXXX`)
- Tissue Source Sites (TSS) map to contributing institutions
- Assumption: Some centers have higher chemotherapy utilization rates independent of patient severity (institutional protocols, trial enrollment)

## Data Limitations for Causal Inference

1. **No explicit randomization**: TCGA is a retrospective cohort; treatment assignment is observational
2. **Incomplete treatment records**: Not all TCGA patients have chemotherapy status documented
3. **Left truncation**: Patients had to survive long enough to be enrolled and biopsied
4. **Selection into TCGA**: Major academic cancer centers are over-represented
5. **No time-varying confounding**: We treat baseline variables as fixed; in reality, treatment decisions evolve with disease progression
6. **Positivity violations**: Some stage/cancer-type combinations have near-zero probability of treatment
7. **Exclusion restriction**: The center instrument may violate exclusion restriction if center quality affects outcomes through channels other than chemotherapy use

These limitations are discussed in detail in each notebook and in notebook 06 (Sensitivity Analysis).

## Processed Data

After running the notebooks, processed datasets will be saved in `data/processed/`:
- `analysis_dataset.parquet`: Main analysis dataset with all derived variables
- `matched_cohort.parquet`: Propensity score matched pairs (from notebook 02)
