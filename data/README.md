# Data Provenance

## Source

**Dataset**: TCGA Pan-Cancer Atlas 2018  
**Provider**: cBioPortal for Cancer Genomics  
**URL**: https://cbioportal-datahub.s3.amazonaws.com/tcga_pan_can_atlas_2018.tar.gz  
**Size**: ~2 GB compressed, ~10 GB extracted  
**License**: TCGA data is publicly available under dbGaP controlled-access and open-tier tiers. Clinical data used here is open-tier.

### Citation

Hoadley KA, et al. (2018). Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer. *Cell*, 173(2), 291–304. https://doi.org/10.1016/j.cell.2018.03.022

## Download Instructions

### Automatic (via notebooks)
The data is downloaded automatically the first time notebook 02 is run. The download and extraction can take 10–30 minutes depending on network speed.

### Manual

```bash
mkdir -p data/raw
cd data/raw
wget https://cbioportal-datahub.s3.amazonaws.com/tcga_pan_can_atlas_2018.tar.gz
tar -xzf tcga_pan_can_atlas_2018.tar.gz
```

Or with curl:

```bash
curl -O https://cbioportal-datahub.s3.amazonaws.com/tcga_pan_can_atlas_2018.tar.gz
tar -xzf tcga_pan_can_atlas_2018.tar.gz
```

## Expected Directory Structure After Extraction

```
data/raw/tcga_pan_can_atlas_2018/
├── acc_tcga_pan_can_atlas_2018/
│   ├── data_clinical_patient.txt
│   ├── data_clinical_sample.txt
│   └── meta_*.txt
├── blca_tcga_pan_can_atlas_2018/
│   ├── data_clinical_patient.txt
│   └── ...
├── ... (33 cancer types total)
└── read_tcga_pan_can_atlas_2018/
```

Each cancer type folder follows the naming convention `{cancer_type}_tcga_pan_can_atlas_2018/`.

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
