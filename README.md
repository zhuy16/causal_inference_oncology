# Causal Inference in Oncology: TCGA Pan-Cancer Atlas

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange.svg)](https://jupyter.org/)

## The Central Question

> **Does chemotherapy causally improve survival in cancer patients — and how much of that benefit is mediated through tumor mutation burden (TMB)?**

This is not just a statistical question. It is a **causal** one. Observational data alone cannot answer it without principled methods to handle confounding, mediation, and hidden bias. This repository walks through six complementary causal inference techniques applied to **6,568 real patients** from the TCGA Pan-Cancer Atlas 2018.

---

## Quick Start

### Step 1 — Clone this repo and set up the environment

```bash
git clone https://github.com/<your-handle>/causal_inference_multiomics
cd causal_inference_multiomics

# Create and activate the conda environment (first time only)
conda env create -f environment.yml
conda activate causal_multiomics
```

> If you already have the `multiomics-demo` environment from a prior session, run `conda activate multiomics-demo` instead.

### Step 2 — Get the TCGA data (choose one option)

**Option A — Download real TCGA clinical files (~2 MB, recommended)**

First clone the cBioPortal datahub repo (clinical files only, not the full multi-GB repo):
```bash
git clone --no-checkout --depth 1 --filter=blob:none \
    https://github.com/cBioPortal/datahub.git ../datahub
cd ../datahub
git sparse-checkout init --cone
git sparse-checkout set $(git ls-tree HEAD public/ | grep pan_can_atlas | awk '{print $4}' | tr '\n' ' ')
git checkout
cd ../causal_inference_multiomics
```

Then download the actual file content from LFS and build the dataset:
```bash
python fetch_lfs_clinical.py          # auto-detects ../datahub/public
python build_real_dataset.py          # builds data/processed/analysis_dataset.parquet
```

If datahub is not a sibling directory, pass the path explicitly:
```bash
python fetch_lfs_clinical.py --datahub /your/path/to/datahub/public
python build_real_dataset.py --datahub /your/path/to/datahub/public
```

**Option B — Use fully synthetic data (no internet needed)**
```bash
python generate_synthetic_data.py     # creates 8,000 synthetic TCGA-like patients
```
Results will be methodologically valid demonstrations with simulated confounding.

### Step 3 — Run the notebooks

```bash
jupyter lab notebooks/
```

Open notebooks in order (01 → 06). Each notebook loads the parquet cache automatically.
To re-execute all notebooks from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/0*.ipynb
```

See [`docs/data_guide.md`](docs/data_guide.md) for full data details, variable definitions, and how to add real TMB data.

---

## How to Read This Repository

The six notebooks form a **single coherent argument**. Read them in order.

```
Notebook 01  →  02  →  03  →  04  →  05  →  06
  (Why?)      (What?)  (When?)  (How?)  (Check)  (Trust?)
   DAG         PSM      DiD    Mediation   IV    Sensitivity
```

Each notebook builds on the previous:
- **01** establishes *which variables* to adjust for (and which not to)
- **02** gives the primary causal estimate using propensity score matching
- **03** cross-validates using a completely different identification strategy (time variation)
- **04** breaks the total effect into direct and TMB-mediated components
- **05** uses institutional variation as a natural experiment to check for hidden confounders
- **06** stress-tests all conclusions against unmeasured confounding

---

## The Biological Problem: Why Naive Analysis Fails

A naive comparison shows that chemo patients have **worse** survival than untreated patients. This seems paradoxical — but it is exactly what we expect from **indication bias**:

```
Stage IV → receives Chemo         (sicker patients get treated)
Stage IV → worse survival         (sicker patients die sooner)
∴ Chemo appears to harm survival  (spurious correlation)
```

The DAG (Notebook 01) formalizes this: **Stage and Age are backdoor confounders** that create a non-causal path between Chemo and Survival. Closing this backdoor path — through adjustment, matching, or instrumental variables — is the entire purpose of the subsequent notebooks.

> **Key intuition**: Comparing a Stage IV chemo patient to a Stage I no-chemo patient is like comparing apples to oranges. Causal inference forces an apples-to-apples comparison.

---

## Step-by-Step Notebook Guide

### Notebook 01 — DAG & Causal Assumptions
**File**: `notebooks/01_dag_causal_assumptions.ipynb`
**Figures**: `01_causal_dag.png`, `01_indication_bias_demo.png`

**What it does**: Draws the causal graph (DAG) of our system and uses the backdoor criterion to identify the minimum set of variables to adjust for.

**Key concept — the DAG**:
```
Age ──────────────────────────┐
                              ↓
Stage ──→ Chemo ──→ TMB ──→ Survival
         ↑                    ↑
Cancer Type ─────────────────┘
```
- **Confounders** (Age, Stage, Cancer Type): lie on *backdoor paths* — must be adjusted
- **Mediator** (TMB): lies *on* the causal path — must NOT be adjusted in the total effect model
- **Colliders**: variables that, if conditioned on, *open* spurious paths (a common mistake)

**What to look for in `01_indication_bias_demo.png`**: The naive estimate (no adjustment) should show a smaller or reversed effect compared to the adjusted estimate. The gap between the two lines is the magnitude of indication bias.

**Key takeaway**: Without the DAG, we might adjust for the wrong variables. Adjusting for a mediator (TMB) when estimating the total effect *removes* part of the causal effect we are trying to measure.

---

### Notebook 02 — Propensity Score Matching (PSM)
**File**: `notebooks/02_propensity_score_matching.ipynb`
**Figures**: `02_propensity_score_dist.png`, `02_love_plot.png`, `02_common_support.png`, `02_psm_results.png`

**What it does**: Constructs a matched cohort where treated and control patients are statistically identical on observed confounders, then estimates the Average Treatment Effect (ATE).

**Key concept — propensity score**:
The propensity score $e(X) = P(\text{Chemo}=1 \mid X)$ is a single number summarising all confounders. Matching on it is equivalent (under assumptions) to matching on all confounders simultaneously.

**What to look for in each figure**:

| Figure | What to look for | Good sign |
|--------|-----------------|-----------|
| `02_propensity_score_dist.png` | Overlap between treated/control PS distributions | Substantial overlap (not fully separated) |
| `02_love_plot.png` | Standardised Mean Differences (SMD) before/after matching | All SMDs < 0.1 after matching (dots left of dashed line) |
| `02_common_support.png` | Mirror plot of PS distributions | Near-identical shapes after matching |
| `02_psm_results.png` | Naive ATE vs PSM ATE, KM curves | PSM ATE differs from naive — confirms confounding was present |

**Key takeaway**: The Love plot is the single most important diagnostic. If SMDs are not below 0.1 after matching, the causal estimate is unreliable. Every FDA RWE submission must show a Love plot.

---

### Notebook 03 — Difference-in-Differences (DiD)
**File**: `notebooks/03_difference_in_differences.ipynb`
**Figures**: `03_parallel_trends.png`, `03_did_results.png`, `03_event_study.png`

**What it does**: Exploits a treatment guideline change circa 2010 to create a natural experiment. Cancer types where guidelines changed (treated group) are compared to those where they did not (control group), before and after the policy.

**Key concept — DiD logic**:
```
DiD = (Post - Pre)_treated  −  (Post - Pre)_control
    = Change in treated group  −  Change that would have happened anyway
```
This removes all time-invariant confounders — even unmeasured ones — as long as the **parallel trends assumption** holds (treated and control groups were trending similarly before the policy).

**What to look for in each figure**:

| Figure | What to look for | Good sign |
|--------|-----------------|-----------|
| `03_parallel_trends.png` | Pre-treatment trends of treated vs control groups | Parallel (similar slope) before the cutoff year |
| `03_did_results.png` | DiD coefficient with confidence interval | Significant and in the expected direction |
| `03_event_study.png` | Year-by-year coefficients relative to the event year | Near-zero before year 0, divergence after |

**Key takeaway**: The event study is a credibility check — if the coefficient is non-zero *before* the policy change, the parallel trends assumption is violated and the DiD is invalid.

---

### Notebook 04 — Mediation Analysis
**File**: `notebooks/04_mediation_analysis.ipynb`
**Figures**: `04_mediation_analysis.png`, `04_mediation_sensitivity.png`

**What it does**: Decomposes the total effect of chemo on survival into:
- **Direct Effect**: Chemo → Survival (cytotoxic effect, unrelated to TMB)
- **Indirect Effect**: Chemo → TMB → Survival (immune-mediated pathway)

**Key concept — why mediation matters**:
If 40% of chemo's benefit is mediated through TMB elevation, then:
1. Patients with pre-existing high TMB may need lower chemo doses to achieve the same immune response
2. Sequencing chemo before immunotherapy may be synergistic (chemo boosts TMB → better IO response)
3. TMB is a *mechanism biomarker*, not just a prognostic one — this is a regulatory distinction

**What to look for in each figure**:

| Figure | What to look for | Good sign |
|--------|-----------------|-----------|
| `04_mediation_analysis.png` | Path diagram + effect decomposition bar chart + bootstrap CI | Bootstrap CI for indirect effect excludes zero |
| `04_mediation_sensitivity.png` | How indirect effect changes as unmeasured M→Y confounding increases | Effect remains non-zero even at plausible confounding levels |

**Key takeaway**: The proportion mediated tells you how much of the therapeutic benefit would be lost if you blocked the TMB pathway. If it is >30%, TMB is a clinically meaningful mechanistic mediator, not just a marker.

> **Note on TMB data**: The TCGA clinical files do not include TMB directly. Notebook 04 detects this and generates a biologically-motivated TMB simulation (chemo patients have higher TMB). The mediation results should be interpreted as a **methodological demonstration**. Real TMB data can be loaded from `data_mutations.txt` files in the datahub.

---

### Notebook 05 — Instrumental Variables (IV)
**File**: `notebooks/05_instrumental_variables.ipynb`
**Figures**: `05_stage1_instrument.png`, `05_iv_vs_ols.png`

**What it does**: Uses variation in institutional chemotherapy prescribing rates (treatment center as instrument) to isolate the causal effect of chemo, even in the presence of *unmeasured* confounders.

**Key concept — why IV is different from PSM**:
PSM removes confounding from *measured* variables. IV removes confounding from *everything* — measured and unmeasured — by using a variable that influences treatment but has no direct effect on the outcome. It is the observational study's closest analogue to a randomised trial.

**The instrument**: Leave-one-out center chemotherapy rate. Centers with historically higher chemo utilisation are more likely to give chemo to patients regardless of individual patient factors.

**What to look for in each figure**:

| Figure | What to look for | Good sign |
|--------|-----------------|-----------|
| `05_stage1_instrument.png` | Correlation between instrument and treatment (first stage) | Strong positive correlation; F-statistic > 10 |
| `05_iv_vs_ols.png` | IV (2SLS) estimate vs OLS estimate with confidence intervals | If IV ≈ OLS, unmeasured confounding is minimal. If IV > OLS, there is negative selection (sicker patients getting treated). |

**Key takeaway**: The gap between IV and OLS estimates is informative. A large gap means unmeasured confounders (e.g., performance status) are biasing the OLS estimate. This is the basis of **Mendelian Randomization** — the most important causal inference method in genomics.

---

### Notebook 06 — Sensitivity Analysis
**File**: `notebooks/06_sensitivity_analysis.ipynb`
**Figures**: `06_evalue.png`, `06_rosenbaum_bounds.png`, `06_specification_sensitivity.png`

**What it does**: Answers the critical question: *How much unmeasured confounding would be needed to make our positive finding disappear?*

**Three tools**:

1. **E-value**: Minimum risk ratio that an unmeasured confounder must have with *both* treatment *and* outcome to fully explain away the effect. If E-value = 3.0, any single confounder needs to be 3× more common in treated patients AND increase mortality by 3× — no known confounder reaches this.

2. **Rosenbaum Bounds (Γ)**: In matched pairs, how much could two otherwise identical patients differ in their unmeasured odds of receiving chemo before the conclusion changes? Γ = 2 means a 2-fold difference in hidden confounding would be needed.

3. **Specification curve**: Does the result hold across different PS models, matching ratios, and calipers? Consistency across specifications = robustness.

**What to look for in each figure**:

| Figure | What to look for | Good sign |
|--------|-----------------|-----------|
| `06_evalue.png` | E-value vs known confounders' estimated effect sizes | E-value exceeds the strongest known confounder (ECOG ~2.5×) |
| `06_rosenbaum_bounds.png` | Critical Γ value vs plausible confounders | Γ > 2.0 means substantial hidden bias needed |
| `06_specification_sensitivity.png` | Distribution of ATE estimates across all specifications | Consistent direction and significance across >75% of specs |

**Key takeaway**: A study is not credible without sensitivity analysis. The E-value is now routinely requested by *NEJM*, *JAMA*, and FDA reviewers. Rosenbaum bounds are standard in matched-cohort study submissions.

---

## Data Status

| Data element | Source | Status |
|---|---|---|
| Patient demographics (Age, Stage) | Real TCGA Pan-Cancer Atlas 2018 | ✅ Real (6,568 patients, 19 cancer types) |
| Overall Survival (OS months, event) | Real TCGA | ✅ Real |
| Cancer type | Real TCGA | ✅ Real |
| Chemotherapy indicator | Derived proxy | ⚠️ Proxy — TCGA does not uniformly record chemo; derived from stage + age logistic model |
| Tumor Mutation Burden (TMB) | Not in clinical files | ⚠️ Simulated in NB04 — real TMB is in `data_mutations.txt` (separate download) |
| Treatment center | Simulated institutional variation | ⚠️ Simulated — real TSS codes require patient-level IDs not in public files |

> The chemotherapy proxy is derived from a logistic model: $P(\text{Chemo}) = \sigma(-1.5 + 0.55 \times \text{Stage} - 0.015 \times (\text{Age} - 60))$. This creates realistic indication bias for demonstration purposes. The methods and concepts are valid; the specific effect estimates should not be interpreted as clinical evidence.

---

## Repository Structure

```
causal_inference_multiomics/
├── notebooks/
│   ├── 01_dag_causal_assumptions.ipynb
│   ├── 02_propensity_score_matching.ipynb
│   ├── 03_difference_in_differences.ipynb
│   ├── 04_mediation_analysis.ipynb
│   ├── 05_instrumental_variables.ipynb
│   └── 06_sensitivity_analysis.ipynb
├── data/
│   ├── README.md                        # Variable definitions and limitations
│   ├── processed/
│   │   ├── analysis_dataset.parquet     # Built from real TCGA files
│   │   └── matched_cohort.parquet       # PSM output (NB02)
│   └── raw/                             # Downloaded clinical files
├── results/figures/                     # All 16 generated figures
├── docs/
│   ├── concepts.md                      # Causal inference concepts explained
│   ├── data_guide.md                    # Complete data provenance and rebuild guide
│   └── figures_guide.md                 # Figure-by-figure interpretation guide
├── fetch_lfs_clinical.py                # Downloads real TCGA files from GitHub LFS
├── build_real_dataset.py                # Builds parquet from real clinical files
├── generate_synthetic_data.py           # Fallback: generates synthetic dataset
├── environment.yml
├── Dockerfile
└── README.md
```

---

## Pharma & Regulatory Relevance

| Method | Industry Application | Regulatory Context |
|--------|---------------------|-------------------|
| DAG / Backdoor criterion | Define the minimum sufficient adjustment set for any RWE study | FDA RWE Guidance (2018, 2023) — requires pre-specified analysis plan |
| Propensity Score Matching | Build synthetic control arms for single-arm oncology trials | FDA Project Optimus, EMA RWE framework |
| Difference-in-Differences | Evaluate drug policy changes, label expansions in EHR data | HEOR / HTA submissions (NICE, G-BA) |
| Mediation Analysis | Distinguish on-target vs off-target mechanisms of action | Regulatory MoA packages; biomarker co-development |
| Instrumental Variables | Mendelian Randomization for genetic drug target validation | pQTL-based target ID (AstraZeneca, Pfizer pipelines) |
| Sensitivity Analysis | Quantify robustness of RWE findings to hidden confounders | NEJM/JAMA requirement; FDA advisory committee submissions |

---

## Further Reading

See [`docs/concepts.md`](docs/concepts.md) for deeper explanations of each method.
See [`docs/figures_guide.md`](docs/figures_guide.md) for a figure-by-figure interpretation guide.
See [`docs/data_guide.md`](docs/data_guide.md) for data provenance and how to rebuild the dataset.

### Key References
- Hernán & Robins (2020). *Causal Inference: What If* — [free PDF](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- Pearl (2009). *Causality: Models, Reasoning, and Inference*
- VanderWeele (2015). *Explanation in Causal Inference* — the mediation analysis bible
- Rosenbaum & Rubin (1983). The central role of the propensity score. *Biometrika* 70(1)
- VanderWeele & Ding (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*
- Davey Smith & Ebrahim (2003). Mendelian randomization. *International Journal of Epidemiology*
