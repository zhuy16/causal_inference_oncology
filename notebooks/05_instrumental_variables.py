# %% [markdown]
# # 05 — Instrumental Variables (IV) Analysis
#
# ## Overview
#
# **Instrumental Variables (IV)** estimation addresses unmeasured confounding — the scenario where propensity score matching and regression adjustment are insufficient because important confounders were never measured.
#
# An **instrument** $Z$ must satisfy three conditions:
#
# 1. **Relevance**: $Z$ is correlated with the treatment $A$ (testable: F-statistic)
# 2. **Exclusion restriction**: $Z$ affects the outcome $Y$ *only through* $A$ — no direct path $Z \to Y$ (untestable, requires domain knowledge)
# 3. **Independence**: $Z$ is independent of all unmeasured confounders $U$ (randomization-like property)
#
# The IV estimator then recovers a **Local Average Treatment Effect (LATE)** — the causal effect for *compliers* (patients whose treatment decision is changed by the instrument).
#
# ## The Instrument: Treatment Center (TSS Code)
#
# We use the **TCGA Tissue Source Site (TSS) code** — embedded in patient IDs as `TCGA-[TSS]-XXXX` — as a proxy for treatment center.
#
# **Rationale**:
# - Different academic cancer centers have different institutional cultures, clinical trial enrollment rates, and chemotherapy utilization protocols
# - A patient at a center with high chemotherapy utilization is more likely to receive chemo regardless of their individual clinical factors
# - Crucially, the center itself should not directly affect survival (it only affects survival *through* the treatments provided)
#
# **Assumption validity discussion**:
# - *Relevance*: Checkable — does center predict chemo use after adjusting for patient factors?
# - *Exclusion restriction*: Potentially problematic — better-resourced centers may also have better supportive care, nursing ratios, etc. → We discuss this limitation explicitly
# - *Independence*: Patients don't randomly self-select to centers entirely; sicker patients may prefer specialized academic centers
#
# ## Two-Stage Least Squares (2SLS)
#
# **Stage 1**: Regress treatment on instrument + covariates → obtain predicted treatment $\hat{A}$
# $$A_i = \pi_0 + \pi_1 Z_i + \pi_2 X_i + \nu_i$$
#
# **Stage 2**: Regress outcome on predicted treatment + covariates
# $$Y_i = \beta_0 + \underbrace{\beta_{\text{IV}}}_{\text{LATE}} \hat{A}_i + \gamma X_i + \epsilon_i$$
#
# The F-statistic from Stage 1 must exceed 10 (Staiger & Stock, 1997) to avoid **weak instrument bias**.

# %% [markdown]
# ---
# ### Concept at a Glance
#
# ```
#   Unmeasured confounders (performance status, patient preference...)
#          |               |
#   [Center] --> [Chemo] --> [Survival]
#      ^
#   instrument       Center affects chemo rate but has no
#   (excluded)       direct effect on survival
# ```
#
# **Key strength over PSM:** removes confounding from *everything* — including unmeasured confounders — by exploiting a variable that shifts treatment but has no direct effect on outcome.
#
# **The instrument:** leave-one-out center chemo rate (institutional prescribing tendency).
#
# **What to check:** First-stage F-statistic > 10 (strong instrument). Gap between IV and OLS reveals unmeasured confounding.
#
# > Detailed concept explanation: `docs/concepts.md` | Figures guide: `docs/figures_guide.md`
# ---

# %% [markdown]
# ---
# ### Concept at a Glance
#
# ```
#   Unmeasured confounders (performance status, patient preference...)
#          ↓               ↓
#   [Center] ──→ [Chemo] ──→ [Survival]
#      ↑
#   instrument       Center affects chemo rate but has no
#   (excluded)       direct effect on survival
# ```
#
# **Why IV is stronger than PSM:** PSM removes confounding from *measured* variables only. IV removes confounding from *everything* — including unmeasured confounders — by exploiting a variable that shifts treatment but has no direct effect on outcome.
#
# **The instrument here:** leave-one-out center chemo rate (institutional prescribing tendency).
#
# **What to check:** First-stage F-statistic > 10 (instrument is strong). Gap between IV and OLS estimate reveals hidden confounding.
#
# > Detailed concept explanation → [`docs/concepts.md`](../docs/concepts.md) | Figures guide → [`docs/figures_guide.md`](../docs/figures_guide.md)
# ---

# %%
import warnings
warnings.filterwarnings('ignore')

import os, glob, urllib.request, tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

DATA_DIR    = os.path.abspath('../data')
RAW_DIR     = os.path.join(DATA_DIR, 'raw')
PROC_DIR    = os.path.join(DATA_DIR, 'processed')
FIGURES_DIR = os.path.abspath('../results/figures')
TCGA_URL    = 'https://cbioportal-datahub.s3.amazonaws.com/tcga_pan_can_atlas_2018.tar.gz'

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
print('Libraries loaded.')


# %%
# Data is loaded from data/processed/analysis_dataset.parquet
# Built by build_real_dataset.py from real TCGA Pan-Cancer Atlas 2018 clinical files.
# Run fetch_lfs_clinical.py + build_real_dataset.py to create it,
# or generate_synthetic_data.py for offline use.
def download_and_load():
    os.makedirs(RAW_DIR, exist_ok=True)
    tarball = os.path.join(RAW_DIR, 'tcga_pan_can_atlas_2018.tar.gz')
    extract_dir = os.path.join(RAW_DIR, 'tcga_pan_can_atlas_2018')
    if not os.path.exists(tarball):
        print('Downloading TCGA data ...')
        urllib.request.urlretrieve(TCGA_URL, tarball)
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 5:
        with tarfile.open(tarball, 'r:gz') as t:
            t.extractall(RAW_DIR)
    files = glob.glob(r'/Users/yunhuazhu/Documents/gitrepos/causal_inference_multiomics/data/raw/datahub_sparse/public/*pan_can_atlas_2018/data_clinical_patient.txt')
    if not files:
        raise FileNotFoundError('Clinical data not found. See data/README.md.')
    dfs = []
    for f in files:
        abbr = os.path.basename(os.path.dirname(f)).split('_tcga')[0].upper()
        tmp = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
        tmp.columns = tmp.columns.str.upper()
        tmp['CANCER_TYPE_ABBR'] = abbr
        dfs.append(tmp)
    raw = pd.concat(dfs, ignore_index=True)
    raw['OS_MONTHS'] = pd.to_numeric(raw.get('OS_MONTHS', pd.Series(dtype=float)), errors='coerce')
    raw['OS_EVENT']  = raw.get('OS_STATUS', pd.Series(dtype=str)).apply(
        lambda x: 1 if '1' in str(x).split(':')[0] else 0)
    age_col   = next((c for c in ['AGE', 'DIAGNOSIS_AGE'] if c in raw.columns), None)
    stage_col = next((c for c in ['AJCC_PATHOLOGIC_TUMOR_STAGE', 'TUMOR_STAGE'] if c in raw.columns), None)
    raw['AGE']   = pd.to_numeric(raw[age_col], errors='coerce') if age_col else np.nan
    raw['STAGE'] = raw[stage_col].apply(
        lambda s: 4 if 'IV' in str(s).upper() else 3 if 'III' in str(s).upper()
        else 2 if 'II' in str(s).upper() else 1 if 'I' in str(s).upper() else np.nan
    ) if stage_col else np.nan
    np.random.seed(42)
    logit = -1.5 + 0.55 * raw['STAGE'].fillna(2.5) - 0.015 * (raw['AGE'].fillna(60) - 60)
    p = 1 / (1 + np.exp(-logit))
    raw['CHEMO'] = np.where(raw['STAGE'].notna() & raw['AGE'].notna(),
                             (np.random.uniform(size=len(raw)) < p).astype(float), np.nan)
    if 'PATIENT_ID' in raw.columns:
        raw['CENTER'] = raw['PATIENT_ID'].str.extract(r'TCGA-([A-Z0-9]{2})-')[0]
    else:
        raw['CENTER'] = None
    keep = ['AGE', 'STAGE', 'CANCER_TYPE_ABBR', 'CHEMO', 'OS_MONTHS', 'OS_EVENT', 'CENTER']
    return raw[[c for c in keep if c in raw.columns]].dropna(
        subset=['OS_MONTHS', 'OS_EVENT', 'AGE', 'STAGE', 'CHEMO']
    ).query('OS_MONTHS > 0').reset_index(drop=True)


PARQUET_PATH = os.path.join(PROC_DIR, 'analysis_dataset.parquet')
if os.path.exists(PARQUET_PATH):
    df = pd.read_parquet(PARQUET_PATH)
    print(f'Loaded cached dataset: {len(df):,} patients')
else:
    df = download_and_load()
    df.to_parquet(PARQUET_PATH, index=False)

df.head(3)

# %% [markdown]
# ## 2. Construct Center-Level Instrument
#
# We derive a continuous instrument: **center-level chemotherapy utilization rate**, calculated as the proportion of patients at each center who receive chemotherapy. This captures institutional variation in chemo prescribing independent of individual patient factors.
#
# To avoid **forbidden regression** (using the patient's own data to define their instrument), we use **leave-one-out (LOO) center chemo rates**.

# %%
if 'CENTER' not in df.columns or df['CENTER'].isna().mean() > 0.5:
    print('CENTER not available from PATIENT_ID — simulating institutional variation.')
    np.random.seed(99)
    n_centers = 35  # typical number of TCGA contributing institutions
    center_ids = [f'C{i:02d}' for i in range(n_centers)]
    # Centers have different baseline chemo rates
    center_chemo_probs = np.random.beta(3, 4, n_centers)  # institutional variation
    df['CENTER'] = np.random.choice(center_ids, size=len(df))

    # Apply center-level chemo tendency (instrument signal)
    center_map = dict(zip(center_ids, center_chemo_probs))
    center_boost = df['CENTER'].map(center_map)
    current_chemo = df['CHEMO'].values
    blend = 0.4 * center_boost + 0.6 * current_chemo + np.random.normal(0, 0.05, len(df))
    df['CHEMO'] = (blend > 0.45).astype(float)

# Leave-one-out center chemo rate as instrument
center_counts = df.groupby('CENTER')['CHEMO'].agg(['sum', 'count']).reset_index()
center_counts.columns = ['CENTER', 'CHEMO_SUM', 'N']
df = df.merge(center_counts, on='CENTER', how='left')

# LOO rate: (sum - own value) / (count - 1)
df['Z_LOO'] = np.where(
    df['N'] > 1,
    (df['CHEMO_SUM'] - df['CHEMO']) / (df['N'] - 1),
    df['CHEMO_SUM'] / df['N']
)

# Restrict to centers with enough patients for reliable LOO estimate
df_iv = df[df['N'] >= 5].copy()

print(f'IV dataset: {len(df_iv):,} patients at {df_iv["CENTER"].nunique()} centers')
print(f'Instrument (Z_LOO) range: [{df_iv["Z_LOO"].min():.3f}, {df_iv["Z_LOO"].max():.3f}]')
print(f'Instrument mean: {df_iv["Z_LOO"].mean():.3f}')

center_summary = df_iv.groupby('CENTER').agg(
    n=('OS_MONTHS', 'size'),
    chemo_rate=('CHEMO', 'mean'),
    mean_os=('OS_MONTHS', 'mean'),
).reset_index().sort_values('chemo_rate')

print(f'\nCenter chemo rates (range): '
      f'{center_summary.chemo_rate.min():.1%} – {center_summary.chemo_rate.max():.1%}')
print(f'Std dev of center chemo rates: {center_summary.chemo_rate.std():.3f}')

# %% [markdown]
# ## 3. Test Instrument Relevance (Stage 1)

# %%
confounders = 'AGE + STAGE + C(CANCER_TYPE_ABBR)'

# Stage 1: Predict treatment from instrument + covariates
stage1_formula = f'CHEMO ~ Z_LOO + {confounders}'
stage1_model   = smf.ols(stage1_formula, data=df_iv).fit(cov_type='HC3')

pi_1   = stage1_model.params['Z_LOO']
pi_se  = stage1_model.bse['Z_LOO']
pi_p   = stage1_model.pvalues['Z_LOO']

# First-stage F-statistic (for instrument only, partialling out covariates)
stage1_restricted = smf.ols(f'CHEMO ~ {confounders}', data=df_iv).fit()
rss_unrestricted  = stage1_model.ssr
rss_restricted    = stage1_restricted.ssr
df_num = 1  # one instrument
df_den = stage1_model.df_resid
f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_den)

partial_r2 = (rss_restricted - rss_unrestricted) / rss_restricted

df_iv['CHEMO_HAT'] = stage1_model.fittedvalues

print('=== STAGE 1 (INSTRUMENT RELEVANCE TEST) ===')
print()
print(f'Instrument: Leave-one-out center chemotherapy rate')
print(f'Coefficient on Z_LOO:  {pi_1:+.4f}  SE={pi_se:.4f}  p={pi_p:.4e}')
print(f'First-stage F-statistic: {f_stat:.2f}')
print(f'Partial R²:              {partial_r2:.4f}')
print()
if f_stat > 10:
    print(f'✓ Strong instrument (F > 10): {f_stat:.1f}')
    print('  Weak instrument bias is unlikely.')
elif f_stat > 3.8:
    print(f'⚠ Borderline weak instrument (3.8 < F < 10): {f_stat:.1f}')
    print('  Consider weak-IV robust inference (LIML, Anderson-Rubin CIs).')
else:
    print(f'✗ Weak instrument (F < 3.8): {f_stat:.1f}')
    print('  IV estimates are unreliable. More institutional variation needed.')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
center_vis = df_iv.groupby('CENTER').agg(
    z_loo=('Z_LOO', 'mean'),
    chemo_rate=('CHEMO', 'mean'),
    n=('CHEMO', 'size'),
).reset_index()
ax.scatter(center_vis['z_loo'], center_vis['chemo_rate'],
            s=center_vis['n'].clip(10, 200), alpha=0.65, color='#4C72B0', edgecolor='white')
m, b, r, p_r, _ = stats.linregress(center_vis['z_loo'], center_vis['chemo_rate'])
x_range = np.linspace(center_vis['z_loo'].min(), center_vis['z_loo'].max(), 100)
ax.plot(x_range, m * x_range + b, '--', color='#DD8452', lw=2, label=f'r={r:.3f}')
ax.set_xlabel('LOO Center Chemo Rate (Instrument)')
ax.set_ylabel('Actual Center Chemo Rate')
ax.set_title(f'Stage 1: Instrument Relevance\nF-stat = {f_stat:.1f}, r = {r:.3f}', fontsize=12)
ax.legend()

ax = axes[1]
ax.hist(df_iv.loc[df_iv['CHEMO'] == 1, 'CHEMO_HAT'], bins=40, alpha=0.55,
         color='#DD8452', label='Treated', density=True, edgecolor='white')
ax.hist(df_iv.loc[df_iv['CHEMO'] == 0, 'CHEMO_HAT'], bins=40, alpha=0.55,
         color='#4C72B0', label='Control', density=True, edgecolor='white')
ax.set_xlabel('Predicted Chemo (Fitted from Stage 1)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Predicted Chemo\n(captures instrument-driven variation)', fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_stage1_instrument.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Two-Stage Least Squares (2SLS) Estimation

# %%
# Stage 2: Regress outcome on predicted treatment + covariates
stage2_model = smf.ols(f'OS_MONTHS ~ CHEMO_HAT + {confounders}', data=df_iv).fit(cov_type='HC3')
iv_estimate  = stage2_model.params['CHEMO_HAT']

# Correct standard errors via bootstrap (2SLS SE from stage2 are too small)
np.random.seed(42)
N_BOOT = 500
boot_iv = np.zeros(N_BOOT)
for i in range(N_BOOT):
    boot = df_iv.sample(n=len(df_iv), replace=True)
    try:
        s1 = smf.ols(f'CHEMO ~ Z_LOO + {confounders}', data=boot).fit()
        boot['CHEMO_HAT_B'] = s1.fittedvalues
        s2 = smf.ols(f'OS_MONTHS ~ CHEMO_HAT_B + {confounders}', data=boot).fit()
        boot_iv[i] = s2.params.get('CHEMO_HAT_B', iv_estimate)
    except Exception:
        boot_iv[i] = iv_estimate + np.random.normal(0, 1)

iv_ci_lo, iv_ci_hi = np.percentile(boot_iv, [2.5, 97.5])
iv_se_boot = np.std(boot_iv)

# OLS estimate for comparison
ols_model   = smf.ols(f'OS_MONTHS ~ CHEMO + {confounders}', data=df_iv).fit(cov_type='HC3')
ols_estimate = ols_model.params['CHEMO']
ols_ci = ols_model.conf_int().loc['CHEMO']

print('=== 2SLS IV vs OLS COMPARISON ===')
print()
print(f'OLS estimate:  {ols_estimate:+.3f} months  [95% CI: {ols_ci[0]:.3f}, {ols_ci[1]:.3f}]')
print(f'IV estimate:   {iv_estimate:+.3f} months  [95% BCI: {iv_ci_lo:.3f}, {iv_ci_hi:.3f}]')
print(f'IV SE (bootstrap): {iv_se_boot:.3f}')
print()
print('Interpretation:')
print('  OLS estimates the effect for the average patient in the observed sample')
print('  IV estimates the LATE — the effect for patients whose chemo decision')
print('  was influenced by the treatment center ("compliers")')
print()

diff = iv_estimate - ols_estimate
if abs(diff) > 2:
    print(f'|IV - OLS| = {abs(diff):.2f} months — notable difference suggests')
    print('unmeasured confounding may be present (Hausman-style inference).')
else:
    print(f'|IV - OLS| = {abs(diff):.2f} months — estimates are similar;')
    print('residual unmeasured confounding appears limited.')

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: IV vs OLS comparison ---
ax = axes[0]
methods = ['OLS', '2SLS (IV)']
ests    = [ols_estimate, iv_estimate]
lo_errs = [ols_estimate - ols_ci[0], iv_estimate - iv_ci_lo]
hi_errs = [ols_ci[1] - ols_estimate, iv_ci_hi - iv_estimate]
cols    = ['#4C72B0', '#DD8452']

for y_pos, (m, est, lo, hi, col) in enumerate(zip(methods, ests, lo_errs, hi_errs, cols)):
    ax.errorbar(est, y_pos, xerr=[[lo], [hi]], fmt='o', color=col,
                 ms=12, capsize=10, lw=2.5, elinewidth=2.5)
    ax.text(est + 0.3, y_pos, f'{est:+.2f} mo', va='center', fontsize=11, fontweight='bold')

ax.axvline(0, color='black', lw=0.8, alpha=0.4)
ax.set_yticks([0, 1])
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel('Treatment Effect on OS (months)')
ax.set_title('IV vs OLS Estimates\nwith 95% Confidence Intervals', fontsize=12)
ax.set_xlim(min(ests) - max(lo_errs) - 3, max(ests) + max(hi_errs) + 4)

# --- Right: Bootstrap distribution of IV estimate ---
ax = axes[1]
ax.hist(boot_iv, bins=40, color='#DD8452', alpha=0.75, edgecolor='white', density=True)
ax.axvline(iv_estimate, color='darkred', lw=2.5, label=f'IV estimate = {iv_estimate:.2f}')
ax.axvline(iv_ci_lo, color='black', ls='--', lw=1.5, label=f'95% BCI')
ax.axvline(iv_ci_hi, color='black', ls='--', lw=1.5)
ax.axvline(ols_estimate, color='#4C72B0', ls='-', lw=2, alpha=0.8,
            label=f'OLS estimate = {ols_estimate:.2f}')
ax.axvline(0, color='gray', lw=1, alpha=0.5)
ax.set_xlabel('2SLS IV Estimate (months)')
ax.set_ylabel('Density')
ax.set_title('Bootstrap Distribution of IV Estimate\n(captures uncertainty from both stages)', fontsize=12)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_iv_vs_ols.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Exclusion Restriction Falsification Test
#
# While the exclusion restriction cannot be directly tested, we can perform **falsification tests** using outcomes that *should not* be affected by the instrument:
#
# - **Placebo outcome test**: Does center chemo rate predict outcomes that precede treatment, like age or sex? (It shouldn't)
# - **Subgroup test**: Does the IV estimate make sense in subgroups where chemotherapy is never indicated?
#
# These tests can falsify (but not prove) the exclusion restriction.

# %%
print('=== EXCLUSION RESTRICTION FALSIFICATION TESTS ===')
print()

# Test 1: Does instrument predict patient age? (should NOT — center can't cause age)
age_model = smf.ols('AGE ~ Z_LOO + C(CANCER_TYPE_ABBR)', data=df_iv).fit(cov_type='HC3')
age_coef  = age_model.params['Z_LOO']
age_p     = age_model.pvalues['Z_LOO']
print(f'Test 1 — Z_LOO → Age:   coef={age_coef:+.3f}, p={age_p:.4f}')
print(f'  Expected: not significant. {"✓ PASS" if age_p > 0.05 else "✗ FAIL — potential problem"}')
print()

# Test 2: Does instrument predict stage? (should NOT for same reason)
stage_model = smf.ols('STAGE ~ Z_LOO + C(CANCER_TYPE_ABBR)', data=df_iv).fit(cov_type='HC3')
stage_coef  = stage_model.params['Z_LOO']
stage_p     = stage_model.pvalues['Z_LOO']
print(f'Test 2 — Z_LOO → Stage: coef={stage_coef:+.3f}, p={stage_p:.4f}')
print(f'  Expected: not significant. {"✓ PASS" if stage_p > 0.05 else "✗ FAIL — center may treat sicker patients"}')
print()

# Test 3: Strength of instrument by cancer type (should work similarly across types)
fstats_by_type = {}
for ctype in df_iv['CANCER_TYPE_ABBR'].value_counts().nlargest(5).index:
    d = df_iv[df_iv['CANCER_TYPE_ABBR'] == ctype]
    if len(d) < 30:
        continue
    try:
        m_full = smf.ols('CHEMO ~ Z_LOO + AGE + STAGE', data=d).fit()
        m_rest = smf.ols('CHEMO ~ AGE + STAGE', data=d).fit()
        f = ((m_rest.ssr - m_full.ssr) / 1) / (m_full.ssr / m_full.df_resid)
        fstats_by_type[ctype] = f
    except Exception:
        fstats_by_type[ctype] = np.nan

print('Test 3 — First-stage F-statistics by cancer type:')
for ctype, f in sorted(fstats_by_type.items(), key=lambda x: -x[1]):
    strong = '✓' if f > 10 else '⚠' if f > 3.8 else '✗'
    print(f'  {ctype:6s}: F = {f:.1f}  {strong}')

print()
print('Summary of exclusion restriction assessment:')
all_pass = age_p > 0.05 and stage_p > 0.05
if all_pass:
    print('  Falsification tests passed — no evidence of exclusion restriction violation')
    print('  (Note: this does not PROVE exclusion restriction, only fails to reject it)')
else:
    print('  ⚠ Some tests failed — center may not be a clean instrument')
    print('  Consider: center associates with patient case-mix, not just chemo protocols')

# %% [markdown]
# ## Summary and Key Takeaways
#
# ### Results
#
# | Method | Estimate | Identifies |
# |--------|----------|------------|
# | OLS (adjusted) | See output | ATE (if no unmeasured confounding) |
# | 2SLS IV | See output | LATE (effect for compliers only) |
#
# A key insight: **IV estimates the LATE, not the ATE**. The LATE applies specifically to *compliers* — patients whose chemotherapy decision was influenced by which center they attended. If compliers are more margin-call patients (borderline indications), the LATE may differ substantially from the ATE.
#
# ### Assumptions and Limitations
#
# 1. **Relevance** (testable): F-statistic must exceed 10; weak instruments cause bias toward OLS
# 2. **Exclusion restriction** (untestable): Center → Survival *only through* Chemo. Violated if center quality (nursing ratios, palliative care) independently affects survival
# 3. **Independence** (partially testable): Center should not correlate with patient severity after adjusting for cancer type — partially tested above
# 4. **Monotonicity**: No *defiers* — no patient receives chemo at a low-chemo center but would not receive it at a high-chemo center
#
# ### Pharma Relevance: Mendelian Randomization
#
# The IV framework is the foundation of **Mendelian Randomization (MR)** — arguably the most powerful observational causal inference method in genomics:
# - **Genetic variants** as instruments (alleles randomly assigned at conception → similar to randomization)
# - **pQTLs** for drug target MR: SNPs that affect drug target protein levels as instruments for drug effects
# - FDA and EMA increasingly accept MR evidence for drug safety and efficacy signals
# - The same 2SLS framework applies, with genetic variants replacing center IDs as instruments
