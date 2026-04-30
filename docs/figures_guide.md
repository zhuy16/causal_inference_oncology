# Figure-by-Figure Interpretation Guide

All figures are saved in `results/figures/`. This guide explains what each figure shows, what a good result looks like, and common pitfalls.

---

## Notebook 00 — Survival Analysis Primer

### `00_data_overview.png`
**What it shows**: Three-panel overview — OS distribution histogram, event (death) rate by stage, patient counts per cancer type.

**How to read it**:
- **Left**: OS distribution — right-skewed, many patients with short follow-up (not necessarily dead; could be recently enrolled)
- **Middle**: Event rate by stage — should show a clear gradient: Stage I lowest mortality, Stage IV highest. This is the visual proof that stage is a strong confounder
- **Right**: Patient counts — shows which cancer types dominate the cohort (BRCA, LGG, KIRC are typically largest)

**What to note**: The steep event rate gradient across stages is the empirical foundation for the entire analysis. Any method that ignores stage will be heavily confounded.

---

### `00_km_curves.png`
**What it shows**: Two KM panels — left: naive chemo vs no-chemo comparison; right: KM by stage.

**How to read the left panel**:
- The naive chemo group will typically show *worse* or similar survival to no-chemo
- This is **indication bias** — the red warning box in the plot flags it explicitly
- Log-rank p-value: significant difference, but the direction is misleading

**How to read the right panel**:
- Four curves for Stage I–IV, clearly separated
- This is what the confounder *looks like*: a 30–40 month survival gap between Stage I and IV
- Any analysis that doesn’t stratify by stage is effectively mixing these four populations

**Key takeaway**: The left panel shows the wrong answer (naive); the right panel explains why it’s wrong (stage confounding).

---

### `00_km_by_cancer_type.png`
**What it shows**: 2×4 grid of individual KM curves for the 8 most common cancer types.

**How to read it**:
- Each panel = one cancer type, with n and median survival annotated
- Dashed horizontal line at S(t) = 0.5 shows where the median falls
- Wide confidence bands = small sample or high censoring

**What to note**: Enormous heterogeneity across cancer types. MESO (mesothelioma) and GBM (glioblastoma) have very poor prognosis; KIRC (kidney) and LGG (brain low-grade glioma) have much better outcomes. Pooling all cancer types without adjustment absorbs this variation as noise.

---

### `00_cox_comparison.png`
**What it shows**: Two side-by-side forest plots — unadjusted Cox (CHEMO only) vs adjusted Cox (CHEMO + AGE + STAGE).

**How to read it**:
- Each bar = hazard ratio with 95% CI; dashed line at HR=1
- Red bars = HR > 1 (increased hazard, worse survival); blue bars = HR < 1 (protective)
- **Unadjusted**: CHEMO HR > 1 (appears harmful) — indication bias
- **Adjusted**: CHEMO HR ≈ 1 (null after adjustment), while STAGE HR ≈ 1.6 (strong prognostic effect)

**Key insight**: The CHEMO coefficient flips from harmful to null once stage is controlled. This empirically demonstrates confounding and motivates the causal inference workflow.

---

### `00_cindex.png`
**What it shows**: Left — bar chart comparing C-index for unadjusted vs adjusted Cox model; right — summary table.

**How to read it**:
- **Unadjusted C ≈ 0.53**: A model with only CHEMO barely beats random. Chemo alone is not a good prognostic variable — because it’s assigned to the sickest patients
- **Adjusted C ≈ 0.69**: Adding Stage + Age gives good discrimination. These variables genuinely predict prognosis
- Reference lines at 0.5 (random) and 0.7 (good) provide context

**Why this matters**: C-index improvement from ~0.53 to ~0.69 quantifies the prognostic value of Stage and Age. These are not just statistical controls — they are the dominant drivers of survival in this dataset.

---

### `00_forest_plot.png`
**What it shows**: Per-cancer-type forest plot of adjusted Cox HRs for chemotherapy.

**How to read it**:
- Each row = one cancer type; point = HR, horizontal line = 95% CI; dashed line at HR=1
- Starred points (*) = p < 0.05; circles = non-significant
- Log scale on x-axis
- Sorted by HR (lowest at top)

**Good result**: Most cancer types cluster near HR=1 after adjustment, with wide CIs reflecting small per-type sample sizes. A few types may show significant beneficial or harmful associations — these are candidates for subgroup analysis.

**Note**: Per-type estimates are noisy (n=30–300 per type). The forest plot is exploratory, not confirmatory. Ridge penalisation (penalizer=0.1) is applied to stabilise estimates.

---

## Notebook 01 — DAG & Assumptions

> **These figures do not use real TCGA data and will not change when the dataset is rebuilt.** Notebook 01 is a *conceptual* notebook. The DAG is a theoretical construct; the bias demonstration uses fully simulated data with a hardcoded true ATE (`TRUE_ATE = 4.0`). This is by design — you can only prove that a method recovers the truth if you set the truth yourself. Real data has no ground truth to compare against.

### `01_causal_dag.png`
**What it shows**: The causal directed acyclic graph (DAG) of the system. Nodes are variables; arrows indicate direct causal relationships.

**How to read it**:
- Follow any path from Chemo to Survival — the *directed* paths (following arrow directions) are causal
- Any *undirected* path (going against an arrow) that connects Chemo to Survival through an unadjusted confounder is a *backdoor path*
- The highlighted adjustment set (Stage, Age, Cancer Type) blocks all backdoor paths

**What makes a good DAG**:
- All plausible confounders are included
- Mediators are correctly identified and not over-adjusted
- No cycles (no variable can cause itself through any chain of arrows)

**Common mistake to avoid**: Adjusting for TMB in the total effect model. Because Chemo → TMB → Survival is a causal path, controlling for TMB removes the indirect effect from your estimate. You would be left with only the direct cytotoxic effect — correct for the direct effect question, wrong for the total effect question.

---

### `01_indication_bias_demo.png`
**What it shows**: A simulation demonstrating what happens when you naively compare treated vs. untreated without adjustment.

**How to read it**:
- The naive comparison (ignoring stage) shows chemo patients have *worse* survival — paradoxical
- After stratifying by stage, within each stratum chemo patients do better
- This is Simpson's Paradox in action: the aggregate association reverses when you account for the confounder

**Key insight**: This figure is the entire motivation for everything that follows. If you understand why the unadjusted comparison is misleading, you understand why causal inference methods are necessary.

---

## Notebook 02 — Propensity Score Matching

### `02_propensity_score_dist.png`
**What it shows**: The distribution of propensity scores (PS) for treated (chemo) vs. control (no chemo) patients before matching.

**How to read it**:
- X-axis: propensity score (0 = certain to not receive chemo, 1 = certain to receive)
- Treated patients should skew right; control patients should skew left
- The **overlap region** (where both distributions coexist) is where matching is possible

**Good result**: Substantial overlap in the 0.2–0.8 range. If distributions barely overlap (e.g., treated all have PS > 0.8, control all have PS < 0.2), matching will discard most patients and the remaining sample may not be representative.

**Warning sign**: Bimodal distributions in one group suggest the PS model is misspecified or there are multiple distinct subpopulations that should be analysed separately.

---

### `02_love_plot.png`
**What it shows**: Standardised Mean Differences (SMD) for each covariate, before and after matching.

**How to read it**:
- Each row is one covariate (Age, Stage, cancer type dummies)
- Red circles = SMD before matching; green diamonds = SMD after
- The dashed vertical line at |SMD| = 0.1 is the balance threshold
- Arrows connecting pairs show improvement

**Good result**: All green diamonds to the left of the 0.1 threshold. The Love plot is the key diagnostic for PSM validity.

**What SMD means**:
- SMD = 0: treated and control groups have identical means on this variable
- SMD = 0.5: a half-standard-deviation difference — substantial imbalance
- Standardisation allows comparison across variables with different units (age in years vs. binary cancer type)

**Why SMD is better than p-values for balance**: With large samples, even tiny, clinically irrelevant differences will show p < 0.05. SMD focuses on the magnitude of difference, not its statistical significance.

---

### `02_common_support.png`
**What it shows**: Mirror (butterfly) plot of PS distributions before and after matching.

**How to read it**:
- Top half: treated patients
- Bottom half (reflected): control patients
- After matching: the two halves should look like mirror images

**Good result**: Near-identical shapes in the matched sample. The matched treated and control groups have the same PS distribution — they are balanced.

---

### `02_psm_results.png`
**What it shows**: Left panel — Kaplan-Meier survival curves for matched treated vs. control; Right panel — naive ATE vs. PSM ATE bar chart.

**How to read the KM plot**:
- Y-axis: probability of surviving to time t
- X-axis: time in months
- Shaded areas: 95% confidence bands
- Log-rank p-value: test of whether the two curves are statistically different
- The gap between curves at any time point = survival difference attributable to chemo (in the matched cohort)

**How to read the bar chart**:
- Naive ATE: raw difference in mean survival (biased by indication bias)
- PSM ATE: difference in matched cohort (confounder-adjusted)
- Error bar on PSM ATE = 95% confidence interval
- If PSM ATE > Naive ATE: indication bias was suppressing the true benefit (sicker patients getting chemo)

**Key interpretation**: The direction of the difference between naive and PSM estimates confirms the presence of indication bias and its direction.

---

### `02_ipw_comparison.png`
**What it shows**: Left — three-method Love plot (Unweighted vs PSM vs IPTW); Right — three-bar ATE comparison (Naive / PSM / IPTW) with bootstrap CIs.

**How to read the Love plot**:
- Three symbols per covariate: red circle (unweighted), green diamond (PSM), blue square (IPTW)
- Both PSM and IPTW should move all covariates left of the 0.1 threshold
- If one method achieves better balance than the other, it is preferred for that variable

**How to read the ATE bars**:
- All three estimates should show consistent direction (if the effect is real)
- **Naive**: confounded, typically shows chemo as harmful or neutral
- **PSM** and **IPTW**: both adjusted; should be close to each other
- Error bars = bootstrap 95% CI (1000 draws, seed 42)

**Interpreting PSM vs IPTW agreement**:
- Close agreement: PS model is well-specified; result is robust to the choice of weighting method
- Large disagreement: positivity violations (some patients have extreme PS near 0 or 1) or PS model misspecification — investigate further

**Regulatory context**: FDA RWE guidance recommends reporting both PS matching and weighting estimates as a sensitivity check. Consistency across methods is a key credibility criterion.

---

## Notebook 03 — Difference-in-Differences

### `03_parallel_trends.png`
**What it shows**: Mean survival by diagnosis year for treated cancer types (those affected by the guideline change) vs. control cancer types, before and after the policy year (~2010).

**How to read it**:
- Two lines: treated group and control group
- Vertical dashed line: policy year
- Pre-period: both lines should be trending roughly in parallel
- Post-period: the lines may diverge if the guideline change had an effect

**Good result**: Parallel pre-trends (lines roughly parallel before the dashed line). Divergence after = evidence of a treatment effect.

**Warning sign**: If the lines were already diverging before the policy year, the parallel trends assumption is violated and the DiD estimate is invalid. The event study figure is the formal test of this.

---

### `03_did_results.png`
**What it shows**: The DiD estimate — the change in treated group minus the change in control group — with confidence interval.

**How to read it**:
- The DiD coefficient (β_DiD) is the treatment effect estimate
- Positive value = guideline-increased chemo improved survival relative to the control group's trajectory
- Confidence interval: if it excludes zero, the effect is statistically significant

**Context**: DiD is an independent identification strategy from PSM. If both PSM (NB02) and DiD (NB03) point in the same direction, this cross-validation substantially strengthens confidence in the causal conclusion.

---

### `03_event_study.png`
**What it shows**: Year-by-year treatment effect estimates relative to the event year (year 0 = policy change).

**How to read it**:
- X-axis: years relative to policy change (negative = before, positive = after)
- Y-axis: treatment effect estimate for that year
- Coefficients before year 0 should be near zero (parallel trends)
- Coefficients after year 0 should become positive if treatment works

**Good result**:
- Pre-period coefficients hovering around zero (no pre-trend)
- Post-period coefficients positive and growing
- A clear "break" at year 0

**This is a credibility check**: Pre-trend non-zero = the parallel trends assumption is violated = the DiD estimate cannot be trusted. Many journals now require event study plots alongside DiD results.

---

## Notebook 04 — Mediation Analysis

### `04_mediation_analysis.png`
**What it shows**: Four panels — (top) path diagram with coefficients; (bottom-left) bootstrap distribution of indirect effect; (bottom-middle) effect decomposition bar chart; (bottom-right) TMB distribution by chemo status.

**How to read the path diagram**:
- a-path: Chemo → TMB coefficient (does chemo raise TMB?)
- b-path: TMB → Survival coefficient (does higher TMB improve survival, holding chemo fixed?)
- c'-path: Direct effect of Chemo on Survival (not through TMB)
- Indirect effect = a × b

**How to read the decomposition bar**:
- Three bars: Direct Effect, Indirect Effect, Total Effect
- Error bars: 95% bootstrap confidence intervals
- If the Indirect Effect bar's CI excludes zero → TMB mediates a significant portion of the benefit

**How to read the TMB distribution**:
- Chemo patients should have slightly higher TMB (a-path > 0)
- The separation between distributions visualises the a-path
- Since real TMB data is not available, this panel uses simulated TMB

**The key number**: Proportion Mediated = Indirect / Total. If this is 20–40%, TMB is a meaningful mechanistic contributor.

---

### `04_mediation_sensitivity.png`
**What it shows**: How the indirect effect estimate changes as the strength of unmeasured mediator-outcome confounding increases (parameterised by ρ).

**How to read it**:
- X-axis: ρ = correlation strength of a hypothetical unmeasured confounder with both TMB and Survival
- Y-axis: indirect effect estimate
- The orange vertical line (if present) = ρ where the indirect effect crosses zero (sign change)

**Good result**: The indirect effect remains statistically significant (away from zero) even at substantial ρ values (e.g., ρ = 0.3–0.4), meaning moderate levels of hidden confounding cannot explain away the mediation result.

---

## Notebook 05 — Instrumental Variables

### `05_stage1_instrument.png`
**What it shows**: Left — scatter plot of instrument (LOO center chemo rate) vs actual center chemo rate; Right — distribution of predicted chemo (Â) by actual treatment status.

**How to read the scatter plot**:
- Each point = one treatment center
- X-axis: leave-one-out center chemo rate (the instrument)
- Y-axis: actual center chemo rate
- Correlation = first-stage relevance
- F-statistic annotated in title — must be > 10 for valid IV

**How to read the distributions**:
- Treated patients should have higher predicted values from Stage 1
- Separation confirms the instrument predicts treatment assignment

**Warning sign**: F-statistic < 10 = weak instrument. This causes IV estimates to be biased toward OLS, negating the purpose of using IV.

---

### `05_iv_vs_ols.png`
**What it shows**: Left — IV vs OLS point estimates with 95% confidence intervals; Right — bootstrap distribution of IV estimate.

**How to read the comparison**:
- Each row = one method (OLS, 2SLS)
- Points = point estimates; horizontal lines = CIs
- OLS estimates the effect for the full sample (biased if unmeasured confounders exist)
- IV estimates the LATE for compliers only (unbiased for that subgroup if assumptions hold)

**Interpreting the gap**:
- IV ≈ OLS: Little evidence of unmeasured confounding — OLS is approximately valid
- IV > OLS: Negative selection — sicker patients (worse baseline survival) are being preferentially treated; OLS understates the benefit
- IV < OLS: Positive selection — healthier patients are being treated; OLS overstates the benefit

**Note on wide CIs for IV**: IV estimates typically have much wider CIs than OLS because they only use the instrument-driven variation (a small slice of all variation in treatment). This is the price paid for addressing unmeasured confounding.

---

## Notebook 06 — Sensitivity Analysis

### `06_evalue.png`
**What it shows**: E-value curve for the point estimate (and CI lower bound), with vertical lines marking the estimated effect sizes of known confounders.

**How to read it**:
- X-axis: risk ratio of the unmeasured confounder with treatment
- Y-axis: E-value (the corresponding risk ratio with outcome also required)
- Orange dashed line: E-value for the point estimate
- Red dashed line: E-value for the CI lower bound (more conservative)
- Gray vertical dotted lines: estimated RRs for known confounders (ECOG, comorbidity, SES)

**Good result**: The E-value lines are to the right of the known confounder vertical lines. This means: "No single known unmeasured confounder is strong enough to explain away the result."

**Critical insight**: The E-value for the CI bound is more important than for the point estimate. If the CI bound E-value > strongest known confounder, the finding is robust even accounting for sampling uncertainty.

---

### `06_rosenbaum_bounds.png`
**What it shows**: Left — p-value bounds as a function of Γ (unmeasured confounding sensitivity); Right — estimated Γ values for known confounders vs the critical Γ.

**How to read the p-value bounds plot**:
- X-axis: Γ (1 = no unmeasured confounding; 2 = 2-fold odds ratio allowed)
- Upper bound (red): worst-case p-value assuming confounders work maximally against our finding
- Lower bound (green): best-case p-value
- Black dashed horizontal line: p = 0.05 threshold
- Orange vertical line: critical Γ where upper bound crosses 0.05

**How to read the bar chart**:
- Each bar = estimated Γ for a known unmeasured confounder
- Red vertical line = critical Γ
- If all bars are shorter than the red line → no single known confounder is sufficient to invalidate the finding

**Interpreting critical Γ**:
- Γ < 1.5: Finding is sensitive to even small unmeasured confounders → low confidence
- 1.5 ≤ Γ < 2.5: Moderate robustness
- Γ ≥ 2.5: Strong robustness — substantial hidden bias required to overturn

---

### `06_specification_sensitivity.png`
**What it shows**: Left — forest plot of ATE estimates across all specification variants; Right — heatmap of ATE by matching ratio × caliper.

**How to read the forest plot**:
- Each row = one analytic specification (different PS model, caliper, or matching ratio)
- Point = ATE estimate; horizontal bar = 95% CI
- Green = significantly positive; red = significantly negative; gray = non-significant
- Orange dashed line = main analysis estimate

**Good result**: The vast majority of specifications show positive, statistically significant effects in a consistent direction. One or two outlier specifications are acceptable.

**How to read the heatmap**:
- Each cell = ATE for that combination of matching ratio and caliper
- Green = positive effect; red = negative (colormap centered at zero)
- Numbers in cells = ATE in months

**This answers**: "Are my results a statistical artifact of my specific modeling choices?" If results are consistent across the specification grid, the answer is no.

---

## Notebook 07 — Heterogeneous Treatment Effects

### `07_km_by_stage.png`
**What it shows**: 2×2 grid of KM curves — chemo vs no-chemo within each of Stage I–IV.

**How to read it**:
- Each panel = one stage; red = chemo, blue = no chemo
- The Δ median annotation shows the survival gap in months within that stage
- Log-rank p-value tests whether the within-stage curves differ

**Good result**: The gap between red and blue curves *grows* from Stage I to Stage IV — visually confirming that Stage is an effect modifier. Stage I may show little or no benefit; Stage IV should show the largest gap.

**Warning sign**: If the gap is similar across all stages, stage is not an effect modifier (but this is unlikely given the biological mechanism).

---

### `07_interaction_cox.png`
**What it shows**: Left — bar chart of chemo hazard ratios by stage (from the interaction model); Right — forest plot of all interaction model coefficients.

**How to read the HR-by-stage plot**:
- Each bar = chemo HR at that stage, with 95% CI
- HR < 1: chemo is protective at that stage; HR > 1: harmful (or insufficient data)
- If HR decreases monotonically from Stage I to IV: chemo becomes progressively more protective in advanced disease

**How to read the interaction coefficient**:
- `CHEMO_X_STAGE` coefficient < 0 (HR < 1): the protective effect of chemo *increases* with stage
- `*` annotation = p < 0.05 significant interaction

---

### `07_cate_distribution.png`
**What it shows**: Three panels — CATE distribution histogram, mean CATE by stage, mean CATE by age quartile.

**How to read the histogram**:
- Wide distribution = high heterogeneity; narrow = homogeneous effect
- Red dashed line = mean (≈ ATE from NB02)
- Patients left of zero = estimated to be harmed by chemo

**How to read the subgroup bars**:
- Error bars = 95% CI on the mean CATE within each subgroup
- Consistent gradient (Stage I < II < III < IV) confirms stage as effect modifier
- Age quartile plot: if Q4 (oldest) has lower CATE, age modifies the effect (toxicity trade-off)

---

### `07_hte_importance.png`
**What it shows**: Left — causal forest feature importances; Right — CATE waterfall (all patients sorted by CATE).

**How to read the feature importances**:
- Importance = how much that variable drives CATE variation across the forest
- High importance for AGE and STAGE confirms they are the primary effect modifiers
- High importance for a cancer type dummy = that type responds very differently to chemo

**How to read the waterfall**:
- Red bars = patients with positive CATE (benefit); blue bars = negative CATE (harm)
- Sorted left-to-right from lowest to highest CATE
- The fraction of blue bars indicates the proportion of patients estimated to be harmed

---

### `07_cate_landscape.png`
**What it shows**: Left — scatter of individual CATEs by stage (with jitter); Right — CATE vs Age scatter with trend line.

**How to read it**:
- Left: vertical spread within each stage = within-stage heterogeneity (driven by age, cancer type)
- Right: slope of trend line quantifies the age gradient in treatment benefit
- Colour in right plot = stage; allows visualising the joint Stage × Age effect

**Key insight**: Both plots together show that HTE is a *multidimensional* phenomenon — patients of the same stage but different ages have different CATEs. The causal forest captures this multi-way interaction; simple stratification cannot.
