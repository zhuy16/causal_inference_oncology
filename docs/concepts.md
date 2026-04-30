# Causal Inference Concepts: A Practical Guide

This document explains the core ideas behind each method in the repo, with enough depth to understand the notebook outputs and discuss them in a technical context.

---

## 0. Survival Analysis Fundamentals

Before applying causal inference methods, it is important to understand the outcome we are modelling: **time-to-event data** (overall survival).

### Why ordinary regression is insufficient for survival data:
Patients who are still alive at the end of the study are **censored** — we know they survived *at least* t months, but not how long they would ultimately live. Ignoring censoring (e.g., treating them as deaths) biases estimates downward; dropping them biases estimates upward. Survival methods handle censoring correctly.

### Kaplan-Meier Estimator:
The non-parametric estimate of the survival function S(t) = P(T > t):

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

where `d_i` = events at time `t_i`, `n_i` = patients at risk. KM curves make no distributional assumption and handle censoring exactly.

### Log-Rank Test:
Compares two KM curves under H₀: identical survival distributions. It is weighted toward late events (proportional hazards weighting). p-value interpretation is standard — but note that with large samples, even tiny differences become significant; always inspect the curves themselves.

### Cox Proportional Hazards Model:
Models the hazard (instantaneous event rate) as:

```
h(t | X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)
```

- `h₀(t)`: unspecified baseline hazard (semi-parametric — no distributional assumption)
- `exp(βᵢ)`: **hazard ratio** for covariate Xᵢ — the multiplicative change in hazard per unit increase
- HR > 1: higher risk (worse survival); HR < 1: lower risk (better survival)
- **Proportional hazards assumption**: the ratio of hazards between any two patients is constant over time

### C-index (Concordance Index):
Measures model discrimination — how well predicted risk scores rank patients by actual survival order:

> *Of all pairs where one patient died before the other, in what fraction did the model give the patient who died first a higher predicted risk?*

| C-index | Interpretation |
|---------|----------------|
| 0.5 | Random — no predictive value |
| 0.6–0.7 | Moderate discrimination |
| 0.7–0.8 | Good discrimination |
| > 0.8 | Excellent discrimination |

The C-index is the survival analogue of AUC-ROC. Adding Stage and Age to a chemo-only Cox model should raise C from ~0.53 to ~0.70, confirming that these variables carry genuine prognostic information.

### Why naive KM comparison is biased (indication bias):
Stage IV patients are both more likely to receive chemotherapy *and* more likely to die. A naive KM comparison of chemo vs no-chemo conflates treatment assignment with disease severity. The chemo group looks worse — not because chemo is harmful, but because it was disproportionately given to sicker patients. This is the entire motivation for the causal inference methods in NB01–06.

---

## 1. The Fundamental Problem of Causal Inference

The key challenge: **we can never observe both potential outcomes for the same person at the same time.**

A patient either receives chemo or they do not. We observe one outcome; the other is the *counterfactual* — what would have happened under the alternative treatment. This is the *fundamental problem of causal inference* (Holland, 1986).

We write:
- `Y(1)` = survival time *if* treated (whether or not they actually were)
- `Y(0)` = survival time *if* untreated

The **individual causal effect** = `Y(1) - Y(0)` — unobservable.

The **Average Treatment Effect (ATE)** = `E[Y(1) - Y(0)]` — estimable from data *if assumptions hold*.

**Why naive regression fails**: Regression compares patients who *chose* (or were assigned) different treatments. If the choice was influenced by factors that also affect the outcome (confounders), the regression estimate is biased.

---

## 2. Directed Acyclic Graphs (DAGs)

A DAG encodes causal assumptions as a graph:
- **Nodes** = variables
- **Directed edges** = direct causal effects
- **No cycles** = we cannot cause our own cause

### Three structural patterns every analyst must know:

**Chain (Mediator)**: `A → M → Y`
- M is on the causal path from A to Y
- Do NOT adjust for M when estimating the total effect of A
- Adjusting for M blocks the indirect pathway — you get only the direct effect

**Fork (Confounder)**: `A ← C → Y`
- C causes both treatment and outcome
- MUST adjust for C to remove the backdoor path
- Failing to adjust = biased estimate

**Collider**: `A → C ← Y`
- C is caused by both A and Y
- Do NOT adjust for C — conditioning on a collider *opens* a spurious path
- Example: adjusting for disease severity when it is caused by both the drug and genetics creates a spurious drug-genetics correlation

### The Backdoor Criterion

A set of variables **Z** satisfies the backdoor criterion for (A, Y) if:
1. Z blocks every backdoor path from A to Y
2. Z does not include any descendants of A

If you find such a Z, then controlling for Z gives you the causal effect: `P(Y | do(A)) = Σ_z P(Y | A, Z=z) P(Z=z)`

---

## 3. Propensity Score Matching (PSM)

The **propensity score** is `e(X) = P(A=1 | X)` — the probability of treatment given covariates.

**Rosenbaum-Rubin theorem**: If there is no unmeasured confounding, conditioning on `e(X)` is sufficient to remove all confounding from X. This is called *balancing property*.

**Why it works**: Two patients with the same propensity score but different treatment assignments are comparable — their treatment was essentially decided by chance (conditional on having the same predicted probability of treatment).

### PSM Steps:
1. Estimate `e(X)` using logistic regression
2. For each treated patient, find the control with the closest `e(X)` (nearest neighbor)
3. Apply caliper: discard matches where `|logit(e_t) - logit(e_c)| > 0.2 × SD(logit(e))`
4. Assess balance: compute SMD for each covariate
5. Estimate ATE: simple mean difference in matched cohort

### Key assumptions:
- **Unconfoundedness (ignorability)**: `Y(a) ⊥ A | X` — no unmeasured confounders
- **Overlap / positivity**: `0 < e(X) < 1` for all X — every patient has some probability of either treatment
- **SUTVA**: one patient's treatment doesn't affect another's outcome

### What the Love plot tells you:
- Standardized Mean Difference (SMD) = (mean_treated - mean_control) / pooled_SD
- SMD before matching: measures imbalance due to confounding
- SMD after matching: measures residual imbalance — should be < 0.1 for all variables
- If any variable has SMD > 0.1 after matching, the matched comparison is biased for that dimension

### Inverse Probability Weighting (IPTW):

IPTW is an alternative to PSM that *reweights* the full cohort rather than discarding unmatched patients.

Each patient receives a **stabilized weight**:

$$w_i^{\text{stab}} = \frac{T_i \cdot \hat{P}(T=1)}{\hat{e}(X_i)} + \frac{(1-T_i) \cdot \hat{P}(T=0)}{1 - \hat{e}(X_i)}$$

These weights create a *pseudo-population* where treatment is independent of confounders — mimicking a randomised trial.

**Hajek estimator** for the weighted ATE:
$$\hat{\tau}_{\text{IPW}} = \frac{\sum_i w_i T_i Y_i}{\sum_i w_i T_i} - \frac{\sum_i w_i (1-T_i) Y_i}{\sum_i w_i (1-T_i)}$$

**Trimming**: Extreme weights (near 0 or ∞, from near-zero or near-one PS) are trimmed at the 1st/99th percentile to reduce variance at the cost of slight bias.

| | PSM | IPTW |
|-|-----|------|
| Sample | Matched pairs only (some discarded) | Full cohort retained |
| Balance | Exact within matched pairs | Approximate across weighted sample |
| Efficiency | Lower (discards patients) | Higher (uses all data) |
| Sensitivity to PS model | Moderate | High (extreme weights amplify misspecification) |

If PSM ATE ≈ IPTW ATE, this cross-validation strengthens confidence in the result. Large disagreement indicates PS model sensitivity or positivity violations.

---

## 4. Difference-in-Differences (DiD)

DiD uses **temporal variation** rather than covariate matching to identify causal effects.

### The Setup:
- Treatment group: receives the intervention after time T
- Control group: never receives the intervention
- Observe both groups before and after T

### The Estimator:
```
DiD = E[Y_post - Y_pre | treated] - E[Y_post - Y_pre | control]
    = Treatment group trend       - Counterfactual trend
```

### Why it removes confounding:
Any confounder that is **time-invariant** (same before and after) is differenced out. This includes:
- Inherent disease aggressiveness of a cancer type
- Baseline demographics
- Institutional quality

### The Parallel Trends Assumption:
The crucial untestable assumption: *In the absence of treatment, the treated group would have followed the same trend as the control group.*

**How to test it**: Look at pre-period trends. If treated and control had parallel trends before the policy change, this lends credibility.

**The event study**: Instead of a single DiD coefficient, estimate separate effects for each time period relative to the event. Coefficients should be near zero pre-event and diverge post-event.

---

## 5. Mediation Analysis

Mediation decomposes the total causal effect into:
- **Natural Direct Effect (NDE)**: `E[Y(1, M(0))] - E[Y(0, M(0))]` — effect of treatment holding mediator fixed at its control value
- **Natural Indirect Effect (NIE)**: `E[Y(1, M(1))] - E[Y(1, M(0))]` — effect operating through the mediator
- **Total Effect**: `NDE + NIE`

### Baron-Kenny Three-Step Approach:
1. Regress Y on A + covariates → coefficient c (total effect)
2. Regress M on A + covariates → coefficient a (A→M path)
3. Regress Y on A + M + covariates → coefficients c' (direct) and b (M→Y path)
4. Indirect effect = a × b (product of coefficients)
5. Proportion mediated = a×b / c

### Why bootstrap for the indirect effect:
The product `a × b` does not follow a normal distribution, especially when effects are small. The Sobel test (delta method) underestimates uncertainty in many practical settings. Bootstrap percentile CIs are more reliable.

### The critical untestable assumption:
**No unmeasured mediator-outcome confounders**: There must be no variable that affects both TMB and survival that was not caused by chemotherapy. This is stronger than the treatment-outcome no-confounding assumption.

### Clinical interpretation:
- **High proportion mediated (>30%)**: TMB is a meaningful mechanistic node — blocking it would substantially reduce treatment benefit
- **Low proportion mediated (<10%)**: TMB may be a prognostic marker but the therapeutic benefit is primarily through direct cytotoxicity

---

## 6. Instrumental Variables (IV) and 2SLS

IV addresses unmeasured confounding by exploiting **exogenous variation** in treatment assignment.

### The three IV assumptions:
1. **Relevance**: Z affects A (testable — check F-statistic)
2. **Exclusion restriction**: Z affects Y *only through* A (untestable — requires domain argument)
3. **Independence**: Z is independent of unmeasured confounders (requires randomization or quasi-randomization)

### Two-Stage Least Squares (2SLS):
**Stage 1**: `A = π₀ + π₁Z + π₂X + ν` → predict Â
**Stage 2**: `Y = β₀ + β_IV·Â + γX + ε`

The key: Stage 2 only uses the **instrument-driven variation** in treatment. Any variation in A due to confounders is removed because Z is independent of confounders.

### What IV estimates — the LATE:
IV does not estimate the ATE. It estimates the **Local Average Treatment Effect (LATE)**: the causal effect for **compliers** — patients whose treatment was determined by the instrument.
- Always-takers (take chemo regardless of center) → not identified
- Never-takers (never take chemo regardless) → not identified
- Compliers (take chemo because they're at a high-chemo center) → identified

### Mendelian Randomization (MR):
The most powerful application of IV in biology. Genetic variants (SNPs) serve as instruments:
- Alleles are randomly assigned at conception (Mendel's laws → independence assumption)
- pQTLs (protein quantitative trait loci) instrument protein levels of drug targets
- MR replaces the target engagement experiment: if the SNP that lowers protein X also lowers disease Y, protein X is causally relevant

---

## 7. Sensitivity Analysis

No observational study is immune to unmeasured confounding. Sensitivity analysis quantifies robustness.

### E-value (VanderWeele & Ding, 2017):
For a risk ratio RR effect:
```
E-value = RR + sqrt(RR × (RR - 1))
```
Interpretation: An unmeasured confounder would need to have *at least* this risk ratio with both treatment and outcome to fully explain away the observed effect. Report both point estimate E-value and confidence interval bound E-value.

**Good E-values**: Greater than the strongest plausible confounder's effect size. For most clinical studies, E-value > 3 is considered robust.

### Rosenbaum Bounds:
For matched pairs, Γ parameterises how much two matched patients can differ in their unmeasured probability of treatment:
- Γ = 1: matched pairs are identical (standard PSM assumption)
- Γ = 2: one patient could be 2× more likely to be treated due to unmeasured factors

We find the **critical Γ**: the smallest value at which our p-value crosses 0.05. If Γ_critical > 2, substantial hidden bias is required to overturn the finding.

### Specification curve:
Run the same analysis across many reasonable analytic choices:
- PS model: logistic with C=0.1, 0.5, 1.0
- Matching ratio: 1:1, 1:2, 1:3
- Caliper: 0.1, 0.2, 0.3 SD

Plot all estimates. If the vast majority point in the same direction and are statistically significant, the finding is robust to analytic flexibility.

---

## 8. Heterogeneous Treatment Effects (HTE)

All previous methods estimate the **Average Treatment Effect (ATE)** — a single number for the whole population. HTE analysis asks: does the effect vary across subgroups?

### Key concepts:
- **CATE** (Conditional Average Treatment Effect): `E[Y(1) - Y(0) | X = x]` — the treatment effect as a function of patient characteristics X
- **Effect modifier**: a variable that *changes the magnitude or direction* of the treatment effect (distinct from a confounder, which biases the estimate)
- **Interaction term**: in a regression, `Chemo × Stage` tests whether Stage modifies the chemo effect

### Three approaches (in order of complexity):

**1. Stratified analysis**: fit the model separately within each subgroup (Stage I, II, III, IV) and compare estimates. Simple, interpretable, but loses power in small subgroups.

**2. Interaction term in Cox/regression**: include `Chemo × Stage` as a covariate. The interaction coefficient tests whether the effect differs across stages. Standard, parametric — assumes linearity.

**3. Causal Forest (Wager & Athey, 2018)**:
- Uses Double Machine Learning to partial out confounders: fit `Ỹ = Y - E[Y|X]` and `T̃ = T - E[T|X]`
- Grows a forest of trees that search for regions of X where T̃ and Ỹ have different correlations (i.e., where the treatment effect differs)
- Provides per-patient CATE estimates with honest confidence intervals (using sample splitting)
- **Feature importance** from the forest identifies which variables drive heterogeneity

### The CATE waterfall plot:
Sort all patients by their estimated CATE. If many patients have CATE < 0, there is a subgroup experiencing harm — a strong signal for treatment targeting.

### Caution:
- Individual CATEs have wide confidence intervals — they are most useful for *subgroup characterisation*, not individual-level prediction
- The causal forest assumes the same identification conditions as PSM (no unmeasured confounders)
- Survival outcomes require special handling (RMST pseudo-observations or weighted estimators) when using continuous-outcome CATE methods

### Clinical application:
If Stage IV patients benefit 7× more than Stage I, an **enrichment trial design** — enrolling only Stage III–IV — maximises power and reduces unnecessary toxicity in low-benefit patients. This is the precision oncology paradigm.
