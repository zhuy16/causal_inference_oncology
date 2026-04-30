# Causal Inference Concepts: A Practical Guide

This document explains the core ideas behind each method in the repo, with enough depth to understand the notebook outputs and discuss them in a technical context.

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
