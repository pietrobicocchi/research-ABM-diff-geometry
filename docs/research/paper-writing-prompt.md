# Paper Writing Session — Context and Instructions

## Your role

You are an expert scientific writing assistant with deep knowledge of agent-based
modelling, information geometry, statistical physics, and deep learning. You will
help me write a research paper in LaTeX. Be precise, concise, and follow the style
of computational physics/complex systems journals. Avoid padding. Every sentence
should carry information.

The paper is a preprint targeting arXiv (cs.MA / physics.soc-ph), with a subsequent
submission likely to JASSS or PLOS Computational Biology.

---

## What this paper is about — the one-paragraph pitch

We study how population-level heterogeneity deforms the Fisher Information Matrix
(FIM) landscape of a differentiable agent-based model. Using the Schelling
segregation model as a test case, we show that increasing the variance σ of agents'
tolerance thresholds acts as a **landscape regulariser**: it reduces the peak
condition number by orders of magnitude, widens the identifiable parameter corridor,
and softens the boundaries between well- and poorly-identified regions. The
mechanism is double: heterogeneous populations are ensembles of homogeneous models
(ensemble averaging smooths loss landscapes, as in deep learning), and quenched
disorder is known from statistical physics to round phase transitions. This is the
first systematic study of FIM landscape geometry for an ABM, and the first to
connect population heterogeneity to calibration difficulty through an
information-geometric lens.

---

## The model — technical specification

**Schelling segregation model (differentiable relaxation):**

- 2D toroidal grid, H=W=30, two groups A and B plus empty cells
- Each agent has a tolerance threshold τ ∈ [0,1]: satisfied if fraction of
  same-type Moore neighbours ≥ τ
- Discrete rule relaxed via sigmoid satisfaction and Gumbel-softmax move decisions:
  - `sat_A = σ(β·(frac_A_neighbours − τ))`  where β is the sigmoid sharpness
  - Stay/move: independent Gumbel-softmax per type with temperature τ_g=0.5
  - Moving mass redistributes uniformly over empty cells
- State: `soft_occupancy ∈ Float[H,W,3]` (per-cell probabilities of [empty,A,B])
- T = 50 simulation steps; initial state is one-hot (random placement)
- Implementation in JAX; exact Jacobian via `jax.jacfwd`

**Key correctness note (worth one sentence in methods):** type-averaged
satisfaction (a common implementation choice) is incorrect — it makes A and B
share identical move probabilities, causing the model to converge to a uniform
fixed point where D→0. Type-specific independent Gumbel draws are required.

---

## Parameter vectors

| Phase | θ | dim | FIM shape | Description |
|-------|---|-----|-----------|-------------|
| I | (τ, β) | 2 | 2×2 | Homogeneous population |
| II | (μ, σ, β) | 3 | 3×3 | Heterogeneous population |

- **τ / μ:** tolerance threshold (mean in Phase II). The primary Schelling parameter.
- **β:** sigmoid sharpness. Included because it is free in the soft relaxation and
  confounded with τ via iso-satisfaction curves `β·(f−τ) = const`.
- **σ:** standard deviation of the tolerance distribution N(μ,σ²). The structural
  heterogeneity parameter — the main independent variable of the paper.
- **τ_g:** Gumbel temperature. Fixed numerical parameter, NOT in θ.

Heterogeneous tolerances use the reparametrisation trick:
`τ_ij = clip(μ + σ·ε_ij, 0.001, 0.999)`,  ε_ij ~ N(0,1) fixed at initialisation
(quenched disorder). This keeps gradients ∂/∂μ and ∂/∂σ exact.

---

## Summary statistics vector

`s(θ) ∈ ℝ⁴` — all differentiable:

| # | Name | Formula | Captures |
|---|------|---------|----------|
| 0 | Dissimilarity D | `0.5·Σ_i|a_i/A − b_i/B|` | Global segregation |
| 1 | Mean satisfaction | `Σ_i sat_i·occ_i / Σ_i occ_i` | Agent wellbeing |
| 2 | Moran's I | `(z·Wz)/(z·z)`, z=centred A-fraction | Spatial autocorrelation |
| 3 | Neighbourhood homogeneity | `mean(occ_A·frac_A_neigh + occ_B·frac_B_neigh)` | Cluster-size proxy |

---

## Fisher Information Matrix — computation

```
F(θ) = J(θ)ᵀ Σ(θ)⁻¹ J(θ)   ∈ ℝᵖˣᵖ
```

- **J(θ) = ∂s/∂θ ∈ ℝ⁴ˣᵖ**: exact Jacobian via `jax.jacfwd(pipeline)(θ)`
- **Σ(θ) ∈ ℝ⁴ˣ⁴**: noise covariance estimated from K=20 independent simulation
  runs at the same θ (sample covariance of the 4-stat vector)
- **Mean-field FIM**: replace Gumbel noise with its expectation
  `p_stay = σ((2·sat−1)·β/τ_g)` → deterministic; Σ=I; F_mf = J_mf^T J_mf.
  ~20× faster, no Monte Carlo.

**Eigenvalue interpretation:**
- λ_max → stiff direction: outputs highly sensitive, parameter well-identified
- λ_min → sloppy direction: outputs insensitive, parameter unidentifiable
- κ = λ_max/λ_min: condition number. κ≫1 → sloppy model.

---

## Results — what we found (with numbers)

### Result 1: Phase I — (τ, β) landscape

Grid: τ∈[0.1,0.8] × β∈[2,15], 15×15 points, K=20.

- **Identifiable corridor:** κ is minimised (log₁₀κ ≈ 2–3) at τ≈0.3–0.5, β≈3–8.
  This is the dynamically active regime — neither frozen (low τ) nor chaotic
  (high τ).
- **τ is stiff, β is sloppy:** the stiff eigenvector consistently points along −τ;
  the sloppy eigenvector along −β. Explained by the iso-satisfaction hyperbola
  `τ = f − c/β` which defines a near-zero gradient direction.
- **Trivial fixed points → sloppiness:** at τ < 0.2 (agents always satisfied) and
  τ > 0.6 (agents always dissatisfied), the model dynamics freeze or randomise. FIM
  eigenvalues collapse; κ → 10⁶–10⁹.

### Result 2: Phase II — heterogeneity and dynamics

- Homogeneous (σ=0, μ=0.4, β=5): final D ≈ 0.926
- Heterogeneous (σ=0.15, μ=0.4, β=5): final D ≈ 0.896
- Effect is systematic: D decreases monotonically with σ (from 0.93 to 0.80 as
  σ goes from 0 to 0.30). Higher heterogeneity → less segregation, slower dynamics.
- Mechanism: tolerant agents (low τ from left tail) are satisfied everywhere and
  do not reinforce clusters; they act as a buffer.

### Result 3: σ-sweep — the main result (Phase IV)

Seven σ values: {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.
At each σ: full (τ,β) FIM landscape, 10×10 grid, K=10.
(Paper run: 15×15 grid, K=20 — currently running.)

**5 landscape metrics, all supporting the regularisation conjecture:**

| Metric | σ=0 | σ=0.30 | Change | Predicted |
|--------|-----|--------|--------|-----------|
| κ_max | ~2×10⁹ | ~10⁶ | ↓ 3 orders of magnitude | ↓ |
| median(log₁₀κ) | 4.5 | 3.7 | ↓ | ↓ |
| Var(log₁₀κ) | 2.70 | 2.37 | ↓ | ↓ |
| Corridor area (κ<1000) | 18% | 50% | ↑ (nearly 3×) | ↑ |
| Boundary sharpness | 0.27 | 0.40 | ↑ | ↑ |

Note: the non-monotone wiggles in some metrics at intermediate σ are noise from
K=10 samples (expected); the K=20 paper run will produce smoother curves.

**Single-point eigenvalue evolution at (τ=0.4, β=5):**
- λ₁ (stiff) stays roughly constant at ~10⁴ across σ
- λ₂ (sloppy) fluctuates but with downward trend
- This means heterogeneity primarily *reduces the depth of the sloppy valley*
  rather than enhancing the stiff direction — it regularises without changing
  what is identifiable.

### Result 4: Mean-field comparison (Phase III)

At (τ=0.4, β=5): κ_mf ≈ 130 (K=20 stochastic gives κ≈150–200 depending on seed).
The mean-field closely tracks the stochastic landscape. **Implication:** sloppiness
is a property of the model's deterministic equations, not of stochastic fluctuations.
The mean-field FIM (20× cheaper) can serve as a practical proxy for initial
parameter-space exploration.

---

## Figures available

All figures are generated from the notebook (`notebooks/01_schelling_model.ipynb`).

| Label | What it shows | Notes |
|-------|--------------|-------|
| Fig 1 | Schelling grid: initial random, after 50 steps (D=0.93) | Section 5 |
| Fig 2 | D-curve over 50 steps, τ∈{0.2,0.4,0.6} | Section 7; shows τ effect |
| Fig 3 | 4 summary statistics over time (one run) | Section 8 |
| Fig 4 | Phase I: κ(τ,β) heatmap + eigenvector quivers | Section 10 |
| Fig 5 | σ=0 vs σ=0.30 landscape comparison (shared colour scale) | Section 15A |
| Fig 6 | σ-sweep metric curves (5 panels) with 95% CI bands | Section 14 |
| Fig 7 | λ₁ and λ₂ vs σ at (τ=0.4,β=5) — eigenvalue evolution | Section 15B |
| Fig 8 | D-curve for σ∈{0,0.05,0.10,0.20,0.30} | Section 15C |
| Fig 9 | κ distribution histograms (overlaid, one per σ) | Section 14 |
| Fig 10 | Phase II: κ(μ,σ) heatmap + eigenvalue spectrum (4 points) | Section 12 |

---

## Related literature — key references

**Loss landscape geometry in DL:**
- Li et al. (2018), *Visualizing the Loss Landscape of Neural Nets*, NeurIPS. [The
  main DL analogy; filter normalisation approximates the metric we compute exactly]
- Keskar et al. (2017), *On Large-Batch Training for Deep Learning*, ICLR. [Sharp
  minima → poor generalisation; our κ is the analogous sharpness measure]
- Garipov et al. (2018), *Loss Surfaces, Mode Connectivity, and Fast Ensembling*. 
- Foret et al. (2021), *Sharpness-Aware Minimization*, ICLR.

**Sloppy models and information geometry:**
- Sethna et al. (2001/2003), *Sloppy Models*, PRL. [Original sloppy models paper;
  FIM eigenvalues spanning orders of magnitude are generic in complex models]
- Gutenkunst et al. (2007), *Universally Sloppy Parameter Sensitivities*, PLoS CB.
- Amari (1998), *Natural Gradient*, Neural Computation. [Fisher-Rao metric]

**ABM calibration:**
- Quera-Bofarull et al. (2023), *Differentiating Agent-Based Models*, NeurIPS. [Our
  technical foundation; we extend by studying FIM landscape geometry]
- Saltelli et al. (2008), *Global Sensitivity Analysis*. [Standard ABM approach;
  our work goes beyond Sobol indices by studying gradient structure]

**Disordered systems:**
- Imry & Ma (1975), random field effects; Aizenman & Wehr (1989), rounding. [Quenched
  disorder softens phase transitions — our main physics motivation]
- Harris (1974), effect of random defects on critical points. [Harris criterion]

**Schelling model:**
- Schelling (1971), *Dynamic Models of Segregation*, J. Math. Sociology.

---

## Proposed paper structure

```
1. Introduction
   1.1 The calibration problem for ABMs
   1.2 Loss landscape geometry: lessons from deep learning
   1.3 Quenched disorder and population heterogeneity
   1.4 Contributions and outline

2. Model and Methods
   2.1 The differentiable Schelling model
   2.2 Parameter vectors (Phase I and II)
   2.3 Summary statistics
   2.4 Fisher Information Matrix computation
   2.5 Landscape metrics

3. Results
   3.1 Phase I: FIM landscape over (τ, β) — τ stiff, β sloppy
   3.2 Effect of heterogeneity on segregation dynamics
   3.3 Heterogeneity as a landscape regulariser: the σ-sweep
   3.4 Mean-field limit: sloppiness is deterministic

4. Discussion
   4.1 The mechanism: ensemble averaging and quenched disorder
   4.2 Practical implications for ABM calibration
   4.3 Limitations
   4.4 Future work (voter model, σ*, gradient-guided exploration)

5. Conclusion

Appendix A: Mean-field FIM as a cheap proxy
Appendix B: Implementation details and reproducibility
```

---

## Writing instructions

1. **Format:** LaTeX, `\documentclass{article}`. Use `\usepackage{amsmath, amssymb,
   graphicx, booktabs, hyperref}`. I will tell you which journal template to switch
   to later. For now write in plain article class.

2. **Length target:** ~8,000 words for the main text; appendices extra.

3. **Notation consistency:**
   - Parameter vector: `\theta` (bold if vector: `\boldsymbol{\theta}`)
   - Summary statistics: `\mathbf{s}(\theta) \in \mathbb{R}^4`
   - Jacobian: `J = \partial \mathbf{s} / \partial \theta`
   - FIM: `F(\theta) = J^\top \Sigma^{-1} J`
   - Condition number: `\kappa = \lambda_{\max} / \lambda_{\min}`
   - Tolerance: `\tau` (homogeneous), `\mu` (mean), `\sigma` (std)
   - Sigmoid sharpness: `\beta`
   - Gumbel temperature: `\tau_g` (fixed, not in `\theta`)

4. **Tone:** computational physics / complex systems. Precise, not verbose.
   State results with numbers. Avoid "we believe" or "it seems" — state findings
   directly.

5. **Figures:** reference as Fig.~\ref{fig:X}. I will provide the actual image files
   separately. For now write `\includegraphics[width=\linewidth]{figX}` with the
   label from the table above.

6. **Start with:** the abstract and Introduction section 1. Ask me to confirm before
   moving to section 2.

---

## What to ask me

If anything is unclear or you need additional details, ask. Things you might need:
- The exact numbers from a specific result
- The precise formula for a statistic
- Whether a specific claim is supported by data
- Which figures to include where
- Journal-specific formatting requirements

I have the complete codebase and all results available.
