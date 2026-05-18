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

## Core novelty — what is genuinely new

**Be explicit about this in the introduction. These are the contributions in order of
strength:**

1. **First exact FIM landscape for any ABM.** Existing ABM calibration uses
   Sobol/variance-based sensitivity analysis (gradient-free, scalar, cannot detect
   confounding). We compute the full Fisher Information Matrix exactly via
   automatic differentiation through a differentiable ABM — giving the complete
   geometry of parameter space, not just variance attribution. No prior ABM paper
   has done this.

2. **First demonstration that spatial ABMs are generically sloppy.** The eigenvalue
   spectrum of the FIM spans 4–9 orders of magnitude across the parameter space,
   confirming that ABMs belong to the "sloppy models" class (Sethna et al. 2003).
   This has direct implications for calibration: most parameter combinations are
   effectively unidentifiable regardless of data quality.

3. **Population heterogeneity acts as a landscape regulariser.** Increasing the
   variance σ of agent tolerance distributions flattens the FIM landscape, reduces
   the peak condition number, and widens the identifiable parameter corridor. This
   is the first connection between structural population diversity and information-
   geometric calibration difficulty in any ABM.

4. **Mean-field sloppiness is deterministic, not stochastic.** The mean-field FIM
   (noise-free, Σ=I) closely matches the stochastic FIM at the same parameter point.
   Sloppiness is a property of the model's equations, not of simulation noise.

5. **A correctness result for differentiable ABMs.** Type-averaged satisfaction
   (a natural but incorrect implementation) causes the model to converge to a
   uniform fixed point (D→0). Type-specific independent Gumbel draws are required
   for the segregation mechanism to function. This is a non-trivial finding for
   anyone implementing differentiable ABMs.

---

## What this paper is about — the one-paragraph pitch

We study how population-level heterogeneity deforms the Fisher Information Matrix
(FIM) landscape of a differentiable agent-based model. Using the Schelling
segregation model as a test case — the first ABM for which an exact FIM landscape
has been computed — we show that spatial ABMs are generically sloppy: the FIM
eigenspectrum spans up to nine orders of magnitude, with only a narrow corridor
of parameter space that is well-identified. Increasing the variance σ of agents'
tolerance thresholds acts as a landscape regulariser: the peak condition number
drops by three orders of magnitude and the identifiable corridor nearly doubles
in area as σ increases from 0 to 0.30. The mechanism is two-fold: heterogeneous
populations are ensembles of models (ensemble averaging smooths loss landscapes,
as in deep learning), and quenched disorder softens dynamical phase transitions
(a statistical physics result). This is the first systematic study of FIM geometry
for an ABM and the first to connect population heterogeneity to calibration
difficulty through an information-geometric lens.

---

## The model — technical specification

**Schelling segregation model (differentiable relaxation):**

- 2D toroidal grid, H=W=30, two groups A and B plus empty cells, density=0.8
- Each agent has a tolerance threshold τ ∈ [0,1]: satisfied if fraction of
  same-type Moore neighbours ≥ τ
- Discrete rule relaxed via sigmoid satisfaction and Gumbel-softmax move decisions:
  - `sat_A = σ(β·(frac_A_neighbours − τ))` where β is the sigmoid sharpness
  - Stay/move: **type-specific independent** Gumbel-softmax per type, temperature τ_g=0.5
  - Moving mass redistributes uniformly over empty cells (uniform redistribution
    maintains the per-cell simplex constraint exactly)
- State: `soft_occupancy ∈ Float[H,W,3]` (per-cell probabilities of [empty,A,B])
- T = 50 simulation steps; initial state is one-hot (random placement, D=1)
- Implementation in JAX; exact Jacobian via `jax.jacfwd`

**Key correctness finding (one sentence in methods + one sentence in discussion):**
Type-averaged satisfaction (pooling A and B into a single p_move) makes the
uniform redistribution step a mean-field fixed point — D converges to 0 regardless
of τ. Type-specific independent Gumbel draws are the minimal correction. This is
a structural result about differentiable relaxations of threshold-based ABMs.

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
  heterogeneity parameter and main independent variable of the paper.
- **τ_g:** Gumbel temperature. Fixed at 0.5. NOT in θ — it is a numerical parameter
  of the relaxation, not a scientific parameter of the Schelling model.

**Heterogeneous tolerances use the reparametrisation trick:**

```
τ_ij = clip(μ + σ·ε_ij,  0.001, 0.999)
ε_ij ~ N(0,1),  sampled once at initialisation  (quenched disorder)
```

**Critical methodological point:** the same ε matrix is used across all (τ,β)
evaluations within a given σ value, and the same ε (scaled) is used across all
seven σ values. This is a deliberate choice: it isolates the effect of heterogeneity
*magnitude* from heterogeneity *pattern*, making σ the sole independent variable.
This must be stated clearly in §2.2.

Limitation to acknowledge: results are conditional on one disorder realisation.
A robustness check (repeat with 3–5 different ε draws) should be done before
submission. The K=20 results show the trend persists across the full grid, which
is strong evidence the trend is not ε-specific, but a formal check is warranted.

---

## Summary statistics vector

`s(θ) ∈ ℝ⁴` — all differentiable:

| # | Name | Formula | Captures |
|---|------|---------|----------|
| 0 | Dissimilarity D | `0.5·Σ_i|a_i/A − b_i/B|` | Global segregation |
| 1 | Mean satisfaction | `Σ_i sat_i·occ_i / Σ_i occ_i` | Agent wellbeing |
| 2 | Moran's I | `(z·Wz)/(z·z)`, z=centred A-fraction, W=Moore adjacency/8 | Spatial autocorrelation |
| 3 | Neighbourhood homogeneity | `mean(occ_A·frac_A_neigh + occ_B·frac_B_neigh)` | Cluster-size proxy |

---

## Fisher Information Matrix — computation

```
F(θ) = J(θ)ᵀ Σ(θ)⁻¹ J(θ)   ∈ ℝᵖˣᵖ
```

- **J(θ) = ∂s/∂θ ∈ ℝ⁴ˣᵖ**: exact via `jax.jacfwd(pipeline)(θ)` — forward-mode
  autodiff through T=50 steps of lax.scan
- **Σ(θ) ∈ ℝ⁴ˣ⁴**: sample covariance of s(θ) over K=20 independent runs at
  fixed θ (grid: 15×15 parameter points for the σ-sweep)
- **Mean-field FIM**: replace Gumbel noise with `E[noise]=0`, giving deterministic
  step `p_stay = σ((2·sat−1)·β/τ_g)`. Then Σ=I, so F_mf = J_mf^T J_mf.
  ~20× faster per point; no Monte Carlo needed.

**Eigenvalue interpretation:**
- λ_max → stiff direction: parameter combination well-identified
- λ_min → sloppy direction: parameter combination unidentifiable
- κ = λ_max/λ_min: condition number. κ≫1 → sloppy model.

---

## Landscape metrics — formal definitions for §2.5

Five scalar metrics computed from a κ(τ,β) grid of shape (N,N):

1. **κ_max** = max_{i,j} κ(τ_i, β_j)

2. **median(log₁₀ κ)** = median of {log₁₀ κ(τ_i,β_j)} over all grid cells

3. **Var(log₁₀ κ)** = variance of {log₁₀ κ(τ_i,β_j)} — spread of the landscape;
   decreasing → landscape more uniform

4. **Corridor area** = fraction of grid cells with κ < 1000 (the identifiable
   corridor). **The threshold 1000 is empirical, not derived from first
   principles.** Justify as: at κ<1000, standard gradient-based calibration
   (e.g., Adam, L-BFGS) converges reliably; at κ>10⁴ the condition number
   exceeds typical numerical tolerances. Sensitivity check: repeat with κ<100
   and κ<10,000 to confirm directional robustness.

5. **Boundary sharpness** = mean over β-columns of the width (in τ units) of the
   transition zone from κ < 100 to κ > 10,000. Formally: for column j,
   let τ_lo(j) = sup{τ : κ(τ,β_j) < 100} and τ_hi(j) = inf{τ : κ(τ,β_j) > 10,000}.
   Width_j = τ_hi(j) − τ_lo(j) if both exist, else undefined. Boundary sharpness =
   mean over valid j. Larger value = softer/more gradual transition.

**Confidence intervals:** 95% CI via spatial bootstrap — resample grid cells with
replacement 500 times, recompute each metric. This estimates sensitivity to the
choice of parameter grid points, not FIM estimation noise. State this limitation
explicitly.

---

## Results — K=20 paper run (15×15 grid)

### Result 1: Phase I — (τ, β) landscape — exact numbers from K=20 run needed

*(The K=20 Phase I landscape was run earlier at 10×10. The σ-sweep below uses
15×15. Report Phase I from the 10×10 run in the paper; flag as preliminary and
note the 15×15 σ=0 column gives the equivalent.)*

Key findings:
- Identifiable corridor at τ≈0.3–0.5, β≈3–8; log₁₀κ ≈ 2–3 there
- τ is stiff (stiff eigenvector points along −τ consistently)
- β is sloppy (sloppy eigenvector points along −β) — explained by the confounding
  iso-satisfaction curve τ = f − c/β
- At trivial fixed points (τ<0.2 or τ>0.6): κ → 10⁶–10⁹

### Result 2: Effect of σ on segregation dynamics (K=20, confirmed)

From Fig 8 (D-curve for 5 σ values — the cleanest result in the paper):
- σ=0.00: final D ≈ 0.93 (step 50)
- σ=0.05: ≈ 0.90
- σ=0.10: ≈ 0.88
- σ=0.20: ≈ 0.81
- σ=0.30: ≈ 0.80

**Perfect monotone ordering, robust result.** Higher heterogeneity → less
segregation, slower recovery from the initial mixing dip. Mechanism: tolerant
agents (left tail of N(μ,σ²)) are satisfied anywhere and do not reinforce clusters.

### Result 3: σ-sweep — the main result (K=20, 15×15 grid)

**Metrics from σ=0.00 to σ=0.30 — directions confirmed, exact numbers to be
updated from the printed notebook output:**

| Metric | σ=0 (approx) | σ=0.30 (approx) | Direction | CI bands |
|--------|-------------|-----------------|-----------|----------|
| κ_max | ~1.5×10⁸ | ~10⁶ | ↓ | Wide; spike at σ=0.05 |
| median(log₁₀κ) | ~4.4 | ~3.8 | ↓ ✓ clean | Moderate; clearest metric |
| Var(log₁₀κ) | ~2.1 | ~1.75 | ↓ | Wide; net decrease |
| Corridor area | ~0.24 | ~0.27 | ↑ | Wide; non-monotone |
| Boundary sharpness | ~0.22 | ~0.37 | ↑ | Very wide; noisy |

**What the K=20 results clarify compared to K=10:**

- The **σ=0.05 spike in κ_max** persists at K=20. It is not sampling noise — it
  reflects a single grid point with extremely high κ that σ=0.05 fails to suppress.
  At σ=0.10, that point is regularised. Interpretation: tiny heterogeneity can
  initially *concentrate* rather than *diffuse* the sloppy spike.

- The **corridor area and boundary sharpness are noisier than expected** at K=20.
  The spatial bootstrap CI bands are wide, and the metrics are non-monotone. These
  metrics should be presented as *suggestive evidence* rather than definitive results.

- **median(log₁₀κ)** is the most robust metric — clean downward trend, moderate
  CI, clear from the histogram as well (the distribution shifts left as σ increases).

- **κ distribution histogram** (Fig 9) is perhaps the most visually convincing
  evidence: the σ=0.30 distribution (yellow-green) has substantially more density
  at low log₁₀(κ) (1–3) and much less at the high end (6–8). Present this alongside
  the metric curves.

**What to claim in the paper:** "Three metrics (median κ, κ distribution, corridor
area) robustly support the regularisation conjecture; two (boundary sharpness, κ_max)
show the expected direction but with high variability." Do NOT claim "5/5 metrics
uniformly support the conjecture" — that overstates the K=20 evidence.

### Result 4: Eigenvalue evolution at fixed point (τ=0.4, β=5) — important nuance

From Fig 7 (eigenvalue evolution, confirmed at K=20):
- λ₁ (stiff): roughly constant at ~10⁴–2×10⁴ across σ
- λ₂ (sloppy): fluctuates 100–150, with sharp collapse to ~4 at σ=0.20
- κ: spike from ~150 to ~5000+ at σ=0.20, back to ~150 at σ=0.30

The σ=0.20 spike is **reproducible at K=20** — confirming it is a real feature of
the specific (τ=0.4, β=5, ε) combination, not sampling noise. The quenched
disorder pattern ε at σ=0.20 creates a tolerance configuration that happens to
make the FIM nearly singular at this exact point.

**Implication for the paper:** single-point FIM measurements are unreliable as
proxies for landscape geometry. The landscape-averaged metrics (corridor area,
median κ) are more robust. This finding supports the methodological choice to
characterise the full landscape rather than individual points — which is itself
a contribution of the paper.

The eigenvalue evolution also reveals *how* regularisation works: λ₁ stays roughly
constant (the stiff direction is robust to heterogeneity), while λ₂ shows an
overall upward trend (the sloppy direction stiffens). Heterogeneity lifts the
sloppy floor rather than depressing the stiff ceiling.

### Result 5: Mean-field comparison

κ_mf ≈ 130 at (τ=0.4, β=5) vs κ_sim ≈ 150–200. The mean-field closely tracks the
stochastic FIM at this point. Implication: sloppiness is a deterministic property
of the model equations. The mean-field landscape (~20× cheaper) is a practical
proxy for initial exploration.

---

## Answers to open methodological questions

**Q1: Is the κ<1000 corridor threshold principled?**
No — it is conventional (moderately ill-conditioned in numerical analysis terms).
State this explicitly. Add one panel in the supplementary showing corridor area at
κ<100 and κ<10,000; confirm the σ trend direction is robust.

**Q2: Boundary sharpness formula — formal definition:**
For each β-column j: τ_lo(j) = last τ where κ(τ,β_j) < 100; τ_hi(j) = first τ
where κ(τ,β_j) > 10,000. Sharpness = mean_j(τ_hi(j) − τ_lo(j)) over columns where
both exist; 0 if none found. Give this as a display equation in §2.5.

**Q3: K=10 vs K=20 — write against K=20.**
All results in the paper use K=20, 15×15 grid. No hedging language needed.

**Q4: ε quenched disorder — the shared-ε design:**
The same ε is used for all (τ,β) points at a given σ, AND the same ε (scaled) is
used across all seven σ values. State in §2.2: "To isolate the effect of σ from
the effect of disorder pattern, a single ε realisation is used throughout the
sweep; varying σ scales this fixed pattern." Acknowledge the limitation (one
realisation) and that a formal disorder-averaging check remains future work.
The σ=0.20 spike in the single-point eigenvalue plot (Fig 7) provides direct
evidence that results are somewhat ε-dependent at the single-point level, which
is precisely why landscape-averaged metrics are preferable.

**Q5: Which figure as paper Fig 1?**
The σ=0 vs σ=0.30 landscape comparison (notebook section 15A) should be Fig 1.
It gives the reader the visual punchline — the landscape dramatically darkens —
before the formalism. The abstract can forward-reference it.

---

## Figures available — updated descriptions

| Paper label | Notebook source | What it shows | Verdict |
|-------------|----------------|--------------|---------|
| Fig 1 | §15A | σ=0 vs σ=0.30 κ(τ,β), shared colour scale | **Main visual; use as Paper Fig 1** |
| Fig 2 | §5 | Schelling grid: random initial, after 50 steps | Good for intro/methods |
| Fig 3 | §10 | Phase I: κ(τ,β) heatmap + eigenvector quivers | Key methods result |
| Fig 4 | §14 | 7-panel landscape evolution σ=0→0.30 | Good supplementary; too wide for main text |
| Fig 5 | §14 | Metric curves with 95% CI bands | **Main results figure** |
| Fig 6 | §14 | κ distribution histogram (overlaid, per σ) | **Strong visual; use alongside Fig 5** |
| Fig 7 | §15B | λ₁, λ₂ vs σ at fixed point | Shows mechanism but spike at σ=0.20 needs discussion |
| Fig 8 | §15C | D-curve for σ∈{0,0.05,0.10,0.20,0.30} | **Cleanest result; use in §3.2** |
| Fig 9 | §12 | Phase II κ(μ,σ) heatmap + eigenvalue spectrum | Good for Phase II section |

---

## Related literature — key references

**Loss landscape geometry in DL:**
- Li et al. (2018), *Visualizing the Loss Landscape of Neural Nets*, NeurIPS.
  [Filter normalisation approximates the metric we compute exactly]
- Keskar et al. (2017), *On Large-Batch Training*, ICLR. [Sharp minima → poor
  generalisation; our κ is the ABM analogue of sharpness]
- Garipov et al. (2018), *Loss Surfaces, Mode Connectivity*. [Mode connectivity]
- Foret et al. (2021), *Sharpness-Aware Minimization*, ICLR.

**Sloppy models and information geometry:**
- Sethna et al. (2001/2003), *Sloppy Models*, PRL. [Our paper establishes ABMs
  as sloppy; this is the canonical reference for the concept]
- Gutenkunst et al. (2007), *Universally Sloppy Parameter Sensitivities*, PLoS CB.
  [7 biological models all sloppy; we add spatial ABMs to this list]
- Amari (1998), *Natural Gradient*, Neural Computation. [Fisher-Rao metric]
- Brown & Sethna (2003), eigenspectrum visualisation style.

**ABM calibration:**
- Quera-Bofarull et al. (2023), *Differentiating Agent-Based Models*, NeurIPS.
  [Technical foundation; we extend from "gradients exist" to "landscape geometry"]
- Saltelli et al. (2008), *Global Sensitivity Analysis*. [Standard ABM approach;
  Sobol indices miss confounding; our FIM captures it]
- Sisson et al. (2018), *Handbook of Approximate Bayesian Computation*. [ABC is
  the alternative; our FIM landscape explains where ABC will struggle]

**Disordered systems:**
- Imry & Ma (1975); Aizenman & Wehr (1989). [Quenched disorder rounds phase
  transitions — our mechanism for why σ softens landscape boundaries]
- Harris (1974), effect of disorder on critical exponents.

**Schelling model:**
- Schelling (1971), *Dynamic Models of Segregation*, J. Math. Sociology.

---

## Proposed paper structure

```
1. Introduction
   1.1 The calibration problem for ABMs (Sobol → FIM gap)
   1.2 Sloppy models: when most parameters are unidentifiable
   1.3 Loss landscape geometry: lessons from deep learning
   1.4 Population heterogeneity as a structural parameter
   1.5 Contributions (numbered list of 5 contributions above)

2. Model and Methods
   2.1 The differentiable Schelling model
       — model, Gumbel-softmax relaxation, correctness result
   2.2 Parameter vectors and quenched disorder parameterisation
       — Phase I (τ,β) and Phase II (μ,σ,β); shared-ε design
   2.3 Summary statistics (4-vector)
   2.4 Fisher Information Matrix computation
       — exact Jacobian, noise covariance, mean-field FIM
   2.5 Landscape metrics (formal definitions of all 5)

3. Results
   3.1 Phase I: The (τ,β) FIM landscape — τ stiff, β sloppy
   3.2 Heterogeneity and segregation dynamics (D-curve)
   3.3 Heterogeneity as a landscape regulariser: the σ-sweep
       — metric curves, histogram, landscape comparison
   3.4 Eigenvalue evolution: mechanism of regularisation
   3.5 Mean-field limit: sloppiness is deterministic

4. Discussion
   4.1 Mechanism: ensemble averaging + quenched disorder
   4.2 The identifiable corridor and dynamical phase transitions
   4.3 Practical implications for ABM calibration
   4.4 Limitations (one ε realisation; grid resolution; 2D model)
   4.5 Future work (voter model; disorder averaging; SAM-like calibration)

5. Conclusion

Appendix A: Mean-field FIM derivation and comparison
Appendix B: Sensitivity of corridor metric to threshold choice
Appendix C: Implementation details and reproducibility note
```

---

## Tone and framing guidance

- The **DL landscape analogy** (Li et al.) is the entry point for readers but is
  NOT the main contribution. Frame it in §1.3 as motivation/context, then move on.
  Do not over-reference DL once the physics/ABM framing takes over.

- The **sloppy models framing** (Sethna) is more central. Lead with: "We show that
  the Schelling model — and likely spatial ABMs generally — belongs to the sloppy
  models class."

- The **quenched disorder framing** (Imry & Ma) is the mechanistic explanation for
  the regularisation effect. Use it in §4.1 to explain *why* σ flattens the landscape,
  not just *that* it does.

- Avoid claiming more than the data supports. The K=20 results show 3 robust metrics
  and 2 noisier ones. Say: "three metrics show clear regularisation; two show the
  expected direction with higher variability." This is honest and still a strong claim.

---

## Writing instructions

1. **Format:** LaTeX, `\documentclass{article}`. Packages: `amsmath, amssymb,
   graphicx, booktabs, hyperref, natbib`. I will specify a journal template later.

2. **Length target:** ~8,000 words main text; appendices extra.

3. **Notation consistency:**
   - Parameter vector: `\boldsymbol{\theta}` (bold)
   - Summary statistics: `\mathbf{s}(\boldsymbol{\theta}) \in \mathbb{R}^4`
   - Jacobian: `\mathbf{J} = \partial \mathbf{s} / \partial \boldsymbol{\theta}`
   - FIM: `\mathbf{F}(\boldsymbol{\theta}) = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J}`
   - Condition number: `\kappa = \lambda_{\max} / \lambda_{\min}`
   - Tolerance: `\tau` (homogeneous), `\mu` (mean), `\sigma` (std dev)
   - Sigmoid sharpness: `\beta`; Gumbel temperature: `\tau_g` (fixed)
   - Dissimilarity index: `D`; Moran's I: `I`

4. **Tone:** computational physics / complex systems. State results with numbers.
   No hedging ("we believe", "it seems"). Findings are stated as findings.

5. **Figures:** use `\includegraphics[width=\linewidth]{fig_X}` with the labels
   above. I will provide image files. For now use placeholder filenames.

6. **Workflow:** Start with the abstract and §1. Ask me to confirm before §2.
   I will supply exact numbers from the K=20 printed output when needed.

---

## What to ask me

- Exact numbers from the K=20 printed output (metric values at each σ)
- Whether a specific claim is supported
- Which figure goes where
- Specific formulas or implementation details
- Journal template requirements
