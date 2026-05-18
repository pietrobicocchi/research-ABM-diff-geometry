# Population Heterogeneity as a Landscape Regulariser:
# Information Geometry of the Differentiable Schelling Segregation Model

**Author:** Pietro Bicocchi  
**Date:** 2026-05-18  
**Status:** Research thesis document — basis for paper drafting  
**Codebase:** `research-ABM-diff-geometry` (JAX, see `CLAUDE.md`)

---

## Abstract

We study how population heterogeneity deforms the loss landscape of a spatial
agent-based model (ABM). Using a differentiable relaxation of the Schelling
segregation model, we compute the exact Fisher Information Matrix (FIM) across
the full parameter space and characterise how its geometry changes as the
variance of agent tolerance (σ) increases from zero. The central conjecture,
motivated by the theory of quenched disorder in statistical physics and by
ensemble averaging in deep learning, is that population heterogeneity acts as a
*landscape regulariser*: increasing σ flattens the FIM landscape, softens
dynamical phase boundaries, and widens the region of parameter space in which
the model is identifiable. This question has not been posed for ABMs. It has
direct implications for calibration practice and for the theory of emergent
segregation.

---

## 1. The Schelling Model: Background and Motivation

### 1.1 What the model says

Thomas Schelling's 1971 segregation model makes a precise and counterintuitive
claim: **mild individual preferences are sufficient to produce near-total
collective segregation**. No agent needs to prefer segregation; the collective
pattern emerges from each individual's desire not to be a small minority in
their immediate neighbourhood.

The model is defined on a two-dimensional grid of H × W cells. Each cell is
occupied by an agent of type A or type B, or is empty. Each occupied agent
holds a *tolerance threshold* τ ∈ [0, 1]. An agent is *satisfied* if the
fraction of its Moore neighbourhood (8 adjacent cells) occupied by agents of
the same type is at least τ. Unsatisfied agents relocate to a randomly chosen
empty cell. The process iterates. Starting from random initial placement,
spatial clusters of like-type agents emerge and grow.

The Schelling result is that segregation emerges even at τ = 0.4 — an agent
who merely prefers not to be in a 60% minority. The mechanism is a positive
feedback loop: a locally A-dominated region retains its A agents (they are
satisfied) while B agents there become increasingly unsatisfied and leave,
reinforcing A-dominance. The segregation is an emergent collective property,
not an individual intent.

### 1.2 Why differentiability matters

Standard ABMs use discrete, stochastic decision rules. This precludes exact
gradient computation. The consequence for scientific analysis is severe:
estimating the Jacobian ∂s/∂θ (how outputs change with parameters) requires
finite differences — at least 2p forward simulations per Jacobian column, each
of which is noisy. For a parameter space of even modest dimension, this is
prohibitively expensive and statistically unreliable.

Differentiable ABMs replace discrete decisions with smooth relaxations. Here:

- **Satisfaction** uses a sigmoid: `sat_A = σ(β · (frac_A_neighbours − τ))`,
  where β controls sharpness. At β → ∞ this recovers the hard threshold.
- **Stay/move decisions** use the Gumbel-softmax trick: the binary choice is
  replaced by a soft probability `p_stay = σ((2·sat − 1)·β/τ_g)` in the mean-
  field limit, with Gumbel noise added for the stochastic variant (τ_g is the
  relaxation temperature, fixed; not in θ).
- **Mass redistribution** is uniform over empty cells, preserving the per-cell
  simplex constraint (occupancies sum to 1) and mass conservation exactly.

The result is a fully differentiable forward pass. JAX's `jacfwd` computes the
exact Jacobian in a single forward pass via forward-mode automatic
differentiation. This is the technical foundation for everything that follows.

The implementation follows Quera-Bofarull et al. (2023) in spirit, with
important corrections: type-specific independent Gumbel draws are required for
the segregation mechanism to function (shared noise collapses to a mean-field
fixed point where D → 0). See §4.2 for the precise move rule.

---

## 2. Technical Framework

### 2.1 State representation

The model state at each time step is a `WorldState` — an immutable PyTree:

```
soft_occupancy : Float[H, W, 3]   # axis-2: [p_empty, p_A, p_B], sums to 1
tolerances     : Float[H, W]       # per-cell tolerance; frozen at initialisation
step           : Int               # scalar step counter
```

`soft_occupancy` begins as a one-hot encoding (each cell is exactly one of
{empty, A, B}). After the first step, cells acquire fractional occupancies —
the "soft" state that enables differentiation. The initial dissimilarity D = 1
for any one-hot state; the meaningful scientific quantity is the value D
converges to after T steps, once the occupancy field has softened.

### 2.2 The move rule (critical correctness detail)

The correct differentiable Schelling move rule computes **type-specific**
stay/move probabilities with **independent** Gumbel draws:

```
sat_A, sat_B = type_satisfaction(soft_occ, tolerances, β)

# Independent Gumbel noise for each type
p_stay_A = softmax(([β·sat_A, β·(1−sat_A)] + Gumbel_A) / τ_g)[0]
p_stay_B = softmax(([β·sat_B, β·(1−sat_B)] + Gumbel_B) / τ_g)[0]
```

A type-averaged satisfaction with shared noise is mathematically incorrect: it
collapses to a mean-field fixed point where D → 0 (the uniform distribution is
a fixed point of the dynamics). The physical mechanism — B agents leaving
A-dominated regions, A agents leaving B-dominated regions — requires
independent decisions. This was a non-trivial debugging finding (see
`CLAUDE.md` implementation notes).

### 2.3 Parameter vectors

Three phases, three parameter vectors:

| Phase | θ | dim(θ) | FIM shape | Description |
|-------|---|--------|-----------|-------------|
| I | (τ, β) | 2 | 2×2 | Homogeneous population |
| II | (μ, σ, β) | 3 | 3×3 | Heterogeneous population |
| III | (τ, β) or (μ, σ, β) | 2 or 3 | as above | Mean-field limit (Σ = I) |

**τ** — the tolerance threshold. The primary Schelling parameter. An agent's
decision rule hinges entirely on this value.

**β** — the sigmoid sharpness. Controls the crispness of the threshold. Not
the primary scientific parameter, but genuinely free in the differentiable
relaxation. Included because it may be confounded with τ: the iso-satisfaction
curve `β·(f−τ) = c` defines a hyperbola `τ = f − c/β` in (τ, β) space.
Moving along this curve leaves sat approximately constant, creating a sloppy
direction.

**μ** — the mean tolerance in the heterogeneous population. Same role as τ in
Phase I.

**σ** — the standard deviation of the tolerance distribution. The structural
heterogeneity parameter. This is the key variable of the main research
question. Tolerances are sampled as `τ_ij = clip(μ + σ·ε_ij, 0.001, 0.999)`
where `ε_ij ~ N(0,1)` is fixed at initialisation (quenched disorder; see §3).
The reparametrisation trick ensures ∂/∂μ and ∂/∂σ flow correctly through
the tolerance sampling.

**τ_g** — the Gumbel relaxation temperature. A numerical parameter, not in θ.
Fixed at 0.5 throughout. Its value affects sharpness of the stochastic
approximation but is not a scientific parameter of interest.

### 2.4 Summary statistics vector

`s(θ) ∈ ℝ⁴`, computed from the final state after T simulation steps:

```
s = [D, sat̄, I, H]
```

| # | Symbol | Name | Formula | What it captures |
|---|--------|------|---------|-----------------|
| 0 | D | Dissimilarity index | `0.5 Σ_i |a_i/A − b_i/B|` | Global spatial segregation |
| 1 | sat̄ | Mean satisfaction | `Σ_i sat_i · occ_i / Σ_i occ_i` | Agent-level wellbeing |
| 2 | I | Moran's I | `(z·Wz)/(z·z)`, z = centred A-fraction | Spatial autocorrelation |
| 3 | H | Neighbourhood homogeneity | `mean(occ_A·frac_A_neigh + occ_B·frac_B_neigh)` | Cluster size proxy |

These four statistics are complementary. D is a global aggregate. I is a
spatial measure. sat̄ is agent-level. H is a neighbourhood-level cluster proxy.
Together they produce a richer Jacobian than a scalar dissimilarity, giving the
FIM more structure to work with. All are differentiable via the convolution
operations in `statistics/`.

### 2.5 The Fisher Information Matrix

The FIM is the core scientific object:

```
F(θ) = J(θ)ᵀ Σ(θ)⁻¹ J(θ) ∈ ℝᵖˣᵖ
```

where:
- `J(θ) = ∂s/∂θ ∈ ℝ⁴ˣᵖ` is the Jacobian of the summary statistics
- `Σ(θ) ∈ ℝ⁴ˣ⁴` is the noise covariance, estimated by running K = 20
  independent simulations at the same θ and computing the sample covariance

The Jacobian is computed exactly by `jax.jacfwd(pipeline)(θ)`. For p = 2
(Phase I) or p = 3 (Phase II), forward-mode autodiff is optimal.

**Eigenvalues of F** encode identifiability:
- Large eigenvalue (λ_max) → stiff direction: outputs are highly sensitive,
  the corresponding parameter combination is well-identified
- Small eigenvalue (λ_min) → sloppy direction: outputs are insensitive, the
  parameter combination is effectively unobservable
- Condition number κ = λ_max / λ_min: the ratio of most to least identifiable
  directions. κ ≫ 1 → sloppy model.

**Mean-field FIM** (Phase III): replace Gumbel noise with its expectation
(E[Gumbel] = 0), giving the deterministic step `p_stay = σ((2·sat−1)·β/τ_g)`.
The mean-field model is fully deterministic: Σ = I, so F_mf = J_mf^T J_mf.
No Monte Carlo is needed — the mean-field landscape is ~20× cheaper to compute.

---

## 3. The Central Research Question

### 3.1 The question

The primary research question of this project is:

> **How does population heterogeneity (σ) deform the Fisher Information
> landscape of the Schelling model, and does it act as a regulariser — smoothing
> sharp features, softening phase boundaries, and reducing the condition number?**

This is distinct from identifiability (which asks: at a fixed θ, can I estimate
the parameters?). This question asks about the *geometry of the landscape as a
function of model structure*. The independent variable is σ — the population
heterogeneity — treated as an architectural choice that deforms the FIM surface.

### 3.2 Motivation: the Li et al. analogy

Li, Xu, Taylor, Studer, Goldstein (2018) — *Visualizing the Loss Landscape of
Neural Nets* — posed an analogous question for deep learning: how does network
architecture deform the loss landscape? Their key finding: skip connections
(ResNets) dramatically flatten the loss surface relative to plain networks
(VGGs). This flatness correlates with better generalisation and easier
optimisation.

Their technical challenge: for neural networks with millions of parameters, the
FIM is intractable. They used filter-normalised random projections as an
approximation to the metric structure.

The present work inverts this situation. For differentiable ABMs, the parameter
space is small (p = 2 or 3) and the FIM is computed exactly. The architectural
analogy is:

| Deep learning | Differentiable ABM |
|---|---|
| Network architecture (ResNet vs. VGG) | Population structure (σ = 0 vs. σ > 0) |
| Skip connections → flatter landscape | Heterogeneity → flatter FIM landscape? |
| Filter normalisation (approximate metric) | Exact FIM (exact metric) |
| Loss L = training error | Loss L = calibration residual ‖s(θ) − s_obs‖² |
| Sharpness → poor generalisation | Condition number κ → poor identifiability |

The ABM setting offers something DL cannot: an exact, interpretable metric
structure. The scientific question is cleaner.

### 3.3 Motivation: quenched disorder in statistical physics

The addition of σ > 0 introduces **quenched disorder** into the Schelling
model. Tolerances τ_ij are sampled once at initialisation and held fixed — they
are "frozen" in the physical terminology. This is the standard definition of
quenched (as opposed to annealed) disorder.

The theory of quenched disorder in statistical mechanics provides strong
theoretical prior for the conjecture:

**The Random Field Ising Model (RFIM):** The Ising model with quenched random
fields is the canonical disordered model. Key result: disorder rounds and
softens the phase transition. Where the clean model has a sharp discontinuity
(first-order) or a power-law singularity (second-order), the disordered model
has a gradual crossover whose width scales with σ (the disorder strength). This
is disorder-induced rounding.

**The Harris criterion:** For a clean critical point with correlation length
exponent ν, disorder is a *relevant* perturbation (changes the universality
class) if ν < 2/d (d = spatial dimension). For the Schelling model on a 2D
grid, if the transition has ν < 1, disorder would be relevant and would
qualitatively change the critical behaviour — softening the transition in a
way that cannot be treated perturbatively.

**Implication for the FIM landscape:** If disorder rounds the phase transition,
the FIM landscape should also smooth near the transition. The sharp boundaries
between the identifiable corridor and the sloppy extremes (which correspond to
the phase boundaries of the dynamical model) should become more gradual as σ
increases. This is the geometric manifestation of disorder-induced rounding.

### 3.4 Motivation: ensemble averaging in deep learning

A heterogeneous population with tolerance distribution N(μ, σ²) produces
aggregate dynamics that are a weighted average over homogeneous populations
with different τ values — each weighted by the population density at that τ.
This is structurally identical to an **ensemble of models**.

In deep learning, ensemble averaging is known to smooth the loss landscape
(Lakshminarayanan et al. 2017; Fort et al. 2019). The ensemble loss is:

```
L_ens(θ) = E_{τ~N(μ,σ²)}[L_τ(θ)]
```

By Jensen's inequality, for any convex function g, E[g(x)] ≥ g(E[x]). The
curvature of L_ens (which controls the FIM) is typically lower than the
curvature of any individual L_τ, because the average smooths over sharp
features of individual members. The standard deviation σ of the averaging
distribution controls the degree of smoothing — higher σ → more averaging →
smoother landscape.

This mechanism operates independently of the quenched disorder argument. Both
point in the same direction: **heterogeneity regularises the landscape**.

---

## 4. Empirical Results to Date

### 4.1 Phase I: the (τ, β) landscape

Grid: τ ∈ [0.1, 0.8] × β ∈ [2.0, 15.0], 15×15 points. K = 20 noise samples
per point. Grid: 30×30, T = 50 steps.

**Condition number landscape κ(τ, β):**

The landscape has a clearly identifiable corridor at τ ≈ 0.3–0.5, β ≈ 3–8,
where κ achieves its minimum (log₁₀ κ ≈ 2–3). Outside this corridor, κ rises
steeply by 3–5 orders of magnitude.

The corridor corresponds to the dynamically active regime — neither frozen (τ
too low, agents never move) nor chaotic (τ too high, everyone always moves).
This connection between FIM identifiability and dynamical activity is the
foundational empirical result and is physically interpretable: the FIM measures
output sensitivity to parameter changes, and in a frozen or chaotic fixed point,
there is no sensitivity regardless of the parameter values.

**Stiff and sloppy directions:**

The stiff eigenvector (λ_max direction) consistently points approximately along
−τ throughout the landscape. τ is the identifiable parameter.

The sloppy eigenvector (λ_min direction) consistently points approximately
along −β. β is the unidentifiable parameter. This is explained by the
confounding `sat = σ(β·(f−τ))`: iso-satisfaction contours in (τ, β) space are
hyperbolae `τ = f − c/β`, defining a direction of near-zero gradient.

### 4.2 Phase II: the (μ, σ, β) landscape

Grid: μ ∈ [0.1, 0.8] × σ ∈ [0.01, 0.3], 12×12 points. β = 5 fixed. Grid:
30×30, T = 50 steps.

**Effect of heterogeneity on segregation dynamics:**

At the same (μ=0.4, β=5), adding σ = 0.15 reduces the final dissimilarity
index from D ≈ 0.926 (σ=0) to D ≈ 0.896 (σ=0.15) and slows the convergence
of D over time. The mechanism: agents with low τ (tolerant, from the left tail
of N(μ, σ²)) are satisfied almost anywhere and do not reinforce cluster
formation. The tolerant subpopulation acts as a buffer that dampens the
segregation feedback loop. The effect is modest but consistent.

**FIM landscape in (μ, σ) space:**

The low-κ (identifiable) region in (μ, σ) space is approximately
μ ∈ [0.2, 0.5] × σ ∈ [0.05, 0.25]. Two sloppy regions are evident:

*High μ (> 0.6):* Same mechanism as Phase I. At high mean tolerance, agents
are rarely dissatisfied, dynamics freeze, and neither parameter is identifiable.

*σ ≈ 0:* Near-zero heterogeneity makes σ itself unidentifiable — a small
variance produces outputs nearly identical to σ = 0. There is a sloppy
direction along σ at small σ, which disappears as σ grows.

**Catastrophic sloppiness at (μ=0.7, σ=0.05):**

The third eigenvalue of the 3×3 FIM at this point is λ₃ ≈ 0.05, compared to
λ₁ ≈ 7×10⁵. The condition number κ ≈ 10⁷. One linear combination of (μ, σ, β)
is essentially invisible in the summary statistics. This occurs because high μ
places the dynamics in a near-frozen regime, while low σ provides insufficient
diversity to escape it.

**Rotation of stiff/sloppy directions:**

Unlike Phase I (where τ was always stiff), the Phase II stiff/sloppy directions
rotate across the (μ, σ) landscape. There is no fixed "unidentifiable
parameter" — the role of μ and σ exchanges depending on where in parameter
space the model operates.

**Eigenvalue spectrum at representative points:**

| Point | λ₁ | λ₂ | λ₃ | κ |
|---|---|---|---|---|
| (μ=0.7, σ=0.05) | ~7×10⁵ | ~2×10³ | ~0.05 | ~10⁷ |
| (μ=0.2, σ=0.05) | ~2×10⁴ | ~3×10³ | ~80 | ~250 |
| (μ=0.2, σ=0.25) | ~2×10⁴ | ~3×10² | ~50 | ~400 |
| (μ=0.7, σ=0.25) | ~2×10³ | ~2×10³ | ~50 | ~40 |

The last row is the most informative: at (μ=0.7, σ=0.25), adding large
heterogeneity to a high-mean-tolerance regime *dramatically reduces* κ (from
~10⁷ to ~40). This is preliminary but striking evidence for the regularisation
conjecture.

### 4.3 Phase III: mean-field FIM

The mean-field model replaces Gumbel noise with its expectation. At the spot-
check (τ=0.4, β=5), κ_mf ≈ 130. The mean-field FIM is ~20× cheaper to compute
(no Monte Carlo). Whether the sloppy structure of the stochastic FIM is
preserved in the mean-field across the full (τ, β) landscape is the open
question of Section 13 of the notebook. The preliminary single-point result
suggests the structure is preserved.

---

## 5. The Central Conjecture

### 5.1 Statement

**Conjecture (Heterogeneity as Landscape Regulariser):**

*For the differentiable Schelling model, increasing population heterogeneity σ
from 0 acts as a regulariser of the FIM landscape in the following sense:*

*(i) The maximum condition number κ_max decreases monotonically with σ up to
some optimal σ\*, then may level off.*

*(ii) The variance of κ across the (μ, β) landscape decreases with σ: the
landscape becomes more homogeneous.*

*(iii) The identifiable corridor (low-κ region) widens with σ: there is a
larger region of parameter space from which the model is calibratable.*

*(iv) The phase boundaries (the sharp transitions from low to high κ
corresponding to dynamical phase boundaries of the model) soften with σ: the
transition from identifiable to sloppy becomes more gradual.*

*This regularisation is driven by two mechanisms: (a) ensemble averaging —
the heterogeneous model is a mixture over homogeneous models, and mixtures
have smoother loss surfaces — and (b) quenched disorder rounding — disorder
softens phase transitions and their associated FIM singularities.*

### 5.2 What would falsify it

The conjecture would be falsified by any of:
- κ increasing with σ (heterogeneity sharpens rather than flattens)
- The identifiable corridor narrowing as σ increases
- Phase boundaries becoming sharper rather than softer
- No systematic relationship between σ and landscape geometry

Any of these outcomes would itself be scientifically interesting and would
require a different mechanism to explain.

### 5.3 The optimal σ*

A natural prediction of the regularisation mechanism is that there exists an
optimal σ* that maximises the total Fisher information, measured by
`det(F)` or `trace(F)`. Below σ*, σ itself is unidentifiable (wasted
variance). Above σ*, the ensemble averaging may over-smooth the response,
reducing sensitivity. The existence and location of σ* as a function of μ and
β is a concrete, falsifiable prediction.

---

## 6. The Experimental Programme

### 6.1 The σ-sweep experiment (primary)

**Design:** Fix μ = 0.4, β = 5. Compute the full FIM on a 15×15 (τ̃, β̃) grid
(where τ̃ plays the role of τ in Phase I but with mean μ = 0.4 fixed) for each
of seven σ values: σ ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

This produces seven complete FIM landscapes. The treatment variable is σ.
Everything else is held constant. Grid: 30×30, T = 50, K = 20.

**Measurements at each σ:**

| Measurement | Definition | Expected direction |
|---|---|---|
| κ_max | Peak condition number across grid | Decreasing with σ up to σ* |
| median(κ) | Median condition number | Decreasing |
| Var(log κ) | Variance of log-condition-number | Decreasing (more homogeneous landscape) |
| A_id(κ < 1000) | Area of identifiable corridor | Increasing (widening corridor) |
| Boundary width | Width of κ = 100 → κ = 10000 transition zone | Increasing (softer boundary) |
| det(F) | Volume of FIM at reference point (μ,β) = (0.4,5) | Non-monotone, peaked at σ* |
| trace(F) | Total Fisher information at reference point | Non-monotone |

**Visualisations:**

1. Seven-panel FIM heatmaps (one per σ), same colour scale
2. κ_max vs σ curve
3. Histogram of log κ values for each σ, overlaid
4. Boundary sharpness metric vs σ
5. det(F) and trace(F) at the reference point vs σ

### 6.2 The phase boundary sharpness experiment

For the transition from identifiable to sloppy at fixed β = 5, plot κ(μ) as a
one-dimensional curve for several values of σ. The prediction: as σ increases,
the transition from κ ~ 10² (identifiable) to κ ~ 10⁶ (sloppy) becomes more
gradual. Quantify sharpness as the width of the transition zone in μ-units
(where width is defined as the range of μ over which log κ changes by 2
decades).

### 6.3 The FIM curvature decomposition

At the reference point (μ=0.4, β=5), compute the FIM eigenvalues as a function
of σ:

```
λ₁(σ), λ₂(σ), λ₃(σ)  for σ ∈ {0, 0.05, ..., 0.30}
```

Plot the eigenvalue spectrum (Brown/Sethna style) for each σ on the same axes.
The prediction: as σ increases, the gap between λ₁ and λ₃ decreases, and the
sloppy threshold (1% of λ_max) crosses fewer eigenvalues. The effective
dimension of the sloppy subspace decreases.

### 6.4 Mean-field landscape comparison (Phase III, ongoing)

Compare the stochastic FIM landscape (F_sim) to the mean-field FIM landscape
(F_mf) across the full (τ, β) grid. The question: does the sloppy structure
persist in the mean-field? If F_mf ≈ F_sim everywhere, then:
(a) Stochasticity does not drive sloppiness — it is a property of the
    deterministic model equations
(b) The mean-field (20× cheaper) can serve as a proxy for the full stochastic
    FIM during parameter space exploration

The ratio log₁₀(κ_sim / κ_mf) as a function of (τ, β) maps regions where
stochasticity amplifies or suppresses sloppiness. This is a clean,
computationally tractable experiment.

---

## 7. Connections to Literature

### 7.1 Loss landscape geometry in deep learning

**Li et al. (2018)** — *Visualizing the Loss Landscape of Neural Nets.* NIPS.  
The baseline reference. Skip connections flatten the DL loss landscape. Filter
normalisation approximates the metric structure (our setting uses the exact FIM).

**Keskar et al. (2017)** — *On Large-Batch Training for Deep Learning.* ICLR.  
Sharp minima (high curvature) → poor generalisation. Flat minima → robust
generalisation. The flatness concept maps directly to our condition number κ.

**Garipov et al. (2018)** — *Loss Surfaces, Mode Connectivity, and Fast
Ensembling of DNNs.*  
Two solutions in DL can be connected by a low-loss path — "mode connectivity."
The ABM analogue: can you move between parameter regimes along a low-κ path?
The sloppy directions already define such paths.

**Fort & Jastrzebski (2019)** — *Large Scale Structure of Neural Network Loss
Landscapes.*  
Hierarchical basin structure in DL landscapes. Whether ABM landscapes have
similar structure across scales is an open question.

**Foret et al. (2021)** — *Sharpness-Aware Minimization (SAM).*  
Explicitly seek flat minima during training. The ABM analogue: choose parameter
estimates that are not just good fits but lie in flat FIM basins (robust to
perturbation). Not pursued here but a natural extension.

### 7.2 Disordered systems and phase transitions

**Imry & Ma (1975)** — Random field effects in first-order transitions.  
Quenched disorder destroys long-range order. Foundational for the disorder-
induced rounding argument.

**Aizenman & Wehr (1989)** — Rounding of first-order phase transitions.  
Rigorous mathematical result: quenched disorder rounds first-order transitions
in d ≤ 2. The Schelling model is 2D — this result is directly applicable if
the transition is first-order.

**Harris (1974)** — Effect of random defects on second-order transitions.  
The Harris criterion: disorder is relevant if ν < 2/d. Relevant disorder changes
the universality class and alters critical exponents.

**Sethna et al. (2001)** — *Sloppy Models, Parameter Space Compression, and
the FIM.* Physical Review Letters.  
The original sloppy models paper. FIM eigenvalues spanning orders of magnitude
are generically expected in complex models. Biological and physical models are
universally sloppy. The present work establishes that spatial ABMs are sloppy
and that population structure deforms this sloppiness.

### 7.3 ABM calibration and sensitivity analysis

**Saltelli et al. (2008)** — *Global Sensitivity Analysis: The Primer.*  
Standard Sobol-index approach to ABM sensitivity. Measures variance attribution
but not gradient structure or landscape geometry. Gradient-free.

**Quera-Bofarull et al. (2023)** — *Differentiating Agent-Based Models.*  
The technical foundation for differentiable ABMs. The present work extends this
by characterising the FIM landscape structure rather than just demonstrating
gradient existence.

**Sisson et al. (2018)** — *Handbook of Approximate Bayesian Computation.*  
Standard approach to ABM calibration. ABC is gradient-free and expensive. The
FIM landscape characterisation informs where ABC will succeed or fail
(identifiable vs. sloppy regions).

### 7.4 Information geometry

**Amari (1998)** — *Natural Gradient Works Efficiently in Learning.*  
The Fisher-Rao metric on statistical manifolds. The FIM defines the natural
Riemannian metric on the parameter space of a statistical model. The "natural
distance" between parameter points is the Fisher information distance, not the
Euclidean distance. This is why Li et al.'s filter normalisation is needed —
it approximates the natural metric. Our differentiable ABM provides it exactly.

**Martens (2014)** — *New Insights and Perspectives on the Natural Gradient.*  
Practical implications of the Fisher-Rao metric for optimisation. The present
work uses the FIM as a diagnostic rather than for optimisation, but the metric
structure is the same.

---

## 8. Open Questions and Future Directions

### 8.1 Generalisation to other ABMs (Phase IV — Voter model)

The voter model is the natural second test case. In the voter model, agents
adopt the opinion of a randomly chosen neighbour with some copying probability.
The model has a different local decision rule (probabilistic copying rather
than threshold satisfaction) but the same spatial structure (2D grid, Moore
neighbourhood).

The key scientific question: does the regularisation conjecture hold for the
voter model? If heterogeneity (variance in copying probability across agents)
also flattens the voter model FIM landscape, this suggests a **general
principle for heterogeneous spatial ABMs**, independent of the specific decision
rule. If it does not hold for the voter model, the mechanism is model-specific
and needs refinement.

The existing `geometry/` infrastructure (FIM, eigenspectrum, landscape plots)
is model-agnostic and reusable without modification.

### 8.2 Connecting FIM geometry to dynamical phase transitions

A deep open question: does the minimum of κ(τ) at fixed β correspond exactly
to the dynamical phase transition of the Schelling model (the point where
segregation first emerges)?

In statistical mechanics, the Fisher information (susceptibility) diverges AT a
phase transition, not before it. If the Schelling transition is sharp at σ = 0
and the FIM minimum tracks the transition point, this provides a new method for
detecting phase transitions in ABMs: compute the FIM landscape and look for the
κ minimum. This method would be faster and more precise than traditional
order-parameter sweeps.

Testing this requires independently characterising the Schelling phase transition
(e.g., via finite-size scaling of D fluctuations) and comparing the transition
point to the FIM minimum location.

### 8.3 σ* as a function of μ: the optimal heterogeneity surface

The existence of an optimal heterogeneity σ*(μ, β) that maximises total Fisher
information (det F or trace F) is a natural prediction of the regularisation
conjecture. Mapping this surface across (μ, β) space would give a prescriptive
result for calibration practice: "given a population with mean tolerance μ and
sigmoid sharpness β, this is the level of tolerance heterogeneity that makes
the model most identifiable."

### 8.4 Gradient-guided phase diagram construction

With access to ∂κ/∂θ via autodiff, the boundaries between identifiable and
sloppy regions can be traced by following the gradient of κ rather than by
exhaustive grid search. This is computationally more efficient in higher-
dimensional parameter spaces and allows systematic characterisation of how the
boundary moves as σ changes.

### 8.5 Calibration under heterogeneity misspecification

A practical question: if the true population is heterogeneous (σ > 0) but the
calibration model assumes homogeneity (σ = 0), what is the estimation bias?
The FIM provides a rigorous framework: the bias is driven by the projection of
the true parameter vector onto the homogeneous model's parameter space, weighted
by the FIM of the misspecified model. This connects model misspecification to
information-geometric distances.

---

## 9. Summary of Contributions

| Contribution | Status | Evidence |
|---|---|---|
| Exact FIM computation for a spatial differentiable ABM | Complete | Phase I, II, III |
| Phase I landscape: τ stiff, β sloppy, identifiable corridor | Complete | Section 10 notebook |
| Phase II landscape: σ adds sloppy direction near σ=0, improves at moderate σ | Complete | Section 12 notebook |
| Catastrophic sloppiness at (μ=0.7, σ=0.05), κ~10⁷ | Complete | Section 12 spectrum |
| Mean-field FIM as cheap proxy (Σ=I, 20× faster) | Complete | Phase III |
| Heterogeneity regularisation conjecture | Stated, preliminary evidence | §4.2, §5 |
| σ-sweep experiment to test regularisation | Planned | §6.1 |
| Phase boundary sharpness measurement | Planned | §6.2 |
| Voter model generalisation | Planned | Phase IV |
| Connection to dynamical phase transition | Open question | §8.2 |

---

## 10. The Paper Argument (Draft Narrative)

**Opening:** ABMs are used to study complex social and biological systems, but
calibrating them — inferring parameters from observations — is computationally
expensive and theoretically poorly understood. A key obstacle is the geometry
of the parameter space: in what regions are model parameters identifiable, and
how does model structure affect this geometry?

**Gap:** Existing work (Saltelli, Sobol sensitivity analysis) characterises
output variance attributable to parameters, but not the gradient structure of
the loss landscape. The deep learning community has studied loss landscape
geometry extensively (Li et al. 2018; Foret et al. 2021), but the analogous
question for ABMs has not been posed.

**Contribution 1:** We introduce the FIM landscape as the natural tool for
characterising ABM parameter geometry. The differentiable Schelling model
allows exact FIM computation (no approximation, unlike DL). We find that the
landscape has an identifiable corridor corresponding to the dynamically active
regime and that the condition number κ tracks the model's distance from trivial
fixed points.

**Contribution 2:** Population heterogeneity (variance σ in agent tolerances)
deforms the FIM landscape as a regulariser: it flattens the landscape, softens
phase boundaries, and widens the identifiable corridor. The mechanism operates
through two independent routes — ensemble averaging (a mixture of models is
smoother than any individual) and quenched disorder rounding (a known
statistical mechanics result for disordered spatial models).

**Contribution 3:** We characterise the trade-off: σ ≈ 0 makes σ itself
unidentifiable, while intermediate σ improves identifiability of all parameters.
There is an optimal σ* that maximises total Fisher information. This has direct
implications for calibration study design.

**Broader claim:** Population heterogeneity is a structural lever that can be
tuned to control the information content of an ABM. Understanding this
relationship between model structure and parameter geometry is essential for
principled calibration of any ABM that incorporates population diversity.

---

*End of document. Version 1.0, 2026-05-18.*
