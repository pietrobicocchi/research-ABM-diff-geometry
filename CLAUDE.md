# CLAUDE.md — research-ABM-diff-geometry

## What this project is

A JAX implementation of the Schelling segregation model designed to study the
**Fisher information geometry of differentiable agent-based models (ABMs)**.

The central question: how does the loss landscape (FIM eigenspectrum, condition number,
sloppy directions) of a Schelling model's summary statistics change across parameter
space, and how does agent *heterogeneity* (variance in tolerance) affect identifiability?

Full design spec: `docs/superpowers/specs/2026-05-16-differentiable-schelling-design.md`

---

## Scientific concepts — read before editing any science module

**Summary statistics** `s(θ) ∈ ℝᵈ`: scalar observables computed from a simulation
run — segregation index, mean satisfaction, cluster-size moments. These are the
quantities we "measure" from the model.

**Jacobian** `J = ∂s/∂θ ∈ ℝᵈˣᵖ`: how summary statistics change with parameters.
Computed via `geometry/jacobian.py`. Use `jacfwd` when p < d (few parameters, many
stats — almost always the case here). Use `jacrev` when p > d.

**Fisher Information Matrix (FIM)** `F = Jᵀ Σ⁻¹ J ∈ ℝᵖˣᵖ`: where Σ is the
covariance of summary stats over simulation noise. Eigenvalues of F tell you which
parameter directions are well-identified (large eigenvalue = **stiff**) and which are
not (small eigenvalue = **sloppy**). Condition number = λ_max / λ_min.

**Sloppy models**: models whose FIM has eigenvalues spanning orders of magnitude.
Most parameter combinations are unidentifiable. This is the expected result and the
object of study, not a bug.

**Differentiable ABM**: forward pass replaces discrete agent decisions with smooth
relaxations so autodiff can compute ∂s/∂θ exactly. Here: sigmoid satisfaction
functions, Gumbel-softmax destination selection (Quera-Bofarull et al. methodology).
Temperature `τ_g` controls sharpness — 0 → hard discrete, ∞ → uniform.

**Schelling model**: agents on a 2D H×W grid, two groups A and B plus empty cells.
Each agent has tolerance τ ∈ [0,1]. An agent is satisfied if the fraction of similar
neighbours ≥ τ. Unsatisfied agents move to empty cells. Segregation emerges even from
mild tolerances. Key output: segregation index, mean satisfaction, cluster statistics.

---

## Scientific narrative — intuition, motivation, and findings

### The Schelling model: why segregation emerges from mildness

The Schelling segregation model (1971) makes a striking claim: *you do not need
racism to get segregation*. All you need is a mild preference — each individual
just wants not to be a tiny minority in their immediate neighbourhood.

Concretely: place agents of two groups (A = red, B = blue) on a grid with some
empty cells. Give each agent a **tolerance threshold τ ∈ [0,1]**: an agent is
*satisfied* if at least fraction τ of their 8 Moore neighbours share their type.
Unsatisfied agents move to a random empty cell. Repeat.

What happens? Even at τ = 0.4 (you just don't want to be in a 60%+ minority),
near-total segregation emerges after ~50 steps. The model does not need agents
to prefer segregation — mildness is enough. The segregation is a collective
emergent property, not an individual intent.

In the differentiable version, the hard threshold is replaced by a sigmoid:
`sat_A = σ(β · (frac_A_neighbours − τ))`. Temperature `β` controls sharpness:
large β → hard threshold, small β → fuzzy gradient. The Gumbel-softmax replaces
discrete move/stay decisions with smooth probabilities so that autodiff can
compute exact gradients through the entire simulation.

### Why make it differentiable?

The goal is to compute the **Fisher Information Matrix (FIM)** — a measure of
how much information the model's observable outputs carry about its parameters.
This requires the Jacobian `J = ∂s/∂θ` (how summary statistics change with
parameters). Computing J by finite differences would require 2p forward
simulations per point and would be very noisy (the model is stochastic).
Autodiff (jacfwd) computes the exact Jacobian in a single forward pass.

### Parameter vectors and their meaning

**Phase I — θ = (τ, β):**

- **τ (tolerance threshold):** the primary scientific parameter. It directly
  controls the decision boundary between satisfied and unsatisfied. This is the
  Schelling parameter.

- **β (sigmoid sharpness):** controls how crisp the threshold is. In the real
  discrete model β = ∞. We include β in θ because: (1) it is genuinely free in
  the differentiable relaxation, and (2) it may be confounded with τ — any
  iso-satisfaction contour `β·(f−τ) = const` defines a curve `τ = f − c/β` in
  parameter space, meaning you can trade β for τ while holding model behaviour
  approximately constant. This confounding is the sloppy direction.

**Phase II — θ = (μ, σ, β):**

- **μ (mean tolerance):** same role as τ in Phase I, now the centre of a
  distribution rather than a single value.

- **σ (tolerance heterogeneity):** the standard deviation of the tolerance
  distribution across agents. σ = 0 reduces to Phase I. σ > 0 means the
  population contains a mix of tolerant (low τ, happy almost anywhere) and
  intolerant (high τ, very selective) individuals.

  **Why introduce σ?** Because in reality individuals differ. A homogeneous
  population where everyone has exactly the same threshold is a convenient but
  unrealistic assumption. The scientific question: does individual variation in
  tolerance matter for the collective outcome, and can we identify σ from
  aggregate observables?

  The reparametrisation trick `τ_ij = μ + σ·ε_ij` (ε ~ N(0,1) fixed at
  initialisation) keeps gradients ∂/∂μ and ∂/∂σ flowing correctly through the
  tolerance sampling.

### The four summary statistics and what they measure

`stats_array_fn(state, β) → Float[4]` returns:

| # | Stat | Intuition |
|---|------|-----------|
| 0 | Dissimilarity D | Global segregation: 0 = fully mixed, 1 = perfectly separated |
| 1 | Mean satisfaction | Average fraction of agents who are "happy" with their neighbourhood |
| 2 | Moran's I | Spatial autocorrelation: are nearby cells similar? Positive = clustered |
| 3 | Neighbourhood homogeneity | Soft cluster-size proxy: how same-type are neighbours on average? |

These four stats capture complementary aspects: D is global, Moran's I is
spatial, satisfaction is agent-level, homogeneity is neighbourhood-level.
Together they make the Jacobian richer and the FIM more informative than a
scalar dissimilarity alone.

### Key empirical findings (as of 2026-05-18)

**Phase I — (τ, β) landscape:**

- The model is in the **"interesting dynamical regime"** at τ ≈ 0.3–0.5,
  β ≈ 3–8. Here the model is neither frozen (everyone satisfied) nor chaotic
  (everyone moves randomly). This is where κ is lowest and parameters are most
  identifiable.

- **τ is the stiff direction** (large eigenvalue): τ directly gates who moves
  and who stays, so the outputs are highly sensitive to it. It is identifiable.

- **β is the sloppy direction** (small eigenvalue): β and τ are confounded —
  a change in β can be compensated by a small change in τ with almost no change
  in observed outputs. The FIM condition number κ typically ranges from 10²–10⁸
  across the landscape.

- At extreme τ (< 0.2 or > 0.6), the model enters trivial fixed points: agents
  barely move (low τ, everyone satisfied) or move randomly (high τ, nobody
  satisfied). In both cases κ → ∞ and neither parameter is identifiable.

**Phase II — (μ, σ, β) landscape:**

- **Heterogeneity slightly reduces total segregation:** same seed, same μ=0.4,
  β=5 gives D ≈ 0.93 (σ=0) vs D ≈ 0.90 (σ=0.15). Tolerant agents (low τ) are
  happy anywhere and do not reinforce cluster formation, dampening the
  segregation feedback loop. The heterogeneous model also converges more slowly.

- **σ near zero is sloppy:** when σ is small, the heterogeneous model is nearly
  identical to the homogeneous one. The summary statistics barely change as σ
  varies from 0 to 0.05, making σ unidentifiable in this region.

- **Moderate σ improves identifiability:** at σ ≈ 0.1–0.2, the spread of
  tolerances creates richer dynamics (genuinely different subpopulations) that
  make the parameters more distinguishable. The FIM condition number actually
  decreases relative to σ = 0 in this regime.

- **High μ with low σ is catastrophically sloppy:** at (μ=0.7, σ=0.05) the
  smallest FIM eigenvalue collapses to ~0.05 (roughly 10⁷ times smaller than
  λ_max). One direction in (μ, σ, β) space is essentially invisible in the
  outputs — the model is in a trivial high-tolerance fixed point and neither
  parameter can be recovered.

- **The stiff/sloppy directions rotate** across (μ, σ) space, unlike Phase I
  where τ was always the stiff direction. There is no single "unidentifiable
  parameter" in Phase II — it depends on where you are in parameter space.

**Phase III — mean-field comparison (open):**

The mean-field replaces Gumbel noise with its expectation:
`p_stay = σ(β·(2·sat−1)/τ_g)` — fully deterministic. The mean-field FIM uses
Σ = I (no noise) and is ~20× faster to compute (no Monte Carlo). At a spot
check (τ=0.4, β=5), κ_mf ≈ 130. Whether the sloppy structure persists across
the full landscape is the open question Section 13 of the notebook addresses.

---

## Architecture — where things live

```
src/abm_geometry/
├── types.py          WorldState (chex.dataclass), SummaryStats, jaxtyping aliases
├── config.py         Config dataclass, YAML loader
├── rng.py            PRNG helpers (splitting, run-key from git sha)
├── schelling/        The differentiable Schelling simulator
│   ├── state.py        init_world(key, cfg) -> WorldState
│   ├── neighbours.py   similar_fraction via 2D convolution (Moore neighbourhood)
│   ├── satisfaction.py soft_satisfaction(occ, tol, beta) -> Float[H,W]
│   ├── move_rules.py   Gumbel-softmax rule; MoveRule protocol added in Phase II
│   ├── step.py         one_step(state, key, params) -> WorldState  [pure]
│   └── simulate.py     simulate(key, params, cfg, T) -> WorldState [jit'd scan]
├── statistics/       Summary statistics — all differentiable
│   ├── segregation.py  dissimilarity_index, moran_i
│   ├── clusters.py     cluster-size distribution (soft surrogate)
│   └── summary.py      stats_fn(state) -> SummaryStats namedtuple
├── geometry/         Scientific core — FIM and landscape (main deliverable)
│   ├── jacobian.py     compute_jacobian(stats_fn, params) — picks fwd/rev by shape
│   ├── fim.py          fisher_information(J, noise_cov) -> FIM
│   └── spectrum.py     eigendecomp, condition_number, sloppy_modes
├── calibrate/        Gradient-based calibration (Phase I+, currently empty)
├── viz/              All plotting — no science logic here
│   ├── grids.py        plot soft_occupancy as RGB
│   ├── spectra.py      eigenvalue spectrum (log scale)
│   └── landscape.py    condition-number heatmaps over parameter space
└── io/runs.py        run_id, save_results, load_results
```

**Data flow for a FIM computation:**
```
Config + seed
   → init_world        (schelling/state.py)
   → simulate          (schelling/simulate.py)    [jit'd lax.scan over T steps]
   → stats_fn          (statistics/summary.py)    [differentiable]
   → compute_jacobian  (geometry/jacobian.py)     [jacfwd, p < d]
   → fisher_information(geometry/fim.py)
   → eigendecomp       (geometry/spectrum.py)
   → save + plot
```

---

## State representation

`WorldState` is a frozen `chex.dataclass` — a PyTree-registered frozen dataclass:

```python
@chex.dataclass(frozen=True)
class WorldState:
    soft_occupancy: Float[Array, "H W 3"]  # axis-2: [empty, A, B], sums to 1
    tolerances:     Float[Array, "H W"]    # per-cell tolerance (ignored if empty)
    step:           Int[Array, ""]         # scalar step counter
```

`tolerances` is *always* a full H×W array. Homogeneous: `jnp.full((H,W), τ)`.
Heterogeneous: sampled from `N(μ, σ²)` once at initialisation, frozen in state.
**The simulator is identical in both phases. Only the tolerance initialiser differs.**

---

## Coding conventions

- **Pure functions everywhere.** Every function is `(inputs) -> outputs`. No global
  state, no mutation, no side effects inside jit-traced code.
- **Type annotations with `jaxtyping`.** Every array parameter and return gets a shape
  annotation. `Float[Array, "H W 3"]` not `jnp.ndarray`.
- **Explicit PRNG keys.** Pass keys explicitly; split inside `lax.scan`. Never store
  a key in a module or global.
- **No Python loops over agents or cells.** Use `jnp` ops, 2D convolution, `vmap`.
  Python loops are silently unrolled by JAX's tracing — huge graphs, no vectorisation.
- **`chex.dataclass(frozen=True)` for all state containers.** Update via
  `dataclasses.replace(state, field=new_val)`.
- **Short, focused modules.** If a file exceeds ~120 lines it probably has two jobs.
- **Experiments are scripts, not libraries.** `experiments/*.py` are entry points.
  Nothing in `src/` imports from them. Reusable logic goes in `src/`.
- **Config is a dataclass, loaded from YAML.** No argparse in science code.
- **Every experiment is reproducible.** Config saved alongside results. `run_id`
  encodes timestamp + git sha.

---

## What to NEVER do

| Don't | Why |
|---|---|
| Python `for` loop over cells / agents | JAX traces static graphs; loops unroll → huge compile time and no vectorisation |
| `np.array(jax_arr)`, `.item()`, `int(x)` inside jit | Breaks tracing; silently returns a compile-time constant |
| `if jax_scalar:` as a Python conditional | JAX can't branch on traced values; use `jnp.where` |
| `print(x)` inside jit | Runs at trace time, not runtime; use `jax.debug.print` |
| `jax.random.seed()` or global RNG state | JAX has no global RNG; always split and pass keys explicitly |
| `class SchellingModel:` or ABM base class | Functional style only; state is a PyTree, not an object |
| Import from `experiments/` | Experiments are entry points, not libraries |
| Storing a JAX array outside a PyTree | JAX can't trace through it; use `chex.dataclass` |
| Hard-coding `H=30, W=30` in simulation code | Grid size lives in `Config`; all ops derive shape from arrays |
| Using `flax` or `equinox` modules | No neural nets here; plain JAX + chex is the right level |
| `from jax import numpy as np` (shadowing NumPy) | Confuses linters; always `import jax.numpy as jnp` |

---

## Phase roadmap (quick reference)

| Phase | Goal | Parameters | Net new code |
|---|---|---|---|
| I | FIM landscape, homogeneous | τ (scalar) | Core simulator + geometry/ |
| II | Heterogeneous agents | (μ, σ²) | Tolerance initialiser only; MoveRule protocol |
| III | Mean-field analytical limit | analytical | `theory/` module + comparison scripts |
| IV | Second model (voter) | model-specific | `voter/` package; geometry/ unchanged |

---

## Running things

```bash
# install
uv sync

# run milestone 0 experiment
python experiments/exp_001_milestone0.py

# tests
pytest tests/ -v

# type check
pyright src/
```

---

## Implementation notes and gotchas

> **For Claude Code sessions:** whenever you discover something non-obvious during
> implementation — a JAX pitfall, a numerics quirk, a design assumption that turned
> out to be wrong — append a bullet here before finishing the task. One line per
> entry, dated. Do not remove entries; they are a record of real surprises.
> Format: `- [YYYY-MM-DD] <what was discovered and why it matters>`

<!-- entries appended below as they are discovered -->
- [2026-05-16] `float(tau)` inside a differentiable pipeline silently breaks autodiff — it extracts a concrete Python float from the traced JAX value at trace time, so the gradient is zero. Always set tolerances via `state.replace(tolerances=jnp.full((H, W), tau))` where `tau` stays a JAX array throughout.
- [2026-05-16] Mass conservation: when density=0.8, total agent mass trying to move (~30 units) can exceed total empty capacity (~19 cells), causing >10% mass loss after renormalisation. Fix: cap moving mass via `cap_scale = jnp.minimum(total_empty / (total_moving + 1e-8), 1.0)` before distributing. Gradient flows through `jnp.minimum`.
- [2026-05-16] Epsilon reuse in move_rules causes micro mass leak: using the same `+1e-8` epsilon in both the cap denominator and the empty-cell weight normaliser makes `sum(empty_weight) < 1`. Keep the two epsilons separate — cap uses `jnp.sum(empty) / (total_moving + 1e-8)`, weight uses `empty / jnp.maximum(jnp.sum(empty), 1e-8)`.
- [2026-05-16] Module identity mismatch: if tests import `from src.abm_geometry.x import Foo` but production code imports `from abm_geometry.x import Foo`, Python treats them as different modules — `isinstance` checks fail and `chex.dataclass` type assertions break. Fix: add `tests/conftest.py` that inserts `src/` into `sys.path`, then use bare `abm_geometry` imports everywhere.
- [2026-05-16] `lax.scan` requires `cfg` to be a static argument (`static_argnums=(2,)`) because `length=cfg.T` must be a compile-time constant. Passing `cfg` as a regular JAX argument causes a trace error.
- [2026-05-16] The Python loop over 8 Moore neighbours in `neighbours.py` is intentional — JAX unrolls it to 8 `jnp.roll` ops at trace time (the loop is over compile-time constants, not array dimensions). It is not a vectorisation bug.
- [2026-05-16] `vmap` over seeds: always split the key before passing to `init_world` and `simulate` separately — `k_init, k_sim = jax.random.split(k)` — otherwise both draws share entropy from the same key, producing correlated initial grids and simulation trajectories.
- [2026-05-17] Type-averaged satisfaction in `move_rules.py` silently kills segregation: using `occ_A*sat_A + occ_B*sat_B` as a single shared p_move means A and B agents at the same cell have identical move rates regardless of type-specific neighbourhood composition. The uniform redistribution then acts as a mean-field fixed point — D converges to 0 instead of increasing. Fix: compute independent `p_move_A` (from `sat_A` + its own Gumbel draw) and `p_move_B` separately, so B agents move out of A-dominated cells and vice versa. See `move_rules.py` and `satisfaction.py:type_satisfaction`.
- [2026-05-17] Cell-level dissimilarity index is always 1.0 for a one-hot initial state (every cell is pure A or pure B), regardless of spatial arrangement. It only becomes meaningful once the soft model creates fractional occupancies. Starting D=1 and falling is therefore NOT evidence of correct Schelling segregation — look for D stabilising at a value > 0 after initial softening.
