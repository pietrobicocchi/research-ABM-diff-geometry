# Design spec: Differentiable Schelling ABM — loss landscape geometry

**Date:** 2026-05-16
**Status:** approved

---

## 1. Project overview

A JAX implementation of the Schelling segregation model designed to study the Fisher
information geometry of differentiable agent-based models (ABMs). The central question:
how does the FIM eigenspectrum and condition number vary across parameter space, and
how does agent heterogeneity (variance in tolerance) affect identifiability?

This spec covers the full architecture, foundational design decisions, and the minimal
first milestone (Milestone 0). Phases II–IV are scoped briefly at the end.

---

## 2. Design decisions

### 2.1 Library stack

**Choice:** plain JAX + `chex` + `jaxtyping` + `optax` (deferred). No equinox, no flax.

`chex.dataclass(frozen=True)` provides PyTree-registered frozen dataclasses — the right
primitive for immutable simulation state. `jaxtyping` provides shape-annotated array
types that serve as documentation. Equinox/flax add abstraction suited for neural
modules that do not exist here.

### 2.2 State representation

`WorldState` is a frozen `chex.dataclass`:

```python
@chex.dataclass(frozen=True)
class WorldState:
    soft_occupancy: Float[Array, "H W 3"]  # [empty, A, B], sums to 1 along axis 2
    tolerances:     Float[Array, "H W"]    # per-cell tolerance; ignored where empty
    step:           Int[Array, ""]         # scalar step counter
```

`tolerances` is always a full array. Homogeneous case: `jnp.full((H, W), τ)`.
Heterogeneous case: sampled from `N(μ, σ²)` once before simulation begins.
**The simulator is identical in both cases.** Only the tolerance initialiser differs.

### 2.3 Differentiable move rule

Following Quera-Bofarull et al. (Arnau group), the move rule uses Gumbel-softmax
with an optional straight-through estimator:

1. **Soft satisfaction:** `sat(i) = σ(β · (similar_frac(i) − τ(i)))`
2. **Move propensity:** `p_move(i) = (1 − sat(i)) · occupied(i)`
3. **Destination logits:** `score(i→j) = β · (similar_frac_at_j − τ(i))` over empty cells
4. **Selection:** Gumbel-softmax with temperature `τ_g`; optional straight-through:
   `x = x_hard + stop_gradient(x_soft − x_hard)`

`similar_frac_at_j` is *not* counterfactual — it is read directly from the global
convolution output (the fraction of same-type occupancy in j's Moore neighbourhood
under the current `soft_occupancy`). This avoids a nested per-agent computation and
is a standard simplification in relaxed ABMs.

Milestone 0 implements Gumbel-softmax only. The `MoveRule` protocol (for Phase II/III
comparisons) is introduced in Phase II.

### 2.4 PRNG management

No key stored in state. The `lax.scan` body carries `(state, key)`, splitting at each
step: `key, sub = jax.random.split(key)`. One master key per experiment, pinned in
`Config`.

### 2.5 Jacobian strategy

Use `jacfwd` when `p ≤ d` (parameters ≤ summary stats) — this is always true for
Phases I–II where `p ∈ {1, 2}` and `d ≈ 5–10`. `jacrev` becomes relevant only in
Phase III if per-cell tolerances become parameters (`p = H·W`).

`compute_jacobian(stats_fn, params)` in `geometry/jacobian.py` selects automatically
based on shapes. Experiment scripts never call `jacfwd`/`jacrev` directly.

### 2.6 Heterogeneous agents

No code path change. `tolerances` field accepts any array. Phase I passes a constant
array; Phase II passes a sample from a distribution. FIM computation is identical.

### 2.7 Experiment scaffolding

Config is a frozen `dataclass`, optionally loaded from YAML via `omegaconf`. No Hydra,
no argparse in science code. Outputs go to `outputs/<run_id>/` where
`run_id = YYYYMMDD-HHMMSS-<git-sha>`. Config is serialised alongside every result.

---

## 3. Directory structure

```
research-ABM-diff-geometry/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore
├── src/abm_geometry/
│   ├── __init__.py
│   ├── types.py               # WorldState, SummaryStats, shape aliases
│   ├── config.py              # Config dataclass + YAML loader
│   ├── rng.py                 # PRNG helpers
│   ├── schelling/
│   │   ├── __init__.py        # re-exports: init_world, simulate
│   │   ├── state.py           # init_world(key, cfg) -> WorldState
│   │   ├── neighbours.py      # similar_fraction via 2D conv
│   │   ├── satisfaction.py    # soft_satisfaction(occ, tol, beta)
│   │   ├── move_rules.py      # Gumbel-softmax rule (+ MoveRule protocol in Phase II)
│   │   ├── step.py            # one_step(state, key, params) -> WorldState
│   │   └── simulate.py        # simulate(...) via jit'd lax.scan
│   ├── statistics/
│   │   ├── __init__.py
│   │   ├── segregation.py     # dissimilarity_index, moran_i
│   │   ├── clusters.py        # cluster-size distribution (soft surrogate)
│   │   └── summary.py         # SummaryStats namedtuple + stats_fn
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── jacobian.py        # compute_jacobian — picks fwd vs rev by shape
│   │   ├── fim.py             # fisher_information(J, noise_cov)
│   │   └── spectrum.py        # eigendecomp, condition_number, sloppy_modes
│   ├── calibrate/
│   │   └── __init__.py        # placeholder for Phase I gradient descent
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── grids.py           # plot soft_occupancy as RGB
│   │   ├── spectra.py         # eigenvalue spectrum (log scale)
│   │   └── landscape.py       # condition-number heatmaps over parameter space
│   └── io/
│       ├── __init__.py
│       └── runs.py            # run_id, save_results, load_results
├── experiments/
│   ├── configs/
│   │   └── default.yaml
│   └── exp_001_milestone0.py
├── notebooks/
│   └── 00_sanity_check.ipynb
├── tests/
│   ├── test_simulator.py
│   ├── test_gradients.py
│   ├── test_invariances.py
│   └── test_statistics.py
└── outputs/                   # gitignored
```

---

## 4. Data flow

```
Config + seed
   → init_world          schelling/state.py
   → simulate            schelling/simulate.py    [jit'd lax.scan over T steps]
   → stats_fn            statistics/summary.py    [differentiable]
   → compute_jacobian    geometry/jacobian.py      [jacfwd, p < d]
   → fisher_information  geometry/fim.py
   → eigendecomp         geometry/spectrum.py
   → save + plot         io/runs.py, viz/
```

---

## 5. Milestone 0 — definition of done

### What gets built

| File | Responsibility |
|---|---|
| `types.py` | `WorldState`, `SchellingParams`, shape aliases |
| `schelling/state.py` | `init_world(key, H, W, density, ratio)` |
| `schelling/neighbours.py` | `similar_fraction(occ)` via Moore conv |
| `schelling/satisfaction.py` | `soft_satisfaction(occ, tolerances, beta)` |
| `schelling/move_rules.py` | Gumbel-softmax rule only |
| `schelling/step.py` | `one_step(state, key, params)` |
| `schelling/simulate.py` | `simulate(key, params, cfg, T)` — jit'd scan |
| `statistics/segregation.py` | `dissimilarity_index(occ)` — one differentiable scalar |
| `geometry/jacobian.py` | `compute_jacobian(stats_fn, params)` via jacfwd |
| `experiments/exp_001_milestone0.py` | Runs, prints Jacobian + condition number, saves grid |
| `tests/test_gradients.py` | Finite-diff vs jacfwd agreement |

### Done criteria (all five must hold)

| # | Condition | Verification |
|---|---|---|
| 1 | `simulate` runs under `jit` without recompilation | Second call is fast |
| 2 | `simulate` is batchable over seeds | `vmap(simulate, in_axes=(0,...))(keys,...)` returns a `WorldState` whose `soft_occupancy` has shape `(B,H,W,3)` |
| 3 | `dissimilarity_index` is differentiable w.r.t. τ | `jax.grad(...)( τ)` returns a finite float |
| 4 | Finite-diff and `jacfwd` agree to `1e-4` | `test_gradients.py` passes |
| 5 | Jacobian and condition number print without NaN/inf | `exp_001_milestone0.py` completes cleanly |

**Not in scope for Milestone 0:** MoveRule protocol, heterogeneous tolerances, FIM
computation, calibration, landscape scanning, voter model.

---

## 6. Phase roadmap

| Phase | Goal | Parameters | Net new code |
|---|---|---|---|
| I | FIM landscape, homogeneous | τ (scalar) | Core simulator + geometry/ |
| II | Heterogeneous agents | (μ, σ²) | Tolerance initialiser only; add MoveRule protocol |
| III | Mean-field analytical limit | analytical | New `theory/` module, comparison scripts |
| IV | Second model (voter) | model-specific | New `voter/` package; geometry/ unchanged |

---

## 7. Coding conventions (summary — full list in CLAUDE.md)

- Pure functions everywhere; no global state.
- `jaxtyping` shape annotations on every array parameter and return.
- No Python loops over agents or cells — use `jnp` ops, convolution, `vmap`.
- `chex.dataclass(frozen=True)` for all state; update via `dataclasses.replace`.
- `experiments/` are entry points, not libraries. Nothing in `src/` imports from them.
- Every run is reproducible: config serialised with results, `run_id` includes git sha.
