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
