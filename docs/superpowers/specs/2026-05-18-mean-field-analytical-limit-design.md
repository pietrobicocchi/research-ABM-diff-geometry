# Phase III: Mean-Field Analytical Limit — Design Spec

**Date:** 2026-05-18
**Phase:** III (mean-field analytical FIM)
**Status:** approved, ready for implementation

---

## Goal

Implement a noise-free (mean-field) version of the Schelling simulator and use it to compute
an analytical FIM. Compare `F_mf` against the stochastic `F_sim` from Phases I–II to
determine whether the sloppy eigenspectrum is a deterministic property of the model
equations or an artefact of stochastic fluctuations.

---

## Core Idea

The stochastic step samples Gumbel noise to decide stay/move. Replacing that noise with
its expected value (zero) gives the **mean-field step** — a deterministic map whose fixed
point is the mean-field equilibrium. Differentiating through this fixed-point trajectory
via `jacfwd` yields `J_mf` and `F_mf = J_mf^T J_mf` (Σ=I, since there is no noise).

The comparison `F_mf` vs `F_sim` is interpretable: any difference is **provably** due to
stochasticity, not numerical error.

---

## Mathematics: Mean-Field Step

In `gumbel_softmax_step`, the stochastic stay probability is:

```
p_stay_A = softmax(([β·sat_A, β·(1−sat_A)] + Gumbel) / τ_g)[0]
```

Taking the expectation over Gumbel(0,1) noise for a symmetric 2-class softmax:

```
E[p_stay_A] = σ( β·(2·sat_A − 1) / τ_g )
```

The **mean-field step** uses this expectation directly — no key, no sampling:

```
p_stay_A = σ( β·(2·sat_A − 1) / τ_g )
p_stay_B = σ( β·(2·sat_B − 1) / τ_g )
```

All remaining logic (cap_scale, redistribution via `empty_weight`) is identical to
the existing `gumbel_softmax_step`.

**`T_mf = 200`** steps (default) ensures convergence to the fixed point; the stochastic
simulation uses `cfg.T = 50`.

---

## Mean-Field FIM

```
J_mf = jacfwd(λ θ: stats_array_fn(simulate_mf(θ), θ[1]))(θ)   # Float[4, p]
F_mf = J_mf^T @ J_mf                                            # Σ = I (no noise)
```

**No Monte Carlo needed** — the mean-field is deterministic, so `estimate_noise_cov`
is skipped entirely. This makes the Phase I landscape sweep ~20× faster.

---

## Module Architecture

### New files only — no changes to existing `schelling/`, `geometry/`, or `config.py`

| File | Purpose |
|------|---------|
| `src/abm_geometry/theory/__init__.py` | package marker (empty) |
| `src/abm_geometry/theory/mean_field.py` | `mean_field_step`, `simulate_mean_field`, `simulate_mean_field_with_params` |
| `src/abm_geometry/viz/comparison.py` | 3 comparison plot functions |
| `experiments/exp_004_mean_field_comparison.py` | runs both landscapes, saves overlay plots |
| `tests/test_theory.py` | tests for mean-field step and FIM |
| `notebooks/01_schelling_model.ipynb` | add section 13 |

---

## `theory/mean_field.py`

### `mean_field_step`

```python
def mean_field_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """Deterministic (noise-free) Schelling step.

    Replaces Gumbel-softmax stay/move sampling with its expected value:
        p_stay_type = σ( β·(2·sat_type − 1) / τ_g )
    All other logic (cap_scale, redistribution) is identical to gumbel_softmax_step.
    beta can be a Python float or JAX traced array.
    """
```

### `simulate_mean_field`

```python
def simulate_mean_field(
    init_state: WorldState,
    cfg: Config,
    T_mf: int = 200,
) -> WorldState:
    """Run T_mf deterministic steps via lax.scan. cfg is static."""
```

### `simulate_mean_field_with_params`

```python
def simulate_mean_field_with_params(
    init_state: WorldState,
    cfg: Config,
    tau,          # JAX traced — gradient flows
    beta,         # JAX traced — gradient flows
    T_mf: int = 200,
) -> WorldState:
    """Like simulate_mean_field but tau and beta are dynamic JAX values.

    Sets tolerances = jnp.full((H,W), tau) at each step so ∂/∂tau flows.
    Use this instead of simulate_mean_field when calling jacfwd.
    cfg is still static (lax.scan length = T_mf).
    """
```

---

## `viz/comparison.py`

### `plot_kappa_comparison`

```python
def plot_kappa_comparison(
    p1_grid, p2_grid,
    kappa_sim: np.ndarray,   # shape (N, M)
    kappa_mf:  np.ndarray,   # shape (N, M)
    xlabel: str, ylabel: str,
) -> plt.Figure:
    """Side-by-side log₁₀ κ heatmaps, shared colour scale.
    Left: stochastic sim. Right: mean-field.
    """
```

### `plot_eigenvalue_comparison`

```python
def plot_eigenvalue_comparison(
    vals_sim_list: list,   # list of Float[p] arrays
    vals_mf_list:  list,   # list of Float[p] arrays — same length
    labels: list[str],
    ax=None,
) -> plt.Axes:
    """Brown/Sethna spectrum overlay.
    Sim eigenvalues: solid lines. Mean-field: dashed lines. Same colour per point.
    """
```

### `plot_kappa_ratio`

```python
def plot_kappa_ratio(
    p1_grid, p2_grid,
    kappa_sim: np.ndarray,
    kappa_mf:  np.ndarray,
    xlabel: str, ylabel: str,
    ax=None,
) -> plt.Axes:
    """Heatmap of log₁₀(κ_sim / κ_mf).
    Positive = stochasticity amplifies sloppiness.
    Negative = stochasticity suppresses sloppiness.
    Zero = mean-field is exact.
    """
```

---

## Experiment `exp_004_mean_field_comparison.py`

1. Sweep (τ, β) on a 15×15 grid (same grid as exp_002)
2. At each point compute both `F_sim` (reuse exp_002 logic) and `F_mf` (new)
3. Save `kappa_sim_grid`, `kappa_mf_grid`, plus eigenvalues at 4 representative points
4. Produce three plots: `plot_kappa_comparison`, `plot_kappa_ratio`, `plot_eigenvalue_comparison`

---

## Notebook Section 13: Mean-Field vs Stochastic FIM

**Cells:**

**Markdown:** Explain what the mean-field is (remove Gumbel noise → deterministic fixed
point), why the comparison matters, and what each outcome means scientifically.

**Code cell 1 — compute mean-field landscape:**
```python
from abm_geometry.theory.mean_field import simulate_mean_field_with_params
from abm_geometry.viz.comparison import (
    plot_kappa_comparison, plot_eigenvalue_comparison, plot_kappa_ratio
)

# Reuse tau_vals, beta_vals, kappa_grid10 from Section 10
# Compute mean-field landscape (fast — no noise cov estimation)
kappa_grid_mf = np.zeros((15, 15))
vals_mf_at_points = {}   # same 4 representative points as Section 12

for i, tau_v in enumerate(tau_vals):
    for j, beta_v in enumerate(beta_vals):
        params_ij = jnp.array([tau_v, beta_v])

        def _pipe_mf(p, ki=k10_init, c=cfg):
            t, b = p[0], p[1]
            s = init_world(ki, c)
            final = simulate_mean_field_with_params(s, c, t, b)
            return stats_array_fn(final, b)

        J_mf_ij = compute_jacobian(_pipe_mf, params_ij)
        F_mf_ij = J_mf_ij.T @ J_mf_ij    # Σ = I
        v_mf, _ = eigendecomp(F_mf_ij)
        kappa_grid_mf[i, j] = float(condition_number(v_mf))
    print(f"τ={tau_v:.2f} ✓", end="  ", flush=True)
```

**Code cell 2 — comparison plots:**
```python
fig13a = plot_kappa_comparison(
    tau_vals, beta_vals, kappa_grid10, kappa_grid_mf,
    xlabel="τ", ylabel="β"
)
plt.suptitle("Stochastic vs mean-field condition number κ(τ,β)", fontsize=13, y=1.02)
plt.show()

fig13b, ax13b = plt.subplots(figsize=(7, 4))
plot_kappa_ratio(tau_vals, beta_vals, kappa_grid10, kappa_grid_mf,
                 xlabel="τ", ylabel="β", ax=ax13b)
plt.title("log₁₀(κ_sim / κ_mf): where does stochasticity matter?")
plt.show()
```

**Code cell 3 — eigenvalue spectrum overlay at 2 points:**
```python
# Pick a high-κ and a low-κ point from the sim landscape
# Compute mean-field eigenvalues at same points and overlay
```

---

## Tests (`tests/test_theory.py`)

| Test | What it checks |
|------|---------------|
| `test_mean_field_step_shape` | output shape matches input |
| `test_mean_field_step_sums_to_one` | simplex constraint holds |
| `test_mean_field_step_non_negative` | all values ≥ 0 |
| `test_mean_field_step_no_key_needed` | function signature has no key arg |
| `test_simulate_mean_field_step_counter` | step counter reaches T_mf |
| `test_mean_field_gradient_tau` | ∂D/∂τ is finite via jacfwd |
| `test_mean_field_gradient_beta` | ∂D/∂β is finite via jacfwd |
| `test_mean_field_fim_shape` | F_mf = J^T J has shape (2,2) |

---

## Constraints & Non-Goals

- `T_mf = 200` passed as function argument (not added to Config)
- No changes to `schelling/`, `geometry/`, `config.py`, or existing tests
- No algebraic/symbolic derivation of fixed point (Approach B — deferred)
- No per-cell noise analysis (uniform mean-field only)
- `exp_004` re-runs `F_sim` from scratch (doesn't depend on cached exp_002 results)
