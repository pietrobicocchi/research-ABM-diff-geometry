# Phase IV: σ-Sweep Experiment — Design Spec

**Date:** 2026-05-18  
**Phase:** IV (heterogeneity as landscape regulariser — primary experiment)  
**Status:** approved, ready for implementation  

---

## Goal

Test the central conjecture from `docs/research/thesis-landscape-heterogeneity.md`:
population heterogeneity σ acts as a landscape regulariser — increasing σ
flattens the FIM landscape, softens phase boundaries, and widens the
identifiable corridor.

This is the primary scientific experiment. Results use the full stochastic FIM
(defensible, directly comparable to Phase I/II landscapes). Mean-field
comparison is deferred to an appendix (future work).

---

## Experiment Design

**Independent variable:** σ ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30} — 7 values.

**Fixed parameters:** μ = 0.4, β = 5.0, τ_g = 0.5 (all from default config).

**At each σ:** compute the full stochastic FIM on a 10×10 (τ, β) grid:
- τ ∈ [0.1, 0.8], 10 points
- β ∈ [2.0, 15.0], 10 points
- Noise covariance: K = 10 independent runs per grid point
- Simulation: H=30, W=30, T=50 (default grid)
- FIM: F = J^T Σ⁻¹ J (stochastic, not mean-field)

**Total computation:** 7 × 10 × 10 = 700 FIM points. Estimated runtime: ~45 min.

**Scalability note:** bumping to N_GRID=15, K=20 for the final paper run requires
no code changes — only configuration constants in `exp_005_sigma_sweep.py`.

---

## Conjecture Predictions

If σ acts as a regulariser, the following should hold as σ increases:

| Metric | Predicted direction | Falsification |
|--------|---------------------|---------------|
| κ_max | Decreases (then levels off at σ*) | Increases or no trend |
| median(log κ) | Decreases | Increases or flat |
| Var(log κ) | Decreases (landscape becomes more uniform) | Increases |
| Corridor area (κ < 1000) | Increases (widens) | Decreases |
| Boundary sharpness | Decreases (softer transition) | Increases or no trend |

Both confirmation and falsification are scientifically interesting.

---

## Module Architecture

### New files

| File | Purpose |
|------|---------|
| `src/abm_geometry/geometry/landscape_metrics.py` | 5 scalar metrics from a κ grid |
| `src/abm_geometry/viz/sigma_evolution.py` | 3 plot functions for σ-sweep results |
| `experiments/exp_005_sigma_sweep.py` | main sweep script — runs all 7 landscapes, saves to disk |
| `notebooks/01_schelling_model.ipynb` | add section 14: loads pre-computed results, plots, concludes |
| `tests/test_landscape_metrics.py` | tests for the 5 metric functions |

### No changes to existing files

All existing `schelling/`, `geometry/`, `statistics/`, `config.py` are
untouched. The sweep reuses `simulate_with_beta`, `estimate_noise_cov`,
`fisher_information`, `eigendecomp`, `condition_number` from existing modules.

---

## `geometry/landscape_metrics.py`

Five functions, each taking a `kappa_grid: np.ndarray` of shape `(N, M)`:

```python
def kappa_max(kappa_grid: np.ndarray) -> float:
    """Peak condition number across the grid."""

def kappa_median(kappa_grid: np.ndarray) -> float:
    """Median condition number (log scale: median of log10(kappa))."""

def log_kappa_variance(kappa_grid: np.ndarray) -> float:
    """Variance of log10(κ) — spread of the landscape."""

def corridor_area(kappa_grid: np.ndarray, threshold: float = 1000.0) -> float:
    """Fraction of grid cells with κ < threshold (identifiable corridor)."""

def boundary_sharpness(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_grid: np.ndarray,
    low_threshold: float = 100.0,
    high_threshold: float = 10000.0,
) -> float:
    """Width of the transition zone from low-κ to high-κ, in p1 units.

    Computed as the mean distance (in p1) over all p2 columns between the
    first cell exceeding low_threshold and the first cell exceeding
    high_threshold. Larger value = softer/broader boundary.
    """
```

All functions operate on plain numpy arrays (not JAX). They are called after
the JAX computation is complete.

---

## `viz/sigma_evolution.py`

Three functions for visualising how the landscape changes with σ:

```python
def plot_sigma_landscape_panel(
    sigma_values: list[float],
    kappa_grids: list[np.ndarray],    # len = 7, each (N, M)
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    xlabel: str = "τ",
    ylabel: str = "β",
) -> plt.Figure:
    """7-panel heatmap showing κ(τ,β) for each σ, shared colour scale.
    Title of each panel: σ = {value}.
    """

def plot_sigma_metric_curves(
    sigma_values: list[float],
    metrics: dict,           # keys: "kappa_max", "kappa_median",
                             #       "log_kappa_variance", "corridor_area",
                             #       "boundary_sharpness"
) -> plt.Figure:
    """4-panel figure (one subplot per metric) showing metric vs σ.
    Each panel has σ on x-axis and the metric on y-axis.
    Predicted direction annotated with arrow/label.
    """

def plot_sigma_kappa_histograms(
    sigma_values: list[float],
    kappa_grids: list[np.ndarray],
) -> plt.Figure:
    """Overlaid histograms of log10(κ) for each σ, one colour per σ value.
    Shows compression of the distribution toward lower κ as σ increases.
    """
```

---

## `experiments/exp_005_sigma_sweep.py`

```
Inputs:  none (all config hardcoded; see SIGMA_VALUES, N_GRID, K constants)
Outputs: runs/<run_id>/
           sigma_sweep_results.npz    — all kappa grids + metrics table
           landscapes_panel.png       — 7-panel heatmap
           metric_curves.png          — 4 metric curves vs σ
           kappa_histograms.png       — overlaid histograms
           sweep_summary.txt          — printed conclusions
```

**Key constants at top of file (adjust for paper-quality run):**
```python
SIGMA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_GRID       = 10     # increase to 15 for paper run
K_NOISE      = 10     # increase to 20 for paper run
CFG          = Config(H=30, W=30, T=50, seed=42, tau=0.4, beta=5.0)
TAU_GRID     = np.linspace(0.1, 0.8, N_GRID)
BETA_GRID    = np.linspace(2.0, 15.0, N_GRID)
```

**Loop structure:**
```
for sigma in SIGMA_VALUES:
    for i, tau in enumerate(TAU_GRID):
        for j, beta in enumerate(BETA_GRID):
            params = [tau, beta]
            J      = compute_jacobian(pipeline(sigma, tau, beta), params)
            Sigma  = estimate_noise_cov(run_fn(sigma, tau, beta), key, K_NOISE)
            F      = fisher_information(J, Sigma)
            v, _   = eigendecomp(F)
            kappa_grid[i, j] = condition_number(v)
    compute 5 metrics from kappa_grid
    store kappa_grid and metrics
```

**Differentiable pipeline for each (σ, τ, β) point:**
```python
def make_pipeline(k_init, k_sim, sigma, tau, beta):
    """Returns stats_fn(params=[tau, beta]) for jacfwd.

    tau is the mean of the tolerance distribution (= homogeneous tau at sigma=0).
    eps is fixed per (sigma, point) so the reparametrisation trick gives
    clean gradients: tol = clip(t + sigma * eps, ...) with t traced by jacfwd.
    """
    eps = jax.random.normal(k_init, (CFG.H, CFG.W))  # fixed noise, not traced

    def pipeline(params):
        t, b = params[0], params[1]
        if sigma == 0.0:
            tol = jnp.full((CFG.H, CFG.W), t)
        else:
            tol = jnp.clip(t + sigma * eps, 0.001, 0.999)
        s = init_world(k_init, CFG).replace(tolerances=tol)
        return stats_array_fn(_SIM_BETA(k_sim, s, CFG, b), b)
    return pipeline
```

**Noise cov run_fn:** same pattern — captures sigma as default arg so vmap
does not trace it.

**Progress:** prints `σ={sigma:.2f}, τ={tau:.2f} ✓` per τ-row.

**Saving:** uses existing `io/runs.py` infrastructure (`make_run_id`,
`save_results`). Results saved as `.npz` for easy loading in the notebook.

---

## Notebook Section 14

**Markdown cell:** Explain the σ-sweep experiment, the conjecture, and what
each plot tests.

**Code cell 1 — load results:**
```python
import numpy as np
results = np.load("runs/<run_id>/sigma_sweep_results.npz", allow_pickle=True)
sigma_values = results["sigma_values"]
kappa_grids  = results["kappa_grids"]   # shape (7, 10, 10)
metrics      = results["metrics"].item()  # dict of arrays, each len 7
```

**Code cell 2 — 7-panel landscape evolution:**
```python
from abm_geometry.viz.sigma_evolution import plot_sigma_landscape_panel
fig = plot_sigma_landscape_panel(sigma_values, kappa_grids, tau_vals, beta_vals)
plt.suptitle("FIM landscape κ(τ,β) as σ increases", fontsize=13)
plt.show()
```

**Code cell 3 — metric curves:**
```python
from abm_geometry.viz.sigma_evolution import plot_sigma_metric_curves
fig = plot_sigma_metric_curves(sigma_values, metrics)
plt.show()
```

**Code cell 4 — κ histograms:**
```python
from abm_geometry.viz.sigma_evolution import plot_sigma_kappa_histograms
fig = plot_sigma_kappa_histograms(sigma_values, kappa_grids)
plt.show()
```

**Code cell 5 — scientific conclusion:**
```python
print("=== Conjecture assessment ===")
for metric_name, values in metrics.items():
    trend = "decreasing" if values[-1] < values[0] else "increasing"
    print(f"  {metric_name}: {trend}  ({values[0]:.2f} → {values[-1]:.2f})")
```

---

## Tests (`tests/test_landscape_metrics.py`)

| Test | What it checks |
|------|---------------|
| `test_kappa_max_uniform` | Returns correct max on a known grid |
| `test_corridor_area_all_below` | Returns 1.0 when all κ < threshold |
| `test_corridor_area_all_above` | Returns 0.0 when all κ > threshold |
| `test_log_kappa_variance_uniform` | Returns 0.0 for uniform κ grid |
| `test_boundary_sharpness_step` | Returns narrow width for a hard step, wide for a smooth transition |

---

## Constraints

- All metrics operate on numpy arrays, not JAX arrays (post-computation)
- `exp_005` saves to disk; notebook only loads — never recomputes in-notebook
- `sigma > 0` uses heterogeneous tolerance initialisation (reparametrisation trick);
  `sigma = 0` is homogeneous (same as Phase I)
- No changes to any existing source file
- Paper-quality run: change `N_GRID=15`, `K_NOISE=20` in `exp_005` constants only
