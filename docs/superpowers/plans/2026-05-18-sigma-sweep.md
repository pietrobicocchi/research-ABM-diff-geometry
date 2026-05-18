# Phase IV: σ-Sweep Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the primary experiment testing whether population heterogeneity σ acts as a landscape regulariser — computing the full stochastic FIM landscape κ(τ,β) for 7 σ values and tracking 5 metrics that should all move in predicted directions if the conjecture holds.

**Architecture:** Four independent components: (1) `geometry/landscape_metrics.py` — 5 scalar metrics from a κ grid; (2) `viz/sigma_evolution.py` — 3 plot functions for the evolution plots; (3) `experiments/exp_005_sigma_sweep.py` — the ~45-min sweep script that saves everything to disk; (4) notebook section 14 that loads pre-computed results and shows plots. The notebook never recomputes — it only loads.

**Tech Stack:** JAX (jacfwd, vmap, lax.scan), numpy, matplotlib, jaxtyping

---

## File Map

| File | Action |
|------|--------|
| `src/abm_geometry/geometry/landscape_metrics.py` | create |
| `src/abm_geometry/viz/sigma_evolution.py` | create |
| `experiments/exp_005_sigma_sweep.py` | create |
| `notebooks/01_schelling_model.ipynb` | modify — add section 14 |
| `tests/test_landscape_metrics.py` | create |

No changes to any existing source file.

---

## Task 1: Create `geometry/landscape_metrics.py`

**Files:**
- Create: `src/abm_geometry/geometry/landscape_metrics.py`
- Create: `tests/test_landscape_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_landscape_metrics.py`:

```python
import numpy as np
import pytest

from abm_geometry.geometry.landscape_metrics import (
    boundary_sharpness,
    corridor_area,
    kappa_max,
    kappa_median,
    log_kappa_variance,
)


def test_kappa_max():
    grid = np.array([[10.0, 100.0], [50.0, 1000.0]])
    assert kappa_max(grid) == 1000.0


def test_kappa_median():
    """Median of log10([10, 100, 1000, 10000]) = median([1,2,3,4]) = 2.5"""
    grid = np.array([[10.0, 100.0], [1000.0, 10000.0]])
    assert abs(kappa_median(grid) - 2.5) < 1e-6


def test_log_kappa_variance_uniform():
    """All same value → log10 values all equal → variance = 0."""
    grid = np.ones((4, 4)) * 100.0
    assert log_kappa_variance(grid) == pytest.approx(0.0, abs=1e-10)


def test_corridor_area_all_below():
    grid = np.ones((3, 3)) * 500.0
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(1.0)


def test_corridor_area_all_above():
    grid = np.ones((3, 3)) * 2000.0
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(0.0)


def test_corridor_area_half():
    """Half below threshold → 0.5."""
    grid = np.array([[500.0, 2000.0], [500.0, 2000.0]])
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(0.5)


def test_boundary_sharpness_hard_step():
    """Hard step from below low_threshold to above high_threshold.

    p1 = [0.1, 0.3, 0.5, 0.7], two p2 columns.
    Rows 0-1: kappa=50 (below low_threshold=100)
    Rows 2-3: kappa=50000 (above high_threshold=10000)

    For each column: last row below low_threshold is index 1 (p1=0.3),
    first row above high_threshold is index 2 (p1=0.5).
    Width = 0.5 - 0.3 = 0.2.
    """
    p1 = np.array([0.1, 0.3, 0.5, 0.7])
    p2 = np.array([1.0, 2.0])
    kappa = np.array([
        [50.0,   50.0],
        [50.0,   50.0],
        [50000.0, 50000.0],
        [50000.0, 50000.0],
    ])
    width = boundary_sharpness(p1, p2, kappa, low_threshold=100.0, high_threshold=10000.0)
    assert abs(width - 0.2) < 1e-6


def test_boundary_sharpness_no_transition():
    """All below low_threshold → no transition found → returns 0.0."""
    p1 = np.array([0.1, 0.3, 0.5, 0.7])
    p2 = np.array([1.0, 2.0])
    kappa = np.ones((4, 2)) * 50.0
    assert boundary_sharpness(p1, p2, kappa) == pytest.approx(0.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
source .venv/bin/activate && python -m pytest tests/test_landscape_metrics.py -v 2>&1 | tail -5
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Create `src/abm_geometry/geometry/landscape_metrics.py`**

```python
import numpy as np


def kappa_max(kappa_grid: np.ndarray) -> float:
    """Peak condition number across the grid."""
    return float(np.max(kappa_grid))


def kappa_median(kappa_grid: np.ndarray) -> float:
    """Median of log10(κ) across the grid."""
    return float(np.median(np.log10(np.clip(kappa_grid, 1.0, None))))


def log_kappa_variance(kappa_grid: np.ndarray) -> float:
    """Variance of log10(κ) — spread of the landscape.

    Zero means all points have the same condition number (maximally uniform).
    Decreasing variance means σ is homogenising the landscape.
    """
    log_k = np.log10(np.clip(kappa_grid, 1.0, None))
    return float(np.var(log_k))


def corridor_area(kappa_grid: np.ndarray, threshold: float = 1000.0) -> float:
    """Fraction of grid cells with κ < threshold (the identifiable corridor)."""
    return float(np.mean(kappa_grid < threshold))


def boundary_sharpness(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_grid: np.ndarray,
    low_threshold: float = 100.0,
    high_threshold: float = 10000.0,
) -> float:
    """Width (in p1 units) of the κ transition zone, averaged over p2 columns.

    For each p2 column: measures the p1 distance between the last cell
    below low_threshold and the first cell above high_threshold.
    A wide transition = soft/gradual boundary. A narrow transition = sharp.
    Returns 0.0 if no valid transition is found in any column.
    """
    widths = []
    for j in range(kappa_grid.shape[1]):
        col = kappa_grid[:, j]
        below = np.where(col < low_threshold)[0]
        above = np.where(col > high_threshold)[0]
        if len(below) > 0 and len(above) > 0:
            p1_lo = p1_grid[below[-1]]   # last p1 value still below low_threshold
            p1_hi = p1_grid[above[0]]    # first p1 value already above high_threshold
            if p1_hi > p1_lo:
                widths.append(p1_hi - p1_lo)
    return float(np.mean(widths)) if widths else 0.0
```

- [ ] **Step 4: Run all tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_landscape_metrics.py -v
```
Expected: 8/8 pass

- [ ] **Step 5: Commit**

```bash
git add src/abm_geometry/geometry/landscape_metrics.py tests/test_landscape_metrics.py
git commit -m "feat: add geometry/landscape_metrics — 5 scalar metrics for FIM landscape characterisation"
```

---

## Task 2: Create `viz/sigma_evolution.py`

**Files:**
- Create: `src/abm_geometry/viz/sigma_evolution.py`

- [ ] **Step 1: Create `src/abm_geometry/viz/sigma_evolution.py`**

```python
import matplotlib.pyplot as plt
import numpy as np

_METRIC_LABELS = {
    "kappa_max":           ("κ_max",                    "Predicted: ↓"),
    "kappa_median":        ("median(log₁₀ κ)",          "Predicted: ↓"),
    "log_kappa_variance":  ("Var(log₁₀ κ)",             "Predicted: ↓"),
    "corridor_area":       ("Corridor area (κ < 1000)", "Predicted: ↑"),
    "boundary_sharpness":  ("Boundary width (τ units)", "Predicted: ↑"),
}


def plot_sigma_landscape_panel(
    sigma_values: list,
    kappa_grids: list,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    xlabel: str = "τ",
    ylabel: str = "β",
) -> plt.Figure:
    """Seven-panel heatmap of κ(τ,β), one panel per σ, shared colour scale."""
    n = len(sigma_values)
    log_grids = [np.log10(np.clip(kg, 1.0, None)) for kg in kappa_grids]
    vmin = min(g.min() for g in log_grids)
    vmax = max(g.max() for g in log_grids)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, sigma, log_kg in zip(axes, sigma_values, log_grids):
        im = ax.pcolormesh(
            p2_grid, p1_grid, log_kg,
            cmap="plasma", shading="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"σ = {sigma:.2f}", fontsize=10)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_ylabel(xlabel, fontsize=9)

    plt.colorbar(im, ax=axes[-1], label="log₁₀(κ)", shrink=0.8)
    plt.tight_layout()
    return fig


def plot_sigma_metric_curves(
    sigma_values: list,
    metrics: dict,
) -> plt.Figure:
    """Four-panel figure: one subplot per metric showing value vs σ.

    metrics: dict with keys matching _METRIC_LABELS (kappa_max,
    kappa_median, log_kappa_variance, corridor_area, boundary_sharpness).
    Boundary_sharpness shares a panel with corridor_area (right y-axis)
    to keep to 4 panels.
    """
    plot_keys = ["kappa_max", "kappa_median", "log_kappa_variance", "corridor_area"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    sigmas = np.array(sigma_values)

    for ax, key in zip(axes, plot_keys):
        label, prediction = _METRIC_LABELS[key]
        ax.plot(sigmas, metrics[key], "o-", color="#2b6cb0", lw=2, ms=7)
        ax.set_xlabel("σ")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n{prediction}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Overlay boundary_sharpness on the corridor_area panel (right y-axis)
        if key == "corridor_area" and "boundary_sharpness" in metrics:
            ax2 = ax.twinx()
            ax2.plot(sigmas, metrics["boundary_sharpness"], "s--",
                     color="#c05621", lw=2, ms=6, label="Boundary width ↑")
            ax2.set_ylabel("Boundary width (τ units)", color="#c05621")
            ax2.tick_params(axis="y", labelcolor="#c05621")
            ax2.legend(loc="upper left", fontsize=8)

    plt.suptitle("Landscape metrics vs population heterogeneity σ", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_sigma_kappa_histograms(
    sigma_values: list,
    kappa_grids: list,
) -> plt.Figure:
    """Overlaid histograms of log10(κ), one per σ value.

    Shows whether the κ distribution compresses toward lower values as σ
    increases (compression = regularisation).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_values)))

    for sigma, kg, color in zip(sigma_values, kappa_grids, colors):
        log_k = np.log10(np.clip(kg.ravel(), 1.0, None))
        ax.hist(log_k, bins=20, alpha=0.45, color=color,
                label=f"σ={sigma:.2f}", density=True)

    ax.set_xlabel("log₁₀(κ)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of log₁₀(κ) across the (τ,β) grid for each σ")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return fig
```

- [ ] **Step 2: Smoke-test imports**

```bash
source .venv/bin/activate && python -c "
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from abm_geometry.viz.sigma_evolution import (
    plot_sigma_landscape_panel, plot_sigma_metric_curves, plot_sigma_kappa_histograms
)
import numpy as np
p1 = np.linspace(0.1, 0.8, 5)
p2 = np.linspace(2.0, 15.0, 5)
grids = [np.random.uniform(10, 10000, (5,5)) for _ in range(3)]
sigmas = [0.0, 0.1, 0.2]
metrics = {k: np.array([1.0, 0.8, 0.6]) for k in
    ['kappa_max','kappa_median','log_kappa_variance','corridor_area','boundary_sharpness']}
plot_sigma_landscape_panel(sigmas, grids, p1, p2); plt.close('all')
plot_sigma_metric_curves(sigmas, metrics); plt.close('all')
plot_sigma_kappa_histograms(sigmas, grids); plt.close('all')
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/abm_geometry/viz/sigma_evolution.py
git commit -m "feat: add viz/sigma_evolution — 3 plot functions for σ-sweep results"
```

---

## Task 3: Create `exp_005_sigma_sweep.py`

**Files:**
- Create: `experiments/exp_005_sigma_sweep.py`

- [ ] **Step 1: Create `experiments/exp_005_sigma_sweep.py`**

```python
"""Phase IV: σ-sweep experiment.

Tests whether population heterogeneity σ acts as a landscape regulariser by
computing the full stochastic FIM κ(τ,β) landscape for 7 σ values and tracking
5 landscape metrics.

Paper-quality run: set N_GRID=15, K_NOISE=20 (no other changes needed).
Runtime at defaults (N_GRID=10, K_NOISE=10): ~45 min on CPU.
Results saved to outputs/<run_id>/sigma_sweep_results.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from abm_geometry.config import Config
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.landscape_metrics import (
    boundary_sharpness,
    corridor_area,
    kappa_max,
    kappa_median,
    log_kappa_variance,
)
from abm_geometry.geometry.spectrum import condition_number, eigendecomp
from abm_geometry.io.runs import make_run_id, save_results
from abm_geometry.rng import make_key
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.viz.sigma_evolution import (
    plot_sigma_kappa_histograms,
    plot_sigma_landscape_panel,
    plot_sigma_metric_curves,
)

# ── Configuration (adjust these two lines for the paper-quality run) ──────────
N_GRID   = 10   # parameter grid size per axis; set to 15 for paper run
K_NOISE  = 10   # noise-cov samples per grid point; set to 20 for paper run

CFG        = Config(H=30, W=30, T=50, seed=42, tau=0.4, beta=5.0)
TAU_GRID   = np.linspace(0.1, 0.8, N_GRID)
BETA_GRID  = np.linspace(2.0, 15.0, N_GRID)
SIGMA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

_SIM_BETA = jax.jit(simulate_with_beta, static_argnums=(2,))


def _make_pipeline(k_init, k_sim, sigma, eps):
    """Returns stats_fn(params=[tau, beta]) for jax.jacfwd.

    sigma and eps are Python/numpy values — not traced by jacfwd.
    Only params[0] (tau) and params[1] (beta) are traced.
    eps is a fixed H×W noise array (quenched disorder reparametrisation).
    """
    def pipeline(params):
        t, b = params[0], params[1]
        if sigma == 0.0:
            tol = jnp.full((CFG.H, CFG.W), t)
        else:
            tol = jnp.clip(t + sigma * eps, 0.001, 0.999)
        s = init_world(k_init, CFG).replace(tolerances=tol)
        return stats_array_fn(_SIM_BETA(k_sim, s, CFG, b), b)
    return pipeline


def _make_run_fn(sigma, tau_val, beta_val, eps):
    """Returns run_fn(key) -> Float[4] for estimate_noise_cov.

    tau_val and beta_val are concrete Python floats (not traced).
    Tolerances are fixed (same quenched eps as pipeline); randomness
    comes only from different init/sim keys across the K runs.
    """
    def run_fn(key):
        ki, ks = jax.random.split(key)
        if sigma == 0.0:
            tol = jnp.full((CFG.H, CFG.W), tau_val)
        else:
            tol = jnp.clip(tau_val + sigma * eps, 0.001, 0.999)
        s = init_world(ki, CFG).replace(tolerances=tol)
        return stats_array_fn(_SIM_BETA(ks, s, CFG, beta_val), beta_val)
    return run_fn


def _compute_landscape(sigma, k_init, k_sim, k_cov, k_eps):
    """Compute the N_GRID×N_GRID κ(τ,β) landscape for a given σ."""
    eps = jax.random.normal(k_eps, (CFG.H, CFG.W)) if sigma > 0.0 else None
    kappa_grid = np.zeros((N_GRID, N_GRID))

    for i, tau in enumerate(TAU_GRID):
        for j, beta in enumerate(BETA_GRID):
            params   = jnp.array([tau, beta])
            pipeline = _make_pipeline(k_init, k_sim, sigma, eps)
            run_fn   = _make_run_fn(sigma, float(tau), float(beta), eps)

            J    = compute_jacobian(pipeline, params)
            Sig  = estimate_noise_cov(run_fn, k_cov, K=K_NOISE)
            F    = fisher_information(J, Sig)
            v, _ = eigendecomp(F)
            kappa_grid[i, j] = float(condition_number(v))

        print(f"    τ={tau:.2f} ✓", end="  ", flush=True)
    print()
    return kappa_grid


def main():
    master_key = make_key(CFG.seed)
    k_init, k_sim, k_cov, k_eps = jax.random.split(master_key, 4)

    all_kappa_grids = []
    metrics = {k: [] for k in [
        "kappa_max", "kappa_median", "log_kappa_variance",
        "corridor_area", "boundary_sharpness",
    ]}

    for sigma in SIGMA_VALUES:
        print(f"\nσ = {sigma:.2f}  ({N_GRID}×{N_GRID} grid, K={K_NOISE})")
        kg = _compute_landscape(sigma, k_init, k_sim, k_cov, k_eps)
        all_kappa_grids.append(kg)

        metrics["kappa_max"].append(kappa_max(kg))
        metrics["kappa_median"].append(kappa_median(kg))
        metrics["log_kappa_variance"].append(log_kappa_variance(kg))
        metrics["corridor_area"].append(corridor_area(kg))
        metrics["boundary_sharpness"].append(
            boundary_sharpness(TAU_GRID, BETA_GRID, kg)
        )
        print(
            f"  κ_max={metrics['kappa_max'][-1]:.1f}  "
            f"corridor={metrics['corridor_area'][-1]:.2f}  "
            f"sharpness={metrics['boundary_sharpness'][-1]:.3f}"
        )

    # Convert to arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])
    kappa_grids_arr = np.array(all_kappa_grids)   # (7, N_GRID, N_GRID)
    sigma_arr       = np.array(SIGMA_VALUES)

    # Save
    run_id  = make_run_id()
    run_dir = save_results(run_id, CFG, {
        "kappa_grids":  kappa_grids_arr,
        "sigma_values": sigma_arr,
        "tau_grid":     TAU_GRID,
        "beta_grid":    BETA_GRID,
        **{k: v for k, v in metrics.items()},
    })
    np.savez(
        run_dir / "sigma_sweep_results.npz",
        kappa_grids=kappa_grids_arr,
        sigma_values=sigma_arr,
        tau_grid=TAU_GRID,
        beta_grid=BETA_GRID,
        **{k: v for k, v in metrics.items()},
    )

    # Plots
    fig1 = plot_sigma_landscape_panel(
        SIGMA_VALUES, all_kappa_grids, TAU_GRID, BETA_GRID
    )
    plt.suptitle("FIM landscape κ(τ,β) as σ increases", fontsize=13, y=1.01)
    fig1.savefig(run_dir / "landscapes_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_sigma_metric_curves(SIGMA_VALUES, metrics)
    fig2.savefig(run_dir / "metric_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plot_sigma_kappa_histograms(SIGMA_VALUES, all_kappa_grids)
    fig3.savefig(run_dir / "kappa_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Summary
    print(f"\n=== Results saved to {run_dir} ===")
    print("Metric evolution (σ=0 → σ=0.30):")
    predicted_up = {"corridor_area", "boundary_sharpness"}
    confirmed = 0
    for name, values in metrics.items():
        direction = "↑" if values[-1] > values[0] else "↓"
        expected  = "↑" if name in predicted_up else "↓"
        match     = "✓" if direction == expected else "✗"
        confirmed += int(direction == expected)
        print(f"  {match} {name:<22} {expected} predicted  "
              f"({values[0]:.3f} → {values[-1]:.3f}  {direction})")
    print(f"\n{confirmed}/5 metrics support the regularisation conjecture.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test a single (σ, τ, β) point**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
import jax, jax.numpy as jnp
from abm_geometry.config import Config
from abm_geometry.rng import make_key
from abm_geometry.schelling.state import init_world
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number

cfg = Config(H=30, W=30, T=50, seed=42, tau=0.4, beta=5.0)
sim = jax.jit(simulate_with_beta, static_argnums=(2,))
key = make_key(42)
ki, ks, kc, ke = jax.random.split(key, 4)

for sigma in [0.0, 0.15]:
    eps = jax.random.normal(ke, (cfg.H, cfg.W)) if sigma > 0 else None

    def pipe(p):
        t, b = p[0], p[1]
        tol = jnp.full((cfg.H, cfg.W), t) if sigma == 0.0 else jnp.clip(t + sigma * eps, 0.001, 0.999)
        s = init_world(ki, cfg).replace(tolerances=tol)
        return stats_array_fn(sim(ks, s, cfg, b), b)

    def run(k):
        ki2, ks2 = jax.random.split(k)
        tol = jnp.full((cfg.H, cfg.W), 0.4) if sigma == 0.0 else jnp.clip(0.4 + sigma * eps, 0.001, 0.999)
        s = init_world(ki2, cfg).replace(tolerances=tol)
        return stats_array_fn(sim(ks2, s, cfg, 5.0), 5.0)

    J = compute_jacobian(pipe, jnp.array([0.4, 5.0]))
    S = estimate_noise_cov(run, kc, K=5)
    F = fisher_information(J, S)
    v, _ = eigendecomp(F)
    print(f'sigma={sigma}  kappa={float(condition_number(v)):.1f}')

print('PASS')
"
```
Expected: prints two finite kappa values and `PASS`

- [ ] **Step 3: Run full test suite to catch regressions**

```bash
source .venv/bin/activate && python -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add experiments/exp_005_sigma_sweep.py
git commit -m "feat: add exp_005_sigma_sweep — Phase IV heterogeneity regularisation experiment"
```

---

## Task 4: Add notebook section 14

**Files:**
- Modify: `notebooks/01_schelling_model.ipynb`

Section 14 loads pre-computed results from the most recent `sigma_sweep_results.npz`
in `outputs/`. It never recomputes — the heavy work runs in `exp_005`.

- [ ] **Step 1: Add section 14 cells via Python**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
source .venv/bin/activate && python3 - <<'EOF'
import json, uuid

NB = "notebooks/01_schelling_model.ipynb"

def cell(cell_type, source):
    return {
        "id": uuid.uuid4().hex[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }

new_cells = [

cell("markdown", """\
## 14. Population Heterogeneity as a Landscape Regulariser

**The conjecture:** increasing population heterogeneity σ *flattens* the FIM
landscape — reducing the peak condition number, widening the identifiable
corridor, and softening the sharp boundary between identifiable and sloppy
regions.

**The mechanism:** a heterogeneous population is an ensemble of agents with
different tolerance thresholds. Ensemble averaging smooths loss landscapes
(a result from both DL theory and the physics of quenched disorder). The
FIM landscape should reflect this smoothing.

We test this by computing the full κ(τ,β) landscape at seven σ values
{0, 0.05, ..., 0.30} and tracking five scalar metrics. Results were
pre-computed by `experiments/exp_005_sigma_sweep.py`.

| Metric | Predicted direction |
|--------|---------------------|
| κ_max | ↓ decreases |
| median(log₁₀ κ) | ↓ decreases |
| Var(log₁₀ κ) | ↓ decreases (landscape becomes more uniform) |
| Corridor area (κ < 1000) | ↑ increases (identifiable region widens) |
| Boundary sharpness | ↑ increases (transition becomes more gradual) |"""),

cell("code", """\
# Load pre-computed results from exp_005_sigma_sweep
from pathlib import Path
import numpy as np
from abm_geometry.viz.sigma_evolution import (
    plot_sigma_landscape_panel, plot_sigma_metric_curves, plot_sigma_kappa_histograms,
)

# Find most recent sigma_sweep_results.npz in outputs/
result_files = sorted(
    Path("../outputs").glob("*/sigma_sweep_results.npz"),
    key=lambda p: p.stat().st_mtime,
)
if not result_files:
    raise FileNotFoundError(
        "No sigma_sweep_results.npz found in outputs/. "
        "Run experiments/exp_005_sigma_sweep.py first."
    )
result_path = result_files[-1]
print(f"Loading: {result_path.parent.name}")

data          = np.load(result_path, allow_pickle=True)
sigma_values  = list(data["sigma_values"])
kappa_grids14 = list(data["kappa_grids"])   # list of (N, N) arrays
tau_vals14    = data["tau_grid"]
beta_vals14   = data["beta_grid"]
metrics14     = {k: data[k] for k in
                 ["kappa_max", "kappa_median", "log_kappa_variance",
                  "corridor_area", "boundary_sharpness"]}

print(f"σ values: {sigma_values}")
print(f"Grid: {kappa_grids14[0].shape}  (N×N parameter grid)")"""),

cell("code", """\
# Seven-panel landscape evolution
fig14a = plot_sigma_landscape_panel(
    sigma_values, kappa_grids14, tau_vals14, beta_vals14, xlabel="τ", ylabel="β"
)
plt.suptitle("FIM landscape κ(τ,β) as σ increases from 0 to 0.30",
             fontsize=13, y=1.02)
plt.show()"""),

cell("code", """\
# Metric evolution curves
fig14b = plot_sigma_metric_curves(sigma_values, metrics14)
plt.show()"""),

cell("code", """\
# κ distribution compression
fig14c = plot_sigma_kappa_histograms(sigma_values, kappa_grids14)
plt.show()

# Scientific conclusion
print("\\n=== Conjecture assessment ===")
predicted_up = {"corridor_area", "boundary_sharpness"}
confirmed = 0
for name, values in metrics14.items():
    direction = "↑" if values[-1] > values[0] else "↓"
    expected  = "↑" if name in predicted_up else "↓"
    match     = "✓" if direction == expected else "✗"
    confirmed += int(direction == expected)
    print(f"  {match} {name:<22}  expected {expected}  "
          f"({values[0]:.3f} → {values[-1]:.3f}  {direction})")
print(f"\\n{confirmed}/5 metrics {'support' if confirmed >= 4 else 'partially support' if confirmed >= 3 else 'do not support'} "
      f"the regularisation conjecture.")"""),
]

with open(NB) as f:
    nb = json.load(f)
nb["cells"].extend(new_cells)
with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Added {len(new_cells)} cells. Total: {len(nb['cells'])}")
EOF
```
Expected: `Added 5 cells. Total: 32`

- [ ] **Step 2: Verify notebook is valid**

```bash
source .venv/bin/activate && python3 -c "
import json
with open('notebooks/01_schelling_model.ipynb') as f:
    nb = json.load(f)
print('Total cells:', len(nb['cells']))
for c in nb['cells'][-5:]:
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'  [{c[\"cell_type\"][:4]}] {src[:60].replace(chr(10),\" \")}')
"
```

- [ ] **Step 3: Run full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all tests pass (including the 8 new landscape_metrics tests)

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_schelling_model.ipynb
git commit -m "feat: add notebook section 14 — σ-sweep results and conjecture assessment"
```

---

## Final verification

- [ ] **Smoke-test the full pipeline end-to-end (single σ)**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
import numpy as np
from abm_geometry.geometry.landscape_metrics import (
    kappa_max, kappa_median, log_kappa_variance, corridor_area, boundary_sharpness
)
import numpy as np

# Synthetic grids representing sigma=0 (sloppy) and sigma=0.3 (less sloppy)
kg0   = np.random.uniform(100, 1e7, (10,10))   # wide range
kg030 = np.random.uniform(100, 1e4, (10,10))   # compressed range

p1 = np.linspace(0.1, 0.8, 10)
p2 = np.linspace(2.0, 15.0, 10)

print('kappa_max:         ', kappa_max(kg0), '->', kappa_max(kg030))
print('kappa_median:      ', kappa_median(kg0), '->', kappa_median(kg030))
print('log_kappa_variance:', log_kappa_variance(kg0), '->', log_kappa_variance(kg030))
print('corridor_area:     ', corridor_area(kg0), '->', corridor_area(kg030))
print('boundary_sharpness:', boundary_sharpness(p1, p2, kg0), '->', boundary_sharpness(p1, p2, kg030))
print('PASS')
"
```
