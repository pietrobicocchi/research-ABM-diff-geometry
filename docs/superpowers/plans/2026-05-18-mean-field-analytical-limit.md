# Phase III: Mean-Field Analytical Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a noise-free Schelling simulator (`theory/mean_field.py`), compute the analytical FIM (`F_mf = J_mf^T J_mf`, Σ=I), and produce a comparison against the stochastic FIM to show whether sloppiness is deterministic or noise-driven.

**Architecture:** A new `theory/` package contains `mean_field_step` — identical to `gumbel_softmax_step` except `p_stay = σ(β·(2·sat−1)/τ_g)` replaces the Gumbel sample. `simulate_mean_field_with_params` threads `tau` and `beta` as JAX-traced values through `lax.scan` so `jacfwd` works. The FIM uses `F_mf = J_mf.T @ J_mf` (Σ=I) — no Monte Carlo needed. All existing `schelling/`, `geometry/`, and `config.py` files are untouched.

**Tech Stack:** JAX (jacfwd, lax.scan), jaxtyping, chex, matplotlib, numpy

---

## File Map

| File | Action |
|------|--------|
| `src/abm_geometry/theory/__init__.py` | create (empty) |
| `src/abm_geometry/theory/mean_field.py` | create — 3 functions |
| `src/abm_geometry/viz/comparison.py` | create — 3 plot functions |
| `experiments/exp_004_mean_field_comparison.py` | create — full sweep + plots |
| `tests/test_theory.py` | create — 8 tests |
| `notebooks/01_schelling_model.ipynb` | modify — add section 13 (3 cells) |

---

## Task 1: Create `theory/mean_field.py`

**Files:**
- Create: `src/abm_geometry/theory/__init__.py`
- Create: `src/abm_geometry/theory/mean_field.py`
- Create: `tests/test_theory.py`

- [ ] **Step 1: Create the package marker**

```bash
touch /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry/src/abm_geometry/theory/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_theory.py`:

```python
import inspect

import jax
import jax.numpy as jnp

from abm_geometry.config import Config
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.theory.mean_field import (
    mean_field_step,
    simulate_mean_field,
    simulate_mean_field_with_params,
)

_CFG = Config(H=8, W=8, T=5, seed=0)
_KEY = jax.random.PRNGKey(0)


def test_mean_field_step_shape():
    state = init_world(_KEY, _CFG)
    new_occ = mean_field_step(
        state.soft_occupancy, state.tolerances, _CFG.beta, _CFG.tau_g
    )
    assert new_occ.shape == (8, 8, 3)


def test_mean_field_step_sums_to_one():
    state = init_world(_KEY, _CFG)
    new_occ = mean_field_step(
        state.soft_occupancy, state.tolerances, _CFG.beta, _CFG.tau_g
    )
    sums = jnp.sum(new_occ, axis=-1)
    assert jnp.allclose(sums, jnp.ones((8, 8)), atol=1e-5)


def test_mean_field_step_non_negative():
    state = init_world(_KEY, _CFG)
    new_occ = mean_field_step(
        state.soft_occupancy, state.tolerances, _CFG.beta, _CFG.tau_g
    )
    assert jnp.all(new_occ >= -1e-6)


def test_mean_field_step_no_key_needed():
    sig = inspect.signature(mean_field_step)
    assert "key" not in sig.parameters


def test_simulate_mean_field_step_counter():
    state = init_world(_KEY, _CFG)
    final = simulate_mean_field(state, _CFG, T_mf=10)
    assert int(final.step) == 10


def test_mean_field_gradient_tau():
    cfg = Config(H=8, W=8, T=5, seed=0)
    ki, _ = jax.random.split(_KEY)
    state = init_world(ki, cfg)
    sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))

    def d_from_tau(tau):
        final = sim_mf(state, cfg, tau, jnp.array(cfg.beta))
        return dissimilarity_index(final.soft_occupancy)

    grad = jax.grad(d_from_tau)(jnp.array(0.4))
    assert jnp.isfinite(grad), f"gradient w.r.t. tau not finite: {grad}"


def test_mean_field_gradient_beta():
    cfg = Config(H=8, W=8, T=5, seed=0)
    ki, _ = jax.random.split(_KEY)
    state = init_world(ki, cfg)
    sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))

    def d_from_beta(beta):
        final = sim_mf(state, cfg, jnp.array(cfg.tau), beta)
        return dissimilarity_index(final.soft_occupancy)

    grad = jax.grad(d_from_beta)(jnp.array(5.0))
    assert jnp.isfinite(grad), f"gradient w.r.t. beta not finite: {grad}"


def test_mean_field_fim_shape():
    cfg = Config(H=8, W=8, T=5, seed=0)
    ki, _ = jax.random.split(_KEY)
    state = init_world(ki, cfg)
    sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))

    def pipeline(params):
        t, b = params[0], params[1]
        final = sim_mf(state, cfg, t, b)
        return stats_array_fn(final, b)

    J = compute_jacobian(pipeline, jnp.array([0.4, 5.0]))
    F_mf = J.T @ J
    assert F_mf.shape == (2, 2)
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
source .venv/bin/activate && python -m pytest tests/test_theory.py::test_mean_field_step_shape -v
```
Expected: `ImportError` or `ModuleNotFoundError` (module does not exist yet)

- [ ] **Step 4: Create `src/abm_geometry/theory/mean_field.py`**

```python
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.config import Config
from abm_geometry.schelling.satisfaction import type_satisfaction
from abm_geometry.types import WorldState


def mean_field_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """Deterministic (noise-free) Schelling step.

    Replaces Gumbel-softmax stay/move sampling with its expected value:
        p_stay_type = σ( β·(2·sat_type − 1) / τ_g )

    All other logic (cap_scale, redistribution) is identical to
    gumbel_softmax_step. beta can be a Python float or a JAX traced array.
    """
    sat_A, sat_B = type_satisfaction(soft_occ, tolerances, beta)

    # Expected stay probability (Gumbel noise expectation = 0)
    p_stay_A = jax.nn.sigmoid((2.0 * sat_A - 1.0) * beta / tau_g)
    p_stay_B = jax.nn.sigmoid((2.0 * sat_B - 1.0) * beta / tau_g)

    p_move_A = 1.0 - p_stay_A
    p_move_B = 1.0 - p_stay_B

    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    empty = soft_occ[..., 0]

    total_moving_uncapped = jnp.sum(occ_A * p_move_A) + jnp.sum(occ_B * p_move_B)
    cap_scale = jnp.minimum(jnp.sum(empty) / (total_moving_uncapped + 1e-8), 1.0)
    p_move_eff_A = p_move_A * cap_scale
    p_move_eff_B = p_move_B * cap_scale

    staying_A = occ_A * (1.0 - p_move_eff_A)
    moving_A  = occ_A * p_move_eff_A
    staying_B = occ_B * (1.0 - p_move_eff_B)
    moving_B  = occ_B * p_move_eff_B

    total_moving_A = jnp.sum(moving_A)
    total_moving_B = jnp.sum(moving_B)

    empty_weight = empty / jnp.maximum(jnp.sum(empty), 1e-8)

    new_A     = staying_A + total_moving_A * empty_weight
    new_B     = staying_B + total_moving_B * empty_weight
    new_empty = 1.0 - new_A - new_B

    return jnp.stack([new_empty, new_A, new_B], axis=-1)


def simulate_mean_field(
    init_state: WorldState,
    cfg: Config,
    T_mf: int = 200,
) -> WorldState:
    """Run T_mf deterministic mean-field steps via lax.scan.

    cfg must be passed as a static argument when jitting:
        jax.jit(simulate_mean_field, static_argnums=(1,))
    Uses cfg.beta and cfg.tau_g directly (Python floats, not differentiated).
    For differentiating w.r.t. tau or beta use simulate_mean_field_with_params.
    """
    def body(state, _):
        new_occ = mean_field_step(
            state.soft_occupancy, state.tolerances, cfg.beta, cfg.tau_g
        )
        return state.replace(soft_occupancy=new_occ, step=state.step + 1), None

    final, _ = jax.lax.scan(body, init_state, xs=None, length=T_mf)
    return final


def simulate_mean_field_with_params(
    init_state: WorldState,
    cfg: Config,
    tau,
    beta,
    T_mf: int = 200,
) -> WorldState:
    """Mean-field simulation with tau and beta as JAX traced values.

    Use this (not simulate_mean_field) when calling jax.jacfwd to differentiate
    w.r.t. tau or beta. Sets tolerances = jnp.full((H,W), tau) at each step so
    the gradient of any loss w.r.t. tau accumulates across all T_mf steps.

    cfg is static (provides H, W, tau_g, T for lax.scan length).
    tau and beta are dynamic — they are traced by jacfwd.

    Jit pattern:
        sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))
        final  = sim_mf(init_state, cfg, tau_jax, beta_jax)
    """
    def body(state, _):
        tol     = jnp.full((cfg.H, cfg.W), tau)
        s       = state.replace(tolerances=tol)
        new_occ = mean_field_step(s.soft_occupancy, s.tolerances, beta, cfg.tau_g)
        return state.replace(
            soft_occupancy=new_occ, tolerances=tol, step=state.step + 1
        ), None

    init   = init_state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final, _ = jax.lax.scan(body, init, xs=None, length=T_mf)
    return final
```

- [ ] **Step 5: Run all theory tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_theory.py -v
```
Expected: 8 tests pass (some may be slow — `test_mean_field_gradient_*` and `test_mean_field_fim_shape` each trigger JAX compilation)

- [ ] **Step 6: Run full suite to catch regressions**

```bash
source .venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1 | tail -10
```
Expected: all existing 54 tests + 8 new = 62 pass

- [ ] **Step 7: Commit**

```bash
git add src/abm_geometry/theory/__init__.py src/abm_geometry/theory/mean_field.py tests/test_theory.py
git commit -m "feat: add theory/mean_field — deterministic Schelling step and FIM pipeline"
```

---

## Task 2: Create `viz/comparison.py`

**Files:**
- Create: `src/abm_geometry/viz/comparison.py`

- [ ] **Step 1: Create `src/abm_geometry/viz/comparison.py`**

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_kappa_comparison(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_sim: np.ndarray,
    kappa_mf: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """Side-by-side log₁₀ κ heatmaps with a shared colour scale.

    p1_grid: 1-D array (N,) — row-axis values (y-axis).
    p2_grid: 1-D array (M,) — column-axis values (x-axis).
    kappa_sim / kappa_mf: shape (N, M).
    """
    log_sim = np.log10(np.clip(kappa_sim, 1.0, None))
    log_mf  = np.log10(np.clip(kappa_mf,  1.0, None))
    vmin = min(log_sim.min(), log_mf.min())
    vmax = max(log_sim.max(), log_mf.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log_k, title in zip(
        axes, [log_sim, log_mf], ["Stochastic sim", "Mean-field"]
    ):
        im = ax.pcolormesh(
            p2_grid, p1_grid, log_k,
            cmap="plasma", shading="auto", vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="log₁₀(κ)")
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_eigenvalue_comparison(
    vals_sim_list: list,
    vals_mf_list: list,
    labels: list[str],
    ax=None,
) -> plt.Axes:
    """Brown/Sethna eigenvalue spectrum: sim (solid) vs mean-field (dashed).

    vals_sim_list / vals_mf_list: lists of 1-D JAX/numpy arrays (one per point).
    Same colour is used for the sim and mf curves at the same parameter point.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    all_vals = [v for vs in [vals_sim_list, vals_mf_list] for v in vs]
    global_max = max(float(np.max(np.abs(np.array(v)))) for v in all_vals)

    for vals_sim, vals_mf, label, color in zip(
        vals_sim_list, vals_mf_list, labels, colors
    ):
        xs = np.arange(1, len(vals_sim) + 1)
        sorted_sim = np.sort(np.abs(np.array(vals_sim)))[::-1]
        sorted_mf  = np.sort(np.abs(np.array(vals_mf)))[::-1]
        ax.semilogy(xs, sorted_sim, "o-",  color=color, lw=2, ms=7, label=f"{label} sim")
        ax.semilogy(xs, sorted_mf,  "o--", color=color, lw=2, ms=7, label=f"{label} mf")

    ax.axhline(0.01 * global_max, color="grey", ls="--", alpha=0.6, label="1% threshold")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks(np.arange(1, max(len(v) for v in vals_sim_list) + 1))
    return ax


def plot_kappa_ratio(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_sim: np.ndarray,
    kappa_mf: np.ndarray,
    xlabel: str,
    ylabel: str,
    ax=None,
) -> plt.Axes:
    """Heatmap of log₁₀(κ_sim / κ_mf).

    Positive (red): stochasticity amplifies sloppiness.
    Negative (blue): stochasticity suppresses sloppiness.
    White contour at ratio=0: mean-field is exact here.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ratio = (
        np.log10(np.clip(kappa_sim, 1.0, None))
        - np.log10(np.clip(kappa_mf, 1.0, None))
    )
    vmax = max(np.abs(ratio).max(), 1e-3)
    im = ax.pcolormesh(
        p2_grid, p1_grid, ratio,
        cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="log₁₀(κ_sim / κ_mf)")
    ax.contour(p2_grid, p1_grid, ratio, levels=[0], colors="white", linewidths=1.5)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    return ax
```

- [ ] **Step 2: Smoke-test imports**

```bash
source .venv/bin/activate && python -c "
from abm_geometry.viz.comparison import plot_kappa_comparison, plot_eigenvalue_comparison, plot_kappa_ratio
import numpy as np, matplotlib; matplotlib.use('Agg')
# quick functional check
import matplotlib.pyplot as plt
fig = plot_kappa_comparison(np.linspace(0,1,5), np.linspace(0,1,5),
    np.ones((5,5))*10, np.ones((5,5))*5, 'x', 'y')
plt.close('all')
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/abm_geometry/viz/comparison.py
git commit -m "feat: add viz/comparison — kappa comparison, ratio, eigenvalue overlay plots"
```

---

## Task 3: Create `exp_004_mean_field_comparison.py`

**Files:**
- Create: `experiments/exp_004_mean_field_comparison.py`

- [ ] **Step 1: Create the experiment script**

```python
"""Phase III: Mean-field vs stochastic FIM comparison over (τ, β) grid.

Produces:
  - kappa_comparison.png: side-by-side stochastic vs mean-field condition number
  - kappa_ratio.png: log₁₀(κ_sim/κ_mf) — where stochasticity matters
  - eigenvalue_comparison.png: spectrum overlay at 2 representative points
Saves plots and arrays to runs/<run_id>/
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
from abm_geometry.geometry.spectrum import condition_number, eigendecomp
from abm_geometry.io.runs import make_run_id, save_results
from abm_geometry.rng import make_key
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.theory.mean_field import simulate_mean_field_with_params
from abm_geometry.viz.comparison import (
    plot_eigenvalue_comparison,
    plot_kappa_comparison,
    plot_kappa_ratio,
)

CFG = Config(H=30, W=30, T=50, seed=42)
_SIM_STOCH = jax.jit(simulate_with_beta, static_argnums=(2,))
_SIM_MF    = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))

TAU_GRID  = np.linspace(0.1, 0.8, 15)
BETA_GRID = np.linspace(2.0, 15.0, 15)

# Two representative points: one low-κ (near τ=0.5, β=5) and one high-κ corner
SPEC_POINTS = {
    "(τ=0.2, β=3)":  (1,  1),
    "(τ=0.6, β=12)": (11, 11),
}


def main():
    master_key = make_key(CFG.seed)
    k_init, k_sim, k_cov = jax.random.split(master_key, 3)

    N, M = len(TAU_GRID), len(BETA_GRID)
    kappa_sim = np.zeros((N, M))
    kappa_mf  = np.zeros((N, M))
    spec_sim, spec_mf = {}, {}

    print(f"Sweeping {N}×{M} = {N*M} grid points…")
    for i, tau in enumerate(TAU_GRID):
        for j, beta in enumerate(BETA_GRID):
            params = jnp.array([tau, beta])

            # ── stochastic FIM ─────────────────────────────────────────────
            def _pipe_sim(p, ki=k_init, ks=k_sim, c=CFG):
                t, b = p[0], p[1]
                s = init_world(ki, c).replace(tolerances=jnp.full((c.H, c.W), t))
                return stats_array_fn(_SIM_STOCH(ks, s, c, b), b)

            def _run_sim(key, tv=tau, bv=beta, c=CFG):
                ki2, ks2 = jax.random.split(key)
                s = init_world(ki2, c).replace(tolerances=jnp.full((c.H, c.W), tv))
                return stats_array_fn(_SIM_STOCH(ks2, s, c, bv), bv)

            J_sim = compute_jacobian(_pipe_sim, params)
            Sig   = estimate_noise_cov(_run_sim, k_cov, K=20)
            F_sim = fisher_information(J_sim, Sig)
            v_sim, _ = eigendecomp(F_sim)
            kappa_sim[i, j] = float(condition_number(v_sim))

            # ── mean-field FIM (Σ = I, no Monte Carlo) ────────────────────
            def _pipe_mf(p, ki=k_init, c=CFG):
                t, b = p[0], p[1]
                s = init_world(ki, c)
                return stats_array_fn(_SIM_MF(s, c, t, b), b)

            J_mf = compute_jacobian(_pipe_mf, params)
            F_mf = J_mf.T @ J_mf
            v_mf, _ = eigendecomp(F_mf)
            kappa_mf[i, j] = float(condition_number(v_mf))

            for label, (ri, rj) in SPEC_POINTS.items():
                if i == ri and j == rj:
                    spec_sim[label] = v_sim
                    spec_mf[label]  = v_mf

        print(f"  τ={tau:.2f} done")

    run_id  = make_run_id()
    run_dir = save_results(run_id, CFG, {"kappa_sim": kappa_sim, "kappa_mf": kappa_mf})

    fig1 = plot_kappa_comparison(
        TAU_GRID, BETA_GRID, kappa_sim, kappa_mf, xlabel="τ", ylabel="β"
    )
    plt.suptitle("Stochastic vs mean-field κ(τ, β)", fontsize=13, y=1.02)
    fig1.savefig(run_dir / "kappa_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_kappa_ratio(
        TAU_GRID, BETA_GRID, kappa_sim, kappa_mf, xlabel="τ", ylabel="β", ax=ax2
    )
    ax2.set_title("log₁₀(κ_sim / κ_mf): where stochasticity matters")
    fig2.savefig(run_dir / "kappa_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    if spec_sim:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        plot_eigenvalue_comparison(
            list(spec_sim.values()),
            list(spec_mf.values()),
            list(spec_sim.keys()),
            ax=ax3,
        )
        ax3.set_title("Eigenvalue spectrum: sim (solid) vs mean-field (dashed)")
        fig3.savefig(run_dir / "eigenvalue_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test a single grid point**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
import jax, jax.numpy as jnp
from abm_geometry.config import Config
from abm_geometry.rng import make_key
from abm_geometry.schelling.state import init_world
from abm_geometry.theory.mean_field import simulate_mean_field_with_params
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number

cfg = Config(H=15, W=15, T=10, seed=0)
sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))
key = make_key(0)
ki, _ = jax.random.split(key)
state = init_world(ki, cfg)

def pipe(p):
    t, b = p[0], p[1]
    s = init_world(ki, cfg)
    return stats_array_fn(sim_mf(s, cfg, t, b), b)

J = compute_jacobian(pipe, jnp.array([0.4, 5.0]))
F_mf = J.T @ J
v, _ = eigendecomp(F_mf)
print('Mean-field kappa =', float(condition_number(v)))
print('PASS')
"
```
Expected: prints a finite kappa and `PASS`

- [ ] **Step 3: Commit**

```bash
git add experiments/exp_004_mean_field_comparison.py
git commit -m "feat: add exp_004 mean-field vs stochastic FIM comparison sweep"
```

---

## Task 4: Add notebook section 13

**Files:**
- Modify: `notebooks/01_schelling_model.ipynb`

Add three cells after the existing section 12 (last cell, id `a9a75dbb` — verify with `python -c "import json; nb=json.load(open('notebooks/01_schelling_model.ipynb')); print(nb['cells'][-1]['id'])"`).

- [ ] **Step 1: Verify last cell id**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
source .venv/bin/activate && python3 -c "
import json
with open('notebooks/01_schelling_model.ipynb') as f:
    nb = json.load(f)
print('Last cell id:', nb['cells'][-1]['id'])
print('Total cells:', len(nb['cells']))
"
```

- [ ] **Step 2: Add all three section-13 cells via Python**

```bash
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
## 13. Mean-Field vs Stochastic FIM

The Gumbel-softmax step introduces stochasticity: at each step, agents sample a random \
stay/move decision. In the **mean-field limit** we replace that sample with its expected \
value — `p_stay = σ(β·(2·sat−1)/τ_g)` — giving a fully deterministic trajectory. \
The resulting mean-field FIM is `F_mf = J_mf^T J_mf` (Σ=I, no Monte Carlo needed).

Comparing `F_mf` and `F_sim` answers the core question: **is sloppiness a property \
of the model equations, or an artefact of stochastic fluctuations?**

- If `κ_mf ≈ κ_sim` everywhere → sloppiness is deterministic; mean-field theory is sufficient.
- If `κ_sim >> κ_mf` → stochasticity amplifies sloppiness; noise matters for identifiability.
- If `κ_sim << κ_mf` → stochasticity reduces sloppiness; noise actually helps identifiability."""),

cell("code", """\
from abm_geometry.theory.mean_field import simulate_mean_field_with_params
from abm_geometry.viz.comparison import (
    plot_kappa_comparison, plot_eigenvalue_comparison, plot_kappa_ratio,
)

_SIM_MF = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))

# Reuse tau_vals, beta_vals, kappa_grid10 from Section 10
kappa_grid_mf = np.zeros((15, 15))

print("Computing mean-field FIM landscape (no noise cov needed)…")
for i, tau_v in enumerate(tau_vals):
    for j, beta_v in enumerate(beta_vals):
        params_ij = jnp.array([tau_v, beta_v])

        def _pipe_mf13(p, ki=k10_init, c=cfg):
            t, b = p[0], p[1]
            s = init_world(ki, c)
            return stats_array_fn(_SIM_MF(s, c, t, b), b)

        J_mf_ij = compute_jacobian(_pipe_mf13, params_ij)
        F_mf_ij = J_mf_ij.T @ J_mf_ij        # Σ = I
        v_mf_ij, _ = eigendecomp(F_mf_ij)
        kappa_grid_mf[i, j] = float(condition_number(v_mf_ij))

    print(f"τ={tau_v:.2f} ✓", end="  ", flush=True)

print(f"\\nMean-field range: κ ∈ [{kappa_grid_mf.min():.1f}, {kappa_grid_mf.max():.1f}]")
print(f"Stochastic range: κ ∈ [{kappa_grid10.min():.1f}, {kappa_grid10.max():.1f}]")"""),

cell("code", """\
# ── Comparison heatmaps ────────────────────────────────────────────────────────
fig13a = plot_kappa_comparison(
    tau_vals, beta_vals, kappa_grid10, kappa_grid_mf, xlabel="τ", ylabel="β"
)
plt.suptitle("Stochastic vs mean-field condition number κ(τ, β)", fontsize=13, y=1.02)
plt.show()

# ── Ratio heatmap ──────────────────────────────────────────────────────────────
fig13b, ax13b = plt.subplots(figsize=(7, 4))
plot_kappa_ratio(
    tau_vals, beta_vals, kappa_grid10, kappa_grid_mf,
    xlabel="τ", ylabel="β", ax=ax13b,
)
ax13b.set_title("log₁₀(κ_sim / κ_mf) — positive = stochasticity amplifies sloppiness")
plt.tight_layout()
plt.show()

print("\\n=== Scientific conclusion ===")
ratio_flat = np.log10(np.clip(kappa_grid10, 1, None)) - np.log10(np.clip(kappa_grid_mf, 1, None))
print(f"Mean log₁₀(κ_sim/κ_mf) = {ratio_flat.mean():+.2f}  "
      f"(+: stochasticity amplifies sloppiness, -: suppresses)")
print(f"Max deviation: {ratio_flat.max():+.2f} at τ={tau_vals[np.unravel_index(ratio_flat.argmax(), ratio_flat.shape)[0]]:.2f}, "
      f"β={beta_vals[np.unravel_index(ratio_flat.argmax(), ratio_flat.shape)[1]]:.1f}")"""),

]

with open(NB) as f:
    nb = json.load(f)

nb["cells"].extend(new_cells)

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(new_cells)} cells. Total: {len(nb['cells'])}")
EOF
```
Expected: `Added 3 cells. Total: 27`

- [ ] **Step 3: Verify notebook is valid JSON and cells are present**

```bash
source .venv/bin/activate && python3 -c "
import json
with open('notebooks/01_schelling_model.ipynb') as f:
    nb = json.load(f)
print('Total cells:', len(nb['cells']))
for c in nb['cells'][-3:]:
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'  [{c[\"cell_type\"][:4]}] {src[:60].replace(chr(10),\" \")}')
"
```
Expected: shows 3 new cells starting with section 13 content

- [ ] **Step 4: Run full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1 | tail -8
```
Expected: 62 tests pass

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_schelling_model.ipynb
git commit -m "feat: add notebook section 13 — mean-field vs stochastic FIM comparison"
```

---

## Final verification

- [ ] **End-to-end smoke test**

```bash
source .venv/bin/activate && python3 - <<'EOF'
import sys; sys.path.insert(0, "src")
import jax, jax.numpy as jnp, numpy as np
from abm_geometry.config import Config
from abm_geometry.rng import make_key
from abm_geometry.schelling.state import init_world
from abm_geometry.theory.mean_field import simulate_mean_field_with_params
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number

cfg = Config(H=15, W=15, T=10, seed=0)
sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))
key = make_key(0)
ki, ks, kc = jax.random.split(key, 3)
state = init_world(ki, cfg)

# Mean-field FIM
def pipe_mf(p):
    s = init_world(ki, cfg)
    return stats_array_fn(sim_mf(s, cfg, p[0], p[1]), p[1])

J_mf = compute_jacobian(pipe_mf, jnp.array([0.4, 5.0]))
F_mf = J_mf.T @ J_mf
v_mf, _ = eigendecomp(F_mf)
print(f"Mean-field κ = {float(condition_number(v_mf)):.1f}")

# Stochastic FIM for comparison
from abm_geometry.schelling.simulate import simulate_with_beta
sim_st = jax.jit(simulate_with_beta, static_argnums=(2,))

def pipe_st(p):
    s = init_world(ki, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), p[0]))
    return stats_array_fn(sim_st(ks, s, cfg, p[1]), p[1])

def run_st(k):
    ki2, ks2 = jax.random.split(k)
    s = init_world(ki2, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), 0.4))
    return stats_array_fn(sim_st(ks2, s, cfg, 5.0), 5.0)

J_st = compute_jacobian(pipe_st, jnp.array([0.4, 5.0]))
Sig  = estimate_noise_cov(run_st, kc, K=5)
F_st = fisher_information(J_st, Sig)
v_st, _ = eigendecomp(F_st)
print(f"Stochastic κ = {float(condition_number(v_st)):.1f}")
print(f"log10(κ_sim/κ_mf) = {float(jnp.log10(condition_number(v_st)/condition_number(v_mf))):.2f}")
print("ALL CHECKS PASSED")
EOF
```
Expected: both kappas finite, `ALL CHECKS PASSED`

- [ ] **Tag the release**

```bash
git tag -a v0.3-mean-field -m "Phase III: mean-field analytical FIM complete"
```
