# FIM Landscape & Heterogeneous Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the full FIM machinery (4-stat summary vector, noise covariance estimation, eigenspectrum), extend to heterogeneous tolerance (Phase II), and add notebook sections 8–12 with publication-quality sloppy-models plots.

**Architecture:** Phase I uses θ=(τ,β) → 2×2 FIM; Phase II uses θ=(μ,σ,β) → 3×3 FIM. A new `simulate_with_beta` function takes β as a traced JAX array (bypassing `cfg.beta`) so `jacfwd` can differentiate through it. Summary stats are returned as `Float[Array,"4"]` from `stats_array_fn(state, beta)`. All sweep loops in the notebook are Python loops over a precompiled jitted simulation.

**Tech Stack:** JAX (jacfwd, vmap, lax.scan), jaxtyping, chex, matplotlib, numpy

---

## File Map

| File | Action |
|------|--------|
| `src/abm_geometry/config.py` | add `sigma_tau: float = 0.0` |
| `src/abm_geometry/statistics/segregation.py` | add `moran_i()` |
| `src/abm_geometry/statistics/clusters.py` | create — `soft_mean_neighbourhood_homogeneity()` |
| `src/abm_geometry/statistics/summary.py` | add `stats_array_fn(state, beta)` |
| `src/abm_geometry/geometry/fim.py` | create — `estimate_noise_cov`, `fisher_information` |
| `src/abm_geometry/geometry/spectrum.py` | create — `eigendecomp`, `condition_number`, `sloppy_modes` |
| `src/abm_geometry/schelling/simulate.py` | add `simulate_with_beta` |
| `src/abm_geometry/schelling/state.py` | add `init_world_heterogeneous` |
| `src/abm_geometry/viz/spectra.py` | create — `plot_eigenvalue_spectrum` |
| `src/abm_geometry/viz/landscape.py` | create — 3 plot functions |
| `src/abm_geometry/viz/grids.py` | add `plot_grid_comparison` |
| `experiments/exp_002_fim_landscape.py` | create — Phase I sweep |
| `experiments/exp_003_heterogeneous.py` | create — Phase II sweep |
| `notebooks/01_schelling_model.ipynb` | add sections 8–12 |
| `tests/test_statistics.py` | add moran_i, homogeneity, stats_array_fn tests |
| `tests/test_geometry.py` | create — FIM and spectrum tests |
| `tests/test_gradients.py` | add simulate_with_beta gradient test |

---

## Task 1: Add sigma_tau to Config

**Files:**
- Modify: `src/abm_geometry/config.py`

- [ ] **Add field after `tau`:**

```python
@dataclass(frozen=True)
class Config:
    H: int = 30
    W: int = 30
    T: int = 50
    density: float = 0.8
    group_ratio: float = 0.5
    beta: float = 5.0
    tau_g: float = 0.5
    tau: float = 0.4
    sigma_tau: float = 0.0   # std of tolerance distribution; 0 = homogeneous
    seed: int = 42
```

- [ ] **Verify existing tests still pass:**
```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
source .venv/bin/activate && python -m pytest tests/test_config.py -v
```
Expected: 2 passed

- [ ] **Commit:**
```bash
git add src/abm_geometry/config.py
git commit -m "feat: add sigma_tau field to Config for heterogeneous tolerance"
```

---

## Task 2: Add moran_i to segregation.py

**Files:**
- Modify: `src/abm_geometry/statistics/segregation.py`
- Test: `tests/test_statistics.py`

- [ ] **Write failing tests first:**

Add to `tests/test_statistics.py`:
```python
from abm_geometry.statistics.segregation import moran_i

def test_moran_i_uniform():
    """All cells equal A fraction → z=0 everywhere → I=0."""
    occ = jnp.zeros((4, 4, 3)).at[..., 1].set(0.5).at[..., 2].set(0.5)
    assert jnp.allclose(moran_i(occ), jnp.array(0.0), atol=1e-5)

def test_moran_i_range():
    """Moran's I should be in [-1, 1] for any valid soft_occ."""
    import jax
    key = jax.random.PRNGKey(7)
    occ = jax.random.dirichlet(key, jnp.ones(3), shape=(8, 8))
    i = float(moran_i(occ))
    assert -1.0 - 1e-4 <= i <= 1.0 + 1e-4

def test_moran_i_segregated():
    """Left half all A, right half all B → strong positive autocorrelation."""
    occ = jnp.zeros((4, 4, 3))
    occ = occ.at[:, :2, 1].set(1.0)   # left: type A
    occ = occ.at[:, 2:, 2].set(1.0)   # right: type B
    assert float(moran_i(occ)) > 0.3
```

- [ ] **Run to confirm they fail:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py::test_moran_i_uniform -v
```
Expected: ImportError or AttributeError (function not yet defined)

- [ ] **Implement `moran_i` in `statistics/segregation.py`:**

```python
from abm_geometry.schelling.neighbours import _sum_moore_neighbours

def moran_i(soft_occ: Float[Array, "H W 3"]) -> Float[Array, ""]:
    """Moran's I spatial autocorrelation of type-A fraction.

    I = (z · Wz) / (z · z), where z is the centred type-A fraction per cell
    and W is the row-normalised Moore adjacency (sum of 8 neighbours / 8).
    Returns values in [-1, 1]: positive = clustered, negative = dispersed.
    """
    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    frac_A = occ_A / (occ_A + occ_B + 1e-8)     # Float[H, W]
    mean_frac_A = jnp.mean(frac_A)
    z = frac_A - mean_frac_A                      # centred

    Wz = _sum_moore_neighbours(z) / 8.0           # row-normalised Moore average
    numerator = jnp.sum(z * Wz)
    denominator = jnp.sum(z * z) + 1e-8
    return numerator / denominator
```

Note: `_sum_moore_neighbours` is already defined in `neighbours.py` — import it directly.

- [ ] **Run tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py -v
```
Expected: all statistics tests pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/statistics/segregation.py tests/test_statistics.py
git commit -m "feat: add moran_i spatial autocorrelation statistic"
```

---

## Task 3: Create clusters.py

**Files:**
- Create: `src/abm_geometry/statistics/clusters.py`
- Test: `tests/test_statistics.py`

- [ ] **Write failing test:**

Add to `tests/test_statistics.py`:
```python
from abm_geometry.statistics.clusters import soft_mean_neighbourhood_homogeneity

def test_homogeneity_all_same_type():
    """Grid all type A → all neighbours same type → homogeneity = 1.0."""
    occ = jnp.zeros((4, 4, 3)).at[..., 1].set(1.0)
    h = soft_mean_neighbourhood_homogeneity(occ)
    assert jnp.allclose(h, jnp.array(1.0), atol=1e-4)

def test_homogeneity_range():
    """Homogeneity should be in [0, 1]."""
    import jax
    occ = jax.random.dirichlet(jax.random.PRNGKey(3), jnp.ones(3), shape=(6, 6))
    h = float(soft_mean_neighbourhood_homogeneity(occ))
    assert 0.0 - 1e-4 <= h <= 1.0 + 1e-4
```

- [ ] **Run to confirm failure:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py::test_homogeneity_all_same_type -v
```

- [ ] **Create `src/abm_geometry/statistics/clusters.py`:**

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.schelling.neighbours import neighbour_type_fraction


def soft_mean_neighbourhood_homogeneity(
    soft_occ: Float[Array, "H W 3"],
) -> Float[Array, ""]:
    """Mean neighbourhood homogeneity — soft cluster-size proxy.

    For each cell, computes the fraction of occupied neighbours that share
    the cell's dominant type (weighted by occupancy probability).
    Returns a scalar in [0, 1]: 1 = perfectly clustered, 0.5 = random.
    Differentiable via the underlying convolution in neighbour_type_fraction.
    """
    neigh_frac = neighbour_type_fraction(soft_occ)  # Float[H, W, 2]
    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    # frac_same: if A-type, how many A-neighbours; if B-type, how many B-neighbours
    frac_same = occ_A * neigh_frac[..., 0] + occ_B * neigh_frac[..., 1]
    return jnp.mean(frac_same)
```

- [ ] **Run tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py -v
```
Expected: all pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/statistics/clusters.py tests/test_statistics.py
git commit -m "feat: add soft_mean_neighbourhood_homogeneity cluster proxy stat"
```

---

## Task 4: Add stats_array_fn to summary.py

**Files:**
- Modify: `src/abm_geometry/statistics/summary.py`
- Test: `tests/test_statistics.py`

- [ ] **Write failing test:**

Add to `tests/test_statistics.py`:
```python
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.config import Config
from abm_geometry.schelling.state import init_world

def test_stats_array_fn_shape():
    import jax
    cfg = Config(H=6, W=6, T=3)
    state = init_world(jax.random.PRNGKey(0), cfg)
    arr = stats_array_fn(state, cfg.beta)
    assert arr.shape == (4,)

def test_stats_array_fn_range():
    import jax
    cfg = Config(H=6, W=6, T=3)
    state = init_world(jax.random.PRNGKey(0), cfg)
    arr = stats_array_fn(state, cfg.beta)
    assert jnp.all(jnp.isfinite(arr))
```

- [ ] **Run to confirm failure:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py::test_stats_array_fn_shape -v
```

- [ ] **Implement in `statistics/summary.py`:**

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.statistics.clusters import soft_mean_neighbourhood_homogeneity
from abm_geometry.statistics.segregation import dissimilarity_index, moran_i
from abm_geometry.schelling.satisfaction import soft_satisfaction
from abm_geometry.types import SummaryStats, WorldState


def stats_fn(state: WorldState) -> SummaryStats:
    """Compute all summary statistics from a simulation state."""
    return SummaryStats(
        dissimilarity=dissimilarity_index(state.soft_occupancy),
    )


def stats_array_fn(state: WorldState, beta) -> Float[Array, "4"]:
    """Differentiable 4-stat summary vector: [D, mean_sat, Moran_I, homogeneity].

    beta can be a Python float or a JAX traced array (for jacfwd through beta).
    """
    occ = state.soft_occupancy
    occupied = 1.0 - occ[..., 0]                           # Float[H, W]

    d = dissimilarity_index(occ)

    sat = soft_satisfaction(occ, state.tolerances, beta)    # Float[H, W]
    mean_sat = jnp.sum(sat * occupied) / (jnp.sum(occupied) + 1e-8)

    i = moran_i(occ)

    h = soft_mean_neighbourhood_homogeneity(occ)

    return jnp.array([d, mean_sat, i, h])
```

- [ ] **Run all statistics tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_statistics.py -v
```
Expected: all pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/statistics/summary.py tests/test_statistics.py
git commit -m "feat: add stats_array_fn returning Float[4] for FIM computation"
```

---

## Task 5: Add simulate_with_beta to simulate.py

**Files:**
- Modify: `src/abm_geometry/schelling/simulate.py`
- Test: `tests/test_gradients.py`

- [ ] **Write failing test first:**

Add to `tests/test_gradients.py`:
```python
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.statistics.summary import stats_array_fn

def test_gradient_flows_through_beta():
    """jacfwd should produce a finite gradient w.r.t. beta."""
    cfg_b = Config(H=8, W=8, T=5, seed=3)
    key_b = jax.random.PRNGKey(cfg_b.seed)
    k_init_b, k_sim_b = jax.random.split(key_b)
    state_b = init_world(k_init_b, cfg_b)
    sim_beta = jax.jit(simulate_with_beta, static_argnums=(2,))

    def d_from_beta(beta):
        final = sim_beta(k_sim_b, state_b, cfg_b, beta)
        return stats_array_fn(final, beta)[0]   # dissimilarity

    grad = jax.grad(d_from_beta)(jnp.array(5.0))
    assert jnp.isfinite(grad), f"gradient w.r.t. beta not finite: {grad}"
```

- [ ] **Run to confirm failure:**
```bash
source .venv/bin/activate && python -m pytest tests/test_gradients.py::test_gradient_flows_through_beta -v
```

- [ ] **Add `simulate_with_beta` to `simulate.py`:**

```python
def simulate_with_beta(
    key: PRNGKeyArray,
    init_state: WorldState,
    cfg: Config,
    beta_jax,
) -> WorldState:
    """Like simulate, but accepts beta as a JAX traced value.

    Use this instead of simulate when differentiating w.r.t. beta.
    cfg is still static (for lax.scan length); beta_jax is dynamic.
    """
    from abm_geometry.schelling.move_rules import gumbel_softmax_step

    def body(carry, _):
        state, k = carry
        k, subkey = jax.random.split(k)
        new_occ = gumbel_softmax_step(
            state.soft_occupancy, state.tolerances, subkey, beta_jax, cfg.tau_g
        )
        return (state.replace(soft_occupancy=new_occ, step=state.step + 1), k), None

    (final_state, _), _ = jax.lax.scan(
        body, (init_state, key), xs=None, length=cfg.T
    )
    return final_state
```

- [ ] **Run all gradient tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_gradients.py -v
```
Expected: all 3 pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/schelling/simulate.py tests/test_gradients.py
git commit -m "feat: add simulate_with_beta for differentiating through beta parameter"
```

---

## Task 6: Create geometry/fim.py

**Files:**
- Create: `src/abm_geometry/geometry/fim.py`
- Create: `tests/test_geometry.py`

- [ ] **Write failing tests:**

Create `tests/test_geometry.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np

from abm_geometry.config import Config
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.summary import stats_array_fn

_CFG = Config(H=8, W=8, T=5, seed=11)
_KEY = jax.random.PRNGKey(0)


def _make_run_fn(cfg, tau_val, beta_val):
    """Returns a function: PRNGKey → Float[4] for noise cov estimation."""
    def run_fn(key):
        k_init, k_sim = jax.random.split(key)
        state = init_world(k_init, cfg).replace(
            tolerances=jnp.full((cfg.H, cfg.W), tau_val)
        )
        final = simulate_with_beta(k_sim, state, cfg, beta_val)
        return stats_array_fn(final, beta_val)
    return run_fn


def test_estimate_noise_cov_shape():
    run_fn = _make_run_fn(_CFG, 0.4, 5.0)
    Sigma = estimate_noise_cov(run_fn, _KEY, K=10)
    assert Sigma.shape == (4, 4)


def test_estimate_noise_cov_psd():
    """Noise covariance must be positive semi-definite."""
    run_fn = _make_run_fn(_CFG, 0.4, 5.0)
    Sigma = estimate_noise_cov(run_fn, _KEY, K=15)
    vals = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(vals >= -1e-6), f"Sigma not PSD: min eigenvalue={float(vals.min())}"


def test_fisher_information_shape():
    J = jnp.ones((4, 2))   # 4 stats, 2 params
    Sigma = jnp.eye(4)
    F = fisher_information(J, Sigma)
    assert F.shape == (2, 2)


def test_fisher_information_identity_sigma():
    """With Sigma=I, F = J^T J."""
    J = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    F = fisher_information(J, jnp.eye(4))
    expected = J.T @ J
    assert jnp.allclose(F, expected, atol=1e-5)
```

- [ ] **Run to confirm failures:**
```bash
source .venv/bin/activate && python -m pytest tests/test_geometry.py -v
```

- [ ] **Create `src/abm_geometry/geometry/fim.py`:**

```python
import warnings
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray


def estimate_noise_cov(
    run_fn: Callable,
    key: PRNGKeyArray,
    K: int = 20,
) -> Float[Array, "4 4"]:
    """Estimate noise covariance Σ of summary statistics via Monte Carlo.

    run_fn: PRNGKey → Float[4]  — one simulation run returning 4 stats.
    Draws K independent runs, returns the sample covariance matrix.
    Falls back to identity if the result is ill-conditioned (κ > 1e6).
    """
    keys = jax.random.split(key, K)
    samples = jax.jit(jax.vmap(run_fn))(keys)   # Float[K, 4]
    Sigma = jnp.cov(samples.T)                   # Float[4, 4]
    cond = float(jnp.linalg.cond(Sigma))
    if cond > 1e6:
        warnings.warn(
            f"Noise covariance ill-conditioned (κ={cond:.2e}); using identity Σ=I.",
            stacklevel=2,
        )
        return jnp.eye(4)
    return Sigma


def fisher_information(
    J: Float[Array, "4 p"],
    noise_cov: Float[Array, "4 4"],
) -> Float[Array, "p p"]:
    """Fisher Information Matrix F = J^T Σ^{-1} J.

    J: Jacobian of summary stats w.r.t. parameters, shape (4, p).
    noise_cov: 4×4 noise covariance Σ.
    Returns F of shape (p, p).
    """
    Sigma_inv = jnp.linalg.inv(noise_cov)
    return J.T @ Sigma_inv @ J
```

- [ ] **Run geometry tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_geometry.py -v
```
Expected: all 4 pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/geometry/fim.py tests/test_geometry.py
git commit -m "feat: add geometry/fim.py — estimate_noise_cov and fisher_information"
```

---

## Task 7: Create geometry/spectrum.py

**Files:**
- Create: `src/abm_geometry/geometry/spectrum.py`
- Test: `tests/test_geometry.py`

- [ ] **Add failing tests to `tests/test_geometry.py`:**

```python
from abm_geometry.geometry.spectrum import condition_number, eigendecomp, sloppy_modes


def test_eigendecomp_shape():
    F = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    vals, vecs = eigendecomp(F)
    assert vals.shape == (2,)
    assert vecs.shape == (2, 2)


def test_eigendecomp_descending():
    F = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    vals, _ = eigendecomp(F)
    assert float(vals[0]) >= float(vals[1])


def test_condition_number_diagonal():
    """Diagonal FIM with known ratio."""
    F = jnp.diag(jnp.array([100.0, 1.0]))
    vals, _ = eigendecomp(F)
    kappa = condition_number(vals)
    assert jnp.allclose(kappa, jnp.array(100.0), atol=1e-3)


def test_sloppy_modes_threshold():
    """With threshold=0.1, the small eigenvalue should be identified as sloppy."""
    F = jnp.diag(jnp.array([100.0, 5.0]))
    vals, vecs = eigendecomp(F)
    modes = sloppy_modes(vals, vecs, threshold=0.1)
    assert len(modes) == 1
    assert jnp.allclose(modes[0][0], jnp.array(5.0), atol=1e-3)
```

- [ ] **Create `src/abm_geometry/geometry/spectrum.py`:**

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def eigendecomp(
    F: Float[Array, "p p"],
) -> tuple[Float[Array, "p"], Float[Array, "p p"]]:
    """Eigendecomposition of symmetric FIM, sorted descending.

    Returns (eigenvalues, eigenvectors) where eigenvectors[:, i] is the
    i-th eigenvector corresponding to eigenvalues[i].
    Uses jnp.linalg.eigh (stable for symmetric matrices, real eigenvalues).
    """
    vals, vecs = jnp.linalg.eigh(F)          # ascending by default
    idx = jnp.argsort(vals)[::-1]            # descending
    return vals[idx], vecs[:, idx]


def condition_number(eigenvalues: Float[Array, "p"]) -> Float[Array, ""]:
    """κ = λ_max / λ_min.  Large κ means sloppy (poorly identified) model."""
    return eigenvalues[0] / (eigenvalues[-1] + 1e-12)


def sloppy_modes(
    eigenvalues: Float[Array, "p"],
    eigenvectors: Float[Array, "p p"],
    threshold: float = 0.01,
) -> list[tuple]:
    """Return (value, vector) pairs where λ < threshold * λ_max."""
    lam_max = float(eigenvalues[0])
    return [
        (eigenvalues[i], eigenvectors[:, i])
        for i in range(len(eigenvalues))
        if float(eigenvalues[i]) < threshold * lam_max
    ]
```

- [ ] **Run all geometry tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_geometry.py -v
```
Expected: all 8 pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/geometry/spectrum.py tests/test_geometry.py
git commit -m "feat: add geometry/spectrum.py — eigendecomp, condition_number, sloppy_modes"
```

---

## Task 8: Add init_world_heterogeneous to state.py

**Files:**
- Modify: `src/abm_geometry/schelling/state.py`
- Test: `tests/test_simulator.py`

- [ ] **Add failing tests to `tests/test_simulator.py`:**

```python
from abm_geometry.schelling.state import init_world_heterogeneous

CFG_HET = Config(H=10, W=10, T=5, density=0.8, tau=0.4, sigma_tau=0.15)


def test_heterogeneous_tolerances_vary():
    """With sigma_tau > 0, tolerances should not all be equal."""
    key = jax.random.PRNGKey(0)
    state = init_world_heterogeneous(key, CFG_HET)
    tol = state.tolerances
    assert float(jnp.std(tol)) > 0.01


def test_heterogeneous_tolerances_clipped():
    """Tolerances must stay in [0.001, 0.999]."""
    key = jax.random.PRNGKey(0)
    state = init_world_heterogeneous(key, CFG_HET)
    assert jnp.all(state.tolerances >= 0.001)
    assert jnp.all(state.tolerances <= 0.999)


def test_heterogeneous_gradient_flows_sigma():
    """Gradient of dissimilarity w.r.t. sigma_tau must be finite."""
    from abm_geometry.schelling.simulate import simulate_with_beta
    from abm_geometry.statistics.segregation import dissimilarity_index

    cfg_g = Config(H=8, W=8, T=5, seed=5, tau=0.4)
    key_g = jax.random.PRNGKey(0)
    k_init_g, k_sim_g = jax.random.split(key_g)

    def d_from_sigma(sigma):
        eps = jax.random.normal(k_init_g, (cfg_g.H, cfg_g.W))
        tol = jnp.clip(cfg_g.tau + sigma * eps, 0.001, 0.999)
        base = init_world(k_init_g, cfg_g).replace(tolerances=tol)
        final = simulate_with_beta(k_sim_g, base, cfg_g, jnp.array(cfg_g.beta))
        return dissimilarity_index(final.soft_occupancy)

    grad = jax.grad(d_from_sigma)(jnp.array(0.1))
    assert jnp.isfinite(grad)
```

- [ ] **Run to confirm failures:**
```bash
source .venv/bin/activate && python -m pytest tests/test_simulator.py::test_heterogeneous_tolerances_vary -v
```

- [ ] **Add `init_world_heterogeneous` to `state.py`:**

```python
def init_world_heterogeneous(key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Place agents on the grid with tolerance sampled from N(cfg.tau, cfg.sigma_tau²).

    Uses the reparametrisation trick: tolerances = clip(cfg.tau + cfg.sigma_tau * ε)
    so gradients w.r.t. cfg.tau (μ) and cfg.sigma_tau (σ) flow correctly when
    those values are passed as JAX traced arrays via state.replace().
    Tolerances are clipped to [0.001, 0.999] to keep sigmoid inputs bounded.
    """
    key_cells, key_groups, key_tol = jax.random.split(key, 3)

    is_occupied = jax.random.bernoulli(key_cells, cfg.density, shape=(cfg.H, cfg.W))
    is_A = jax.random.bernoulli(key_groups, cfg.group_ratio, shape=(cfg.H, cfg.W))
    is_A = is_A & is_occupied
    is_B = is_occupied & ~is_A

    soft_occ = jnp.stack(
        [
            (~is_occupied).astype(jnp.float32),
            is_A.astype(jnp.float32),
            is_B.astype(jnp.float32),
        ],
        axis=-1,
    )

    eps = jax.random.normal(key_tol, (cfg.H, cfg.W))
    tolerances = jnp.clip(cfg.tau + cfg.sigma_tau * eps, 0.001, 0.999)

    return WorldState(
        soft_occupancy=soft_occ,
        tolerances=tolerances,
        step=jnp.int32(0),
    )
```

- [ ] **Run all simulator tests:**
```bash
source .venv/bin/activate && python -m pytest tests/test_simulator.py -v
```
Expected: all pass (original 20 + 3 new = 23)

- [ ] **Commit:**
```bash
git add src/abm_geometry/schelling/state.py tests/test_simulator.py
git commit -m "feat: add init_world_heterogeneous with reparametrisation trick for sigma gradient"
```

---

## Task 9: Create viz/spectra.py

**Files:**
- Create: `src/abm_geometry/viz/spectra.py`

- [ ] **Create the file:**

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalue_spectrum(
    eigenvalues_list: list,
    labels: list[str],
    ax=None,
    title: str = "FIM eigenvalue spectrum",
) -> plt.Axes:
    """Brown/Sethna-style eigenvalue spectrum plot.

    eigenvalues_list: list of 1-D arrays, one per parameter-space point.
    labels: one label per array, shown in the legend.
    Horizontal grey dashed line marks the 1% sloppy threshold relative to
    the maximum eigenvalue across all points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eigenvalues_list)))
    global_max = max(float(np.max(np.abs(v))) for v in eigenvalues_list)

    for vals, label, color in zip(eigenvalues_list, labels, colors):
        vals_np = np.array(vals)
        idx = np.argsort(vals_np)[::-1]
        sorted_vals = vals_np[idx]
        xs = np.arange(1, len(sorted_vals) + 1)
        ax.semilogy(xs, sorted_vals, "o-", color=color, label=label, lw=2, ms=7)

    ax.axhline(
        0.01 * global_max,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="1% sloppy threshold",
    )
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xticks(np.arange(1, max(len(v) for v in eigenvalues_list) + 1))
    return ax
```

- [ ] **Smoke-test import:**
```bash
source .venv/bin/activate && python -c "from abm_geometry.viz.spectra import plot_eigenvalue_spectrum; print('OK')"
```
Expected: `OK`

- [ ] **Commit:**
```bash
git add src/abm_geometry/viz/spectra.py
git commit -m "feat: add viz/spectra.py — Brown/Sethna eigenvalue spectrum plot"
```

---

## Task 10: Create viz/landscape.py

**Files:**
- Create: `src/abm_geometry/viz/landscape.py`

- [ ] **Create the file:**

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_condition_heatmap(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    ax=None,
    title: str = "",
) -> plt.Axes:
    """Log10 condition-number heatmap over a 2D parameter grid.

    p1_grid: 1-D array of shape (N,) — x-axis values (rows of kappa_grid).
    p2_grid: 1-D array of shape (M,) — y-axis values (columns of kappa_grid).
    kappa_grid: shape (N, M) — condition numbers κ = λ_max/λ_min.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    log_kappa = np.log10(np.clip(kappa_grid, 1.0, None))
    im = ax.pcolormesh(
        p2_grid, p1_grid, log_kappa,
        cmap="plasma", shading="auto",
    )
    plt.colorbar(im, ax=ax, label="log₁₀(κ)")

    # Contour lines at κ = 10, 100, 1000
    ax.contour(
        p2_grid, p1_grid, log_kappa,
        levels=[1, 2, 3],
        colors="white",
        linewidths=0.8,
        alpha=0.7,
    )
    ax.set_xlabel(ylabel)   # p2 on x-axis
    ax.set_ylabel(xlabel)   # p1 on y-axis
    if title:
        ax.set_title(title)
    return ax


def plot_eigenvector_quivers(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    stiff_vecs: np.ndarray,
    sloppy_vecs: np.ndarray,
    ax=None,
    stride: int = 3,
) -> plt.Axes:
    """Quiver plot of stiff (blue) and sloppy (red) eigenvectors.

    stiff_vecs / sloppy_vecs: shape (N, M, 2) — eigenvectors at each grid point.
    stride: plot every `stride`-th point to avoid clutter.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    P1, P2 = np.meshgrid(p1_grid, p2_grid, indexing="ij")

    def _quiver(vecs, color, label):
        s = stride
        ax.quiver(
            P2[::s, ::s], P1[::s, ::s],
            vecs[::s, ::s, 1], vecs[::s, ::s, 0],
            color=color, alpha=0.8, scale=8, width=0.005, label=label,
        )

    _quiver(stiff_vecs, "#2563eb", "Stiff (λ_max)")
    _quiver(sloppy_vecs, "#dc2626", "Sloppy (λ_min)")
    ax.legend(fontsize=9, loc="upper right")
    return ax


def plot_jacobian_components(
    param_values: np.ndarray,
    J_values: np.ndarray,
    param_name: str,
    stat_names: list[str],
    ax=None,
) -> plt.Axes:
    """Plot ∂s_i/∂θ vs a swept scalar parameter.

    param_values: shape (N,)
    J_values: shape (N, 4) — one row per parameter value, one col per stat.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = ["#2b6cb0", "#c05621", "#276749", "#6b46c1"]
    for i, (name, color) in enumerate(zip(stat_names, colors)):
        ax.plot(param_values, J_values[:, i], label=name, color=color, lw=2)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(param_name)
    ax.set_ylabel(f"∂s/∂{param_name}")
    ax.legend(fontsize=9)
    ax.set_title(f"Jacobian components vs {param_name}")
    return ax
```

- [ ] **Smoke-test:**
```bash
source .venv/bin/activate && python -c "from abm_geometry.viz.landscape import plot_condition_heatmap, plot_eigenvector_quivers, plot_jacobian_components; print('OK')"
```

- [ ] **Commit:**
```bash
git add src/abm_geometry/viz/landscape.py
git commit -m "feat: add viz/landscape.py — heatmap, quiver, Jacobian component plots"
```

---

## Task 11: Add plot_grid_comparison to viz/grids.py

**Files:**
- Modify: `src/abm_geometry/viz/grids.py`

- [ ] **Append to `grids.py`:**

```python
def plot_grid_comparison(
    states: list,
    titles: list[str],
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Side-by-side grid panels with a shared colour legend."""
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, len(states), figsize=figsize)
    if len(states) == 1:
        axes = [axes]
    for state, title, ax in zip(states, titles, axes):
        plot_grid(state.soft_occupancy, title=title, ax=ax)

    legend = [
        mpatches.Patch(color="#CC3333", label="Group A"),
        mpatches.Patch(color="#3333CC", label="Group B"),
        mpatches.Patch(facecolor="#F0F0F0", edgecolor="lightgrey", label="Empty"),
    ]
    axes[-1].legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    return fig
```

- [ ] **Smoke-test:**
```bash
source .venv/bin/activate && python -c "from abm_geometry.viz.grids import plot_grid_comparison; print('OK')"
```

- [ ] **Run full test suite to catch regressions:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```
Expected: all pass

- [ ] **Commit:**
```bash
git add src/abm_geometry/viz/grids.py
git commit -m "feat: add plot_grid_comparison to viz/grids"
```

---

## Task 12: Create exp_002_fim_landscape.py (Phase I sweep)

**Files:**
- Create: `experiments/exp_002_fim_landscape.py`

- [ ] **Create the experiment:**

```python
"""Phase I FIM landscape: sweep (τ, β) on a 15×15 grid.

Produces:
  - condition_number heatmap over (τ, β)
  - eigenvector quivers
  - Jacobian component curves vs τ (at β=5)
Saves plots to runs/<run_id>/
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
from abm_geometry.viz.landscape import (
    plot_condition_heatmap,
    plot_eigenvector_quivers,
    plot_jacobian_components,
)

CFG = Config(H=30, W=30, T=50, seed=42)
_SIM = jax.jit(simulate_with_beta, static_argnums=(2,))

TAU_GRID = np.linspace(0.1, 0.8, 15)
BETA_GRID = np.linspace(2.0, 15.0, 15)
STAT_NAMES = ["Dissimilarity D", "Mean satisfaction", "Moran's I", "Homogeneity"]


def make_pipeline(key_init, key_sim, tau_val, beta_val):
    def stats(p):
        t, b = p[0], p[1]
        state = init_world(key_init, CFG).replace(
            tolerances=jnp.full((CFG.H, CFG.W), t)
        )
        return stats_array_fn(_SIM(key_sim, state, CFG, b), b)
    return stats

def make_run_fn(tau_val, beta_val):
    def run(key):
        ki, ks = jax.random.split(key)
        state = init_world(ki, CFG).replace(
            tolerances=jnp.full((CFG.H, CFG.W), tau_val)
        )
        return stats_array_fn(_SIM(ks, state, CFG, beta_val), beta_val)
    return run


def main():
    master_key = make_key(CFG.seed)
    k_init, k_sim, k_cov = jax.random.split(master_key, 3)

    N = len(TAU_GRID)
    M = len(BETA_GRID)
    kappa_grid = np.zeros((N, M))
    stiff_vecs = np.zeros((N, M, 2))
    sloppy_vecs = np.zeros((N, M, 2))

    print(f"Sweeping {N}×{M} = {N*M} grid points…")
    for i, tau in enumerate(TAU_GRID):
        for j, beta in enumerate(BETA_GRID):
            params = jnp.array([tau, beta])
            pipeline = make_pipeline(k_init, k_sim, tau, beta)
            J = compute_jacobian(pipeline, params)          # Float[4, 2]
            Sigma = estimate_noise_cov(make_run_fn(tau, beta), k_cov, K=20)
            F = fisher_information(J, Sigma)
            vals, vecs = eigendecomp(F)
            kappa_grid[i, j] = float(condition_number(vals))
            stiff_vecs[i, j] = np.array(vecs[:, 0])
            sloppy_vecs[i, j] = np.array(vecs[:, 1])
        print(f"  τ={tau:.2f} done")

    # --- Plot heatmap + quivers ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_condition_heatmap(TAU_GRID, BETA_GRID, kappa_grid,
                           xlabel="τ", ylabel="β", ax=axes[0],
                           title="Condition number κ(τ, β)")
    plot_eigenvector_quivers(TAU_GRID, BETA_GRID, stiff_vecs, sloppy_vecs, ax=axes[1])
    axes[1].set_xlabel("β"); axes[1].set_ylabel("τ")
    axes[1].set_title("Stiff / sloppy directions")
    plt.tight_layout()

    run_id = make_run_id()
    run_dir = save_results(run_id, CFG, {"kappa_grid": kappa_grid})
    fig.savefig(run_dir / "fim_landscape_phase1.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {run_dir}/fim_landscape_phase1.png")

    # --- Jacobian components vs τ at β=5 ---
    beta_fixed = 5.0
    J_tau = []
    for tau in TAU_GRID:
        params = jnp.array([tau, beta_fixed])
        J_ij = compute_jacobian(make_pipeline(k_init, k_sim, tau, beta_fixed), params)
        J_tau.append(np.array(J_ij[:, 0]))   # ∂s/∂τ

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_jacobian_components(TAU_GRID, np.array(J_tau), "τ", STAT_NAMES, ax=ax2)
    fig2.savefig(run_dir / "jacobian_vs_tau.png", dpi=150, bbox_inches="tight")
    print(f"Saved Jacobian plot.")


if __name__ == "__main__":
    main()
```

- [ ] **Smoke-test (quick):**
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
from experiments.exp_002_fim_landscape import make_pipeline, make_run_fn, CFG
import jax, jax.numpy as jnp
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number
from abm_geometry.rng import make_key
key = make_key(42)
ki, ks, kc = jax.random.split(key, 3)
p = jnp.array([0.4, 5.0])
J = compute_jacobian(make_pipeline(ki, ks, 0.4, 5.0), p)
Sigma = estimate_noise_cov(make_run_fn(0.4, 5.0), kc, K=5)
F = fisher_information(J, Sigma)
vals, _ = eigendecomp(F)
print('kappa =', float(condition_number(vals)))
print('PASS')
"
```
Expected: prints a finite kappa and `PASS`

- [ ] **Commit:**
```bash
git add experiments/exp_002_fim_landscape.py
git commit -m "feat: add exp_002_fim_landscape Phase I (τ,β) sweep experiment"
```

---

## Task 13: Create exp_003_heterogeneous.py (Phase II sweep)

**Files:**
- Create: `experiments/exp_003_heterogeneous.py`

- [ ] **Create the experiment:**

```python
"""Phase II FIM landscape: sweep (μ, σ) at fixed β=5 on a 12×12 grid.

Produces:
  - condition number heatmap over (μ, σ)
  - eigenvector quivers (3D FIM projected onto μ-σ subspace)
  - Brown/Sethna eigenvalue spectrum at 4 representative points
Saves plots to runs/<run_id>/
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
from abm_geometry.viz.landscape import plot_condition_heatmap, plot_eigenvector_quivers
from abm_geometry.viz.spectra import plot_eigenvalue_spectrum

BASE_CFG = Config(H=30, W=30, T=50, seed=42, beta=5.0, sigma_tau=0.0)
_SIM = jax.jit(simulate_with_beta, static_argnums=(2,))
BETA_FIXED = 5.0

MU_GRID = np.linspace(0.1, 0.8, 12)
SIGMA_GRID = np.linspace(0.01, 0.3, 12)


def make_pipeline_phase2(key_init, key_sim, mu_val, sigma_val):
    """Returns stats_fn(params=[mu,sigma,beta]) for jacfwd."""
    def pipeline(params):
        mu, sigma, beta = params[0], params[1], params[2]
        eps = jax.random.normal(key_init, (BASE_CFG.H, BASE_CFG.W))
        tol = jnp.clip(mu + sigma * eps, 0.001, 0.999)
        state = init_world(key_init, BASE_CFG).replace(tolerances=tol)
        final = _SIM(key_sim, state, BASE_CFG, beta)
        return stats_array_fn(final, beta)
    return pipeline


def make_run_fn_phase2(key_base, mu_val, sigma_val, beta_val):
    def run(key):
        ki, ks = jax.random.split(key)
        eps = jax.random.normal(ki, (BASE_CFG.H, BASE_CFG.W))
        tol = jnp.clip(mu_val + sigma_val * eps, 0.001, 0.999)
        state = init_world(ki, BASE_CFG).replace(tolerances=tol)
        final = _SIM(ks, state, BASE_CFG, beta_val)
        return stats_array_fn(final, beta_val)
    return run


def main():
    master_key = make_key(BASE_CFG.seed)
    k_init, k_sim, k_cov = jax.random.split(master_key, 3)

    N, M = len(MU_GRID), len(SIGMA_GRID)
    kappa_grid = np.zeros((N, M))
    stiff_vecs = np.zeros((N, M, 2))   # projected onto (μ, σ) subspace
    sloppy_vecs = np.zeros((N, M, 2))

    # Store full eigenvalues at 4 representative points for spectrum plot
    spectrum_points = {
        "low μ, low σ": (MU_GRID[2], SIGMA_GRID[1]),
        "high μ, low σ": (MU_GRID[-3], SIGMA_GRID[1]),
        "low μ, high σ": (MU_GRID[2], SIGMA_GRID[-2]),
        "high μ, high σ": (MU_GRID[-3], SIGMA_GRID[-2]),
    }
    spectrum_eigenvalues = {}

    print(f"Sweeping {N}×{M} = {N*M} grid points…")
    for i, mu in enumerate(MU_GRID):
        for j, sigma in enumerate(SIGMA_GRID):
            params = jnp.array([mu, sigma, BETA_FIXED])
            pipeline = make_pipeline_phase2(k_init, k_sim, mu, sigma)
            J = compute_jacobian(pipeline, params)           # Float[4, 3]
            Sigma = estimate_noise_cov(
                make_run_fn_phase2(k_cov, mu, sigma, BETA_FIXED), k_cov, K=20
            )
            F = fisher_information(J, Sigma)                 # Float[3, 3]
            vals, vecs = eigendecomp(F)
            kappa_grid[i, j] = float(condition_number(vals))
            # Project stiff/sloppy directions onto μ-σ plane (first two params)
            stiff_vecs[i, j] = np.array(vecs[:2, 0])
            sloppy_vecs[i, j] = np.array(vecs[:2, 1])

            # Store spectra at representative points
            for label, (mu_s, sigma_s) in spectrum_points.items():
                if abs(mu - mu_s) < 1e-6 and abs(sigma - sigma_s) < 1e-6:
                    spectrum_eigenvalues[label] = vals
        print(f"  μ={mu:.2f} done")

    # --- Heatmap + quivers ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_condition_heatmap(MU_GRID, SIGMA_GRID, kappa_grid,
                           xlabel="μ", ylabel="σ", ax=axes[0],
                           title="Condition number κ(μ, σ)  [β=5 fixed]")
    plot_eigenvector_quivers(MU_GRID, SIGMA_GRID, stiff_vecs, sloppy_vecs, ax=axes[1])
    axes[1].set_xlabel("σ"); axes[1].set_ylabel("μ")
    axes[1].set_title("Stiff / sloppy directions (μ-σ plane)")
    plt.tight_layout()

    run_id = make_run_id()
    run_dir = save_results(run_id, BASE_CFG, {"kappa_grid": kappa_grid})
    fig.savefig(run_dir / "fim_landscape_phase2.png", dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {run_dir}")

    # --- Eigenvalue spectrum ---
    if spectrum_eigenvalues:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plot_eigenvalue_spectrum(
            list(spectrum_eigenvalues.values()),
            list(spectrum_eigenvalues.keys()),
            ax=ax2,
            title="FIM eigenvalue spectrum at 4 (μ, σ) points",
        )
        fig2.savefig(run_dir / "eigenvalue_spectrum_phase2.png", dpi=150, bbox_inches="tight")
        print("Saved eigenvalue spectrum.")


if __name__ == "__main__":
    main()
```

- [ ] **Smoke-test (single point):**
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
import jax, jax.numpy as jnp
from abm_geometry.rng import make_key
from experiments.exp_003_heterogeneous import make_pipeline_phase2, make_run_fn_phase2, BASE_CFG
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number
key = make_key(42)
ki, ks, kc = jax.random.split(key, 3)
p = jnp.array([0.4, 0.15, 5.0])
J = compute_jacobian(make_pipeline_phase2(ki, ks, 0.4, 0.15), p)
Sigma = estimate_noise_cov(make_run_fn_phase2(kc, 0.4, 0.15, 5.0), kc, K=5)
F = fisher_information(J, Sigma)
vals, _ = eigendecomp(F)
print('eigenvalues:', vals)
print('kappa =', float(condition_number(vals)))
print('PASS')
"
```

- [ ] **Commit:**
```bash
git add experiments/exp_003_heterogeneous.py
git commit -m "feat: add exp_003_heterogeneous Phase II (μ,σ,β) sweep experiment"
```

---

## Task 14: Add notebook sections 8–12

**Files:**
- Modify: `notebooks/01_schelling_model.ipynb`

Add five new sections after the existing section 7 cell. Use `NotebookEdit` to append each section's cells. The content below is the complete source for each cell.

### Section 8 cells

- [ ] **Add markdown cell:**
```markdown
## 8. Summary Statistics Vector

To study the Fisher Information Matrix we need a **vector** of summary statistics $\mathbf{s}(\theta) \in \mathbb{R}^4$, not just a scalar. We use four differentiable statistics:

| # | Stat | What it measures |
|---|------|-----------------|
| 0 | Dissimilarity $D$ | global spatial segregation level |
| 1 | Mean satisfaction | average agent happiness (fraction of similar neighbours ≥ τ) |
| 2 | Moran's $I$ | spatial autocorrelation of type-A fraction; positive = clustered |
| 3 | Neighbourhood homogeneity | fraction of neighbours sharing the cell's type — a soft cluster-size proxy |

All four are fully differentiable via the underlying convolution operations.
```

- [ ] **Add code cell:**
```python
import numpy as np
from abm_geometry.statistics.summary import stats_array_fn

STAT_NAMES = ["Dissimilarity $D$", "Mean satisfaction", "Moran's $I$", "Neighbourhood\nhomogeneity"]
COLORS = ["#2b6cb0", "#c05621", "#276749", "#6b46c1"]

# Collect stats at every step
state8 = init_world(k_init, cfg)
step_keys8 = jax.random.split(k_sim, cfg.T)
stats_over_time = []
s8 = state8
for k8 in step_keys8:
    s8 = one_step(s8, k8, cfg)
    stats_over_time.append(np.array(stats_array_fn(s8, cfg.beta)))
stats_arr = np.array(stats_over_time)  # (T, 4)

fig, ax = plt.subplots(figsize=(9, 4))
for i, (name, color) in enumerate(zip(STAT_NAMES, COLORS)):
    ax.plot(range(1, cfg.T + 1), stats_arr[:, i], label=name, color=color, lw=2)
ax.set_xlabel("Step")
ax.set_ylabel("Statistic value")
ax.legend(ncol=2, fontsize=9)
ax.set_title("All four summary statistics over time")
plt.tight_layout()
plt.show()

# Table
init_stats = np.array(stats_array_fn(state8, cfg.beta))
print(f"{'Statistic':<30} {'t=0':>8} {'t=25':>8} {'t=50':>8}")
print("-" * 58)
for i, name in enumerate([s.replace("\n", " ") for s in STAT_NAMES]):
    print(f"{name:<30} {init_stats[i]:>8.3f} {stats_arr[24, i]:>8.3f} {stats_arr[-1, i]:>8.3f}")
```

### Section 9 cells

- [ ] **Add markdown cell:**
```markdown
## 9. Fisher Information Matrix (Phase I)

The **Fisher Information Matrix** quantifies how much information the summary statistics carry about the parameters $\theta$. It is:

$$F(\theta) = J^\top \Sigma^{-1} J, \quad J = \frac{\partial \mathbf{s}}{\partial \theta} \in \mathbb{R}^{4 \times p}$$

where $\Sigma$ is the noise covariance (estimated by re-running the simulation with $K=20$ different seeds at the same $\theta$).

**Eigenvalues of $F$** tell us which parameter directions are *stiff* (large $\lambda$ — well identified) vs *sloppy* (small $\lambda$ — hard to identify). A large condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ signals a **sloppy model**.

For Phase I we use $\theta = (\tau, \beta)$ — both the tolerance threshold and the sigmoid sharpness.
```

- [ ] **Add code cell:**
```python
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.spectrum import condition_number, eigendecomp
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.schelling.simulate import simulate_with_beta

_SIM_BETA = jax.jit(simulate_with_beta, static_argnums=(2,))

# Fix keys so Jacobian and noise cov use the same init
key9 = jax.random.PRNGKey(77)
k9_init, k9_sim, k9_cov = jax.random.split(key9, 3)

# Point in parameter space
params9 = jnp.array([0.4, 5.0])   # [tau, beta]

def stats_from_params9(params):
    tau, beta = params[0], params[1]
    state = init_world(k9_init, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final = _SIM_BETA(k9_sim, state, cfg, beta)
    return stats_array_fn(final, beta)

def run_fn9(key):
    ki, ks = jax.random.split(key)
    tau, beta = float(params9[0]), float(params9[1])
    state = init_world(ki, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final = _SIM_BETA(ks, state, cfg, params9[1])
    return stats_array_fn(final, params9[1])

print("Computing Jacobian…")
J9 = compute_jacobian(stats_from_params9, params9)          # Float[4, 2]
print("Estimating noise covariance (K=20 runs)…")
Sigma9 = estimate_noise_cov(run_fn9, k9_cov, K=20)
F9 = fisher_information(J9, Sigma9)                          # Float[2, 2]
vals9, vecs9 = eigendecomp(F9)
kappa9 = condition_number(vals9)

print(f"\nJacobian J (4 stats × 2 params):")
jac_df_labels = [s.replace('\n',' ') for s in STAT_NAMES]
for i, name in enumerate(jac_df_labels):
    print(f"  {name:<30}  ∂/∂τ={float(J9[i,0]):+.4f}   ∂/∂β={float(J9[i,1]):+.4f}")

print(f"\nFIM F = J^T Σ^{{-1}} J:")
print(np.array(F9).round(4))
print(f"\nEigenvalues: λ₁={float(vals9[0]):.4f},  λ₂={float(vals9[1]):.6f}")
print(f"Condition number κ = {float(kappa9):.1f}")
if float(kappa9) > 100:
    print("→ Sloppy model: parameters are poorly co-identified.")
else:
    print("→ Well-conditioned: both parameters are identifiable at this point.")
```

### Section 10 cells

- [ ] **Add markdown cell:**
```markdown
## 10. FIM Landscape over (τ, β)

We now sweep $\tau \in [0.1, 0.8]$ and $\beta \in [2, 15]$ on a $15 \times 15$ grid, computing the full FIM (and its eigendecomposition) at each point.

- **Left panel:** condition number $\kappa(\tau, \beta)$ on a log scale. Bright regions are sloppy (large $\kappa$); dark regions are stiff.
- **Right panel:** eigenvector quivers. Blue arrows point in the stiff direction (most information); red arrows point in the sloppy direction (least information).

*This takes ~3–5 minutes. The simulation is jit-compiled so each grid point runs in seconds.*
```

- [ ] **Add code cell:**
```python
from abm_geometry.viz.landscape import plot_condition_heatmap, plot_eigenvector_quivers

tau_vals = np.linspace(0.1, 0.8, 15)
beta_vals = np.linspace(2.0, 15.0, 15)

key10 = jax.random.PRNGKey(101)
k10_init, k10_sim, k10_cov = jax.random.split(key10, 3)

kappa_grid10 = np.zeros((15, 15))
stiff_vecs10  = np.zeros((15, 15, 2))
sloppy_vecs10 = np.zeros((15, 15, 2))

for i, tau_v in enumerate(tau_vals):
    for j, beta_v in enumerate(beta_vals):
        params_ij = jnp.array([tau_v, beta_v])

        def _pipeline(p, ki=k10_init, ks=k10_sim, c=cfg):
            t, b = p[0], p[1]
            s = init_world(ki, c).replace(tolerances=jnp.full((c.H, c.W), t))
            return stats_array_fn(_SIM_BETA(ks, s, c, b), b)

        def _run(key, tv=tau_v, bv=beta_v, c=cfg):
            ki, ks = jax.random.split(key)
            s = init_world(ki, c).replace(tolerances=jnp.full((c.H, c.W), tv))
            return stats_array_fn(_SIM_BETA(ks, s, c, bv), bv)

        J_ij   = compute_jacobian(_pipeline, params_ij)
        Sig_ij = estimate_noise_cov(_run, k10_cov, K=20)
        F_ij   = fisher_information(J_ij, Sig_ij)
        v_ij, e_ij = eigendecomp(F_ij)

        kappa_grid10[i, j]  = float(condition_number(v_ij))
        stiff_vecs10[i, j]  = np.array(e_ij[:, 0])
        sloppy_vecs10[i, j] = np.array(e_ij[:, 1])
    print(f"τ={tau_v:.2f} ✓", end="  ", flush=True)

fig10, axes10 = plt.subplots(1, 2, figsize=(13, 5))
plot_condition_heatmap(tau_vals, beta_vals, kappa_grid10,
                       xlabel="τ", ylabel="β", ax=axes10[0],
                       title="log₁₀ κ(τ, β)")
plot_eigenvector_quivers(tau_vals, beta_vals, stiff_vecs10, sloppy_vecs10, ax=axes10[1])
axes10[1].set_xlabel("β"); axes10[1].set_ylabel("τ")
axes10[1].set_title("Stiff (blue) / sloppy (red) eigenvectors")
plt.suptitle("Phase I FIM landscape", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

### Section 11 cells

- [ ] **Add markdown cell:**
```markdown
## 11. Heterogeneous Agents (Phase II)

So far every agent had the same tolerance $\tau$. In reality, individuals differ. We now model tolerance as **normally distributed**: $\tau_{ij} \sim \mathcal{N}(\mu, \sigma^2)$, clipped to $[0.001, 0.999]$.

This introduces a second scientific parameter $\sigma$ — the **heterogeneity** of the population. The reparametrisation trick ($\tau_{ij} = \mu + \sigma \varepsilon_{ij}$, $\varepsilon \sim \mathcal{N}(0,1)$) keeps gradients $\partial/\partial\mu$ and $\partial/\partial\sigma$ flowing correctly.

Our Phase II parameter vector is $\theta = (\mu, \sigma, \beta) \in \mathbb{R}^3$, giving a $3 \times 3$ FIM.
```

- [ ] **Add code cell:**
```python
from abm_geometry.schelling.state import init_world_heterogeneous
from abm_geometry.viz.grids import plot_grid_comparison

cfg_het = Config(H=cfg.H, W=cfg.W, T=cfg.T, density=cfg.density,
                 tau=0.4, sigma_tau=0.15, seed=cfg.seed)

key11 = jax.random.PRNGKey(11)
k11a, k11b = jax.random.split(key11)

state_hom = init_world(k11a, cfg)
state_het = init_world_heterogeneous(k11b, cfg_het)

final_hom = _SIM_BETA(k11a, state_hom, cfg, jnp.array(cfg.beta))
final_het = _SIM_BETA(k11b, state_het, cfg_het, jnp.array(cfg_het.beta))

# Side-by-side grids
plot_grid_comparison(
    [final_hom, final_het],
    [f"Homogeneous (σ=0)\nD={float(dissimilarity_index(final_hom.soft_occupancy)):.3f}",
     f"Heterogeneous (σ=0.15)\nD={float(dissimilarity_index(final_het.soft_occupancy)):.3f}"],
)
plt.suptitle(f"Same μ=0.4, β=5 — heterogeneity changes final segregation", y=1.02)
plt.show()

# Tolerance distribution for heterogeneous case
fig11, ax11 = plt.subplots(figsize=(6, 3))
tol_vals = np.array(state_het.tolerances).ravel()
ax11.hist(tol_vals[tol_vals > 0.01], bins=25, color="#276749", edgecolor="white", alpha=0.85)
ax11.axvline(0.4, color="red", lw=1.5, ls="--", label="μ = 0.4")
ax11.set_xlabel("Tolerance τ"); ax11.set_ylabel("Count")
ax11.set_title("Tolerance distribution (σ=0.15)")
ax11.legend(); plt.tight_layout(); plt.show()

# D-curve comparison
print("Running D-curve comparison…")
d_hom, d_het = [float(dissimilarity_index(state_hom.soft_occupancy))], [float(dissimilarity_index(state_het.soft_occupancy))]
sh, sh2 = state_hom, state_het
step_keys11 = jax.random.split(key11, cfg.T)
for kk in step_keys11:
    sh  = one_step(sh,  kk, cfg)
    sh2 = one_step(sh2, kk, cfg_het)
    d_hom.append(float(dissimilarity_index(sh.soft_occupancy)))
    d_het.append(float(dissimilarity_index(sh2.soft_occupancy)))

fig11b, ax11b = plt.subplots(figsize=(8, 4))
ax11b.plot(d_hom, label="Homogeneous (σ=0)",     color="#2b6cb0", lw=2)
ax11b.plot(d_het, label="Heterogeneous (σ=0.15)", color="#c05621", lw=2, ls="--")
ax11b.set_xlabel("Step"); ax11b.set_ylabel("Dissimilarity $D$")
ax11b.legend(); ax11b.set_title("Effect of tolerance heterogeneity on segregation dynamics")
plt.tight_layout(); plt.show()
```

### Section 12 cells

- [ ] **Add markdown cell:**
```markdown
## 12. FIM Landscape (Phase II)

With $\theta = (\mu, \sigma, \beta)$ and $\beta$ fixed at 5, we sweep $(\mu, \sigma)$ on a $12 \times 12$ grid. The FIM is now $3 \times 3$; we project stiff/sloppy directions onto the $(\mu, \sigma)$ plane.

**Key question:** does $\sigma$ add a genuinely new sloppy direction, or are $\mu$ and $\sigma$ well-separated in information content?
```

- [ ] **Add code cell:**
```python
from abm_geometry.viz.spectra import plot_eigenvalue_spectrum

mu_vals    = np.linspace(0.1, 0.8, 12)
sigma_vals = np.linspace(0.01, 0.3, 12)
BETA_FIXED = 5.0

key12 = jax.random.PRNGKey(202)
k12_init, k12_sim, k12_cov = jax.random.split(key12, 3)

kappa_grid12 = np.zeros((12, 12))
stiff12      = np.zeros((12, 12, 2))
sloppy12     = np.zeros((12, 12, 2))

# Points for eigenvalue spectrum
SPEC_POINTS = {
    "(μ=0.2, σ=0.05)": (1, 1),
    "(μ=0.7, σ=0.05)": (10, 1),
    "(μ=0.2, σ=0.25)": (1, 10),
    "(μ=0.7, σ=0.25)": (10, 10),
}
spec_eigs = {}

for i, mu_v in enumerate(mu_vals):
    for j, sig_v in enumerate(sigma_vals):
        params_ij = jnp.array([mu_v, sig_v, BETA_FIXED])

        def _pipe12(p, ki=k12_init, ks=k12_sim, c=cfg):
            mu, sigma, beta = p[0], p[1], p[2]
            eps = jax.random.normal(ki, (c.H, c.W))
            tol = jnp.clip(mu + sigma * eps, 0.001, 0.999)
            s = init_world(ki, c).replace(tolerances=tol)
            return stats_array_fn(_SIM_BETA(ks, s, c, beta), beta)

        def _run12(key, mv=mu_v, sv=sig_v, bv=BETA_FIXED, c=cfg):
            ki, ks = jax.random.split(key)
            eps = jax.random.normal(ki, (c.H, c.W))
            tol = jnp.clip(mv + sv * eps, 0.001, 0.999)
            s = init_world(ki, c).replace(tolerances=tol)
            return stats_array_fn(_SIM_BETA(ks, s, c, bv), bv)

        J12   = compute_jacobian(_pipe12, params_ij)
        Sig12 = estimate_noise_cov(_run12, k12_cov, K=20)
        F12   = fisher_information(J12, Sig12)
        v12, e12 = eigendecomp(F12)

        kappa_grid12[i, j] = float(condition_number(v12))
        stiff12[i, j]      = np.array(e12[:2, 0])
        sloppy12[i, j]     = np.array(e12[:2, 1])

        for label, (ri, rj) in SPEC_POINTS.items():
            if i == ri and j == rj:
                spec_eigs[label] = v12
    print(f"μ={mu_v:.2f} ✓", end="  ", flush=True)

# Heatmap + quivers
fig12, axes12 = plt.subplots(1, 2, figsize=(13, 5))
plot_condition_heatmap(mu_vals, sigma_vals, kappa_grid12,
                       xlabel="μ", ylabel="σ", ax=axes12[0],
                       title="log₁₀ κ(μ, σ)  [β=5 fixed]")
plot_eigenvector_quivers(mu_vals, sigma_vals, stiff12, sloppy12, ax=axes12[1])
axes12[1].set_xlabel("σ"); axes12[1].set_ylabel("μ")
axes12[1].set_title("Stiff (blue) / sloppy (red) in (μ, σ) plane")
plt.suptitle("Phase II FIM landscape", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# Eigenvalue spectrum — Brown/Sethna style
if spec_eigs:
    fig12b, ax12b = plt.subplots(figsize=(6, 4))
    plot_eigenvalue_spectrum(
        list(spec_eigs.values()),
        list(spec_eigs.keys()),
        ax=ax12b,
        title="FIM eigenvalue spectrum at 4 (μ, σ) points",
    )
    plt.tight_layout()
    plt.show()

    print("\nCondition numbers at representative points:")
    for label, v in spec_eigs.items():
        print(f"  {label}: κ = {float(condition_number(v)):.1f}")
```

- [ ] **Run full test suite to verify nothing broken:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Commit all notebook changes:**
```bash
git add notebooks/01_schelling_model.ipynb
git commit -m "feat: add notebook sections 8-12 — FIM machinery and Phase II heterogeneous agents"
```

---

## Final verification

- [ ] **Run complete test suite:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v --tb=short
```
Expected: all pass

- [ ] **Smoke-test the Phase I experiment (single point only, fast):**
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'src')
import jax, jax.numpy as jnp
from abm_geometry.rng import make_key
from abm_geometry.config import Config
from abm_geometry.schelling.state import init_world, init_world_heterogeneous
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.statistics.summary import stats_array_fn
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.geometry.spectrum import eigendecomp, condition_number

cfg = Config(H=15, W=15, T=20, seed=0)
key = make_key(0)
ki, ks, kc = jax.random.split(key, 3)
sim = jax.jit(simulate_with_beta, static_argnums=(2,))

def pipe(p):
    s = init_world(ki, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), p[0]))
    return stats_array_fn(sim(ks, s, cfg, p[1]), p[1])

def run(k):
    ki2, ks2 = jax.random.split(k)
    s = init_world(ki2, cfg).replace(tolerances=jnp.full((cfg.H, cfg.W), 0.4))
    return stats_array_fn(sim(ks2, s, cfg, 5.0), 5.0)

p = jnp.array([0.4, 5.0])
J = compute_jacobian(pipe, p)
S = estimate_noise_cov(run, kc, K=10)
F = fisher_information(J, S)
v, _ = eigendecomp(F)
print('Phase I kappa =', float(condition_number(v)))

# Phase II
cfg2 = Config(H=15, W=15, T=20, seed=0, sigma_tau=0.15)
def pipe2(p):
    mu, sigma, beta = p[0], p[1], p[2]
    eps = jax.random.normal(ki, (cfg2.H, cfg2.W))
    tol = jnp.clip(mu + sigma * eps, 0.001, 0.999)
    s = init_world(ki, cfg2).replace(tolerances=tol)
    return stats_array_fn(sim(ks, s, cfg2, beta), beta)

p2 = jnp.array([0.4, 0.15, 5.0])
J2 = compute_jacobian(pipe2, p2)
S2 = estimate_noise_cov(run, kc, K=10)
F2 = fisher_information(J2, S2)
v2, _ = eigendecomp(F2)
print('Phase II kappa =', float(condition_number(v2)))
print('ALL CHECKS PASSED')
"
```

- [ ] **Final commit tag:**
```bash
git tag -a v0.2-fim-phase2 -m "FIM machinery + heterogeneous agents (Phase I & II complete)"
```
