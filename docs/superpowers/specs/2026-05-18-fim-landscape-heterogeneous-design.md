# FIM Landscape & Heterogeneous Agents — Design Spec

**Date:** 2026-05-18
**Phases covered:** I (complete), II (heterogeneous tolerance)
**Status:** approved, ready for implementation

---

## Goal

Wire up the full Fisher Information Matrix machinery, extend the model to
heterogeneous agent tolerance (Phase II), and add five notebook sections with
publication-quality scientific plots inspired by the sloppy-models literature
(Brown/Sethna 2003, Gutenkunst 2007).

---

## Parameter Vectors

| Phase | θ | dim | FIM shape |
|-------|---|-----|-----------|
| I     | (τ, β) | 2 | 2×2 |
| II    | (μ, σ, β) | 3 | 3×3 |

`τ_g` (Gumbel temperature) is a fixed numerical parameter, not in θ.
`β` (sigmoid sharpness) is included because it genuinely changes model
behaviour and may be sloppy/stiff relative to τ — a testable prediction.

---

## Summary Statistics Vector

`stats_array_fn(state) → Float[Array, "4"]` in order:

| idx | name | formula |
|-----|------|---------|
| 0 | dissimilarity D | existing `dissimilarity_index` |
| 1 | mean satisfaction | `mean(soft_satisfaction * occupancy_mask)` over occupied cells |
| 2 | Moran's I | `(z · Wz) / (z · z)`, z = centred type-A fraction per cell, W = Moore adjacency (row-normalised) |
| 3 | mean neighbourhood homogeneity | `mean(frac_same_neigh * (occ_A + occ_B))` — soft cluster-size proxy, fully differentiable via convolution |

The existing `SummaryStats` namedtuple and `stats_fn` in `types.py` /
`statistics/summary.py` are kept for human-readable display. A new
`stats_array_fn` entry point returns the plain array used by geometry code.

---

## Module Architecture

### New / modified source files

| File | Action | Purpose |
|------|--------|---------|
| `statistics/segregation.py` | modify | add `moran_i()` |
| `statistics/clusters.py` | create | `soft_mean_neighbourhood_homogeneity()` (convolution-based) |
| `statistics/summary.py` | modify | add `stats_array_fn` returning `Float[Array, "4"]` |
| `geometry/fim.py` | create | `estimate_noise_cov`, `fisher_information` |
| `geometry/spectrum.py` | create | `eigendecomp`, `condition_number`, `sloppy_modes` |
| `schelling/state.py` | modify | add `init_world_heterogeneous` |
| `config.py` | modify | add `sigma_tau: float = 0.0` |
| `viz/spectra.py` | create | eigenvalue spectrum (log scale, sloppy-models style) |
| `viz/landscape.py` | create | Jacobian components, condition-number heatmap, eigenvector quivers |
| `viz/grids.py` | modify | add `plot_grid_comparison` |
| `experiments/exp_002_fim_landscape.py` | create | Phase I (τ,β) sweep |
| `experiments/exp_003_heterogeneous.py` | create | Phase II (μ,σ) sweep |
| `notebooks/01_schelling_model.ipynb` | modify | add sections 8–12 |

---

## Geometry: FIM Computation

### `geometry/fim.py`

```
estimate_noise_cov(
    stats_fn,   # WorldState → Float[4]
    params,     # Float[p] — current parameter point
    cfg,        # Config (static)
    key,        # PRNGKeyArray
    K=20,       # number of Monte Carlo samples
) → Float[Array, "4 4"]
```

Procedure:
1. vmap `stats_fn` over K independent keys (each key → different init + sim seed)
2. Collect `Float[K, 4]` sample matrix
3. Return `jnp.cov(samples.T)`  (shape 4×4)
4. If `jnp.linalg.cond(Σ) > 1e6`: warn, fall back to identity

```
fisher_information(
    J,          # Float[Array, "4 p"]  — Jacobian
    noise_cov,  # Float[Array, "4 4"] — Σ
) → Float[Array, "p p"]
    returns J.T @ inv(Σ) @ J
```

### `geometry/spectrum.py`

```
eigendecomp(F)        → (vals Float["p"], vecs Float["p p"])   # jnp.linalg.eigh, descending
condition_number(vals) → scalar   # vals[0] / (vals[-1] + 1e-12)
sloppy_modes(vals, vecs, threshold=0.01)
    → list[(value, vector)] where value < threshold * vals[0]
```

---

## Statistics Implementation

### `moran_i`

```python
# W is the Moore-neighbourhood adjacency (row-normalised, no self-loop)
# implemented as sum_moore_neighbours / 8  (already available in neighbours.py)
z = occ_A / (occ_A + occ_B + 1e-8) - mean_frac_A   # centred type-A fraction
Wz = sum_moore_neighbours(z) / 8
return jnp.sum(z * Wz) / (jnp.sum(z * z) + 1e-8)
```

### `soft_mean_neighbourhood_homogeneity`

```python
# frac_same[h,w] = frac_A_neigh if cell is A-leaning, frac_B_neigh if B-leaning
frac_same = occ_A * neigh_frac_A + occ_B * neigh_frac_B  # already computed in satisfaction
return jnp.mean(frac_same)  # scalar in [0, 1]
```

---

## Phase II: Heterogeneous Tolerance

### `init_world_heterogeneous(key, cfg) → WorldState`

```python
key_cells, key_groups, key_tol = jax.random.split(key, 3)
# Reparametrisation trick so gradients flow w.r.t. mu and sigma:
eps = jax.random.normal(key_tol, (cfg.H, cfg.W))
tolerances = jnp.clip(cfg.tau + cfg.sigma_tau * eps, 0.001, 0.999)
```

`cfg.tau` is reused as μ (mean tolerance). `cfg.sigma_tau` is new (default 0.0,
fully backward-compatible). The reparametrisation keeps gradients ∂/∂μ and
∂/∂σ flowing correctly through the tolerance sampling.

### `config.py` addition

```python
sigma_tau: float = 0.0   # std of tolerance distribution; 0 = homogeneous
```

### Differentiable pipeline for Phase II

```python
def stats_from_params(params):   # params = [mu, sigma, beta]
    mu, sigma, beta = params
    state = init_world_heterogeneous(key, cfg_with(tau=mu, sigma_tau=sigma))
    final = simulate(k_sim, state, cfg_with(beta=beta))
    return stats_array_fn(final)

J = jax.jacfwd(stats_from_params)(jnp.array([mu, sigma, beta]))   # Float[4, 3]
Σ = estimate_noise_cov(...)
F = fisher_information(J, Σ)   # Float[3, 3]
```

---

## Visualisation

### `viz/spectra.py` — `plot_eigenvalue_spectrum`

- x-axis: eigenvalue index (1 … p), sorted descending
- y-axis: eigenvalue magnitude, **log scale**
- Multiple θ points overlaid with distinct colours + legend
- Horizontal grey dashed line at 1% of max eigenvalue ("sloppy threshold")
- Style: clean white background, markers at each index, lines connecting

### `viz/landscape.py` — three functions

**`plot_jacobian_components(param_grid, J_grid, param_name, stat_names)`**
- One subplot per parameter direction
- One line per summary statistic showing ∂si/∂θj vs swept param
- Identifies which stat provides signal where

**`plot_condition_heatmap(p1_grid, p2_grid, kappa_grid, xlabel, ylabel)`**
- 2D heatmap, log-scale colorbar (log10 κ)
- Contour lines at κ = 10, 100, 1000
- Title shows which parameter is fixed

**`plot_eigenvector_quivers(p1_grid, p2_grid, stiff_vecs, sloppy_vecs, ax)`**
- Overlaid on heatmap
- Blue arrows = stiff direction (λ_max eigenvector)
- Red arrows = sloppy direction (λ_min eigenvector)
- Subset of grid points (every 3rd) to avoid clutter

### `viz/grids.py` addition — `plot_grid_comparison`

Side-by-side grid panels with shared legend, for Phase II heterogeneous vs
homogeneous comparison.

---

## Notebook Sections (8–12)

### Section 8: Summary Statistics Vector
- Run single simulation, plot all 4 stats over time on one figure
- Table: stat values at t=0, t=25, t=50
- One paragraph explaining each stat and why it matters for identifiability

### Section 9: Fisher Information Matrix (Phase I)
- 3-sentence FIM explainer with the formula F = J^T Σ^{-1} J
- Compute at θ=(τ=0.4, β=5.0): show the 2×2 matrix, eigenvalues, condition number
- Explain "sloppy" (κ >> 1) vs "stiff" (κ ≈ 1) in plain language

### Section 10: FIM Landscape over (τ, β)
- 15×15 grid sweep of τ ∈ [0.1, 0.8], β ∈ [2, 15]
- Left panel: condition-number heatmap (log10 scale)
- Right panel: eigenvector quivers (stiff=blue, sloppy=red)
- Commentary: where is the model identifiable? What's the sloppy direction?

### Section 11: Heterogeneous Agents
- Side-by-side grids: σ=0 vs σ=0.15 (same μ=0.4, same seed)
- Tolerance distribution histogram for σ=0.15
- D curve comparison: does heterogeneity increase/decrease segregation?
- Introduce θ=(μ, σ, β) parameter vector

### Section 12: FIM Landscape (Phase II)
- Fix β=5, sweep (μ, σ) on 12×12 grid: μ ∈ [0.1, 0.8], σ ∈ [0.01, 0.3]
- Condition-number heatmap with eigenvector quivers
- Eigenvalue spectrum panel: plot 3 eigenvalues at 4 representative (μ,σ) points
  (corner + centre) — Brown/Sethna style with log y-axis
- Key question answered: does σ add a sloppy direction, or does it always remain stiff?

---

## Constraints & Non-Goals

- `τ_g` stays fixed (numerical parameter, not in θ)
- No per-agent state tracking (stays as per-cell soft occupancy)
- No cost-function contour plots (Phase III contribution, needs analytical limit)
- Grid sweeps run on CPU with jit — expect ~2–5 min for the 15×15 sweep
- All sweep loops are Python loops over a precompiled jitted function (not vmap
  over params, since cfg must be static)
