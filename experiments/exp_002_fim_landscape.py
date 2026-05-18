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
            J = compute_jacobian(pipeline, params)
            Sigma = estimate_noise_cov(make_run_fn(tau, beta), k_cov, K=20)
            F = fisher_information(J, Sigma)
            vals, vecs = eigendecomp(F)
            kappa_grid[i, j] = float(condition_number(vals))
            stiff_vecs[i, j] = np.array(vecs[:, 0])
            sloppy_vecs[i, j] = np.array(vecs[:, 1])
        print(f"  τ={tau:.2f} done")

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

    beta_fixed = 5.0
    J_tau = []
    for tau in TAU_GRID:
        params = jnp.array([tau, beta_fixed])
        J_ij = compute_jacobian(make_pipeline(k_init, k_sim, tau, beta_fixed), params)
        J_tau.append(np.array(J_ij[:, 0]))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_jacobian_components(TAU_GRID, np.array(J_tau), "τ", STAT_NAMES, ax=ax2)
    fig2.savefig(run_dir / "jacobian_vs_tau.png", dpi=150, bbox_inches="tight")
    print(f"Saved Jacobian plot.")


if __name__ == "__main__":
    main()
