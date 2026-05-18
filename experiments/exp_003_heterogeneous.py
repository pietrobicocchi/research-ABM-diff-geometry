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
    stiff_vecs = np.zeros((N, M, 2))
    sloppy_vecs = np.zeros((N, M, 2))

    SPEC_POINTS = {
        "(μ=0.2, σ=0.05)": (1, 1),
        "(μ=0.7, σ=0.05)": (10, 1),
        "(μ=0.2, σ=0.25)": (1, 10),
        "(μ=0.7, σ=0.25)": (10, 10),
    }
    spectrum_eigenvalues = {}

    print(f"Sweeping {N}×{M} = {N*M} grid points…")
    for i, mu in enumerate(MU_GRID):
        for j, sigma in enumerate(SIGMA_GRID):
            params = jnp.array([mu, sigma, BETA_FIXED])
            pipeline = make_pipeline_phase2(k_init, k_sim, mu, sigma)
            J = compute_jacobian(pipeline, params)
            Sigma = estimate_noise_cov(
                make_run_fn_phase2(k_cov, mu, sigma, BETA_FIXED), k_cov, K=20
            )
            F = fisher_information(J, Sigma)
            vals, vecs = eigendecomp(F)
            kappa_grid[i, j] = float(condition_number(vals))
            stiff_vecs[i, j] = np.array(vecs[:2, 0])
            sloppy_vecs[i, j] = np.array(vecs[:2, 1])

            for label, (ri, rj) in SPEC_POINTS.items():
                if i == ri and j == rj:
                    spectrum_eigenvalues[label] = vals
        print(f"  μ={mu:.2f} done")

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
