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

# Two representative points: one low-κ and one high-κ corner
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
