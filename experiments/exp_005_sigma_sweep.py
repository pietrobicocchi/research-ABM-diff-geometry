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
