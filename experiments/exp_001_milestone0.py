"""Milestone 0 experiment: validate the end-to-end differentiable Schelling stack.

Runs the simulation, computes the Jacobian of the dissimilarity index w.r.t. tau,
prints a summary, saves results and a grid plot.
"""

import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from abm_geometry.config import Config, load_config
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.io.runs import make_run_id, save_results
from abm_geometry.rng import make_key
from abm_geometry.schelling import init_world, simulate
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.viz.grids import plot_grid

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"

_SIM_JIT = jax.jit(simulate, static_argnums=(2,))


def dissimilarity_from_tau(
    tau: jnp.ndarray, init_state, key, cfg: Config
) -> jnp.ndarray:
    """Differentiable pipeline: tau (scalar JAX array) → dissimilarity (scalar).

    Tolerances are set via jnp.full (differentiable). Never use float(tau) to
    build a Config — that breaks the autodiff trace.
    """
    state = init_state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final = _SIM_JIT(key, state, cfg)
    return dissimilarity_index(final.soft_occupancy)


def main():
    cfg = load_config(CFG_PATH)
    key = make_key(cfg.seed)

    print(f"Running Milestone 0 — H={cfg.H}, W={cfg.W}, T={cfg.T}, tau={cfg.tau}")

    # --- Forward simulation ---
    state = init_world(key, cfg)
    final_state = _SIM_JIT(key, state, cfg)

    d = dissimilarity_index(final_state.soft_occupancy)
    print(f"Dissimilarity index after {cfg.T} steps: {float(d):.4f}")

    # --- Jacobian (gradient of D w.r.t. tau) ---
    tau = jnp.array(cfg.tau)
    J = compute_jacobian(
        lambda t: dissimilarity_from_tau(t, state, key, cfg), tau
    )
    print(f"dD/dtau = {float(J):.4f}")

    # --- Save results ---
    run_id = make_run_id()
    run_dir = save_results(
        run_id, cfg,
        {"dissimilarity": d, "jacobian": J, "final_occupancy": final_state.soft_occupancy},
    )
    print(f"Results saved to: {run_dir}")

    # --- Grid plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_grid(state.soft_occupancy, title="Initial", ax=axes[0])
    plot_grid(final_state.soft_occupancy, title=f"After {cfg.T} steps", ax=axes[1])
    plot_path = run_dir / "grid.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Grid plot saved to: {plot_path}")
    plt.close()

    # --- Verify all 5 done criteria ---
    print("\n--- Milestone 0 verification ---")
    # 1. jit (already ran above without error)
    print("[1] jit: PASS")
    # 2. vmap — use plain simulate (not sim_jit) inside vmap; cfg is a closure constant
    keys = jax.random.split(key, 4)
    batch = jax.jit(jax.vmap(lambda k: simulate(k, init_world(k, cfg), cfg)))(keys)
    assert batch.soft_occupancy.shape == (4, cfg.H, cfg.W, 3)
    print("[2] vmap: PASS")
    # 3. differentiability
    grad = jax.grad(lambda t: dissimilarity_from_tau(t, state, key, cfg))(tau)
    assert jnp.isfinite(grad)
    print(f"[3] grad finite: PASS  (dD/dtau = {float(grad):.4f})")
    # 4. finite-diff agreement (checked by test suite)
    print("[4] finite-diff agreement: run pytest tests/test_gradients.py")
    # 5. no NaN/inf in Jacobian
    assert jnp.isfinite(J)
    print("[5] Jacobian finite: PASS")


if __name__ == "__main__":
    main()
