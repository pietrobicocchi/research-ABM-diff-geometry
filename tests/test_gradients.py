import jax
import jax.numpy as jnp
import pytest

from abm_geometry.config import Config
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.schelling.simulate import simulate, simulate_with_beta
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.statistics.summary import stats_array_fn

CFG = Config(H=8, W=8, T=10, beta=5.0, tau_g=0.3, seed=7)

# Pre-build a fixed initial grid. Tolerances are overridden inside the
# differentiable function via .replace() + jnp.full so gradients flow.
# NEVER do Config(tau=float(traced_tau)) — float() on a traced value
# silently breaks autodiff by extracting the concrete value at trace time.
_KEY = jax.random.PRNGKey(CFG.seed)
_INIT_STATE = init_world(_KEY, CFG)
_SIM_JIT = jax.jit(simulate, static_argnums=(2,))


def _dissimilarity_from_tau(tau: jnp.ndarray) -> jnp.ndarray:
    """Differentiable pipeline: tau (scalar JAX array) → dissimilarity index (scalar)."""
    state = _INIT_STATE.replace(tolerances=jnp.full((CFG.H, CFG.W), tau))
    final = _SIM_JIT(_KEY, state, CFG)
    return dissimilarity_index(final.soft_occupancy)


def test_gradient_is_finite():
    tau = jnp.array(0.4)
    grad = jax.grad(_dissimilarity_from_tau)(tau)
    assert jnp.isfinite(grad), f"gradient is not finite: {grad}"


def test_jacfwd_matches_finite_difference():
    """jacfwd gradient should agree with central finite difference.

    atol=1e-2: Gumbel noise in the move rule makes the landscape slightly rough
    at finite step count (T=10), so 1e-4 is too tight for a small test grid.
    Phase I uses larger grids and more steps where the landscape is smoother.
    """
    tau = jnp.array(0.4)
    eps = 1e-3

    # Autodiff gradient
    grad_auto = compute_jacobian(_dissimilarity_from_tau, tau)

    # Central finite difference
    f_plus = _dissimilarity_from_tau(tau + eps)
    f_minus = _dissimilarity_from_tau(tau - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)

    assert jnp.allclose(grad_auto, grad_fd, atol=1e-2), (
        f"autodiff={float(grad_auto):.4f}, finite-diff={float(grad_fd):.4f}"
    )


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
