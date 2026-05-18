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
