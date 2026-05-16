"""Smoke-tests for the public API used by the tutorial notebook."""
import jax
import jax.numpy as jnp

from abm_geometry.config import Config
from abm_geometry.rng import make_key
from abm_geometry.schelling import init_world, one_step, simulate
from abm_geometry.statistics.segregation import dissimilarity_index


def _default_cfg():
    return Config(H=10, W=10, T=5, density=0.8, tau=0.4)


def test_one_step_exported():
    """one_step must be importable directly from abm_geometry.schelling."""
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    k1, _ = jax.random.split(key)
    next_state = one_step(state, k1, cfg)
    assert next_state.soft_occupancy.shape == (10, 10, 3)
    assert int(next_state.step) == 1


def test_simulate_returns_final_state():
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    final = simulate(key, state, cfg)
    assert final.soft_occupancy.shape == (10, 10, 3)
    assert int(final.step) == cfg.T


def test_dissimilarity_in_unit_range():
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    final = simulate(key, state, cfg)
    d = dissimilarity_index(final.soft_occupancy)
    assert 0.0 <= float(d) <= 1.0


def test_tolerance_sweep_shapes():
    """Running simulate with different τ values must all return valid states."""
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    _sim = jax.jit(simulate, static_argnums=(2,))
    for tau_val in [0.2, 0.4, 0.6]:
        s_tau = state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau_val))
        final = _sim(key, s_tau, cfg)
        assert final.soft_occupancy.shape == (10, 10, 3)
