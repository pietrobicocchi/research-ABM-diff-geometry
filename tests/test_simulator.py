import jax
import jax.numpy as jnp
import pytest

from src.abm_geometry.config import Config
from src.abm_geometry.schelling.state import init_world

CFG = Config(H=10, W=10, T=5, density=0.8, group_ratio=0.5, tau=0.4)


def test_init_world_occupancy_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.soft_occupancy.shape == (10, 10, 3)


def test_init_world_tolerances_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.tolerances.shape == (10, 10)


def test_init_world_sums_to_one():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sums = jnp.sum(state.soft_occupancy, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-5)


def test_init_world_tolerances_constant():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert jnp.allclose(state.tolerances, jnp.full((10, 10), 0.4))


def test_init_world_step_zero():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert int(state.step) == 0


from src.abm_geometry.schelling.neighbours import neighbour_type_fraction


def test_neighbour_fraction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    assert frac.shape == (10, 10, 2)


def test_neighbour_fraction_sums_to_one():
    """frac_A + frac_B should sum to 1 everywhere (by definition of fraction)."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    total = frac[..., 0] + frac[..., 1]
    assert jnp.allclose(total, jnp.ones((10, 10)), atol=1e-5)


def test_neighbour_fraction_all_same_type():
    """Grid entirely filled with type A → frac_A = 1 everywhere."""
    # All type A, no empty
    occ = jnp.zeros((5, 5, 3)).at[..., 1].set(1.0)  # all A
    frac = neighbour_type_fraction(occ)
    assert jnp.allclose(frac[..., 0], jnp.ones((5, 5)), atol=1e-5)
    assert jnp.allclose(frac[..., 1], jnp.zeros((5, 5)), atol=1e-5)


from src.abm_geometry.schelling.satisfaction import soft_satisfaction


def test_satisfaction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert sat.shape == (10, 10)


def test_satisfaction_range():
    """Satisfaction values should be in (0, 1) since they are sigmoid outputs."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert jnp.all(sat >= 0.0) and jnp.all(sat <= 1.0)


def test_satisfaction_zero_tolerance():
    """With τ=0, every agent is satisfied (any non-zero similar fraction ≥ 0)."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    # Replace tolerances with 0
    state_zero_tau = state.replace(tolerances=jnp.zeros((10, 10)))
    sat = soft_satisfaction(state_zero_tau.soft_occupancy, state_zero_tau.tolerances, beta=10.0)
    # Most cells should be highly satisfied; none should be < 0.5 at high beta
    assert jnp.mean(sat) > 0.6


def test_satisfaction_high_tolerance():
    """With τ=1, no agent is ever fully satisfied."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    state_high_tau = state.replace(tolerances=jnp.ones((10, 10)))
    sat = soft_satisfaction(state_high_tau.soft_occupancy, state_high_tau.tolerances, beta=10.0)
    assert jnp.mean(sat) < 0.4
