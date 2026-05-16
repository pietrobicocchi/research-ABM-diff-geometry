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
