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
