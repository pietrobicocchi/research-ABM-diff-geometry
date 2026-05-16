import jax.numpy as jnp
import pytest

from src.abm_geometry.statistics.segregation import dissimilarity_index
from src.abm_geometry.statistics.summary import stats_fn
from src.abm_geometry.types import WorldState


def test_dissimilarity_fully_segregated():
    """All A on left half, all B on right half → D = 1."""
    occ = jnp.zeros((1, 2, 3))
    occ = occ.at[0, 0, 1].set(1.0)  # cell (0,0) is type A
    occ = occ.at[0, 1, 2].set(1.0)  # cell (0,1) is type B
    d = dissimilarity_index(occ)
    assert jnp.allclose(d, jnp.array(1.0), atol=1e-5)


def test_dissimilarity_fully_integrated():
    """Equal A and B in every cell → D = 0."""
    occ = jnp.zeros((2, 2, 3)).at[..., 1].set(0.5).at[..., 2].set(0.5)
    d = dissimilarity_index(occ)
    assert jnp.allclose(d, jnp.array(0.0), atol=1e-5)


def test_dissimilarity_range():
    """D should always be in [0, 1]."""
    import jax

    key = jax.random.PRNGKey(0)
    occ_raw = jax.random.dirichlet(key, jnp.ones(3), shape=(5, 5))
    d = dissimilarity_index(occ_raw)
    assert 0.0 <= float(d) <= 1.0 + 1e-5


def test_stats_fn_returns_summary_stats():
    from src.abm_geometry.types import SummaryStats
    import jax

    key = jax.random.PRNGKey(0)
    from src.abm_geometry.config import Config
    from src.abm_geometry.schelling.state import init_world

    cfg = Config(H=5, W=5, T=3)
    state = init_world(key, cfg)
    stats = stats_fn(state)
    assert isinstance(stats, SummaryStats)
    assert stats.dissimilarity.shape == ()
