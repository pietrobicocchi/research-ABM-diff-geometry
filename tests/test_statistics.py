import jax.numpy as jnp

from abm_geometry.statistics.segregation import dissimilarity_index, moran_i
from abm_geometry.statistics.summary import stats_fn
from abm_geometry.types import WorldState


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


def test_moran_i_uniform():
    """All cells equal A fraction → z=0 everywhere → I=0."""
    occ = jnp.zeros((4, 4, 3)).at[..., 1].set(0.5).at[..., 2].set(0.5)
    assert jnp.allclose(moran_i(occ), jnp.array(0.0), atol=1e-5)


def test_moran_i_range():
    """Moran's I should be in [-1, 1] for any valid soft_occ."""
    import jax
    key = jax.random.PRNGKey(7)
    occ = jax.random.dirichlet(key, jnp.ones(3), shape=(8, 8))
    i = float(moran_i(occ))
    assert -1.0 - 1e-4 <= i <= 1.0 + 1e-4


def test_moran_i_segregated():
    """Left half all A, right half all B → strong positive autocorrelation.

    Uses an 8×8 grid (not 4×4): on a toroidal 4×4 grid with only 4 columns
    the periodic wrap makes every boundary cell see 3/8 opposite-type neighbours,
    analytically giving I=0.25. An 8×8 grid yields I≈0.625.
    """
    occ = jnp.zeros((8, 8, 3))
    occ = occ.at[:, :4, 1].set(1.0)   # left half: type A
    occ = occ.at[:, 4:, 2].set(1.0)   # right half: type B
    assert float(moran_i(occ)) > 0.3


def test_stats_fn_returns_summary_stats():
    from abm_geometry.types import SummaryStats
    import jax

    key = jax.random.PRNGKey(0)
    from abm_geometry.config import Config
    from abm_geometry.schelling.state import init_world

    cfg = Config(H=5, W=5, T=3)
    state = init_world(key, cfg)
    stats = stats_fn(state)
    assert isinstance(stats, SummaryStats)
    assert stats.dissimilarity.shape == ()


def test_homogeneity_all_same_type():
    """Grid all type A → all neighbours same type → homogeneity = 1.0."""
    from abm_geometry.statistics.clusters import soft_mean_neighbourhood_homogeneity

    occ = jnp.zeros((4, 4, 3)).at[..., 1].set(1.0)
    h = soft_mean_neighbourhood_homogeneity(occ)
    assert jnp.allclose(h, jnp.array(1.0), atol=1e-4)


def test_homogeneity_range():
    """Homogeneity should be in [0, 1]."""
    import jax
    from abm_geometry.statistics.clusters import soft_mean_neighbourhood_homogeneity

    occ = jax.random.dirichlet(jax.random.PRNGKey(3), jnp.ones(3), shape=(6, 6))
    h = float(soft_mean_neighbourhood_homogeneity(occ))
    assert 0.0 - 1e-4 <= h <= 1.0 + 1e-4
