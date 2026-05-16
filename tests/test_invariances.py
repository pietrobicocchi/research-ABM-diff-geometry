import jax
import jax.numpy as jnp

from abm_geometry.config import Config
from abm_geometry.schelling.state import init_world
from abm_geometry.schelling.simulate import simulate
from abm_geometry.statistics.segregation import dissimilarity_index

CFG = Config(H=8, W=8, T=5, seed=3)


def _swap_AB(occ):
    """Swap type-A and type-B channels: axis-2 [empty, A, B] → [empty, B, A]."""
    return jnp.stack([occ[..., 0], occ[..., 2], occ[..., 1]], axis=-1)


def test_dissimilarity_invariant_under_AB_swap():
    """Swapping labels A↔B should leave the dissimilarity index unchanged."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = jax.jit(simulate, static_argnums=(2,))(key, state, CFG)

    d_original = dissimilarity_index(final.soft_occupancy)
    d_swapped = dissimilarity_index(_swap_AB(final.soft_occupancy))
    assert jnp.allclose(d_original, d_swapped, atol=1e-5), (
        f"D={float(d_original):.4f} vs swapped={float(d_swapped):.4f}"
    )


def test_tolerances_constant_across_cells():
    """In the homogeneous case, all tolerances should equal cfg.tau."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert jnp.allclose(state.tolerances, jnp.full((CFG.H, CFG.W), CFG.tau))
