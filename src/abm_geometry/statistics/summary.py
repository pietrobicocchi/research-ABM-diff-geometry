import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.statistics.clusters import soft_mean_neighbourhood_homogeneity
from abm_geometry.statistics.segregation import dissimilarity_index, moran_i
from abm_geometry.schelling.satisfaction import soft_satisfaction
from abm_geometry.types import SummaryStats, WorldState


def stats_fn(state: WorldState) -> SummaryStats:
    """Compute all summary statistics from a simulation state."""
    return SummaryStats(
        dissimilarity=dissimilarity_index(state.soft_occupancy),
    )


def stats_array_fn(state: WorldState, beta) -> Float[Array, "4"]:
    """Differentiable 4-stat summary vector: [D, mean_sat, Moran_I, homogeneity].

    beta can be a Python float or a JAX traced array (for jacfwd through beta).
    """
    occ = state.soft_occupancy
    occupied = 1.0 - occ[..., 0]                           # Float[H, W]

    d = dissimilarity_index(occ)

    sat = soft_satisfaction(occ, state.tolerances, beta)    # Float[H, W]
    mean_sat = jnp.sum(sat * occupied) / (jnp.sum(occupied) + 1e-8)

    i = moran_i(occ)

    h = soft_mean_neighbourhood_homogeneity(occ)

    return jnp.array([d, mean_sat, i, h])
