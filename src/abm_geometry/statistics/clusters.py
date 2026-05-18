import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.schelling.neighbours import neighbour_type_fraction


def soft_mean_neighbourhood_homogeneity(
    soft_occ: Float[Array, "H W 3"],
) -> Float[Array, ""]:
    """Mean neighbourhood homogeneity — soft cluster-size proxy.

    For each cell, computes the fraction of occupied neighbours that share
    the cell's dominant type (weighted by occupancy probability).
    Returns a scalar in [0, 1]: 1 = perfectly clustered, 0.5 = random.
    Differentiable via the underlying convolution in neighbour_type_fraction.
    """
    neigh_frac = neighbour_type_fraction(soft_occ)  # Float[H, W, 2]
    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    # frac_same: if A-type, how many A-neighbours; if B-type, how many B-neighbours
    frac_same = occ_A * neigh_frac[..., 0] + occ_B * neigh_frac[..., 1]
    return jnp.mean(frac_same)
