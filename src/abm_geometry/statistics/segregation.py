import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.schelling.neighbours import _sum_moore_neighbours


def dissimilarity_index(soft_occ: Float[Array, "H W 3"]) -> Float[Array, ""]:
    """Dissimilarity index D ∈ [0, 1].

    D = 0.5 * Σ_i |a_i/A - b_i/B|
    where a_i = type-A mass in cell i, b_i = type-B mass, A = total A, B = total B.

    D=0: perfectly integrated. D=1: perfectly segregated.
    Differentiable w.r.t. soft_occ except at the isolated zero set of |frac_A - frac_B|.
    """
    total_A = jnp.sum(soft_occ[..., 1]) + 1e-8
    total_B = jnp.sum(soft_occ[..., 2]) + 1e-8
    frac_A = soft_occ[..., 1] / total_A  # Float[H, W]
    frac_B = soft_occ[..., 2] / total_B  # Float[H, W]
    return 0.5 * jnp.sum(jnp.abs(frac_A - frac_B))


def moran_i(soft_occ: Float[Array, "H W 3"]) -> Float[Array, ""]:
    """Moran's I spatial autocorrelation of type-A fraction.

    I = (z · Wz) / (z · z), where z is the centred type-A fraction per cell
    and W is the row-normalised Moore adjacency (sum of 8 neighbours / 8).
    Returns values in [-1, 1]: positive = clustered, negative = dispersed.
    """
    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    frac_A = occ_A / (occ_A + occ_B + 1e-8)     # Float[H, W]
    mean_frac_A = jnp.mean(frac_A)
    z = frac_A - mean_frac_A                      # centred

    Wz = _sum_moore_neighbours(z) / 8.0           # row-normalised Moore average
    numerator = jnp.sum(z * Wz)
    denominator = jnp.sum(z * z) + 1e-8
    return numerator / denominator
