import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


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
