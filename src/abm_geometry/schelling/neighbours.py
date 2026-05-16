import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def _sum_moore_neighbours(x: Float[Array, "H W"]) -> Float[Array, "H W"]:
    """Sum values across the 8 Moore neighbours (periodic / toroidal boundary)."""
    result = jnp.zeros_like(x)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            result = result + jnp.roll(x, shift=(di, dj), axis=(0, 1))
    return result
    # Note: Python loop over 8 constants — JAX unrolls this at trace time.


def neighbour_type_fraction(
    soft_occ: Float[Array, "H W 3"],
) -> Float[Array, "H W 2"]:
    """Fraction of type-A and type-B mass in each cell's Moore neighbourhood.

    Returns Float[H, W, 2] where axis-2 = (frac_A, frac_B).
    frac_A[h, w] + frac_B[h, w] = 1 by definition.
    """
    neigh_A = _sum_moore_neighbours(soft_occ[..., 1])  # Float[H, W]
    neigh_B = _sum_moore_neighbours(soft_occ[..., 2])  # Float[H, W]
    neigh_total = neigh_A + neigh_B + 1e-8  # avoid division by zero in empty regions

    return jnp.stack([neigh_A / neigh_total, neigh_B / neigh_total], axis=-1)
