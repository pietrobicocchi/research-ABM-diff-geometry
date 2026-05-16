import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.schelling.neighbours import neighbour_type_fraction


def soft_satisfaction(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta: float,
) -> Float[Array, "H W"]:
    """Occupancy-weighted soft satisfaction at each cell.

    For a type-A agent: sat_A = σ(β * (frac_A_neighbours - τ))
    For a type-B agent: sat_B = σ(β * (frac_B_neighbours - τ))
    Returns p_A * sat_A + p_B * sat_B — the expected satisfaction over type uncertainty.
    """
    neigh_frac = neighbour_type_fraction(soft_occ)  # Float[H, W, 2]

    sat_A = jax.nn.sigmoid(beta * (neigh_frac[..., 0] - tolerances))  # Float[H, W]
    sat_B = jax.nn.sigmoid(beta * (neigh_frac[..., 1] - tolerances))  # Float[H, W]

    # Weight by occupancy probability of each type
    return soft_occ[..., 1] * sat_A + soft_occ[..., 2] * sat_B
