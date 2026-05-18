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
    sat_A, sat_B = type_satisfaction(soft_occ, tolerances, beta)
    return soft_occ[..., 1] * sat_A + soft_occ[..., 2] * sat_B


def type_satisfaction(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta: float,
) -> tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """Per-type soft satisfaction: (sat_A, sat_B).

    sat_A = σ(β * (frac_A_neighbours - τ))
    sat_B = σ(β * (frac_B_neighbours - τ))
    """
    neigh_frac = neighbour_type_fraction(soft_occ)  # Float[H, W, 2]
    sat_A = jax.nn.sigmoid(beta * (neigh_frac[..., 0] - tolerances))
    sat_B = jax.nn.sigmoid(beta * (neigh_frac[..., 1] - tolerances))
    return sat_A, sat_B
