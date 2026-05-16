import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from abm_geometry.schelling.satisfaction import soft_satisfaction


def gumbel_softmax_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    key: PRNGKeyArray,
    beta: float,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """One differentiable Schelling step via Gumbel-softmax.

    Each occupied cell independently decides whether to stay or move.
    Moving mass redistributes to cells weighted by their empty probability.
    This is a relaxed (soft) approximation of the discrete Schelling move rule.

    tau_g controls relaxation sharpness: smaller → harder (closer to discrete).

    The move probability is capped so that total moving mass never exceeds total
    empty capacity, ensuring the simplex constraint (A + B + empty = 1, all ≥ 0)
    holds exactly without renormalization.
    """
    H, W, _ = soft_occ.shape

    sat = soft_satisfaction(soft_occ, tolerances, beta)  # Float[H, W]

    # Stay/move logits and Gumbel noise
    logits = jnp.stack(
        [beta * sat, beta * (1.0 - sat)], axis=-1
    )  # Float[H, W, 2]; axis-2 = [stay, move]
    u = jax.random.uniform(key, (H, W, 2), minval=1e-6, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))  # Gumbel(0,1) samples
    p_stay = jax.nn.softmax((logits + gumbel) / tau_g, axis=-1)[..., 0]  # Float[H, W]

    occupied = soft_occ[..., 1:]  # Float[H, W, 2]; axis-2 = [A, B]
    empty = soft_occ[..., 0]  # Float[H, W]
    total_empty = jnp.sum(empty) + 1e-8  # scalar

    # Scale down move probability so total moving mass ≤ total empty capacity.
    # This ensures the redistribution never overfills any cell and new_empty ≥ 0.
    p_move = 1.0 - p_stay
    total_moving_uncapped = jnp.sum(occupied * p_move[..., None])
    cap_scale = jnp.minimum(total_empty / (total_moving_uncapped + 1e-8), 1.0)
    p_move_eff = p_move * cap_scale
    p_stay_eff = 1.0 - p_move_eff

    # Split occupied mass into staying and moving components
    staying = occupied * p_stay_eff[..., None]  # Float[H, W, 2]
    moving = occupied * p_move_eff[..., None]  # Float[H, W, 2]

    # Redistribute moving mass to cells proportional to their emptiness
    total_moving = jnp.sum(moving, axis=(0, 1))  # Float[2]
    empty_weight = empty / total_empty  # Float[H, W]; normalized so sum = 1

    new_A = staying[..., 0] + total_moving[0] * empty_weight
    new_B = staying[..., 1] + total_moving[1] * empty_weight
    # new_empty is exact (non-negative by cap_scale construction) — no clip needed
    new_empty = 1.0 - new_A - new_B

    return jnp.stack([new_empty, new_A, new_B], axis=-1)
