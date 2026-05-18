import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from abm_geometry.schelling.satisfaction import type_satisfaction


def gumbel_softmax_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    key: PRNGKeyArray,
    beta: float,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """One differentiable Schelling step via Gumbel-softmax.

    Each type makes an *independent* stay/move decision driven by its own
    satisfaction (frac_same_neighbours ≥ τ), with independent Gumbel noise.
    This replicates the core Schelling mechanism: type A moves away from
    B-dominated cells, type B moves away from A-dominated cells, so clusters
    self-reinforce.  Moving mass of both types redistributes proportionally to
    the current empty space (uniform over empty cells), preserving the per-cell
    simplex constraint exactly.

    tau_g controls relaxation sharpness: smaller → harder (closer to discrete).
    """
    H, W, _ = soft_occ.shape

    sat_A, sat_B = type_satisfaction(soft_occ, tolerances, beta)

    # Independent Gumbel-softmax stay/move for each type
    key_A, key_B = jax.random.split(key)

    logits_A = jnp.stack([beta * sat_A, beta * (1.0 - sat_A)], axis=-1)
    u_A = jax.random.uniform(key_A, (H, W, 2), minval=1e-6, maxval=1.0)
    gumbel_A = -jnp.log(-jnp.log(u_A))
    p_move_A = 1.0 - jax.nn.softmax((logits_A + gumbel_A) / tau_g, axis=-1)[..., 0]

    logits_B = jnp.stack([beta * sat_B, beta * (1.0 - sat_B)], axis=-1)
    u_B = jax.random.uniform(key_B, (H, W, 2), minval=1e-6, maxval=1.0)
    gumbel_B = -jnp.log(-jnp.log(u_B))
    p_move_B = 1.0 - jax.nn.softmax((logits_B + gumbel_B) / tau_g, axis=-1)[..., 0]

    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    empty = soft_occ[..., 0]

    # Cap total moving mass so it never exceeds total empty capacity.
    # Gradient flows through jnp.minimum.
    total_moving_uncapped = jnp.sum(occ_A * p_move_A) + jnp.sum(occ_B * p_move_B)
    cap_scale = jnp.minimum(jnp.sum(empty) / (total_moving_uncapped + 1e-8), 1.0)
    p_move_eff_A = p_move_A * cap_scale
    p_move_eff_B = p_move_B * cap_scale

    staying_A = occ_A * (1.0 - p_move_eff_A)
    moving_A = occ_A * p_move_eff_A
    staying_B = occ_B * (1.0 - p_move_eff_B)
    moving_B = occ_B * p_move_eff_B

    total_moving_A = jnp.sum(moving_A)
    total_moving_B = jnp.sum(moving_B)

    # Redistribute moving mass to cells weighted by emptiness (sum = 1).
    # cap_scale ensures (total_moving_A + total_moving_B) ≤ sum(empty),
    # so new_empty = 1 - new_A - new_B ≥ 0 exactly.
    empty_weight = empty / jnp.maximum(jnp.sum(empty), 1e-8)

    new_A = staying_A + total_moving_A * empty_weight
    new_B = staying_B + total_moving_B * empty_weight
    new_empty = 1.0 - new_A - new_B

    return jnp.stack([new_empty, new_A, new_B], axis=-1)
