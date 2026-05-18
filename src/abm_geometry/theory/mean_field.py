import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.config import Config
from abm_geometry.schelling.satisfaction import type_satisfaction
from abm_geometry.types import WorldState


def mean_field_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """Deterministic (noise-free) Schelling step.

    Replaces Gumbel-softmax stay/move sampling with its expected value:
        p_stay_type = σ( β·(2·sat_type − 1) / τ_g )
    All other logic (cap_scale, redistribution) is identical to
    gumbel_softmax_step. beta can be a Python float or a JAX traced array.
    """
    sat_A, sat_B = type_satisfaction(soft_occ, tolerances, beta)

    # Expected stay probability: E[Gumbel-softmax] with logits [β·sat, β·(1-sat)]
    p_stay_A = jax.nn.sigmoid((2.0 * sat_A - 1.0) * beta / tau_g)
    p_stay_B = jax.nn.sigmoid((2.0 * sat_B - 1.0) * beta / tau_g)

    p_move_A = 1.0 - p_stay_A
    p_move_B = 1.0 - p_stay_B

    occ_A = soft_occ[..., 1]
    occ_B = soft_occ[..., 2]
    empty = soft_occ[..., 0]

    total_moving_uncapped = jnp.sum(occ_A * p_move_A) + jnp.sum(occ_B * p_move_B)
    cap_scale = jnp.minimum(jnp.sum(empty) / (total_moving_uncapped + 1e-8), 1.0)
    p_move_eff_A = p_move_A * cap_scale
    p_move_eff_B = p_move_B * cap_scale

    staying_A = occ_A * (1.0 - p_move_eff_A)
    moving_A  = occ_A * p_move_eff_A
    staying_B = occ_B * (1.0 - p_move_eff_B)
    moving_B  = occ_B * p_move_eff_B

    total_moving_A = jnp.sum(moving_A)
    total_moving_B = jnp.sum(moving_B)

    empty_weight = empty / jnp.maximum(jnp.sum(empty), 1e-8)

    new_A     = staying_A + total_moving_A * empty_weight
    new_B     = staying_B + total_moving_B * empty_weight
    new_empty = 1.0 - new_A - new_B

    return jnp.stack([new_empty, new_A, new_B], axis=-1)


def simulate_mean_field(
    init_state: WorldState,
    cfg: Config,
    T_mf: int = 200,
) -> WorldState:
    """Run T_mf deterministic mean-field steps via lax.scan.

    cfg must be passed as a static argument when jitting:
        jax.jit(simulate_mean_field, static_argnums=(1,))
    Uses cfg.beta and cfg.tau_g directly. For differentiating w.r.t.
    tau or beta use simulate_mean_field_with_params instead.
    """
    def body(state, _):
        new_occ = mean_field_step(
            state.soft_occupancy, state.tolerances, cfg.beta, cfg.tau_g
        )
        return state.replace(soft_occupancy=new_occ, step=state.step + 1), None

    final, _ = jax.lax.scan(body, init_state, xs=None, length=T_mf)
    return final


def simulate_mean_field_with_params(
    init_state: WorldState,
    cfg: Config,
    tau,
    beta,
    T_mf: int = 200,
) -> WorldState:
    """Mean-field simulation with tau and beta as JAX traced values.

    Use this (not simulate_mean_field) when calling jax.jacfwd to differentiate
    w.r.t. tau or beta. Sets tolerances = jnp.full((H,W), tau) at each step so
    the gradient of any loss w.r.t. tau accumulates across all T_mf steps.

    cfg is static (provides H, W, tau_g for lax.scan).
    tau and beta are dynamic — they are traced by jacfwd.

    Jit pattern:
        sim_mf = jax.jit(simulate_mean_field_with_params, static_argnums=(1,))
        final  = sim_mf(init_state, cfg, tau_jax, beta_jax)
    """
    def body(state, _):
        tol     = jnp.full((cfg.H, cfg.W), tau)
        s       = state.replace(tolerances=tol)
        new_occ = mean_field_step(s.soft_occupancy, s.tolerances, beta, cfg.tau_g)
        return state.replace(
            soft_occupancy=new_occ, tolerances=tol, step=state.step + 1
        ), None

    init   = init_state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final, _ = jax.lax.scan(body, init, xs=None, length=T_mf)
    return final
