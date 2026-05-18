import jax
from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.schelling.step import one_step
from abm_geometry.types import WorldState


def simulate(key: PRNGKeyArray, init_state: WorldState, cfg: Config) -> WorldState:
    """Run cfg.T steps of the Schelling model, returning the final state.

    Uses lax.scan for JIT-friendly iteration — no Python-level loop at runtime.
    cfg must be passed as a static argument when jitting:
        jax.jit(simulate, static_argnums=(2,))
    """

    def body(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        return (one_step(state, subkey, cfg), key), None

    (final_state, _), _ = jax.lax.scan(
        body, (init_state, key), xs=None, length=cfg.T
    )
    return final_state


def simulate_with_beta(
    key: PRNGKeyArray,
    init_state: WorldState,
    cfg: Config,
    beta_jax,
) -> WorldState:
    """Like simulate, but accepts beta as a JAX traced value.

    Use this instead of simulate when differentiating w.r.t. beta.
    cfg is still static (for lax.scan length); beta_jax is dynamic.

    The import of gumbel_softmax_step is inside the function body to avoid
    a circular import.
    """
    from abm_geometry.schelling.move_rules import gumbel_softmax_step

    def body(carry, _):
        state, k = carry
        k, subkey = jax.random.split(k)
        new_occ = gumbel_softmax_step(
            state.soft_occupancy, state.tolerances, subkey, beta_jax, cfg.tau_g
        )
        return (state.replace(soft_occupancy=new_occ, step=state.step + 1), k), None

    (final_state, _), _ = jax.lax.scan(
        body, (init_state, key), xs=None, length=cfg.T
    )
    return final_state
