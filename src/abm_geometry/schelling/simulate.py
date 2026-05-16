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
