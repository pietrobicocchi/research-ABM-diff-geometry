from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.schelling.move_rules import gumbel_softmax_step
from abm_geometry.types import WorldState


def one_step(state: WorldState, key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Apply one Gumbel-softmax Schelling step and increment the step counter."""
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, cfg.beta, cfg.tau_g
    )
    return state.replace(soft_occupancy=new_occ, step=state.step + 1)
