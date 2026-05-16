import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.types import WorldState


def init_world(key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Randomly place agents on the grid according to cfg.density and cfg.group_ratio.

    soft_occupancy is a one-hot encoding: each cell has exactly one of
    {empty, A, B} set to 1.0. tolerances are initialised uniformly to cfg.tau.
    """
    key_cells, key_groups = jax.random.split(key)

    is_occupied = jax.random.bernoulli(key_cells, cfg.density, shape=(cfg.H, cfg.W))
    is_A = jax.random.bernoulli(key_groups, cfg.group_ratio, shape=(cfg.H, cfg.W))
    is_A = is_A & is_occupied
    is_B = is_occupied & ~is_A

    soft_occ = jnp.stack(
        [
            (~is_occupied).astype(jnp.float32),
            is_A.astype(jnp.float32),
            is_B.astype(jnp.float32),
        ],
        axis=-1,
    )  # Float[H, W, 3]

    return WorldState(
        soft_occupancy=soft_occ,
        tolerances=jnp.full((cfg.H, cfg.W), cfg.tau),
        step=jnp.int32(0),
    )
