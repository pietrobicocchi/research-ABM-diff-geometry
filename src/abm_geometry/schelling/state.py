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


def init_world_heterogeneous(key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Place agents on the grid with tolerance sampled from N(cfg.tau, cfg.sigma_tau²).

    Uses the reparametrisation trick: tolerances = clip(cfg.tau + cfg.sigma_tau * ε)
    so gradients w.r.t. cfg.tau (μ) and cfg.sigma_tau (σ) flow correctly when
    those values are passed as JAX traced arrays via state.replace().
    Tolerances are clipped to [0.001, 0.999] to keep sigmoid inputs bounded.
    """
    key_cells, key_groups, key_tol = jax.random.split(key, 3)

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
    )

    eps = jax.random.normal(key_tol, (cfg.H, cfg.W))
    tolerances = jnp.clip(cfg.tau + cfg.sigma_tau * eps, 0.001, 0.999)

    return WorldState(
        soft_occupancy=soft_occ,
        tolerances=tolerances,
        step=jnp.int32(0),
    )
