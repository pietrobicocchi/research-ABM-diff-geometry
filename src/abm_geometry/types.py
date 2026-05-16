from typing import NamedTuple

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int


@chex.dataclass(frozen=True)
class WorldState:
    """Immutable simulation state. All fields are JAX arrays."""

    soft_occupancy: Float[Array, "H W 3"]
    # axis=-1 encodes [p_empty, p_A, p_B]; rows sum to 1 along axis=-1

    tolerances: Float[Array, "H W"]
    # per-cell tolerance threshold; value ignored where cell is empty

    step: Int[Array, ""]
    # scalar step counter


class SummaryStats(NamedTuple):
    """Differentiable summary statistics of the simulation."""

    dissimilarity: Float[Array, ""]
    # dissimilarity index D ∈ [0, 1]; 0 = fully integrated, 1 = fully segregated
