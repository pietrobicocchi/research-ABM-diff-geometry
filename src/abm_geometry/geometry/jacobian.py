from typing import Callable

import jax
from jax import Array
from jaxtyping import Float


def compute_jacobian(
    fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    params: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute the Jacobian of fn at params using forward-mode autodiff.

    Uses jacfwd, which is efficient when the number of parameters p is small
    relative to the number of outputs d (always true for Phases I-II where p ≤ 2).

    For Phase III+ where p grows large (per-cell tolerances), switch to jacrev.
    """
    return jax.jacfwd(fn)(params)
