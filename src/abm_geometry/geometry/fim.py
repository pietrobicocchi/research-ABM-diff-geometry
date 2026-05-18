import warnings
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray


def estimate_noise_cov(
    run_fn: Callable,
    key: PRNGKeyArray,
    K: int = 20,
) -> Float[Array, "4 4"]:
    """Estimate noise covariance Σ of summary statistics via Monte Carlo.

    run_fn: PRNGKey → Float[4]  — one simulation run returning 4 stats.
    Draws K independent runs, returns the sample covariance matrix.
    Falls back to identity if the result is ill-conditioned (κ > 1e6).
    """
    keys = jax.random.split(key, K)
    samples = jax.jit(jax.vmap(run_fn))(keys)   # Float[K, 4]
    Sigma = jnp.cov(samples.T)                   # Float[4, 4]
    cond = float(jnp.linalg.cond(Sigma))
    if cond > 1e6:
        warnings.warn(
            f"Noise covariance ill-conditioned (κ={cond:.2e}); using identity Σ=I.",
            stacklevel=2,
        )
        return jnp.eye(4)
    return Sigma


def fisher_information(
    J: Float[Array, "4 p"],
    noise_cov: Float[Array, "4 4"],
) -> Float[Array, "p p"]:
    """Fisher Information Matrix F = J^T Σ^{-1} J."""
    Sigma_inv = jnp.linalg.inv(noise_cov)
    return J.T @ Sigma_inv @ J
