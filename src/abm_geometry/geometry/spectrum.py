import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def eigendecomp(
    F: Float[Array, "p p"],
) -> tuple[Float[Array, "p"], Float[Array, "p p"]]:
    """Eigendecomposition of symmetric FIM, sorted descending.

    Returns (eigenvalues, eigenvectors) where eigenvectors[:, i] is the
    i-th eigenvector corresponding to eigenvalues[i].
    Uses jnp.linalg.eigh (stable for symmetric matrices, real eigenvalues).
    """
    vals, vecs = jnp.linalg.eigh(F)          # ascending by default
    idx = jnp.argsort(vals)[::-1]            # descending
    return vals[idx], vecs[:, idx]


def condition_number(eigenvalues: Float[Array, "p"]) -> Float[Array, ""]:
    """κ = λ_max / λ_min.  Large κ means sloppy (poorly identified) model."""
    return eigenvalues[0] / (eigenvalues[-1] + 1e-12)


def sloppy_modes(
    eigenvalues: Float[Array, "p"],
    eigenvectors: Float[Array, "p p"],
    threshold: float = 0.01,
) -> list[tuple]:
    """Return (value, vector) pairs where λ < threshold * λ_max."""
    lam_max = float(eigenvalues[0])
    return [
        (eigenvalues[i], eigenvectors[:, i])
        for i in range(len(eigenvalues))
        if float(eigenvalues[i]) < threshold * lam_max
    ]
