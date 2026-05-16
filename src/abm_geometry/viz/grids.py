import numpy as np
import matplotlib.pyplot as plt
from jax import Array
from jaxtyping import Float


def plot_grid(
    soft_occ: Float[Array, "H W 3"],
    title: str = "",
    ax=None,
):
    """Render soft_occupancy as RGB. Empty=white, type A=red, type B=blue.

    Returns the matplotlib Axes object.
    """
    occ = np.array(soft_occ)
    # RGB: R channel = type A, B channel = type B, G channel = 0
    empty = occ[..., 0:1]
    rgb = np.concatenate(
        [
            occ[..., 1:2] + empty,  # R: A mass + empty whiteness
            empty,                   # G: just empty whiteness
            occ[..., 2:3] + empty,  # B: B mass + empty whiteness
        ],
        axis=-1,
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
