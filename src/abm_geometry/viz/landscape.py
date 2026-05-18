import matplotlib.pyplot as plt
import numpy as np


def plot_condition_heatmap(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    ax=None,
    title: str = "",
) -> plt.Axes:
    """Log10 condition-number heatmap over a 2D parameter grid.

    p1_grid: 1-D array of shape (N,) — row-axis values (y-axis of plot).
    p2_grid: 1-D array of shape (M,) — column-axis values (x-axis of plot).
    kappa_grid: shape (N, M) — condition numbers κ = λ_max/λ_min.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    log_kappa = np.log10(np.clip(kappa_grid, 1.0, None))
    im = ax.pcolormesh(
        p2_grid, p1_grid, log_kappa,
        cmap="plasma", shading="auto",
    )
    plt.colorbar(im, ax=ax, label="log₁₀(κ)")

    ax.contour(
        p2_grid, p1_grid, log_kappa,
        levels=[1, 2, 3],
        colors="white",
        linewidths=0.8,
        alpha=0.7,
    )
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    if title:
        ax.set_title(title)
    return ax


def plot_eigenvector_quivers(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    stiff_vecs: np.ndarray,
    sloppy_vecs: np.ndarray,
    ax=None,
    stride: int = 3,
) -> plt.Axes:
    """Quiver plot of stiff (blue) and sloppy (red) eigenvectors.

    stiff_vecs / sloppy_vecs: shape (N, M, 2) — eigenvectors at each grid point.
    stride: plot every stride-th point to avoid clutter.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    P1, P2 = np.meshgrid(p1_grid, p2_grid, indexing="ij")

    def _quiver(vecs, color, label):
        s = stride
        ax.quiver(
            P2[::s, ::s], P1[::s, ::s],
            vecs[::s, ::s, 1], vecs[::s, ::s, 0],
            color=color, alpha=0.8, scale=8, width=0.005, label=label,
        )

    _quiver(stiff_vecs, "#2563eb", "Stiff (λ_max)")
    _quiver(sloppy_vecs, "#dc2626", "Sloppy (λ_min)")
    ax.legend(fontsize=9, loc="upper right")
    return ax


def plot_jacobian_components(
    param_values: np.ndarray,
    J_values: np.ndarray,
    param_name: str,
    stat_names: list[str],
    ax=None,
) -> plt.Axes:
    """Plot ∂s_i/∂θ vs a swept scalar parameter.

    param_values: shape (N,)
    J_values: shape (N, 4) — one row per parameter value, one col per stat.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    colors = ["#2b6cb0", "#c05621", "#276749", "#6b46c1"]
    for i, (name, color) in enumerate(zip(stat_names, colors)):
        ax.plot(param_values, J_values[:, i], label=name, color=color, lw=2)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(param_name)
    ax.set_ylabel(f"∂s/∂{param_name}")
    ax.legend(fontsize=9)
    ax.set_title(f"Jacobian components vs {param_name}")
    return ax
