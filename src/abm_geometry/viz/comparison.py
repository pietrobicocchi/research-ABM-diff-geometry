import matplotlib.pyplot as plt
import numpy as np


def plot_kappa_comparison(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_sim: np.ndarray,
    kappa_mf: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """Side-by-side log₁₀ κ heatmaps with a shared colour scale.

    p1_grid: 1-D array (N,) — row-axis values (y-axis).
    p2_grid: 1-D array (M,) — column-axis values (x-axis).
    kappa_sim / kappa_mf: shape (N, M).
    """
    log_sim = np.log10(np.clip(kappa_sim, 1.0, None))
    log_mf  = np.log10(np.clip(kappa_mf,  1.0, None))
    vmin = min(log_sim.min(), log_mf.min())
    vmax = max(log_sim.max(), log_mf.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log_k, title in zip(
        axes, [log_sim, log_mf], ["Stochastic sim", "Mean-field"]
    ):
        im = ax.pcolormesh(
            p2_grid, p1_grid, log_k,
            cmap="plasma", shading="auto", vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="log₁₀(κ)")
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_eigenvalue_comparison(
    vals_sim_list: list,
    vals_mf_list: list,
    labels: list[str],
    ax=None,
) -> plt.Axes:
    """Brown/Sethna eigenvalue spectrum: sim (solid) vs mean-field (dashed).

    vals_sim_list / vals_mf_list: lists of 1-D JAX/numpy arrays (one per point).
    Same colour is used for the sim and mf curves at the same parameter point.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    all_vals = [v for vs in [vals_sim_list, vals_mf_list] for v in vs]
    global_max = max(float(np.max(np.abs(np.array(v)))) for v in all_vals)

    for vals_sim, vals_mf, label, color in zip(
        vals_sim_list, vals_mf_list, labels, colors
    ):
        xs = np.arange(1, len(vals_sim) + 1)
        sorted_sim = np.sort(np.abs(np.array(vals_sim)))[::-1]
        sorted_mf  = np.sort(np.abs(np.array(vals_mf)))[::-1]
        ax.semilogy(xs, sorted_sim, "o-",  color=color, lw=2, ms=7, label=f"{label} sim")
        ax.semilogy(xs, sorted_mf,  "o--", color=color, lw=2, ms=7, label=f"{label} mf")

    ax.axhline(0.01 * global_max, color="grey", ls="--", alpha=0.6, label="1% threshold")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks(np.arange(1, max(len(v) for v in vals_sim_list) + 1))
    return ax


def plot_kappa_ratio(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_sim: np.ndarray,
    kappa_mf: np.ndarray,
    xlabel: str,
    ylabel: str,
    ax=None,
) -> plt.Axes:
    """Heatmap of log₁₀(κ_sim / κ_mf).

    Positive (red): stochasticity amplifies sloppiness.
    Negative (blue): stochasticity suppresses sloppiness.
    White contour at ratio=0: mean-field is exact here.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ratio = (
        np.log10(np.clip(kappa_sim, 1.0, None))
        - np.log10(np.clip(kappa_mf, 1.0, None))
    )
    vmax = max(np.abs(ratio).max(), 1e-3)
    im = ax.pcolormesh(
        p2_grid, p1_grid, ratio,
        cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="log₁₀(κ_sim / κ_mf)")
    ax.contour(p2_grid, p1_grid, ratio, levels=[0], colors="white", linewidths=1.5)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    return ax
