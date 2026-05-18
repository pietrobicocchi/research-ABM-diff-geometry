import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalue_spectrum(
    eigenvalues_list: list,
    labels: list[str],
    ax=None,
    title: str = "FIM eigenvalue spectrum",
) -> plt.Axes:
    """Brown/Sethna-style eigenvalue spectrum plot.

    eigenvalues_list: list of 1-D arrays, one per parameter-space point.
    labels: one label per array, shown in the legend.
    Horizontal grey dashed line marks the 1% sloppy threshold relative to
    the maximum eigenvalue across all points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eigenvalues_list)))
    global_max = max(float(np.max(np.abs(v))) for v in eigenvalues_list)

    for vals, label, color in zip(eigenvalues_list, labels, colors):
        vals_np = np.array(vals)
        idx = np.argsort(vals_np)[::-1]
        sorted_vals = vals_np[idx]
        xs = np.arange(1, len(sorted_vals) + 1)
        ax.semilogy(xs, sorted_vals, "o-", color=color, label=label, lw=2, ms=7)

    ax.axhline(
        0.01 * global_max,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="1% sloppy threshold",
    )
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xticks(np.arange(1, max(len(v) for v in eigenvalues_list) + 1))
    return ax
