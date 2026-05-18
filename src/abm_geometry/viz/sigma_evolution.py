import matplotlib.pyplot as plt
import numpy as np

_METRIC_LABELS = {
    "kappa_max":           ("κ_max",                    "Predicted: ↓"),
    "kappa_median":        ("median(log₁₀ κ)",          "Predicted: ↓"),
    "log_kappa_variance":  ("Var(log₁₀ κ)",             "Predicted: ↓"),
    "corridor_area":       ("Corridor area (κ < 1000)", "Predicted: ↑"),
    "boundary_sharpness":  ("Boundary width (τ units)", "Predicted: ↑"),
}


def plot_sigma_landscape_panel(
    sigma_values: list,
    kappa_grids: list,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    xlabel: str = "τ",
    ylabel: str = "β",
) -> plt.Figure:
    """Seven-panel heatmap of κ(τ,β), one panel per σ, shared colour scale."""
    n = len(sigma_values)
    log_grids = [np.log10(np.clip(kg, 1.0, None)) for kg in kappa_grids]
    vmin = min(g.min() for g in log_grids)
    vmax = max(g.max() for g in log_grids)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, sigma, log_kg in zip(axes, sigma_values, log_grids):
        im = ax.pcolormesh(
            p2_grid, p1_grid, log_kg,
            cmap="plasma", shading="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"σ = {sigma:.2f}", fontsize=10)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_ylabel(xlabel, fontsize=9)

    plt.colorbar(im, ax=axes[-1], label="log₁₀(κ)", shrink=0.8)
    plt.tight_layout()
    return fig


def plot_sigma_metric_curves(
    sigma_values: list,
    metrics: dict,
) -> plt.Figure:
    """Four-panel figure: one subplot per metric showing value vs σ.

    metrics: dict with keys matching _METRIC_LABELS (kappa_max,
    kappa_median, log_kappa_variance, corridor_area, boundary_sharpness).
    boundary_sharpness shares a panel with corridor_area (right y-axis).
    """
    plot_keys = ["kappa_max", "kappa_median", "log_kappa_variance", "corridor_area"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    sigmas = np.array(sigma_values)

    for ax, key in zip(axes, plot_keys):
        label, prediction = _METRIC_LABELS[key]
        ax.plot(sigmas, metrics[key], "o-", color="#2b6cb0", lw=2, ms=7)
        ax.set_xlabel("σ")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n{prediction}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Overlay boundary_sharpness on the corridor_area panel (right y-axis)
        if key == "corridor_area" and "boundary_sharpness" in metrics:
            ax2 = ax.twinx()
            ax2.plot(sigmas, metrics["boundary_sharpness"], "s--",
                     color="#c05621", lw=2, ms=6, label="Boundary width ↑")
            ax2.set_ylabel("Boundary width (τ units)", color="#c05621")
            ax2.tick_params(axis="y", labelcolor="#c05621")
            ax2.legend(loc="upper left", fontsize=8)

    plt.suptitle("Landscape metrics vs population heterogeneity σ", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_sigma_kappa_histograms(
    sigma_values: list,
    kappa_grids: list,
) -> plt.Figure:
    """Overlaid histograms of log10(κ), one per σ value."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_values)))

    for sigma, kg, color in zip(sigma_values, kappa_grids, colors):
        log_k = np.log10(np.clip(kg.ravel(), 1.0, None))
        ax.hist(log_k, bins=20, alpha=0.45, color=color,
                label=f"σ={sigma:.2f}", density=True)

    ax.set_xlabel("log₁₀(κ)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of log₁₀(κ) across the (τ,β) grid for each σ")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return fig
