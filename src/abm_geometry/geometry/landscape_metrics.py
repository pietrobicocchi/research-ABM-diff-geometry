import numpy as np


def kappa_max(kappa_grid: np.ndarray) -> float:
    """Peak condition number across the grid."""
    return float(np.max(kappa_grid))


def kappa_median(kappa_grid: np.ndarray) -> float:
    """Median of log10(κ) across the grid."""
    return float(np.median(np.log10(np.clip(kappa_grid, 1.0, None))))


def log_kappa_variance(kappa_grid: np.ndarray) -> float:
    """Variance of log10(κ) — spread of the landscape.

    Zero means all points have the same condition number (maximally uniform).
    Decreasing variance means σ is homogenising the landscape.
    """
    log_k = np.log10(np.clip(kappa_grid, 1.0, None))
    return float(np.var(log_k))


def corridor_area(kappa_grid: np.ndarray, threshold: float = 1000.0) -> float:
    """Fraction of grid cells with κ < threshold (the identifiable corridor)."""
    return float(np.mean(kappa_grid < threshold))


def boundary_sharpness(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    kappa_grid: np.ndarray,
    low_threshold: float = 100.0,
    high_threshold: float = 10000.0,
) -> float:
    """Width (in p1 units) of the κ transition zone, averaged over p2 columns.

    For each p2 column: measures the p1 distance between the last cell
    below low_threshold and the first cell above high_threshold.
    A wide transition = soft/gradual boundary. A narrow transition = sharp.
    Returns 0.0 if no valid transition is found in any column.
    """
    widths = []
    for j in range(kappa_grid.shape[1]):
        col = kappa_grid[:, j]
        below = np.where(col < low_threshold)[0]
        above = np.where(col > high_threshold)[0]
        if len(below) > 0 and len(above) > 0:
            p1_lo = p1_grid[below[-1]]   # last p1 value still below low_threshold
            p1_hi = p1_grid[above[0]]    # first p1 value already above high_threshold
            if p1_hi > p1_lo:
                widths.append(p1_hi - p1_lo)
    return float(np.mean(widths)) if widths else 0.0


def bootstrap_landscape_metrics(
    kappa_grids: list,
    n_bootstrap: int = 500,
    threshold: float = 1000.0,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """95% confidence intervals for landscape metrics via spatial bootstrap.

    Resamples grid cells with replacement to estimate metric uncertainty due
    to the finite parameter grid. Does NOT capture FIM estimation noise (which
    requires saving raw noise samples from estimate_noise_cov); it captures
    spatial sampling variability — i.e., how sensitive each metric is to
    which specific (τ,β) points happen to be included in the grid.

    Covers kappa_max, kappa_median, log_kappa_variance, corridor_area.
    boundary_sharpness is excluded because it depends on spatial ordering
    that the cell-resample bootstrap destroys.

    Returns dict: metric_name → (lower_ci, upper_ci), each Float[n_sigma].
    """
    alpha = (1.0 - confidence) / 2.0
    rng   = np.random.default_rng(seed)

    _fns = {
        "kappa_max":          lambda kg: kappa_max(kg),
        "kappa_median":       lambda kg: kappa_median(kg),
        "log_kappa_variance": lambda kg: log_kappa_variance(kg),
        "corridor_area":      lambda kg: corridor_area(kg, threshold),
    }

    lowers: dict = {n: [] for n in _fns}
    uppers: dict = {n: [] for n in _fns}

    for kg in kappa_grids:
        flat = kg.ravel()
        n    = len(flat)
        boot: dict = {k: np.empty(n_bootstrap) for k in _fns}

        for b in range(n_bootstrap):
            resample = rng.choice(flat, n, replace=True).reshape(kg.shape)
            for name, fn in _fns.items():
                boot[name][b] = fn(resample)

        for name in _fns:
            lowers[name].append(np.percentile(boot[name], 100.0 * alpha))
            uppers[name].append(np.percentile(boot[name], 100.0 * (1.0 - alpha)))

    return {n: (np.array(lowers[n]), np.array(uppers[n])) for n in _fns}
