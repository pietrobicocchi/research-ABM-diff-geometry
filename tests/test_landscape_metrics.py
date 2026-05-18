import numpy as np
import pytest

from abm_geometry.geometry.landscape_metrics import (
    boundary_sharpness,
    corridor_area,
    kappa_max,
    kappa_median,
    log_kappa_variance,
)


def test_kappa_max():
    grid = np.array([[10.0, 100.0], [50.0, 1000.0]])
    assert kappa_max(grid) == 1000.0


def test_kappa_median():
    """Median of log10([10, 100, 1000, 10000]) = median([1,2,3,4]) = 2.5"""
    grid = np.array([[10.0, 100.0], [1000.0, 10000.0]])
    assert abs(kappa_median(grid) - 2.5) < 1e-6


def test_log_kappa_variance_uniform():
    """All same value → log10 values all equal → variance = 0."""
    grid = np.ones((4, 4)) * 100.0
    assert log_kappa_variance(grid) == pytest.approx(0.0, abs=1e-10)


def test_corridor_area_all_below():
    grid = np.ones((3, 3)) * 500.0
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(1.0)


def test_corridor_area_all_above():
    grid = np.ones((3, 3)) * 2000.0
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(0.0)


def test_corridor_area_half():
    """Half below threshold → 0.5."""
    grid = np.array([[500.0, 2000.0], [500.0, 2000.0]])
    assert corridor_area(grid, threshold=1000.0) == pytest.approx(0.5)


def test_boundary_sharpness_hard_step():
    """Hard step from below low_threshold to above high_threshold.

    p1 = [0.1, 0.3, 0.5, 0.7], two p2 columns.
    Rows 0-1: kappa=50 (below low_threshold=100)
    Rows 2-3: kappa=50000 (above high_threshold=10000)

    For each column: last row below low_threshold is index 1 (p1=0.3),
    first row above high_threshold is index 2 (p1=0.5).
    Width = 0.5 - 0.3 = 0.2.
    """
    p1 = np.array([0.1, 0.3, 0.5, 0.7])
    p2 = np.array([1.0, 2.0])
    kappa = np.array([
        [50.0,   50.0],
        [50.0,   50.0],
        [50000.0, 50000.0],
        [50000.0, 50000.0],
    ])
    width = boundary_sharpness(p1, p2, kappa, low_threshold=100.0, high_threshold=10000.0)
    assert abs(width - 0.2) < 1e-6


def test_boundary_sharpness_no_transition():
    """All below low_threshold → no transition found → returns 0.0."""
    p1 = np.array([0.1, 0.3, 0.5, 0.7])
    p2 = np.array([1.0, 2.0])
    kappa = np.ones((4, 2)) * 50.0
    assert boundary_sharpness(p1, p2, kappa) == pytest.approx(0.0)
