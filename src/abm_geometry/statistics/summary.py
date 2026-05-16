from src.abm_geometry.statistics.segregation import dissimilarity_index
from src.abm_geometry.types import SummaryStats, WorldState


def stats_fn(state: WorldState) -> SummaryStats:
    """Compute all summary statistics from a simulation state."""
    return SummaryStats(
        dissimilarity=dissimilarity_index(state.soft_occupancy),
    )
