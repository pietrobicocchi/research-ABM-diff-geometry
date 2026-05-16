from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """All simulation hyperparameters. Frozen so it can be used as a JAX static arg."""

    H: int = 30
    W: int = 30
    T: int = 50           # number of simulation steps
    density: float = 0.8  # fraction of cells that are occupied
    group_ratio: float = 0.5  # of occupied cells, fraction that are type A
    beta: float = 5.0     # inverse temperature for satisfaction sigmoid
    tau_g: float = 0.5    # Gumbel temperature; smaller → harder moves
    tau: float = 0.4      # homogeneous tolerance threshold (Phase I)
    seed: int = 42


def load_config(path: Path) -> Config:
    """Load a Config from a YAML file. Missing keys fall back to defaults."""
    import omegaconf

    raw = omegaconf.OmegaConf.load(path)
    overrides = omegaconf.OmegaConf.to_container(raw, resolve=True)
    return Config(**overrides)
