import dataclasses
import datetime
import json
from pathlib import Path

import numpy as np

from abm_geometry.rng import get_git_sha


def make_run_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{get_git_sha()}"


def save_results(
    run_id: str,
    config,
    results: dict,
    base_dir: Path = Path("outputs"),
) -> Path:
    """Save config as JSON and numpy arrays as .npz. Returns the run directory."""
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    np.savez(run_dir / "results.npz", **{k: np.array(v) for k, v in results.items()})
    return run_dir
