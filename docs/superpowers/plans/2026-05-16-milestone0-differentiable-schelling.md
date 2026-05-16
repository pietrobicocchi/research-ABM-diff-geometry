# Milestone 0 — Differentiable Schelling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal end-to-end stack that validates the architecture: a jit-able, vmappable Schelling simulator whose dissimilarity index is differentiable w.r.t. τ and whose Jacobian agrees with finite differences.

**Architecture:** Pure-functional JAX with `chex.dataclass` state, Gumbel-softmax move rule (Quera-Bofarull style), and `jacfwd` for Jacobian computation. The simulation runs as a `lax.scan` loop so it is JIT-friendly and vmappable.

**Tech Stack:** Python ≥3.11, JAX ≥0.4.25 (CPU), chex ≥0.1.86, jaxtyping ≥0.2.28, optax, omegaconf, matplotlib, seaborn, pytest, pyright.

---

## File map

Files to create (in order):

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata and dependencies |
| `.gitignore` | Ignore outputs/, .venv/, __pycache__ |
| `src/abm_geometry/__init__.py` | Package marker |
| `src/abm_geometry/types.py` | `WorldState`, `SummaryStats` |
| `src/abm_geometry/config.py` | `Config` dataclass + YAML loader |
| `src/abm_geometry/rng.py` | `make_key`, `get_git_sha` |
| `src/abm_geometry/schelling/__init__.py` | Re-exports public API |
| `src/abm_geometry/schelling/state.py` | `init_world` |
| `src/abm_geometry/schelling/neighbours.py` | `neighbour_type_fraction` |
| `src/abm_geometry/schelling/satisfaction.py` | `soft_satisfaction` |
| `src/abm_geometry/schelling/move_rules.py` | `gumbel_softmax_step` |
| `src/abm_geometry/schelling/step.py` | `one_step` |
| `src/abm_geometry/schelling/simulate.py` | `simulate` |
| `src/abm_geometry/statistics/__init__.py` | Package marker |
| `src/abm_geometry/statistics/segregation.py` | `dissimilarity_index` |
| `src/abm_geometry/statistics/summary.py` | `SummaryStats`, `stats_fn` |
| `src/abm_geometry/geometry/__init__.py` | Package marker |
| `src/abm_geometry/geometry/jacobian.py` | `compute_jacobian` |
| `src/abm_geometry/calibrate/__init__.py` | Empty placeholder |
| `src/abm_geometry/viz/__init__.py` | Package marker |
| `src/abm_geometry/viz/grids.py` | `plot_grid` |
| `src/abm_geometry/io/__init__.py` | Package marker |
| `src/abm_geometry/io/runs.py` | `make_run_id`, `save_results` |
| `experiments/configs/default.yaml` | Default Config values |
| `experiments/exp_001_milestone0.py` | Entry point for Milestone 0 |
| `tests/__init__.py` | Test package marker |
| `tests/test_simulator.py` | Shape, mass conservation, jit, vmap |
| `tests/test_gradients.py` | Finite-diff vs jacfwd agreement |
| `tests/test_invariances.py` | A↔B relabelling invariance |
| `tests/test_statistics.py` | Known-grid dissimilarity values |
| `outputs/.gitkeep` | Keep outputs/ tracked but empty |

---

## Key implementation notes

**`simulate` takes `cfg: Config` as a static JAX argument.** `lax.scan` requires a
compile-time `length`; passing Config as a frozen dataclass and marking it static via
`jax.jit(simulate, static_argnums=(2,))` handles this cleanly.

**`chex.dataclass` `.replace()` method.** Use `state.replace(field=new_val)` to
create updated state — not `dataclasses.replace`. Both work but chex's `.replace()`
is idiomatic.

**Gumbel-softmax semantics.** `tau_g` → 0 makes moves harder (more like argmax).
`tau_g` → ∞ makes moves uniform. Typical values: 0.3–0.7.

**No `jacfwd` inside `jit`.** `compute_jacobian` is called outside jit. The
differentiated function `stats_from_tau` calls `jit`-compiled `simulate` internally.

---

## Task 1 — Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `outputs/.gitkeep`
- Create: all `__init__.py` package markers

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "abm-geometry"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jax[cpu]>=0.4.25",
    "chex>=0.1.86",
    "jaxtyping>=0.2.28",
    "optax>=0.2.2",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "omegaconf>=2.3",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pyright>=1.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.hatch.build.targets.wheel]
packages = ["src/abm_geometry"]
```

- [ ] **Step 2: Write `.gitignore`**

```
outputs/
!outputs/.gitkeep
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
dist/
.pyright/
```

- [ ] **Step 3: Create directory structure and empty `__init__.py` files**

Run:
```bash
mkdir -p src/abm_geometry/schelling
mkdir -p src/abm_geometry/statistics
mkdir -p src/abm_geometry/geometry
mkdir -p src/abm_geometry/calibrate
mkdir -p src/abm_geometry/viz
mkdir -p src/abm_geometry/io
mkdir -p experiments/configs
mkdir -p tests
mkdir -p outputs
touch src/abm_geometry/__init__.py
touch src/abm_geometry/schelling/__init__.py
touch src/abm_geometry/statistics/__init__.py
touch src/abm_geometry/geometry/__init__.py
touch src/abm_geometry/calibrate/__init__.py
touch src/abm_geometry/viz/__init__.py
touch src/abm_geometry/io/__init__.py
touch tests/__init__.py
touch outputs/.gitkeep
```

- [ ] **Step 4: Install dependencies**

Run:
```bash
uv sync
```

Expected: dependencies install without error. `uv sync --extra dev` for dev tools.

- [ ] **Step 5: Verify JAX works**

Run:
```bash
uv run python -c "import jax; import chex; import jaxtyping; print(jax.__version__)"
```

Expected: prints a version string ≥ 0.4.25 with no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ outputs/ experiments/
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2 — Core types

**Files:**
- Create: `src/abm_geometry/types.py`

- [ ] **Step 1: Write `types.py`**

```python
from typing import NamedTuple

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int


@chex.dataclass(frozen=True)
class WorldState:
    """Immutable simulation state. All fields are JAX arrays."""

    soft_occupancy: Float[Array, "H W 3"]
    # axis=-1 encodes [p_empty, p_A, p_B]; rows sum to 1 along axis=-1

    tolerances: Float[Array, "H W"]
    # per-cell tolerance threshold; value ignored where cell is empty

    step: Int[Array, ""]
    # scalar step counter


class SummaryStats(NamedTuple):
    """Differentiable summary statistics of the simulation."""

    dissimilarity: Float[Array, ""]
    # dissimilarity index D ∈ [0, 1]; 0 = fully integrated, 1 = fully segregated
```

- [ ] **Step 2: Verify the types import and WorldState is a JAX PyTree**

Run:
```bash
uv run python -c "
import jax
from src.abm_geometry.types import WorldState, SummaryStats
import jax.numpy as jnp

state = WorldState(
    soft_occupancy=jnp.ones((5, 5, 3)) / 3,
    tolerances=jnp.full((5, 5), 0.4),
    step=jnp.int32(0),
)
leaves, treedef = jax.tree_util.tree_flatten(state)
print('leaves:', len(leaves), '  shapes:', [l.shape for l in leaves])
"
```

Expected output: `leaves: 3   shapes: [(5, 5, 3), (5, 5), ()]`

- [ ] **Step 3: Commit**

```bash
git add src/abm_geometry/types.py
git commit -m "feat: add WorldState and SummaryStats types"
```

---

## Task 3 — Config and RNG helpers

**Files:**
- Create: `src/abm_geometry/config.py`
- Create: `src/abm_geometry/rng.py`
- Create: `experiments/configs/default.yaml`

- [ ] **Step 1: Write `config.py`**

```python
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
```

- [ ] **Step 2: Write `rng.py`**

```python
import subprocess

import jax
from jaxtyping import PRNGKeyArray


def make_key(seed: int) -> PRNGKeyArray:
    return jax.random.PRNGKey(seed)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"
```

- [ ] **Step 3: Write `experiments/configs/default.yaml`**

```yaml
H: 30
W: 30
T: 50
density: 0.8
group_ratio: 0.5
beta: 5.0
tau_g: 0.5
tau: 0.4
seed: 42
```

- [ ] **Step 4: Write test for Config loading**

Create `tests/test_config.py`:

```python
from pathlib import Path
from src.abm_geometry.config import Config, load_config


def test_config_defaults():
    cfg = Config()
    assert cfg.H == 30
    assert cfg.T == 50
    assert cfg.tau == 0.4


def test_load_config_from_yaml(tmp_path):
    yaml_content = "H: 10\nW: 10\nT: 5\n"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(p)
    assert cfg.H == 10
    assert cfg.T == 5
    assert cfg.tau == 0.4  # default preserved
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/abm_geometry/config.py src/abm_geometry/rng.py \
        experiments/configs/default.yaml tests/test_config.py
git commit -m "feat: add Config dataclass, YAML loader, and RNG helpers"
```

---

## Task 4 — World initialisation

**Files:**
- Create: `src/abm_geometry/schelling/state.py`
- Create: `tests/test_simulator.py` (first batch of tests)

- [ ] **Step 1: Write failing tests for `init_world`**

Create `tests/test_simulator.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from src.abm_geometry.config import Config
from src.abm_geometry.schelling.state import init_world

CFG = Config(H=10, W=10, T=5, density=0.8, group_ratio=0.5, tau=0.4)


def test_init_world_occupancy_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.soft_occupancy.shape == (10, 10, 3)


def test_init_world_tolerances_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.tolerances.shape == (10, 10)


def test_init_world_sums_to_one():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sums = jnp.sum(state.soft_occupancy, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-5)


def test_init_world_tolerances_constant():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert jnp.allclose(state.tolerances, jnp.full((10, 10), 0.4))


def test_init_world_step_zero():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert int(state.step) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_simulator.py -v
```

Expected: ImportError or ModuleNotFoundError — `init_world` not yet defined.

- [ ] **Step 3: Write `schelling/state.py`**

```python
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.types import WorldState


def init_world(key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Randomly place agents on the grid according to cfg.density and cfg.group_ratio.

    soft_occupancy is a one-hot encoding: each cell has exactly one of
    {empty, A, B} set to 1.0. tolerances are initialised uniformly to cfg.tau.
    """
    key_cells, key_groups = jax.random.split(key)

    is_occupied = jax.random.bernoulli(key_cells, cfg.density, shape=(cfg.H, cfg.W))
    is_A = jax.random.bernoulli(key_groups, cfg.group_ratio, shape=(cfg.H, cfg.W))
    is_A = is_A & is_occupied
    is_B = is_occupied & ~is_A

    soft_occ = jnp.stack(
        [
            (~is_occupied).astype(jnp.float32),
            is_A.astype(jnp.float32),
            is_B.astype(jnp.float32),
        ],
        axis=-1,
    )  # Float[H, W, 3]

    return WorldState(
        soft_occupancy=soft_occ,
        tolerances=jnp.full((cfg.H, cfg.W), cfg.tau),
        step=jnp.int32(0),
    )
```

- [ ] **Step 4: Update `schelling/__init__.py` to export `init_world`**

```python
from abm_geometry.schelling.state import init_world

__all__ = ["init_world"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_simulator.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/abm_geometry/schelling/state.py src/abm_geometry/schelling/__init__.py \
        tests/test_simulator.py
git commit -m "feat: add init_world — random grid initialisation"
```

---

## Task 5 — Neighbourhood computation

**Files:**
- Create: `src/abm_geometry/schelling/neighbours.py`
- Modify: `tests/test_simulator.py` (add neighbour tests)

- [ ] **Step 1: Write failing test for `neighbour_type_fraction`**

Append to `tests/test_simulator.py`:

```python
from src.abm_geometry.schelling.neighbours import neighbour_type_fraction


def test_neighbour_fraction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    assert frac.shape == (10, 10, 2)


def test_neighbour_fraction_sums_to_one():
    """frac_A + frac_B should sum to 1 everywhere (by definition of fraction)."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    total = frac[..., 0] + frac[..., 1]
    assert jnp.allclose(total, jnp.ones((10, 10)), atol=1e-5)


def test_neighbour_fraction_all_same_type():
    """Grid entirely filled with type A → frac_A = 1 everywhere."""
    # All type A, no empty
    occ = jnp.zeros((5, 5, 3)).at[..., 1].set(1.0)  # all A
    frac = neighbour_type_fraction(occ)
    assert jnp.allclose(frac[..., 0], jnp.ones((5, 5)), atol=1e-5)
    assert jnp.allclose(frac[..., 1], jnp.zeros((5, 5)), atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_simulator.py::test_neighbour_fraction_shape -v
```

Expected: ImportError.

- [ ] **Step 3: Write `schelling/neighbours.py`**

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def _sum_moore_neighbours(x: Float[Array, "H W"]) -> Float[Array, "H W"]:
    """Sum values across the 8 Moore neighbours (periodic / toroidal boundary)."""
    result = jnp.zeros_like(x)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            result = result + jnp.roll(x, shift=(di, dj), axis=(0, 1))
    return result
    # Note: Python loop over 8 constants — JAX unrolls this at trace time.


def neighbour_type_fraction(
    soft_occ: Float[Array, "H W 3"],
) -> Float[Array, "H W 2"]:
    """Fraction of type-A and type-B mass in each cell's Moore neighbourhood.

    Returns Float[H, W, 2] where axis-2 = (frac_A, frac_B).
    frac_A[h, w] + frac_B[h, w] = 1 by definition.
    """
    neigh_A = _sum_moore_neighbours(soft_occ[..., 1])  # Float[H, W]
    neigh_B = _sum_moore_neighbours(soft_occ[..., 2])  # Float[H, W]
    neigh_total = neigh_A + neigh_B + 1e-8  # avoid division by zero in empty regions

    return jnp.stack([neigh_A / neigh_total, neigh_B / neigh_total], axis=-1)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_simulator.py -v -k "neighbour"
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/abm_geometry/schelling/neighbours.py tests/test_simulator.py
git commit -m "feat: add neighbour_type_fraction via Moore convolution"
```

---

## Task 6 — Soft satisfaction

**Files:**
- Create: `src/abm_geometry/schelling/satisfaction.py`
- Modify: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests for `soft_satisfaction`**

Append to `tests/test_simulator.py`:

```python
from src.abm_geometry.schelling.satisfaction import soft_satisfaction


def test_satisfaction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert sat.shape == (10, 10)


def test_satisfaction_range():
    """Satisfaction values should be in (0, 1) since they are sigmoid outputs."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert jnp.all(sat >= 0.0) and jnp.all(sat <= 1.0)


def test_satisfaction_zero_tolerance():
    """With τ=0, every agent is satisfied (any non-zero similar fraction ≥ 0)."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    # Replace tolerances with 0
    state_zero_tau = state.replace(tolerances=jnp.zeros((10, 10)))
    sat = soft_satisfaction(state_zero_tau.soft_occupancy, state_zero_tau.tolerances, beta=10.0)
    # Most cells should be highly satisfied; none should be < 0.5 at high beta
    assert jnp.mean(sat) > 0.6


def test_satisfaction_high_tolerance():
    """With τ=1, no agent is ever fully satisfied."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    state_high_tau = state.replace(tolerances=jnp.ones((10, 10)))
    sat = soft_satisfaction(state_high_tau.soft_occupancy, state_high_tau.tolerances, beta=10.0)
    assert jnp.mean(sat) < 0.4
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_simulator.py -v -k "satisfaction"
```

Expected: ImportError.

- [ ] **Step 3: Write `schelling/satisfaction.py`**

```python
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from abm_geometry.schelling.neighbours import neighbour_type_fraction


def soft_satisfaction(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    beta: float,
) -> Float[Array, "H W"]:
    """Occupancy-weighted soft satisfaction at each cell.

    For a type-A agent: sat_A = σ(β * (frac_A_neighbours - τ))
    For a type-B agent: sat_B = σ(β * (frac_B_neighbours - τ))
    Returns p_A * sat_A + p_B * sat_B — the expected satisfaction over type uncertainty.
    """
    neigh_frac = neighbour_type_fraction(soft_occ)  # Float[H, W, 2]

    sat_A = jax.nn.sigmoid(beta * (neigh_frac[..., 0] - tolerances))  # Float[H, W]
    sat_B = jax.nn.sigmoid(beta * (neigh_frac[..., 1] - tolerances))  # Float[H, W]

    # Weight by occupancy probability of each type
    return soft_occ[..., 1] * sat_A + soft_occ[..., 2] * sat_B
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_simulator.py -v -k "satisfaction"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/abm_geometry/schelling/satisfaction.py tests/test_simulator.py
git commit -m "feat: add soft_satisfaction with sigmoid relaxation"
```

---

## Task 7 — Gumbel-softmax move rule

**Files:**
- Create: `src/abm_geometry/schelling/move_rules.py`
- Modify: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests for `gumbel_softmax_step`**

Append to `tests/test_simulator.py`:

```python
from src.abm_geometry.schelling.move_rules import gumbel_softmax_step


def test_move_rule_output_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    assert new_occ.shape == (10, 10, 3)


def test_move_rule_sums_to_one():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    sums = jnp.sum(new_occ, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-5)


def test_move_rule_total_agent_mass_conserved():
    """Total mass of type A + type B should be conserved across a step."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    total_before = jnp.sum(state.soft_occupancy[..., 1:])
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    total_after = jnp.sum(new_occ[..., 1:])
    assert jnp.allclose(total_before, total_after, atol=1e-4)


def test_move_rule_non_negative():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    assert jnp.all(new_occ >= -1e-6)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_simulator.py -v -k "move_rule"
```

Expected: ImportError.

- [ ] **Step 3: Write `schelling/move_rules.py`**

```python
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from abm_geometry.schelling.satisfaction import soft_satisfaction


def gumbel_softmax_step(
    soft_occ: Float[Array, "H W 3"],
    tolerances: Float[Array, "H W"],
    key: PRNGKeyArray,
    beta: float,
    tau_g: float,
) -> Float[Array, "H W 3"]:
    """One differentiable Schelling step via Gumbel-softmax.

    Each occupied cell independently decides whether to stay or move.
    Moving mass redistributes to cells weighted by their empty probability.
    This is a relaxed (soft) approximation of the discrete Schelling move rule.

    tau_g controls relaxation sharpness: smaller → harder (closer to discrete).
    """
    H, W, _ = soft_occ.shape

    sat = soft_satisfaction(soft_occ, tolerances, beta)  # Float[H, W]

    # Stay/move logits and Gumbel noise
    logits = jnp.stack(
        [beta * sat, beta * (1.0 - sat)], axis=-1
    )  # Float[H, W, 2]; axis-2 = [stay, move]
    u = jax.random.uniform(key, (H, W, 2), minval=1e-6, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))  # Gumbel(0,1) samples
    p_stay = jax.nn.softmax((logits + gumbel) / tau_g, axis=-1)[..., 0]  # Float[H, W]

    # Split occupied mass into staying and moving components
    occupied = soft_occ[..., 1:]  # Float[H, W, 2]; axis-2 = [A, B]
    staying = occupied * p_stay[..., None]  # Float[H, W, 2]
    moving = occupied * (1.0 - p_stay[..., None])  # Float[H, W, 2]

    # Redistribute moving mass to cells proportional to their emptiness
    total_moving = jnp.sum(moving, axis=(0, 1))  # Float[2]
    empty_weight = soft_occ[..., 0] / (jnp.sum(soft_occ[..., 0]) + 1e-8)  # Float[H, W]

    new_A = staying[..., 0] + total_moving[0] * empty_weight
    new_B = staying[..., 1] + total_moving[1] * empty_weight
    new_empty = jnp.clip(1.0 - new_A - new_B, 0.0, 1.0)

    new_occ = jnp.stack([new_empty, new_A, new_B], axis=-1)
    # Renormalize to absorb floating-point drift
    return new_occ / (jnp.sum(new_occ, axis=-1, keepdims=True) + 1e-8)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_simulator.py -v -k "move_rule"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/abm_geometry/schelling/move_rules.py tests/test_simulator.py
git commit -m "feat: add Gumbel-softmax move rule with mass conservation"
```

---

## Task 8 — Step and simulation loop

**Files:**
- Create: `src/abm_geometry/schelling/step.py`
- Create: `src/abm_geometry/schelling/simulate.py`
- Modify: `src/abm_geometry/schelling/__init__.py`
- Modify: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests for `simulate`**

Append to `tests/test_simulator.py`:

```python
from src.abm_geometry.schelling.simulate import simulate


def test_simulate_output_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    assert final.soft_occupancy.shape == (10, 10, 3)


def test_simulate_step_counter():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    assert int(final.step) == CFG.T


def test_simulate_mass_conservation():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    sums = jnp.sum(final.soft_occupancy, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-4)


def test_simulate_jit():
    """Running jitted simulate twice with same inputs should give identical results."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sim_jit = jax.jit(simulate, static_argnums=(2,))
    out1 = sim_jit(key, state, CFG)
    out2 = sim_jit(key, state, CFG)
    assert jnp.allclose(out1.soft_occupancy, out2.soft_occupancy)


def test_simulate_vmap():
    """vmap over keys should produce a batch of independent simulations."""
    B = 4
    keys = jax.random.split(jax.random.PRNGKey(99), B)

    def run_one(key):
        state = init_world(key, CFG)
        return simulate(key, state, CFG)

    batch = jax.jit(jax.vmap(run_one))(keys)
    assert batch.soft_occupancy.shape == (B, 10, 10, 3)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_simulator.py -v -k "simulate"
```

Expected: ImportError.

- [ ] **Step 3: Write `schelling/step.py`**

```python
from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.schelling.move_rules import gumbel_softmax_step
from abm_geometry.types import WorldState


def one_step(state: WorldState, key: PRNGKeyArray, cfg: Config) -> WorldState:
    """Apply one Gumbel-softmax Schelling step and increment the step counter."""
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, cfg.beta, cfg.tau_g
    )
    return state.replace(soft_occupancy=new_occ, step=state.step + 1)
```

- [ ] **Step 4: Write `schelling/simulate.py`**

```python
import jax
from jaxtyping import PRNGKeyArray

from abm_geometry.config import Config
from abm_geometry.schelling.step import one_step
from abm_geometry.types import WorldState


def simulate(key: PRNGKeyArray, init_state: WorldState, cfg: Config) -> WorldState:
    """Run cfg.T steps of the Schelling model, returning the final state.

    Uses lax.scan for JIT-friendly iteration — no Python-level loop at runtime.
    cfg must be passed as a static argument when jitting:
        jax.jit(simulate, static_argnums=(2,))
    """

    def body(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        return (one_step(state, subkey, cfg), key), None

    (final_state, _), _ = jax.lax.scan(
        body, (init_state, key), xs=None, length=cfg.T
    )
    return final_state
```

- [ ] **Step 5: Update `schelling/__init__.py`**

```python
from abm_geometry.schelling.simulate import simulate
from abm_geometry.schelling.state import init_world

__all__ = ["init_world", "simulate"]
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_simulator.py -v
```

Expected: all simulator tests pass (≥14 tests).

- [ ] **Step 7: Commit**

```bash
git add src/abm_geometry/schelling/step.py src/abm_geometry/schelling/simulate.py \
        src/abm_geometry/schelling/__init__.py tests/test_simulator.py
git commit -m "feat: add one_step and simulate via lax.scan — jit and vmap verified"
```

---

## Task 9 — Summary statistics

**Files:**
- Create: `src/abm_geometry/statistics/segregation.py`
- Create: `src/abm_geometry/statistics/summary.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_statistics.py`:

```python
import jax.numpy as jnp
import pytest

from src.abm_geometry.statistics.segregation import dissimilarity_index
from src.abm_geometry.statistics.summary import stats_fn
from src.abm_geometry.types import WorldState


def _make_occ(empty, A, B):
    """Helper: build a uniform (H=1, W=2) soft_occupancy from three scalars."""
    row = jnp.array([[empty, A, B], [empty, A, B]], dtype=jnp.float32)
    return row[None, ...]  # shape (1, 2, 3)


def test_dissimilarity_fully_segregated():
    """All A on left half, all B on right half → D = 1."""
    occ = jnp.zeros((1, 2, 3))
    occ = occ.at[0, 0, 1].set(1.0)  # cell (0,0) is type A
    occ = occ.at[0, 1, 2].set(1.0)  # cell (0,1) is type B
    d = dissimilarity_index(occ)
    assert jnp.allclose(d, jnp.array(1.0), atol=1e-5)


def test_dissimilarity_fully_integrated():
    """Equal A and B in every cell → D = 0."""
    occ = jnp.zeros((2, 2, 3)).at[..., 1].set(0.5).at[..., 2].set(0.5)
    d = dissimilarity_index(occ)
    assert jnp.allclose(d, jnp.array(0.0), atol=1e-5)


def test_dissimilarity_range():
    """D should always be in [0, 1]."""
    import jax

    key = jax.random.PRNGKey(0)
    occ_raw = jax.random.dirichlet(key, jnp.ones(3), shape=(5, 5))
    d = dissimilarity_index(occ_raw)
    assert 0.0 <= float(d) <= 1.0 + 1e-5


def test_stats_fn_returns_summary_stats():
    from src.abm_geometry.types import SummaryStats
    import jax

    key = jax.random.PRNGKey(0)
    from src.abm_geometry.config import Config
    from src.abm_geometry.schelling.state import init_world

    cfg = Config(H=5, W=5, T=3)
    state = init_world(key, cfg)
    stats = stats_fn(state)
    assert isinstance(stats, SummaryStats)
    assert stats.dissimilarity.shape == ()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_statistics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `statistics/segregation.py`**

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def dissimilarity_index(soft_occ: Float[Array, "H W 3"]) -> Float[Array, ""]:
    """Dissimilarity index D ∈ [0, 1].

    D = 0.5 * Σ_i |a_i/A - b_i/B|
    where a_i = type-A mass in cell i, b_i = type-B mass, A = total A, B = total B.

    D=0: perfectly integrated. D=1: perfectly segregated.
    Differentiable w.r.t. soft_occ except at the isolated zero set of |frac_A - frac_B|.
    """
    total_A = jnp.sum(soft_occ[..., 1]) + 1e-8
    total_B = jnp.sum(soft_occ[..., 2]) + 1e-8
    frac_A = soft_occ[..., 1] / total_A  # Float[H, W]
    frac_B = soft_occ[..., 2] / total_B  # Float[H, W]
    return 0.5 * jnp.sum(jnp.abs(frac_A - frac_B))
```

- [ ] **Step 4: Write `statistics/summary.py`**

```python
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.types import SummaryStats, WorldState


def stats_fn(state: WorldState) -> SummaryStats:
    """Compute all summary statistics from a simulation state."""
    return SummaryStats(
        dissimilarity=dissimilarity_index(state.soft_occupancy),
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_statistics.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/abm_geometry/statistics/segregation.py \
        src/abm_geometry/statistics/summary.py \
        tests/test_statistics.py
git commit -m "feat: add dissimilarity_index and stats_fn"
```

---

## Task 10 — Jacobian computation (gradient test)

**Files:**
- Create: `src/abm_geometry/geometry/jacobian.py`
- Create: `tests/test_gradients.py`

- [ ] **Step 1: Write failing gradient test**

Create `tests/test_gradients.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from src.abm_geometry.config import Config
from src.abm_geometry.geometry.jacobian import compute_jacobian
from src.abm_geometry.schelling.simulate import simulate
from src.abm_geometry.schelling.state import init_world
from src.abm_geometry.statistics.segregation import dissimilarity_index

CFG = Config(H=8, W=8, T=10, beta=5.0, tau_g=0.3, seed=7)

# Pre-build a fixed initial grid. Tolerances are overridden inside the
# differentiable function via .replace() + jnp.full so gradients flow.
# NEVER do Config(tau=float(traced_tau)) — float() on a traced value
# silently breaks autodiff by extracting the concrete value at trace time.
_KEY = jax.random.PRNGKey(CFG.seed)
_INIT_STATE = init_world(_KEY, CFG)
_SIM_JIT = jax.jit(simulate, static_argnums=(2,))


def _dissimilarity_from_tau(tau: jnp.ndarray) -> jnp.ndarray:
    """Differentiable pipeline: tau (scalar JAX array) → dissimilarity index (scalar)."""
    state = _INIT_STATE.replace(tolerances=jnp.full((CFG.H, CFG.W), tau))
    final = _SIM_JIT(_KEY, state, CFG)
    return dissimilarity_index(final.soft_occupancy)


def test_gradient_is_finite():
    tau = jnp.array(0.4)
    grad = jax.grad(_dissimilarity_from_tau)(tau)
    assert jnp.isfinite(grad), f"gradient is not finite: {grad}"


def test_jacfwd_matches_finite_difference():
    """jacfwd gradient should agree with central finite difference to 1e-3."""
    tau = jnp.array(0.4)
    eps = 1e-3

    # Autodiff gradient
    grad_auto = compute_jacobian(_dissimilarity_from_tau, tau)

    # Central finite difference
    f_plus = _dissimilarity_from_tau(tau + eps)
    f_minus = _dissimilarity_from_tau(tau - eps)
    grad_fd = (f_plus - f_minus) / (2 * eps)

    assert jnp.allclose(grad_auto, grad_fd, atol=1e-2), (
        f"autodiff={float(grad_auto):.4f}, finite-diff={float(grad_fd):.4f}"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_gradients.py -v
```

Expected: ImportError (`compute_jacobian` not defined yet).

- [ ] **Step 3: Write `geometry/jacobian.py`**

```python
from typing import Callable

import jax
from jax import Array
from jaxtyping import Float


def compute_jacobian(
    fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    params: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute the Jacobian of fn at params using forward-mode autodiff.

    Uses jacfwd, which is efficient when the number of parameters p is small
    relative to the number of outputs d (always true for Phases I-II where p ≤ 2).

    For Phase III+ where p grows large (per-cell tolerances), switch to jacrev.
    """
    return jax.jacfwd(fn)(params)
```

- [ ] **Step 4: Run gradient tests**

```bash
uv run pytest tests/test_gradients.py -v
```

Expected: 2 passed. If `test_jacfwd_matches_finite_difference` fails, check that `tau_g` is not too small (< 0.1 makes the step discontinuous and breaks finite-diff agreement). The default `tau_g=0.3` should work.

- [ ] **Step 5: Commit**

```bash
git add src/abm_geometry/geometry/jacobian.py tests/test_gradients.py
git commit -m "feat: add compute_jacobian (jacfwd) — finite-diff agreement verified"
```

---

## Task 11 — Invariance tests

**Files:**
- Create: `tests/test_invariances.py`

- [ ] **Step 1: Write invariance tests**

Create `tests/test_invariances.py`:

```python
import jax
import jax.numpy as jnp

from src.abm_geometry.config import Config
from src.abm_geometry.schelling.state import init_world
from src.abm_geometry.schelling.simulate import simulate
from src.abm_geometry.statistics.segregation import dissimilarity_index

CFG = Config(H=8, W=8, T=5, seed=3)


def _swap_AB(occ):
    """Swap type-A and type-B channels: axis-2 [empty, A, B] → [empty, B, A]."""
    return jnp.stack([occ[..., 0], occ[..., 2], occ[..., 1]], axis=-1)


def test_dissimilarity_invariant_under_AB_swap():
    """Swapping labels A↔B should leave the dissimilarity index unchanged."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = jax.jit(simulate, static_argnums=(2,))(key, state, CFG)

    d_original = dissimilarity_index(final.soft_occupancy)
    d_swapped = dissimilarity_index(_swap_AB(final.soft_occupancy))
    assert jnp.allclose(d_original, d_swapped, atol=1e-5), (
        f"D={float(d_original):.4f} vs swapped={float(d_swapped):.4f}"
    )


def test_tolerances_constant_across_cells():
    """In the homogeneous case, all tolerances should equal cfg.tau."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert jnp.allclose(state.tolerances, jnp.full((CFG.H, CFG.W), CFG.tau))
```

- [ ] **Step 2: Run invariance tests**

```bash
uv run pytest tests/test_invariances.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_invariances.py
git commit -m "test: add A↔B relabelling and homogeneous-tolerance invariance tests"
```

---

## Task 12 — IO, visualisation, and experiment script

**Files:**
- Create: `src/abm_geometry/io/runs.py`
- Create: `src/abm_geometry/viz/grids.py`
- Create: `experiments/exp_001_milestone0.py`

- [ ] **Step 1: Write `io/runs.py`**

```python
import dataclasses
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np


def make_run_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        sha = "nogit"
    return f"{ts}-{sha}"


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
```

- [ ] **Step 2: Write `viz/grids.py`**

```python
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
```

- [ ] **Step 3: Write `experiments/exp_001_milestone0.py`**

```python
"""Milestone 0 experiment: validate the end-to-end differentiable Schelling stack.

Runs the simulation, computes the Jacobian of the dissimilarity index w.r.t. tau,
prints a summary, saves results and a grid plot.
"""

import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from abm_geometry.config import Config, load_config
from abm_geometry.geometry.jacobian import compute_jacobian
from abm_geometry.io.runs import make_run_id, save_results
from abm_geometry.rng import make_key
from abm_geometry.schelling import init_world, simulate
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.viz.grids import plot_grid

CFG_PATH = Path(__file__).parent / "configs" / "default.yaml"


_SIM_JIT = jax.jit(simulate, static_argnums=(2,))


def dissimilarity_from_tau(
    tau: jnp.ndarray, init_state, key, cfg: Config
) -> jnp.ndarray:
    """Differentiable pipeline: tau (scalar JAX array) → dissimilarity (scalar).

    Tolerances are set via jnp.full (differentiable). Never use float(tau) to
    build a Config — that breaks the autodiff trace.
    """
    state = init_state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau))
    final = _SIM_JIT(key, state, cfg)
    return dissimilarity_index(final.soft_occupancy)


def main():
    cfg = load_config(CFG_PATH)
    key = make_key(cfg.seed)

    print(f"Running Milestone 0 — H={cfg.H}, W={cfg.W}, T={cfg.T}, tau={cfg.tau}")

    # --- Forward simulation ---
    state = init_world(key, cfg)
    sim_jit = jax.jit(simulate, static_argnums=(2,))
    final_state = sim_jit(key, state, cfg)

    d = dissimilarity_index(final_state.soft_occupancy)
    print(f"Dissimilarity index after {cfg.T} steps: {float(d):.4f}")

    # --- Jacobian (gradient of D w.r.t. tau) ---
    tau = jnp.array(cfg.tau)
    J = compute_jacobian(
        lambda t: dissimilarity_from_tau(t, state, key, cfg), tau
    )
    print(f"dD/dtau = {float(J):.4f}")
    print(f"Condition number (scalar case): N/A — only 1 param")

    # --- Save results ---
    run_id = make_run_id()
    run_dir = save_results(
        run_id, cfg,
        {"dissimilarity": d, "jacobian": J, "final_occupancy": final_state.soft_occupancy},
    )
    print(f"Results saved to: {run_dir}")

    # --- Grid plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_grid(state.soft_occupancy, title="Initial", ax=axes[0])
    plot_grid(final_state.soft_occupancy, title=f"After {cfg.T} steps", ax=axes[1])
    plot_path = run_dir / "grid.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Grid plot saved to: {plot_path}")
    plt.close()

    # --- Verify all 5 done criteria ---
    print("\n--- Milestone 0 verification ---")
    # 1. jit (already ran above without error)
    print("[1] jit: PASS")
    # 2. vmap — use plain simulate (not sim_jit) inside vmap; cfg is a closure constant
    keys = jax.random.split(key, 4)
    batch = jax.jit(jax.vmap(lambda k: simulate(k, init_world(k, cfg), cfg)))(keys)
    assert batch.soft_occupancy.shape == (4, cfg.H, cfg.W, 3)
    print("[2] vmap: PASS")
    # 3. differentiability
    grad = jax.grad(lambda t: dissimilarity_from_tau(t, state, key, cfg))(tau)
    assert jnp.isfinite(grad)
    print(f"[3] grad finite: PASS  (dD/dtau = {float(grad):.4f})")
    # 4. finite-diff agreement (checked by test suite — run pytest to verify)
    print("[4] finite-diff agreement: run pytest tests/test_gradients.py")
    # 5. no NaN/inf in Jacobian
    assert jnp.isfinite(J)
    print("[5] Jacobian finite: PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the experiment**

```bash
uv run python experiments/exp_001_milestone0.py
```

Expected output (values will vary):
```
Running Milestone 0 — H=30, W=30, T=50, tau=0.4
Dissimilarity index after 50 steps: 0.XXXX
dD/dtau = X.XXXX
Results saved to: outputs/20260516-XXXXXX-XXXXXXX
Grid plot saved to: outputs/20260516-XXXXXX-XXXXXXX/grid.png

--- Milestone 0 verification ---
[1] jit: PASS
[2] vmap: PASS
[3] grad finite: PASS  (dD/dtau = X.XXXX)
[4] finite-diff agreement: run pytest tests/test_gradients.py
[5] Jacobian finite: PASS
```

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass with no errors or failures.

- [ ] **Step 6: Commit**

```bash
git add src/abm_geometry/io/runs.py src/abm_geometry/viz/grids.py \
        experiments/exp_001_milestone0.py
git commit -m "feat: add IO helpers, grid visualisation, and Milestone 0 experiment script"
```

---

## Milestone 0 done criteria checklist

Run these after Task 12 to confirm the milestone is complete:

```bash
# All 5 criteria
uv run python experiments/exp_001_milestone0.py   # criteria 1, 2, 3, 5
uv run pytest tests/test_gradients.py -v           # criterion 4
uv run pytest tests/ -v                            # full suite
```

| # | Condition | Expected |
|---|---|---|
| 1 | `simulate` runs under `jit` | No error; second call is fast |
| 2 | `simulate` is batchable via `vmap` | `soft_occupancy` shape `(4, H, W, 3)` |
| 3 | `dissimilarity_index` is differentiable w.r.t. τ | Finite float from `jax.grad` |
| 4 | `jacfwd` agrees with finite differences | `test_jacfwd_matches_finite_difference` passes |
| 5 | Jacobian is finite | No NaN or inf in output |
