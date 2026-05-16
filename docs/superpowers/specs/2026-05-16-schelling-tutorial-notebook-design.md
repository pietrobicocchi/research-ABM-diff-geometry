# Design: Schelling Model Tutorial Notebook

**Date:** 2026-05-16  
**Status:** Approved  
**Scope:** Single Jupyter notebook introducing the Schelling segregation model to PhD-level readers new to ABMs. No JAX internals, no differentiability — pure model narrative and visualisations.

---

## Audience

PhD students who know Python well but are new to agent-based models and to this codebase. Explanations assume familiarity with NumPy-style arrays and scientific Python; they do not assume knowledge of JAX, ABMs, or segregation models.

---

## File layout

```
notebooks/
└── 01_schelling_model.ipynb   ← new

pyproject.toml                 ← add jupyter, ipykernel to [project.optional-dependencies] dev
src/abm_geometry/schelling/__init__.py  ← export one_step alongside init_world, simulate
```

No new source code in `src/`. The notebook calls only the existing public API.

---

## Notebook structure

The notebook lives at `notebooks/01_schelling_model.ipynb` and tells a single linear story in 7 sections. Each section is one markdown cell (explanation) followed by one or two short code cells (≤10 lines each).

### Section 1 — Introduction (markdown only)

Three paragraphs:
1. The original 1971 Schelling paper: mild individual preferences produce strong collective segregation — a classic emergence result.
2. The key model rule in plain English: an agent moves if fewer than τ fraction of its neighbours share its type.
3. What this notebook covers: building intuition for the model by running it and watching segregation emerge.

No code in this section.

### Section 2 — Setup

One code cell: imports and config. Uses `load_config` with the default YAML. Prints the key parameters (H, W, T, τ, density) so the reader sees what they're working with.

```python
from abm_geometry.config import load_config
from abm_geometry.rng import make_key
from abm_geometry.schelling import init_world, one_step, simulate
from abm_geometry.statistics.segregation import dissimilarity_index
from abm_geometry.viz.grids import plot_grid
import jax, jax.numpy as jnp, matplotlib.pyplot as plt

cfg = load_config("../experiments/configs/default.yaml")
key = make_key(cfg.seed)
print(cfg)
```

### Section 3 — The grid

Markdown explains: two groups (A = blue, B = orange), empty cells (grey), soft occupancy representation.

Code: `init_world` → `plot_grid`. Single figure, no axes, clean colour bar legend manually added as a patch legend (group A / group B / empty).

### Section 4 — One step

Markdown explains satisfaction and movement: an agent counts neighbours of the same type; if the fraction is below τ it is "unsatisfied" and may move to a random empty cell.

Two-panel figure: before and after one `one_step` call. Cells that changed highlighted via a thin border overlay (computed as `abs(after - before).sum(axis=-1) > 0.1`).

### Section 5 — Full simulation

Markdown: "Let's fast-forward T=50 steps."

Code: `simulate` → two-panel figure (initial / final). Below the figure, print the dissimilarity index before and after.

### Section 6 — Dissimilarity over time

Markdown explains D: the dissimilarity index measures segregation (0 = fully mixed, 1 = fully segregated). Formula in LaTeX inline.

Code: Python loop over `one_step` for T steps, collecting D at each step. Plot D(t) as a line, annotated with two text labels: "agents start moving" (step 1) and "segregation stabilises" (first step t where `|D(t) - D(t-5)| < 0.01`, fallback to step 35). Y-axis 0–1, x-axis 0–T.

### Section 7 — Tolerance sweep

Markdown: "What happens if agents are more or less tolerant?"

Three-panel figure (1×3), τ ∈ {0.2, 0.4, 0.6}, same seed, same T. Each panel shows the final state grid. Subtitle of each panel shows the τ value and the final D. Clear visual: low tolerance → high D (segregated); high tolerance → low D (mixed).

---

## Visual style

| Element | Choice |
|---|---|
| Group A colour | `#4C72B0` (seaborn blue) |
| Group B colour | `#DD8452` (seaborn orange) |
| Empty colour | `#F0F0F0` (light grey) |
| Figure style | `seaborn-v0_8-whitegrid`, axes off for grid plots |
| Panel size | 5×5 inches per grid panel |
| Font | Default matplotlib, titles in plain English |
| LaTeX | Only for the D formula in Section 6 |

---

## Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies] dev`:
- `jupyter>=1.0`
- `ipykernel>=6.0`

Install: `uv sync --extra dev`  
Run: `uv run jupyter lab` (note at top of notebook)

---

## What is NOT in scope

- JAX autodiff, Jacobians, FIM — deferred to a future notebook
- ipywidgets — deferred unless this notebook proves out well
- Heterogeneous tolerances (Phase II) — not covered here
- Any new source code in `src/` beyond the `one_step` export
