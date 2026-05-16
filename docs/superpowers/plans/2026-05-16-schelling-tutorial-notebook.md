# Schelling Tutorial Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/01_schelling_model.ipynb`, a polished linear tutorial introducing the Schelling segregation model to Python-literate PhD readers who are new to ABMs.

**Architecture:** No new source code in `src/`. The notebook calls the existing public API (`init_world`, `simulate`, `one_step`, `dissimilarity_index`, `plot_grid`). The only code changes are exporting `one_step` from the schelling package and adding Jupyter to dev dependencies. The notebook covers 7 sections: intro → setup → grid → one step → full simulation → D(t) curve → tolerance sweep.

**Tech Stack:** JAX, matplotlib, existing `abm_geometry` package, Jupyter.

---

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/abm_geometry/schelling/__init__.py` | export `one_step` |
| Modify | `pyproject.toml` | add `jupyter`, `ipykernel` to dev extras |
| Create | `notebooks/01_schelling_model.ipynb` | the tutorial notebook |
| Create | `tests/test_tutorial_api.py` | smoke-tests that all notebook API calls work |

---

### Task 1: Export `one_step` from the schelling package

**Files:**
- Modify: `src/abm_geometry/schelling/__init__.py`
- Create: `tests/test_tutorial_api.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_tutorial_api.py`:

```python
"""Smoke-tests for the public API used by the tutorial notebook."""
import jax
import jax.numpy as jnp

from abm_geometry.config import Config
from abm_geometry.rng import make_key
from abm_geometry.schelling import init_world, one_step, simulate
from abm_geometry.statistics.segregation import dissimilarity_index


def _default_cfg():
    return Config(H=10, W=10, T=5, density=0.8, tau=0.4)


def test_one_step_exported():
    """one_step must be importable directly from abm_geometry.schelling."""
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    k1, k2 = jax.random.split(key)
    next_state = one_step(state, k1, cfg)
    assert next_state.soft_occupancy.shape == (10, 10, 3)
    assert int(next_state.step) == 1


def test_simulate_returns_final_state():
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    final = simulate(key, state, cfg)
    assert final.soft_occupancy.shape == (10, 10, 3)
    assert int(final.step) == cfg.T


def test_dissimilarity_in_unit_range():
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    final = simulate(key, state, cfg)
    d = dissimilarity_index(final.soft_occupancy)
    assert 0.0 <= float(d) <= 1.0


def test_tolerance_sweep_shapes():
    """Running simulate with different τ values must all return valid states."""
    cfg = _default_cfg()
    key = make_key(0)
    state = init_world(key, cfg)
    _sim = jax.jit(simulate, static_argnums=(2,))
    for tau_val in [0.2, 0.4, 0.6]:
        s_tau = state.replace(tolerances=jnp.full((10, 10), tau_val))
        final = _sim(key, s_tau, cfg)
        assert final.soft_occupancy.shape == (10, 10, 3)
```

- [ ] **Step 1.2: Run the test to confirm it fails on `one_step` import**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
uv run pytest tests/test_tutorial_api.py -v
```

Expected: `ImportError: cannot import name 'one_step' from 'abm_geometry.schelling'`

- [ ] **Step 1.3: Add `one_step` to the schelling package exports**

Edit `src/abm_geometry/schelling/__init__.py`:

```python
from abm_geometry.schelling.simulate import simulate
from abm_geometry.schelling.state import init_world
from abm_geometry.schelling.step import one_step

__all__ = ["init_world", "one_step", "simulate"]
```

- [ ] **Step 1.4: Run the tests and confirm all pass**

```bash
uv run pytest tests/test_tutorial_api.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 1.5: Run the full test suite to check no regressions**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 1.6: Commit**

```bash
git add src/abm_geometry/schelling/__init__.py tests/test_tutorial_api.py
git commit -m "feat: export one_step from schelling package; add tutorial API smoke-tests"
```

---

### Task 2: Add Jupyter to dev dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 2.1: Add `jupyter` and `ipykernel` to dev extras in `pyproject.toml`**

The current `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pyright>=1.1",
]
```

Replace with:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pyright>=1.1",
    "jupyter>=1.0",
    "ipykernel>=6.0",
]
```

- [ ] **Step 2.2: Sync the environment**

```bash
uv sync --extra dev
```

Expected: resolves without errors, installs jupyter + ipykernel.

- [ ] **Step 2.3: Verify Jupyter is available**

```bash
uv run jupyter --version
```

Expected: prints version string (e.g. `jupyter core: 5.x.x`), no error.

- [ ] **Step 2.4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add jupyter and ipykernel to dev dependencies"
```

---

### Task 3: Create the tutorial notebook

**Files:**
- Create: `notebooks/01_schelling_model.ipynb`

The notebook has 14 cells (alternating markdown + code). Each cell's full source is given below. Create the file as valid Jupyter notebook JSON.

> **Colour note:** `viz/grids.py` renders type-A cells with the red channel, type-B with the blue channel, and empty cells as white. The legend uses `#CC3333` (red) for A, `#3333CC` (blue) for B, and `#F0F0F0` (grey) for empty.

- [ ] **Step 3.1: Create the `notebooks/` directory and write the notebook file**

Create `notebooks/01_schelling_model.ipynb` with the following content (valid `.ipynb` JSON):

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The Schelling Segregation Model\n",
    "\n",
    "> **How to run:** from the repo root, `uv run jupyter lab` then open this file.\n",
    "\n",
    "A hands-on introduction to agent-based modelling: how mild individual preferences produce strong collective segregation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Introduction\n",
    "\n",
    "In 1971, the economist Thomas Schelling described a striking result: even if every individual is willing to live in a mixed neighbourhood — they just prefer *not* to be a tiny minority — the collective outcome is near-total segregation. No central planner, no overt discrimination; the pattern emerges purely from local movement decisions.\n",
    "\n",
    "The model is simple. Agents of two groups (A and B) are placed on a grid. At each time step, every agent checks what fraction of its eight immediate neighbours share its type. If that fraction is below a **tolerance threshold τ**, the agent is *unsatisfied* and moves to a random empty cell. Satisfied agents stay put. After enough steps, clusters form and segregation stabilises.\n",
    "\n",
    "This notebook walks through the model step by step: we will build intuition for the rules, watch segregation emerge in real time, and explore how τ controls the final outcome."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Setup\n",
    "\n",
    "We load the default configuration and set up a reproducible random key."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "sys.path.insert(0, str(Path(\"..\").resolve() / \"src\"))\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches\n",
    "import numpy as np\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "\n",
    "from abm_geometry.config import load_config\n",
    "from abm_geometry.rng import make_key\n",
    "from abm_geometry.schelling import init_world, one_step, simulate\n",
    "from abm_geometry.statistics.segregation import dissimilarity_index\n",
    "from abm_geometry.viz.grids import plot_grid\n",
    "\n",
    "plt.style.use(\"seaborn-v0_8-whitegrid\")\n",
    "%matplotlib inline\n",
    "\n",
    "cfg = load_config(\"../experiments/configs/default.yaml\")\n",
    "key = make_key(cfg.seed)\n",
    "print(f\"Grid: {cfg.H}\\u00d7{cfg.W}  |  Steps T: {cfg.T}  |  Tolerance \\u03c4: {cfg.tau}  |  Density: {cfg.density}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. The Grid\n",
    "\n",
    "The world is a **30×30 grid** where each cell is one of:\n",
    "- **Empty** (white) — no agent present\n",
    "- **Group A** (red) — an agent of type A\n",
    "- **Group B** (blue) — an agent of type B\n",
    "\n",
    "Initially, agents are placed at random. With `density = 0.8`, 80% of cells are occupied. Half are group A, half group B. There is no structure yet — this is the baseline before any movement."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "state = init_world(key, cfg)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(5, 5))\n",
    "plot_grid(state.soft_occupancy, title=f\"Initial random placement  (D = {float(dissimilarity_index(state.soft_occupancy)):.3f})\", ax=ax)\n",
    "legend = [\n",
    "    mpatches.Patch(color=\"#CC3333\", label=\"Group A\"),\n",
    "    mpatches.Patch(color=\"#3333CC\", label=\"Group B\"),\n",
    "    mpatches.Patch(facecolor=\"#F0F0F0\", edgecolor=\"lightgrey\", label=\"Empty\"),\n",
    "]\n",
    "ax.legend(handles=legend, loc=\"lower right\", fontsize=9, framealpha=0.85)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. One Step: Satisfaction and Movement\n",
    "\n",
    "At each time step the model does the following for every agent:\n",
    "\n",
    "1. Count the agent's **8 Moore-neighbourhood** cells (the cells directly and diagonally adjacent).\n",
    "2. Compute the fraction of those occupied cells that share the agent's type.\n",
    "3. If that fraction is **≥ τ**, the agent is *satisfied* — it stays.\n",
    "4. If it is **< τ**, the agent is *unsatisfied* — it relocates to a randomly chosen empty cell.\n",
    "\n",
    "Below we apply a single step and highlight the cells that changed (red border)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "k1, _ = jax.random.split(key)\n",
    "after_one = one_step(state, k1, cfg)\n",
    "\n",
    "changed = np.array(jnp.abs(after_one.soft_occupancy - state.soft_occupancy).sum(axis=-1) > 0.1)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "plot_grid(state.soft_occupancy, title=\"Before (step 0)\", ax=axes[0])\n",
    "plot_grid(after_one.soft_occupancy, title=\"After (step 1)\", ax=axes[1])\n",
    "\n",
    "for r, c in zip(*np.where(changed)):\n",
    "    axes[1].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,\n",
    "                                     fill=False, edgecolor=\"crimson\", linewidth=1.2))\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(f\"Cells that moved: {int(changed.sum())}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Full Simulation: Segregation Emerges\n",
    "\n",
    "Let's run all T = 50 steps and compare the initial and final states.\n",
    "\n",
    "We measure segregation with the **dissimilarity index** D:\n",
    "$$D = \\frac{1}{2} \\sum_i \\left|\\frac{a_i}{A} - \\frac{b_i}{B}\\right|$$\n",
    "where $a_i$ and $b_i$ are the type-A and type-B mass in cell $i$, and $A$, $B$ are the totals. D = 0 means perfect integration; D = 1 means complete segregation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "_sim = jax.jit(simulate, static_argnums=(2,))\n",
    "final_state = _sim(key, state, cfg)\n",
    "\n",
    "d_init  = float(dissimilarity_index(state.soft_occupancy))\n",
    "d_final = float(dissimilarity_index(final_state.soft_occupancy))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "plot_grid(state.soft_occupancy,       title=f\"Initial   D = {d_init:.3f}\",         ax=axes[0])\n",
    "plot_grid(final_state.soft_occupancy, title=f\"After {cfg.T} steps   D = {d_final:.3f}\", ax=axes[1])\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(f\"Dissimilarity: {d_init:.3f} \\u2192 {d_final:.3f}  (+{d_final - d_init:.3f})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Segregation Over Time\n",
    "\n",
    "How quickly does D rise, and when does it stabilise? We track the dissimilarity index at every step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "d_curve = [float(dissimilarity_index(state.soft_occupancy))]\n",
    "s = state\n",
    "step_keys = jax.random.split(key, cfg.T)\n",
    "for k in step_keys:\n",
    "    s = one_step(s, k, cfg)\n",
    "    d_curve.append(float(dissimilarity_index(s.soft_occupancy)))\n",
    "\n",
    "# First t >= 10 where |D(t) - D(t-5)| < 0.01; fallback to step 35\n",
    "stabilise_t = 35\n",
    "for t in range(10, cfg.T):\n",
    "    if abs(d_curve[t] - d_curve[t - 5]) < 0.01:\n",
    "        stabilise_t = t\n",
    "        break\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "ax.plot(range(cfg.T + 1), d_curve, color=\"#2b6cb0\", linewidth=2)\n",
    "ax.axvline(1, color=\"grey\", linestyle=\"--\", alpha=0.6)\n",
    "ax.text(1.5, d_curve[1] - 0.04, \"agents start\\nmoving\", fontsize=9, color=\"grey\", va=\"top\")\n",
    "ax.axvline(stabilise_t, color=\"grey\", linestyle=\"--\", alpha=0.6)\n",
    "ax.text(stabilise_t + 0.5, d_curve[stabilise_t] + 0.01,\n",
    "        \"segregation\\nstabilises\", fontsize=9, color=\"grey\")\n",
    "ax.set_xlabel(\"Step\")\n",
    "ax.set_ylabel(\"Dissimilarity index $D$\")\n",
    "ax.set_ylim(0, 1)\n",
    "ax.set_title(\"Segregation grows rapidly then stabilises\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. The Role of Tolerance\n",
    "\n",
    "The key parameter is τ. A lower τ means agents are **more demanding**: they need a higher fraction of like-minded neighbours to be satisfied. Below we run the same simulation from the same starting state with τ ∈ {0.2, 0.4, 0.6}.\n",
    "\n",
    "Notice how even a mild tolerance (τ = 0.4 means you are happy as long as 40% of neighbours share your type) still produces clear segregation — the core Schelling result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "taus = [0.2, 0.4, 0.6]\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "for ax, tau_val in zip(axes, taus):\n",
    "    s_tau = state.replace(tolerances=jnp.full((cfg.H, cfg.W), tau_val))\n",
    "    final_tau = _sim(key, s_tau, cfg)\n",
    "    d_tau = float(dissimilarity_index(final_tau.soft_occupancy))\n",
    "    plot_grid(final_tau.soft_occupancy, title=f\"\\u03c4 = {tau_val}\\nD = {d_tau:.3f}\", ax=ax)\n",
    "\n",
    "plt.suptitle(f\"Final state after {cfg.T} steps — lower \\u03c4 means more segregation\",\n",
    "             y=1.02, fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ]
}
```

- [ ] **Step 3.2: Commit the notebook**

```bash
git add notebooks/01_schelling_model.ipynb
git commit -m "feat: add Schelling tutorial notebook (sections 1-7)"
```

---

### Task 4: Execute the notebook end-to-end and verify output

- [ ] **Step 4.1: Execute the notebook non-interactively**

```bash
cd /Users/pietrobicocchi/Documents/dev/phd/projects/research-ABM-diff-geometry
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=120 \
    --output notebooks/01_schelling_model_executed.ipynb \
    notebooks/01_schelling_model.ipynb
```

Expected: exits 0, writes `01_schelling_model_executed.ipynb`.

- [ ] **Step 4.2: Confirm all cells executed without errors**

```bash
uv run python -c "
import json, sys
nb = json.load(open('notebooks/01_schelling_model_executed.ipynb'))
errors = [
    (i, o)
    for i, cell in enumerate(nb['cells'])
    if cell['cell_type'] == 'code'
    for o in cell.get('outputs', [])
    if o.get('output_type') == 'error'
]
if errors:
    for i, e in errors:
        print(f'Cell {i}: {e[\"ename\"]}: {e[\"evalue\"]}')
    sys.exit(1)
else:
    print('All cells executed cleanly.')
"
```

Expected: `All cells executed cleanly.`

- [ ] **Step 4.3: Remove the executed copy (keep only the clean source notebook)**

```bash
rm notebooks/01_schelling_model_executed.ipynb
```

- [ ] **Step 4.4: Final commit**

```bash
git add notebooks/ pyproject.toml uv.lock
git commit -m "chore: verify tutorial notebook executes end-to-end"
```

---

## Self-review checklist (completed inline)

**Spec coverage:**
- [x] Section 1 Introduction → Cell 2 markdown
- [x] Section 2 Setup → Task 2 (deps) + Cell 4 code
- [x] Section 3 Grid → Cell 5–6
- [x] Section 4 One step → Cell 7–8
- [x] Section 5 Full simulation → Cell 9–10
- [x] Section 6 D(t) curve → Cell 11–12; stabilisation heuristic matches spec exactly
- [x] Section 7 Tolerance sweep → Cell 13–14; τ ∈ {0.2, 0.4, 0.6} as specified
- [x] `one_step` export + test → Task 1
- [x] Jupyter deps → Task 2
- [x] Colour scheme: A=red (`#CC3333`), B=blue (`#3333CC`), empty=white — matches actual `plot_grid` RGB logic

**Placeholder scan:** No TBDs, no "similar to Task N", all code complete.

**Type consistency:** `one_step(state, key, cfg)` signature used consistently in test (Task 1) and notebook cells (Task 3). `simulate(key, state, cfg)` consistent throughout. `dissimilarity_index(soft_occupancy)` consistent.
