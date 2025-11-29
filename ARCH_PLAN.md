# Code Organization & Architecture Plan

**Goal**: Clarify and stabilize the `neurospatial` architecture and public API while keeping all existing import paths working and all tests passing.

**Constraints**

- Do not break existing imports (especially `from neurospatial import ...` and current submodules).
- Prefer additive facades + shims over disruptive moves.
- Back every structural change with tests (`tests/test_import_paths.py` plus existing suites).

---

## Milestone 1: Make Architecture Explicit (Docs-First)

### Task 1.1: Add Architecture Overview Doc

**File**: `docs/user-guide/architecture.md` (new)

- Describe major layers: `environment`, `layout`, `regions`, `spatial`, `distance`, `field_ops`, `metrics`, `behavior(al)`, `segmentation`, `simulation`, `animation`, `nwb`, `io`, `visualization`, `annotation`.
- Sketch allowed dependency directions (e.g., `layout` → `environment` → `metrics/behavior`, but not the reverse).
- Mark which modules are “core API” vs “advanced/integration”.

**Success**: `uv run mkdocs build` passes; new page answers “where does this concept live?” for new contributors.

### Task 1.2: Wire Architecture Doc into MkDocs

**File**: `mkdocs.yml`

- Under `User Guide`, add:
  - `Architecture: user-guide/architecture.md`
- Optionally add one-line links from:
  - `docs/user-guide/environments.md`
  - `docs/index.md`

**Success**: Page appears under “User Guide”; nav has no broken links.

### Task 1.3: Add Module Map to `CLAUDE.md`

**File**: `CLAUDE.md`

- Add “Module Map” with three buckets:
  - **Core**: `environment`, `layout`, `regions`, `spatial`, `distance`, `field_ops`, `transforms`, `kernels`, `primitives`.
  - **Analysis**: `metrics`, `behavioral`, `segmentation`, `spike_field`, `reward`.
  - **I/O & Integration**: `io`, `nwb`, `animation`, `annotation`, `visualization`, `simulation`.
- Note which are preferred entry points (e.g., `Environment` from top-level or `environment`, behavior from `behavior` once added).

**Success**: Claude tasks can quickly pick appropriate “neighborhoods”; content is consistent with the architecture doc.

---

## Milestone 2: Central Public-API Facade

### Task 2.1: Add `neurospatial.api`

**File**: `src/neurospatial/api.py` (new)

- Re-export a curated set of “blessed” symbols, grouped by category:
  - Core classes: `Environment`, `EnvironmentNotFittedError`, `Region`, `Regions`, `CompositeEnvironment`.
  - Spatial ops: `map_points_to_bins`, `distance_field`, `neighbors_within`, `resample_field`.
  - Field ops & kernels: `normalize_field`, `combine_fields`, `clamp`, `compute_diffusion_kernels`, `apply_kernel`, `convolve`, `neighbor_reduce`.
  - Neuroscience metrics: core place/grid/border/population metrics.
  - Behavioral helpers: `path_progress`, `distance_to_region`, `time_to_goal`, etc. (wired to `behavior` / `behavioral`).
  - I/O: `to_file`, `from_file`, `to_dict`, `from_dict`.
- Define `__all__` as the canonical curated surface.
- Docstring: explain `neurospatial.api` as the “one obvious place” for main building blocks.

**Success**: `from neurospatial.api import Environment, compute_place_field` works and covers most common workflows.

### Task 2.2: Delegate `neurospatial.__init__` to `api`

**File**: `src/neurospatial/__init__.py`

- For any symbol that exists in `neurospatial.api.__all__`, import it from `neurospatial.api`.
- Keep legacy/advanced exports imported from their original modules.
- Preserve the existing `__all__` list (no removals), but let `api.py` be the source of truth for the core subset.
- Update top-level docstring with an “Import Patterns” section:
  - Recommend `from neurospatial import ...` or `from neurospatial.api import ...`.

**Success**: All existing tests’ imports still work; `neurospatial.api` and top-level exports are consistent.

### Task 2.3: Strengthen Import-Path Tests

**File**: `tests/test_import_paths.py`

- Add assertions that key symbols are identical across modules, e.g.:
  - `Environment`, `compute_place_field`, `map_points_to_bins`, one or two metrics.
- Use identity comparisons (`is`) to ensure no divergence.

**Success**: Tests fail loudly if `api.py` and `__init__.py` drift apart.

---

## Milestone 3: Behavioral / Trajectory Namespace

### Task 3.1: Create `neurospatial.behavior`

**Files**: `src/neurospatial/behavior/__init__.py`, `src/neurospatial/behavior/core.py` (new)

- Move implementations from `src/neurospatial/behavioral.py` into `behavior/core.py`:
  - `trials_to_region_arrays`, `path_progress`, `distance_to_region`, `time_to_goal`,
    `goal_pair_direction_labels`, `graph_turn_sequence`, `heading_direction_labels`,
    `compute_trajectory_curvature`, etc.
- In `behavior/__init__.py`, import and re-export those symbols with a docstring:
  - “Behavioral and trajectory analysis utilities.”

**Success**: `from neurospatial.behavior import path_progress, distance_to_region` works.

### Task 3.2: Turn `behavioral.py` into a Shim

**File**: `src/neurospatial/behavioral.py`

- Replace implementations with imports from `neurospatial.behavior.core`, e.g.:

  ```python
  """Compatibility shim; prefer neurospatial.behavior."""
  from neurospatial.behavior.core import (
      compute_trajectory_curvature,
      cost_to_goal,
      distance_to_region,
      goal_pair_direction_labels,
      graph_turn_sequence,
      heading_direction_labels,
      path_progress,
      time_to_goal,
      trials_to_region_arrays,
  )

  __all__ = [
      "compute_trajectory_curvature",
      "cost_to_goal",
      "distance_to_region",
      "goal_pair_direction_labels",
      "graph_turn_sequence",
      "heading_direction_labels",
      "path_progress",
      "time_to_goal",
      "trials_to_region_arrays",
  ]
  ```

- Keep docstrings where possible; mention `neurospatial.behavior` as the new home.

**Success**: `from neurospatial.behavioral import path_progress` keeps working; `tests/test_behavioral.py` remains green.

### Task 3.3: Move Trajectory Metrics Behind Behavior

**Files**: `src/neurospatial/behavior/metrics.py` (new), `src/neurospatial/metrics/trajectory.py`

- Move implementations from `metrics/trajectory.py` into `behavior/metrics.py`:
  - `compute_home_range`, `compute_step_lengths`, `compute_turn_angles`, `mean_square_displacement`, etc.
- In `behavior/metrics.py`, export these and label them as trajectory metrics.
- Turn `metrics/trajectory.py` into a shim that re-exports from `behavior.metrics`, similar to `behavioral.py`.

**Success**: Old imports (`from neurospatial.metrics import compute_home_range`) and new ones (`from neurospatial.behavior.metrics import compute_home_range`) both work.

### Task 3.4: Document Behavior Namespace

**Files**: `docs/user-guide/behavior.md` (new), `mkdocs.yml`

- Add a short “Behavior & Trajectory Analysis” page:
  - Show how `segmentation`, `behavior`, and `behavior.metrics` fit together.
  - Provide recommended imports from `neurospatial.behavior`.
- Add page to `User Guide` nav.

**Success**: Docs build; example snippets run if copied into a notebook.

---

## Milestone 4: Fields & Kernels Facade

### Task 4.1: Add `neurospatial.fields`

**File**: `src/neurospatial/fields/__init__.py` (new)

- Re-export field-related utilities:
  - From `field_ops.py`: `normalize_field`, `combine_fields`, `clamp`.
  - From `kernels.py`: `compute_diffusion_kernels`, `apply_kernel`.
  - From `primitives.py`: `convolve`, `neighbor_reduce`.
- Define `__all__` and a docstring: “Field-level signal processing and kernels.”

**Success**: `from neurospatial.fields import normalize_field, compute_diffusion_kernels` works.

### Task 4.2: Integrate Fields Facade into API & Docs

**Files**: `src/neurospatial/api.py`, `docs/user-guide/spatial-analysis.md`

- In `api.py`, import these functions from `neurospatial.fields` instead of their original modules.
- In `spatial-analysis.md`, add a subsection “Field Operations & Kernels” that demonstrates imports from `neurospatial.fields`.

**Success**: API tests still pass; docs clearly point users to the new facade.

---

## Milestone 5: Lighten `environment.core` Introspection (Optional)

### Task 5.1: Extract Representation Helpers

**Files**: `src/neurospatial/environment/introspection.py` (new), `src/neurospatial/environment/core.py`

- Move pure formatting logic from `Environment.__repr__` and `_repr_html_` into helpers:
  - `format_environment_repr(env: Environment) -> str`
  - `format_environment_html(env: Environment) -> str`
- Have the methods call these helpers.

**Success**: `repr(env)` and `_repr_html_()` outputs remain unchanged for a simple test environment.

### Task 5.2: Add Introspection Tests

**File**: `tests/environment/test_introspection.py` (new)

- Create a small `Environment` and assert:
  - `repr(env)` mentions name, dimensions, layout type.
  - `_repr_html_()` contains basic labels like “Layout Type” and “Dimensions”.

**Success**: Tests catch accidental changes to representations.

---

## Milestone 6: Repo-Root Hygiene (Optional, Non-Code)

### Task 6.1: Move Planning / Scratch Docs Under `dev/`

**Candidates**: `PLAN.md`, `PLAN2.md`, `LLM_PLAN.md`, `DECODING_PLAN.md`, `TRANSITION_PLAN.md`, `TASKS.md`, `SCRATCHPAD.md`, `TODO.md` (adjust as needed).

- Create `dev/` (or `notes/`) at repo root.
- Move planning/scratch files into `dev/`.
- Add `dev/README.md` explaining these are internal and how to use them.
- Update any doc links that reference these files directly.

**Success**: Root looks like a typical library (`src`, `tests`, `docs`, `examples`, `benchmarks`, `scripts`, `pyproject.toml`, `README.md`); only paths to these docs change.

---

## Recommended Implementation Order

1. Milestone 2 – `neurospatial.api` + import-path tests.
2. Milestone 1 – Architecture doc + module map (docs-only).
3. Milestone 3 – `behavior` namespace + shims.
4. Milestone 4 – `fields` facade.
5. Milestone 5 – Environment introspection extraction (optional).
6. Milestone 6 – Root hygiene (optional).

---

## Quick Commands

```bash
# Run all tests
uv run pytest tests -v

# Focus on import paths
uv run pytest tests/test_import_paths.py -v

# Build docs
uv run mkdocs build
```

---

## Verification Checklist

- [ ] `uv run pytest tests/` passes with all new facades and shims.
- [ ] `uv run pytest tests/test_import_paths.py` passes, including new checks.
- [ ] `uv run mkdocs build` passes; new docs pages appear in nav.
- [ ] Old imports (`neurospatial.behavioral`, `metrics.trajectory`, top-level imports) still work.
- [ ] New facades (`neurospatial.api`, `neurospatial.behavior`, `neurospatial.fields`) are documented and used in at least one example.
