# Phase 2 — Make the quickstart and simulators work out of the box

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [docs/getting-started/quickstart.md:22-100](../../../../docs/getting-started/quickstart.md) — the arena built from 2000 uniform points (line 38), and the claim at lines 97-98.
- [docs/user-guide/workflows.md:424-430](../../../../docs/user-guide/workflows.md) — the existing `from_polygon` workaround to copy.
- [src/neurospatial/simulation/examples.py:22-31](../../../../src/neurospatial/simulation/examples.py) — `_square_arena`, which uses `from_samples` and declares no walls.
- [src/neurospatial/simulation/mazes/t_maze.py:150-160,240-256](../../../../src/neurospatial/simulation/mazes/t_maze.py) — the point regions.
- [src/neurospatial/layout/_plotting.py:181](../../../../src/neurospatial/layout/_plotting.py) — `NotImplementedError` for non-2-D grids.
- [src/neurospatial/environment/_visualization.py:273](../../../../src/neurospatial/environment/_visualization.py) — `plot_1d`'s refusal, whose Fix points back to `env.plot()`.
- [src/neurospatial/io/nwb/_behavior.py:148-150](../../../../src/neurospatial/io/nwb/_behavior.py) — the doctest that writes `position.nwb` into the current directory.
- `.claude/rules/executable-docs.md` — the docs-test markers.

## Tasks

- **Quickstart arena.**
  - Replace quickstart.md:38 with
    `Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")`,
    adding the `shapely.geometry.box` import.
  - Re-run the block and update the printed-value comments: bin count and
    peak rate.
  - Add a fence asserting the map covers the arena:
    `assert np.isfinite(result.firing_rate).mean() > 0.8`. Keep the assert
    visible; it shows the reader how to check coverage.
  - Rewrite the sentence at lines 97-98 so it matches the new figure.
- **Simulator arena walls.** `_square_arena` builds from
  `Environment.from_polygon(box(0, 0, arena_size, arena_size), bin_size=…, units="cm")`,
  so walls are declared and border scores are available. After the change,
  check that `open_field_session(...)` place cells label as `place` in
  `spatial_cell_type_significance(...).summary_table()["cell_type"]`.
  `border_cell_session` and `grid_cell_session` use the same helper; check
  their `validate_simulation` still passes.
- **T-maze regions.** Give each T-maze region (`start`, `junction`, arm
  ends) a radius equal to half the corridor width, so `detect_trials` accepts
  them. If `Region` takes no radius, use a disk polygon
  (`Point(...).buffer(corridor_width / 2)`). Update
  `docs/user-guide/simulated-mazes.md` if it shows the regions.
- **1-D `env.plot`.**
  - Implement the 1-D regular-grid branch in `layout/_plotting.py` (`:181`):
    draw bin edges as ticks along a horizontal line, with the x-axis in
    `env.units`, and return the `Axes`.
  - Fix the `plot_1d` refusal at `environment/_visualization.py:273` so its
    Fix line no longer points to the call that just failed.
  - Remove the `NotImplementedError` path.
- **README.**
  - README.md:271 and :579 say `max_gap=0.5`; change them to `1.0`.
  - Update the package map (README.md around line 637): remove
    `composite.py`, and add `arena`, `binning`, `landmarks` and `testing`.
- **Workflows dead snippets.**
  - Rewrite Workflow 3 (workflows.md around lines 840-890, which uses the
    non-existent `interpolate_position` and undefined `position_data`/`plt`)
    and "Handling Edge Cases" (around line 1754) with
    `compute_spatial_rate`/`env.occupancy`.
  - Add `<!-- docs-test: run -->` (or `run setup=<name>`) markers so they
    execute.
- **Doctest writes.** Make the `_behavior.py:148` example write to a
  `tempfile.TemporaryDirectory()` path. Grep for other examples that write
  into the current directory: `grep -rn '">>> .*\.nwb"\|to_file("' src/neurospatial | grep ">>>"`.
- **Docs.** CHANGELOG "Unreleased" → "Fixed": quickstart figure, simulator
  walls/regions, 1-D plotting, README defaults.

## Deliberately not in this phase

- `seed` → `rng` in simulators belongs to phase 3a.
- Lap direction on 1-D tracks belongs to phase 4.
- Start Here blocks and glossary belong to phase 7.
- Plot styling of `env.plot` (hidden ticks, overwritten titles) belongs to
  phase 5.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest tests/docs/test_executable_docs.py -n 0 -k quickstart` | Quickstart runs, and the finite-fraction assert passes. |
| `uv run pytest tests/docs/test_executable_docs.py -n 0 -k workflows` | The rewritten workflow fences run. |
| `test_open_field_session_labels_place_cells` | Over half of the simulated place cells get `cell_type == "place"`, and none are `"undetermined"` because of a missing border score. |
| `test_tmaze_session_regions_work_with_detect_trials` | `detect_trials(env, times, positions, regions=[...])` on `tmaze_alternation_session` returns about `n_trials` trials. |
| `test_plot_1d_regular_grid` | `linear_track_session().env.plot()` returns an `Axes` with an x-label in cm. |
| `uv run pytest --doctest-modules src/neurospatial/io -n 0` then `git status --short` | No file left behind in the repo root. |

## Fixtures

Simulators only. Mark any test slower than 5 s `@pytest.mark.slow`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
