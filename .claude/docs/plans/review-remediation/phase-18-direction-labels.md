# Phase 18 — Lap/run → direction-labels bridge

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Closes the linear-track journey dead-end (DESIGN-REVIEW Med; journey "Linear/W-track directional place fields"). `compute_directional_place_fields` needs a per-timepoint direction-label array, but only `goal_pair_direction_labels` produces one — and only from a `Trial`, so a user reaching for `detect_laps`/`detect_runs_between_regions` dead-ends. This phase adds the bridge and a first-class inbound/outbound labeler.

**Inputs to read first:**

- `behavior/segmentation.py` — `detect_laps`, `detect_runs_between_regions`, `segment_trials`, and `goal_pair_direction_labels` (the existing bridge that only accepts `Trial`). Read their return types so the new helpers produce the same per-timepoint label array (with the `"other"` sentinel).
- `encoding/` — `compute_directional_place_fields`: read the exact `direction_labels` argument shape/dtype it consumes, so the new helpers' output is drop-in.

## Tasks

- Add to `behavior/segmentation.py` (follow the module's argument-order convention — `position_bins, times, env, *, ...`):
  - `laps_to_direction_labels(laps, times, *, ...) -> NDArray` and `runs_to_direction_labels(runs, times, *, ...) -> NDArray` — convert the lap/run objects (whatever `detect_laps`/`detect_runs_between_regions` return) into the per-timepoint object array `compute_directional_place_fields` expects, with `"other"` for unlabeled samples. Reuse the labeling logic already in `goal_pair_direction_labels` (extract a shared private helper rather than duplicating).
  - `running_direction_labels(position_bins, times, env, *, start_region, end_regions, ...) -> NDArray` returning labels drawn from `{"inbound", "outbound", "other"}` for linear tracks — a first-class named primitive so inbound/outbound no longer requires inventing end-region names.
- Cross-link: add a "See Also" / one-line note in `detect_laps` and `detect_runs_between_regions` docstrings pointing to the new label functions (so the lap-centric path no longer dead-ends).
- Export the new functions from `neurospatial.behavior` `__all__`.
- Docstrings with a runnable Example (registered in phase 23): `detect_laps(...)` → `laps_to_direction_labels(...)` → `compute_directional_place_fields(...)`. CHANGELOG entry.

## Deliberately not in this phase

- The `compute_directional_place_fields` API itself — unchanged; this phase only feeds it.
- VTE / decision-region helpers — unrelated.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_laps_to_direction_labels_shape` | Output length equals `len(times)`; unlabeled samples are `"other"`; labeled samples match the lap directions on a fixture. |
| `test_running_direction_labels_inbound_outbound` | A synthesized out-and-back linear-track trajectory yields contiguous `"outbound"` then `"inbound"` runs with `"other"` between. |
| `test_labels_feed_directional_place_fields` | The produced labels are accepted by `compute_directional_place_fields` without reshaping (integration). |

## Fixtures

Synthesize an out-and-back trajectory on a linearized track env in `conftest`; reuse for all three tests.

## Review

Dispatch `code-reviewer`. Confirm: output is drop-in for `compute_directional_place_fields` (verified against its real signature); the shared labeling helper is extracted from `goal_pair_direction_labels`, not duplicated; cross-links added; integration test passes; CHANGELOG updated; no plan/phase references in code/test names.
