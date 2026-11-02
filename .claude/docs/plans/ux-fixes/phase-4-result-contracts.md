# Phase 4 — Every result class meets the result contract

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [.claude/rules/public-api.md](../../../../.claude/rules/public-api.md) — "Result classes", "Cell types", "Peaks and unit identity".
- [tests/encoding/test_terminal_verbs_contract.py](../../../../tests/encoding/test_terminal_verbs_contract.py) — the contract test this phase widens.
- [src/neurospatial/_results.py](../../../../src/neurospatial/_results.py) — `ResultMixin`, the repr built from `summary()`, `_repr_html_`.
- [src/neurospatial/encoding/spatial.py:149](../../../../src/neurospatial/encoding/spatial.py) — `PlaceFieldsResult`; `:5223` `detect_place_fields`; `:6276` `place_cell_significance`; `:7102` `spatial_cell_type_significance` (the pattern to copy); `:2312` `SpatialRatesResult.plot` (the actionable no-index error to copy).
- [src/neurospatial/encoding/directional.py:1827](../../../../src/neurospatial/encoding/directional.py) and [src/neurospatial/encoding/tuning.py:677](../../../../src/neurospatial/encoding/tuning.py) — population `plot(idx, ...)` with `idx` required.
- [src/neurospatial/events/_core.py:1300](../../../../src/neurospatial/events/_core.py) — `PopulationPeriEventResult.plot`.
- [src/neurospatial/stats/shuffle.py:414](../../../../src/neurospatial/stats/shuffle.py) — `ShuffleTestResult`.
- [src/neurospatial/behavior/vte.py:1520-1535](../../../../src/neurospatial/behavior/vte.py) — VTE classification with zero head-sweep variance.
- [src/neurospatial/behavior/segmentation.py:848](../../../../src/neurospatial/behavior/segmentation.py) — lap direction (clockwise/counter-clockwise/unknown).
- `docs/reviews/UX_REVIEW_2026-11-02.md` §4 — the reproductions.

**Contracts referenced:**

- [Verdict vs screen](shared-contracts.md#verdict-vs-screen).
- [Argument vocabulary](shared-contracts.md#argument-vocabulary) — `plot(idx=None, *, ax=None)`.

## Tasks

- **Widen the contract test first.**
  - Parametrize `test_terminal_verbs_contract.py` over **every** public
    result class (collect from `tests/data/public_api.txt` entries ending in
    `Result`).
  - Give each class an explicit expected-verb row, so an intentional
    exception (for example `ResampledPathResult.samples_table()`) is visible
    in the table rather than skipped.
  - Assert per class:
    - `summary()` is a flat dict of scalars;
    - the `repr` mentions only names that are attributes or properties;
    - `plot` (if present) returns `Axes`;
    - `to_dataframe()` has `unit_id` when the result is per-unit;
    - tables carry `attrs["units"]`.
- **Population `plot`.**
  - Every population result takes `plot(idx=None, *, ax=None, ...)`.
  - With `idx=None`:
    - `PopulationPeriEventResult`, `TuningCurvesResult` and
      `DirectionalRatesResult` draw a units × time/value/direction heatmap,
      sorted by peak, with a colorbar in Hz.
    - `SpatialRatesResult` and other 2-D map results raise the shared
      actionable error from `spatial.py:2312`, moved to a helper in
      `_results.py` so every class uses one message.
  - A population PSTH titled as a mean says "Population mean of N units".
- **`PlaceFieldsResult`.**
  - `detect_place_fields(env, rates_or_array, *, bandwidth=None, ...)`
    accepts a `SpatialRateResult`/`SpatialRatesResult`. With a result,
    `bandwidth=None` inherits the result's recorded bandwidth. With an array,
    `bandwidth` stays required, with a Fix line saying where to get it.
  - Add `unit_id`, and per-field columns: `field`, `n_bins`, `area`
    (env units²), `peak_rate` (Hz), `centroid_*` (env units), `mean_rate`.
  - Add `summary_table()` (one row per unit: `n_fields`, largest-field area,
    peak), `attrs["units"]`, `plot(ax=)` (outlines over the rate map) and
    `peak_location()`.
  - Reuse `field_size`, `rate_map_centroid` and `field_shape_metrics`;
    don't reimplement them.
- **Population significance result.**
  - `place_cell_significance` returns a `PlaceCellSignificanceResult`, which
    holds:
    - `unit_ids`;
    - per-unit `observed`, `p_value`, `null_mean`, `null_std` and `z`;
    - `statistic="spatial_information"`, `statistic_units="bits/spike"`;
    - `n_shuffles`, `alpha` and `analysis_settings`.

    It provides `summary_table()` indexed by `unit_id`, `classify()`,
    `__getitem__`/`sel` via `UnitSelectionMixin`, and `plot(idx)` (null
    histogram with the observed value).
  - Delete the `dict[int, ShuffleTestResult]` return.
  - Check `spatial_cell_type_significance` to see whether the two can share
    one table builder.
  - `ShuffleTestResult` gains `statistic` and `statistic_units` fields,
    shown in its repr.
- **Path efficiency semantics.**
  - Efficiency above 1 comes from comparing the bin-graph geodesic with the
    continuous path length. Confirm first by measuring on
    `tmaze_alternation_session`.
  - Either compute both lengths in the same space, or clamp to 1 and record
    `attrs["efficiency_clamped"]`. Document the choice in the docstring.
- **Units in tables.**
  - Add `attrs["units"]` (per-column dict) to:
    - `DecodingResult.to_dataframe()`;
    - PSTH tables;
    - `trials_table`/`intervals_table`;
    - behavior result tables.
  - Make xarray `attrs["units"]` the same per-variable spelling ("rad", not
    "radians").
- **Environment repr.** It includes `units` and the nominal bin size:
  `Environment(name='', 2D, 2601 bins @ 2 cm, RegularGrid)`.
- **Properties vs methods.**
  - The rule: a no-argument scalar that is cheap to compute is a property;
    anything with options or real cost stays a method.
  - Apply it to `mean_vector_length` and peers. Find candidates with
    `grep -rn "def mean_vector_length\|def preferred_direction\b" src/neurospatial/encoding`.
  - The repr shows only what the rule makes accessible as `obj.name`.
  - `CellTypeSignificanceResult` either gets `n_place` etc. as properties, or
    drops them from the repr.
  - Add the rule to `.claude/rules/public-api.md`.
- **Single-unit `unit_id`.** `PeriEventResult` uses `pd.NA` like the rate
  results; its summary-table index shows `<NA>`, not `None`.
- **Directional peak vs preferred direction.** Document both in the class
  docstring (bin peak vs circular-mean direction). `summary_table` keeps one
  angle unit (rad), with degrees in the docs only.
- **VTE with zero sweep variance.** When `std_head_sweep` is at
  floating-point noise (`< 1e-12`), set `is_vte=<NA>` and say so in the
  existing warning (`vte.py` around line 1520). Report `mean_head_sweep` as
  `0.0` when it is below the same tolerance.
- **Lap direction on 1-D tracks.** For environments without a loop, report
  `"increasing"`/`"decreasing"` along the linear coordinate. Update the
  `Literal` at `segmentation.py:254` and `:848`. If phase 1's S8 refusal
  means `detect_laps` never sees a 1-D track, drop this item and note it in
  the PR.
- **Docs.**
  - Update the result-class docstrings.
  - CHANGELOG **Breaking**: `place_cell_significance` return type, removed
    dict.
  - "Added": `PlaceFieldsResult` tables, population heatmap plots.

## Deliberately not in this phase

- The visual fixes inside existing single-unit plots (colorbar labels, axes
  in cm, titles) belong to phase 5.
- Errors for passing a result where an array is expected belong to phase 6.
- No new metrics.

## Validation slice

| Test | Asserts |
| --- | --- |
| Widened `test_terminal_verbs_contract.py` | Every public result class matches its expected-verb row. |
| `test_population_plot_without_idx` | Heatmap for PSTH, tuning and directional results (`Axes`, colorbar label contains "Hz"). The shared actionable `ValueError` for spatial results. |
| `test_detect_place_fields_accepts_rate_result` | Bandwidth is inherited. Fields equal the array path with the same bandwidth. `to_dataframe()` has `unit_id`, `area`, `peak_rate`. |
| `test_place_cell_significance_population_result` | `summary_table()` index is `unit_id`. `p_value` matches the old dict values for `rng=0`. `statistic_units == "bits/spike"`. |
| `test_path_efficiency_not_above_one` | On `tmaze_alternation_session`, every efficiency is ≤ 1, or clamped and flagged. |
| `test_tables_carry_units` | Decode, PSTH, trials and behavior tables have `attrs["units"]`. |
| `test_repr_names_are_accessible` | For each result class, every `name=` in the repr is a `getattr`-able non-callable. |
| `test_vte_zero_sweep_is_na` | Constant-heading trials → `is_vte` is NA, not True. |
| `test_environment_repr_shows_units` | The repr contains the units and bin size. |

## Fixtures

The simulators (`open_field_session`, `linear_track_session`,
`tmaze_alternation_session`, `simulated_reactivation_session`), all with
`rng=0`. Put a session-scoped fixture in `tests/conftest.py` if several test
files need the same session.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
