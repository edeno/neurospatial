# Phase 3b — Consistent defaults and one return type per function

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/neurospatial/encoding/spatial.py:5837-5900](../../../../src/neurospatial/encoding/spatial.py) — `is_place_cell` (required `criterion`, `None` sentinels for `min_info`/`alpha`/`n_shuffles`).
- [src/neurospatial/encoding/spatial.py:1867-1880](../../../../src/neurospatial/encoding/spatial.py) — `SpatialRateResult.is_place_cell(criterion=...)`, whose only allowed value is `"threshold"`.
- The four sibling predicates: `encoding/directional.py:2918`, `encoding/egocentric.py:3525,3782`, `encoding/view.py:2223`.
- [src/neurospatial/behavior/navigation.py:3922-3940](../../../../src/neurospatial/behavior/navigation.py) — `compute_path_efficiency -> PathEfficiencyResult | pd.DataFrame`.
- [src/neurospatial/regions/ops.py:425-470](../../../../src/neurospatial/regions/ops.py) — `regions_containing_points` overloads on `return_dataframe`.
- [src/neurospatial/simulation/trajectory.py:975](../../../../src/neurospatial/simulation/trajectory.py) — `simulate_trajectory_laps(return_metadata=)`.
- [tests/encoding/test_predicate_contract.py](../../../../tests/encoding/test_predicate_contract.py) — the predicate contract to update.

**Contracts referenced:**

- [Verdict vs screen](shared-contracts.md#verdict-vs-screen) — implemented here.
- [Argument vocabulary](shared-contracts.md#argument-vocabulary) — the new defaults use the 3a names.

## Tasks

- **Predicates default to the shuffle.**
  - Set `criterion: Literal["threshold", "shuffle"] = "shuffle"` on all five
    `is_*_cell` free functions.
  - Rewrite each first docstring paragraph per the contract: "Test whether
    … with a circular-shift shuffle (default) …". State the default
    `n_shuffles`, that `rng` makes it reproducible, and that
    `criterion="threshold"` is the fast screen.
  - Remove `criterion=` from `SpatialRateResult.is_place_cell` and the
    equivalent result methods (`grep -n "criterion: Literal\[\"threshold\"\]" -r src/neurospatial/encoding`).
  - Rename those methods' first line to "Screen …".
  - Update `CLAUDE.md` "Scientific conventions" (the `has_place_field` /
    shuffle bullet) and `.claude/rules/public-api.md` "Cell types" to match.
- **Real defaults instead of `None` sentinels.**
  - Where `None` always resolves to one constant, put the constant in the
    signature: `min_info=0.5`, `alpha=0.05`, `n_shuffles=1000`, and the
    similar cases.
  - Find them with `grep -rn "None resolves to\|None means\|None is [0-9]" src/neurospatial`
    and check each one.
  - Keep `None` only where the value depends on the data or the environment
    (for example `bandwidth=None`). For those, the parameter doc's first
    sentence states the resolved value and how it is computed.
  - Write angular defaults in readable form in the docs ("π/6 rad (30°)")
    where the signature shows a float such as `0.5235987755982988`.
- **`n_shuffles` = 1000 everywhere**, unless a docstring cites a reason:
  - `has_phase_precession` (`encoding/phase_precession.py:964`) 200 → 1000
  - `pairwise_order_significance` 300 → 1000
  - `bias_similarity_significance` 100 → 1000

  If a lower default exists for runtime, measure the 1000-shuffle runtime on
  the doc example. Keep the lower value only with a docstring sentence giving
  the measured reason.
- **`min_occupancy`.**
  - Find what `None` means on `compute_spatial_rate(s)`.
  - If it resolves to a constant, make every encoding function use that
    constant.
  - If it means "method-dependent", document that in each `0.0`-default
    sibling's docs, or move all of them to `None` with the same resolution.
  - Do one or the other; don't leave the two families diverging without a
    stated reason.
- **`min_speed`.** It is required on seven behavior functions:
  - `compute_pre_decision_metrics`
  - `pre_decision_heading_stats`
  - `compute_vte_trial`
  - `head_sweep_from_positions`
  - `heading_direction_labels`
  - `headings_to_direction_labels`
  - `segment_by_velocity`

  Give each a justified default per the defaults-over-required policy. Cite a
  source or derive the value, put the reason in the docstring, and use one
  value across the VTE family.
- **One return type per function.**
  - `compute_path_efficiency` always returns `PathEfficiencyResult`. With
    `intervals`, the result holds per-trial arrays and a
    `summary_table()`/`to_dataframe()` with `units` in `attrs`.
  - Missing `time_efficiency` becomes NaN (float), never `None`.
  - Remove the DataFrame branch.
  - `regions_containing_points` always returns a DataFrame; delete the
    `return_dataframe` flag and its overloads. Users wanting arrays call
    `.to_numpy()`.
  - `simulate_trajectory_laps` always returns its metadata. Pick a named
    result or a fixed tuple, check callers, and delete the flag.
- **Docs.** CHANGELOG **Breaking**: the criterion default, removed
  `criterion=` on result methods, changed defaults (with old → new values),
  and removed return-type flags.

## Deliberately not in this phase

- `PlaceFieldsResult`, the population significance result and other result
  contract work belong to phase 4.
- The efficiency > 1 discretization issue in `compute_path_efficiency`
  belongs to phase 4 (result semantics), not here.
- No renames; 3a owns those.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_free_predicates_default_to_shuffle` | `inspect.signature(...).parameters["criterion"].default == "shuffle"` for all five. |
| `test_default_predicate_matches_explicit_shuffle` | `is_place_cell(env, st, t, pos, rng=0)` equals `criterion="shuffle", rng=0`, and agrees with `place_cell_significance` for the same unit and `rng` (existing single/population agreement rule). Mark `slow` if over 5 s. |
| `test_result_screens_take_no_criterion` | `SpatialRateResult.is_place_cell(criterion=…)` → `TypeError`. |
| `test_no_constant_none_sentinels` | For an explicit list of parameters (`min_info`, `alpha`, `n_shuffles`), the default is not `None` on any public function. |
| `test_n_shuffles_defaults_agree` | Every `n_shuffles` default is 1000, apart from an allowlist with documented reasons. |
| `test_min_speed_has_default` | No public `min_speed` is required. |
| `test_path_efficiency_single_return_type` | With and without `intervals` → `PathEfficiencyResult`. `summary_table().attrs["units"]` is present. No object-dtype columns. |
| `test_regions_containing_points_returns_frame` | Always a DataFrame. |
| `tests/encoding/test_predicate_contract.py` | Updated to the new contract and passing. |

## Fixtures

Existing simulator fixtures. Use `rng=0` everywhere.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
