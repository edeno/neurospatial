# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Argument vocabulary](#argument-vocabulary) — one name per concept (phases 3a, 3b, 4, 5, 6, 7).
- [Discarded-data warning](#discarded-data-warning) — when and how the boundary warns or raises (phases 1, 6).
- [Verdict vs screen](#verdict-vs-screen) — `is_*_cell` defaults and result-method screens (phases 3b, 4, 7).
- [Zero-warning golden path](#zero-warning-golden-path) — the noise invariant (all phases).

## Argument vocabulary

These decisions are settled by the maintainer. Each row becomes one test in
`tests/test_argument_conventions.py` (phase 3a), and the table is copied into
`.claude/rules/public-api.md` under "Names and arguments". Do not weaken it.

| Concept | Rule | Replaces |
| --- | --- | --- |
| Randomness | `rng: np.random.Generator \| int \| None = None`, keyword-only; `np.random.default_rng(rng)` semantics, so `rng=0` and `rng=np.random.default_rng(0)` give the same result | `seed=` (13 functions, all of `simulation` plus `animation.estimate_colormap_range_from_subset`); `line_seed=0` in `replay.detect_trajectory_radon` |
| One region | `region=` (a name, or a `Region`) | `region_name=` |
| Several regions | `regions=` | — |
| Two region roles in one call | a role prefix on each: `decision_region=` + `goal_regions=` | used only when two roles coexist; `goal_region=` alone (`detect_goal_directed_runs`) becomes `region=` or `goal=`, matching the #256 goal functions |
| Time-bin width | `dt=` (seconds) | `bin_size=` on `peri_event_histogram` (`events/alignment.py:388`) and `population_peri_event_histogram` (`:541`). Spatial, angular and distance `bin_size` (for example `peri_event_distance_histogram`, `events/distance.py:291`) are **unchanged** |
| Pre-entry window length | `window_duration=` | `pre_window=` (`compute_decision_analysis`, `behavior/decisions.py:1584`) |
| Paired inputs | suffix `_a` / `_b` | `_1/_2` (`field_stability`, `compute_field_emd`, `field_shift_distance`, `population_vector_correlation`), `_i/_j` (`field_overlap`), `angles1/angles2` (`circular_circular_correlation`) |
| Unit selection | `unit_ids=` | `cells=` (`replay.null_events`, `replay.spike_words`) |
| Temporal smoothing width | `smooth_window=` is seconds; a width counted in bins says so: `smooth_bins=` | `smoothing_window=` (`stats.generate_inhomogeneous_poisson_surrogates`, a bin count) → `smooth_bins=` (maintainer decision 2026-11-02) |
| Significance level | `alpha=` means only this | VTE mixing weight `alpha=` (`compute_vte_index`, `compute_vte_session`) → `speed_weight=` |
| Matplotlib target | `ax=` keyword-only, never positional, never `axes=` | positional `ax` in free plot functions and in result `plot` methods (after `idx` where present); `axes=` in `plot_peri_event_response` → `ax=` taking a pair |
| Field resampling | `f(source_env, target_env, values, *, …)` | `resample_field(field, src_env, dst_env)` → `resample_field(source_env, target_env, field)` to match `map_probabilities` |
| Graph input in `ops` | `graph=` | `G` (`distance_field`, `neighbors_within`, `geodesic_distance_matrix`, `geodesic_distance_between_points`, `pairwise_distances`) |
| Sample times | `times` | `in_epochs(t, …)` |

No old name is kept as an alias; the old keyword raises the normal Python
`TypeError`.

## Discarded-data warning

The public boundary warns whenever it silently discards input the user passed,
or sees input that contradicts a declared unit.

- **Threshold.** Warn when more than `DISCARD_WARN_FRACTION = 0.05` of the
  input is discarded. Measure it in the unit the analysis uses: seconds of
  occupancy for position gates, number of spikes for spike gates. Raise when
  nothing usable is left (phase 6).
- **Home.** `DISCARD_WARN_FRACTION` lives in `src/neurospatial/_constants.py`
  and replaces `_DROP_WARN_THRESHOLD` (`encoding/_spatial_counts.py:68`).
- **Message.** Built with `_format_error`
  (`src/neurospatial/_exceptions.py:68`), with three parts:
  - **What**, with numbers: `"{caller}: {n} of {total} s of position samples
    fall outside the environment (env x∈[…] cm; positions x∈[…])."`
  - **Why**: the most likely cause, such as units, frame or clock.
  - **Fix**: the exact argument or call.
- **Mechanics.**
  - Category `UserWarning`, with `stacklevel=_external_stacklevel()`
    (`src/neurospatial/_stacklevel.py`).
  - At most one discarded-data warning per call.
  - Silenced by the existing `warn_on_drop=False` where that parameter exists.

## Verdict vs screen

- **Free predicates on raw arrays give verdicts.**
  - `is_place_cell` (`encoding/spatial.py:5837`), `is_head_direction_cell`
    (`encoding/directional.py:2918`), `is_object_vector_cell`
    (`encoding/egocentric.py:3525`), `is_egocentric_object_vector_cell`
    (`:3782`) and `is_spatial_view_cell` (`encoding/view.py:2223`) all default
    to `criterion="shuffle"`.
  - `criterion="threshold"` stays available.
- **Result methods and `classify()` / `label_cell_types()` give screens.**
  They take no `criterion=` argument. Today `SpatialRateResult.is_place_cell`
  (`encoding/spatial.py:1867`) requires `criterion` with the single allowed
  value `"threshold"`; remove that argument.
- **Docstrings.** The first line of every predicate says which it is: "Test
  whether …" for verdicts, "Screen …" for result methods.

## Zero-warning golden path

`tests/test_golden_path_warnings.py` (new in phase 1) runs the documented
workflow with `warnings.simplefilter("error")`:

1. simulated open field;
2. `Environment.from_polygon`;
3. `compute_spatial_rates`;
4. `summary_table`;
5. `decoding.decode_session`;
6. `peri_event_histogram`;
7. `behavior.compute_speed`.

Every later phase keeps it green. A new warning that fires on this path is a
bug in the warning, not in the test.
