# Phase 3a — One name per concept

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [tests/test_argument_conventions.py](../../../../tests/test_argument_conventions.py) — the existing rule tests and the `public_functions()` iterator to reuse.
- [.claude/rules/public-api.md](../../../../.claude/rules/public-api.md) — "Names and arguments".
- [CLAUDE.md](../../../../CLAUDE.md) — "Canonical argument order"; the `events` line says `*, bin_size` and changes to `*, dt`.
- [src/neurospatial/_random.py](../../../../src/neurospatial/_random.py) — the shared `rng` handling (`_entropy`, `_stream`) the simulators should adopt.

**Contracts referenced:**

- [Argument vocabulary](shared-contracts.md#argument-vocabulary) — the full rename table. This phase implements every row. Do not weaken it, and add no aliases.

## Tasks

- **Write the rule tests first.**
  - Add one test per vocabulary row to `tests/test_argument_conventions.py`,
    using `public_functions()`. Each test fails on `main` and lists the
    offenders. Suggested tests:
    - `test_randomness_is_rng`: no parameter named `seed` or `*_seed`.
    - `test_single_region_is_region`: no `region_name`.
    - `test_time_bin_width_is_dt`: a parameter named `bin_size` whose
      docstring entry says "seconds" or "s" is an offender. Parse with
      `numpydoc.docscrape` if available, or a regex on the Parameters block.
    - `test_paired_inputs_use_a_b`: no `*_1`/`*_2`/`*_i`/`*_j` pairs and no
      `angles1`.
    - `test_unit_selection_is_unit_ids`: no `cells`.
    - `test_ax_is_keyword_only`: every `ax` is `KEYWORD_ONLY`, and no `axes`.
    - `test_alpha_is_significance_level`: an allowlist of the functions where
      `alpha` appears, all significance uses.
    - `test_graph_argument_name`: no `G`.
    - `test_times_not_t`.
  - Extend the iterator to result-class `plot` methods for the `ax` rule.
- **Randomness.**
  - Rename `seed` → `rng` in every `simulation` function (13 sites,
    `grep -rn "seed" src/neurospatial/simulation | grep "def \|seed:"`) and in
    `animation.estimate_colormap_range_from_subset`.
  - Route them through `_random._entropy`/`np.random.default_rng` so `int`,
    `Generator` and `None` all work.
  - `replay.detect_trajectory_radon(line_seed=0)`
    (`replay/trajectory.py:1599`) → `rng=0`. Keep the deterministic default,
    because a fixed line set is part of the method's reproducibility.
  - `SimulationSession.metadata`, and any other place that records `seed`,
    records `rng` entropy instead.
- **Regions.** Rename across `behavior` and `regions`:
  - `region_name` → `region`: `distance_to_boundary`, `detect_region_crossings`,
    `region_reward_field`, `regions.mask_to_region`.
  - `detect_goal_directed_runs(goal_region=)` → `goal=`, consistent with the
    #256 goal functions. Check `behavior/navigation.py` for the shared goal
    resolver and reuse it.
  - `compute_decision_analysis` keeps `decision_region` + `goal_regions`
    (two roles).
  - `compute_vte_session(decision_region=)` and
    `decision_region_entry_time(region=)` — `compute_vte_session` has a
    single region role, so it becomes `region=`.
  - `RegionNotFoundError(argument=...)` defaults to `"region_name"`
    (`_exceptions.py` around line 95). Change the default to `"region"`.
- **Time-bin width.**
  - `peri_event_histogram` (`events/alignment.py:388`),
    `population_peri_event_histogram` (`:541`) and the top-level
    re-export: `bin_size` → `dt`.
  - Result fields named `bin_size` on PSTH results become `dt`.
  - Leave the spatial, angular and distance `bin_size` (including
    `events/distance.py:291`) alone.
- **Windows.** `compute_decision_analysis(pre_window=)`
  (`behavior/decisions.py:1584`) → `window_duration=`.
- **Paired suffixes.** `_a`/`_b` on `field_stability`, `compute_field_emd`,
  `field_shift_distance` (including `env_1/env_2` → `env_a/env_b`),
  `population_vector_correlation`, `field_overlap`, and
  `circular_circular_correlation`.
- **Units.** `cells` → `unit_ids` on `replay.null_events`
  (`replay/calibration.py:1504`) and `replay.spike_words`
  (`replay/sequence_match.py:193`).
- **Smoothing.** `stats.generate_inhomogeneous_poisson_surrogates(smoothing_window=)`
  (`stats/surrogates.py:165`) → `smooth_window=`. Check the docstring unit;
  if it is a bin count rather than seconds, note that in the docstring rather
  than changing meaning.
- **VTE weight.** `compute_vte_index(alpha=)` (`behavior/vte.py:906`) and
  `compute_vte_session(alpha=)` → `speed_weight=`. Update `VTESessionResult`
  fields and `analysis_settings` that record it.
- **`ax`.**
  - Make `ax` keyword-only in every free plot function and every result
    `plot` method.
  - `plot_peri_event_response(axes=)` (`events/_plotting.py:29`) →
    `ax=(raster_ax, psth_ax)`.
  - Update every internal call that passes `ax` by position:
    `grep -rnE "\.plot\([^)]*, ?ax\b|\.plot\(ax\b" src docs examples`.
  - The population `plot(idx, ax)` signature becomes `plot(idx=None, *, ax=None, ...)`.
    The `idx=None` behavior itself is phase 4; here keep today's behavior
    when `idx` is missing.
- **Resampling order.** `resample_field(field, src_env, dst_env)`
  (`ops/mapping.py:350`) → `resample_field(source_env, target_env, field, *, …)`.
- **Graph.** `G` → `graph` across `ops` distance functions.
- **`in_epochs(t, …)`** (`behavior/epochs.py:148`) → `times`.
- **Repo-wide call-site sweep.** Grep `src/`, `tests/`, `docs/`, `examples/`
  (notebooks: edit the paired `.py` via jupytext, then re-run
  `docs/sync_notebooks.py`), `benchmarks/` and the README for every renamed
  keyword.
- **Snapshot and contract.**
  - Regenerate `tests/data/public_api.txt` and review the diff; only the
    renamed parameters may change.
  - Copy the vocabulary table into `.claude/rules/public-api.md`.
  - Update the CLAUDE.md `events` line to `f(spike_times, event_times, window, *, dt)`.
- **CHANGELOG.** A single **Breaking** entry with an old → new table.

## Deliberately not in this phase

- Default values (`n_shuffles`, `min_speed`, `min_occupancy`, `criterion`)
  and return types belong to phase 3b.
- `idx=None` behavior of population plots belongs to phase 4.
- Renaming `compute_` outliers belongs to phase 7, after the rule is written.
- No aliases for old names.

## Validation slice

| Test | Asserts |
| --- | --- |
| New rule tests in `tests/test_argument_conventions.py` | Each one fails on `main` (record the offender list in the PR description) and passes after. |
| `tests/test_public_api_snapshot.py` | Snapshot diff contains only the listed renames. |
| `uv run pytest --doctest-modules src/neurospatial/ -n 0` | All examples use the new names. |
| `uv run pytest tests/docs/test_executable_docs.py -n 0` | Docs fences run. |
| `test_simulation_rng_reproducible` | `open_field_session(rng=0)` twice → identical spikes. `rng=np.random.default_rng(0)` matches `rng=0`. |
| `test_detect_trajectory_radon_default_is_deterministic` | Two default calls give identical scores. |
| `test_golden_path_warnings.py` | Still zero warnings. |
| Full default suite | Passes (cross-cutting change). |

## Fixtures

None new. Existing tests are updated in place.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
