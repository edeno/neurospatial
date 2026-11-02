# Phase 6 — Errors name the call to make

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/neurospatial/_exceptions.py](../../../../src/neurospatial/_exceptions.py) — the hierarchy and `_format_error` (`:68`).
- [docs/errors.md](../../../../docs/errors.md) — the documented error story; its `NeurospatialError` claim is wrong today.
- [src/neurospatial/_keyword_hint.py](../../../../src/neurospatial/_keyword_hint.py) — the existing keyword-hint decorator.
- [src/neurospatial/_spikes.py:182](../../../../src/neurospatial/_spikes.py) — where a list of trains passed to a single-unit function becomes a numpy "inhomogeneous shape" error.
- [src/neurospatial/decoding/posterior.py:630-640](../../../../src/neurospatial/decoding/posterior.py) — "encoding_models must be 2-D … got shape ()" when a result object is passed.
- [src/neurospatial/encoding/_smoothing.py:712](../../../../src/neurospatial/encoding/_smoothing.py) — the thin invalid-`method` message.
- [src/neurospatial/_intervals.py:1699](../../../../src/neurospatial/_intervals.py) — `warn_excluded_intervals`, whose all-excluded branch becomes a raise.
- `docs/reviews/UX_REVIEW_2026-11-02.md` §6. The weak-site list from the audit (`scratchpad/errors/weak2.txt`) is not in the repo; regenerate it with the AST heuristic described there if needed.

**Contracts referenced:**

- [Discarded-data warning](shared-contracts.md#discarded-data-warning) — "raise when nothing usable is left".

## Tasks

- **`InputError`.**
  - Add `class InputError(NeurospatialError, ValueError)` to
    `_exceptions.py` and export it at the top level.
  - Convert user-input validation at the public boundary to it:
    - the shared trajectory validator (`encoding/_validation.py` around
      line 295);
    - neuron-count mismatch;
    - `unit_ids` length;
    - the new phase-1 refusals.
  - It subclasses `ValueError`, so existing `except ValueError` code keeps
    working.
  - Update `docs/errors.md` so `except NeurospatialError` is described
    accurately.
- **Wrong-kind inputs, detected at the boundary.** Add one helper,
  `_refuse_wrong_kind(value, *, name, caller, expected)`, in
  `src/neurospatial/_validation.py` that recognizes:
  - **A list or tuple of 1-D arrays** passed to a single-unit spike argument
    → `TypeError`:
    - **What:** "{caller}() takes one unit's spike times; got {n} trains."
    - **Fix:** "use {population_caller}(...) for a population, or pass
      spike_times[i]."
    - **Decoding:** "bin them first with bin_spikes_in_time(spike_times,
      dt=...)".
    - Call it before `_spikes.py:182` coerces.
  - **A neurospatial result object** passed where an array is expected →
    `TypeError` naming the result's method or attribute:
    - `spatial_information(r, ...)` → `r.spatial_information()`.
    - `decode_position(env, counts, single_result)` → "pass the population
      result from compute_spatial_rates".
    - `population_coverage(env, rates_result)` → accept the result or name
      `.firing_rates`.

    Keep a small mapping from result class to its suggestion.
  - **An integer-valued 2-D array passed to `BayesianDecoder.fit`
    (spike_times):** "this looks like a count matrix; use
    decode_position(env, counts, rates, dt=)". Remove the misleading "MATLAB
    zero padding" Fix for this input.
  - **Equal-length trains given as a 2-D array** to a single-unit function:
    the same message as the list case. The current "must be 1-D" Why line
    lists forms this function rejects; fix it.
- **Raise when everything is excluded.**
  - In `warn_excluded_intervals`, the all-excluded branch raises
    `InputError` with the existing `_all_excluded_message`.
  - Keep a warning (not a raise) only for callers where an empty result is
    a valid answer. Check each of the 11 call sites, and add a
    `raise_if_empty: bool = True` parameter for the exceptions.
  - This covers times in ms and `pd.DatetimeIndex` times.
  - It is also the only guard for positions in the wrong units (review S1).
    Phase 1 deferred the partial outside-environment warning: every
    drop-fraction or margin rule also fired on the documented
    `env.subset(...)` region-of-interest recipes
    (`spatial-analysis.md:1030`, `environments.md:932`), which keep as
    little time as a ×8 pixel error does. An untested idea for a later PR:
    a units error clusters the kept samples in a small corner of the
    environment, while a region-of-interest analysis covers the subset.
    Adopt it only if measured cases (×0.1, ×2.54, ×8, ×10, ×100 versus the
    subset recipes) separate cleanly.
- **Typo and choice hints.**
  - Simulator wrappers (`open_field_session`, `linear_track_session`,
    `simulate_session`) validate `**kwargs` against the inner function they
    forward to, but report the error under their own name with a close
    match.
  - `decode_session`'s kwargs error adds `difflib.get_close_matches` over
    all its accepted names, explicit parameters included.
  - The shared choice validator used at `_smoothing.py:712` (the
    `validate_choice`-style helper; find it with
    `grep -rn "def validate_choice\|def _validate_choice" src/neurospatial`)
    names the caller and adds "Did you mean …?". This fixes every `method=`
    message at once.
- **Style.**
  - Normalize the 80 `WHY:` sites to `Why:` (`grep -rn "WHY:" src/neurospatial`).
  - Remove the duplicated Fix in E1006 (`Environment()`).
  - Add a test that no `src/` string contains `"WHY:"`.
- **Public-boundary weak messages.** Rewrite the public-path sites listed in
  the review §6 with `_format_error`:
  - `encoding/grid.py:766`
  - `events/_core.py:316`
  - `io/files.py:508,662`
  - `decoding/estimates.py:509`
  - `decoding/metrics.py:470`
  - `encoding/population.py:441`
  - `environment/_visualization.py:298`
  - `decoding/posterior.py:760`
  - `replay/criteria.py:646`

  Re-read each one first; some may already be fixed by earlier phases.
- **Docs.** `docs/errors.md`: the `InputError` section, the wrong-kind
  examples, and a "what raises vs warns" table. CHANGELOG: "Added"
  `InputError`; **Breaking** all-excluded now raises.

## Deliberately not in this phase

- The 930 weak private raise sites. Only the public-boundary sites listed
  above are in scope.
- New unit heuristics; phase 1 owns those.
- No LLM redirect aliases for non-existent names (rejected).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_input_error_is_value_error_and_neurospatial_error` | `issubclass` both. The trajectory validator raises it. |
| `test_list_of_trains_to_single_unit_function` | `compute_spatial_rate(env, [a, b], t, p)` → `TypeError` mentioning `compute_spatial_rates`. The same for `is_place_cell` and `decode_position`. |
| `test_result_object_where_array_expected` | `spatial_information(r, occupancy=...)` → `TypeError` mentioning `r.spatial_information()`. `decode_position(env, counts, single)` names `compute_spatial_rates`. |
| `test_fit_count_matrix_hint` | `BayesianDecoder.fit(counts_2d_int)` message contains `decode_position` and not "MATLAB". |
| `test_all_intervals_excluded_raises` | Times in ms → `InputError` with "probably not in seconds". |
| `test_simulator_kwarg_typo_names_caller` | `open_field_session(n_plce_cells=3)` → the message names `open_field_session` and suggests `n_place_cells`. |
| `test_decode_session_kwarg_typo_did_you_mean` | `bandwdth=` → "Did you mean 'bandwidth'". |
| `test_invalid_method_did_you_mean` | `method="gaussian"` → mentions the caller and `gaussian_kde`. |
| `test_no_uppercase_why_labels` | No `"WHY:"` in `src/`. |
| `test_golden_path_warnings.py` | Still zero warnings, and no new raises on valid input. |

## Fixtures

Simulators with `rng=0`. Each error test constructs its one bad input inline.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
