# Phase 6b — One argument convention, checked

**Requires:** Phase 6a.

[← back to PLAN.md](PLAN.md) · [executing](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Read [executing.md](executing.md) first: branch and PR workflow, definition of done, CHANGELOG-per-commit, and what to do when the plan and reality disagree. This file holds only what is specific to Phase 6b.

This phase:

- makes every signature follow the [input conventions](shared-contracts.md#input-conventions): `times` before `positions`, `env` first, spike parameters named `spike_times`;
- adds one times/positions validator whose error names a swap, shared with Phase 4a's encoding validator;
- removes the `PositionLike` dual form from every analysis signature (decision 4);
- splits `decode_session`'s two modes, so `positions` is a required argument (error contract, rule 1);
- adds `BayesianDecoder.fit(unit_ids=)`.

It is mostly mechanical call-site updates across about 100 files. Regenerate the API snapshot (Phase 6a) at the end and put its diff in the PR.

**Inputs to read first** (verified on `main` at `da631a47`; earlier phases shift line numbers):

- [src/neurospatial/_validation.py](../../../../src/neurospatial/_validation.py) — `validate_finite`, `validate_lengths`.
- [src/neurospatial/environment/trajectory.py:338-378](../../../../src/neurospatial/environment/trajectory.py) — `env.occupancy`'s swap-aware times check (the model for 6b.1), and `env.bin_sequence` (:636-650). Both require **2-D** positions: `tests/environment/test_trajectory_gaps.py:363` expects `"2-dimensional array"` from `bin_sequence`.
- [src/neurospatial/encoding/_validation.py](../../../../src/neurospatial/encoding/_validation.py) — `validate_trajectory`, which Phase 4a gave a problem list and the swap message "did you pass positions before times?". Phase 4a's `tests/test_first_run_errors.py::test_swapped_times_positions` asserts that phrase.
- [src/neurospatial/encoding/spatial.py:2542](../../../../src/neurospatial/encoding/spatial.py) — `compute_spatial_rate(env, spike_times, times: NDArray | PositionLike, positions=None)`, the dual form (adapter at :2874; same in `compute_spatial_rates` :3026), `decode_session(_summary)` (`decoding/session.py:92, :679`, adapter at :479-500) and `BayesianDecoder.fit`/`predict`/`predict_summary`/`score` (`decoding/estimator.py:278, :404, :448, :498`). There is no `fit_predict`.
- **`decode_session` on `main`** (probed in this plan's remediation run): signature `decode_session(env, spike_times, times, positions=None, *, ..., encoding_models=None, ...)`. With neither `positions` nor `encoding_models`, it raises `ValueError: as_times_positions received a timestamp array but no positions …`. With `encoding_models=`, `positions` is never read: passing `positions` filled with 999 gave a posterior array-equal to omitting it. `BayesianDecoder.predict` and `predict_summary` reach the decode through this `encoding_models=` path with `positions=None` (`estimator.py:437, :486`).
- **Files earlier phases already changed** (search by symbol):
  - `decoding/estimator.py`: Phase 1 Task 6 (`_unit_ids_generated`, `_align_to_fitted_units`) and Phase 3c (`epochs`/`spike_window` replace `epoch`; per-run decode bins).
  - `decoding/session.py`: Phase 3c (per-run bins, `_evolve`, `spike_window` on results).
  - `events/alignment.py`: Phase 2a Task 6 (spike-group input to `population_peri_event_histogram`) and Phase 3c (event filtering). This phase renames the parameter only.
  - `behavior/vte.py`, `navigation.py`, `decisions.py`: Phase 2a Task 7 (VTE windows), Phase 2b Tasks 4 and 5 (stationary filters, `_velocity_heading_and_speed`), Phases 3d and 3e (`max_gap`/`epochs`, per-run kinematics). This phase reorders their arguments only.
  - `ops/egocentric.py`: Phase 2b Task 3 (heading interpolation) and Phase 3e (`heading_from_velocity(positions, times, …)` replaced `dt`).
  - `_results.py::resolve_unit_ids`: Phase 1 Task 7 added `input_ids=` and duplicate-label rejection.

**Contracts referenced:**

- [Input conventions](shared-contracts.md#input-conventions) — do not weaken the times-before-positions rule or the swap-naming error. 6b.5 implements the Population-identity rule for `BayesianDecoder.fit`.
- [Error-message contract](shared-contracts.md#error-message-contract) — rule 1 (separate functions instead of a `None` slot that raises; this decides `decode_session`), rule 4 (the swap error lists every problem at once) and the `Fix:` line.

## Tasks

**6b.1 One times/positions validator.** Add to `src/neurospatial/_validation.py`:

```python
def times_positions_problems(
    t: NDArray[np.float64], p: NDArray[np.float64]
) -> tuple[list[str], bool]:
    """Return every times/positions problem and whether the pair looks swapped."""
    problems: list[str] = []
    if t.ndim != 1:
        problems.append(f"times must be 1-D (n_samples,), got shape {t.shape}.")
    else:
        # Finiteness applies to every 1-D timestamp array, including a single
        # sample; only monotonicity needs two or more samples.
        finite = np.isfinite(t)
        if not finite.all():
            problems.append(f"times has {int((~finite).sum())} non-finite value(s), "
                            f"first at index {int(np.argmin(finite))}.")
        elif t.size > 1:
            down = np.flatnonzero(np.diff(t) < 0)
            if down.size:
                k = int(down[0])
                problems.append(
                    f"times must be monotonically non-decreasing; it decreases at "
                    f"{down.size} place(s), first {float(t[k])!r} -> {float(t[k + 1])!r} "
                    f"at index {k}.")
    if p.ndim not in (1, 2):
        problems.append(f"positions must be (n_samples, n_dims), got shape {p.shape}.")
    if t.ndim >= 1 and p.ndim >= 1 and len(t) != len(p):
        problems.append(f"times and positions must have the same length; times has "
                        f"{len(t)} samples, positions has {len(p)}.")
    looks_swapped = t.ndim == 2 or (
        p.ndim == 1 and p.size > 1 and bool(np.all(np.diff(p) >= 0)))
    return problems, looks_swapped


def validate_times_positions(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    call: str,
    order: Literal["times, positions", "positions, times"] = "times, positions",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a ``(times, positions)`` pair and name an argument swap.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Sample timestamps in seconds; must be 1-D, finite and non-decreasing.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        One position row per timestamp.
    call : str
        Public function name, used in the message.
    order : {"times, positions", "positions, times"}, default="times, positions"
        The order in which ``call`` takes the two arguments, so the message
        names the swap the caller actually made.

    Returns
    -------
    times, positions : ndarray
        float64 arrays; shapes are not changed.

    Raises
    ------
    ValueError
        Listing every problem, with a ``Fix:`` line that names the swap when the
        arguments look swapped.
    """
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    problems, looks_swapped = times_positions_problems(t, p)
    if not problems:
        return t, p
    raise ValueError(format_times_positions_error(problems, looks_swapped, call=call, order=order))


def format_times_positions_error(
    problems: list[str], looks_swapped: bool, *, call: str, order: str
) -> str:
    first, second = order.split(", ")
    fix = (f"Fix: did you pass {second} before {first}? Call {call}(..., {order}, ...)."
           if looks_swapped else
           "Fix: pass times as a sorted 1-D array of timestamps in seconds with one "
           "positions row per timestamp.")
    return (f"Invalid times/positions passed to {call}():\n- " + "\n- ".join(problems)
            + "\nWhy: each interval [times[k], times[k+1]) is weighted by its duration, "
            "so mis-shaped or unsorted timestamps give wrong numbers.\n" + fix)
```

- **Phase 4a's phrase is kept.** For the default order the swap line reads "did you pass positions before times?", exactly what `tests/test_first_run_errors.py::test_swapped_times_positions` asserts. Do not change that test.
- **Message substrings existing tests match.** The problem lines keep `"same length"` and `"monotonically non-decreasing"`. Before committing, `grep -rn 'match=' tests | grep -E 'same length|monotonic|1-dimensional|2-dimensional|swapped|before times'` and re-run every hit. Two tests match the old phrase `"positions and times must have same length"` and must change to `"same length"`, because the reordered functions now name `times` first: `tests/behavior/test_goal_directed.py:408` and `tests/behavior/test_path_efficiency.py:351`.
- **One swap detector.** Phase 4a's `encoding/_validation.py::validate_trajectory` calls `times_positions_problems` for the times/positions problems, appends its own `n_dims` problems, and raises once with `format_times_positions_error` (rule 4). It keeps the `n_dims` check and nothing else of its own swap logic.
- **`env.occupancy` and `env.bin_sequence`.** Replace their inline times checks (`trajectory.py:338-378`, :636-650) with `validate_times_positions(times, positions, call="Environment.occupancy")` (and `"Environment.bin_sequence"`). **Keep** their 2-D positions check after it, with its current `"2-dimensional array"` wording: the validator accepts 1-D positions, these methods do not, and `tests/environment/test_trajectory_gaps.py:363` asserts that text.
- Call `validate_times_positions` first in every public function that takes both arrays (6b.2). Positions-first functions pass `order="positions, times"`.

**6b.2 Signature changes** (decision 1, no shims). Validate, and update internal call sites, tests and doctests:

- **`(positions, times)` → `(times, positions)`:**
  - `behavior` (16): `approach_rate`, `compute_decision_analysis`, `compute_goal_directed_metrics`, `compute_path_efficiency`, `compute_pre_decision_metrics`, `compute_vte_session`, `compute_vte_trial`, `extract_pre_decision_window`, `goal_bias`, `head_sweep_from_positions`, `instantaneous_goal_alignment`, `mean_square_displacement`, `pre_decision_heading_stats`, `pre_decision_speed_stats`, `segment_by_velocity`, `time_efficiency`;
  - `ops.visibility_occupancy(env, positions, headings, times)` → `(env, times, positions, headings)`;
  - `simulation.generate_population_spikes(models, positions, times, *, headings=None, …)` → `(models, times, positions, *, headings=None, …)`;
  - `ops.heading_from_velocity(positions, times, *, …)` (Phase 3e) → `(times, positions, *, …)`. Update its internal callers (`behavior/vte.py`, `behavior/decisions.py`, `behavior/navigation.py`, `simulation/spikes.py`), Phase 2b's `_velocity_heading_and_speed` if it still takes `(positions, times)`, and the docs Phase 3e updated (CLAUDE.md pattern 7, `.claude/QUICKSTART.md`, `docs/api/index.md`, examples 22, 24 and 25).
- **Positions-first functions keep their order**, because `times` is optional there: `compute_trajectory_curvature(positions, times=None, …)` and `heading_direction_labels(positions=None, times=None, …)`. They call `validate_times_positions(times, positions, call=…, order="positions, times")` whenever `times` is given, so a swap raises "did you pass times before positions?". (`heading_direction_labels`'s own optional-argument modes are out of scope here.)
- **`env` first** (the field-metric majority — `field_size`, `rate_map_centroid`, `detect_place_fields`, `border_score` — is already env-first):
  - `encoding.compute_region_coverage(field_bins, env)`;
  - `encoding.field_shape_metrics(firing_rate, field_bins, env)`;
  - `encoding.rate_map_coherence(firing_rate, env)`;
  - `ops.map_points_to_bins(points, env)`;
  - `stats.shuffle_place_fields_circular_2d(encoding_models, env)`;
  - `animation.calibrate_video(video_path, env)`.
  - Segmentation-form functions keep env after the data. These are the functions whose first parameter starts with `position_bins` or is `trials` (`detect_laps`, `trajectory_similarity`, `trials_to_region_arrays`, …).
- **Spike parameter named `spike_times`:**
  - `decoding.bin_spikes_in_time(spike_trains, …)`;
  - `events.population_peri_event_histogram(spike_trains, …)`;
  - `behavior.restrict_spike_trains(trains, epochs)` (parameter only);
  - `simulation.validate_simulation(spike_trains=…)`. Phase 6c replaces this function's raw keyword form entirely; rename it here so this phase's rule test passes.
- **`PositionLike` dual form removed from every analysis signature** (decision 4). Every `times` parameter becomes a plain 1-D array; no `times` slot accepts a `PositionLike`. pynapple users pass `tsd.t, tsd.values`.
  - Delete the adapter branches: `encoding/spatial.py:2874` (and the plural's), `decoding/session.py:479-500`, and the estimator branches.
  - `_typing.PositionLike`, `as_times_positions` and `_is_position_like` stay for now: `recording.py` (`Session`) and `io/pynapple.py::from_pynapple` still use them. Phase 6c deletes them with `recording.py`.
  - Tests: delete the `PositionLike` cases (`_FakeTsdFrame` as `times`) in `tests/encoding/test_spatial_adapters.py` (:157-190) and `tests/decoding/test_session_adapters.py` (:97, :130); keep their spike-group parity cases. Run the pynapple job locally after `uv sync --all-extras`: `uv run pytest -m "pynapple and not napari" -n 0` (CI: `.github/workflows/test_pynapple.yml`).

**6b.3 Positions are required wherever they are used.** Which functions need positions follows Phase 3c:

- **Positions required** (encoding needs tracking): `compute_spatial_rate(s)`, `BayesianDecoder.fit(spike_times, times, positions, *, ...)` and `BayesianDecoder.score(spike_times, times, positions, *, ...)` (ground truth). Their `positions` loses its `=None` default.
- **No positions** (decoding without tracking): `BayesianDecoder.predict(spike_times, times, *, epochs=None, spike_window=None)` and `predict_summary(spike_times, times, *, time_chunk=1024, epochs=None, spike_window=None)`. Here `times` are the timestamps whose valid runs are tiled with decode bins; Phase 3c documents `times=np.arange(t0, t1, dt)` for spans without tracking.
- **`decode_session(_summary)` keeps one mode** (error contract, rule 1). On `main` it has two: encode-and-decode, which needs `positions`, and decode-with-given-models, in which `positions` is silently ignored (probe above). The modes become separate functions:
  - `decode_session(env, spike_times, times, positions, *, dt=…, …, max_gap=…, epochs=None, spike_window=None, …)` and `decode_session_summary(…)` take `positions` as an ordinary required argument and **lose `encoding_models=`**.
  - The decode-with-models path becomes a private `decoding/session.py::_decode_with_models(env, spike_times, times, encoding_models, *, dt, max_gap, epochs, spike_window, warn_on_drop, dtype)` (and a `_summary` sibling), extracted from the current passthrough branch with Phase 3c's per-run binning. `decode_session` calls it after encoding; `BayesianDecoder.predict`/`predict_summary` call it with the fitted models.
  - Public routes for models a user already has: `BayesianDecoder` (fit once, predict many), or `decode_position(env, bin_spikes_in_time(...), models, dt)` on binned counts. State both in `decode_session`'s See Also.
  - Tests: `tests/decoding/test_decode_session.py` passes `encoding_models=` at about :181, :204, :385, :435, :444, :536, :776, :887-890 and :1189, and `tests/decoding/test_scaling.py:157` does too. Port each to `BayesianDecoder(...).fit(...).predict(...)` or to `_decode_with_models` according to what it tests; delete tests whose only subject is the removed keyword (for example the one at :408, "Passing encoding_models= skips encoding step").
  - Re-check every `decode_session` call in docs and docstrings; Phase 4b's executable docs fail on any that still passes `encoding_models=`.

**6b.4 `tests/test_argument_conventions.py`,** a rule test over the snapshot namespaces with no name lists. For every public *function* (not class):

- if `times` and `positions` are both positional without defaults, `times` comes first;
- no parameter is named `spike_trains`, `trains` or `trajectory`;
- a positional `env` is the first parameter, unless the first parameter's name starts with `position_bins`, or is `trials` or `nwbfile`. NWB writers take the file container first, as pynwb does (`write_environment(nwbfile, env, …)`, `write_occupancy`, `write_place_field`); the exception is needed because the snapshot namespaces include `neurospatial.io.nwb`.

The archive branch is the reason this guard checks rules, not names. Its convention lint grew into prose-parsing tests.

**6b.5 `BayesianDecoder.fit(spike_times, times, positions, *, unit_ids=None, …)`** ([Population identity](shared-contracts.md#input-conventions)).

- The new keyword-only argument comes first among the keywords. Labels are never overridden, as in the encoders since Phase 1 Task 7. If the input is a labelled group and `unit_ids=` is also passed, they must be identical in the same order, or the call raises listing both. Otherwise whichever is present is used.
- Resolve with `resolve_unit_ids(unit_ids, n_units, input_ids=extracted_ids, context="BayesianDecoder.fit")` and set Phase 1's `_unit_ids_generated = unit_ids is None and extracted_ids is None`. A decoder fitted with `unit_ids=` then aligns a labelled predict input by label (Phase 1 Task 6).
- **Duplicate labels** are rejected by `resolve_unit_ids` itself (Phase 1 Task 7), so `fit(unit_ids=[3, 3, 7])` raises with no code here. This phase does not add duplicate checks anywhere else.
- Document the keyword and the pairing rule in `fit`.

**6b.5b Decode with precomputed rate maps: `BayesianDecoder.from_rates(rates, *, dt=0.025)`.** 6b.3 removes `decode_session(encoding_models=)`, so this classmethod is the public way to decode with rate maps the caller already computed, for example with a chosen bandwidth or on a training epoch. It returns a fitted, frozen decoder, the same object `fit` returns, so `.predict(spike_times, times)` works unchanged.
- **Input:** a `SpatialRatesResult` only. Any other type raises `TypeError` naming `compute_spatial_rates`.
- **Fields taken from `rates`:** `env`, `firing_rates` as the encoding models, `unit_ids`, and `spike_window`, which are carried for the record.
- **NaN bins.** Non-finite rate bins are treated as zero-rate, exactly as `decode_position` already does: warn once and point to `fill_value=0.0`.
- **Label alignment needs to know whether the labels were caller-supplied** ([Population identity](shared-contracts.md#input-conventions)).
  - Add the private init field `_unit_ids_generated: bool = field(default=False, repr=False, compare=False, kw_only=True)` to `SpatialRatesResult`, per [Population identity](shared-contracts.md#input-conventions).
  - In `__post_init__`, *before* `unit_ids` is resolved (`spatial.py:1414`), set it to `True` when `self.unit_ids is None`. Use `object.__setattr__`, as that method already does for `unit_ids`. A caller who constructs `SpatialRatesResult(..., unit_ids=[10, 20])` directly therefore gets caller-supplied labels, and `replace` preserves the flag because it is an init field.
  - The plural compute functions pass `unit_ids=None` to the result when the caller gave neither `unit_ids=` nor labelled input, rather than a pre-resolved `arange`. They pass the resolved labels otherwise.
  - `from_rates` copies `rates._unit_ids_generated` into the decoder's `_unit_ids_generated`.
  - A rate map built from a plain list therefore pairs a labelled `TsGroup` by position, as `fit` does.
- **Docstring:** an Examples section showing `rates = compute_spatial_rates(env, spikes, t, pos, epochs=train)` then `BayesianDecoder.from_rates(rates).predict(spikes, t, epochs=test)`.

**6b.6 Regenerate the API snapshot.** `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py`. The count does not change (expected 419, since `from_rates` is a method, not an export); every changed line is a reorder, rename or removed default from 6b.2–6b.5b. Then run Phase 4b's executable docs, the full suite and the slow tests (`-m "slow and not napari"`).

**6b.7 Documentation** (part of this PR; each commit adds its own CHANGELOG bullet per [executing.md](executing.md), and this task checks they are all present):

- `README.md`: the `generate_population_spikes` call at :213.
- `docs/getting-started/quickstart.md` and CLAUDE.md: the Canonical Argument Order block, with behavior as `(env, times, positions)`; pattern 7 (`heading_from_velocity(times, positions, …)`).
- `.claude/QUICKSTART.md` (:911-913 `spike_trains`) and `.claude/API_REFERENCE.md`.
- `docs/user-guide/interoperability.md`: the pynapple `PositionLike` section becomes the explicit `tsd.t, tsd.values` recipe.
- `examples/20_bayesian_decoding.py` and other examples using `spike_trains`, `encoding_models=` with `decode_session`, or the swapped behavior order (`grep -rln` over `examples/`). Sync them with `uv run jupytext --sync` and `uv run python docs/sync_notebooks.py`.
- `CHANGELOG.md` `[Unreleased]` (breaking): every reorder and rename in 6b.2; `PositionLike` no longer accepted; `positions` required in `compute_spatial_rate(s)`, `fit`, `score`, `decode_session(_summary)`; `decode_session(_summary)` no longer takes `encoding_models=`; `predict`/`predict_summary` take no positions and a plain `times` array; `fit(unit_ids=)`, which raises when it disagrees with a labelled group.

## Deliberately not in this phase

- **Duplicate-label rejection.** Phase 1 Task 7 put it in `resolve_unit_ids`; nothing to add here.
- **Data holders, `Session`/`load_session`/`recording.py`, and deleting `_typing.PositionLike`/`as_times_positions`.** Phase 6c.
- **`env.track` linear frame and linearization properties** (design-review High #11). Phase 2a fixes the W-maze bug; a track frame is new functionality.
- **Assembly results carrying `unit_ids`, and `bin_spikes_in_time` returning labelled counts.** Output work for Phase 7 or later; this phase only renames the parameter.
- **Behavior `epochs=`.** Phase 3 adds it; this phase only reorders.
- **Other decoder constructors.** `from_rates` takes only `SpatialRatesResult`. Constructors for directional or view models wait for a concrete need.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_argument_conventions.py::test_times_before_positions` | for all public functions with both parameters required, the rule holds. Run against `main` it failed for exactly the 18 functions reordered in 6b.2 (measured in the original dry run). At the start of this phase it also fails for `heading_from_velocity(positions, times)` from Phase 3e: 19 expected. Record the observed list in the PR; any extra name is a function an earlier phase added, and it gets reordered too |
| `…::test_no_spike_trains_parameter_name` | no public function parameter is named `spike_trains`/`trains`/`trajectory`. On `main` it fails for exactly `bin_spikes_in_time`, `population_peri_event_histogram`, `restrict_spike_trains`, `validate_simulation` |
| `…::test_env_is_first` | the rule holds. On `main` it fails for exactly the 6 env-first functions in 6b.2 (measured with the `nwbfile` exception; without it the three NWB writers also fail) |
| `tests/test_validation.py::test_swapped_times_positions_names_swap` | `validate_times_positions(positions_2d, times_1d, call="f")` raises `ValueError`; message contains `"did you pass positions before times"`, `"f(..., times, positions, ...)"`, `"Fix:"`, and both problems (shape and length are listed together when both apply) |
| `…::test_positions_first_order_names_reverse_swap` | `validate_times_positions(positions_2d, times_1d, call="compute_trajectory_curvature", order="positions, times")` (the function's `times` slot received positions) → message contains `"did you pass times before positions"` and `"compute_trajectory_curvature(..., positions, times, ...)"` |
| `…::test_unsorted_times_reports_first_decrease` | times `[0, 1, 0.5, 2]` → message names index 1 and `1.0 -> 0.5` |
| `tests/test_first_run_errors.py::test_swapped_times_positions` (Phase 4a, unchanged) | still passes: `"did you pass positions before times"`, two bullet lines for two problems |
| `tests/environment/test_trajectory_gaps.py:363` (unchanged) | `bin_sequence` with 1-D positions still raises `"2-dimensional array"` |
| `tests/behavior/test_argument_order.py::test_old_order_raises` (parametrized over the 16 reordered behavior functions and `ops.heading_from_velocity`, plus `compute_trajectory_curvature` called with times in the positions slot) | calling with the swapped order raises `ValueError` containing `"did you pass"` |
| `…::test_1d_column_swap_no_longer_silent` | 1-D env, `x` shape (600, 1), `times` shape (600, 1), the audit case: `compute_path_efficiency(env, x, times, goal)` (old order) raises; on `main` it returned efficiency 4.758 (correct 0.083) |
| `tests/decoding/test_estimator.py::test_fit_unit_ids_enable_label_alignment` | `BayesianDecoder(env).fit(trains, t, p, unit_ids=[10, 11, 12])`, then `predict` on a spike-group double keyed `[12, 11, 10]` (reordered trains) equals `predict(trains, t)`; keyed `[10, 11, 13]` → `ValueError` with `missing: [12]`, `unexpected: [13]` |
| `…::test_fit_unit_ids_must_match_group_labels` | `fit(group_keyed_[10, 20], …, unit_ids=[20, 10])` raises `ValueError` whose message contains both `[20, 10]` and `[10, 20]` and a `Fix:` line. `unit_ids=[10, 20]` is accepted, with `unit_ids == [10, 20]`; a plain list with `unit_ids=[20, 10]` gives `[20, 10]`; `unit_ids=[3, 3, 7]` raises (via Phase 1's resolver) |
| `…::test_from_rates_matches_fit` | On the two-epoch fixture, `BayesianDecoder.from_rates(compute_spatial_rates(env, spikes, t, pos, fill_value=0.0)).predict(spikes, t)` has a posterior `assert_allclose` (atol 1e-12) to `BayesianDecoder(env).fit(spikes, t, pos).predict(spikes, t)` with matching encoder parameters |
| `…::test_from_rates_label_alignment` | Rates from a plain list: predict on a `TsGroup` keyed `3, 7, 9` pairs by position (equal to the list predict). Rates from a `TsGroup` keyed `[10, 11, 12]`: predict on the same group reordered `[12, 11, 10]` equals the in-order predict; keyed `[10, 11, 13]` raises `ValueError` listing missing `[12]` and unexpected `[13]` |
| `…::test_from_rates_constructor_labels` | `SpatialRatesResult(firing_rates=..., env=env, ..., unit_ids=[10, 20])` built directly (not by a compute function). Then `BayesianDecoder.from_rates(r).predict(group keyed [20, 10])` pairs by label: unit 20's spikes meet unit 20's map, equal to predicting the group keyed `[10, 20]`. The same result built without `unit_ids` pairs by position. `rates[0]` and `replace(r, …)` keep the flag |
| `…::test_from_rates_rejects_other_types` | `from_rates(directional_rates)` raises `TypeError` naming `compute_spatial_rates` |
| `…::test_positions_required_where_used` | `list(inspect.signature(BayesianDecoder.predict).parameters)` is `["self", "spike_times", "times", "epochs", "spike_window"]`, and `predict_summary`'s is the same plus `time_chunk`. `fit`, `score`, `decode_session`, `decode_session_summary`, `compute_spatial_rate(s)` have a `positions` parameter with no default, and neither decode function has an `encoding_models` parameter. `decode_session(env, spikes, t)` raises `TypeError` naming `'positions'` |
| `…::test_predict_matches_decode_session` | `BayesianDecoder(env, dt=…).fit(spikes, t, p).predict(spikes, t).posterior` equals `decode_session(env, spikes, t, p, dt=…).posterior` (array-equal), so the extraction of `_decode_with_models` changed nothing |
| `tests/io_tests/test_pynapple.py` (pynapple job) | still passes; `from_pynapple(tsdframe)` still returns `(times, positions)` |
| `tests/test_public_api_snapshot.py` | passes with the regenerated file |
| Phase 4b executable-docs test | passes with every doc updated in 6b.7 |

## Fixtures

- **Behavior order tests:** reuse the existing behavior test trajectories (600 samples at 30 Hz on a 100 cm grid env). The 1-D case is the audit script `claim4c`, reproduced inline: `x = (50 + 45·sin(linspace(0, 6π, 600)))[:, None]`, `env = Environment.from_samples(linspace(0, 100, 500)[:, None], bin_size=2.0)`, `goal = [95.0]`.
- **Decoder tests:** reuse `tests/decoding` fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- Every task is implemented as specified; no public signature has a `None` default that raises when omitted.
- The "Deliberately not in this phase" list is honored.
- Validation slice tests pass; slow tests are marked.
- Tests aren't trivial: they exercise the asserted behavior, not tautologies (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are actually removed (the `PositionLike` adapter branches, the `encoding_models=` passthrough keyword, `validate_trajectory`'s private swap logic).
- User-facing documentation listed as tasks is updated, not deferred.
- Run `scientific-code-change-audit` on the `decode_session` split.
