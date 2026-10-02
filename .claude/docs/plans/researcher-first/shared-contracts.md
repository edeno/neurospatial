# Shared contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

- [Time-window semantics](#time-window-semantics) — Phases 3, 4 and 6
- [Error-message contract](#error-message-contract) — every phase (1–7)
- [Input conventions](#input-conventions) — Phases 1, 3 and 6
- [API snapshot](#api-snapshot) — Phase 6, and every later PR

---

## Time-window semantics

**Concepts.** Each data stream covers some time span:

- **Position (and heading) samples** `times`, shape `(n_samples,)`, sorted. Consecutive samples form intervals `k = [times[k], times[k+1])`.
- **Spike observation.** The time the electrophysiology was recording. It cannot be inferred from sparse spikes.
- **Analysis epochs.** An optional caller restriction, for example "only the run sessions".

**Rule.** An analysis uses only intervals inside *every* window relevant to the streams it consumes:

| Analysis kind | Valid interval `k` iff |
| --- | --- |
| spike + position (rates, decoding, phase precession) | gap/speed/bounds valid **and** `[times[k], times[k+1]) ⊆ epochs ∩ spike_window` |
| position only (behavior, segmentation, occupancy) | gap/speed/bounds valid **and** `[times[k], times[k+1]) ⊆ epochs` |
| spike only (PSTH, event regressors) | no intervals; an event is kept iff `[event + window[0], event + window[1]) ⊆ epochs ∩ spike_window` |

"Gap/speed/bounds valid" is the existing rule in `interval_valid_mask`, shown here at `src/neurospatial/environment/trajectory.py:90-121` on `main`:

```
(max_gap is None or dt_k <= max_gap)
and (min_speed is None or speed_k >= min_speed)
and start_bin_k >= 0
```

**Defaults.**

- `epochs=None` means unrestricted.
- `spike_window=None` means spikes are **assumed** observed wherever position is. This is the common case, ephys running at least as long as tracking. It is an assumption, not a finding, and the library never claims otherwise.
  - **The assumption stays visible.** Every spike + position result records `spike_window` (the `(n, 2)` windows actually applied, or None) and `spike_window_assumed: bool`. Both appear in `summary()` and in `to_xarray().attrs`.
  - **Loaders preserve acquisition windows.** When the source data carries observation intervals, such as NWB `units.obs_intervals`, the loader's data holder exposes them as a `spike_window` attribute, ready to pass to the analysis.
  - The population-silence warning below is a **heuristic** that catches one common mistake. It cannot establish recording coverage: it can't for a single unit, or for outages shorter than its threshold. Documentation must not present it as proof of coverage.
- `max_gap=0.5` (seconds) stays the default, and it is the gap detector everywhere. No new gap parameter is introduced.

**Accepted forms.** `epochs` and `spike_window` both accept:

- `None`;
- a length-2 sequence `(start, stop)`;
- an `(n, 2)` array-like of `[start, stop)` rows;
- any object with `.start` and `.end` 1-D array attributes, which is pynapple `IntervalSet`. Detect it by duck typing and never import pynapple for this.

**Normalization.** All forms go through one private helper, `neurospatial/_intervals.py::as_intervals(value, *, name) -> NDArray[np.float64]`, shape `(n, 2)`. It:

- raises `ValueError` (following the [error contract](#error-message-contract)) on non-finite values, `stop <= start`, or a wrong shape;
- sorts the rows and merges overlapping or touching rows;
- returns `None` for `None`.

**Containment check.** One vectorized private helper, `_intervals.py::intervals_contain(windows, starts, stops) -> NDArray[np.bool_]`. It returns True iff each `[starts[i], stops[i])` lies within a single row of `windows`. Implement it with `np.searchsorted(windows[:, 0], starts, side="right") - 1`, then check `stops <= windows[idx, 1]` and `idx >= 0`.

**Single implementation point.**

- `interval_valid_mask` gains keyword-only `epochs=None` and `spike_window=None`, already-normalized `(n, 2)` arrays or None. It ANDs `intervals_contain(...)` into the mask.
- `env` and `positions` become optional (`None`), because directional analyses have neither. When they are absent, the bounds rule (`start_bin_k >= 0`) is skipped, and the speed rule applies only if `speed` is given.
- Every family computes one mask through it and applies that mask identically to its numerator and its denominator. That's the existing invariant documented at `src/neurospatial/encoding/_binning.py:216-233`.
- No family implements its own gap logic.

**Decoding.** Time bins are formed separately within each maximal run of valid intervals. Bins are half-open. No decode bin spans an invalid interval. Spikes outside valid runs are not counted.

**Boundary rule, for every counter: bin edges never exceed their window.** For each window `[a, b)`, bin edges are computed as `a + dt·k`. The final right edge is clamped to exactly `b`, and edges that float rounding pushes beyond `b` are clamped as well. A spike at `t` is counted in a bin iff `left <= t < right`, so a spike exactly at a window stop is never counted. A fixed absolute epsilon such as `+1e-9` must not be used to decide bin counts. Required tests:
- decimal boundaries (`[0.1, 0.3)` with `dt=0.1` must not count a spike at `0.3`);
- large time offsets (`t0 = 1e9`);
- chunked and full computations giving identical counts.

The same rule applies to the per-sample (frame) counters. A spike exactly at `times[-1]` lies outside every interval `[times[k], times[k+1])` and is not counted.

The runs come from the gap, `epochs` and `spike_window` rules only, not from speed or bounds. Decoding rest and replay periods is legitimate, and `predict` receives no positions.

**Segmentation (laps, trials, runs, crossings).** A segment never spans an invalid interval. A segment in progress at an invalid interval is terminated there and is not counted as complete.

**Warnings.**

- *Do not* warn merely because gaps exist. Dropped frames and pauses are normal, and warnings on normal data train users to ignore warnings.
- Keep `main`'s existing all-intervals-excluded warning (`src/neurospatial/encoding/_binning.py:398-478`).
- **Population-silence warning.** Spike + position population functions only. If `n_units >= 5` and the merged spike train of all units has a silent stretch of at least 60 s inside the analyzed window, emit one `UserWarning`. Count the stretch from the window start to the first spike and from the last spike to the window end too. The message:

  > All {n} units are silent from {a:.1f} s to {b:.1f} s ({b-a:.0f} s) while position is tracked. If the electrophysiology was not recording then, pass spike_window=(start, stop) so that time is excluded from occupancy.

---

## Error-message contract

Every exception a user can trigger satisfies all of these:

1. **The message names what, why and how.** `str(exc)` states what was wrong (with the offending value or shape), why it matters, and a concrete fix as a line beginning `Fix:` that shows the corrected call or argument. The model is `[E1006]` in `src/neurospatial/environment/core.py` (on `main`).
   - **Exception for missing required arguments.** Python's own `TypeError` ("f() missing 1 required keyword-only argument: 'criterion'") is left as it is. It names the argument exactly, and type checkers and IDEs report it before the code runs. Required arguments are ordinary required parameters, never sentinels or `=None` defaults that raise.
   - **Exception for `KeyError` subclasses.** Python quotes `str(KeyError)` and escapes its newlines. So a `KeyError` subclass overrides `__str__` itself, and a bare `KeyError` carries its fix in the final sentence rather than on a separate `Fix:` line.
2. **Errors use public types.**
   - `src/neurospatial/_exceptions.py` gains `class NeurospatialError(Exception)`, the base for all library-defined exceptions.
   - Each existing class there (`RegionNotFoundError`, `BinIndexOutOfRangeError`, `IncompatibleEnvironmentError`, `LayoutNotBuiltError`), plus `EnvironmentNotFittedError` and `GraphValidationError`, adds `NeurospatialError` as a second base. Their stdlib base comes first, so `except ValueError` still works.
   - `NeurospatialError` is exported from the package root.
   - New exception classes are added only when callers need to catch that case specifically. Otherwise raise a stdlib type.
   - No leading-underscore exception class may reach users.
3. **The domain word fits the call.** A PSTH error says "peri-event", not "encoding".
4. **Every problem is reported at once.** When several arguments are invalid, the error lists all of them rather than failing one at a time.

---

## Input conventions

- **Spatial raw form:**

  ```
  func(env, spike_times, times, positions, [headings], [targets], *, epochs=None, spike_window=None, max_gap=0.5, ...)
  ```

  In population (plural) functions, `spike_times` is a sequence of 1-D arrays or a pynapple `TsGroup`. The plural functions accept `unit_ids=` (keyword, defaulting to `np.arange(n_units)`) and carry it on the result.
- **Directional raw form** (no env): `(spike_times, times, headings, *, ...)`. This is the documented exception in CLAUDE.md.
- **Egocentric operations:** `(positions, headings, targets)`.
- **Behavior:** `(env, times, positions, ...)` when an env is needed, otherwise `(times, positions, ...)`. Times always come before positions, matching the encoding order `spike_times, times, positions`. Arguments that a swap could confuse are validated (`times` must be 1-D and non-decreasing), and the error names the swap.
- **Segmentation:** `(position_bins, times, env, *, region_params)` (unchanged).
- **Population identity:** results index units by `unit_ids`.
  - **Labels are never overridden.** If an input already carries identity (a labelled `TsGroup`) and the caller also passes `unit_ids=`, the two must be identical, or the call raises an error listing both. Intentional relabelling happens upstream, on the input itself. This applies in every encoder and in `BayesianDecoder.fit`.
  - Decoders align encoding models to spike inputs **by label only when both sides carry caller-supplied labels**: the encoder or `fit` was given `unit_ids=` (or a labelled `TsGroup`), and so was the spike input. Otherwise they align by position and require equal unit counts.
  - A default `np.arange(n)` label is not caller-supplied and never triggers label alignment.
  - A label mismatch raises an error listing the missing and unexpected labels.
  - Duplicate `unit_ids` raise `ValueError` at the point they're supplied.
- **Optional `times`:** a function that can run without timestamps takes `times=None` as a keyword, never as a leading positional argument.

---

## API snapshot

The governance replacement is exactly two files:

- **`tests/data/public_api.txt`.** For each public namespace in this fixed list:

  ```
  neurospatial, neurospatial.encoding, neurospatial.decoding, neurospatial.behavior,
  neurospatial.events, neurospatial.ops, neurospatial.stats, neurospatial.simulation,
  neurospatial.io, neurospatial.io.nwb, neurospatial.animation, neurospatial.annotation,
  neurospatial.regions, neurospatial.layout
  ```

  it lists one line per `__all__` name: `<namespace>.<name><signature>`. The lines are sorted.
  - For a function, `<signature>` is `inspect.signature` with annotations removed and memory addresses in defaults stripped, so the snapshot doesn't change across Python or NumPy versions.
  - Otherwise it is ` (class)`, ` (module)` or ` (constant)`.
- **`tests/test_public_api_snapshot.py`.** The test regenerates the text in memory and compares it to the file. On a mismatch it prints a unified diff and the command to accept the change: `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py`. With that environment variable set, the test rewrites the file instead of asserting.

There are no reference counts, inventories, tiers or metadata. A deliberate API change is one regenerated file in the PR diff, which the reviewer reads.
