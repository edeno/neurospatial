# Phase 6 — Behavior: binning & dt correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#input-validation-helpers)

Seven silent-correctness defects in the behavior layer, all scoped to `src/neurospatial/behavior/`. They fall into three families: (1) out-of-bounds bin indices (`-1`, or `>= n_bins`) silently wrap when used to gather from region masks / Voronoi label arrays, so an out-of-environment trajectory sample is mis-attributed to a real region; (2) velocity / rate / lag computations divide by `dt` (or a median `dt`) with no guard against zero, negative, or duplicate timestamps, producing `inf`/`nan` or a `ZeroDivisionError`; (3) path-efficiency helpers treat an `inf` geodesic length as a valid `0.0` and mix Euclidean and geodesic optimal distances. Each produces a plausible-but-wrong number (or a swallowed crash) rather than an obvious failure, so each fix ships with a fail-before / pass-after regression test.

**Soft dependency on phase 4.** This phase imports `validate_finite` from `src/neurospatial/_validation.py`, the new top-level module created in **phase 4** (see [shared contracts](shared-contracts.md#input-validation-helpers)). Land phase 4 first, or — if phase 6 lands first — create that module here per the contract verbatim and let phase 4 adopt it. Do **not** fork a private copy inside `behavior/`. Only `validate_finite` is used in this phase (timestamp-finiteness guards); `validate_lengths` is not needed here because every entry point already has its own `len(...) != len(...)` length check.

---

**Inputs to read first:**

- [../../../../src/neurospatial/behavior/segmentation.py:194](../../../../src/neurospatial/behavior/segmentation.py) — `detect_region_crossings` (def at 194). The bug: line 295 `in_region = region_mask[position_bins]` gathers from a `(n_bins,)` bool mask using raw bins; a `-1` bin wraps to the last bin and a `>= n_bins` bin raises or wraps depending on dtype. An out-of-env sample is silently treated as "in region".
- [../../../../src/neurospatial/behavior/segmentation.py:330](../../../../src/neurospatial/behavior/segmentation.py) — `detect_runs_between_regions` (def at 330). Same defect twice: lines 477–478 `in_source = source_mask[position_bins]` / `in_target = target_mask[position_bins]`. Also the dt defect: line 533 `velocities = distances / dt` where `dt = np.diff(run_times)` (line 532) — duplicate timestamps in a run give `inf` velocity and corrupt the `min_speed` filter.
- [../../../../src/neurospatial/behavior/segmentation.py:552](../../../../src/neurospatial/behavior/segmentation.py) — `segment_by_velocity` (def at 552). dt defect: line 673 `dt = np.diff(times)`, line 674 `velocities = distances / dt`; `median_dt = np.median(dt)` (line 679) also feeds the smoothing window-width and goes to `0`/negative on duplicate/unsorted timestamps.
- [../../../../src/neurospatial/behavior/segmentation.py:1096](../../../../src/neurospatial/behavior/segmentation.py) — `segment_trials` (def at 1096). Mask-gather defect: lines 1313–1314 `in_start = start_mask[position_bins]` and `in_end_regions = {region: mask[position_bins] for ...}`.
- [../../../../src/neurospatial/behavior/decisions.py:733](../../../../src/neurospatial/behavior/decisions.py) — `detect_boundary_crossings` (def at 733). Label-gather defect: line 767 `trajectory_labels = voronoi_labels[position_bins]`. `voronoi_labels` is `(n_bins,)` and already uses `-1` as an "unreachable" sentinel (filtered at lines 775–777), so an out-of-env `-1` bin wraps to the last bin's *real* label and is wrongly counted as reachable.
- [../../../../src/neurospatial/behavior/decisions.py:797](../../../../src/neurospatial/behavior/decisions.py) — `compute_decision_analysis` (def at 797). Same label-gather defect: line 904 `trajectory_labels = voronoi_labels[position_bins]` (this array is stored in `DecisionBoundaryMetrics.goal_labels`).
- [../../../../src/neurospatial/behavior/decisions.py:582](../../../../src/neurospatial/behavior/decisions.py) — `geodesic_voronoi_labels` (def at 582) — read to confirm the `(n_bins,)` shape and `-1` sentinel semantics the gather fix must preserve.
- [../../../../src/neurospatial/behavior/navigation.py:1787](../../../../src/neurospatial/behavior/navigation.py) — `approach_rate` (def at 1787). dt defect: line 1853 `dt = np.diff(times)`, line 1857 `rates[1:] = d_distance / dt`; duplicate timestamps yield `inf`/`nan` approach rates with no diagnostic.
- [../../../../src/neurospatial/behavior/navigation.py:1523](../../../../src/neurospatial/behavior/navigation.py) — `compute_path_efficiency` (def at 1523). Two defects. (a) Line 1591 `shortest_length=shortest if np.isfinite(shortest) else 0.0` — when `shortest_path_length` returns `inf` (disconnected goal), the result reports `shortest_length=0.0`, which reads as "goal is at the start", the opposite of "unreachable". (b) Lines 1583–1585 call `time_efficiency(...)` which internally uses a **Euclidean** optimal distance (see below) even when `compute_path_efficiency` was called with `metric="geodesic"`, so the time-efficiency number silently mixes metrics.
- [../../../../src/neurospatial/behavior/navigation.py:1315](../../../../src/neurospatial/behavior/navigation.py) — `time_efficiency` (def at 1315). Line 1363 `shortest_dist = float(np.linalg.norm(goal - start))` is straight-line; lines 1357–1367 build `optimal_time` from it. It has no `env`/`metric` parameter, so it cannot honor a geodesic request — the fix must route the optimal distance through the caller, not recompute it here.
- [../../../../src/neurospatial/behavior/navigation.py:1268](../../../../src/neurospatial/behavior/navigation.py) — `shortest_path_length` (def at 1268). Confirms the geodesic branch returns `inf` for a disconnected goal (docstring line 1292; `float(distances[start_bin])` at 1312). This is the source of the `inf` that `compute_path_efficiency` mishandles.
- [../../../../src/neurospatial/behavior/trajectory.py:454](../../../../src/neurospatial/behavior/trajectory.py) — `mean_square_displacement` (def at 454). dt defect: line 617 `dt = np.median(time_diffs)`, line 618 `n_lags = int(max_tau / dt)`, line 621 `tau_values = np.linspace(dt, max_tau, n_lags)`. A median `dt` of `0` (duplicate-dominated timestamps) raises `ZeroDivisionError` at line 618; a negative median (unsorted) makes `n_lags` negative and `np.linspace` ill-formed.
- [../../../../src/neurospatial/ops/binning.py:372](../../../../src/neurospatial/ops/binning.py) — `regions_to_mask` (def at 372) — confirms the returned mask is `(n_bins,)` bool (docstring line 401). The `_safe_gather` helper relies on this `(n_bins,)` length matching `env.n_bins`.
- [../../../../src/neurospatial/_validation.py](../../../../src/neurospatial/_validation.py) — `validate_finite` (created in phase 4 per shared-contracts). Used here to guard timestamp arrays at the entry points where non-finite times are a hard error.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — `validate_finite(a, *, name, allow_nan=False)` from `src/neurospatial/_validation.py`. Use `validate_finite(times, name="times")` to reject `inf`/`nan` timestamps before any `np.diff`. **Do not weaken**: `validate_finite` raises on Inf always and on NaN unless `allow_nan=True`, and never silently coerces. It does not check monotonicity or duplicates — that is a separate, local guard added in this phase (see Task 2's `_positive_dt`).

---

## Tasks

### Task 1 — Stop out-of-bounds bins from silently wrapping into real regions/labels

Five gather sites index a `(n_bins,)` array (a region bool mask or a Voronoi label array) with `position_bins`, which may legitimately contain `-1` (point off the environment / unmapped) and could contain `>= n_bins` from a caller that built bins differently. NumPy fancy-indexing wraps `-1` to the last element and raises on `>= n_bins` only for the in-bounds-int path — the practical bug seen in review is the `-1 → last bin` wrap, which mis-labels an out-of-env sample as belonging to the *last* region/Voronoi cell.

Because the same `(labels_or_mask)[position_bins]` pattern recurs five times, add **one** private helper and route every site through it. Add the helper near the top of `segmentation.py` (after the imports block, before the first public function at line 194) and import it into `decisions.py`.

**1a. Add `_safe_gather` to `segmentation.py`** (insert after the module imports, before `detect_region_crossings`):

```python
def _safe_gather(
    values: NDArray,
    position_bins: NDArray[np.int64],
    *,
    fill: object,
) -> NDArray:
    """Gather ``values[position_bins]`` with out-of-bounds bins mapped to ``fill``.

    ``position_bins`` may contain ``-1`` (sample off the environment / unmapped)
    or indices ``>= len(values)``. Plain fancy indexing would wrap ``-1`` to the
    last element and silently mis-attribute an out-of-environment sample to a
    real bin. This helper instead returns ``fill`` for every out-of-bounds
    position, so callers can treat those samples as "outside" rather than
    accidentally "inside the last region".

    Parameters
    ----------
    values : NDArray, shape (n_bins,) or (n_bins, ...)
        Per-bin array to gather from (e.g. a boolean region mask or an integer
        Voronoi-label array).
    position_bins : NDArray[np.int64], shape (n_samples,)
        Bin index per trajectory sample. May contain ``-1`` or out-of-range
        indices.
    fill : object
        Value substituted wherever ``position_bins`` is out of bounds. Use
        ``False`` for boolean region masks and ``-1`` for integer label arrays
        (matching the ``-1`` "unreachable" sentinel those arrays already use).

    Returns
    -------
    NDArray
        Array of length ``len(position_bins)`` (leading axis) with out-of-bounds
        rows set to ``fill``.
    """
    position_bins = np.asarray(position_bins, dtype=np.int64)
    n_bins = values.shape[0]
    in_bounds = (position_bins >= 0) & (position_bins < n_bins)
    safe_idx = np.where(in_bounds, position_bins, 0)
    gathered = values[safe_idx]
    gathered[~in_bounds] = fill
    return gathered
```

Notes for the executor: `gathered[~in_bounds] = fill` requires `gathered` to be a fresh array (it is — fancy indexing copies). For a boolean mask, `fill=False`; for an integer label array, `fill=-1`. The helper deliberately does **not** raise on out-of-bounds bins — out-of-env samples are a normal, expected occurrence in trajectory data, and the correct behavior is to exclude them, not to crash.

**1b. `segmentation.py:295` — `detect_region_crossings`.** Replace:

```python
    in_region = region_mask[position_bins]
```

with (after the existing `position_bins`/`times` length check; cast bins to int64 first as the other functions already do):

```python
    position_bins = np.asarray(position_bins, dtype=np.int64)
    in_region = _safe_gather(region_mask, position_bins, fill=False)
```

This makes an out-of-env sample read as "not in region", so it cannot create a spurious entry/exit crossing at the `np.diff(in_region.astype(np.int8))` step (line 300).

**1c. `segmentation.py:477-478` — `detect_runs_between_regions`.** This function already does `position_bins = np.asarray(position_bins, dtype=np.int64)` (line 468). Replace:

```python
    in_source = source_mask[position_bins]
    in_target = target_mask[position_bins]
```

with:

```python
    in_source = _safe_gather(source_mask, position_bins, fill=False)
    in_target = _safe_gather(target_mask, position_bins, fill=False)
```

**1d. `segmentation.py:1313-1314` — `segment_trials`.** Cast and replace:

```python
    in_start = start_mask[position_bins]
    in_end_regions = {region: mask[position_bins] for region, mask in end_masks.items()}
```

with:

```python
    position_bins = np.asarray(position_bins, dtype=np.int64)
    in_start = _safe_gather(start_mask, position_bins, fill=False)
    in_end_regions = {
        region: _safe_gather(mask, position_bins, fill=False)
        for region, mask in end_masks.items()
    }
```

**1e. `decisions.py` — import `_safe_gather` and fix both label gathers.** Add to the imports block of `decisions.py`:

```python
from neurospatial.behavior.segmentation import _safe_gather
```

At `decisions.py:767` (`detect_boundary_crossings`), replace:

```python
    trajectory_labels = voronoi_labels[position_bins]
```

with (the function already does `position_bins = np.asarray(position_bins)` at line 763 — make it int64):

```python
    position_bins = np.asarray(position_bins, dtype=np.int64)
    trajectory_labels = _safe_gather(voronoi_labels, position_bins, fill=-1)
```

`fill=-1` matches the existing "unreachable" sentinel, so the downstream `(prev_labels != -1) & (curr_labels != -1)` mask (lines 775–777) automatically excludes out-of-env samples from crossing detection — no further change needed there.

At `decisions.py:904` (`compute_decision_analysis`), replace:

```python
    trajectory_labels = voronoi_labels[position_bins]
```

with:

```python
    position_bins = np.asarray(position_bins, dtype=np.int64)
    trajectory_labels = _safe_gather(voronoi_labels, position_bins, fill=-1)
```

`trajectory_labels` is stored as `DecisionBoundaryMetrics.goal_labels`; out-of-env samples now read `-1` ("no goal") instead of the last goal's label.

> Import-direction note: `decisions.py` already lives in the same package as `segmentation.py` and importing one private helper from a sibling module introduces no cycle (`segmentation.py` does not import `decisions.py`). Verify `uv run python -c "import neurospatial.behavior.decisions"` still imports cleanly after the change.

### Task 2 — Guard every `dt` division against zero / negative / duplicate timestamps

Four sites divide by a per-sample `dt = np.diff(times)` (or a `np.median` of it) with no guard. Add one shared local guard `_positive_dt` next to `_safe_gather` in `segmentation.py`, and apply `validate_finite` on the timestamp array at each public entry before differencing.

**2a. Add `_positive_dt` to `segmentation.py`** (next to `_safe_gather`):

```python
def _positive_dt(times: NDArray[np.float64], *, name: str = "times") -> NDArray[np.float64]:
    """Return strictly-positive consecutive time deltas, or raise.

    Computes ``np.diff(times)`` and rejects any delta that is not strictly
    positive (duplicate or out-of-order timestamps) with an actionable message.
    Use this before dividing distances/displacements by a per-sample ``dt``.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Monotonically increasing timestamps in seconds.
    name : str, optional
        Argument name, used in the error message. Default ``"times"``.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Consecutive time deltas, all ``> 0``.

    Raises
    ------
    ValueError
        If any ``np.diff(times) <= 0`` (duplicate or non-increasing timestamps).
    """
    dt = np.diff(np.asarray(times, dtype=np.float64))
    bad = dt <= 0
    if bad.any():
        n = int(bad.sum())
        first = int(np.argmax(bad))
        raise ValueError(
            f"{name} must be strictly increasing for a per-sample time step, "
            f"but {n} consecutive delta(s) are <= 0 "
            f"(first between index {first} and {first + 1}: "
            f"dt={dt[first]!r}). Sort and de-duplicate timestamps before calling."
        )
    return dt
```

Import `validate_finite` at the top of each affected module:

```python
from neurospatial._validation import validate_finite
```

(Add to `segmentation.py`, `navigation.py`, and `trajectory.py` import blocks. `_positive_dt` lives in `segmentation.py`; import it into `navigation.py` and `trajectory.py` the same way `_safe_gather` is imported into `decisions.py`.)

**2b. `segmentation.py:532-533` — `detect_runs_between_regions` velocity filter.** Replace:

```python
            dt = np.diff(run_times)
            velocities = distances / dt
```

with:

```python
            dt = _positive_dt(run_times, name="times")
            velocities = distances / dt
```

(`run_times` is a contiguous slice of the validated `times`; the guard catches duplicate frames inside a run.) Guard the full array once at the top of the function, right after the existing length check (before line 468's cast):

```python
    validate_finite(times, name="times")
```

**2c. `segmentation.py:670-680` — `segment_by_velocity`.** Replace:

```python
    dt = np.diff(times)
    velocities = distances / dt
```

with:

```python
    dt = _positive_dt(times, name="times")
    velocities = distances / dt
```

Add the finiteness guard near the top of the function, after the existing `positions`/`times` length check (around line 659) and `min_speed`/`hysteresis` checks:

```python
    validate_finite(times, name="times")
```

The `median_dt = np.median(dt)` smoothing-window computation (line 679) is now safe because every `dt` is `> 0`, so `median_dt > 0` and `int(smooth_window / median_dt)` is well-defined.

**2d. `navigation.py:1853-1857` — `approach_rate`.** Replace:

```python
    dt = np.diff(times)
    d_distance = np.diff(distances)

    rates = np.full(len(positions), np.nan)
    rates[1:] = d_distance / dt
```

with:

```python
    dt = _positive_dt(times, name="times")
    d_distance = np.diff(distances)

    rates = np.full(len(positions), np.nan)
    rates[1:] = d_distance / dt
```

Add the finiteness guard near the top of `approach_rate`, after the `metric`/`env` validation (around line 1840), once `times` is known to be the timestamp array:

```python
    validate_finite(times, name="times")
```

`approach_rate` has no `len(positions) != len(times)` check today — add one alongside, since `np.diff` would otherwise broadcast-mismatch silently in the geodesic branch:

```python
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}."
        )
```

**2e. `trajectory.py:613-621` — `mean_square_displacement`.** The lag grid is built from a **median** `dt`, so a single duplicate timestamp is tolerable but an all-duplicate or non-increasing record is not. Replace:

```python
    time_diffs = np.diff(times)
    # Create lag times by accumulating time differences
    # We'll use a simple approach: sample lag times at regular intervals
    # from the time step up to max_tau
    dt = np.median(time_diffs)  # Typical time step
    n_lags = int(max_tau / dt)
    n_lags = max(1, min(n_lags, n_samples // 2))  # At least 1, at most n_samples/2
```

with:

```python
    time_diffs = np.diff(times)
    # Create lag times by accumulating time differences. Use the median delta as
    # the typical time step; reject records whose typical step is not positive
    # (all-duplicate or non-increasing timestamps), which would make the lag grid
    # ill-defined (ZeroDivisionError or negative n_lags).
    dt = float(np.median(time_diffs))  # Typical time step
    if dt <= 0:
        raise ValueError(
            f"times must be increasing: the median consecutive time step is "
            f"{dt!r} (<= 0), so the lag grid is undefined. Sort and "
            f"de-duplicate timestamps before calling."
        )
    n_lags = int(max_tau / dt)
    n_lags = max(1, min(n_lags, n_samples // 2))  # At least 1, at most n_samples/2
```

Add the finiteness guard near the top of `mean_square_displacement`, after the existing `positions.ndim` / length / `metric` validation (around line 604):

```python
    validate_finite(times, name="times")
```

> Why a local `dt <= 0` check here instead of `_positive_dt`? MSD legitimately accepts the occasional duplicate or slightly-jittered timestamp (it uses `np.median` and a `dt/2` matching tolerance, lines 617 & 638). Rejecting *every* non-positive delta would over-constrain it. The guard fires only when the *typical* step collapses to zero/negative.

### Task 3 — Path efficiency: surface unreachable goals, and keep the metric consistent

**3a. `navigation.py:1589-1598` — `compute_path_efficiency` report `inf` / `nan`, not `0.0`, for an unreachable goal.** The `efficiency` field (line 1579) already becomes `np.nan` when `shortest` is non-finite, but `shortest_length` is clamped to `0.0` (line 1591), which reads as "goal at start". Stop clamping. Replace:

```python
    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest if np.isfinite(shortest) else 0.0,
        efficiency=eff,
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=positions[0] if len(positions) > 0 else np.array([]),
        goal_position=goal,
        metric=metric,
    )
```

with:

```python
    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest,  # may be inf if the goal is unreachable
        efficiency=eff,            # already nan when shortest is non-finite
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=positions[0] if len(positions) > 0 else np.array([]),
        goal_position=goal,
        metric=metric,
    )
```

Document the `inf` possibility in the `compute_path_efficiency` docstring `Returns` block and in `PathEfficiencyResult`'s `shortest_length` attribute doc (find the `@dataclass` for `PathEfficiencyResult` above this function — its summary methods at lines 195/212 are part of the same class; verify `is_efficient` / `summary` tolerate `inf`/`nan` without crashing, and if `summary()` formats `shortest_length`, ensure it prints `inf` rather than raising).

**3b. `navigation.py:1581-1585` & `1315-1367` — keep `time_efficiency` on the same metric as the caller.** `compute_path_efficiency(metric="geodesic")` already computed the correct geodesic `shortest` length (line 1575). Don't let `time_efficiency` silently recompute a Euclidean one. Two coordinated edits:

First, give `time_efficiency` an explicit optimal-distance input so it cannot pick a metric on its own. Change its signature and body. Replace lines 1315–1367 (`def time_efficiency` through `return float(optimal_time / actual_time)`) so that the optimal *distance* is passed in rather than recomputed:

```python
def time_efficiency(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    reference_speed: float,
    optimal_distance: float,
) -> float:
    """Compute time efficiency: ratio of optimal to actual travel time.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    reference_speed : float
        Reference speed in environment units per second.
    optimal_distance : float
        Shortest-path distance from start to goal under the caller's chosen
        metric (Euclidean or geodesic). The optimal time is
        ``optimal_distance / reference_speed``. Pass ``np.inf`` for an
        unreachable goal; the result is then ``nan``.

    Returns
    -------
    float
        Time efficiency ratio ``T_optimal / T_actual``. ``nan`` if the
        trajectory has < 2 samples, ``actual_time <= 0``, or
        ``optimal_distance`` is non-finite.

    Examples
    --------
    >>> eff = time_efficiency(
    ...     positions, times, reference_speed=20.0, optimal_distance=50.0
    ... )  # doctest: +SKIP
    >>> print(f"Time efficiency: {eff:.1%}")  # doctest: +SKIP
    """
    if len(positions) < 2:
        return np.nan

    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    if not np.isfinite(optimal_distance):
        return np.nan

    actual_time = times[-1] - times[0]
    if actual_time <= 0:
        return np.nan

    optimal_time = optimal_distance / reference_speed

    return float(optimal_time / actual_time)
```

This removes the `goal` parameter and the internal `np.linalg.norm(goal - start)`, so `time_efficiency` no longer has any hidden metric. Search the codebase for other callers before landing:

```bash
uv run python -c "import subprocess" >/dev/null  # placeholder; use ripgrep below
rg -n 'time_efficiency\(' src tests
```

The only in-package caller is `compute_path_efficiency`. Update its call (lines 1583–1585) to pass the geodesic-or-euclidean `shortest` it already computed:

```python
    time_eff = None
    if reference_speed is not None and len(positions) >= 2:
        time_eff = time_efficiency(
            positions,
            times,
            reference_speed=reference_speed,
            optimal_distance=shortest,
        )
```

Now `time_efficiency` honors whatever `metric` the caller passed, and an unreachable goal (`shortest == inf`) yields `time_eff = nan` instead of a Euclidean fiction.

> If `rg` finds external/test callers of the old `time_efficiency(positions, times, goal, *, reference_speed=...)` signature, update each to the new keyword-only `optimal_distance=` form in this same PR (it is a behavior-fixing breaking change within the same scope, pre-1.0). Do not leave a `goal`-based shim — the whole point is to remove the metric ambiguity.

### Task 4 — Documentation touch-ups (public API)

- `compute_path_efficiency` docstring `Returns`: note `shortest_length` (and therefore `efficiency`/`time_efficiency`) can be `inf`/`nan` when the goal is unreachable under the geodesic metric.
- `time_efficiency` docstring: already rewritten in 3b (parameter set changed — `goal` removed, `optimal_distance` added).
- `approach_rate`, `segment_by_velocity`, `detect_runs_between_regions`, `mean_square_displacement`: add a one-line `Raises`/note that timestamps must be finite and strictly increasing (the new guards), so the behavior is discoverable.

No `CLAUDE.md`/`QUICKSTART.md` snippets reference these functions' internals, so no top-level doc edits are required — but grep to confirm before closing the phase:

```bash
rg -n 'time_efficiency|approach_rate|compute_path_efficiency|mean_square_displacement' .claude README.md
```

---

## Deliberately not in this phase

- **`path_progress` (`navigation.py:483`), `cost_to_goal` (`navigation.py:754`), `compute_vte_session` (`vte.py:593`) "non-runnable doc" bugs.** These have docstring `Examples` that don't execute and/or signature-vs-doc drift. They are documentation-execution correctness, batched into **phase 23** (runnable-docs sweep). Touching them here would mix doc-only edits into a numeric-correctness PR.
- **`env`-placement / `KeyError`-vs-`ValueError` convention drift.** Several behavior functions take `env` in a non-canonical argument position and raise `KeyError` where the project convention is `ValueError` (e.g. region-lookup paths). That API-consistency sweep is **phase 22**; do not re-order parameters or change exception types here, even where the fixed lines sit next to a convention violation.
- **`heading_direction_labels` `min_speed` unit documentation (`navigation.py:1076`).** The `min_speed` default (`5.0`) has ambiguous units (cm/s vs units/s) in its docstring. Units/docs clarification is **phase 22**; this phase does not touch `heading_direction_labels`.
- **Anything outside `src/neurospatial/behavior/`.** The `_validation.py` module is owned by phase 4; if it must be created here as a soft-dependency fallback, copy it verbatim from shared-contracts and do not extend it. `regions_to_mask` (`ops/binning.py`) and `shortest_path_length`'s `inf` semantics are read-only references — do not modify them.
- **Vectorizing the Python `for`-loops** in `detect_runs_between_regions` / `segment_trials`. Tempting while in the file, but it is a performance change with its own correctness surface; out of scope for a correctness-only PR.

---

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_detect_region_crossings_ignores_out_of_env_sample` | A trajectory whose middle sample has `bin == -1` (out of env), with the surrounding samples genuinely outside the region, produces **no** entry/exit crossing — i.e. the `-1` is not wrapped into the last bin's region membership. Fail-before: spurious crossing(s) appear. |
| `test_detect_runs_between_regions_excludes_out_of_env_from_target` | A run that passes through an out-of-env (`-1`) sample whose wrapped last-bin would be in `target` is **not** marked `success=True` on account of that sample. Fail-before: run wrongly succeeds. |
| `test_segment_trials_out_of_env_not_counted_in_end_region` | An out-of-env (`-1`) sample is not treated as entering an `end_region`; trial segmentation matches the same data with the `-1` sample removed. |
| `test_detect_boundary_crossings_minus_one_bin_is_unreachable` | An out-of-env (`-1`) bin yields label `-1` (unreachable) and so does **not** create a boundary crossing with its neighbors. Fail-before: the wrapped last-bin label creates a phantom crossing. |
| `test_compute_decision_analysis_goal_labels_minus_one_for_out_of_env` | `result.boundary.goal_labels` is `-1` at out-of-env samples, not the last goal's label. |
| `test_segment_by_velocity_duplicate_timestamp_raises` | Passing `times` with a duplicated timestamp raises `ValueError` naming `times` (from `_positive_dt`) instead of returning `inf`-velocity segments. |
| `test_detect_runs_between_regions_duplicate_timestamp_raises` | A run containing duplicate frame times raises `ValueError` from the `min_speed` velocity filter rather than dividing by zero. |
| `test_approach_rate_duplicate_timestamp_raises` | Duplicate timestamps raise `ValueError` (naming `times`) instead of producing `inf`/`nan` approach rates. |
| `test_approach_rate_nonfinite_times_raises` | `times` containing `inf` raises `ValueError` from `validate_finite` before any division. |
| `test_mean_square_displacement_all_duplicate_times_raises` | `times` whose median consecutive delta is `0` raises `ValueError` (no `ZeroDivisionError`, no negative `n_lags`). |
| `test_mean_square_displacement_tolerates_single_duplicate` | A record with one duplicated timestamp but a positive median step still computes an MSD curve (the guard is on the *typical* step, not every delta). |
| `test_compute_path_efficiency_unreachable_goal_reports_inf` | For a goal on a disconnected component (`metric="geodesic"`), `result.shortest_length == inf` and `result.efficiency` is `nan` — **not** `shortest_length == 0.0`. Fail-before: `shortest_length == 0.0`. |
| `test_compute_path_efficiency_time_efficiency_is_geodesic` | With `metric="geodesic"` on an environment where the geodesic optimal distance exceeds the Euclidean one (e.g. an L-shaped/obstacle track), `result.time_efficiency` is computed from the geodesic `shortest_length`, not the Euclidean straight line. Fail-before: the Euclidean value. |
| `test_time_efficiency_infinite_optimal_distance_is_nan` | `time_efficiency(..., optimal_distance=np.inf)` returns `nan`. |

Mark none of these `slow` — all run on tiny synthetic fixtures. The geodesic-vs-Euclidean efficiency test needs a small obstacle/L-shaped environment but is still sub-second.

## Fixtures

All synthesized in the test module (or `tests/behavior/conftest.py` if shared across files) — no checked-in data, no real-data slice needed:

- **Out-of-env trajectory:** build a small 2D `Environment.from_samples(positions, bin_size=...)`, then construct a `position_bins` array by hand that includes a `-1` at a chosen index (or place a real sample outside the env bounds and map it with `env.bin_at`, asserting it returns `-1` first, so the test documents *why* the `-1` is there). Pair with a region added via `env.regions` so a mask exists.
- **Voronoi-label fixtures** (`decisions.py` tests): two goal bins → `geodesic_voronoi_labels(env, goal_bins)`; inject an out-of-env `-1` sample into `position_bins`.
- **Duplicate-/non-increasing-timestamp arrays:** start from `np.linspace(0, T, n)` and overwrite one element to equal its predecessor (duplicate) — reused across the `segment_by_velocity`, `detect_runs_between_regions`, `approach_rate`, and `mean_square_displacement` dt tests; factor into a `conftest.py` helper, not copy-pasted.
- **Disconnected / obstacle environment** (path-efficiency tests): the existing `tests/behavior/test_path_efficiency.py` already constructs efficiency-test environments — reuse or extend its fixtures (e.g. an environment with a wall so the geodesic path detours), and add a fully disconnected variant where `shortest_path_length` returns `inf`.

Place new shared synthetic builders in `tests/behavior/conftest.py`; per-test one-offs stay inline. Reuse existing fixtures in `tests/behavior/` rather than re-deriving environments where one already fits.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases (no `path_progress`/`cost_to_goal`/`compute_vte_session` doc fixes, no `env`-placement or exception-type changes, no `heading_direction_labels` edits, no loop vectorization, no edits outside `src/neurospatial/behavior/` except the phase-4-owned `_validation.py` fallback).
- Validation slice tests pass (`uv run pytest tests/behavior tests/segmentation`); none are mis-marked `slow`.
- Tests aren't trivial — they exercise the asserted behavior (a phantom crossing actually disappears; an `inf` is actually surfaced), not tautologies (no `assert True`; no assertions that only re-check the fixture). Shared setup (out-of-env trajectory, duplicate-timestamp arrays, disconnected env) lives in `tests/behavior/conftest.py`, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its phase numbers.
- `_safe_gather` and `_positive_dt` are imported, not re-defined, in `decisions.py` / `navigation.py` / `trajectory.py`; `validate_finite` comes from `neurospatial._validation`, not a private fork.
- No old code path is left orphaned — the Euclidean-only branch inside the old `time_efficiency` is removed (the function no longer takes `goal`), and every caller is updated to the `optimal_distance=` signature.
- `uv run ruff check . && uv run ruff format . && uv run mypy src/neurospatial/behavior/` are clean.
- User-facing docstring updates (Task 4) are present, not deferred.
