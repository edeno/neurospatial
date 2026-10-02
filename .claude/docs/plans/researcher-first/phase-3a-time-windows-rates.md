# Phase 3a — Time windows: shared helpers and the four rate families

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics)

Phase 3 implements the [time-window contract](shared-contracts.md#time-window-semantics) for every analysis family on `main`. It ships as **three PRs** because the inventory below covers about 60 public entry points across five subpackages. A single diff would be about 4–5k lines (code, docstrings and tests), which is too large to review. Each PR is self-contained and revertible:

| PR | Scope | Depends on |
| --- | --- | --- |
| **3a (this file)** | `neurospatial/_intervals.py`, the `interval_valid_mask` extension, `env.occupancy(epochs=)`, one normalizer for `behavior.restrict`/`in_epochs`, and the four spike+position **rate** families (spatial, directional, view, egocentric) plus their predicates, with the population-silence warning | Phases 1–2 |
| [3b](phase-3b-time-windows-decoding-events.md) | Decoding (per-run time bins, `BayesianDecoder`), `bin_spikes_in_time`, and the PETH functions | 3a |
| [3c](phase-3c-time-windows-behavior.md) | Position-only behavior: segmentation, kinematics, `heading_from_velocity`, `env.bin_sequence`/`transitions`, `events.add_positions` | 3a |

3b and 3c are independent of each other.

**Inputs to read first:**

- [src/neurospatial/environment/trajectory.py:90](../../../../src/neurospatial/environment/trajectory.py#L90). This is `interval_valid_mask`, the gate this phase extends. Its body is at lines 158–187: the gap gate at 170–172, the speed gate at 174–177, and the bounds gate at 179–185. Today it requires both `positions` and `env`.
- [src/neurospatial/environment/trajectory.py:197](../../../../src/neurospatial/environment/trajectory.py#L197). This is `Environment.occupancy`. The mask call is at lines 450–457, and the `"start"` allocation (`np.bincount` over `bin_indices[:-1][valid_mask]`) is at 464–475.
- [src/neurospatial/encoding/_binning.py:175](../../../../src/neurospatial/encoding/_binning.py#L175). This is `_bin_spike_train_with_stats`, the spike kernel. The interval gate (`gate_active`, searchsorted, clip to `n-2`) is at lines 291–313; that is the alignment model the other families copy. It also holds:
  - `_resolve_interval_mask` at :339;
  - `_emit_all_excluded_intervals_warning` at :398;
  - `bin_spike_train` at :572;
  - `compute_occupancy` at :723 (which delegates to `env.occupancy` at 818–836);
  - `bin_spike_trains` at :841.
- [src/neurospatial/encoding/spatial.py:2542](../../../../src/neurospatial/encoding/spatial.py#L2542) (`compute_spatial_rate`) and [:3026](../../../../src/neurospatial/encoding/spatial.py#L3026) (`compute_spatial_rates`). These are the only families with `max_gap` on `main`. Note how the mask is resolved for the warning (2896–2912) and then recomputed separately inside the spike and occupancy helpers (2915–2937).
- [src/neurospatial/encoding/spatial.py:4009](../../../../src/neurospatial/encoding/spatial.py#L4009) (`_subset_spikes_by_time_mask`) and [:4111](../../../../src/neurospatial/encoding/spatial.py#L4111) (`compute_directional_place_fields`). This function slices the arrays per label and concatenates the pieces, which creates artificial joins between label segments.
- [src/neurospatial/encoding/_directional_binning.py:44](../../../../src/neurospatial/encoding/_directional_binning.py#L44). This holds `compute_directional_occupancy` (occupancy at 177–197, with no gap gate), `_bin_spikes_with_precomputed_directional_bins` at :246, `bin_directional_spike_train` at :289, and `bin_directional_spike_trains` at :367.
- [src/neurospatial/encoding/_view_binning.py:48](../../../../src/neurospatial/encoding/_view_binning.py#L48). This holds:
  - `_precompute_view_bins`;
  - `_bin_spikes_with_precomputed_view_bins` at :108;
  - `compute_occupancy` at :169 (occupancy block at 295–323);
  - `bin_view_spike_train` at :326;
  - `bin_view_spike_trains` at :475 (occupancy at 596–603).

  None of these has a gap gate.
- [src/neurospatial/encoding/_egocentric_binning.py:375](../../../../src/neurospatial/encoding/_egocentric_binning.py#L375). This holds `compute_egocentric_occupancy` (occupancy at 518–531), `bin_egocentric_spike_train` at :534, and `bin_egocentric_spike_trains` at :700 (occupancy at 851–857; the per-neuron `_bin_single_neuron` closure follows it). None of these has a gap gate.
- [src/neurospatial/encoding/directional.py:1629](../../../../src/neurospatial/encoding/directional.py#L1629), [:1866](../../../../src/neurospatial/encoding/directional.py#L1866) and [:2184](../../../../src/neurospatial/encoding/directional.py#L2184). These are `compute_directional_rate`, `compute_directional_rates` and `is_head_direction_cell`, which forwards at :2281.
- [src/neurospatial/encoding/view.py:1097](../../../../src/neurospatial/encoding/view.py#L1097), [:1373](../../../../src/neurospatial/encoding/view.py#L1373) and [:1735](../../../../src/neurospatial/encoding/view.py#L1735). These are `compute_view_rate`, `compute_view_rates` and `is_spatial_view_cell`, which forwards at :1807.
- [src/neurospatial/encoding/egocentric.py:1311](../../../../src/neurospatial/encoding/egocentric.py#L1311), [:1600](../../../../src/neurospatial/encoding/egocentric.py#L1600) and [:2141](../../../../src/neurospatial/encoding/egocentric.py#L2141). These are `compute_egocentric_rate`, `compute_egocentric_rates` and `is_object_vector_cell`, which forwards at :2230.
- [src/neurospatial/encoding/spatial.py:4450](../../../../src/neurospatial/encoding/spatial.py#L4450). This is `is_place_cell`, which forwards to `compute_spatial_rate` at :4524.
- [src/neurospatial/behavior/epochs.py:84](../../../../src/neurospatial/behavior/epochs.py#L84). This is `_as_intervals`, a **second** interval normalizer with different semantics:
  - zero-width rows are allowed;
  - there is a parallel `(starts, ends)` form;
  - there is an "Ambiguous" error;
  - rows are not merged.

  `in_epochs` (:243), `restrict` (:295) and `restrict_spike_trains` (:376) use it.
- Evidence (session scratchpad, not in the repo): `branch-triage.md` §A2 and `catA/gap.py`. They report the following, using two epochs `[0, 100)` and `[1100, 1200)` s at 50 Hz with a true rate of 5 Hz:

  | Family on `main` | Result |
  | --- | --- |
  | Directional | Occupancy totals 1200 s, the worst bin holds 1003 s, and the minimum rate is 0.019 Hz |
  | View | The worst bin holds 1000 s |
  | Egocentric | 0.024 Hz |
  | Spatial | Already correct, at 4.97–5.00 Hz |

- Archive reference only, never cherry-picked: `aad59e74` (directional), `a0ff995a` (view), `d373cc18` (egocentric). These commits are entangled with `TemporalSupport` and do not apply to `main`.
- **Files earlier phases already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `encoding/directional.py`, `view.py` and `egocentric.py`: Phase 1 Task 7 routes spike-group input through `as_spike_trains_with_ids` in the three plural functions, Phase 1 Task 9 changed the directional `to_xarray` attrs, and Phase 2 Task 2 changed the polar plots. Keep the spike-group routing when adding the new keywords.
  - `tests/conftest.py`: Phase 1 added the `make_spike_group` fixture. Add this phase's fixtures alongside it.
  - `CHANGELOG.md`: Phases 1 and 2 added `### Fixed` bullets under `[Unreleased]`. Add this phase's section after them.

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics). This phase creates the single implementation point (`_intervals.py` and the extended `interval_valid_mask`). It also implements:
  - the spike+position row for the rate families;
  - the boundary rule for the per-sample (frame) counters: a spike exactly at `times[-1]` lies in no interval and is not counted;
  - the visible spike-window assumption (`spike_window` and `spike_window_assumed` on every result);
  - the population-silence warning.

  Do not weaken any rule.
- [Error-message contract](shared-contracts.md#error-message-contract). Every error from `as_intervals` states what, why, and a `Fix:` line. It reports every problem in both arguments at once.
- [Input conventions](shared-contracts.md#input-conventions). The new keywords are keyword-only and follow `max_gap` in the order `max_gap, epochs, spike_window`. The directional raw form stays `(spike_times, times, headings, *, ...)`.

**Designs referenced:** none. No `designs.md` exists for this plan; the algorithms are small enough to give in full below.

## Inventory (this PR's slice)

On `main`, every one of these consumes `times` together with spikes and positions or headings. "Bridges gaps" means an interval with `dt > 0.5 s` is charged to one bin as occupancy and its spikes are counted.

| Function (file:line) | Category | Current gap behavior on `main` | Change in 3a |
| --- | --- | --- | --- |
| `compute_spatial_rate` spatial.py:2542 | spike+position | `max_gap=0.5` handled (shared mask) | add `epochs=None, spike_window=None`; resolve one mask, pass it to both the spike and occupancy helpers |
| `compute_spatial_rates` spatial.py:3026 | spike+position (plural) | handled | same, plus the population-silence warning |
| `is_place_cell` spatial.py:4450 | spike+position | inherits (default `max_gap`, not exposed) | add `max_gap=0.5, epochs=None, spike_window=None`; forward them |
| `compute_directional_place_fields` spatial.py:4111 | spike+position | `max_gap` handled per label slice. Slicing joins label segments: a join shorter than 0.5 s is counted as occupancy, while the spikes between segments are excluded | add `max_gap, epochs, spike_window`; label subsets become **label epochs** passed as `epochs=` (removes `_subset_spikes_by_time_mask`) |
| `compute_directional_rate` directional.py:1629 | spike+position (headings) | **bridges gaps** (no gate; NaN headings only) | add `max_gap=0.5, epochs=None, spike_window=None`; shared mask |
| `compute_directional_rates` directional.py:1866 | spike+position (plural) | **bridges gaps** | same, plus silence warning |
| `is_head_direction_cell` directional.py:2184 | spike+position | **bridges gaps** | add and forward the three keywords |
| `compute_view_rate` view.py:1097 | spike+position | **bridges gaps** (gates only on view-bin validity) | add the three keywords; shared mask with `start_bin=view_bins` |
| `compute_view_rates` view.py:1373 | spike+position (plural) | **bridges gaps** | same, plus silence warning |
| `is_spatial_view_cell` view.py:1735 | spike+position | **bridges gaps** | add and forward |
| `compute_egocentric_rate` egocentric.py:1311 | spike+position | **bridges gaps** (gates only on polar-bin validity) | add the three keywords; shared mask with `start_bin=polar_bins` |
| `compute_egocentric_rates` egocentric.py:1600 | spike+position (plural) | **bridges gaps** | same, plus silence warning |
| `is_object_vector_cell` egocentric.py:2141 | spike+position | **bridges gaps** | add and forward |
| `Environment.occupancy` trajectory.py:197 | position-only | `max_gap` handled | add `epochs=None` |
| `behavior.in_epochs` / `restrict` / `restrict_spike_trains` epochs.py:243/295/376 | epoch utilities | n/a | normalize through `_intervals.as_intervals` (the second normalizer is removed) |

Phase 3 totals across 3a–3c, from reading the code on this branch:

- **Spike+position: 19.** Thirteen are in 3a. Six are in 3b: `decode_session`, `decode_session_summary`, and `BayesianDecoder.fit`/`predict`/`predict_summary`/`score`.
- **Spike-only: 7.**
  - Three change in 3b: `peri_event_histogram`, `population_peri_event_histogram` and `bin_spikes_in_time`.
  - Four are deferred, with a stated reason (see 3b and the overview's Open Questions): `align_spikes_to_events`, `time_to_nearest_event`, `event_count_in_window` and `event_indicator`.
- **Position-only: 39.**
  - One changes in 3a (`env.occupancy`) and 29 change in 3c.
  - Nine need no change, with stated reasons (see 3c):
    - five label mappers;
    - `mean_square_displacement`;
    - `time_efficiency`;
    - `decision_region_entry_time`;
    - `distance_to_reward`.
  - `trajectory_similarity` takes no `times`, so it is not counted.
- **Not applicable: phase precession.** `phase_precession` and `has_phase_precession` (`encoding/phase_precession.py`) take per-spike `positions` and `phases` and no `times` array, so there is no interval to gate.

## Tasks

### 1. `src/neurospatial/_intervals.py` (new, private)

This module is the single parser and containment test for `epochs` and `spike_window`. It also provides two helpers that later PRs reuse: run extraction and interval intersection. Write it exactly as follows (formatting may change under ruff).

```python
"""Time-window normalization and containment tests.

Every analysis that combines data streams restricts itself to time inside the
optional ``epochs`` and ``spike_window`` arguments. This module is the single
place those arguments are parsed, merged and tested, so every analysis family
applies identical semantics: each row is a half-open ``[start, stop)`` window in
seconds, on the same clock as ``times``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_ACCEPTED_FORMS = (
    "None, a (start, stop) pair, an (n, 2) array-like of [start, stop) rows, "
    "or an object with 1-D .start and .end arrays (for example a pynapple "
    "IntervalSet)"
)


def _format_rows(idx: NDArray[np.intp]) -> str:
    """Format row indices as ``"0, 3, 7"``, truncated after five entries."""
    shown = ", ".join(str(int(i)) for i in idx[:5])
    return shown + (f" (+{idx.size - 5} more)" if idx.size > 5 else "")


def _parse_intervals(
    value: Any, name: str
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Convert ``value`` to an unsorted ``(n, 2)`` array and list every problem.

    Returns ``(rows, problems)``. ``rows`` is ``None`` only when the shape is
    unusable; ``problems`` is empty when ``rows`` is valid.
    """
    if hasattr(value, "start") and hasattr(value, "end"):
        starts = np.asarray(value.start, dtype=np.float64)
        stops = np.asarray(value.end, dtype=np.float64)
        if starts.ndim != 1 or stops.shape != starts.shape:
            return None, [
                f"{name}.start and {name}.end must be 1-D arrays of equal "
                f"length, got shapes {starts.shape} and {stops.shape}"
            ]
        rows = np.column_stack([starts, stops])
    else:
        try:
            rows = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None, [
                f"{name} could not be read as numbers "
                f"(got {type(value).__name__}: {value!r:.80})"
            ]
        if rows.shape == (2,):
            rows = rows.reshape(1, 2)
        if rows.ndim != 2 or rows.shape[1] != 2:
            return None, [f"{name} has shape {rows.shape}; expected (2,) or (n, 2)"]
    if rows.shape[0] == 0:
        return None, [f"{name} has no rows, so it would exclude all data"]

    problems: list[str] = []
    finite = np.isfinite(rows).all(axis=1)
    nonfinite = np.flatnonzero(~finite)
    if nonfinite.size:
        problems.append(
            f"{name} row(s) {_format_rows(nonfinite)} contain NaN or inf "
            f"(e.g. {rows[nonfinite[0]].tolist()})"
        )
    inverted = np.flatnonzero(finite & (rows[:, 1] <= rows[:, 0]))
    if inverted.size:
        problems.append(
            f"{name} row(s) {_format_rows(inverted)} have stop <= start "
            f"(e.g. {rows[inverted[0]].tolist()})"
        )
    return rows, problems


def _merge(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sort rows by start and merge overlapping or touching rows (vectorized)."""
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    running_stop = np.maximum.accumulate(rows[:, 1])
    opens_new = np.ones(rows.shape[0], dtype=bool)
    opens_new[1:] = rows[1:, 0] > running_stop[:-1]
    first = np.flatnonzero(opens_new)
    return np.column_stack([rows[first, 0], np.maximum.reduceat(rows[:, 1], first)])


def _raise_invalid(problems: list[str]) -> None:
    raise ValueError(
        "Invalid time window: "
        + "; ".join(problems)
        + ".\nWhy: epochs and spike_window are half-open [start, stop) windows "
        "in seconds on the same clock as `times`; every row needs a finite "
        f"start < stop. Accepted forms: {_ACCEPTED_FORMS}.\n"
        "Fix: pass e.g. epochs=[(0.0, 100.0), (1100.0, 1200.0)] or "
        "spike_window=(0.0, 1200.0), or None for no restriction."
    )


def as_intervals(value: Any, *, name: str) -> NDArray[np.float64] | None:
    """Normalize a time-window argument to sorted, merged ``(n, 2)`` rows.

    Parameters
    ----------
    value : None, (start, stop), array-like of shape (n, 2), or IntervalSet-like
        The window(s). An object exposing 1-D ``.start`` and ``.end`` arrays
        (for example a pynapple ``IntervalSet``) is accepted by duck typing;
        pynapple is never imported.
    name : str
        Argument name used in error messages (``"epochs"``, ``"spike_window"``).

    Returns
    -------
    ndarray of float64, shape (n_windows, 2), or None
        Rows sorted by start, with overlapping or touching rows merged.
        ``None`` when ``value`` is ``None``.

    Raises
    ------
    ValueError
        On a wrong shape, zero rows, non-finite values or ``stop <= start``.
        Every problem is listed in one message.
    """
    if value is None:
        return None
    rows, problems = _parse_intervals(value, name)
    if problems:
        _raise_invalid(problems)
    assert rows is not None  # _parse_intervals returns rows when no problems
    return _merge(rows)


def resolve_time_windows(
    epochs: Any, spike_window: Any
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Normalize ``epochs`` and ``spike_window`` together.

    Problems in both arguments are reported in a single ``ValueError``.
    """
    resolved: list[NDArray[np.float64] | None] = []
    problems: list[str] = []
    for name, value in (("epochs", epochs), ("spike_window", spike_window)):
        if value is None:
            resolved.append(None)
            continue
        rows, found = _parse_intervals(value, name)
        problems.extend(found)
        resolved.append(None if found or rows is None else _merge(rows))
    if problems:
        _raise_invalid(problems)
    return resolved[0], resolved[1]


def intervals_contain(
    windows: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Test whether each ``[starts[i], stops[i])`` lies inside one window row.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, merged rows (the output of :func:`as_intervals`).
    starts, stops : ndarray, shape (n,)
        Query intervals.

    Returns
    -------
    ndarray of bool, shape (n,)
        ``True`` iff ``windows[j, 0] <= starts[i]`` and
        ``stops[i] <= windows[j, 1]`` for a single row ``j``. NaN queries are
        ``False``.
    """
    starts = np.asarray(starts, dtype=np.float64)
    stops = np.asarray(stops, dtype=np.float64)
    if windows.shape[0] == 0:
        return np.zeros(starts.shape, dtype=bool)
    idx = np.searchsorted(windows[:, 0], starts, side="right") - 1
    return (idx >= 0) & (stops <= windows[np.maximum(idx, 0), 1])


def intersect_intervals(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Intersect two sorted, merged interval sets (vectorized).

    Returns
    -------
    ndarray, shape (n, 2)
        Sorted, disjoint rows; shape ``(0, 2)`` when the sets do not overlap.
    """
    lo = np.searchsorted(b[:, 1], a[:, 0], side="right")  # first b row ending after a_start
    hi = np.searchsorted(b[:, 0], a[:, 1], side="left")  # b rows starting before a_stop
    counts = np.maximum(hi - lo, 0)
    a_idx = np.repeat(np.arange(a.shape[0]), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    b_idx = np.repeat(lo, counts) + offsets
    starts = np.maximum(a[a_idx, 0], b[b_idx, 0])
    stops = np.minimum(a[a_idx, 1], b[b_idx, 1])
    keep = stops > starts
    return np.column_stack([starts[keep], stops[keep]])


def run_sample_bounds(interval_mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Return the first and last sample of every maximal run of valid intervals.

    Parameters
    ----------
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Per-interval validity (``interval_valid_mask`` output).

    Returns
    -------
    ndarray of intp, shape (n_runs, 2)
        Row ``r`` is ``(first_sample, last_sample)``, inclusive, so the run's
        samples are ``slice(first_sample, last_sample + 1)``. A sample whose
        two neighbouring intervals are both invalid belongs to no run.
    """
    padded = np.concatenate([[False], np.asarray(interval_mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return np.column_stack([edges[0::2], edges[1::2]]).astype(np.intp)


def run_time_bounds(
    times: NDArray[np.float64], interval_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Return ``[times[first], times[last]]`` for every maximal valid run.

    Returns
    -------
    ndarray, shape (n_runs, 2)
        Half-open ``[start, stop)`` time windows covering exactly the valid
        intervals.
    """
    bounds = run_sample_bounds(interval_mask)
    return np.column_stack([times[bounds[:, 0]], times[bounds[:, 1]]])
```

Check `run_sample_bounds`: for `mask=[T, T, F, T]`, the padded array is `[F, T, T, F, T, F]`, the edges are `[0, 2, 3, 4]`, and the rows are `[(0, 2), (3, 4)]`. That is samples 0–2 and samples 3–4. Correct.

### 2. Extend `interval_valid_mask` (trajectory.py:90)

The changes, as specified by the contract's [single implementation point](shared-contracts.md#time-window-semantics):

- `positions` and `env` become optional (`None`).
- The bounds gate runs only when `start_bin` or `env` is given; the speed gate only when `speed` is given.
- Two keyword-only arguments, `epochs` and `spike_window`, are added. Both are already normalized (`as_intervals` output) or `None`.
- Update the docstring math block. Interval `k` is valid iff, additionally, `[t_k, t_{k+1}) ⊆ epochs` (when given) and `⊆ spike_window` (when given).

Replace the signature and body (lines 90–100 and 158–187) with:

```python
def interval_valid_mask(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    env: EnvironmentProtocol | None = None,
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    start_bin: NDArray[np.intp] | None = None,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
) -> NDArray[np.bool_]:
    # docstring: as today, plus the epochs/spike_window parameters and the rule
    # that the bounds gate needs start_bin, or env together with positions.
    from neurospatial._intervals import intervals_contain

    times = np.asarray(times, dtype=np.float64)
    n_samples = len(times)
    if n_samples < 2:
        return np.zeros(max(n_samples - 1, 0), dtype=bool)

    dt = np.diff(times)
    valid_mask = np.ones(len(dt), dtype=bool)
    if max_gap is not None:
        valid_mask &= dt <= max_gap
    if min_speed is not None and speed is not None:
        valid_mask &= np.asarray(speed, dtype=np.float64)[:-1] >= min_speed
    if start_bin is None and env is not None:
        pos = np.asarray(positions, dtype=np.float64)
        start_bin = env.bin_at(pos.reshape(-1, 1) if pos.ndim == 1 else pos)
    if start_bin is not None:
        valid_mask &= np.asarray(start_bin)[:-1] >= 0
    for windows in (epochs, spike_window):
        if windows is not None:
            valid_mask &= intervals_contain(windows, times[:-1], times[1:])
    return valid_mask
```

Also add one module-level helper to trajectory.py. It is the single implementation of start-allocated occupancy, and `env.occupancy`, the spatial encoders and the frame-binned families all use it:

```python
def start_allocated_occupancy(
    start_bin: NDArray[np.intp],
    dt: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    n_bins: int,
    *,
    return_seconds: bool = True,
) -> NDArray[np.float64]:
    """Sum each valid interval's duration into the bin of its start sample.

    Parameters
    ----------
    start_bin : ndarray of intp, shape (n_samples,)
        Bin of every sample (``-1`` = invalid; such intervals must already be
        excluded by ``interval_mask``).
    dt : ndarray, shape (n_samples - 1,)
        ``np.diff(times)``.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        The shared validity mask.
    n_bins : int
        Number of bins.
    return_seconds : bool, default=True
        Weight by ``dt`` (seconds) or count intervals.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    bins = start_bin[:-1][interval_mask]
    weights = dt[interval_mask] if return_seconds else None
    return np.bincount(bins, weights=weights, minlength=n_bins)[:n_bins].astype(np.float64)
```

`Environment.occupancy` changes as follows:

- It gains keyword-only `epochs=None`, placed after `max_gap`.
- It normalizes the argument with `as_intervals(epochs, name="epochs")` and passes it to `interval_valid_mask` (lines 450–457).
- It replaces the `"start"` branch (464–475) with `start_allocated_occupancy(bin_indices, dt, valid_mask, self.n_bins, return_seconds=return_seconds)`.
- The `"linear"` branch already receives `valid_mask`, so it needs no change.

`env.occupancy` is position-only per the contract, so it does **not** take `spike_window`. Spike-rate denominators come from Task 3 instead.

### 3. Spatial encoders: one mask, applied to the numerator and the denominator

In `encoding/_binning.py`:

- **`_resolve_interval_mask` (:339).**
  - It gains `epochs` and `spike_window` (normalized) and forwards them.
  - It now **always** returns an array: it calls `interval_valid_mask(..., start_bin=env.bin_at(positions))` even when every gate is off.
  - This removes the `None`/`gate_active` path (lines 382–384). That path let spikes in out-of-bounds-start intervals reach the numerator when `max_gap=None`, while `env.occupancy` dropped those intervals from the denominator. The asymmetry exists on `main` today. CHANGELOG it.
- **`_bin_spike_train_with_stats` (:175).**
  - It takes `interval_mask` as **required** and drops `speed`/`min_speed`/`max_gap`.
  - The gate block (291–313) becomes unconditional: searchsorted, then index the mask.
  - **Boundary rule.** The time filter at 267–268 becomes `(spike_times >= times[0]) & (spike_times < times[-1])`, and the clip to `n-2` (and its comment) is removed. A spike at `t < times[-1]` always has `searchsorted(times, t, "right") - 1 <= n - 2`. Interval `k` is `[times[k], times[k+1])`, so a spike exactly at `times[-1]` lies in no interval. On `main` it is gated by the last interval and counted: `times = [0, 0.5, 1]` with a spike at `1.0` counts 1 (probe, `bin_spike_train`). It now counts as time-dropped (`n_time_dropped`).
  - The fallback that recomputes the mask (lines 298–309) is removed.
- **`bin_spike_train` (:572), `compute_occupancy` (:723) and `bin_spike_trains` (:841).**
  - Each keeps `speed`/`min_speed`/`max_gap` and adds `epochs=None, spike_window=None, interval_mask=None`.
  - When `interval_mask` is `None`, each calls `_resolve_interval_mask` once.
  - `compute_occupancy` stops delegating to `env.occupancy` (818–836). It returns `start_allocated_occupancy(env.bin_at(positions), np.diff(times), interval_mask, env.n_bins)`.
  - `bin_spike_trains` resolves the mask once and uses it for both the occupancy and every per-neuron count. Its line-978 call already does this for the counts.
- **`_emit_all_excluded_intervals_warning` (:398).**
  - It gains `epochs` and `spike_window`, and names `epochs` and/or `spike_window` among the active gates.
  - It adds the fix: "check that epochs/spike_window overlap `times` (same clock, seconds)".
  - It is now always given an array, so its `interval_mask is None` early return goes.

In `compute_spatial_rate` (spatial.py:2542):

- Add the keyword-only parameters `epochs=None, spike_window=None` after `max_gap`.
- After `resolve_speed`, call `E, S = resolve_time_windows(epochs, spike_window)`.
- Compute `interval_mask = _resolve_interval_mask(env, times, positions_2d, speed=..., min_speed=..., max_gap=..., epochs=E, spike_window=S)` **once**. It is no longer under `if warn_on_drop:`; only the warning call is.
- Pass `interval_mask=` to both `bin_spike_train` and `compute_occupancy`.

`compute_spatial_rates` (spatial.py:3026) changes the same way. Its four helper calls are at 3474, 3484, 3549 and 3580, and all receive the same mask. Then emit the population-silence warning (Task 6).

`is_place_cell` (spatial.py:4450) gains `max_gap=0.5, epochs=None, spike_window=None` and forwards them at :4524.

`compute_directional_place_fields` (spatial.py:4111):

- Add `max_gap=0.5, epochs=None, spike_window=None`.
- Replace the per-label slicing (the loop body that calls `_subset_spikes_by_time_mask` and `positions[mask]`) with **label epochs**. Interval `k` carries the label of its start sample:

  ```python
  E, S = resolve_time_windows(epochs, spike_window)
  for label in unique_labels:
      label_mask = labels_arr[:-1] == label  # per interval
      label_windows = run_time_bounds(times, label_mask)
      if label_windows.shape[0] == 0:
          continue  # label only on the final sample: no interval to analyze
      windows = label_windows if E is None else intersect_intervals(label_windows, E)
      if windows.shape[0] == 0:
          continue
      single = compute_spatial_rate(
          env, spike_times, times, positions,
          method=method, bandwidth=bandwidth, min_occupancy=min_occupancy,
          max_gap=max_gap, epochs=windows, spike_window=S,
      )
  ```

- **Remove `_subset_spikes_by_time_mask` (spatial.py:4009)** and its direct tests. That is the old code path; nothing else in `src/` uses it.

### 4. One frame-binning kernel for the directional, view and egocentric families

All three families assign every sample a bin (`frame_bins`, with `-1` for invalid). Each charges interval `k` to `frame_bins[k]` and looks up spikes at the most recent frame. Today each family carries its own copy of this logic: 6 occupancy blocks and 4 spike-lookup helpers. Replace them all with one kernel in `encoding/_binning.py`, next to `start_allocated_occupancy`:

```python
def count_spikes_by_frame(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    frame_bins: NDArray[np.intp],
    interval_mask: NDArray[np.bool_],
    n_bins: int,
) -> NDArray[np.float64]:
    """Count spikes per bin from the most recent frame, gated by interval validity.

    Interval ``k`` is the half-open ``[times[k], times[k+1])``. A spike at ``t``
    lies in interval ``frame = searchsorted(times, t, "right") - 1``, takes that
    frame's bin, and is kept iff the interval is valid and
    ``frame_bins[frame] >= 0``. A spike before ``times[0]`` or at or after
    ``times[-1]`` lies in no interval and is not counted. Interval validity is
    the SAME mask used for the occupancy denominator
    (``start_allocated_occupancy``), so numerator and denominator drop identical
    intervals.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    spikes = spike_times[(spike_times >= times[0]) & (spike_times < times[-1])]
    frame = np.searchsorted(times, spikes, side="right") - 1  # <= n_samples - 2
    bins = frame_bins[frame]
    keep = interval_mask[frame] & (bins >= 0)
    return np.bincount(bins[keep], minlength=n_bins).astype(np.float64)
```

Per family:

- **Directional (`_directional_binning.py`).**
  - Add `directional_frame_bins(headings, bin_size, *, angle_unit) -> (frame_bins, bin_centers)`. It is built from `_precompute_directional_bins` (:200) as follows:
    - `np.digitize(wrapped, bin_edges) - 1` on finite headings;
    - `>= n_bins → 0`;
    - non-finite headings → `-1`.
  - `compute_directional_occupancy` (:44) keeps its validation (lines 115–175) and gains keyword-only `max_gap, epochs, spike_window`. It then builds the mask and calls `start_allocated_occupancy(frame_bins, np.diff(times), mask, n_bins)`, where the mask is:

    ```python
    interval_valid_mask(times, start_bin=frame_bins, max_gap=max_gap,
                        epochs=epochs, spike_window=spike_window)
    ```

  - `bin_directional_spike_train(s)` (:289, :367) gain the same keywords and call `count_spikes_by_frame`.
  - **Remove `_bin_spikes_with_precomputed_directional_bins` (:246).** It is replaced by `count_spikes_by_frame`.
  - The NaN-heading exclusion is preserved, because `-1` frames fail the bounds gate.
- **View (`_view_binning.py`).**
  - `compute_occupancy` (:169), `bin_view_spike_train` (:326) and `bin_view_spike_trains` (:475) gain `max_gap, epochs, spike_window`.
  - After `_precompute_view_bins`, each builds `mask = interval_valid_mask(times, start_bin=view_bins, ...)` and uses `start_allocated_occupancy` together with `count_spikes_by_frame`.
  - **Remove `_bin_spikes_with_precomputed_view_bins` (:108)** and the two hand-written occupancy blocks (295–323 and 596–603).
- **Egocentric (`_egocentric_binning.py`).**
  - `compute_egocentric_occupancy` (:375), `bin_egocentric_spike_train` (:534) and `bin_egocentric_spike_trains` (:700) change the same way, with `start_bin=bin_indices` (the polar bins).
  - **Remove the inline occupancy blocks (518–531 and 851–857) and the `_bin_single_neuron` closure in `bin_egocentric_spike_trains`.**
  - `n_jobs` keeps its meaning: the joblib map calls `count_spikes_by_frame` per neuron.

Public functions:

- `compute_directional_rate(s)`, `compute_view_rate(s)` and `compute_egocentric_rate(s)` gain the keyword-only parameters `max_gap: float | None = 0.5, epochs=None, spike_window=None`, placed after `bin_size`/`view_distance`/`n_direction_bins` respectively and before `method`.
- Each calls `resolve_time_windows` once and threads the normalized arrays down.
- The predicates `is_head_direction_cell`, `is_spatial_view_cell` and `is_object_vector_cell` gain the same three parameters and forward them (at :2281, :1807 and :2230).
- In every family, a spike exactly at `times[-1]` is no longer counted (the contract's boundary rule). On `main` it is counted by the spatial and directional kernels (probe: `times = [0, 0.5, 1]`, spike at `1.0` → 1 count from `bin_spike_train` and from `bin_directional_spike_train`), and by the egocentric kernel without any gate. This snippet was probed: on 3000 frames with 5000 random spikes (none exactly at `times[-1]`) it matches the old `<=`/clip version exactly, and it differs only for a spike at `times[-1]`.

### 5. One normalizer for `behavior.restrict` / `in_epochs` / `restrict_spike_trains`

`behavior/epochs.py` changes as follows:

- **Delete `_as_intervals` (:84) and `_stack_start_end`/`_sequence_length`.**
- `in_epochs`, `restrict` and `restrict_spike_trains` call `as_intervals(epochs, name="epochs")`.
- `_mask_in_intervals` keeps its `closed=` point semantics, because these functions test *points*, not intervals.

The documented consequences, all of which go in the CHANGELOG:

- The parallel `(starts, ends)` two-array form is no longer accepted. Use an `(n, 2)` array or an `IntervalSet`. The "Ambiguous" error disappears, because `[[0, 5], [10, 15]]` now always means two rows.
- Zero-width rows (`start == stop`) are rejected.
- Overlapping rows are merged, which does not change the result for point masks.

Update `tests/behavior/test_epochs.py` accordingly:

- Delete the parallel-arrays and "Ambiguous" tests (around :52–:95).
- Keep the `(n, 2)`, the IntervalSet, and the `closed=` tests.

This is the only place the plan changes `restrict`'s behaviour. Phase 6 removes `restrict` from the root namespace (it stays in `behavior`) and deletes `Session`.

### 6. Population-silence warning (plural spike+position functions)

Add `_warn_if_population_silent` to `encoding/_binning.py` next to the other warnings:

```python
_SILENCE_MIN_UNITS = 5
_SILENCE_MIN_SECONDS = 60.0


def _warn_if_population_silent(
    spike_trains: Sequence[NDArray[np.float64]],
    observed_runs: NDArray[np.float64],
    *,
    stacklevel: int = 3,
) -> None:
    """Warn once if every unit is silent for >= 60 s of tracked time.

    ``observed_runs`` are the maximal runs of intervals passing the max_gap and
    epochs gates (``run_time_bounds``). A silent stretch is measured inside a
    single run, from the run start to the first spike, between consecutive
    spikes of the merged train, and from the last spike to the run end, so a
    stretch never spans an untracked pause.
    """
    n_units = len(spike_trains)
    if n_units < _SILENCE_MIN_UNITS or observed_runs.shape[0] == 0:
        return
    nonempty = [np.asarray(s, dtype=np.float64) for s in spike_trains if len(s)]
    spikes = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)
    run_idx = np.searchsorted(observed_runs[:, 0], spikes, side="right") - 1
    inside = run_idx >= 0
    inside[inside] = spikes[inside] < observed_runs[run_idx[inside], 1]
    n_runs = observed_runs.shape[0]
    marks = np.concatenate([observed_runs[:, 0], observed_runs[:, 1], spikes[inside]])
    owner = np.concatenate([np.arange(n_runs), np.arange(n_runs), run_idx[inside]])
    order = np.lexsort((marks, owner))  # by run, then time
    marks, owner = marks[order], owner[order]
    silence = np.where(owner[1:] == owner[:-1], np.diff(marks), -np.inf)
    k = int(np.argmax(silence))
    if silence[k] < _SILENCE_MIN_SECONDS:
        return
    a, b = float(marks[k]), float(marks[k + 1])
    warnings.warn(
        f"All {n_units} units are silent from {a:.1f} s to {b:.1f} s "
        f"({b - a:.0f} s) while position is tracked. If the electrophysiology "
        f"was not recording then, pass spike_window=(start, stop) so that time "
        f"is excluded from occupancy.",
        UserWarning,
        stacklevel=stacklevel,
    )
```

Call it in `compute_spatial_rates`, `compute_directional_rates`, `compute_view_rates` and `compute_egocentric_rates`. Each call goes after the spike trains and times are validated, and **only when `spike_window is None`**: an explicit spike window means the caller has already stated when ephys was recording. The observed runs are:

```python
run_time_bounds(times, interval_valid_mask(times, max_gap=max_gap, epochs=E))
```

They exclude the speed and bounds gates, because a population silent while the animal sits still may also mean that ephys was off. The warning is a heuristic for that one common mistake: it never confirms that recording covered the analyzed time, and its absence is not evidence of coverage. Do not call it in the singular functions or the predicates (the contract scopes the warning to populations). The decoders inherit it through `compute_spatial_rates` (3b).

### 7. Results record the spike window

The default `spike_window=None` is an assumption, not a finding, so every result says which one applied ([time-window semantics, Defaults](shared-contracts.md#time-window-semantics)).

- **Field.** Each of the eight rate result classes gains `spike_window: NDArray[np.float64] | None = field(default=None, kw_only=True, compare=False)`. The classes are `SpatialRateResult` and `SpatialRatesResult` (`spatial.py:500, :1188`), `DirectionalRateResult` and `DirectionalRatesResult` (`directional.py:134, :959`), `ViewRateResult` and `ViewRatesResult` (`view.py:109, :437`), and `EgocentricRateResult` and `EgocentricRatesResult` (`egocentric.py:108, :508`). The field holds the normalized `(n, 2)` rows actually applied (the `S` from `resolve_time_windows`), or `None`.
- **Property.** `SpatialResultMixin` (`_base.py:162`) gains a read-only property `spike_window_assumed -> bool`, which returns `self.spike_window is None`. A property rather than a second field means the two can never disagree. Its docstring: "True when no `spike_window` was passed, so spikes were assumed recorded wherever position was. The population-silence warning catches one common violation of this assumption; it cannot establish recording coverage."
- **Who sets it.**
  - Every `compute_*_rate(s)` passes `spike_window=S`. So do the predicates' internal compute calls and each per-label result of `compute_directional_place_fields`, which records the caller's `S`; label windows are `epochs`, not `spike_window`.
  - `__getitem__`/`__iter__` on the four plural classes (`spatial.py:1430`, `directional.py:1155`, `view.py:637`, `egocentric.py:713`) pass it to the child.
- **`summary()`.** `SpatialResultMixin.summary` (`_base.py:333`) adds `"spike_window_assumed"` (bool) and `"spike_window"` (`self.spike_window.tolist()`, or `None`).
- **`to_xarray()` attrs.** The four plural `attrs` dicts (`spatial.py:1697`, `directional.py:1131`, `view.py:595`, `egocentric.py:670`) add two entries:
  - `attrs["spike_window_assumed"] = int(self.spike_window_assumed)`;
  - `attrs["spike_window"] = self.spike_window.ravel()` (flat `[start0, stop0, start1, stop1, ...]`), only when `spike_window` is not None.

  NetCDF has no bool or None. A probe with the scipy engine showed a `bool` attr reading back as `int8(1)`, a `None` attr raising `TypeError`, and a flat float64 array round-tripping exactly. Phase 7 moves these dicts into one `_xarray_attrs()` per family, unchanged.

### 8. Public docstrings for every touched public function (own task)

The functions:

- `compute_spatial_rate`, `compute_spatial_rates`, `is_place_cell` and `compute_directional_place_fields`;
- `compute_directional_rate`, `compute_directional_rates` and `is_head_direction_cell`;
- `compute_view_rate`, `compute_view_rates` and `is_spatial_view_cell`;
- `compute_egocentric_rate`, `compute_egocentric_rates` and `is_object_vector_cell`;
- `Environment.occupancy`, `in_epochs`, `restrict` and `restrict_spike_trains`.

Each gets NumPy-style `Parameters` entries for the new keywords. Use this text verbatim for all three (shorten only for position-only functions, which have no `spike_window`):

```text
max_gap : float or None, default=0.5
    Longest sampling interval (seconds) treated as continuous recording.
    Longer intervals (dropped frames, pauses between sessions) are excluded
    from occupancy and their spikes are not counted. ``None`` disables the
    gap check.
epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
    Restrict the analysis to these half-open [start, stop) windows (seconds,
    same clock as ``times``). An interval counts only if it lies entirely
    inside one window. ``None`` (default) means unrestricted.
spike_window : same forms as ``epochs``, or None
    When the electrophysiology was recording. Intervals outside it are
    excluded from occupancy (and their spikes are not counted). ``None``
    (default) assumes spikes were recorded whenever position was; this is an
    assumption, not something the function checks. Pass it when tracking
    started before, or continued after, the spike recording. The result
    records the window applied (``result.spike_window``) and whether it was
    assumed (``result.spike_window_assumed``).
```

The plural functions also get a `Warns` entry for the population-silence warning. It must call the warning a heuristic: it fires only when at least 5 units are all silent for at least 60 s, so it cannot detect an outage for a single unit, or one shorter than 60 s, and its silence is not proof that recording coverage is correct. `Notes` gains one paragraph: "An interval is analyzed only if it passes the gap, speed and bounds checks and lies inside `epochs ∩ spike_window`. The same intervals are removed from the spike counts and the occupancy." Docstrings must not mention plans or phases.

### 9. CHANGELOG, README and quickstart (own task)

- **`CHANGELOG.md` `[Unreleased]`.** Add one "Changed — recording gaps and time windows in rate maps" section. It lists:
  - the bug fixed: directional, view and egocentric rates charged a recording pause to one bin (for example 0.02 Hz instead of 5 Hz);
  - the new keywords;
  - the `max_gap=None` alignment fix;
  - the label-epoch change in `compute_directional_place_fields`;
  - the `restrict`/`in_epochs` form changes;
  - the population-silence warning;
  - `result.spike_window` and `result.spike_window_assumed` in `summary()` and `to_xarray().attrs`;
  - **behavior change:** a spike exactly at the last position timestamp `times[-1]` is no longer counted by any rate family. On `main` the spatial and directional kernels counted it, gated by the last interval, and the egocentric kernel counted it ungated.
- **`README.md`.** Directly after the "Your First Place Field" example (its closing paragraph ends just before `## Core Concepts`), add a subsection "### Recording gaps and time windows" of at most 12 lines, with a 4-line code block. It covers:
  - Gaps longer than `max_gap=0.5` s are detected from `times` and excluded automatically.
  - `epochs=` restricts the analysis to chosen windows.
  - `spike_window=` covers ephys that started late or stopped early. Without it, spikes are *assumed* recorded whenever position was, and `result.spike_window_assumed` says so.
  - An example `compute_spatial_rate(env, spike_times, times, positions, epochs=run_epochs, spike_window=(t_ephys_start, t_ephys_stop))`.
- **`docs/getting-started/quickstart.md`.** Add the same paragraph, without the code block, as a fourth bold item at the end of "## What just happened", before `## Next steps`. Phase 4 executes both snippets, so they must run.

## Deliberately not in this phase

- **Decoding, `bin_spikes_in_time`, PETH.** These belong to 3b, which needs the per-run time-bin helper and result-field changes. Do not touch `decoding/` or `events/`.
- **Segmentation, behavior kinematics, `heading_from_velocity`, `env.bin_sequence`/`transitions`, `events.add_positions`.** These belong to 3c.
- **New gating knobs** (for example `min_speed` for the view, directional or egocentric families). The contract adds only `epochs`/`spike_window` and reuses `max_gap`.
- **Argument reordering or renaming** (for example the behavior `(times, positions)` order, or removing `Session`). Phase 6 owns that.
- **Inferring gaps from non-finite positions.** The contract's gap detector is `max_gap` on `times`. NaN positions keep today's per-family handling.
- **Performance work.** The kernel replacement must not regress runtime; see the benchmark row in the validation slice.
- **Phase precession.** Its API has no `times` (see inventory).

## Validation slice

The fixture `two_epoch_recording` is described under Fixtures. "Pooled rate" means `Σ firing_rate·occupancy / Σ occupancy` over bins with occupancy > 0, using `method="binned"` and no smoothing, so it equals `Σ counts / Σ occupancy`.

| Test | Asserts |
| --- | --- |
| `tests/test_intervals.py::test_as_intervals_accepted_forms` | `(0, 10)` → `[[0, 10]]`. `[[20, 30], [0, 10]]` → sorted. `[[0, 10], [10, 20]]` → `[[0, 20]]` (touching rows merge). `[[0, 10], [5, 15]]` → `[[0, 15]]`. `None` → `None`. |
| `tests/test_intervals.py::test_as_intervals_duck_types_intervalset` | A plain class with `.start = np.array([20., 0.])` and `.end = np.array([30., 10.])` → `[[0, 10], [20, 30]]`. No pynapple import (assert `"pynapple" not in sys.modules` after the call, when it was absent before). |
| `tests/test_intervals.py::test_as_intervals_real_intervalset` (`@pytest.mark.pynapple`) | `nap.IntervalSet(start=[0, 20], end=[10, 30])` gives the same rows. |
| `tests/test_intervals.py::test_as_intervals_errors_follow_contract` | Parametrized over: shape `(3,)`; zero rows `np.empty((0, 2))`; `[[0, np.nan]]`; `[[5, 5]]`; an object with mismatched `.start`/`.end`. Each raises `ValueError` whose message names the argument, quotes the offending row or shape, and contains `"Why:"` and a line starting `"Fix:"`. |
| `tests/test_intervals.py::test_as_intervals_reports_every_problem` | `[[1, 0], [np.nan, 1]]` → one error naming row 0 (`stop <= start`) and row 1 (`NaN`). `resolve_time_windows([[1, 0]], [[np.inf, 2]])` → one error naming both `epochs` and `spike_window`. |
| `tests/test_intervals.py::test_intervals_contain` | Windows `[[0, 10], [20, 30]]` with queries `[0,10)`, `[5,10)`, `[9,11)`, `[10,20)`, `[-1,0)`, `[20,30)`, `[25,31)`, `[29.5,30)`, `[nan,1)` → `[T, T, F, F, F, T, F, T, F]`. |
| `tests/test_intervals.py::test_intersect_intervals` | `[[0,10],[20,30]] ∩ [[5,25]]` → `[[5,10],[20,25]]`; disjoint inputs → shape `(0, 2)`. A property check on 200 random merged sets (seed 0) agrees with a brute-force point-sampling reference. |
| `tests/test_intervals.py::test_run_bounds` | `run_sample_bounds([T,T,F,T])` → `[[0,2],[3,4]]`; all-False → shape `(0, 2)`. `run_time_bounds` returns matching times. |
| `tests/environment/test_occupancy.py::test_occupancy_epochs_restricts_and_matches_slicing` | On a continuous 200 s, 50 Hz grid, `env.occupancy(t, p, epochs=[(0., 100.)])` equals `env.occupancy(t[t <= 100], p[t <= 100])` with `rtol=1e-12`. The sum is `100.0 ± 1e-9`. |
| `tests/environment/test_interval_valid_mask_windows.py` | With `epochs=[(0,1)]` on `times=[0, .5, 1, 1.5]` → `[T, T, F]`. With `spike_window` alone the result is the same. With both `[(0,1)]` and `[(0.5,2)]` → `[F, T, F]`. `env=None` with `start_bin=[0,-1,0,0]` → `[T, F, T]`. With neither, only the gap gate applies. |
| `tests/encoding/test_recording_gaps.py::test_rate_family_recovers_true_rate_across_pause` | Parametrized over spatial (binned), directional (`bandwidth=None`), view (`fixed_distance`, `view_distance=5`) and egocentric (`object (50,50)`, `distance_range=(0,100)`), each in singular and plural form. Asserts: pooled rate = `5.0 ± 5%` (true value 1000 spikes / 199.96 s = 5.001 Hz); total occupancy ≤ 199.96 s, and for directional/egocentric `== 199.96 ± 1e-6`; max per-bin occupancy < 10 s. **Fails on `main`** for directional (total 1200 s, worst bin ≈1003 s), view (worst bin ≈1000 s) and egocentric (pooled ≈0.8 Hz). Passes on `main` for spatial, as a regression guard. |
| `tests/encoding/test_recording_gaps.py::test_spatial_all_methods_recover_rate` (`slow`) | `compute_spatial_rate` with `method ∈ {diffusion_kde, gaussian_kde, glm}` on the fixture: total occupancy `199.96 ± 1e-6`; `Σ rate·occ / Σ occ = 5 ± 10%` (smoothing redistributes mass). |
| `tests/encoding/test_recording_gaps.py::test_predicates_forward_time_windows` | For each `is_*_cell`, a call with `epochs=[(0, 100)]` uses only the first epoch. Spy on the underlying `compute_*` (via `monkeypatch` wrapping, which records kwargs) and assert that `epochs`, `spike_window` and `max_gap` arrive unchanged. |
| `tests/encoding/test_time_windows.py::test_epochs_equal_slicing` | Parametrized over the four families. On a continuous 200 s recording, `epochs=[(0., 100.)]` gives `firing_rate` and `occupancy` equal (`rtol=1e-12`, `equal_nan=True`) to slicing samples to `times <= 100` and spikes to `< 100`. Uses `method="binned"` and `"diffusion_kde"` for spatial. |
| `tests/encoding/test_time_windows.py::test_spike_window_restores_true_rate` | Tracking covers `[0, 200)` s; 5 units fire regularly at 5 Hz only in `[100, 200)`, unit `u` offset by `0.04·u` s. For each plural family: with no `spike_window`, the pooled rate is `2.5 ± 5%` and a `UserWarning` matching `r"All 5 units are silent from 0\.0 s to 100\.\d s"` fires. With `spike_window=(100., 200.)`, the pooled rate is `5.0 ± 5%` and no warning fires (`warnings.simplefilter("error")`). |
| `tests/encoding/test_time_windows.py::test_population_silence_thresholds` | No warning with: 5 units and a clean 200 s recording; 4 units and a 100 s silence; 5 units and a 59 s silence. A warning, exactly once, with 5 units and a 61 s silence in the middle of the recording. No warning when the silence spans an untracked pause: on `two_epoch_recording`, all 5 units are silent over `[70, 100)` and `[1100, 1131)` (61 s of tracked time in total, but only 30 s and 31 s within each run). No warning when `spike_window` is passed. |
| `tests/encoding/test_time_windows.py::test_epochs_and_spike_window_errors` | `compute_spatial_rate(..., epochs=[[5, 1]], spike_window="bad")` raises one `ValueError` that names both arguments. |
| `tests/encoding/test_time_windows.py::test_all_excluded_warning_names_epochs` | `epochs=[(5000., 6000.)]`, which lies outside `times`, → the existing all-intervals-excluded warning, now naming `epochs`. |
| `tests/encoding/test_directional_place_fields.py::test_label_epochs_do_not_join_segments` | Labels `A,A,other,A` with a 0.3 s `other` gap: occupancy for `A` equals the sum of the `A` intervals only. On `main`, slicing counted the joined 0.3 s interval toward `A`, so the old result is `+0.3 s`. |
| `tests/encoding/test_interval_mask_alignment.py::test_spike_at_last_sample_not_counted` | Parametrized over `bin_spike_train` (spatial), `bin_directional_spike_train`, `bin_view_spike_train` and `bin_egocentric_spike_train`, plus `count_spikes_by_frame` directly. With `times = [0, 0.5, 1.0]` (intervals `[0, 0.5)` and `[0.5, 1.0)`) and spikes `[0.5, 1.0]`, the total count is 1: the spike at 0.5 is counted and the one at 1.0 is not. `main` counts 2 (probe: spatial and directional each count the 1.0 spike; old frame kernel `[0, 1, 1]`, new `[0, 1, 0]`). |
| `tests/encoding/test_time_windows.py::test_results_record_spike_window` | Parametrized over the four families, singular and plural. By default, `result.spike_window is None`, `result.spike_window_assumed is True`, and `summary()` has `spike_window_assumed=True` and `spike_window=None`. With `spike_window=(100., 200.)`, `result.spike_window` array-equals `[[100., 200.]]`, `spike_window_assumed is False`, and `summary()["spike_window"] == [[100.0, 200.0]]`. `rates[0].spike_window` equals the parent's. Constructing a result directly gives `spike_window_assumed is True`. |
| `tests/encoding/test_spatial_xarray_interop.py::test_spike_window_attrs_roundtrip` | Runs in the `test_xarray.yml` job. Plural spatial and directional with `spike_window=(100., 200.)`: `attrs["spike_window_assumed"] == 0` and `attrs["spike_window"]` equals `[100., 200.]`. With the default, `attrs["spike_window_assumed"] == 1` and there is no `spike_window` key. Both round-trip through scipy-engine `to_netcdf` and `load_dataset` with equal values. |
| `tests/encoding/test_interval_mask_alignment.py::test_max_gap_none_drops_out_of_bounds_spikes` | With `max_gap=None`, a spike inside an interval whose start sample is out of bounds is not counted, matching the occupancy. On `main` it was counted. |
| `tests/behavior/test_epochs.py` (updated) | `(n, 2)`, IntervalSet, `closed=` semantics are unchanged. A parallel `(starts, ends)` input now reads as rows, or raises on a shape mismatch. A zero-width row raises the contract error. |
| `tests/benchmarks/` (`benchmark`, existing marker) | `compute_view_rates`, `compute_egocentric_rates` and `compute_directional_rates` on 50 units × 60 s at 50 Hz are no slower than the baseline recorded before the change (record both numbers in the PR description). |

## Fixtures

Add these to `tests/conftest.py` (session scope) so that 3b and 3c reuse them. Build all timestamps as `np.arange(n) / fs` so that grid points such as 100.0 are exact.

- **`two_epoch_recording`.** A frozen dataclass with:
  - `times = np.r_[np.arange(5000) / 50, 1100 + np.arange(5000) / 50]` (`[0, 100)` and `[1100, 1200)` s at 50 Hz);
  - `positions = np.c_[50 + 40*np.sin(times/3.1), 50 + 40*np.cos(times/4.7)]`;
  - `headings = np.random.default_rng(0).uniform(-π, π, times.size)`;
  - `spike_times = np.r_[np.arange(0.1, 100, 0.2), np.arange(1100.1, 1200, 0.2)]`, which is exactly 1000 spikes at a regular 5 Hz, so the pooled rate is deterministic;
  - `env = Environment.from_samples(positions, bin_size=5.0)` with `units="cm"`.

  The same geometry as the audit repro (`catA/gap.py`).
- **`continuous_recording`.** The same construction over `[0, 200)` s at 50 Hz (`np.arange(10000) / 50`), with regular 5 Hz spikes. Used by the epochs-equivalence, spike-window and silence tests.
- **Population variants.** These are built inside the tests from the fixtures above by offsetting the spike train per unit (`+0.04·u` s). No new data files.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also confirm:

- Every family obtains its mask from `interval_valid_mask` exactly once per call, and applies that one array to both the counts and the occupancy. `grep -n "np.diff(times)" src/neurospatial/encoding/_*binning.py` shows no hand-rolled occupancy accumulation.
- `scientific-code-change-audit` has been run on the kernel replacement. Outputs on gap-free data must be unchanged: `rtol=1e-12` against `main` for all four families on `continuous_recording`, except the documented `max_gap=None` out-of-bounds alignment fix and a spike exactly at `times[-1]`. `continuous_recording` has none: its last spike is 199.9 s and `times[-1]` is 199.98 s.
