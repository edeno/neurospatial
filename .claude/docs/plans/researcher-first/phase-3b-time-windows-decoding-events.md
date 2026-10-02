# Phase 3b — Time windows: decoding and peri-event analysis

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics) · [3a](phase-3a-time-windows-rates.md) · [3c](phase-3c-time-windows-behavior.md)

This is the second of three Phase 3 PRs (see the split table in [3a](phase-3a-time-windows-rates.md)). It needs 3a's `neurospatial/_intervals.py`, the extended `interval_valid_mask` and the encoder keywords. It does not depend on 3c.

**Inputs to read first:**

- [phase-3a-time-windows-rates.md](phase-3a-time-windows-rates.md), Tasks 1–2. This is the `_intervals` API that 3b uses: `resolve_time_windows`, `as_intervals`, `intervals_contain`, `run_time_bounds`, and `interval_valid_mask(times, *, max_gap, epochs, spike_window)` with no env.
- [src/neurospatial/decoding/session.py:92](../../../../src/neurospatial/decoding/session.py#L92). `decode_session`.
  - [:368](../../../../src/neurospatial/decoding/session.py#L368) is `_build_encoding_model`. Its global decode grid is at lines 594–605: `n_time = floor((t_stop - t_start)/dt + 1e-9)`, then `edges` and `bin_centers` over the whole span, so the grid **invents bins inside recording gaps**.
  - [:610](../../../../src/neurospatial/decoding/session.py#L610) is `_encode_and_bin`, with a global `np.histogram` at 672–674.
  - [:679](../../../../src/neurospatial/decoding/session.py#L679) is `decode_session_summary`. Its streamed block loop starts at :887; it slices global `edges`, special-cases the right-closed last bin, and has two `RuntimeError` guards for edge drift.
- [src/neurospatial/decoding/_binning.py:64](../../../../src/neurospatial/decoding/_binning.py#L64). `bin_spikes_in_time` holds a second copy of the same grid math (lines 190–200, `np.histogram` with a right-closed last bin).
- [src/neurospatial/decoding/estimator.py:278](../../../../src/neurospatial/decoding/estimator.py#L278). This is `BayesianDecoder.fit`, with `epoch=` at :286. Its body slices the arrays with `behavior.restrict`/`restrict_spike_trains` at lines 353–371.
  - `predict` (:404) does not forward `self.max_gap`.
  - The other methods are `predict_summary` (:448) and `score` (:498).
- [src/neurospatial/decoding/_result.py:219](../../../../src/neurospatial/decoding/_result.py#L219). `DecodingResult.plot` uses `imshow` with `extent=[times[0], times[-1]]` (lines 305–330), which assumes uniformly spaced bins.
- [src/neurospatial/events/alignment.py:201](../../../../src/neurospatial/events/alignment.py#L201) is `peri_event_histogram` (`n_events = len(event_times)` at :300; the mean over events is at :325). [:343](../../../../src/neurospatial/events/alignment.py#L343) is `population_peri_event_histogram` (`n_events` at :456; the mean is at :493). Neither has a notion of epochs. Each normalizes by every event, even when the event's window runs into unrecorded time.
- [src/neurospatial/events/_core.py:30](../../../../src/neurospatial/events/_core.py#L30) is `PeriEventResult` (fields at 77–84; `summary` at :138; `plot` at :187). [:208](../../../../src/neurospatial/events/_core.py#L208) is `PopulationPeriEventResult` (fields at 267–278; `__getitem__` builds a `PeriEventResult` at :351; `summary` at :492; `plot` builds one at :560).
- Evidence (session scratchpad): `branch-triage.md` §A2. Its repro uses two epochs, `[0, 100)` and `[1100, 1200)`.
  - **Decoding:** 4,000 of 4,799 decode bins fall inside the unrecorded gap, all with a finite MAP.
  - **PETH:** windows that cross the gap are normalized as if fully observed. A flat 10 Hz rate reads as 0 Hz in the late bins.
- Archive reference only: `c95789c8` and `0d1a89f4` (decode grid), and `4505b564` (PETH). Do not cherry-pick them; they depend on `TemporalSupport`.
- **Files earlier phases already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `decoding/estimator.py`: Phase 1 Task 6 added `_unit_ids_supplied` and `_align_to_fitted_units`, which `predict` and `predict_summary` call on their spike input. Keep that call when adding `epochs`/`spike_window`; it runs before decoding.
  - `decoding/_result.py`: Phase 1 Task 12 made `DecodingResult` a frozen dataclass.
    - **Public constructor:** `__post_init__` always copies `posterior` and `times` into read-only arrays.
    - **Internal path:** `decode_position` builds its result through the private `_from_owned_posterior`, which keeps the posterior it just allocated without copying and copies only the small metadata arrays.
    - **This phase:** Task 4's plot change does not touch this. Task 6 adds the `spike_window` field, which is copied on both paths, and the private `_evolve` method, so `decode_session` can attach metadata without a second posterior allocation.
  - `decoding/posterior.py`: Phase 1 Task 2 changed the prior handling. This phase does not edit it.
  - `events/alignment.py`: Phase 2 Task 8 made `population_peri_event_histogram` normalize its input with `as_spike_trains_with_ids`.
  - `CHANGELOG.md`: append after the Phase 1, 2 and 3a sections.

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics) defines the rules this phase implements:
  - **Decoding:** time bins are formed separately within each maximal run of valid intervals, and spikes outside valid runs are not counted.
  - **Boundary rule:** bin edges never exceed their window, and there is no absolute epsilon in the bin count.
  - **Spike only:** an event is kept iff `[event + window[0], event + window[1]) ⊆ epochs ∩ spike_window`.
  - **Visible assumption:** decoding results record `spike_window` and `spike_window_assumed`.
- [Error-message contract](shared-contracts.md#error-message-contract) covers the new errors: no decode bin fits; all events dropped; `epochs` combined with `t_start`. Each states what, why and a `Fix:` line. A PETH error says "peri-event".
- [Input conventions](shared-contracts.md#input-conventions): the new arguments are keyword-only.

**Designs referenced:** none.

## Inventory (this PR's slice)

| Function (file:line) | Category | Current gap behavior on `main` | Change in 3b |
| --- | --- | --- | --- |
| `decode_session` session.py:92 | spike+position | encoding is gap-aware (`max_gap`); the **decode grid spans the whole recording**, so bins inside a pause get a posterior | add `epochs=None, spike_window=None`; per-run time bins |
| `decode_session_summary` session.py:679 | spike+position | same as above | same; streamed counting uses per-run bins |
| `BayesianDecoder.fit` estimator.py:278 | spike+position | `epoch=` slices the arrays, which concatenates epochs; `max_gap` then drops the joins | `epoch=` is **replaced** by `epochs=None, spike_window=None`, routed through the mask; the slicing branch is removed |
| `BayesianDecoder.predict` / `predict_summary` estimator.py:404/448 | spike + times (decode) | global grid; does not forward `self.max_gap` | add `epochs`, `spike_window`; forward `max_gap=self.max_gap`; per-run bins |
| `BayesianDecoder.score` estimator.py:498 | spike+position | inherits `predict` | add `epochs`, `spike_window`; forward |
| `bin_spikes_in_time` _binning.py:64 | spike only (primitive) | single `[t_start, t_stop)` grid | add `epochs=None`: bins tile each epoch row separately; shares the decoder's binning helpers |
| `peri_event_histogram` alignment.py:201 | spike only | every event is normalized, even when its window leaves the recording | add `epochs=None, spike_window=None`; drop events whose window is not contained; `n_events_dropped` on the result |
| `population_peri_event_histogram` alignment.py:343 | spike only | same as above | same |
| `align_spikes_to_events` alignment.py:68 | spike only | per-event list, no normalization | **deferred** (see below) |
| `time_to_nearest_event`, `event_count_in_window`, `event_indicator` (events/regressors.py:28/223/360) | spike only (regressors) | per-sample values, computed across gaps | **deferred** (overview Open Questions; see below) |

## Tasks

### 1. Per-run decode time bins (one implementation for decoders and `bin_spikes_in_time`)

Add two private helpers to `src/neurospatial/decoding/_binning.py`. They replace both copies of the global grid (session.py:594–605 and _binning.py:190–200) and every `np.histogram` counting site (session.py:672–674, the streamed loop from :887, and _binning.py:197).

```python
def time_bins_in_windows(
    windows: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tile each ``[start, stop)`` window with whole bins of width ``dt``.

    Each window is tiled independently, so no bin spans the space between two
    windows. A remainder shorter than one bin at the end of a window is
    dropped, because the decoder's Poisson likelihood assumes every bin is
    ``dt`` long. A window that is a whole number of bins up to float rounding
    keeps its last bin, whose right edge is clamped to exactly ``stop``.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, disjoint windows (seconds).
    dt : float
        Bin width (seconds), already validated.

    Returns
    -------
    left, right : ndarray, shape (n_time_bins,)
        Bin edges, ``left = start + dt * k``. Within a window
        ``right[i] == left[i + 1]`` exactly (same ``k``), and every
        ``right <= stop`` of its window.
    """
    start, stop = windows[:, 0], windows[:, 1]
    # Rounding error of the edges is a few ulp of the timestamps' magnitude.
    # If that is not small against dt, the edges cannot be represented to bin
    # precision, so refuse instead of inventing or merging bins.
    rounding = 4.0 * np.spacing(np.maximum(np.abs(start), np.abs(stop)))
    imprecise = rounding > 1e-2 * dt
    if imprecise.any():
        w = int(np.flatnonzero(imprecise)[0])
        raise ValueError(
            f"Time bins of dt={dt:g} s cannot be represented at timestamps near "
            f"{start[w]:.6g} s: float64 rounding there is {rounding[w]:.3g} s, more "
            f"than 1% of dt.\n"
            f"Fix: subtract a time origin first (e.g. times - times[0], and the same "
            f"offset from spike times and windows), or use a larger dt."
        )
    ratio = (stop - start) / dt
    # A shortfall below the rounding allowance is rounding, not a partial bin.
    # After the precision check the allowance is at most ~0.01, so it can
    # never add a whole bin. A final bin may therefore be up to 1% short; it
    # is kept and clamped to the window stop.
    slack = rounding / dt + 4.0 * np.finfo(np.float64).eps * ratio
    n_per = np.maximum(np.floor(ratio + slack), 0).astype(np.int64)
    window_idx = np.repeat(np.arange(windows.shape[0]), n_per)
    k = np.arange(int(n_per.sum())) - np.repeat(np.cumsum(n_per) - n_per, n_per)
    left = start[window_idx] + dt * k
    right = np.minimum(start[window_idx] + dt * (k + 1), stop[window_idx])
    return left, right


def count_spikes_in_time_bins(
    spike_trains: Sequence[NDArray[np.float64]],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Count spikes per half-open bin ``[left[i], right[i])``.

    Spikes that fall between bins (outside every window) are not counted.

    Returns
    -------
    ndarray of int64, shape (n_time_bins, n_neurons)
    """
    counts = np.zeros((left.size, len(spike_trains)), dtype=np.int64)
    if left.size == 0:
        return counts
    for unit, train in enumerate(spike_trains):
        s = np.asarray(train, dtype=np.float64)
        idx = np.searchsorted(left, s, side="right") - 1
        inside = idx >= 0
        inside[inside] = s[inside] < right[idx[inside]]
        counts[:, unit] = np.bincount(idx[inside], minlength=left.size)
    return counts
```

**Semantics change: every bin is half-open, and no bin exceeds its window**, as the contract's [Decoding rule and Boundary rule](shared-contracts.md#time-window-semantics) require. Two `main` behaviors are gone everywhere:

- `np.histogram` closed the final bin on the right, so a spike exactly on the last edge was counted.
- The old `floor((b - a)/dt + 1e-9)` decided bin counts with an absolute epsilon.

A probe of the old helper and the snippet above shows the difference:

| Case | Old helper | `time_bins_in_windows` above |
| --- | --- | --- |
| `[0.1, 0.3)`, `dt=0.1`, spike at 0.3 | last bin `[0.2, 0.30000000000000004)`, so the spike **is counted** | bins exactly `[[0.1, 0.2], [0.2, 0.3]]`; the spike is not counted |
| `[1e9+0.1, 1e9+0.3)`, `dt=0.1` | **1 bin**: the quotient is `1.99999928`, and `1e-9` cannot absorb an error of `7e-7` | 2 bins, last right edge `== 1e9+0.3` |
| 20,000 random windows that are whole multiples (`t0 ∈ {0, 1e3, 1e6, 1e9}`, `dt ∈ {1, 2, 10, 25, 100, 200, 300}` ms, 1–2000 bins) | 3,404 with a right edge past `stop` or the wrong bin count | 0 past `stop`, 0 wrong counts, 0 spikes at `stop` counted |
| two-epoch fixture runs `[0, 99.98]`, `[1100, 1199.98]`, `dt=0.025` | 7998 | 7998 (unchanged) |
| `[0, 1.05)`, `dt=0.25` | 4 bins | 4 bins; the 0.05 s remainder is dropped |

Counting full versus in 1000-bin blocks (scoping each train to `[left[start], right[stop-1])`) gave identical counts on 7998 bins.

- Update the `bin_spikes_in_time` docstring Notes, which currently say "the last bin is closed on the right". Replace that with: bins are half-open `[left, right)`, a trailing remainder shorter than `dt` is dropped, and a spike exactly at `t_stop` (or at any window stop) is not counted.
- Update any test that relies on the right-closed edge. `tests/decoding/test_spike_binning.py` has the related edge-case test near :233–:245.
- Add a CHANGELOG line.

`bin_spikes_in_time(spike_trains, dt, t_start=None, t_stop=None, *, epochs=None, orient=...)`:

- If `epochs is not None` and either `t_start` or `t_stop` is given, raise this `ValueError`:

  > bin_spikes_in_time got both epochs and t_start/t_stop. Why: epochs already defines where bins are formed. Fix: pass either `epochs=[(start, stop), ...]` or `t_start=..., t_stop=...`, not both.

- `windows = as_intervals(epochs, name="epochs")` when given; otherwise `windows = np.array([[t_start, t_stop]])` after the existing defaulting and validation (lines 180–195).
- `left, right = time_bins_in_windows(windows, dt)`. Keep the existing "span smaller than one bin" error when `left.size == 0`, worded for windows.
- Count with `count_spikes_in_time_bins`; set `bin_centers = left + dt / 2`.

### 2. `decode_session` / `decode_session_summary` form bins per valid run

Changes to `_build_encoding_model` (session.py:368):

- It gains `epochs` and `spike_window`, both already normalized.
- It returns `(trains, firing_rates, bin_left, bin_right)` instead of `(trains, firing_rates, n_time, edges, bin_centers)`.
- In the encoding branch, it passes `max_gap`, `epochs=E` and `spike_window=S` to `compute_spatial_rates`. That call inherits the population-silence warning from 3a; do not warn twice.
- The decode grid replaces lines 594–605:

```python
from neurospatial._intervals import run_time_bounds
from neurospatial.decoding._binning import time_bins_in_windows
from neurospatial.environment.trajectory import interval_valid_mask

# Decode bins exist only where the recording was observed: the gap, epochs and
# spike_window gates. The speed and out-of-bounds gates restrict only the
# ENCODING step, so periods of immobility (for example replay) are still
# decoded.
observed = interval_valid_mask(
    times_arr, max_gap=max_gap, epochs=epochs, spike_window=spike_window
)
runs = run_time_bounds(times_arr, observed)
bin_left, bin_right = time_bins_in_windows(runs, dt)
if bin_left.size == 0:
    longest = float(np.max(np.diff(runs, axis=1), initial=0.0))
    raise ValueError(
        f"No decode time bin fits: the {runs.shape[0]} observed recording "
        f"run(s) are at most {longest:.3g} s long, shorter than dt={dt}. "
        f"Why: time bins are formed only inside runs of samples with gaps "
        f"<= max_gap={max_gap} s that lie inside epochs and spike_window. "
        f"Fix: use a smaller dt, widen epochs/spike_window, or pass a larger "
        f"max_gap (max_gap=None decodes across gaps)."
    )
```

Everything downstream follows from the new grid:

- **`decode_session` (via `_encode_and_bin`).** `counts = count_spikes_in_time_bins(trains, bin_left, bin_right)` replaces the `np.histogram` stack (672–674), and `centers = bin_left + dt / 2`.
- **`decode_session_summary`.**
  - `n_time = bin_left.size`.
  - The block loop (from :887) slices `bin_left[start:stop]` and `bin_right[start:stop]`, scopes each train to `[bin_left[start], bin_right[stop - 1])`, and counts with `count_spikes_in_time_bins`.
  - **Remove the right-edge special case and the two `RuntimeError` drift guards.** Bins are half-open and come from one array, so they cannot double-count or drift.
  - The 2-D prior check keeps using `n_time`.
- **Public signatures.** `decode_session` and `decode_session_summary` gain the keyword-only `epochs=None, spike_window=None` after `max_gap`. Each calls `resolve_time_windows` once.
- **Out-of-window warning.** `_warn_if_spikes_out_of_window` (session.py:33) keeps the `[times.min(), times.max()]` units check. It is unchanged.

The decode grid uses only the gap, `epochs` and `spike_window` gates, as the [contract's Decoding rule](shared-contracts.md#time-window-semantics) specifies: speed and bounds restrict encoding only, so rest and replay periods are still decoded, and `predict` (which has no positions) uses the same rule.

### 3. `BayesianDecoder` routes every call through the same windows

- **`fit` (estimator.py:278).**
  - Replace `epoch: Any = None` with `epochs: Any = None, spike_window: Any = None`.
  - **Remove the `restrict`/`restrict_spike_trains` branch (lines 353–371).** Pass the normalized windows to `_build_encoding_model`.
  - Restricting through the mask drops the intervals that straddle an epoch boundary. Slicing used to concatenate the epochs, after which `max_gap` dropped the join.
  - Update the example at :342 to `epochs=(0.0, 60.0)`.
- **`predict`, `predict_summary` and `score`.** Each gains keyword-only `epochs=None, spike_window=None` and forwards `max_gap=self.max_gap`; `predict` did not forward `max_gap` before. The docstrings state that `times` are the tracking timestamps: decode bins tile each run of `times` with gaps no longer than `max_gap`. To decode a span without tracking samples, pass `times=np.arange(t0, t1, dt)`.

### 4. `DecodingResult.plot` with non-contiguous bins

In `_result.py` (lines 305–330), when `times` is set and the bins are non-contiguous (`np.any(np.diff(times) > 1.5 * np.min(np.diff(times)))`), plot the columns against the bin index. A time `extent` would visually compress each pause. Then:

- draw a dashed vertical line (`ax.axvline`) at each run break;
- set the x label to `"Time bin (dashed lines: recording gaps)"`.

Contiguous results keep today's time `extent`, with no change.

### 5. Peri-event histograms keep only fully observed windows

In `events/alignment.py`, add this private helper:

```python
def _keep_observed_events(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], int]:
    """Drop events whose window [e + w0, e + w1) is not inside epochs ∩ spike_window.

    Non-finite events are kept so the existing NaN/inf validation in
    ``align_spikes_to_events`` still raises on them.
    """
    keep = np.ones(event_times.shape, dtype=bool)
    finite = np.isfinite(event_times)
    for windows in (epochs, spike_window):
        if windows is not None:
            keep[finite] &= intervals_contain(
                windows, event_times[finite] + window[0], event_times[finite] + window[1]
            )
    n_dropped = int(event_times.size - keep.sum())
    if n_dropped == event_times.size:
        raise ValueError(
            f"All {event_times.size} events were dropped from the peri-event "
            f"histogram: no event's window [event{window[0]:+g} s, "
            f"event{window[1]:+g} s) lies entirely inside epochs ∩ spike_window. "
            f"Why: a window that reaches unrecorded time would be averaged as "
            f"if no spikes occurred there. Fix: check that event_times, epochs "
            f"and spike_window share one clock (seconds), or narrow `window`."
        )
    return event_times[keep], n_dropped
```

Changes to the two PETH functions:

- `peri_event_histogram` and `population_peri_event_histogram` gain keyword-only `epochs=None, spike_window=None`.
- After the empty-events check and before `n_events = len(event_times)` (:300 and :456), each runs:

  ```python
  E, S = resolve_time_windows(epochs, spike_window)
  event_times, n_dropped = _keep_observed_events(event_times, window, E, S)
  ```

- `n_events` is the number of kept events, so the mean and SEM normalize by observed events only.

Changes to the result classes (`_core.py`):

- **Fields.** `PeriEventResult` gains `n_events_dropped: int = 0`, after `unit_id`. `PopulationPeriEventResult` gains `n_events_dropped: int = field(default=0, compare=False)`, after `unit_table`.
- **Builders.** `__getitem__` (:351) and `plot` (:560) pass the count through.
- **Summaries.** Both `summary()` methods (:138, :492) report `n_events_dropped`.
- **Docstrings.** Update the Attributes docstrings.

There is no warning on a partial drop. The count is on the result, and the contract forbids warning merely because gaps exist.

With both arguments `None` there is nothing to filter against. Spike-only analyses have no position stream, so the "spikes observed wherever position is" default cannot apply. Document this in both docstrings: "pass `spike_window` (or `epochs`) when the recording has gaps or edges inside your event windows."

### 6. Decoding results record the spike window

Decoding is a spike + position analysis, so its results carry the same visible assumption as 3a's rate results ([time-window semantics, Defaults](shared-contracts.md#time-window-semantics)).

- **Fields.** `DecodingResult` (`_result.py:25`, frozen since Phase 1) and `DecodingSummary` (`_result.py:733`, already frozen) each gain `spike_window: NDArray[np.float64] | None = field(default=None, kw_only=True, compare=False)`. `DecodingResult.__post_init__` passes a non-None `spike_window` through Phase 1's `_read_only_copy`. Phase 1's trusted `_from_owned_posterior` does the same: it copies `spike_window` and `times`, which are small, and never `posterior`.
- **Changing fields without copying the posterior.** The public `dataclasses.replace` re-runs `__post_init__`, so it copies the posterior. That is correct for callers, but it would add a second full posterior allocation inside the library. Add a private method that reuses the result's own posterior through the trusted path:

  ```python
  def _evolve(self, **changes: Any) -> "DecodingResult":
      """Return a copy with ``changes`` applied, sharing this result's posterior.

      Internal use only. ``self.posterior`` is already a read-only array this
      result owns, so sharing it with the new result cannot create a writable alias.
      """
      if "posterior" in changes:
          raise ValueError("_evolve never replaces the posterior; use dataclasses.replace.")
      fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "posterior"}
      fields.update(changes)
      return type(self)._from_owned_posterior(self.posterior, **fields)
  ```
- **Property.** Each class gets a read-only property `spike_window_assumed -> bool`, which returns `self.spike_window is None`. Its docstring is 3a's text: an assumption, which the population-silence warning cannot verify.
- **Who sets it.**
  - `decode_session` returns `decode_position(...)._evolve(spike_window=S)`. That is one posterior allocation for the whole call, made by `decode_position`. Do not use `dataclasses.replace` here, because it copies.
  - `decode_session_summary` passes `spike_window=S` to its `DecodingSummary(...)` (`session.py:961`).
  - `BayesianDecoder.predict`, `predict_summary` and `score` inherit this through those functions.
  - `decode_position` and `decode_position_summary` take precomputed counts, which carry no spike-window information. They leave the default `None`, which reads as "assumed": nothing restricted the counts.
- **`summary()` and `to_xarray()`.** Both classes' `summary()` (`_result.py:365, :880`) add `"spike_window_assumed"` and `"spike_window"` (`.tolist()` or `None`). Both `to_xarray()` `attrs` dicts (`:576`, `:1030`) add `spike_window_assumed` as `int` and, when it is not None, `spike_window` as a flat float64 array. This is 3a's encoding: NetCDF has no bool or None.

### 7. Public docstrings for every touched public function (own task)

- **Functions to cover:** `decode_session`, `decode_session_summary`, `BayesianDecoder.fit`/`predict`/`predict_summary`/`score`, `bin_spikes_in_time`, `peri_event_histogram` and `population_peri_event_histogram`.
- **Parameter text:** reuse 3a's `epochs`/`spike_window` Parameters text verbatim.
- **Decoders' `Notes`:** "Decode time bins are formed separately within each run of samples whose gaps are no longer than `max_gap` and that lie inside `epochs` and `spike_window`; no bin spans a pause, and spikes between runs are not counted. `result.times` may therefore be non-contiguous."
- **PETH `Returns`:** document `n_events_dropped`.
- **`bin_spikes_in_time` Notes:** update the half-open note.

### 8. CHANGELOG, README and user guide (own task)

- **`CHANGELOG.md` `[Unreleased]`:** add the section "Changed — decoding and peri-event histograms respect recording gaps". It covers:
  - the decode-bin fix (on the audit repro, 4,000 of 4,799 bins fell inside a pause);
  - `epochs`/`spike_window`;
  - the `BayesianDecoder.fit(epoch=)` → `epochs=` replacement;
  - half-open time bins whose edges never pass their window: a spike exactly at the stop of a run or epoch is no longer counted, and the bin count no longer uses an absolute `1e-9` epsilon, which dropped a bin at large time offsets (for example `[1e9+0.1, 1e9+0.3)` with `dt=0.1` gave 1 bin);
  - `spike_window` and `spike_window_assumed` on `DecodingResult` and `DecodingSummary`;
  - PETH event dropping and `n_events_dropped`.
- **`README.md` decoding paragraph** (the one that begins "To go the other way" and introduces `decode_session`): add one sentence: "Pauses in tracking are never decoded: time bins are formed only within recorded stretches, and `epochs=`/`spike_window=` restrict them further."
- **`docs/user-guide/interoperability.md:247`:** `epoch=(0.0, 60.0)` → `epochs=(0.0, 60.0)`.
- **Decoding tutorial:** if `examples/20_bayesian_decoding.py` (paired `.ipynb`, mirrored in `docs/examples/`) relies on contiguous `result.times`, update it and re-sync the pair with jupytext.

## Deliberately not in this phase

- **The rate families and `_intervals.py`.** These are done in 3a. Do not re-edit them, except to call them.
- **Behavior, segmentation, `add_positions`.** These belong to 3c.
- **`align_spikes_to_events`.** It returns one entry per input event with no normalization. Dropping events would change its length and index alignment. It stays a primitive. Callers who need observed-only events filter `event_times` first, or use the PETH functions.
- **GLM event regressors (`time_to_nearest_event`, `event_count_in_window`, `event_indicator`).** They return one value per *sample* and drop no events. The contract's spike-only row ("keep the event iff its window ⊆ …") does not map onto them; the natural rule would mask samples whose window leaves the epochs. That changes return dtypes (`int64`/`bool` cannot hold NaN), so it needs a maintainer decision. It stays deferred and is listed in the overview's Open Questions, with this trigger: decide before Phase 4 documents the regressors.
- **Replay-detection helpers** (`detect_trajectory_radon`, `fit_linear_trajectory`, `fit_isotonic_trajectory`) on windows that span a gap. They receive user-chosen posterior windows and are unchanged.
- **The TsGroup handling bug in `population_peri_event_histogram`.** Phase 2 Task 8 fixed it (the function now normalizes its input with `as_spike_trains_with_ids`). Apply this phase's event filtering to those normalized trains.
- **Argument reordering, or removing `Session`.** These belong to Phase 6.

## Validation slice

The fixtures come from 3a (`two_epoch_recording`, `continuous_recording`).

| Test | Asserts |
| --- | --- |
| `tests/decoding/test_decode_gaps.py::test_decode_session_has_no_bins_in_pause` | `decode_session(env, [spikes]*5 (offset per unit), times, positions, dt=0.025)` on the two-epoch fixture. `result.times.size == 7998` (3999 per run, since `floor(99.98/0.025) = 3999`). No center lies in `(99.98, 1100)`. `posterior.shape == (7998, env.n_bins)`. **Fails on `main`**, which has 47999 bins, about 40000 of them in the pause. |
| `tests/decoding/test_decode_gaps.py::test_summary_matches_full_decode` | `decode_session_summary(..., time_chunk=1000)` gives `times` exactly equal to `decode_session(...).times`, and `map_bin` equal to `posterior.argmax(1)` (array-equal). The block size is chosen so that blocks straddle the run break. |
| `tests/decoding/test_decode_gaps.py::test_spikes_in_pause_are_ignored` | Adding 500 spikes uniformly in `[200, 1000)` to every unit leaves the decode `posterior` array-equal, and the encoding models unchanged. |
| `tests/decoding/test_decode_gaps.py::test_epochs_restrict_decode_bins` | `epochs=[(0., 100.)]` gives 3999 bins, all below 100. `spike_window=(1100., 1200.)` gives 3999 bins, all at or above 1100. |
| `tests/decoding/test_decode_gaps.py::test_no_bin_fits_error` | `dt=200.0` raises a `ValueError` matching `"No decode time bin fits"`, with a line starting `"Fix:"`. |
| `tests/decoding/test_decode_gaps.py::test_immobility_still_decoded` | With `min_speed` large enough to exclude every interval in the second epoch from encoding, decode bins still cover the second epoch (3999 bins at or above 1100). |
| `tests/decoding/test_estimator.py::test_fit_epochs_matches_decode_session` | `BayesianDecoder(env).fit(spikes, t, p, epochs=[(0, 100)]).encoding_models` equals the encoding models from `decode_session(..., epochs=[(0, 100)])` (array-equal). `fit(..., epoch=...)` raises `TypeError`. |
| `tests/decoding/test_estimator.py::test_predict_forwards_max_gap_and_windows` | `BayesianDecoder(env, max_gap=2000.).fit(...).predict(spikes, times)` bridges the pause (one run: `floor(1199.98/0.025) = 47999` bins). With the default `max_gap` it gives 7998. `predict(..., epochs=[(1100, 1200)])` gives 3999. |
| `tests/decoding/test_spike_binning.py::test_epochs_tile_each_window` | `bin_spikes_in_time([np.array([0.01, 1.5, 2.01])], dt=0.25, epochs=[(0, 1), (2, 3)])` gives 8 bins with centers `[0.125 … 0.875, 2.125 … 2.875]`. The spike at 1.5 is not counted, and the column sum is 2. Passing `epochs` together with `t_start` raises the "not both" error. |
| `tests/decoding/test_spike_binning.py::test_time_bins_decimal_boundary` | `time_bins_in_windows([[0.1, 0.3]], 0.1)` gives `np.c_[left, right]` array-equal to `[[0.1, 0.2], [0.2, 0.3]]`, and a spike at `0.3` is not counted. The old helper gave a last right edge of `0.30000000000000004` and counted it. |
| `tests/decoding/test_spike_binning.py::test_time_bins_large_offset` | `t0 = 1e9`: `[[t0+0.1, t0+0.3]]` with `dt=0.1` gives 2 bins with `right[-1] == t0+0.3` (the old helper gave 1). `[[t0+0.1, t0+100.1]]` with `dt=0.025` gives 4000 bins, and a spike at `t0+100.1` is not counted. |
| `tests/decoding/test_spike_binning.py::test_bins_never_exceed_window` | 2,000 seeded random whole-multiple windows (`t0 ∈ {0, 1e3, 1e6, 1e9}` plus `U(0, 10)`, `dt ∈ {1, 2, 10, 25, 100, 200, 300}` ms, 1–2000 bins). For each: `n_bins == n`, `left < right`, `right <= stop` for every bin, and a spike at `stop` is not counted. The probe found 0 failures in 20,000, against 3,404 for the old helper. |
| `tests/decoding/test_spike_binning.py::test_time_bins_reject_insufficient_precision` | `[[1e9, 1e9 + 1e-6]]` with `dt=2e-7` raises `ValueError` whose message contains `Fix: subtract a time origin`. The previous allowance turned the same input into 7 bins, 2 of them with `right <= left`. |
| `tests/decoding/test_spike_binning.py::test_time_bins_unix_epoch_timestamps` | `[[1.7e9, 1.7e9 + 10]]` with `dt=5e-4` gives 20000 bins and with `dt=2e-3` gives 5000, all with `right > left` and width ≥ 0.99·dt. Unix-epoch timestamps with millisecond bins keep working without an origin shift. |
| `tests/decoding/test_session.py::test_decode_session_allocates_one_posterior` | Wrap `DecodingResult._from_owned_posterior` to record each `posterior` argument. After `decode_session(...)` on the two-epoch fixture, `result.posterior is` the array `decode_position` passed first, and `np.shares_memory` holds between them. So no second posterior was allocated, and `result.spike_window` equals the resolved windows. A guard asserts that `dataclasses.replace(result, spike_window=None)` *does* copy (`not np.shares_memory`), which documents why `_evolve` exists. |
| `tests/decoding/test_result.py::test_evolve_rejects_posterior_and_keeps_fields` | `r._evolve(posterior=x)` raises `ValueError`. `r._evolve(spike_window=w)` keeps every other field equal to `r`'s, and its `spike_window` is a read-only copy of `w`. |
| `tests/decoding/test_spike_binning.py::test_partial_bin_dropped` | `[0, 1.05)` with `dt=0.25` gives 4 bins ending at 1.0. Spikes at `0.99`, `1.0` and `1.04` count 1 in total. |
| `tests/decoding/test_spike_binning.py::test_chunked_counts_equal_full` | On the two-epoch runs with `dt=0.025` (7998 bins), five seeded trains of 4000 spikes give `count_spikes_in_time_bins` counts in 1000-bin blocks (trains scoped to `[left[start], right[stop-1])`) that array-equal the full call. |
| `tests/decoding/test_decode_gaps.py::test_results_record_spike_window` | `decode_session(...)` with the default gives `spike_window is None`, `spike_window_assumed is True`, and the same values in `summary()`. With `spike_window=(1100., 1200.)` it gives `spike_window == [[1100., 1200.]]` and `spike_window_assumed is False`. `decode_session_summary` and `BayesianDecoder.predict` behave the same. With xarray installed (`test_xarray.yml`), `to_xarray().attrs["spike_window_assumed"]` is `1` or `0`, and `attrs["spike_window"]` is the flat `[1100., 1200.]`. |
| `tests/decoding/test_spike_binning.py::test_bins_are_half_open` | A spike exactly at `t_stop` (when `t_stop` is a whole number of bins from `t_start`) is not counted. The test documents the change from the old right-closed last bin. |
| `tests/decoding/test_result.py::test_plot_marks_recording_gaps` | `DecodingResult` with the two-run `times`: `ax.get_xlabel()` starts with `"Time bin"`, and exactly 1 dashed vertical line is drawn. A contiguous result keeps the `"Time (s)"` label. |
| `tests/events/test_peth_time_windows.py::test_drops_events_whose_window_leaves_epochs` | Events at `[10, 50, 99.5, 1100.2, 1150]` with `window=(-0.5, 1.0)` and `epochs=[(0,100),(1100,1200)]` give `n_events == 3` and `n_events_dropped == 2` (99.5 + 1.0 > 100; 1100.2 − 0.5 < 1100). With `spike_window=(0., 100.)` instead: `n_events == 2`, `n_events_dropped == 3`. The population function gives the same counts, and `result[0].n_events_dropped == 2`. |
| `tests/events/test_peth_time_windows.py::test_flat_rate_recovered_at_recording_edges` | Spikes fire regularly at 10 Hz (period 0.1 s) only inside the two epochs. 200 events are placed uniformly (seed 0) in `[0, 1200)`, with `window=(-1, 1)` and `bin_size=0.1`. With `epochs=[(0,100),(1100,1200)]`, every PETH bin's `firing_rate` is `10.0 ± 1%`. Without `epochs` (the `main` behavior), the minimum bin rate is below 9.0 Hz. This documents why the argument exists. |
| `tests/events/test_peth_time_windows.py::test_all_events_dropped_error` | `epochs=[(5000., 6000.)]` raises a `ValueError` containing `"peri-event"`, the event count, and `"Fix:"`. |
| `tests/events/test_peth_time_windows.py::test_nan_event_still_raises` | `event_times=[np.nan, 10.]` with `epochs` set still raises the existing NaN error. It is not silently dropped. |
| `tests/events/test_peth_time_windows.py::test_summary_reports_dropped` | `result.summary()["n_events_dropped"] == 2` for the first case above. |

No test in this slice exceeds about 2 s. Mark any decode test over 5 s on CI `@pytest.mark.slow`.

## Fixtures

- `two_epoch_recording` and `continuous_recording` from `tests/conftest.py` (added in 3a).
- **Five-unit populations.** Build them in the test module from the fixture's regular 5 Hz train, shifted by `0.04·u` s. Encoding uses `method="binned"` (fast) unless the test is about smoothing.
- **Regular 10 Hz PETH train.** Build it inline with `np.r_[np.arange(0.05, 100, 0.1), np.arange(1100.05, 1200, 0.1)]`.

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

- `grep -n "np.histogram" src/neurospatial/decoding/` returns nothing. Every decode and time-binning path uses `time_bins_in_windows` and `count_spikes_in_time_bins`.
- The `epoch=` branch and the edge-drift guards are gone.
- On gap-free data, `decode_session` gives `times` and `posterior` equal to `main`'s (`rtol=1e-12`), except when a spike lies exactly on the final edge (the documented half-open change). On gap-free data the bin count is unchanged wherever the old `+1e-9` and the new relative slack agree, which is every window in the sweep above except those whose quotient fell more than `1e-9` below a whole number. Run `scientific-code-change-audit` on the diff.
