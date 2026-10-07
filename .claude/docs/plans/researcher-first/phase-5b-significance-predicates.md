# Phase 5b — Opt-in shuffle significance and one predicate contract

**Requires:** Phase 5a.

[← back to PLAN.md](PLAN.md) · [executing](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Read [executing.md](executing.md) first: branch and PR workflow, definition of done, CHANGELOG-per-commit, and what to do when the plan and reality disagree. This file holds only what is specific to Phase 5b.

## Checkpoint additions (2026-10-07)

- In the updated object-vector tutorial, keep an explicit place-cell control
  and distinguish a candidate information screen from cell identity. The
  checkpoint's control passed at 1.0982 bits/spike; that is corroborating
  evidence for this phase's existing bias work, not a new threshold target.
- Reconcile raw/smoothed estimator guidance. The checkpoint measured raw
  information slightly above smoothed information for both OVC and control;
  do not promise that raw binning is universally lower or more conservative.
  Show which estimator/criterion each printed verdict used and preserve
  same-estimator agreement between the result method and convenience predicate.

## Original scope

Each threshold classifier keeps its current criterion (decision 5), with one exception, place cells (decision 5, revised):

- Field detection keeps its fast behavior under the honest name `has_place_field()`, as a free function and a method.
- `is_place_cell()` (free function and method) has **no default criterion**; the caller passes `criterion="spatial_info"` or `criterion="shuffle"`.

This phase:

- adds seeded circular time-shift shuffle significance as an **explicit computation on raw arrays**. One free `*_significance` function per family recomputes the map from the arrays passed in that call, and the free `is_*_cell(criterion="shuffle")` predicates use it. Results never retain their inputs (decision 4), so no result method runs a shuffle;
- gives the free predicates, the result methods and the batch `classify` one set of threshold keywords, resolved from one named constant per family;
- documents how biased the threshold criteria are.

**Inputs to read first** (line numbers verified on `main` at `da631a47`; earlier phases shift them, so search by name):

- [src/neurospatial/stats/shuffle.py:802](../../../../src/neurospatial/stats/shuffle.py) — `ShuffleTestResult` (reused as the per-unit return type), `compute_shuffle_pvalue` (:946, Phipson–Smyth). No spike-train time-shift shuffle exists (only `shuffle_spikes_isi` at :1217, which permutes ISIs). [stats/_utils.py:11](../../../../src/neurospatial/stats/_utils.py) `_ensure_rng`.
- [src/neurospatial/encoding/spatial.py](../../../../src/neurospatial/encoding/spatial.py) — `SpatialRateResult.is_place_cell` :1114 (place-*field detection*; renamed `has_place_field` in 5b.4), `label_cell_types` :1959 (gate fixed in Phase 5a), deprecated `detect_cell_types` :2105, `classify` :2144 (*spatial information* `>= min_spatial_info`), free `is_place_cell` :4450 (field detection; swallows `ValueError`/`RuntimeError` → `False` at :4532).
- [src/neurospatial/encoding/directional.py](../../../../src/neurospatial/encoding/directional.py) — method :785 (positional args), deprecated `detect_hd_cells` :1479, `classify` :1421, free :2184 (swallows at :2289).
- [src/neurospatial/encoding/view.py](../../../../src/neurospatial/encoding/view.py) — method :354 (positional), deprecated `detect_view_cells` :952, `classify` :897, free :1735 (swallows at :1819).
- [src/neurospatial/encoding/egocentric.py](../../../../src/neurospatial/encoding/egocentric.py) — after Phase 5a: `ObjectVectorRateResult.is_object_vector_cell` (positional `min_info`), `ObjectVectorRatesResult.classify`, deprecated `detect_ovcs` :1062, the free `is_object_vector_cell` (allocentric) and `is_egocentric_object_vector_cell` (both still swallow errors), the "How was 0.3 chosen?" note at :436 and the doctest at :497.
- **Files earlier phases already changed** (search by name):
  - `encoding/spatial.py`: Phase 3a added `max_gap`/`epochs`/`spike_window` to every compute function and `is_*_cell` predicate and factored `_resolve_interval_mask`; Phase 1 Task 7 added `resolve_unit_ids(..., input_ids=)` and its duplicate-label rejection.
  - `encoding/directional.py`, `view.py`, `egocentric.py`: Phase 3b (the frame kernel and time-window keywords), Phase 1 Task 7 (spike-group input in the plural functions), Phase 5a (object-vector frames, `_object_vector_interval_mask`).
  - Phase 4b's `tests/docs/test_flagship_docstrings.py` and `tests/docs/test_docstring_sections.py` (below). Use Phase 4a's `_format_error` for new messages.
  - `CHANGELOG.md`: append under `[Unreleased]`.

**Contracts referenced:**

- [Overview decision 4](overview.md#settled-design-decisions) — raw arrays are the only input form. Results keep no inputs, so a shuffle test is a free computation on the arrays passed in that call (5b.3).
- [Overview decision 5](overview.md#settled-design-decisions) (revised) — threshold criteria stay; place field detection becomes `has_place_field()`; `is_place_cell()` has no default criterion.
- [Time-window semantics](shared-contracts.md#time-window-semantics) — the shuffle shifts spikes only within the *same* valid intervals the observed map used: `run_time_bounds(times, mask)` of the family's interval mask, which includes `epochs ∩ spike_window`. Never shift into a gap.
- [Error-message contract](shared-contracts.md#error-message-contract) — every new error has what / why / `Fix:`. Rule 1: object-vector frames are separate functions (Phase 5a), so the significance functions follow the same split.
- [Input conventions](shared-contracts.md#input-conventions) — each significance function keeps its family's raw form. Their `unit_ids=` follows "Labels are never overridden", through Phase 1 Task 7's `resolve_unit_ids`, which also rejects duplicate labels.

## Evidence (measured on `main`, `da631a47`)

20 independent homogeneous 0.5 Hz Poisson units, one 30 Hz OU trajectory in a 100 × 100 cm arena (Phase 5a's `ou_trajectory` fixture, seed 0; spikes seed 1); place and view maps use `bin_size=5`, `diffusion_kde`, `bandwidth=5`; object-vector maps the default 10 × 12 binned polar grid. On `main` the only object-vector map is egocentric. Counts are units classified as the cell type.

| Duration (≈spikes) | Egocentric OVC info, median (`min_info=0.3`) | Place SI median (`classify`, 0.5) | Place field detection (`main`'s `is_place_cell`, now `has_place_field`) | View `classify` (0.5) | HD `classify` |
| --- | --- | --- | --- | --- | --- |
| 1 min (30) | 2.12 → **20/20** | 0.86 → 19/20 | **20/20** | 19/20 | 0/20 |
| 2 min (60) | 1.41 → **20/20** | 0.53 → 15/20 | **20/20** | 19/20 | 0/20 |
| 5 min (150) | 0.70 → **20/20** | 0.22 → 0/20 | **20/20** | 0/20 | 0/20 |
| 10 min (300) | 0.41 → **20/20** | 0.11 → 0/20 | **20/20** | 0/20 | 0/20 |
| 20 min (600) | 0.21 → 0/20 | 0.05 → 0/20 | **20/20** | 0/20 | 0/20 |

The audit's independent run (20 trajectories) agrees for object-vector cells: 20/20 at 5 and 10 min, 2/20 at 15 min, 0/20 at 20 min. Plug-in Skaggs information is biased upward by roughly `(n_bins − 1) / (2 ln 2 · N_spikes)`, so a fixed cutoff thresholds the spike count. A prototype of the shuffle below (200 shuffles, `min_shift=20`, 10 min) flagged 1/20 noise units (object-vector), 2/20 (place), and 3/100 pooled over 5 more seeds. It gave a true object-vector cell p = 1/201.

**Fast-path probe (this plan's dry run, on `main`).** On the 2-min trajectory, a field cell `PlaceCellModel(env, center=[70, 50], width=10, max_rate=20)` (spikes seed 3) fires 87 spikes. Its observed spatial information (`compute_spatial_rates`, `diffusion_kde`, bandwidth 5) was 2.54 bits/spike. Across 5 RNG seeds × 20 circular shifts drawn from `[20, T − 20]`, the largest shifted value was 1.93, so `p = 1/21` for every seed. The fixture's narrower field (`width=6, max_rate=10`) fires only 17 spikes in 2 min and its margin was 0.03 bits, too thin for a CI assertion.

## Tasks

**5b.1 Circular time-shift primitive.** Add to `src/neurospatial/stats/shuffle.py`, export from `neurospatial.stats.__all__`, and list it in the module's "Shuffle Categories" table:

```python
def shuffle_spike_times_circular(
    spike_times: NDArray[np.float64],
    windows: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly time-shift one spike train within the analyzed time.

    The rows of ``windows`` are joined into one circular axis of length
    ``T = sum(stop - start)``. Each shuffle adds one offset drawn uniformly from
    ``[min_shift, T - min_shift]`` to every spike and wraps it on that axis, so
    spikes never land in a gap and the count and spike-train structure are kept;
    only the alignment to behavior is broken (Skaggs et al. 1993 style null).

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike times in seconds. Spikes outside every window are dropped.
    windows : ndarray, shape (n_windows, 2)
        Sorted, non-overlapping ``[start, stop)`` rows of analyzed time, seconds.
    n_shuffles : int, default=1000
        Number of shifted trains to yield.
    min_shift : float, default=20.0
        Smallest shift, in seconds of analyzed time, in either direction.
    rng : numpy.random.Generator, int or None, default=None
        Random source; an int seeds ``numpy.random.default_rng``.

    Yields
    ------
    ndarray, shape (n_spikes_in_windows,)
        Sorted shifted spike times, seconds.

    Raises
    ------
    ValueError
        If the analyzed time is not longer than ``2 * min_shift``.
    """
    windows = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    offsets = np.concatenate([[0.0], np.cumsum(windows[:, 1] - windows[:, 0])])
    total = float(offsets[-1])
    if total <= 2.0 * min_shift:
        raise ValueError(
            f"The analyzed time is {total:.1f} s, not longer than 2 * min_shift = "
            f"{2.0 * min_shift:.1f} s, so no circular shift moves every spike by at "
            "least min_shift.\n"
            f"Fix: pass a smaller min_shift (e.g. min_shift={total / 4:.1f}) or "
            "analyze a longer recording."
        )
    idx = np.searchsorted(windows[:, 0], spike_times, side="right") - 1
    inside = (idx >= 0) & (spike_times < windows[np.maximum(idx, 0), 1])
    idx = idx[inside]
    compressed = offsets[idx] + (spike_times[inside] - windows[idx, 0])
    generator = _ensure_rng(rng)
    for _ in range(n_shuffles):
        wrapped = np.mod(compressed + generator.uniform(min_shift, total - min_shift), total)
        j = np.minimum(np.searchsorted(offsets, wrapped, side="right") - 1, len(windows) - 1)
        yield np.sort(windows[j, 0] + (wrapped - offsets[j]))
```

**5b.2 One significance engine.** New private module `src/neurospatial/encoding/_significance.py`. Every family calls it, and no family has its own shuffle loop. It works on arrays the caller-facing function has already copied (5b.3); it never sees a result object.

```python
def _stream_key(label: Hashable) -> int:
    """Stable 64-bit key for a unit label (``3``, ``np.int64(3)`` and ``3`` agree)."""
    value = np.asarray(label).item()
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], "little")


def _entropy(rng: np.random.Generator | int | None) -> int:
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(2**63))
    if rng is None:
        return int(np.random.SeedSequence().entropy)
    return int(rng)


def run_shuffle_test(
    statistic: Callable[[list[NDArray[np.float64]]], ArrayLike],
    spike_times: list[NDArray[np.float64]],
    windows: NDArray[np.float64],
    unit_ids: NDArray[Any],
    *,
    n_shuffles: int,
    min_shift: float,
    rng: np.random.Generator | int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return observed ``(n_units, n_stats)`` and null ``(n_shuffles, n_units, n_stats)``.

    ``statistic(trains)`` recomputes the family's plural map from arrays the
    public function already copied and returns one row per train. Unit ``i``'s
    shifts come from a random stream keyed by ``unit_ids[i]``, so a unit gets
    identical shifts whether it is tested alone or in any population, in any
    order, given the same integer ``rng``. ``unit_ids`` are unique
    (``resolve_unit_ids`` rejects duplicates), so no two units share a stream.
    """
    entropy = _entropy(rng)
    streams = [
        shuffle_spike_times_circular(
            train, windows, n_shuffles=n_shuffles, min_shift=min_shift,
            rng=np.random.default_rng(
                np.random.SeedSequence(entropy, spawn_key=(_stream_key(uid),))
            ),
        )
        for train, uid in zip(spike_times, unit_ids, strict=True)
    ]

    def evaluate(trains: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        return np.asarray(statistic(trains), dtype=np.float64).reshape(len(trains), -1)

    observed = evaluate(list(spike_times))
    null = np.empty((n_shuffles, *observed.shape))
    # Warnings from the observed recompute above reach the caller once. The
    # shifted recomputes would repeat them n_shuffles times, and shifted trains
    # can trip data-quality heuristics (such as the population-silence warning)
    # that say nothing about the caller's data, so they are silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n_shuffles):
            null[k] = evaluate([next(s) for s in streams])
    return observed, null


def shuffle_pvalues(observed, null):
    """Phipson–Smyth ``(1 + #{null >= observed}) / (1 + n_finite)``; NaN where observed is NaN."""
    finite = np.isfinite(null)
    exceed = np.sum(finite & (null >= observed[None]), axis=0)
    p = (1.0 + exceed) / (1.0 + np.sum(finite, axis=0))
    return np.where(np.isfinite(observed), p, np.nan)
```

Also add the following helpers:

- `check_criterion(criterion, allowed: tuple[str, ...], *, call: str)`. It raises `ValueError` listing every allowed value, with a `Fix:` line.
- `check_mode_keywords(criterion, *, threshold: dict[str, Any], shuffle: dict[str, Any], call: str) -> None` (see 5b.4).
- `to_shuffle_results(observed, null, p, unit_ids, column=0) -> dict[Hashable, ShuffleTestResult]`. It returns results keyed by unit label, in input order, and fills `shuffle_type="circular_time_shift"`. It takes `z = (obs − mean) / std` over the finite nulls, or NaN when the std is 0. It asserts `len(result) == len(unit_ids)`: a label-keyed dict would silently drop a unit if two labels were equal, which Phase 1 Task 7's duplicate rejection in `resolve_unit_ids` rules out at the input.
- Private `_SHUFFLE_DEFAULTS = MappingProxyType({"n_shuffles": 1000, "min_shift": 20.0, "alpha": 0.05})`, used by the free predicates' shuffle mode.

The windows are the family's valid runs, `run_time_bounds(times, mask)` (Phase 3a's `_intervals.py`). Do not add a second run extractor.

The engine was probed against `main`'s `compute_spatial_rates` (binned, 3 units, 120 s, `n_shuffles=50`, `rng=0`, labels `[10, 20, 30]`):

- each unit's p-value was identical tested alone and in the population: `0.6078`, `0.2745` and `0.0588`;
- the p-values were identical after reordering the population;
- unit 20's shifted trains were array-equal alone and in the population;
- the smallest `|null − observed|` was `1.1e-3`.

**5b.3 One significance function per family; results keep no inputs.**

- **Why not on the result.** An earlier design stored the caller's arrays and a population partial on every result. That design is unsound, for two reasons:
  - mutating a caller array after the call silently changes a later shuffle;
  - `rates[i]` inherited population-bound `unit_ids`, so a single-unit recompute raised a length mismatch (reproduced in review).

  Decision 4 settles it: raw arrays in, results out. A result holds no spike times, timestamps, positions, headings or recompute closure. No `shuffle_test()` method, `classify(criterion="shuffle")` or `label_cell_types(criterion="shuffle")` is added.
- **Public functions** (one per family and frame, added to `encoding.__all__`; each keeps its family's raw argument order):

  ```text
  place_cell_significance(env, spike_times, times, positions, *, unit_ids=None, <compute_spatial_rates keywords>, n_shuffles=1000, min_shift=20.0, rng=None)
  head_direction_cell_significance(spike_times, times, headings, *, unit_ids=None, <compute_directional_rates keywords>, n_shuffles=1000, min_shift=20.0, rng=None)
  object_vector_cell_significance(env, spike_times, times, positions, object_positions, *, unit_ids=None, <compute_object_vector_rates keywords>, n_shuffles=1000, min_shift=20.0, rng=None)
  egocentric_object_vector_cell_significance(env, spike_times, times, positions, headings, object_positions, *, unit_ids=None, <compute_egocentric_rates keywords>, n_shuffles=1000, min_shift=20.0, rng=None)
  spatial_view_cell_significance(env, spike_times, times, positions, headings, *, unit_ids=None, <compute_view_rates keywords>, n_shuffles=1000, min_shift=20.0, rng=None)
  ```

  Spell out every forwarded compute keyword in the signature (no `**kwargs`), so the signature is truthful and each keyword is documented in Parameters. Phase 4b's `test_docstring_sections.py` checks every parameter of a `FLAGSHIP` callable, and these five join `FLAGSHIP` (5b.7).

  Each returns `dict[unit_id, ShuffleTestResult]` in input order. The statistic is the one the threshold path uses:

  | Function | Statistic |
  | --- | --- |
  | place | `spatial_information()` |
  | head direction | `mean_vector_lengths()` |
  | object-vector (both frames) | `spatial_information()` (renamed in Phase 5a) |
  | view | `view_spatial_information()` |

  `ShuffleTestResult.is_significant` is fixed at 0.05, so the docstrings say to compare `p_value < alpha`. Per family a function, not one generic `shuffle_significance(kind, ...)`, because the raw forms differ: no env or positions for head direction, object positions for object-vector, headings for view and the egocentric frame. One string-dispatched signature would need `None` slots, which the [error contract](shared-contracts.md#error-message-contract) rules out.
- **Body pattern** (place; the others differ only in the compute function, the statistic and the mask helper):

  ```python
  if method == "glm":
      raise ValueError(_format_error(
          "place_cell_significance does not support method='glm'.",
          why="pooled REML couples units, so a recompute on shifted trains is not "
              "the same model as the observed fit",
          fix="method='diffusion_kde' (default), 'gaussian_kde' or 'binned'.",
      ))
  trains, input_ids = as_spike_trains_with_ids(spike_times)  # from encoding._spikes
  # Copy once. Every shuffle recomputes from these arrays, so a caller who edits
  # theirs while this (slow) loop runs cannot change the result.
  trains = [np.array(t, dtype=np.float64) for t in trains]
  times = np.array(times, dtype=np.float64)
  positions = np.array(positions, dtype=np.float64)
  speed = None if speed is None else np.array(speed, dtype=np.float64)
  ids = resolve_unit_ids(unit_ids, len(trains), input_ids=input_ids,
                         context="place_cell_significance")
  E, S = resolve_time_windows(epochs, spike_window)
  mask = _spatial_interval_mask(env, times, positions, speed=speed, min_speed=min_speed,
                                max_gap=max_gap, epochs=E, spike_window=S)
  windows = run_time_bounds(times, mask)

  def statistic(shifted):
      return compute_spatial_rates(
          env, shifted, times, positions, method=method, bandwidth=bandwidth,
          min_occupancy=min_occupancy, speed=speed, min_speed=min_speed,
          max_gap=max_gap, epochs=E, spike_window=S, warn_on_drop=False,
      ).spatial_information()

  observed, null = run_shuffle_test(statistic, trains, windows, ids,
                                    n_shuffles=n_shuffles, min_shift=min_shift, rng=rng)
  return to_shuffle_results(observed, null, shuffle_pvalues(observed, null), ids)
  ```

  Import `as_spike_trains_with_ids` from `neurospatial.encoding._spikes`; Phase 6a removes it from `encoding.__all__`.
- **Copy every array-valued input, not just spikes and positions.** Each `*_significance` function copies, once and up front, **every** array-like argument its recompute closure reads:
  - `spike_times`, `times`, `positions` and `speed`;
  - `headings`, `object_positions` and the view-target arrays, in the families that take them;
  - any array-valued keyword, such as an explicit `bandwidth` array.

  The closure must close over these copies only. A caller who edits `speed` while the loop runs would otherwise change the null distribution and leave the observed statistic unchanged: an external probe moved five untuned units from 0/5 to 5/5 "significant" that way. A unit test enforces the rule for every family (see the Validation slice).
- **One mask per family.** Factor each family's mask construction into a private helper that both its compute functions and its significance function call, so the shuffle windows are exactly the intervals the observed map used: `_spatial_interval_mask` (Phase 3a's `_resolve_interval_mask`), `_directional_interval_mask`, `_view_interval_mask` and Phase 5a's `_object_vector_interval_mask` (both frames). A probe on `main` showed that an engine copying its inputs this way returns identical p-values when the caller's `positions` and spike trains are overwritten after the first recompute.
- **`unit_ids`** follows [Labels are never overridden](shared-contracts.md#input-conventions) through Phase 1 Task 7's `resolve_unit_ids(..., input_ids=)`. That resolver rejects duplicate labels, which is what keeps the label-keyed return dict from silently merging two units.

**5b.4 One predicate contract.** Current inconsistencies on `main`:

| Cell type | Free function | Result method | Batch `classify` |
| --- | --- | --- | --- |
| place | field detection (`threshold`, `min_size`, `max_mean_rate`, `detect_subfields`) | field detection | spatial information `>= min_spatial_info` |
| head direction | MVL `>` and Rayleigh p `<` `alpha` | same, **positional** args | same, keyword-only |
| object-vector | info `>` `min_info`; cannot pass `method=` | **positional** `min_info` | keyword-only |
| spatial view | info `>` `min_info` | **positional** `min_info` | keyword-only |

All the free functions return `False` on any `ValueError`/`RuntimeError`, which hides bad input. The target:

```text
# Free predicates: raw arrays, one unit, both criteria
is_place_cell(env, spike_times, times, positions, *, criterion, min_info=None,
              alpha=None, n_shuffles=None, min_shift=None, rng=None, unit_id=None, <compute keywords>)
is_head_direction_cell(spike_times, times, headings, *, criterion="threshold", min_mvl=None,
              alpha=None, n_shuffles=None, min_shift=None, rng=None, unit_id=None, <compute keywords>)
is_object_vector_cell(env, spike_times, times, positions, object_positions, *, criterion="threshold",
              min_info=None, alpha=None, n_shuffles=None, min_shift=None, rng=None, unit_id=None,
              <compute_object_vector_rate keywords>)
is_egocentric_object_vector_cell(env, spike_times, times, positions, headings, object_positions, *,
              criterion="threshold", min_info=None, alpha=None, n_shuffles=None, min_shift=None,
              rng=None, unit_id=None, <compute_egocentric_rate keywords>)
is_spatial_view_cell(env, spike_times, times, positions, headings, *, criterion="threshold",
              min_info=None, alpha=None, n_shuffles=None, min_shift=None, rng=None,
              unit_id=None, <compute keywords>)
has_place_field(env, spike_times, times, positions, *, threshold=0.2, min_size=None,
              max_mean_rate=10.0, detect_subfields=True, <compute keywords>)

# Result methods and batch classify: threshold criteria only
SpatialRateResult.is_place_cell(*, criterion, min_info=None)  # required; "spatial_info" only
SpatialRateResult.has_place_field(*, threshold=0.2, min_size=None, max_mean_rate=10.0, detect_subfields=True)
DirectionalRateResult.is_head_direction_cell(*, min_mvl=None, alpha=None)
ObjectVectorRateResult.is_object_vector_cell(*, min_info=None)
ViewRateResult.is_spatial_view_cell(*, min_info=None)
<plural result>.classify(*, <that family's threshold keywords>) -> NDArray[np.bool_]
SpatialRatesResult.label_cell_types(*, min_spatial_info=None, min_grid_score=None, min_border_score=None)
```

- **`has_place_field`** is `main`'s `is_place_cell`, renamed in place: the same detector, keywords, defaults and Phase 3a time-window keywords. Add it to `encoding.__all__` explicitly (it is a new name there; the old `is_place_cell` entry stays for the new verdict). Its docstring summary says it detects a field and is **not** a cell-type verdict, and gives the measured false-positive rate (5b.6). The method is renamed the same way (`spatial.py:1114`). `SpatialRatesResult.classify`'s `.. note::` (`:2156-2162`) now points to `has_place_field` for field detection. Its free form loses `main`'s `try/except → False`, as every predicate does (below).
- **`is_place_cell` requires `criterion` as an ordinary required keyword-only argument**: `def is_place_cell(env, spike_times, times, positions, *, criterion: Literal["spatial_info", "shuffle"], ...)`. There is no sentinel and no default. A plain required keyword is what IDEs, type checkers and `help()` already understand: mypy and pyright flag a missing `criterion` before the code runs, and the signature reads `criterion` with no default. Calling without it gives Python's own message, `is_place_cell() missing 1 required keyword-only argument: 'criterion'`. That names the argument exactly, and the [error contract](shared-contracts.md#error-message-contract) exempts missing-argument `TypeError`s. The docstring's summary paragraph carries the guidance a custom message would have:

  > `criterion` has no default because there is no default verdict. Use `"shuffle"` when the verdict must control false positives (slow), or `"spatial_info"` for a fast screen (biased upward at low spike counts). To ask whether the map has a field, use `has_place_field()`. Field detection flags spatially untuned 0.5 Hz Poisson units 20/20, so it is not a cell-type verdict.

  An invalid value such as `criterion="threshold"` raises `ValueError`, following the error contract, listing both valid values and naming `has_place_field`.
  The method has the same required keyword (`criterion: Literal["spatial_info"]`) and accepts only `"spatial_info"`. For `criterion="shuffle"` it raises `ValueError`: "a result does not keep the spike times it was computed from, so it cannot run a shuffle test. Fix: `is_place_cell(env, spike_times, times, positions, criterion='shuffle')` or `place_cell_significance(...)`."
- **Agreement.** Each free predicate in threshold mode equals `compute_*_rate(...).<method>(...)`, and each method equals `classify()[i]` for every family and frame. That now includes place (`criterion="spatial_info"` against `classify()`), because field detection is no longer an `is_place_cell` criterion.
- **Keywords belong to one mode, and a keyword for the other mode raises** (free predicates only; the methods have no shuffle mode). Silently ignoring an argument would contradict the plan's guiding principle, so `None` means "not passed" and is resolved inside the function:

  | Group | Keywords (resolved default) | Used when | Passed (not `None`) with the other criterion |
  | --- | --- | --- | --- |
  | threshold | place: `min_info` (0.5); view: `min_info` (0.5); object-vector, both frames: `min_info` (0.3 bits/spike); head direction: `min_mvl` (0.4); `label_cell_types` (method only): `min_spatial_info` (0.5), `min_grid_score` (0.4), `min_border_score` (0.5) | `criterion="spatial_info"` (place) or `"threshold"` (the others) | `ValueError` |
  | shuffle | `n_shuffles` (1000), `min_shift` (20.0 s), `rng` (unseeded), `unit_id` (label 0, which keys the unit's random stream); `alpha` (0.05) for place, view and both object-vector frames | `criterion="shuffle"` | `ValueError` |
  | both | head direction `alpha` (0.05): the Rayleigh level in threshold mode and the p-value level in shuffle mode, so its current default behaviour is unchanged | both | — |

  `check_mode_keywords` lists **every** offending keyword in one message (contract rule 4):

  > is_place_cell(criterion="shuffle") got min_info=0.3, which applies only when criterion="spatial_info".
  > Why: the shuffle test decides by its p-value < alpha, so this value would be silently ignored.
  > Fix: drop it, or pass criterion="spatial_info".
- **Threshold constants.** The resolved defaults live in one module constant per family, not in the signatures. Phase 7 reads them by these exact names for `df.attrs`, so do not rename them. Each is a read-only `types.MappingProxyType`, and none is added to `__all__`:

  | Constant | Module | Value | Used by |
  | --- | --- | --- | --- |
  | `PLACE_FIELD_DETECTION_DEFAULTS` | `encoding/spatial.py` | `{"threshold": 0.2, "min_size": None, "max_mean_rate": 10.0, "detect_subfields": True}` | `has_place_field` (free and method). It keeps literal signature defaults, because it has one mode; a test asserts they equal this constant |
  | `PLACE_SPATIAL_INFO_THRESHOLDS` | `encoding/spatial.py` | `{"min_info": 0.5}` | `is_place_cell(criterion="spatial_info")`, the method, `SpatialRatesResult.classify` |
  | `PLACE_GRID_BORDER_THRESHOLDS` | `encoding/spatial.py` | `{"min_spatial_info": 0.5, "min_grid_score": 0.4, "min_border_score": 0.5}` | `label_cell_types` |
  | `HEAD_DIRECTION_THRESHOLDS` | `encoding/directional.py` | `{"min_mvl": 0.4, "alpha": 0.05}` | `is_head_direction_cell` (both modes for `alpha`), the method, `classify`. `alpha` is no longer a literal `0.05` signature default anywhere |
  | `VIEW_THRESHOLDS` | `encoding/view.py` | `{"min_info": 0.5}` | `is_spatial_view_cell`, the method, `classify` |
  | `OBJECT_VECTOR_THRESHOLDS` | `encoding/egocentric.py` | `{"min_info": 0.3}` | both object-vector frames' predicates, the method, `classify` |

  The values equal `main`'s literal defaults (verified with `inspect.signature` on `main`), so no default changes (decision 5).
- **Threshold rule:** comparisons become `>=` everywhere.
- **Shuffle rule:** a unit is the cell type iff its shuffle p-value is `< alpha`, using the statistic table in 5b.3.
- **Free-function compute keywords:** the free functions accept the compute keywords of their family. The object-vector predicates gain `method=` and `bandwidth=`.
- **Removed:**
  - every free predicate's `try/except → False` block (errors now propagate);
  - the deprecated `detect_cell_types`, `detect_hd_cells`, `detect_view_cells` and `detect_ovcs` methods (decision 1);
  - `classify(min_spatial_info=)`, renamed to `min_info`.
- **Keyword-only:** the method thresholds (directional :785, view :354 and object-vector :421 are positional on `main`) and the `label_cell_types` thresholds become keyword-only.

**5b.5 Shuffle in the free predicates.**

- **What the free predicate does.** `is_*_cell(..., criterion="shuffle")` calls its family's significance function on the one train, `significance(..., [spike_times], ..., unit_ids=[0 if unit_id is None else unit_id])[label]`, and returns `p_value < alpha`. `is_object_vector_cell` calls `object_vector_cell_significance`; `is_egocentric_object_vector_cell` calls `egocentric_object_vector_cell_significance`.
- **Agreement with a population call.** With the same integer `rng`, unit `u` gets the same shifts alone (`unit_id=u`) as inside `*_significance(..., unit_ids=[..., u, ...])`, in any population order. That is because the stream is keyed by label (5b.2).
  - The statistics themselves agree only to about 1e-15, since the plural kernels batch units. Measured max `|single − batch|` on `main`: spatial binned 1.8e-15, `diffusion_kde` 8.5e-16, directional 4.2e-17, view 8.3e-16, egocentric 0.
  - So the p-values agree unless a null value lies within that distance of the observed value. The agreement test asserts that margin explicitly, so a flip is diagnosed rather than mysterious.
- **Docstrings.**
  - **Runtime.** Each free predicate and significance function states that it costs about `n_shuffles` recomputes of the plural map. Measured on `main`: 200 shuffles of 22 units took 1.2 s (egocentric object-vector) and 3.4 s (place, `diffusion_kde`).
  - **Methods and `classify`.** Each says: "For a shuffle test, call `<family>_significance(...)` or `is_<celltype>_cell(..., criterion='shuffle')` with the raw arrays; a result does not keep the arrays it was computed from."

**5b.6 Document the bias** in each predicate's, `classify`'s, `has_place_field`'s and `label_cell_types`'s Notes section:

- give the measured table rows for that cell type (spike count, median information, fraction flagged);
- give the `(n_bins − 1)/(2 ln 2 N)` bias formula;
- give guidance: the threshold is a screening heuristic whose false-positive rate grows as spikes fall. Report a shuffle test (`criterion="shuffle"` or the `*_significance` functions) for publication.

The place docstrings state the measured behaviour **in their summary paragraph**, not only in Notes.

- **`has_place_field`:** "Detects whether the rate map has a place field; this is not a cell-type verdict. Independent 0.5 Hz Poisson units with no spatial tuning had a detected field 20/20 at every recording length from 1 to 20 min (100 × 100 cm arena, 5 cm bins). Use `is_place_cell(..., criterion="shuffle")` for a verdict that controls false positives."
- **`is_place_cell`:** "There is no default criterion. `criterion="spatial_info"` thresholds plug-in spatial information, which flagged 19/20 and 15/20 such noise units at 1 and 2 min. `criterion="shuffle"` tests it against circularly shifted spike trains."

Delete the "How was 0.3 chosen? … (Hoydal et al., 2019)" justification at `egocentric.py:436` and its twin in the free function, and state the value is this library's heuristic. Fix the doctest at `egocentric.py:497`: it shows 100 uniform random spikes classified as an object-vector cell at `min_info=0.5`. Keep the example but say that is the bias, and show the free `is_object_vector_cell(..., criterion="shuffle", n_shuffles=50, rng=0)` returning `False`. Head direction's Notes mention, without fixing, that Rayleigh on rate-weighted MVL is occupancy-biased.

**5b.7 Documentation** (each is part of this PR; each commit adds its own CHANGELOG bullet per [executing.md](executing.md), and this task checks they are all present):

- CLAUDE.md "Cell-type API":
  - the shipped predicates become `has_place_field` (field detection, not a verdict), `is_place_cell` (required `criterion`), `is_head_direction_cell`, `is_object_vector_cell`, `is_egocentric_object_vector_cell` and `is_spatial_view_cell`;
  - the five `*_significance` functions are the population shuffle;
  - result methods and `classify` are threshold-only.
- CLAUDE.md pattern 2 and any other `is_place_cell` call in CLAUDE.md, README, `docs/getting-started/quickstart.md` and `.claude/QUICKSTART.md` ("Spatial View Cells", "Object-Vector Cells"): pass `criterion=` or switch to `has_place_field`.
- `.claude/API_REFERENCE.md` (new functions); `docs/glossary.md` (shuffle significance); `docs/user-guide/neuroscience-metrics.md` (the bias table and when to use the shuffle).
- `examples/22` and `25` (keyword-only predicate calls) and `24` (shuffle example). Run `uv run jupytext --sync` on each, then `uv run python docs/sync_notebooks.py`.
- Phase 4b's `FLAGSHIP` tuple: add `has_place_field`, `place_cell_significance`, `head_direction_cell_significance`, `object_vector_cell_significance`, `egocentric_object_vector_cell_significance` and `spatial_view_cell_significance`. Their examples use ≤ 60 s of simulated data and `n_shuffles=20`. The `is_place_cell` example passes `criterion=`. Keep every edited example self-contained and executable, including the `egocentric.py:497` doctest.
- `CHANGELOG.md` `[Unreleased]`:
  - **breaking:** `is_place_cell` → `has_place_field` (same behavior), and the new `is_place_cell` requires `criterion="spatial_info" | "shuffle"`;
  - `shuffle_spike_times_circular` and the five `*_significance` functions;
  - the free predicates' `criterion=` keyword and the mode-keyword `ValueError`;
  - free predicates now raise on bad input instead of returning `False`;
  - results do not keep their inputs, and methods are threshold-only;
  - threshold keywords now default to `None` (resolved from the named constants), comparisons are `>=`, method thresholds are keyword-only, `classify(min_spatial_info=)` → `min_info`;
  - the removed `detect_*` aliases.

## Deliberately not in this phase

- **Changing a threshold classifier's default to `criterion="shuffle"`.** Decision 5 keeps the head-direction, view and object-vector thresholds as defaults. Only `is_place_cell` loses its default, and it gains none.
- **Shuffle mode on result methods, `classify` or `label_cell_types`.** Results keep no inputs (decision 4). A shuffled `label_cell_types` would also need grid- and border-score significance. Make it a follow-up only if a user asks; the trigger is a request for shuffled labels.
- **Recording the thresholds on `summary_table()` output** (`df.attrs`). Phase 7 owns table layout and attrs; it reads this phase's named constants.
- **The `label_cell_types` gate fix and the object-vector frames.** Phase 5a.
- **Occupancy-weighted Rayleigh correction for head direction.** HD flagged 0/20 noise above. Mention it in the HD docstring Notes only.
- **Shuffle support for `method="glm"`.** Pooled REML couples units, so a single-unit recompute would differ from the batch. It raises.
- **`min_size` in bins making field detection bin-size dependent.** That is `detect_place_fields`, which `has_place_field` keeps unchanged.
- **Speeding the shuffle up** (reusing kernels, vectorizing over shuffles). Opt-in slowness is allowed (overview Non-Goals).
- **Shuffle support for `has_place_field`.** It is field detection, not a verdict; the shuffle verdict is `is_place_cell(criterion="shuffle")`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/stats/test_shuffle_spike_times_circular.py::test_count_preserved_and_inside_windows` | windows `[[0,100],[200,300]]`, 500 uniform spikes in them: every yielded train has 500 spikes, all inside a window, sorted |
| `…::test_shift_respects_min_shift` | one spike at 10 s, window `[0,100]`, `min_shift=20`, 500 shuffles: every value in `[30, 90]` |
| `…::test_drops_spikes_outside_windows` | a spike at 150 s with windows `[[0,100],[200,300]]` is absent from every shuffle |
| `…::test_seeded_reproducible` | `rng=7` twice gives identical arrays; `rng=8` differs |
| `…::test_rejects_too_short` | window 30 s, `min_shift=20` → `ValueError` whose message contains `Fix:` |
| `tests/encoding/test_cell_type_significance.py::test_threshold_flags_noise_as_object_vector_cell` | the 10-min noise fixture: `compute_egocentric_rates(...).classify()` flags 20/20 (measured on `main`); `compute_object_vector_rates(...).classify()` flags ≥ 18/20 (not measurable on `main`; record the observed count in the commit body). This documents the bias |
| `…::test_shuffle_detects_field_cell_fast` **(not slow; runs in CI)** | `ou_2min`, the strong field cell (Fixtures), `place_cell_significance(..., n_shuffles=20, rng=0)`: `p_value == 1/21`. Dry-run probe: observed 2.54 bits/spike against a largest shifted value of 1.93 over 5 seeds × 20 shifts |
| `…::test_shuffle_noise_false_positives` **(slow)** | the same 20 noise units, `n_shuffles=200, rng=0`: `object_vector_cell_significance`, `egocentric_object_vector_cell_significance` and `place_cell_significance` each give `p_value < 0.05` for ≤ 3/20 units (prototype: 1/20, 2/20) |
| `…::test_shuffle_pooled_false_positive_rate` **(slow)** | 100 noise units (5 spike seeds × 20) through the significance functions, `n_shuffles=200`: ≤ 10/100 flagged (prototype 3/100; Binomial(100, 0.05) 99th pct ≈ 11) |
| `…::test_shuffle_detects_true_cells` **(slow)** | the 10-min allocentric field cell through `object_vector_cell_significance` and the egocentric OVC through `egocentric_object_vector_cell_significance`: `p_value == 1/201` with `n_shuffles=200`. Prototype: 0.0050 for the field cell and for an omnidirectional OVC; the egocentric OVC's information is 2.94 against noise ≈ 0.4 |
| `…::test_free_and_population_shuffles_agree` | For each family and frame, 3 units with `unit_ids=[10, 20, 30]`, `n_shuffles=50`, `rng=0`: `is_*_cell(train_i, criterion="shuffle", unit_id=ids[i], n_shuffles=50, rng=0) == (sig[ids[i]].p_value < 0.05)`, and the p-values are equal (`assert_array_equal`). The test also asserts `min \|null − observed\| > 1e-12`, so a 1e-15 batch-vs-single statistic difference (measured) cannot flip a tie unnoticed. Reordering the population to `[30, 10, 20]` leaves every unit's p-value unchanged. Probe on `main` (spatial, binned): p `[0.6078, 0.2745, 0.0588]` alone, in the population and reordered; margin `1.1e-3` |
| `…::test_threshold_free_method_classify_agree` | `criterion="threshold"` (`"spatial_info"` for place): for every family and frame and unit, the free predicate equals the method on `compute_*_rate(...)`, and the method equals `classify()[i]`. Place is no longer excluded |
| `…::test_threshold_constants_are_the_defaults` | For each constant in 5b.4, calling the predicate, method and `classify` with no threshold keyword equals calling it with the constant's values passed explicitly. `inspect.signature(has_place_field)` defaults equal `PLACE_FIELD_DETECTION_DEFAULTS`. No threshold or `alpha` keyword has a literal non-`None` default in any predicate, method or `classify` signature |
| `…::test_has_place_field_flags_noise` (guard) | 2-min noise fixture plus the allocentric field cell (seeds fixed): `has_place_field()` (free and method) equals golden values recorded from `main`'s `is_place_cell()` before the rename (noise: 20/20 `True`, as in Evidence; field cell: record its value). The behavior is unchanged; only the name is |
| `…::test_is_place_cell_requires_criterion` | The free function and the method without `criterion` each raise `TypeError` mentioning `'criterion'`. `inspect.signature(is_place_cell).parameters['criterion']` is `KEYWORD_ONLY` with `default is inspect.Parameter.empty`. `criterion="threshold"` raises `ValueError` listing `'spatial_info'` and `'shuffle'` and naming `has_place_field` |
| `…::test_is_place_cell_spatial_info` | On the 2-min noise fixture: `[is_place_cell(env, tr, t, p, criterion="spatial_info") for tr in noise]` equals `compute_spatial_rates(env, noise, t, p).classify()` elementwise (Evidence: 15/20 at 2 min; record the observed count), and `rates[i].is_place_cell(criterion="spatial_info") == classify()[i]` |
| `…::test_is_place_cell_shuffle_rejects_noise` | 5 of the 2-min noise units, `criterion="shuffle", n_shuffles=50, rng=0`: at most 2 of 5 flagged (under the null each unit is flagged with probability 2/51 ≈ 0.04, so P(≥ 3 of 5) < 0.001), while `has_place_field` flags 5/5. Not `slow`, so CI runs it |
| `…::test_methods_have_no_shuffle` | `rates[0].is_place_cell(criterion="shuffle")` raises `ValueError` whose `Fix:` names `is_place_cell(env, spike_times, times, positions, criterion='shuffle')`. `rates.classify(criterion="shuffle")` raises `TypeError`, since there is no such keyword. No result class has a `shuffle_test` attribute |
| `…::test_mode_keywords_raise` | Free predicates, per family and frame: a threshold keyword with `criterion="shuffle"` (e.g. `is_place_cell(..., criterion="shuffle", min_info=0.3)`) and a shuffle keyword with the threshold criterion (e.g. `is_spatial_view_cell(..., n_shuffles=10)`, `unit_id=3`) each raise `ValueError` with a `Fix:` line. Passing two such keywords names both in one message. Head-direction `alpha` is accepted in both modes |
| `…::test_free_predicates_raise_on_bad_input` | a swapped `(positions, times)` call raises `ValueError` for every free predicate (on `main` it returns `False`); so does `has_place_field` |
| `…::test_results_do_not_retain_inputs` | For each family, singular, plural and an indexed child `rates[0]`: no `np.ndarray` attribute in `vars(result)` (the `env` excluded) shares memory with the caller's `spike_times`, `times`, `positions` or `headings`, and no attribute is a callable or `functools.partial`. Overwriting the caller's `positions` afterwards leaves `firing_rate` and `classify()` unchanged |
| `…::test_significance_isolated_from_caller_mutation` (parametrized over every `*_significance` function and every array-valued argument it takes, including `speed`, `headings` and `object_positions`) | Monkeypatch the plural compute function as the significance module looks it up, with a wrapper that, after its first call, overwrites the parametrized caller array: `[:] = 0`, or `+= 5.0` for spike trains; for `speed`, `[:] = 1e6`, which flips the speed gate. Also covered: the caller's `positions[:] = 0` and `trains[0][:] += 5.0`. `place_cell_significance(..., n_shuffles=20, rng=0)` p-values then equal a clean run on fresh arrays (probe on `main`: equal) |
| `…::test_significance_labels_and_glm` | Results are keyed by unit label, in input order, and `len(result) == n_units`. A group keyed `[10, 20]` with `unit_ids=[20, 10]` raises `ValueError` listing both. `unit_ids=[3, 3, 7]` raises `ValueError` naming `[3]` (Phase 1 Task 7's `resolve_unit_ids`), so no label-keyed entry is silently lost. `method="glm"` raises `ValueError` with `Fix:` |
| `…::test_invalid_criterion` | `is_head_direction_cell(..., criterion="percentile")` → `ValueError` listing `"threshold"` and `"shuffle"`; `is_place_cell(..., criterion="percentile")` lists `"spatial_info"` and `"shuffle"` |
| Phase 4b `tests/docs/test_flagship_docstrings.py` and `test_docstring_sections.py` | pass with the six new `FLAGSHIP` entries; every forwarded compute keyword of each significance function is documented |

Mark every test over 5 s `@pytest.mark.slow`; run them with `uv run pytest -m "slow and not napari" -n 4`. Delete the tests that assert the removed `detect_*` aliases warn. Update the tests that call `is_place_cell` without `criterion`:

- `tests/encoding/test_naming_contract.py` (:199-215) and `tests/encoding/test_method_param.py` call it for field detection; switch them to `has_place_field`.
- Phase 3's `test_predicates_forward_time_windows` should spy on `has_place_field` and on `is_place_cell(criterion="spatial_info")`.

## Fixtures

Reuse Phase 5a's `tests/encoding/conftest.py` fixtures (`_ou_trajectory`, `ou_10min`, `ou_2min`, `noise_trains`, the allocentric field cell, the egocentric OVC, `obj`, `env`). Add:

- **Strong field cell (fast tests):** on `ou_2min`, `PlaceCellModel(env, center=obj + [20, 0], width=10, max_rate=20)`, spikes `generate_poisson_spikes(..., seed=3)` (87 spikes in the dry run).

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- Every task is implemented as specified; the six threshold constants exist under exactly these names.
- The "Deliberately not in this phase" list is honored; no scope creep into adjacent phases.
- Validation slice tests pass; slow tests are marked, and the fast significance tests are **not** marked slow.
- Tests aren't trivial: they exercise the asserted behavior, not tautologies, and shared setup is in fixtures (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are actually removed: the `try/except → False` blocks, the `detect_*` aliases, `classify(min_spatial_info=)`.
- User-facing documentation listed as tasks is updated, not deferred.
- Run `scientific-code-change-audit` on the significance engine and the `>=` change.
