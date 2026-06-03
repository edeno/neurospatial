# Phase 5 — Encoding: directional & phase-precession correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#input-validation-helpers)

Five silent-correctness defects in head-direction and phase-precession analysis, all scoped to `src/neurospatial/encoding/` (directional + phase-precession surface only). Each produces a plausible-but-wrong number or a swallowed error rather than a crash, so each fix ships with a fail-before / pass-after regression test.

**Soft dependency on phase 4.** This phase imports `validate_finite` / `validate_lengths` from `src/neurospatial/_validation.py`, the new top-level module created in **phase 4** (see [shared contracts](shared-contracts.md#input-validation-helpers)). Land phase 4 first, or — if phase 5 lands first — create that module here per the contract verbatim and let phase 4 adopt it. Do **not** fork a private copy inside `encoding/`. The underlying `rayleigh_test` (phase 4) already implements correct count-weighting; this phase only fixes the *caller* that feeds it the wrong weights.

---

**Inputs to read first:**

- [../../../../src/neurospatial/encoding/directional.py:587](../../../../src/neurospatial/encoding/directional.py) — `DirectionalRateResult.rayleigh_pvalue()` (lines 587–662). The bug: line 660 passes `firing_rate` (Hz) as `weights=` to `rayleigh_test`, so the test's effective sample size becomes a sum of *rates*, not spike counts → scale-dependent, invalid p-values.
- [../../../../src/neurospatial/encoding/directional.py:155](../../../../src/neurospatial/encoding/directional.py) — `DirectionalRateResult` field list (lines 155–159): `firing_rate, occupancy, bin_centers, bin_size, bandwidth`. **No spike-count field exists** — the counts must be plumbed in (added as a new optional field) for the Rayleigh fix.
- [../../../../src/neurospatial/encoding/directional.py:929](../../../../src/neurospatial/encoding/directional.py) — `DirectionalRatesResult` field list (lines 929–933) and `__getitem__` (lines 952–979) which rebuilds per-neuron `DirectionalRateResult`s; must forward the new counts field.
- [../../../../src/neurospatial/encoding/directional.py:502](../../../../src/neurospatial/encoding/directional.py) — `tuning_width()` (lines 502–549). The half-max crossing search (lines 522–546) reads `rates[idx]` directly; a NaN neighbour at the crossing makes `rates[idx] < half_max` false forever → the `for/else` returns NaN.
- [../../../../src/neurospatial/encoding/directional.py:1466](../../../../src/neurospatial/encoding/directional.py) — `compute_directional_rate` construction (lines 1466–1525): raw `spike_counts` is computed at line 1472 but discarded; `DirectionalRateResult(...)` at line 1519 must now carry it.
- [../../../../src/neurospatial/encoding/directional.py:1745](../../../../src/neurospatial/encoding/directional.py) — `compute_directional_rates` (lines 1745–1800): `_process_neuron` (line 1746) returns only firing rate; it must also surface counts so the batch result at line 1794 can store `(n_neurons, n_bins)` counts. Empty-neuron branch at line 1729–1743 must store an empty counts array.
- [../../../../src/neurospatial/encoding/directional.py:1884](../../../../src/neurospatial/encoding/directional.py) — `is_head_direction_cell` convenience fn (lines 1884–1895): `try: compute_directional_rate(...) ... except (ValueError, RuntimeError): return False` swallows genuine input errors as "not an HD cell".
- [../../../../src/neurospatial/encoding/_directional_binning.py:182](../../../../src/neurospatial/encoding/_directional_binning.py) — `compute_directional_occupancy` (lines 182–190): `np.digitize(NaN) - 1` yields `n_bins`, then line 185 (`frame_bins[frame_bins >= n_bins] = 0`) folds non-finite headings into **bin 0**, inflating occupancy there.
- [../../../../src/neurospatial/encoding/_directional_binning.py:263](../../../../src/neurospatial/encoding/_directional_binning.py) — `_bin_spikes_with_precomputed_directional_bins` (lines 263–273): identical fold — a spike whose interpolated heading index points at a NaN frame lands in bin 0 (line 271). Occupancy and spike counts must mask the **same** frames.
- [../../../../src/neurospatial/encoding/phase_precession.py:332](../../../../src/neurospatial/encoding/phase_precession.py) — `phase_precession` (lines 332–353): `optimal_slope` is fit by maximizing residual concentration (lines 277–330), but `correlation, pval` come from `circular_linear_correlation(angles=phases, linear_values=positions)` at line 342 — **the fitted slope is ignored**, so the reported p-value tests a different (slope-free) hypothesis than the fit.
- [../../../../src/neurospatial/encoding/phase_precession.py:399](../../../../src/neurospatial/encoding/phase_precession.py) — `has_phase_precession` (lines 399–407): `try: phase_precession(...) ... except ValueError: return False` swallows input errors (bad `angle_unit`, length mismatch, too few spikes presented as a true error) as "no precession".
- [../../../../src/neurospatial/encoding/phase_precession.py:247](../../../../src/neurospatial/encoding/phase_precession.py) — `phase_precession` input handling (lines 247–260): `_validate_paired_input` already drops NaN pairs and enforces `min_spikes`; the new shuffle path must run **after** this on the cleaned arrays.
- [../../../../src/neurospatial/stats/circular.py:485](../../../../src/neurospatial/stats/circular.py) — `rayleigh_test` signature/contract (lines 485–612): `weights` are documented as **counts/frequencies** and the statistic is `z = sum(weights) * R**2`. Confirms the fix is purely caller-side: pass spike counts, not Hz.
- [../../../../src/neurospatial/stats/circular.py:615](../../../../src/neurospatial/stats/circular.py) — `circular_linear_correlation(angles, linear_values, *, angle_unit)` — used to build the shuffle null in the phase-precession fix.
- [../../../../src/neurospatial/encoding/_validation.py:184](../../../../src/neurospatial/encoding/_validation.py) — `validate_trajectory` (lines 184–245): confirms headings are checked only for ndim/length, **not** finiteness — this is why NaN headings reach the binner today.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — `validate_finite(a, *, name, allow_nan=False)` and `validate_lengths(name_to_array)` from `src/neurospatial/_validation.py`. Use `validate_finite(..., allow_nan=False)` for the heading-finiteness *guard at public entry points* only where NaN headings are a hard error; use explicit masking (below) where NaN headings are legitimately dropped. **Do not weaken**: `validate_finite` raises on Inf always, and these helpers never silently coerce. The classifier try/except fix relies on these guards running *outside* the `try` so genuine input errors propagate.

---

## Tasks

### Task 1 — Rayleigh p-value: weight by spike COUNTS, not firing rate

The underlying `rayleigh_test` already does count-weighting correctly (`z = sum(weights) * R²`, stats/circular.py:594). The defect is that `rayleigh_pvalue()` hands it `firing_rate` (Hz). Firing rate = counts / occupancy, so using it as the count weight makes the test statistic depend on bin dwell-time and the absolute rate scale — a cell recorded at 2× the frame rate gets a different p-value for identical tuning.

`DirectionalRateResult` does not currently retain the raw spike counts, so plumb them through as a new **optional** field (keeps all existing positional constructions valid).

**1a. Add `spike_counts` field to `DirectionalRateResult`** (directional.py:155–159). Place it last with a default so existing keyword/positional constructions (including the many docstring `>>>` examples) keep working:

```python
    firing_rate: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    bandwidth: float | None
    spike_counts: ArrayLike | None = None
```

Document the new field in the class `Parameters`/`Attributes` docstring:

```
    spike_counts : ArrayLike or None, optional
        Raw (unsmoothed) spike count per angular bin, shape (n_bins,). Used as
        count weights for the Rayleigh test. None for results constructed
        without counts (e.g. from external tuning curves), in which case
        rayleigh_pvalue() falls back to occupancy-implied counts.
```

**1b. Rewrite `rayleigh_pvalue()`** (directional.py:587–662, body at 649–662):

```python
        from neurospatial.stats.circular import rayleigh_test

        centers = np.asarray(self.bin_centers, dtype=np.float64)
        rates = np.asarray(self.firing_rate, dtype=np.float64)

        # Count weights for the Rayleigh test. The test treats weights as
        # FREQUENCIES (z = sum(weights) * R**2); firing rate in Hz is the wrong
        # quantity because it is occupancy- and rate-scale-dependent.
        if self.spike_counts is not None:
            counts = np.asarray(self.spike_counts, dtype=np.float64)
        else:
            # Fallback for results built without raw counts: reconstruct an
            # integer-like count from rate * occupancy. Still a count, so the
            # statistic remains scale-correct.
            occ = np.asarray(self.occupancy, dtype=np.float64)
            counts = rates * occ

        # Drop bins with no valid weight (unvisited -> NaN rate / NaN count,
        # or zero count). A bin with zero spikes contributes nothing to the
        # resultant and must not be passed as a zero weight that still counts
        # toward n.
        valid = np.isfinite(centers) & np.isfinite(counts) & (counts > 0)
        if valid.sum() < 3:
            return float(np.nan)

        _, pval = rayleigh_test(centers[valid], weights=counts[valid])
        return pval
```

Update the method's `Notes` so the documented statistic reflects counts (the existing prose at lines 600–622 talks about "weighted by firing rates" — correct it to "weighted by spike counts").

**1c. Forward `spike_counts` from the compute functions.**

`compute_directional_rate` (directional.py:1519–1525) — pass the **unsmoothed** `spike_counts` (the variable computed at line 1472, *not* `spike_counts_smooth`):

```python
    return DirectionalRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy,
        bin_centers=bin_centers,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        spike_counts=spike_counts,
    )
```

(If the JAX branch at lines 1511–1517 converts arrays, also convert `spike_counts` for consistency; counts are weights for a NumPy stats call so leaving them as NumPy is acceptable — pick one and keep it consistent with the other fields.)

**1d. Carry counts through `DirectionalRatesResult`.** Add a matching optional field (directional.py:929–933):

```python
    firing_rates: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    bandwidth: float | None
    spike_counts: ArrayLike | None = None  # shape (n_neurons, n_bins)
```

In `__getitem__` (directional.py:952–979) forward the per-neuron slice:

```python
        counts = self.spike_counts
        return DirectionalRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            bandwidth=self.bandwidth,
            spike_counts=(
                None if counts is None else np.asarray(counts)[idx]
            ),
        )
```

In `compute_directional_rates`, have `_process_neuron` (directional.py:1746–1768) return `(firing_rate, spike_counts)` (the unsmoothed counts from line 1748), stack both, and pass the `(n_neurons, n_bins)` counts array to the result at line 1794. The empty-neuron branch (lines 1729–1743) stores `spike_counts=np.empty((0, n_bins))`.

### Task 2 — Reject non-finite headings out of occupancy AND spike counts

A NaN/Inf heading must not be silently counted into bin 0. The two sites that fold out-of-range digitize indices to 0 (`_directional_binning.py:185` for occupancy, `:271` for spikes) treat non-finite the same as an exact-2π edge case, corrupting bin 0 of both arrays. Fix at the binning layer so occupancy and spike counts mask the **same** frames.

**2a. `compute_directional_occupancy`** (`_directional_binning.py`, lines 173–190). After wrapping (line 174) and computing `time_deltas` (line 179), mask non-finite frames out of the per-frame contribution rather than folding them to bin 0:

```python
    # Wrap headings to [0, 2*pi). Non-finite headings stay non-finite (NaN % x
    # == NaN), so we can detect and exclude them below.
    headings_wrapped = headings_rad % (2 * np.pi)

    # Each frame i contributes time until frame i+1; last frame excluded.
    time_deltas = np.diff(times)
    frame_headings = headings_wrapped[:-1]

    # Exclude frames whose heading is non-finite (NaN/Inf). Folding these into
    # bin 0 via digitize would inflate bin-0 occupancy.
    finite = np.isfinite(frame_headings)

    frame_bins = np.digitize(frame_headings[finite], bin_edges) - 1
    # Exact-2*pi edge wraps to bin 0 (legitimate); non-finite never reach here.
    frame_bins[frame_bins >= n_bins] = 0

    occupancy = np.bincount(
        frame_bins, weights=time_deltas[finite], minlength=n_bins
    ).astype(np.float64)

    return occupancy, bin_centers
```

**2b. `_bin_spikes_with_precomputed_directional_bins`** (`_directional_binning.py`, lines 253–273). A spike inherits the heading of its frame via `headings_wrapped[spike_indices]` (line 265). Drop spikes whose frame heading is non-finite so they are not folded to bin 0:

```python
    spike_hd = headings_wrapped[spike_indices]

    # Exclude spikes landing on a frame with a non-finite heading. These would
    # be folded into bin 0 by the >= n_bins clamp below.
    finite = np.isfinite(spike_hd)
    spike_hd = spike_hd[finite]
    if spike_hd.size == 0:
        return spike_counts

    spike_bins = np.digitize(spike_hd, bin_edges) - 1
    spike_bins[spike_bins >= n_bins] = 0

    return np.bincount(spike_bins, minlength=n_bins).astype(np.float64)
```

This keeps occupancy and counts mutually consistent (a frame excluded from occupancy also has its spikes excluded), so firing rate in that bin is unaffected rather than divided by an inflated occupancy.

> **Note on `_precompute_directional_bins`** (`_directional_binning.py:195–238): it wraps but does not mask; leaving it unchanged is correct because masking now happens at each consumer (occupancy + spike binning) where the matching `time_deltas` / `spike_indices` are available.

### Task 3 — `tuning_width`: interpolate across NaN bins at the half-max crossing

`tuning_width()` (directional.py:502–549) walks outward from the peak and stops at the first bin below half-max. A NaN (unvisited) bin between the peak and the crossing makes the `<` comparison false, so the search runs off the end and returns NaN even though a finite crossing exists. Fix by searching over the **finite** bins only, in circular order, and interpolating between consecutive finite samples.

Replace the two directional search loops (lines 520–546) with a helper that operates on finite samples. Insert this private module-level function near the class (or as a static method) and call it for each direction:

```python
def _half_max_halfwidth(
    rates: NDArray[np.float64],
    peak_idx: int,
    half_max: float,
    bin_size: float,
    *,
    step: int,
) -> float:
    """Distance (radians) from peak to the half-max crossing in one direction.

    Walks circularly from ``peak_idx`` in ``step`` (+1 or -1), skipping NaN
    bins, and linearly interpolates the crossing between the last finite bin
    above half-max and the first finite bin below it. ``offset`` counts bins
    from the peak so the geometric distance is preserved even when NaN bins are
    skipped. Returns NaN if no finite below-half-max bin is found.
    """
    n_bins = len(rates)
    prev_offset = 0
    prev_rate = rates[peak_idx]
    for offset in range(1, n_bins // 2 + 1):
        idx = (peak_idx + step * offset) % n_bins
        r = rates[idx]
        if not np.isfinite(r):
            continue  # skip unvisited bin; keep accumulating offset distance
        if r < half_max:
            denom = r - prev_rate
            frac = 0.0 if denom == 0 else (half_max - prev_rate) / denom
            # Interpolate between the last finite-above bin (prev_offset) and
            # this finite-below bin (offset), in units of bins-from-peak.
            return (prev_offset + frac * (offset - prev_offset)) * bin_size
        prev_offset = offset
        prev_rate = r
    return float(np.nan)
```

Then in `tuning_width()`, after computing `peak_idx`, `half_max`, and the flat-curve guard (lines 506–513), replace the right/left loops with:

```python
        right_width = _half_max_halfwidth(
            rates, peak_idx, half_max, self.bin_size, step=+1
        )
        left_width = _half_max_halfwidth(
            rates, peak_idx, half_max, self.bin_size, step=-1
        )

        # Average the finite half-widths. If only one side crosses (the other
        # is masked-out NaN all the way round), report the single finite side
        # rather than NaN.
        halves = np.array([left_width, right_width])
        finite = halves[np.isfinite(halves)]
        if finite.size == 0:
            return float(np.nan)
        return float(finite.mean())
```

Keep the existing flat-curve guard (`np.nanmin(rates) >= half_max → NaN`) unchanged.

### Task 4 — Classifiers: validate inputs OUTSIDE the try; only non-significance maps to False

`is_head_direction_cell` (directional.py:1884–1895) and `has_phase_precession` (phase_precession.py:399–407) wrap the entire computation in `try/except (ValueError ...)` and return `False`. This hides genuine input errors (length mismatch, bad `angle_unit`, non-finite headings) behind a "not a cell / no precession" answer. The narrow case that legitimately maps to `False` is *successful computation that is simply not significant*. Validate inputs first, outside the try.

**4a. Add a shared helper** in `encoding/_validation.py` (this is the domain validator that may call the top-level helpers per the contract):

```python
from neurospatial._validation import validate_finite, validate_lengths


def validate_classifier_trajectory(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    context: str,
) -> None:
    """Validate (spike_times, times, headings) for directional classifiers.

    Raises on genuine input errors so they propagate instead of being
    swallowed as a False classification. Does NOT enforce statistical
    significance — that is decided after the (valid) computation.
    """
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    validate_trajectory(times, headings=headings, context=context)
    validate_finite(times, name="times")
    validate_spike_times(spike_times, context=context)
```

(`headings` are intentionally **not** passed through `validate_finite` here — non-finite headings are a *legitimate, droppable* condition handled by Task 2's masking, not a hard error. The guard is for shape/length/timestamp sanity that should surface as errors.)

**4b. `is_head_direction_cell`** (directional.py:1884–1895):

```python
    # Validate inputs OUTSIDE the try so genuine input errors propagate.
    validate_classifier_trajectory(
        spike_times, times, headings, context="is_head_direction_cell"
    )

    try:
        result = compute_directional_rate(
            spike_times, times, headings,
            bin_size=bin_size, bandwidth=bandwidth, angle_unit=angle_unit,
        )
    except (ValueError, RuntimeError):
        # Computation succeeded validation but produced no usable tuning
        # (e.g. no spikes in any visited bin) -> not an HD cell.
        return False
    return result.is_head_direction_cell(min_mvl=min_mvl, alpha=alpha)
```

Note `result.is_head_direction_cell(...)` is moved out of the `try` so an unexpected error inside the classifier method is not masked either.

**4c. `has_phase_precession`** (phase_precession.py:399–407). Validate the paired inputs and `angle_unit` first; only an *insufficient-data* `ValueError` from the fit maps to `False`:

```python
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")
    positions = np.asarray(positions, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    validate_lengths({"positions": positions, "phases": phases})

    try:
        result = phase_precession(positions, phases, angle_unit=angle_unit)
    except ValueError:
        # Too few spikes after NaN-dropping -> cannot assess precession.
        return False
    return (
        result.pval < alpha
        and result.correlation >= min_correlation
        and result.slope < 0
    )
```

Keep the `min_spikes`-driven `ValueError` from `phase_precession` mapping to `False` (insufficient data is a legitimate "can't tell → False"), but length mismatch and bad `angle_unit` now raise.

> The egocentric (`is_object_vector_cell`) and view (`is_spatial_view_cell`) classifiers share this swallow-the-ValueError pattern but live in `egocentric.py` / `view.py` and are fixed in **phases 22 / their own phases** — see "Deliberately not in this phase". `validate_classifier_trajectory` is written generically so those phases can reuse it.

### Task 5 — Phase-precession p-value must reflect the fitted slope

`phase_precession` (phase_precession.py:332–353) fits `optimal_slope` by maximizing residual concentration, then reports `correlation, pval` from `circular_linear_correlation(phases, positions)` (line 342) — a slope-agnostic statistic. The p-value therefore tests "is there *any* circular-linear association" rather than "is the *fitted precession* significant", and is identical no matter what slope the optimizer lands on. Replace the decoupled p-value with a **shuffle null re-fit at the fitted slope**.

Replace lines 341–344 (the `circular_linear_correlation` call) with a permutation test whose statistic is the mean resultant length of residuals at the *re-fit* slope, matching the quantity the fit maximizes:

```python
    # Observed fit quality = MRL of residuals at the fitted slope.
    observed_mrl = mean_resultant_length  # = -neg_mrl, computed above

    # Shuffle null: break the phase<->position pairing, re-fit the slope on
    # each shuffle, and compare the resulting MRL. This makes the p-value test
    # the SAME hypothesis the slope fit optimizes (a real position-dependent
    # phase relationship), instead of the slope-free circular-linear
    # correlation that ignores the fitted slope entirely.
    rng = np.random.default_rng(rng)
    n_shuffles_eff = int(n_shuffles)
    null_mrls = np.empty(n_shuffles_eff, dtype=np.float64)
    for i in range(n_shuffles_eff):
        shuffled_pos = rng.permutation(positions)
        null_mrls[i] = _best_residual_mrl(
            phases, shuffled_pos, slope_bounds
        )
    # +1 smoothing avoids p == 0 (Phipson & Smyth 2010).
    pval = float((np.sum(null_mrls >= observed_mrl) + 1) / (n_shuffles_eff + 1))

    # Report the circular-linear correlation alongside as a descriptive
    # effect size (documented as slope-independent), NOT as the significance.
    correlation, _ = circular_linear_correlation(
        angles=phases, linear_values=positions
    )

    return PhasePrecessionResult(
        slope=float(optimal_slope),
        slope_units=slope_units,
        offset=offset,
        correlation=correlation,
        pval=pval,
        mean_resultant_length=float(mean_resultant_length),
    )
```

Factor the grid+refine slope search (lines 283–333) into a reusable `_best_residual_mrl(phases, positions, slope_bounds) -> float` so the shuffle loop re-fits identically to the observed fit (do **not** reuse the observed `optimal_slope` for shuffles — that would bias the null). The existing observed fit can call the same helper.

> **Performance (required — address one of these, do not ship the naive form).** The shuffle null re-runs the full adaptive grid + Brent slope search **per shuffle**, and `has_phase_precession` (Task 4c) inherits this on *every* classifier call. At the default `n_shuffles` this is a large latency cliff (a full O(grid×refine) slope fit ×`n_shuffles` per call). The executor MUST pick **one** mitigation and document it:
> - use a **coarse FIXED grid** for the per-shuffle re-fit (e.g. evaluate `_best_residual_mrl` on a fixed coarse slope grid without the Brent refine step) instead of the full adaptive search — the null only needs a comparable statistic, not a precision slope; **or**
> - set a **smaller `n_shuffles` default on the classifier path** (`has_phase_precession` passes e.g. `n_shuffles=200` while `phase_precession` keeps its `n_shuffles=1000` default); **or**
> - add an explicit **performance note + parameter** in both docstrings stating the per-shuffle cost and how to bound it (e.g. exposing the shuffle-grid coarseness), so callers can opt into a cheaper null.
> Keep the correctness of the re-fit unbiased (still re-fit per shuffle on the shuffled pairing); the mitigation only trades slope *precision*/`n_shuffles` for latency, it must not reuse the observed slope.

Add the two new keyword-only parameters to the `phase_precession` signature (after `min_spikes`, phase_precession.py:185):

```python
    n_shuffles: int = 1000,
    rng: int | np.random.Generator | None = None,
```

Update the `phase_precession` docstring:
- `Parameters`: document `n_shuffles` and `rng`.
- `Returns`/`Notes`: state explicitly that `pval` is now a **shuffle p-value at the fitted slope**, and that `correlation` is a slope-independent descriptive circular-linear effect size (resolving the decoupling the review flagged). Update `PhasePrecessionResult.correlation` / `.pval` attribute docs (phase_precession.py:88–91) to match.

`has_phase_precession` (Task 4c) keeps using `result.pval`, which now correctly tracks the fitted slope, plus its existing `slope < 0` directionality gate.

### Task 6 — Docstring / doctest sweep for touched public surface

- `DirectionalRateResult` / `DirectionalRatesResult` doctests (directional.py:35, 135, 899, …) construct results without `spike_counts`; the new optional field keeps them valid — confirm by running doctests. Add one doctest showing `rayleigh_pvalue()` on a result built *with* counts vs. the occupancy-fallback path.
- `phase_precession` examples (phase_precession.py:235–242) — add `rng=0` to any example asserting a specific `pval` so the shuffle p-value is deterministic.
- No QUICKSTART/API_REFERENCE example references `tuning_width`, raw `rayleigh_pvalue` weighting, or the shuffle params today (grep-confirm during implementation); if one is added by another phase, defer to phase 23's doc sweep.

---

## Deliberately not in this phase

- **OVC classifier disagreement + `is_object_vector_cell` delegation + egocentric naming** (`encoding/egocentric.py`) → **phase 22**. The same swallow-`ValueError`-return-`False` anti-pattern lives in `is_object_vector_cell` / `is_spatial_view_cell`; Task 4 writes `validate_classifier_trajectory` generically so phase 22 / the view phase can adopt it, but this PR does **not** edit `egocentric.py` or `view.py`.
- **Egocentric polar bandwidth cm vs. rad** (`from_polar_egocentric`, polar env method disabling) → **phase 19**. Out of scope; do not touch polar binning here.
- **Result-mixin `to_dataframe`/`summary`/`to_xarray` for directional results** → **phases 17 / 20**. This phase only *adds a field*; it does not restructure the result classes onto the new `ResultMixin`.
- **Generalizing the shuffle helper into `stats/shuffle.py`** → the phase-precession shuffle is local to the fit and stays in `phase_precession.py`; the population/stats shuffle p-value fix is **phase 4**. Do not refactor across the stats boundary here.
- **`compute_directional_rate` "env-first" signature** — intentionally heading-native (documented exception in CLAUDE.md and the function docstring); not a naming change for phase 22.

---

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_rayleigh_pvalue_scale_invariant` | Two `DirectionalRateResult`s with identical *tuning shape* but different occupancy/rate scale (e.g. counts ×3, occupancy ×3 → same Hz; and same counts with halved occupancy → doubled Hz) yield the **same** `rayleigh_pvalue()` when built with `spike_counts`. **Fail-before**: rate-weighted p-values differ across the rate-scaled pair. |
| `test_rayleigh_pvalue_matches_counts_weighting` | `result.rayleigh_pvalue()` equals `rayleigh_test(bin_centers[valid], weights=spike_counts[valid])[1]` for a hand-built result. Fail-before: equals the rate-weighted value instead. |
| `test_compute_directional_rate_stores_spike_counts` | `compute_directional_rate(...).spike_counts` is the unsmoothed per-bin count; `sum == number of spikes inside the recording window`. |
| `test_directional_rates_getitem_forwards_counts` | `compute_directional_rates(...)[i].spike_counts` equals row `i` of the batch counts and round-trips `rayleigh_pvalue()`. |
| `test_nan_heading_not_folded_into_bin0` | With a trajectory where a contiguous block of headings is NaN, `compute_directional_occupancy` puts **zero** of that block's time into bin 0; bin-0 occupancy equals the bin-0 occupancy of the same trajectory with NaN frames removed (within float tol). **Fail-before**: bin 0 is inflated by the NaN block's `time_deltas`. |
| `test_nan_heading_spikes_excluded` | Spikes occurring during NaN-heading frames do not increment bin 0 of `bin_directional_spike_train`; total binned spikes == spikes on finite-heading frames. |
| `test_inf_heading_rejected_or_dropped` | An `Inf` heading is excluded from occupancy and counts identically (no bin-0 inflation). |
| `test_tuning_width_skips_nan_neighbor` | A sharply-tuned curve with a single NaN bin inserted *between the peak and the half-max crossing* returns a finite width within tolerance of the same curve without the NaN. **Fail-before**: returns NaN. |
| `test_tuning_width_one_sided_nan` | A curve where one side is entirely NaN (masked) returns the finite single-side half-width, not NaN. |
| `test_is_head_direction_cell_raises_on_bad_input` | `is_head_direction_cell(spike_times, times, headings)` with mismatched `len(times) != len(headings)` **raises `ValueError`** (not returns `False`). Fail-before: returns `False`. |
| `test_is_head_direction_cell_false_on_uniform` | Valid inputs with uniform (non-directional) firing return `False` (the legitimate not-significant path still works). |
| `test_is_head_direction_cell_recovers_hd_cell` *(end-to-end)* | Simulate a von Mises HD cell over a trajectory uniformly covering all directions (use `simulation` if available, else inline Poisson draws with a fixed seed); `is_head_direction_cell(...)` returns `True` and the recovered `preferred_direction()` is within ~10° of the planted direction. |
| `test_has_phase_precession_raises_on_length_mismatch` | `has_phase_precession(positions, phases)` with unequal lengths **raises `ValueError`**. Fail-before: returns `False`. |
| `test_has_phase_precession_false_on_insufficient_spikes` | Fewer than `min_spikes` valid pairs still returns `False` (insufficient-data path preserved). |
| `test_phase_precession_pval_tracks_fitted_slope` | For a synthetic dataset with a strong **negative** planted slope, `pval < 0.05`; for a phase-shuffled copy of the *same* phases against the *same* positions, `pval` is large (≳ 0.5). Both with fixed `rng`. **Fail-before**: pval comes from slope-free `circular_linear_correlation` and does not distinguish the shuffled control from a true fit at the planted slope where the correlation magnitude is unchanged. |
| `test_phase_precession_pval_deterministic` | Same inputs + same `rng` → identical `pval` across two calls. |
| `test_phase_precession_correlation_is_descriptive` | `result.correlation` still in `[0, 1]`; documented as slope-independent (smoke: equals `circular_linear_correlation(phases, positions)[0]`). |
| `test_has_phase_precession_within_time_budget` | A single `has_phase_precession(...)` call (on the seeded `precessing_spikes` fixture, using whatever shuffle mitigation Task 5 chose — coarse fixed grid and/or the smaller classifier-path `n_shuffles` default) completes within a reasonable wall-clock budget (e.g. well under ~1 s on the fixture); guards against the per-shuffle full-adaptive-fit latency cliff. Mark `@pytest.mark.slow` if timing proves noisy in CI. |

Mark `test_is_head_direction_cell_recovers_hd_cell` as the end-to-end recovery test; if it draws many spikes/shuffles, mark `@pytest.mark.slow`. The `phase_precession` shuffle tests use a small `n_shuffles` (e.g. 200) with a fixed seed to stay fast.

## Fixtures

Add to `tests/encoding/conftest.py` (synthesized, seeded — no checked-in data):

- `uniform_heading_trajectory` — `times = np.linspace(0, 60, 1800)`, headings sweeping all directions uniformly; returns `(times, headings)`.
- `von_mises_hd_spikes` — given the trajectory and a planted `preferred_direction` + concentration, draws Poisson spikes with a fixed `np.random.default_rng(seed)`; returns `spike_times`. Reused by the Rayleigh, classifier, and HD-recovery tests.
- `nan_block_heading_trajectory` — a copy of `uniform_heading_trajectory` with a contiguous index block set to `np.nan` (and a parametrized `np.inf` variant) plus the matching `_clean` version with those frames removed, so the "bin-0 not inflated" tests can compare against ground truth.
- `precessing_spikes` — positions sweeping a field with phases following a planted negative slope plus von-Mises jitter (seeded); plus a `phase_shuffled` variant (same phases permuted) for the null-control assertion.

Existing `tests/encoding/test_encoding_directional.py`, `test_encoding_directional_binning.py`, and `test_encoding_phase_precession.py` are the homes for the new tests — extend them rather than adding new modules.

## Review

Before opening the PR, dispatch `code-reviewer` (or `scientific-code-change-audit`, given these are scientific-quantity changes) against the diff. Confirm:

- Every task (1–6) is implemented as specified; the Rayleigh fix weights by **spike counts**, never Hz, and `rayleigh_test` itself is **not** modified (phase 4 owns it).
- The "Deliberately not in this phase" list is honored: **no edits** to `egocentric.py`, `view.py`, `stats/`, polar/factory code, or the result-mixin surface. `validate_classifier_trajectory` is added but only *wired* into the directional classifier.
- NaN/Inf headings are masked out of occupancy **and** spike counts at the same frames (Task 2) — verify a single shared `finite` mask logic, not two divergent ones.
- The phase-precession shuffle null **re-fits** the slope per shuffle (does not reuse the observed slope), is deterministic under `rng`, and the docstring states `pval` is now a fitted-slope shuffle p-value while `correlation` is descriptive/slope-independent.
- Validation slice passes, including the fail-before assertions (reviewer spot-checks at least the scale-invariance, NaN-bin-0, tuning-width-NaN, classifier-raises, and pval-tracks-slope tests by reverting the fix locally).
- Tests aren't trivial — the scale-invariance and shuffle-control tests exercise the asserted behavior, not tautologies; shared setup is in `conftest.py` fixtures, not copy-pasted (`testing-anti-patterns`).
- `uv run pytest tests/encoding/ -q`, `uv run pytest --doctest-modules src/neurospatial/encoding/directional.py src/neurospatial/encoding/phase_precession.py`, `uv run ruff check . && uv run ruff format --check .`, and `uv run mypy src/neurospatial/encoding/` all pass.
- Docstrings, test names, and module names do **not** reference this plan, "phase 5", or any milestone.
- No orphaned code: the old per-direction `tuning_width` loops and the old `circular_linear_correlation`-as-significance line are removed, not left dead alongside the new paths.
