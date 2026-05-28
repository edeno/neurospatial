# Phase 2 — Encoding recovery + JAX parity

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add end-to-end recovery tests that drive the public `compute_*_rate` functions on simulated ground truth and assert that recovered tuning parameters match the truth within one bin / Poisson tolerance. Today, the encoding tests construct result dataclasses from synthetic firing rates and read back the trivial accessors — exercising plumbing, not math. Also closes a NumPy↔JAX parity gap for the directional path.

**Inputs to read first:**

- [src/neurospatial/encoding/spatial.py](../../../src/neurospatial/encoding/spatial.py) — `compute_spatial_rate(env, spike_times, times, positions, *, smoothing_method, bandwidth, min_occupancy, ...)`. Returns `SpatialRateResult` with `.firing_rate`, `.occupancy`, `.env`, `.peak_location()`, `.spatial_information()`.
- [src/neurospatial/encoding/directional.py](../../../src/neurospatial/encoding/directional.py) — `compute_directional_rate(spike_times, times, headings, *, n_bins, ...)` and `compute_directional_rates(...)`. Note the exception to canonical argument order documented in CLAUDE.md (no `env` param — heading is the variable).
- [src/neurospatial/encoding/egocentric.py](../../../src/neurospatial/encoding/egocentric.py) — `compute_egocentric_rate(env, spike_times, times, positions, headings, object_positions, *, distance_range, n_distance_bins, n_direction_bins, ...)`. Returns `EgocentricRateResult` with `.preferred_direction()`, `.preferred_distance()`.
- [src/neurospatial/encoding/view.py](../../../src/neurospatial/encoding/view.py) — `compute_view_rate(env, spike_times, times, positions, headings, *, view_distance, gaze_model, ...)`. `gaze_model` is `"fixed_distance"`, `"ray_cast"`, or `"boundary"`.
- [src/neurospatial/encoding/phase_precession.py](../../../src/neurospatial/encoding/phase_precession.py) — `circular_linear_regression(...)` returns slope, intercept, R². Audit found existing tests only assert `slope < 0`.
- [tests/encoding/test_jax_compute_dispatch.py:110-194, 445-543, 677-731, 777-825](../../../tests/encoding/test_jax_compute_dispatch.py) — the pattern to mirror for parity tests. Compare NumPy and JAX outputs at `rtol=1e-10, atol=1e-14` for matching inputs.
- [tests/encoding/conftest.py](../../../tests/encoding/conftest.py) — existing fixtures including `place_cell_spikes` (audit referenced lines 2782-2804; reads simulator output for a Gaussian field at (60,60) peak 20 Hz).
- [src/neurospatial/simulation/models/place_cells.py](../../../src/neurospatial/simulation/models/place_cells.py) and `head_direction_cells.py`, `object_vector_cells.py` — neural models used to generate ground-truth spike trains.

## Tasks

### 1. Place-field recovery through `compute_spatial_rate`

In [tests/encoding/test_encoding_spatial.py](../../../tests/encoding/test_encoding_spatial.py), after the existing `TestComputeSpatialRateFunction` class (audit referenced line 3145), add a new class `TestSpatialRateRecovery`. Cover all three smoothing methods (`binned`, `gaussian_kde`, `diffusion_kde`).

For each smoothing method:
- Simulate a Poisson place cell with peak at `(60, 60)`, peak rate 20 Hz, Gaussian width 5 cm, for 5000 spikes over 600 s on a 2D open field (`Environment.from_samples` with `bin_size=2.0`, random walk trajectory covering the env).
- Call `compute_spatial_rate(env, spike_times, times, positions, smoothing_method=..., bandwidth=5.0)`.
- Assert `np.linalg.norm(result.peak_location() - np.array([60.0, 60.0])) < env.bin_size` (peak within one bin).
- Assert `0.6 * 20 <= result.firing_rate.max() <= 1.4 * 20` (peak rate within ±40% of truth — Poisson noise budget).
- Fix random seed; mark test `@pytest.mark.slow` if wall-clock > 5s on local dev.

Three tests total (`test_recovery_binned`, `test_recovery_gaussian_kde`, `test_recovery_diffusion_kde`). A scale bug, a coordinate-convention flip, or a smoothing-bandwidth misuse all break these.

### 2. Head-direction recovery through `compute_directional_rate`

In [tests/encoding/test_encoding_directional.py](../../../tests/encoding/test_encoding_directional.py), after the existing function-level smoke test (audit cited lines 2741-2803), add `TestDirectionalRateRecovery`:

- `test_preferred_direction_recovered`: simulate headings uniform on `[-π, π]` over 600 s; spike rate follows von Mises tuning curve `r(h) = baseline + max_rate * exp(kappa * (cos(h - true_pref) - 1))` with `true_pref = π/3`, `kappa = 4.0`, `max_rate=15.0`, `baseline=1.0`. Generate Poisson spikes (~5000 total). Call `compute_directional_rate(spike_times, times, headings, n_bins=36)`. Assert the circular distance between `result.preferred_direction()` and `π/3` is `< 2 * (2π / 36)` (within two bins).
- Parametrize over `true_pref ∈ {0.0, π/2, -π/2, π}` to cover the wrap-around at ±π.

This test catches: (a) circular wrap bugs, (b) bin-edge indexing off-by-one, (c) any divergence in how `n_bins` is interpreted between docs and code.

### 3. Object-vector convention recovery through `compute_egocentric_rate`

In [tests/encoding/test_compute_egocentric_rate.py](../../../tests/encoding/test_compute_egocentric_rate.py), add `TestEgocentricRateConvention`:

- `test_object_ahead_peaks_at_zero_bearing`: stationary animal at `(0, 0)`, `heading = 0` (facing +x) for 600 s. Object at `(10, 0)` (directly ahead). Simulate spikes from an `ObjectVectorCellModel(preferred_direction=0.0, preferred_distance=10.0)`. Call `compute_egocentric_rate`. Assert `abs(result.preferred_direction()) < 2π / n_direction_bins` and `abs(result.preferred_distance() - 10.0) < (distance_range_span / n_distance_bins)`.
- `test_object_left_peaks_at_positive_pi_over_two`: object at `(0, 10)` (animal's left per the CLAUDE.md convention π/2 = left). OVC model with `preferred_direction=π/2`. Assert `circular_distance(result.preferred_direction(), π/2) < one bin`.
- `test_object_right_peaks_at_negative_pi_over_two`: object at `(0, -10)`. OVC model with `preferred_direction=-π/2`. Assert `circular_distance(result.preferred_direction(), -π/2) < one bin`.

These three tests together pin the egocentric convention end-to-end through the actual compute function. A sign flip in `compute_egocentric_bearing` or a swapped left/right convention will fail at least one.

### 4. Phase-precession slope magnitude recovery

In [tests/encoding/test_phase_precession_metrics.py](../../../tests/encoding/test_phase_precession_metrics.py), after the existing line 104 region, add `TestPhasePrecessionSlopeMagnitude`:

- `test_slope_magnitude_recovered`: simulate a noisy linear relationship `phase = wrap_to_2pi(-0.1 * position + intercept + noise)` with `noise ~ N(0, 0.2)` for 200 samples over positions `0..100`. Call the existing slope-fitting function (likely `circular_linear_regression` or `precession_slope`). Assert `np.isclose(result.slope, -0.1, atol=0.02)`.
- `test_slope_sign_with_noise`: as above but with `true_slope = +0.05`; assert `np.isclose(result.slope, +0.05, atol=0.02)`.
- `test_position_range_normalization`: pin the unit convention as documented at [src/neurospatial/encoding/phase_precession.py:203-208](../../../src/neurospatial/encoding/phase_precession.py). The contract is: **with `position_range=None`, slope is in `rad/position_unit`. With `position_range=(pos_min, pos_max)`, positions are normalized to `[0, 1]` before fitting and slope is in `rad/normalized_position` (i.e., phase change across the entire normalized field).** Test by simulating the same precession on positions `0..100` two ways: (a) `position_range=None` → expect `slope ≈ true_slope_per_cm`; (b) `position_range=(0, 100)` → expect `slope ≈ true_slope_per_cm * 100` (because slope is now per-normalized-unit and the field spans 100 raw units). Assert both at `atol=0.02`.

### 5. NumPy↔JAX parity for directional rate

In [tests/encoding/test_jax_compute_dispatch.py](../../../tests/encoding/test_jax_compute_dispatch.py), add `TestComputeDirectionalRateJaxBackend` and `TestComputeDirectionalRatesJaxBackend`, mirroring the structure used for spatial/view/egocentric (audit referenced lines 110-194 for the pattern).

For each:
- Build a deterministic fixture: 500 headings sampled uniformly, 1000 spike times, `n_bins=36`.
- Call `compute_directional_rate(..., backend="numpy")` and `compute_directional_rate(..., backend="jax")`. Assert `np.testing.assert_allclose(jax_result.firing_rate, numpy_result.firing_rate, rtol=1e-10, atol=1e-14)`.
- Repeat for the batched `compute_directional_rates` over 10 cells.
- Parametrize over the angle-wrap edge case (headings concentrated near ±π) to surface any bin-edge divergence.

### 6. Gaze-model geometric correctness in `compute_view_rate`

In [tests/encoding/test_encoding_view_binning.py](../../../tests/encoding/test_encoding_view_binning.py), strengthen the gaze-model tests (audit cited lines 246-280 as smoke-only).

Add `TestGazeModelGeometry` with:
- `test_boundary_clips_at_wall`: rectangular env spanning `x ∈ [0, 100], y ∈ [0, 100]` with no obstacles. Animal at `(5, 50)`, `heading=π` (facing -x toward the `x=0` wall). With `gaze_model="boundary"`, the occupied view bins must peak near `x ≈ 0`. With `gaze_model="fixed_distance"` and `view_distance=10`, view bins peak near `x ≈ -5` — but since `x=-5` is outside the env, the view should fall outside `env.bin_at` (returns -1). Assert `boundary` produces strictly non-negative bin indices throughout, while `fixed_distance` produces ≥ 50% invalid (negative) indices.
- `test_ray_cast_walls_match_boundary_in_convex_env`: in a convex env with no obstacles, `boundary` and `ray_cast` should agree (both stop at the wall). Assert `np.allclose(boundary_view, ray_cast_view, atol=env.bin_size)`.

### 7. NaN-position-input contract for `compute_*_rate`

**Desired contract** (established for this plan): NaN positions are treated as **missing data and excluded** from occupancy and firing-rate computation. The recovered rate must equal — within Poisson tolerance — the rate computed from the same session with the NaN samples removed before passing in. This is the standard tracking-data convention and matches what users expect after a brief tracking dropout.

**Current state** (verified at plan time): [src/neurospatial/encoding/_validation.py:184-260](../../../src/neurospatial/encoding/_validation.py) `validate_trajectory` checks `times` for finiteness via `validate_times`, but does **not** check `positions` or `headings` for NaN. NaN positions propagate downstream into `bin_at`/`map_points_to_bins`, which may produce -1 (outside) indices via KDTree's `inf` distance. Whether the resulting occupancy/rate calculation correctly excludes those samples is **unknown** and is what the test must pin.

For each of `compute_spatial_rate`, `compute_egocentric_rate`, `compute_view_rate`, add:

- `test_nan_positions_treated_as_missing_data`: simulate a session of length N. Pick 10% of indices at random; set their positions to NaN. Call `compute_spatial_rate(env, spike_times, times, positions_with_nan, ...)`. Separately, drop the same indices entirely (positions, times, spike_times-in-window all consistent) and run again. Assert `np.allclose(rate_with_nan, rate_without_nan, rtol=0.1)` (Poisson noise budget).
- `test_nan_headings_treated_as_missing_data` (for `compute_egocentric_rate`, `compute_view_rate`): same pattern on the heading array.

**If the test fails** (current behavior produces wrong rates with NaN inputs), this is a real bug. **Do not bundle the source fix in Phase 2** — surface in the PR description, open a follow-up issue, and mark these tests `@pytest.mark.xfail(reason="NaN-position handling pending — see issue #XXX")`. Phase 2 only adds the regression tests; the fix is out of scope.

#### 7a. Documentation subtask

Independent of the test outcome above, add to each of `compute_spatial_rate`, `compute_egocentric_rate`, `compute_view_rate` docstrings a `positions` Parameter sub-line stating the NaN contract:

```
positions : ndarray, shape (n_samples, n_dims)
    Position coordinates at each ``times[i]``. NaN values are treated as
    missing data and excluded from occupancy and firing-rate computation;
    callers do not need to pre-filter.
```

If the test in Task 7 (above) confirms current behavior matches the contract, the docstring is now accurate. If the test is `xfail`ed, the docstring documents the contract that the follow-up fix will establish.

## Deliberately not in this phase

- **No grid-cell recovery test.** `compute_spatial_rate` on a simulated grid cell would be valuable but adds complexity (need a multi-bump tuning curve and `grid_score` evaluation); Phase 7 strengthens the existing grid-score thresholds. Drop this task here.
- **No `gaussian_kde` NumPy↔JAX parity.** The audit flagged this gap; deferred — directional parity is the higher-leverage fix and `gaussian_kde` is less common.
- **No removal of the existing weak tests** that construct results from hand-built firing rates (audit cited `test_encoding_egocentric.py:933-967`, `test_encoding_directional.py:550-581`, etc.). Add strong tests alongside; weak tests stay for now to document API surface.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/encoding/test_encoding_spatial.py::TestSpatialRateRecovery::test_recovery_*` (3 tests) | Place-cell peak recovered within 1 bin and rate within ±40% for `binned` / `gaussian_kde` / `diffusion_kde`. **Mark `@pytest.mark.slow`.** |
| `tests/encoding/test_encoding_directional.py::TestDirectionalRateRecovery::test_preferred_direction_recovered` (parametrized 4×) | HD preferred direction recovered within 2 bins for cardinal + π/3 + ±π wrap. |
| `tests/encoding/test_compute_egocentric_rate.py::TestEgocentricRateConvention::test_object_*` (3 tests) | Egocentric 0=ahead, π/2=left, -π/2=right convention end-to-end. |
| `tests/encoding/test_phase_precession_metrics.py::TestPhasePrecessionSlopeMagnitude::test_*` (3 tests) | Slope magnitude, sign, and unit convention pinned. |
| `tests/encoding/test_jax_compute_dispatch.py::TestComputeDirectionalRate{,s}JaxBackend` | NumPy↔JAX parity for `compute_directional_rate` and `compute_directional_rates` at `rtol=1e-10`. |
| `tests/encoding/test_encoding_view_binning.py::TestGazeModelGeometry` (2 tests) | `boundary` clips at wall; `boundary` and `ray_cast` agree in convex env. |
| `tests/encoding/test_*::test_nan_in_positions` (3 tests across 3 files) | NaN-position handling is pinned to current behavior. |

## Fixtures

Synthesize in `tests/encoding/conftest.py`:
- `place_cell_session` (deterministic): 600 s random walk on a 100×100 open field, 5000 Poisson spikes from a Gaussian field at `(60, 60)`. Reuse seed `42`.
- `hd_session`: 600 s of uniform headings, 5000 Poisson spikes from a von Mises tuning curve.
- `egocentric_session_with_stationary_animal`: animal at origin, fixed heading, object at parametrized location.

Existing `place_cell_spikes` fixture (cited lines 2782-2804 of `test_encoding_spatial.py`) may be reusable — check first; extend rather than duplicate if so.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Recovery tests drive the **public** `compute_*_rate` functions, not hand-built `*RateResult` dataclasses. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (none in this phase).
- User-facing documentation listed as tasks is updated, not deferred (NaN-position behavior is documented in the source docstrings).
