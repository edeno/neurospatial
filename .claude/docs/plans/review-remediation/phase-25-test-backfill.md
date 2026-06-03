# Phase 25 — Test-coverage backfill

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

One shippable PR. This phase closes the **residual** untested validation/failure
branches identified in [SUMMARY.md "Theme 6 — Untested validation / failure
branches"](../../../reviews/SUMMARY.md) that are **not** already locked in by a
fail-before/pass-after regression test in the correctness phases (1–13). Each
correctness phase already adds the regression test for the bug it fixes; this
phase adds tests for the branches that were merely *uncovered* (no bug fix
attached) plus one small guard (`detect_assemblies` IndexError) that is too
trivial to warrant its own correctness phase but must ship with a test.

All work is **tests**, plus a single ~5-line source guard in
`decoding/assemblies.py` (Task 7). No other source changes.

**Inputs to read first:**

- [../../../../src/neurospatial/layout/validation.py:44-233](../../../../src/neurospatial/layout/validation.py) — `validate_bin_size` (44-115) and `validate_dimension_ranges` (118-233). Public, fully implemented, **zero test coverage** (verified: `grep validate_bin_size tests/` → no hits). Error messages confirmed by running them: `"bin_size must be positive (got -1.0)."`, `"bin_size contains NaN values..."`, `"bin_size contains infinite values..."`, `"dimension_ranges[0] has min >= max: ..."`, `"dimension_ranges has 1 dimensions, expected 2"`, `"dimension_ranges cannot be empty"`, `"dimension_ranges must be a list or tuple, ..."`.
- [../../../../src/neurospatial/layout/helpers/regular_grid.py:258-275](../../../../src/neurospatial/layout/helpers/regular_grid.py) — `_infer_active_bins_from_regular_grid` boundary-trim block. The **1D** branch (259-265) is covered by `tests/layout/test_regular_grid_utils.py:77` (`test_infer_active_bins_boundary_exists`). The **N-D** branch (266-273, the per-axis outer-shell loop) is **uncovered**. Exposed publicly via `Environment.from_samples(..., add_boundary_bins=True)`.
- [../../../../src/neurospatial/ops/alignment.py:388-453](../../../../src/neurospatial/ops/alignment.py) — `_map_inverse_distance_weighted`; `map_probabilities(..., mode="inverse-distance-weighted")` at 456-600. `IDW_MIN_DISTANCE = 1e-08` is module-public. **No IDW test exists** (verified: `grep inverse_distance tests/` → no hits; existing `tests/ops/test_ops_alignment.py` tests only `mode="nearest"`).
- [../../../../src/neurospatial/ops/normalize.py:31-275](../../../../src/neurospatial/ops/normalize.py) — `normalize_field` (31-108) untested branches: NaN (80-85), Inf (86-91), all-zeros (101-105), `eps<=0` (72-76). `combine_fields` (161-275) untested branches: mismatched shapes (223-226), weights-wrong-length (237-241), weights-with-non-mean-mode (231-235), weights-not-summing-to-1 (244-248), unknown-mode (271-274).
- [../../../../src/neurospatial/behavior/navigation.py:1835-1851](../../../../src/neurospatial/behavior/navigation.py) — `approach_rate` geodesic branch (1845-1851). **Untested** (verified: every `approach_rate` test in `tests/behavior/` passes `metric="euclidean"`).
- [../../../../src/neurospatial/animation/rendering.py:555-590](../../../../src/neurospatial/animation/rendering.py) — `field_to_rgb_for_napari` 2D-grid orientation transform: transpose `(n_x, n_y)→(n_y, n_x)` then `np.flip(axis=0)`. Verified mapping: env `(min x, max y)` → image pixel `(row 0, col 0)`; `(min x, min y)` → `(row H-1, col 0)`; `(max x, max y)` → `(row 0, col W-1)`. Existing tests (`tests/animation/test_rendering.py:104-161`) assert dtype/shape/clipping only — **orientation never verified**.
- [../../../../src/neurospatial/simulation/trajectory.py:460-595](../../../../src/neurospatial/simulation/trajectory.py) — `simulate_trajectory_sinusoidal`; requires `env.is_linearized_track`. Existing behavioral tests (`tests/simulation/test_trajectory_sim.py:203-216`) are all `pytest.skip("Requires 1D environment with GraphLayout")` — **zero behavioral coverage**. A 1D env is buildable via `Environment.from_graph(...)` (no extra deps; verified). `seed` is documented no-op.
- [../../../../src/neurospatial/simulation/trajectory.py:62-303](../../../../src/neurospatial/simulation/trajectory.py) — `simulate_trajectory_ou`, `coherence_time` at 71/303. Existing autocorrelation test (`tests/simulation/test_trajectory_sim.py:45-81`) uses a 150% tolerance, so it passes for almost any decay ratio — a near-tautology. See Task 6 for the scoped, robust replacement (and the deliberate exclusion of a fragile comparative test).
- [../../../../src/neurospatial/ops/visibility.py:380-430](../../../../src/neurospatial/ops/visibility.py) — `visible_cues` returns `(visible, distances, bearings)`. Existing test (`tests/ops/test_visibility.py:605-632`) asserts lengths/dtype only — **numerical bearings never asserted**. Bearings flow through `compute_egocentric_bearing → _wrap_angle`; **this couples to phase 8** (see Task 5 note).
- [../../../../src/neurospatial/decoding/assemblies.py:487-587](../../../../src/neurospatial/decoding/assemblies.py) — `detect_assemblies`. **Verified crash**: with `n_time_bins < n_components ≤ n_neurons` (e.g. shape `(8, 3)`, `n_components=5`), `_detect_pca`'s SVD yields only `min(n_neurons, n_time_bins)=3` patterns, so the loop `for i in range(n_comp)` at line 569 indexes `member_masks[i]` out of bounds → `IndexError: index 3 is out of bounds for axis 0 with size 3` (pca, ica) / `ValueError` (nmf). Task 7 adds a guard.

**Designs referenced:** none (this phase is mechanical test backfill).

## Tasks

### Task 1 — Cover the public layout validators (`tests/layout/`)

Add a new file `tests/layout/test_validation_helpers.py` covering
`validate_bin_size` and `validate_dimension_ranges` end-to-end. These are
public functions with documented `Raises` clauses and **no** existing test.

```python
"""Behavioral tests for the public layout validators."""

import numpy as np
import pytest

from neurospatial.layout.validation import (
    validate_bin_size,
    validate_dimension_ranges,
)


class TestValidateBinSize:
    def test_scalar_promoted_to_array(self):
        np.testing.assert_array_equal(validate_bin_size(2.0), np.array([2.0]))

    def test_per_dimension_array_passthrough(self):
        np.testing.assert_array_equal(
            validate_bin_size(np.array([2.0, 3.0])), np.array([2.0, 3.0])
        )

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(0.0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(-1.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_bin_size(np.nan)

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="infinite"):
            validate_bin_size(np.inf)

    def test_one_bad_entry_in_array_rejected(self):
        # Mixed valid/invalid: the negative entry must still trip the guard.
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(np.array([2.0, -1.0]))


class TestValidateDimensionRanges:
    def test_valid_ranges_returned_as_float_tuples(self):
        result = validate_dimension_ranges([(0, 100), (0, 200)])
        assert result == [(0.0, 100.0), (0.0, 200.0)]

    def test_n_dims_match_ok(self):
        result = validate_dimension_ranges([(0, 100), (0, 200)], n_dims=2)
        assert result == [(0.0, 100.0), (0.0, 200.0)]

    def test_n_dims_mismatch_rejected(self):
        with pytest.raises(ValueError, match="expected 2"):
            validate_dimension_ranges([(0, 100)], n_dims=2)

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="min >= max"):
            validate_dimension_ranges([(100, 0)])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dimension_ranges([])

    def test_non_sequence_rejected(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            validate_dimension_ranges("not-a-list")  # type: ignore[arg-type]

    def test_non_finite_bound_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_dimension_ranges([(0.0, np.inf)])

    def test_wrong_tuple_length_rejected(self):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_dimension_ranges([(0.0, 1.0, 2.0)])  # type: ignore[list-item]
```

### Task 2 — Cover N-D boundary trimming in `_infer_active_bins_from_regular_grid` (`tests/layout/`)

The 1D branch is covered; the **N-D** branch (regular_grid.py:266-273) is not.
Add a 2D and a 3D case to `tests/layout/test_regular_grid_utils.py` (the existing
home of `test_infer_active_bins_boundary_exists`). The behavioral contract: with
`boundary_exists=True`, the **outer shell of bins along every axis** is set
inactive, while the interior is unaffected.

```python
def test_infer_active_bins_boundary_exists_2d():
    """boundary_exists trims the outer shell of a 2D grid on every axis."""
    # Fully populate a 4x4 grid so all interior bins would otherwise be active.
    edges = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
    centers = np.array([[i + 0.5, j + 0.5] for i in range(4) for j in range(4)])
    mask = _infer_active_bins_from_regular_grid(
        centers, edges, bin_count_threshold=0, boundary_exists=True
    )
    assert mask.shape == (4, 4)
    # Outer rows/cols inactive on both axes.
    assert not mask[0, :].any()
    assert not mask[-1, :].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()
    # Interior preserved.
    assert mask[1:-1, 1:-1].all()


def test_infer_active_bins_boundary_exists_3d():
    """boundary_exists trims the outer shell of a 3D grid on every axis."""
    e = np.array([0, 1, 2, 3])
    edges = (e, e, e)  # 3x3x3 grid
    centers = np.array(
        [
            [i + 0.5, j + 0.5, k + 0.5]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ]
    )
    mask = _infer_active_bins_from_regular_grid(
        centers, edges, bin_count_threshold=0, boundary_exists=True
    )
    assert mask.shape == (3, 3, 3)
    # Every outer slab on all three axes is inactive.
    assert not mask[0, :, :].any() and not mask[-1, :, :].any()
    assert not mask[:, 0, :].any() and not mask[:, -1, :].any()
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()
    # The single interior bin (center) is the only one that can survive.
    assert mask[1, 1, 1]
```

### Task 3 — IDW `map_probabilities`: full positive-behavior matrix (`tests/ops/`)

Phase 8 (Task 3) narrowed the swallowed `except Exception` in the IDW path to
make bad input *raise* (and added `test_map_probabilities_nan_centers_raises`).
This phase owns the **positive-behavior** matrix that phase 8 explicitly deferred
(see phase-8 "Deliberately not in this phase"). Add to
`tests/ops/test_ops_alignment.py`, reusing the existing `MockEnvironment`
(`map_probabilities` only reads `.bin_centers`, `.n_bins`, `.n_dims`,
`._is_fitted`). All expected values below are **hand-computed and verified**.

```python
from neurospatial.ops.alignment import IDW_MIN_DISTANCE, map_probabilities


class TestMapProbabilitiesInverseDistanceWeighted:
    def test_two_neighbor_split_is_hand_computed(self):
        # Source mass entirely at (0,0). Targets at distances 1 and 3.
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        out = map_probabilities(
            src, tgt, np.array([1.0]),
            mode="inverse-distance-weighted", n_neighbors=2,
        )
        eps = IDW_MIN_DISTANCE
        w = np.array([1.0 / (1.0 + eps), 1.0 / (3.0 + eps)])
        expected = w / w.sum()  # ~[0.75, 0.25]
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_mass_is_conserved(self):
        src = MockEnvironment(
            np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]), n_dims=2
        )
        tgt = MockEnvironment(
            np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [5.0, 5.0]]), n_dims=2
        )
        src_probs = np.array([0.5, 0.3, 0.2])
        out = map_probabilities(
            src, tgt, src_probs,
            mode="inverse-distance-weighted", n_neighbors=3,
        )
        # IDW weights per source row are normalized, so total mass is preserved.
        assert out.sum() == pytest.approx(src_probs.sum())
        assert out.shape == (tgt.n_bins,)

    def test_n_neighbors_clamped_to_target_count(self):
        # Requesting more neighbors than target bins must clamp to n_tgt,
        # giving the same result as requesting exactly n_tgt.
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        sp = np.array([1.0])
        clamped = map_probabilities(
            src, tgt, sp, mode="inverse-distance-weighted", n_neighbors=5
        )
        exact = map_probabilities(
            src, tgt, sp, mode="inverse-distance-weighted", n_neighbors=2
        )
        np.testing.assert_allclose(clamped, exact, atol=1e-12)

    def test_k_eff_one_is_nearest_only(self):
        # k_eff == 1: all mass goes to the single nearest target (no split).
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        out = map_probabilities(
            src, tgt, np.array([1.0]),
            mode="inverse-distance-weighted", n_neighbors=1,
        )
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-12)
```

> `pytest` is already imported in `tests/ops/test_ops_alignment.py`.

### Task 4 — `normalize_field` / `combine_fields` validation paths (`tests/ops/`)

Extend `tests/ops/test_ops_normalize.py` with the untested guard branches. (The
happy-path and `negative`/`empty`/`lo>hi` cases already exist there; do **not**
duplicate them.)

```python
class TestNormalizeFieldValidation:
    def test_nan_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="NaN"):
            normalize_field(np.array([1.0, np.nan, 3.0]))

    def test_inf_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="Inf"):
            normalize_field(np.array([1.0, np.inf, 3.0]))

    def test_all_zeros_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="all zeros"):
            normalize_field(np.zeros(4))

    def test_non_positive_eps_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="eps must be positive"):
            normalize_field(np.array([1.0, 2.0]), eps=0.0)


class TestCombineFieldsValidation:
    def test_mismatched_shapes_rejected(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="same shape"):
            combine_fields([np.zeros(3), np.zeros(4)])

    def test_weights_only_valid_for_mean(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="mode='mean'"):
            combine_fields(
                [np.ones(3), np.ones(3)], weights=[0.5, 0.5], mode="max"
            )

    def test_weights_length_must_match_fields(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="must match"):
            combine_fields([np.ones(3), np.ones(3)], weights=[1.0])

    def test_weights_must_sum_to_one(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="sum to 1"):
            combine_fields([np.ones(3), np.ones(3)], weights=[0.3, 0.3])

    def test_unknown_mode_rejected(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="Unknown mode"):
            combine_fields([np.ones(3), np.ones(3)], mode="median")  # type: ignore[arg-type]
```

### Task 5 — `visible_cues` numerical bearings (`tests/ops/`)

Extend the existing `TestVisibleCues` class in `tests/ops/test_visibility.py`
with an assertion on the actual egocentric bearing **values** (not just lengths).
Convention: 0 = ahead, +π/2 = left, ±π = behind.

**Phase-8 coupling (important):** the behind-cue bearing routes through
`_wrap_angle`. *Before* phase 8 it returns `-π`; *after* phase 8's
`_wrap_angle` fix it returns `+π`. To keep this test independent of phase
ordering, assert `|bearing| == π` for the behind cue (both ±π are "directly
behind"). The exact-sign assertion is owned by phase 8
(`test_egocentric_bearing_directly_behind_is_plus_pi`); do not duplicate it here.

```python
def test_bearings_match_known_geometry(self):
    """visible_cues returns correct egocentric bearings (0=ahead, +pi/2=left)."""
    from neurospatial import Environment
    from neurospatial.ops.visibility import visible_cues

    rng = np.random.default_rng(42)
    env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=5.0)

    position = np.array([50.0, 50.0])
    heading = 0.0  # facing East (+x)
    cue_positions = np.array(
        [
            [60.0, 50.0],  # directly ahead (East)  -> 0
            [40.0, 50.0],  # directly behind (West) -> +/- pi
            [50.0, 60.0],  # directly left (North)  -> +pi/2
        ]
    )

    visible, distances, bearings = visible_cues(
        env, position, heading, cue_positions
    )

    assert np.all(visible)  # open arena, all cues in line of sight
    np.testing.assert_allclose(distances, [10.0, 10.0, 10.0], atol=1e-6)
    assert bearings[0] == pytest.approx(0.0, abs=1e-6)
    # Behind: +/- pi (sign owned by phase 8; assert magnitude here).
    assert abs(bearings[1]) == pytest.approx(np.pi, abs=1e-6)
    assert bearings[2] == pytest.approx(np.pi / 2, abs=1e-6)
```

> `pytest` and `np` are already imported in `tests/ops/test_visibility.py`.

### Task 6 — `simulate_trajectory_sinusoidal` behavioral coverage + tighten the OU autocorrelation test (`tests/simulation/`)

**6a — Replace the three skipped sinusoidal tests.** In
`tests/simulation/test_trajectory_sim.py`, the `TestSimulateTrajectorysinusoidal`
class has three `pytest.skip` placeholders (lines 203-216) because `simple_1d_env`
is a RegularGrid (`is_linearized_track == False`), not a graph track. Add a
module-level fixture that builds a real 1D track via `from_graph` and write the
behavioral tests against it. (Keep `test_requires_1d_environment` as-is.)

```python
@pytest.fixture
def linear_track_env():
    """A real 1D linearized track (GraphLayout) for sinusoidal motion tests."""
    import networkx as nx

    from neurospatial import Environment

    g = nx.Graph()
    g.add_node(0, pos=(0.0,))
    g.add_node(1, pos=(50.0,))
    g.add_node(2, pos=(100.0,))
    g.add_edge(0, 1, distance=50.0)
    g.add_edge(1, 2, distance=50.0)
    env = Environment.from_graph(
        graph=g, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )
    env.units = "cm"
    return env
```

Then, replacing the three skipped tests:

```python
def test_basic_sinusoidal_generation(self, linear_track_env):
    """Produces (n_time, 1) positions sampled at the requested rate."""
    positions, times = simulate_trajectory_sinusoidal(
        linear_track_env, duration=10.0, sampling_frequency=100.0,
        speed=20.0, period=4.0, seed=1,
    )
    assert positions.shape == (1000, 1)  # 10s * 100Hz
    assert times.shape == (1000,)
    assert times[1] - times[0] == pytest.approx(0.01)

def test_stays_within_track_bounds(self, linear_track_env):
    """Sinusoidal positions never leave the linearized track range."""
    range_min, range_max = linear_track_env.dimension_ranges[0]
    positions, _ = simulate_trajectory_sinusoidal(
        linear_track_env, duration=20.0, sampling_frequency=100.0,
        speed=20.0, period=4.0,
    )
    assert positions.min() >= range_min - 1e-9
    assert positions.max() <= range_max + 1e-9

def test_is_periodic_with_given_period(self, linear_track_env):
    """With pauses off, x(t) repeats every `period` seconds."""
    period = 4.0
    positions, times = simulate_trajectory_sinusoidal(
        linear_track_env, duration=12.0, sampling_frequency=100.0,
        speed=20.0, period=period, pause_at_peaks=False,
    )
    n_per_period = int(period * 100.0)
    # Compare the first period to the second period sample-for-sample.
    first = positions[:n_per_period, 0]
    second = positions[n_per_period : 2 * n_per_period, 0]
    np.testing.assert_allclose(first, second, atol=1e-6)

def test_seed_has_no_effect(self, linear_track_env):
    """`seed` is a documented no-op: output is identical regardless of seed."""
    kw = dict(
        duration=5.0, sampling_frequency=100.0, speed=20.0, period=4.0
    )
    a, _ = simulate_trajectory_sinusoidal(linear_track_env, seed=1, **kw)
    b, _ = simulate_trajectory_sinusoidal(linear_track_env, seed=999, **kw)
    np.testing.assert_array_equal(a, b)
```

**6b — Tighten the OU autocorrelation test (de-tautologize).** The existing
`test_velocity_autocorrelation_matches_coherence_time` (lines 45-81) uses a 150%
tolerance, so it would pass for nearly any decay ratio — it does not actually
verify that velocity is temporally correlated. Replace the loose ratio check with
a **non-tautological** assertion: the velocity autocorrelation at a one-coherence-
time lag is **strictly between 0 and the lag-0 value** (i.e. the process is
genuinely correlated and decaying, not white noise and not a constant). Keep the
same simulation call so coherence_time is still exercised.

```python
def test_velocity_autocorrelation_is_correlated_and_decays(self, simple_2d_env):
    """OU velocity is temporally correlated: 0 < acf(coherence_time) < acf(0)."""
    coherence_time = 0.5
    positions, times = simulate_trajectory_ou(
        simple_2d_env, duration=200.0, dt=0.01,
        coherence_time=coherence_time, seed=42, speed_units="cm",
    )
    dt = times[1] - times[0]
    vx = np.diff(positions, axis=0)[:, 0]
    vx = vx - vx.mean()
    acf0 = np.mean(vx**2)
    lag = int(coherence_time / dt)
    acf_lag = np.mean(vx[:-lag] * vx[lag:])
    # White noise -> ~0; a constant -> ~acf0. A real OU process sits strictly
    # between, confirming coherence_time produces decaying-but-nonzero memory.
    assert 0.0 < acf_lag < acf0
```

> **Scope note (see "Deliberately not in this phase"):** a *comparative* test
> ("larger coherence_time ⇒ slower decay") was investigated and rejected — under
> the discretized OU update the velocity-ACF separation across coherence times is
> masked even after averaging over 8 seeds, making such a test flaky. The
> assertion above is the strongest robust, non-tautological claim available.

### Task 7 — `detect_assemblies`: guard the IndexError + test (`assemblies.py` + `tests/decoding/`)

**Source guard (the only source change in this phase).** In
`src/neurospatial/decoding/assemblies.py`, after `n_comp` is resolved
(currently lines 532-541, just before "Perform dimensionality reduction" at
544), the SVD in `_detect_pca` can only produce `min(n_neurons, n_time_bins)`
patterns, but the membership loop at line 569 iterates `range(n_comp)`. When
`n_time_bins < n_comp` the loop indexes past the patterns array → `IndexError`
(pca/ica) or `ValueError` (nmf `nndsvda`). Clamp `n_comp` to the achievable rank
with a warning:

```python
    # The factorization can yield at most min(n_neurons, n_time_bins)
    # components. Requesting more (short recordings) would index past the
    # SVD output and raise IndexError downstream; clamp with a warning.
    max_rank = min(n_neurons, n_time_bins)
    if n_comp > max_rank:
        warnings.warn(
            f"n_components ({n_comp}) exceeds the achievable rank "
            f"min(n_neurons, n_time_bins) = {max_rank}; using {max_rank}.",
            UserWarning,
            stacklevel=2,
        )
        n_comp = max_rank
```

Place this block immediately after the `n_components`-resolution branch
(after line 541, before the `if algorithm == "pca":` dispatch). `warnings` is
already imported in this module. Do **not** touch the existing
`n_comp > n_neurons` ValueError (lines 538-541) — that remains a hard user error
for the over-request case; the new guard handles the short-recording case where
`n_time_bins` is the binding constraint.

> Note: `detect_assemblies` is also edited by phases 1 and 22; land in order
> 1 → 22 → 25; the edits are disjoint (REV math / signature rename / clamp guard).
> In particular, by the time this phase lands the public RNG argument is `rng`
> (phase 22), so the Task-7 test below calls `detect_assemblies(..., rng=0)`.

**Test.** Add to `tests/decoding/test_assemblies.py` (note: this complements,
and does not overlap, phase 1's REV/EV semantics tests):

```python
class TestDetectAssembliesShortRecording:
    @pytest.mark.parametrize("algorithm", ["pca", "ica", "nmf"])
    def test_n_time_bins_below_n_components_does_not_crash(
        self, rng: np.random.Generator, algorithm: str
    ) -> None:
        """n_time_bins < n_components <= n_neurons must clamp, not IndexError."""
        # 8 neurons, 3 time bins, request 5 components.
        spike_counts = rng.poisson(5, (8, 3)).astype(np.float64)
        with pytest.warns(UserWarning, match="achievable rank"):
            result = detect_assemblies(
                spike_counts, algorithm=algorithm,
                n_components=5, rng=0,  # `rng` (renamed from `random_state` in phase 22)
            )
        # Clamped to min(n_neurons, n_time_bins) == 3 patterns; no crash.
        assert len(result.patterns) <= 3
        assert result.activations.shape[1] == 3  # n_time_bins preserved
```

> **Fail-before:** without the guard, the `pca`/`ica` params raise
> `IndexError: index 3 is out of bounds for axis 0 with size 3` and `nmf` raises
> `ValueError`. `rng` is the existing module fixture (`tests/decoding/test_assemblies.py:27`).

### Task 8 — `approach_rate` geodesic branch (`tests/behavior/`)

Add a geodesic-branch test to `tests/behavior/test_goal_directed.py` (the
existing home of the `approach_rate` tests). The geodesic branch (navigation.py
1845-1851) bins positions/goal into the env and measures geodesic distance. On an
open 2D grid the geodesic distance tracks the Euclidean one, so a trajectory
walking straight toward the goal must produce **negative** approach rates
(distance decreasing), and the first sample is NaN.

```python
def test_approach_rate_geodesic_branch(self):
    """metric='geodesic' returns negative rates when approaching, NaN first."""
    import numpy as np

    from neurospatial import Environment
    from neurospatial.behavior.navigation import approach_rate

    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 100, (2000, 2)), bin_size=5.0)

    # Straight walk from (10,50) toward goal at (90,50).
    xs = np.linspace(10.0, 80.0, 20)
    positions = np.column_stack([xs, np.full_like(xs, 50.0)])
    times = np.linspace(0.0, 4.0, 20)
    goal = np.array([90.0, 50.0])

    rates = approach_rate(positions, times, goal, metric="geodesic", env=env)

    assert np.isnan(rates[0])  # first value is NaN by contract
    assert np.nanmean(rates) < 0  # approaching => distance decreasing


def test_approach_rate_geodesic_requires_env(self):
    """metric='geodesic' without env raises a clear ValueError."""
    import numpy as np

    from neurospatial.behavior.navigation import approach_rate

    positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
    times = np.linspace(0, 5, 11)
    with pytest.raises(ValueError, match="env parameter is required"):
        approach_rate(positions, times, np.array([100.0, 0.0]), metric="geodesic")
```

### Task 9 — `field_to_rgb_for_napari` orientation (`tests/animation/`)

Add an orientation test to `tests/animation/test_rendering.py`. Use an
**identity colormap** (`cmap_lookup[i] = (i, i, i)`) so the rendered pixel value
equals the normalized colormap index, letting us read back *which* bin landed at a
given pixel. A unique high value at a known corner bin must appear at the
predicted `(row, col)` of the napari image. Verified mapping (env → image):
`(min x, max y) → (0, 0)`, `(min x, min y) → (H-1, 0)`, `(max x, max y) → (0, W-1)`.

```python
def test_field_to_rgb_for_napari_orientation():
    """A field peak at a known env corner lands at the predicted napari pixel."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    # Dense 4x4 grid covering 0..30 cm so every bin is active and indexable.
    xs = np.linspace(0, 30, 16)
    ys = np.linspace(0, 30, 16)
    xx, yy = np.meshgrid(xs, ys)
    env = Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=10.0
    )
    assert env.layout.grid_shape == (4, 4)

    # Identity colormap: index i -> (i, i, i). Channel 0 == colormap index.
    cmap_lookup = np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)
    centers = env.bin_centers

    def bright_pixel(flat_bin_index):
        field = np.zeros(env.n_bins)
        field[flat_bin_index] = 1.0  # maps to colormap index 255
        rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)
        assert rgb.shape == (4, 4, 3)
        return np.unravel_index(int(np.argmax(rgb[..., 0])), rgb[..., 0].shape)

    # (min x, max y) -> top-left (row 0, col 0)
    top_left = int(np.lexsort((centers[:, 0], -centers[:, 1]))[0])
    assert bright_pixel(top_left) == (0, 0)

    # (min x, min y) -> bottom-left (row 3, col 0): napari flips Y downward.
    bottom_left = int(np.lexsort((centers[:, 0], centers[:, 1]))[0])
    assert bright_pixel(bottom_left) == (3, 0)

    # (max x, max y) -> top-right (row 0, col 3)
    top_right = int(np.lexsort((-centers[:, 0], -centers[:, 1]))[0])
    assert bright_pixel(top_right) == (0, 3)
```

> `np`, `pytest`, and `Environment` are already imported at the top of
> `tests/animation/test_rendering.py`.

## Deliberately not in this phase

The following Theme-6 candidates are **already covered by a fail-before/pass-after
regression test in an earlier phase** and are intentionally excluded here to avoid
duplicate assertions (re-verified by reading each phase file):

- **Negative-weight weighted-circular stats** → covered by **phase 4**
  (`test_rayleigh_test_negative_weights_raises`,
  `test_circular_mean_negative_weights_raises`). Dropped.
- **`compute_directional_rate` end-to-end HD-cell recovery** → covered by
  **phase 5** (`test_is_head_direction_cell_recovers_hd_cell`, plus
  `test_compute_directional_rate_stores_spike_counts` /
  `test_directional_rates_getitem_forwards_counts`). Dropped.
- **Graph / Polygon `to_file`→`from_file` round-trip** (and the
  `active_mask` / `grid_edges` / `is_linearized_track` round-trip, and the polar
  `coordinate_kind` NWB round-trip) → covered by **phase 9**
  (`test_to_file_roundtrip_all_layouts[Graph|Polygon|Hexagonal|Masked|ImageMask|3D]`,
  `test_to_dict_roundtrip_graph_layout`,
  `test_nwb_roundtrip_preserves_coordinate_kind`). Dropped.
- **`detect_assemblies` REV directional semantics** (EV > REV with a control;
  EV == REV without) → covered by **phase 1**
  (`test_explained_variance_rev_differs_from_ev_with_control`,
  `test_explained_variance_no_control_ev_equals_rev`). Only the **IndexError
  short-recording crash** (a separate, untested code path) is in scope here
  (Task 7).
- **`_wrap_angle` antipode / directly-behind bearing exact sign (+π)** → covered
  by **phase 8** (`test_wrap_angle_antipode_is_plus_pi`,
  `test_egocentric_bearing_directly_behind_is_plus_pi`). Task 5 asserts only the
  bearing **magnitude** (`|bearing| == π`) to stay phase-order-independent.
- **IDW `map_probabilities` "bad input now raises"** → covered by **phase 8**
  (`test_map_probabilities_nan_centers_raises`,
  `test_map_probabilities_empty_env_still_returns_zeros`). Only the **positive**
  IDW matrix (splits, mass conservation, clamp, `k_eff==1`) is in scope here
  (Task 3).
- **Holed-grid linear occupancy mass-conservation** → owned by **phase 2**.
  Not touched here.
- **`edge_order`/`edge_spacing` override staleness after delete** → owned by
  **phase 12** (annotation). Not touched here.

Also out of scope:

- **Any source behavior change beyond the Task-7 `detect_assemblies` clamp.**
  This phase does not "fix while testing"; the validators, IDW, normalize,
  `approach_rate`, `field_to_rgb_for_napari`, and sinusoidal/OU code are tested
  **as-is**.
- **A comparative OU coherence-time autocorrelation test** — investigated and
  rejected as flaky (Task 6b scope note).
- **The reuse-artists video-render-events path** (`animation/_parallel.py:1066-1104`,
  also listed under Theme 6) — left for the animation phase; this phase covers
  only the `field_to_rgb_for_napari` orientation gap.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/layout/test_validation_helpers.py::TestValidateBinSize::*` | `validate_bin_size` promotes scalars, passes arrays, and raises on zero/negative/NaN/Inf and one-bad-entry arrays. |
| `tests/layout/test_validation_helpers.py::TestValidateDimensionRanges::*` | `validate_dimension_ranges` normalizes to float tuples and raises on n_dims mismatch, inverted/empty/non-finite/wrong-length ranges and non-sequence input. |
| `tests/layout/test_regular_grid_utils.py::test_infer_active_bins_boundary_exists_2d` | `boundary_exists=True` deactivates the outer shell on both axes of a full 4×4 grid; interior preserved. |
| `tests/layout/test_regular_grid_utils.py::test_infer_active_bins_boundary_exists_3d` | Same for a 3×3×3 grid on all three axes; only the center bin survives. |
| `tests/ops/test_ops_alignment.py::TestMapProbabilitiesInverseDistanceWeighted::test_two_neighbor_split_is_hand_computed` | IDW split of all-mass-at-(0,0) onto targets at distance 1 and 3 equals `[0.75, 0.25]` (within 1e-9). |
| `…::test_mass_is_conserved` | `out.sum() == src_probs.sum()` for a 3-source/4-target IDW map. |
| `…::test_n_neighbors_clamped_to_target_count` | `n_neighbors=5` clamps to `n_tgt=2`, identical to `n_neighbors=2`. |
| `…::test_k_eff_one_is_nearest_only` | `n_neighbors=1` gives `[1.0, 0.0]` (nearest only). |
| `tests/ops/test_ops_normalize.py::TestNormalizeFieldValidation::*` | `normalize_field` raises on NaN / Inf / all-zeros / `eps<=0`. |
| `tests/ops/test_ops_normalize.py::TestCombineFieldsValidation::*` | `combine_fields` raises on mismatched shapes / weights-with-non-mean-mode / wrong weights length / weights-not-summing-to-1 / unknown mode. |
| `tests/ops/test_visibility.py::TestVisibleCues::test_bearings_match_known_geometry` | Bearings for ahead/behind/left cues are `0`, `±π` (magnitude), `+π/2`; distances all `10.0`. |
| `tests/simulation/test_trajectory_sim.py::TestSimulateTrajectorysinusoidal::*` (4 new) | Shape `(1000,1)` at 100 Hz; positions stay in track range; `x(t)` periodic with `period`; `seed` is a no-op. (Replaces 3 skips.) |
| `tests/simulation/test_trajectory_sim.py::…::test_velocity_autocorrelation_is_correlated_and_decays` | OU velocity ACF at lag = coherence_time satisfies `0 < acf_lag < acf0` (replaces the 150%-tolerance near-tautology). |
| `tests/decoding/test_assemblies.py::TestDetectAssembliesShortRecording::test_n_time_bins_below_n_components_does_not_crash[pca\|ica\|nmf]` | `(8,3)` counts with `n_components=5` clamps with a `UserWarning` and returns `≤3` patterns; no IndexError/ValueError. **Fails before** the Task-7 guard. |
| `tests/behavior/test_goal_directed.py::…::test_approach_rate_geodesic_branch` | `metric='geodesic'` on an open grid yields NaN first sample and negative mean rate when approaching. |
| `tests/behavior/test_goal_directed.py::…::test_approach_rate_geodesic_requires_env` | `metric='geodesic'` with `env=None` raises `ValueError`. |
| `tests/animation/test_rendering.py::test_field_to_rgb_for_napari_orientation` | A field peak at env `(min x,max y)`/`(min x,min y)`/`(max x,max y)` lands at napari pixel `(0,0)`/`(3,0)`/`(0,3)`. |

Only the **Task-7** test (`test_n_time_bins_below_n_components_does_not_crash`)
has a true fail-before state (it pairs with the source guard). All other tests in
this phase pin **current correct behavior** that simply lacked coverage; they pass
against the unmodified source and guard against future regressions.

Mark slow / integration: none of these need `@pytest.mark.slow` — the longest
simulations (OU `duration=200, dt=0.01`; sinusoidal `duration=20`) run in well
under a second each. If CI timing proves otherwise for the OU autocorrelation
test, mark it `@pytest.mark.slow`.

## Fixtures

- **`tests/simulation/test_trajectory_sim.py`** — add a module-local
  `linear_track_env` fixture (Task 6a) building a 3-node `from_graph` 1D track
  with `units="cm"`. Do **not** reuse `simple_1d_env` from
  `tests/simulation/conftest.py` (it is a RegularGrid, `is_linearized_track ==
  False`, which is why the old tests had to skip). Reuse the existing
  `simple_2d_env` fixture for Task 6b.
- **`tests/decoding/test_assemblies.py`** — reuse the existing `rng` fixture
  (module-level, `np.random.default_rng(42)`).
- **`tests/ops/test_ops_alignment.py`** — reuse the existing `MockEnvironment`
  class (only `.bin_centers`/`.n_bins`/`.n_dims`/`._is_fitted` are read).
- **`tests/layout/`, `tests/animation/`, `tests/behavior/`, `tests/ops/`** — all
  test data synthesized inline with seeded `np.random.default_rng(...)`; no
  checked-in fixtures, no new conftest entries.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:

- Every task is implemented as specified; no Theme-6 item listed under
  "Deliberately not in this phase" is re-tested here (no duplicate of phase
  1/4/5/8/9 assertions).
- The **only** source change is the Task-7 `detect_assemblies` clamp guard; no
  other production code is modified (this is a test-backfill phase).
- The Task-7 test fails before the guard (reviewer reverts the clamp locally and
  confirms `IndexError`/`ValueError` for pca/ica/nmf), and every other test
  passes against unmodified source.
- Per `testing-anti-patterns`: tests exercise **real behavior**, not tautologies
  or mock-echo. Specifically — the IDW values are hand-computed (not read back
  from the function under test); the `field_to_rgb_for_napari` orientation test
  uses an identity colormap to decode actual pixel placement; the OU ACF test
  asserts a non-trivial `0 < acf_lag < acf0` band rather than the old 150%
  tolerance; the `MockEnvironment` is a thin data holder for pure functions, not
  a stub whose return value is then asserted.
- The three `pytest.skip` sinusoidal placeholders are **removed**, not left
  alongside the new tests.
- New test names, docstrings, and the fixture name do not reference this plan,
  phase numbers, or milestones.
- Shared setup (the `linear_track_env` graph, the IDW mock envs) lives in
  fixtures/helpers where reused, not copy-pasted across tests.
- `uv run pytest` passes for all new tests (run the touched files explicitly:
  `uv run pytest tests/layout/test_validation_helpers.py tests/layout/test_regular_grid_utils.py tests/ops/test_ops_alignment.py tests/ops/test_ops_normalize.py tests/ops/test_visibility.py tests/simulation/test_trajectory_sim.py tests/decoding/test_assemblies.py tests/behavior/test_goal_directed.py tests/animation/test_rendering.py`).
