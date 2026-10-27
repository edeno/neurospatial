"""Tests for compute_egocentric_rate(None) function.

This module tests the compute_egocentric_rate(None) function for computing egocentric
firing rate maps for object-vector cells.

compute_egocentric_rate():
- Accept single spike_times array
- Accept object_positions array
- Support distance_range and bin count parameters
- Support metric parameter
- Optional env parameter (required for geodesic)
- Apply smoothing via _smoothing.py
- Return ObjectVectorRateResult
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from neurospatial import Environment

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing (required for geodesic)."""
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0, units="cm")


@pytest.fixture
def trajectory_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sample trajectory data (times, positions, headings)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    times = np.linspace(0, 100, n_samples)
    # Random walk within environment
    positions = np.column_stack(
        [
            50 + np.cumsum(rng.standard_normal(n_samples) * 0.5),
            50 + np.cumsum(rng.standard_normal(n_samples) * 0.5),
        ]
    )
    positions = np.clip(positions, 10, 90)  # Keep within bounds
    # Random headings
    headings = rng.uniform(-np.pi, np.pi, n_samples)
    return times, positions, headings


@pytest.fixture
def object_positions() -> np.ndarray:
    """Object positions for testing."""
    return np.array([[50.0, 50.0], [25.0, 75.0]])


@pytest.fixture
def spike_times() -> np.ndarray:
    """Spike times within the trajectory time range."""
    rng = np.random.default_rng(123)
    return np.sort(rng.uniform(0.1, 99.9, 100))


@pytest.fixture
def empty_spike_times() -> np.ndarray:
    """Empty spike train."""
    return np.array([], dtype=np.float64)


# ==============================================================================
# Import Tests
# ==============================================================================


class TestComputeEgocentricRateImport:
    """Test that compute_egocentric_rate can be imported."""

    def test_import_from_egocentric(self) -> None:
        """compute_egocentric_rate can be imported from encoding.egocentric."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        assert compute_egocentric_rate is not None
        assert callable(compute_egocentric_rate)

    def test_import_from_encoding(self) -> None:
        """compute_egocentric_rate can be imported from encoding package."""
        from neurospatial.encoding import compute_egocentric_rate

        assert compute_egocentric_rate is not None
        assert callable(compute_egocentric_rate)

    def test_in_all(self) -> None:
        """compute_egocentric_rate is in __all__."""
        from neurospatial.encoding import egocentric

        assert "compute_egocentric_rate" in egocentric.__all__


# ==============================================================================
# Return Type Tests
# ==============================================================================


class TestComputeEgocentricRateReturnsResult:
    """Test that compute_egocentric_rate returns ObjectVectorRateResult."""

    def test_returns_egocentric_rate_result(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """compute_egocentric_rate should return an ObjectVectorRateResult object."""
        from neurospatial.encoding.egocentric import (
            ObjectVectorRateResult,
            compute_egocentric_rate,
        )

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert isinstance(result, ObjectVectorRateResult)

    def test_firing_rate_shape(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert np.asarray(result.firing_rate).ndim == 1
        assert np.asarray(result.firing_rate).shape == (result.env.n_bins,)

    def test_occupancy_shape(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert np.asarray(result.occupancy).shape == (result.env.n_bins,)

    def test_env_is_environment(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """env should be an Environment."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        from neurospatial import Environment

        # Egocentric rate runs on a distinct polar environment type, not the
        # Cartesian Environment.
        assert isinstance(result.env, Environment)
        assert result.env.coordinate_kind == "polar"


# ==============================================================================
# Distance Range and Bin Count Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateDistanceRange:
    """Test distance_range parameter."""

    @pytest.mark.parametrize(("units", "scale"), [("cm", 1.0), ("m", 0.01)])
    @pytest.mark.parametrize("frame", ["allocentric", "egocentric"])
    def test_default_grid_covers_the_environment_in_5_cm_rings(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        units: str,
        scale: float,
        frame: str,
    ) -> None:
        """Every distance inside the arena is in the map, in whole 5 cm rings."""
        from neurospatial.encoding import (
            compute_egocentric_rate,
            compute_object_vector_rate,
        )

        times, positions, headings = trajectory_data
        env = Environment.from_samples(positions * scale, bin_size=5.0 * scale)
        env = env.with_metadata(units=units)
        if frame == "allocentric":
            result = compute_object_vector_rate(
                env, spike_times, times, positions * scale, object_positions * scale
            )
        else:
            result = compute_egocentric_rate(
                env,
                spike_times,
                times,
                positions * scale,
                headings,
                object_positions * scale,
            )
        diagonal = np.hypot(*[stop - start for start, stop in env.dimension_ranges])
        low, high = result.distance_range
        ring = (high - low) / result.n_distance_bins
        assert low == 0.0
        assert ring == pytest.approx(5.0 * scale)
        assert diagonal < high <= diagonal + ring
        assert result.n_direction_bins == 12
        assert result.analysis_settings[
            "fraction_of_time_in_distance_range"
        ] == pytest.approx(1.0)

    @pytest.mark.parametrize("units", [None, "pixels"])
    def test_unknown_units_need_an_explicit_grid(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        units: str | None,
    ) -> None:
        """5 cm has no meaning without a length scale; no env has no extent."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        env = Environment.from_samples(positions, bin_size=5.0, units=units)
        with pytest.raises(ValueError, match="distance_bin_size is required"):
            compute_egocentric_rate(
                env, spike_times, times, positions, headings, object_positions
            )
        result = compute_egocentric_rate(
            env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_bin_size=5.0,
        )
        assert result.distance_range[1] > 0
        with pytest.raises(ValueError, match="distance_range is required"):
            compute_egocentric_rate(
                None, spike_times, times, positions, headings, object_positions
            )

    @pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde"])
    def test_unknown_units_need_an_explicit_kernel_bandwidth(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        method: str,
    ) -> None:
        """No kernel width can be inferred from polar edges of mixed length."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        settings = {
            "distance_range": (0.0, 50.0),
            "distance_bin_size": 5.0,
            "method": method,
        }
        with pytest.raises(ValueError, match="needs a bandwidth"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                **settings,
            )
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            bandwidth=5.0,
            **settings,
        )
        assert result.analysis_settings["bandwidth"] == 5.0

    def test_custom_distance_range(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom distance_range should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(5.0, 60.0),
            distance_bin_size=5.0,
        )
        assert result.distance_range == (5.0, 60.0)


def _far_object_vector_cell(
    preferred_distance: float,
) -> tuple[Environment, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cell firing 20-60 cm out from an object in the centre of a 150 cm box.

    Positions are sampled uniformly over the box at 30 Hz for 20 minutes; the
    cell fires when the animal is ``preferred_distance`` east of the object
    (sd 5 cm, kappa 4).
    """
    from neurospatial.simulation import generate_poisson_spikes

    rng = np.random.default_rng(0)
    times = np.arange(0.0, 1200.0, 1 / 30)
    positions = rng.uniform(0.0, 150.0, (len(times), 2))
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    obj = np.array([[75.0, 75.0]])
    delta = positions - obj
    distance = np.hypot(*delta.T)
    direction = np.arctan2(delta[:, 1], delta[:, 0])
    rate = 0.2 + 15.0 * np.exp(-0.5 * ((distance - preferred_distance) / 5.0) ** 2) * (
        np.exp(4.0 * (np.cos(direction) - 1.0))
    )
    spikes = generate_poisson_spikes(rate, times, seed=1)
    return env, spikes, times, positions, obj


class TestObjectVectorRangeCoverage:
    """The map covers the arena, and says when an explicit range cut time off."""

    def test_default_range_recovers_a_field_beyond_50_cm(self) -> None:
        from neurospatial.encoding import compute_object_vector_rate

        env, spikes, times, positions, obj = _far_object_vector_cell(60.0)
        result = compute_object_vector_rate(env, spikes, times, positions, obj)
        ring = (result.distance_range[1] - result.distance_range[0]) / (
            result.n_distance_bins
        )
        assert abs(result.preferred_distance() - 60.0) <= ring
        # The animal is east of the object, so the object lies west of it.
        west = np.angle(np.exp(1j * (result.preferred_direction() - np.pi)))
        assert abs(west) <= np.pi / 6

    def test_fraction_of_time_in_range_matches_the_geometry(self) -> None:
        """Uniform positions: the 50 cm disc holds pi 50^2 / 150^2 of the time."""
        from neurospatial.encoding import compute_object_vector_rate

        env, spikes, times, positions, obj = _far_object_vector_cell(60.0)
        result = compute_object_vector_rate(
            env, spikes, times, positions, obj, distance_range=(0.0, 50.0)
        )
        fraction = result.analysis_settings["fraction_of_time_in_distance_range"]
        assert fraction == pytest.approx(np.pi * 50.0**2 / 150.0**2, abs=0.01)
        beyond = result.analysis_settings["time_beyond_distance_range"]
        assert beyond == pytest.approx(
            float(result.occupancy.sum()) * (1 - fraction) / fraction
        )

    def test_peak_in_the_outermost_ring_of_a_cut_range_warns(self) -> None:
        from neurospatial.encoding import compute_object_vector_rate

        env, spikes, times, positions, obj = _far_object_vector_cell(52.0)
        with pytest.warns(UserWarning, match="outermost ring"):
            compute_object_vector_rate(
                env, spikes, times, positions, obj, distance_range=(0.0, 50.0)
            )

    def test_a_field_inside_the_range_does_not_warn(self) -> None:
        import warnings

        from neurospatial.encoding import compute_object_vector_rate

        env, spikes, times, positions, obj = _far_object_vector_cell(30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_object_vector_rate(
                env, spikes, times, positions, obj, distance_range=(0.0, 50.0)
            )
            compute_object_vector_rate(env, spikes, times, positions, obj)


class TestUnsampledObjectDirections:
    """An object outside the arena leaves directions unsampled; say so."""

    @pytest.mark.parametrize("call", ["single", "population", "significance"])
    def test_object_outside_the_arena_warns(self, call: str) -> None:
        from neurospatial.encoding import (
            compute_object_vector_rate,
            compute_object_vector_rates,
            object_vector_cell_significance,
        )

        env, spikes, times, positions, _ = _far_object_vector_cell(30.0)
        outside = np.array([[175.0, 75.0]])  # 25 cm east of the 150 cm box
        with pytest.warns(UserWarning, match="no time was spent") as record:
            if call == "single":
                result = compute_object_vector_rate(
                    env, spikes, times, positions, outside
                )
            elif call == "population":
                result = compute_object_vector_rates(
                    env, [spikes], times, positions, outside
                )[0]
            else:
                object_vector_cell_significance(
                    env, [spikes], times, positions, outside, n_shuffles=3, rng=0
                )
        unsampled = [w for w in record if "no time was spent" in str(w.message)]
        assert len(unsampled) == 1  # once per call, not once per shuffle
        # Seen from the animal, the object is always within 72 degrees of due
        # east, so the directions from north through west to south are empty.
        assert "directions [90, 270) degrees" in str(unsampled[0].message)
        if call != "significance":
            occupancy = np.asarray(result.occupancy).reshape(
                result.n_distance_bins, result.n_direction_bins
            )
            assert not (occupancy > 0).any(axis=0).all()

    def test_object_inside_the_arena_does_not_warn(self) -> None:
        import warnings

        from neurospatial.encoding import compute_object_vector_rate

        env, spikes, times, positions, obj = _far_object_vector_cell(30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_object_vector_rate(env, spikes, times, positions, obj)


class TestComputeEgocentricRateBinCounts:
    """Test n_distance_bins and n_direction_bins parameters."""

    def test_default_n_distance_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default n_distance_bins should be 10."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert result.n_distance_bins == 10

    def test_default_n_direction_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default n_direction_bins should be 12."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert result.n_direction_bins == 12

    def test_custom_n_distance_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom n_distance_bins should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_bin_size=2.5,
            distance_range=(0.0, 50.0),
        )
        assert result.n_distance_bins == 20

    def test_custom_n_direction_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom n_direction_bins should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            direction_bin_size=2 * np.pi / 24,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert result.n_direction_bins == 24

    def test_n_bins_equals_product(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """env.n_bins should equal n_distance_bins * n_direction_bins."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        n_dist = 8
        n_dir = 16
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_bin_size=(50.0 - 0.0) / n_dist,
            direction_bin_size=2 * np.pi / n_dir,
            distance_range=(0.0, 50.0),
        )
        assert result.env.n_bins == n_dist * n_dir


# ==============================================================================
# Distance Metric Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateDistanceMetric:
    """Test metric parameter."""

    def test_default_metric_euclidean(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default metric should be 'euclidean' (no env needed)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        # Should work without env parameter
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert isinstance(result.firing_rate, np.ndarray) or hasattr(
            result.firing_rate, "__array__"
        )

    def test_euclidean_metric_explicit(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Explicit euclidean metric works without env."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="euclidean",
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_geodesic_requires_env(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """geodesic metric requires env parameter."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                metric="geodesic",
                distance_range=(0.0, 50.0),
                distance_bin_size=5.0,
            )

    def test_geodesic_with_env(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """geodesic metric works with env parameter."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="geodesic",
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_invalid_metric_raises(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Invalid metric raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match="Invalid metric"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                metric="manhattan",  # type: ignore[arg-type],
                distance_range=(0.0, 50.0),
                distance_bin_size=5.0,
            )


# ==============================================================================
# Smoothing Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateSmoothing:
    """Test smoothing parameters."""

    @pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
    def test_accepts_valid_methods(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    ) -> None:
        """compute_egocentric_rate should accept all valid smoothing methods."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            method=method,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
            bandwidth=5.0,
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_default_method(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default method should be 'binned'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        explicit_result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            method="binned",
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )

        np.testing.assert_allclose(
            np.asarray(result.firing_rate),
            np.asarray(explicit_result.firing_rate),
            equal_nan=True,
        )

    def test_bandwidth_parameter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """bandwidth parameter should be accepted."""
        from neurospatial.encoding import _smoothing
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        captured: dict[str, object] = {}
        original_smooth_rate_map = _smoothing.smooth_rate_map

        def capture_smooth_rate_map(*args, **kwargs):
            captured["method"] = kwargs["method"]
            captured["bandwidth"] = kwargs["bandwidth"]
            return original_smooth_rate_map(*args, **kwargs)

        monkeypatch.setattr(_smoothing, "smooth_rate_map", capture_smooth_rate_map)

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            method="gaussian_kde",
            bandwidth=10.0,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        assert np.asarray(result.firing_rate).shape[0] > 0
        assert captured == {"method": "gaussian_kde", "bandwidth": 10.0}


# ==============================================================================
# Empty Spike Train Tests
# ==============================================================================


class TestComputeEgocentricRateEmptySpikes:
    """Test behavior with empty spike train."""

    def test_empty_spikes_returns_zero_rate(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should return zero firing rate."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            empty_spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        firing_rate = np.asarray(result.firing_rate)
        # All non-NaN values should be zero
        assert np.allclose(firing_rate[~np.isnan(firing_rate)], 0.0)

    def test_empty_spikes_has_positive_occupancy(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should still have positive occupancy."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            empty_spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        occupancy = np.asarray(result.occupancy)
        assert np.sum(occupancy) > 0


class TestComputeEgocentricRatePrecomputation:
    """Regression tests for single-neuron precomputation efficiency."""

    def test_egocentric_coordinates_computed_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """compute_egocentric_rate should reuse coordinates for counts and occupancy."""
        from neurospatial.encoding import _egocentric_counts
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        call_count = 0
        original = _egocentric_counts._compute_object_coords

        def counting_compute_object_coords(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            _egocentric_counts,
            "_compute_object_coords",
            counting_compute_object_coords,
        )

        compute_egocentric_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="geodesic",
        )

        assert call_count == 1


# ==============================================================================
# Correctness Tests
# ==============================================================================


class TestComputeEgocentricRateCorrectness:
    """Test correctness of computed values."""

    def test_firing_rate_non_negative(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Firing rate should be non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        firing_rate = np.asarray(result.firing_rate)
        # Exclude NaN values
        non_nan = firing_rate[~np.isnan(firing_rate)]
        assert np.all(non_nan >= 0)

    def test_occupancy_non_negative(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Occupancy should be non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        occupancy = np.asarray(result.occupancy)
        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximates_duration(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Total occupancy should approximately equal trajectory duration."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        duration = times[-1] - times[0]
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        occupancy = np.asarray(result.occupancy)
        total_occupancy = np.sum(occupancy)
        # Should be within 10% (some bins may be outside distance range)
        assert total_occupancy <= duration * 1.1
        assert total_occupancy > duration * 0.5  # At least half should be covered


# ==============================================================================
# Result Methods Tests
# ==============================================================================


class TestComputeObjectVectorRateResultMethods:
    """Test that result methods work correctly."""

    def test_plot_returns_axes(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """plot() should return matplotlib Axes."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        import matplotlib.pyplot as plt

        ax = result.plot()
        assert ax is not None
        plt.close()

    def test_preferred_distance_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """preferred_distance() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        dist = result.preferred_distance()
        assert isinstance(dist, float)

    def test_preferred_direction_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """preferred_direction() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        direction = result.preferred_direction()
        assert isinstance(direction, float)

    def test_spatial_information_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """spatial_information() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        info = result.spatial_information()
        assert isinstance(info, float)

    def test_is_ovc_returns_bool(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """is_egocentric_object_vector_cell() should return a bool."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        is_egocentric_object_vector_cell = result.is_object_vector_cell()
        assert isinstance(is_egocentric_object_vector_cell, (bool, np.bool_))


# ==============================================================================
# min_occupancy Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateMinOccupancy:
    """Test min_occupancy parameter."""

    @pytest.mark.parametrize("method", ["binned", "diffusion_kde", "gaussian_kde"])
    def test_nan_exactly_where_occupancy_is_below_min_occupancy(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        method: str,
    ) -> None:
        """min_occupancy is seconds per bin: below it the rate is unmeasured."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        settings = {
            "distance_range": (0.0, 50.0),
            "distance_bin_size": 5.0,
            "method": method,
            "bandwidth": 5.0,
        }
        args = (None, spike_times, times, positions, headings, object_positions)
        occupancy = np.asarray(compute_egocentric_rate(*args, **settings).occupancy)
        threshold = float(np.median(occupancy[occupancy > 0]))
        result = compute_egocentric_rate(*args, min_occupancy=threshold, **settings)
        unsupported = (occupancy == 0) | (occupancy < threshold)
        assert unsupported.any() and not unsupported.all()
        np.testing.assert_array_equal(
            np.isnan(np.asarray(result.firing_rate)), unsupported
        )

    def test_default_min_occupancy_zero(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default min_occupancy should be 0.0."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        # Should work without specifying min_occupancy
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
        )
        # With min_occupancy=0, bins with occupancy>0 should have values
        occupancy = np.asarray(result.occupancy)
        firing_rate = np.asarray(result.firing_rate)
        occupied_mask = occupancy > 0
        # At least some occupied bins should have non-NaN values
        assert np.any(~np.isnan(firing_rate[occupied_mask]))


# ==============================================================================
# Input Validation Tests
# ==============================================================================


class TestComputeEgocentricRateInputValidation:
    """Test input validation."""

    def test_mismatched_times_positions_raises(
        self,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and positions lengths should raise ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times = np.linspace(0, 100, 1000)
        rng = np.random.default_rng(42)
        positions = rng.random((500, 2)) * 100  # Wrong length
        headings = rng.uniform(-np.pi, np.pi, 1000)

        with pytest.raises(ValueError, match=r"times.*positions"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                distance_bin_size=5.0,
            )

    def test_mismatched_times_headings_raises(
        self,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and headings lengths should raise ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times = np.linspace(0, 100, 1000)
        rng = np.random.default_rng(42)
        positions = rng.random((1000, 2)) * 100
        headings = rng.uniform(-np.pi, np.pi, 500)  # Wrong length

        with pytest.raises(ValueError, match=r"times.*headings"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                distance_bin_size=5.0,
            )


# ==============================================================================
# Signature Tests
# ==============================================================================


class TestComputeEgocentricRateSignature:
    """Test function signature follows conventions."""

    def test_argument_order(self) -> None:
        """Arguments should follow canonical order from CLAUDE.md (env first)."""
        import inspect

        from neurospatial.encoding.egocentric import compute_egocentric_rate

        sig = inspect.signature(compute_egocentric_rate)
        params = list(sig.parameters.keys())

        # First 6 positional args should be in this order (env first per CLAUDE.md)
        assert params[0] == "env"
        assert params[1] == "spike_times"
        assert params[2] == "times"
        assert params[3] == "positions"
        assert params[4] == "headings"
        assert params[5] == "object_positions"

    def test_keyword_only_parameters(self) -> None:
        """Optional parameters should be keyword-only."""
        import inspect

        from neurospatial.encoding.egocentric import compute_egocentric_rate

        sig = inspect.signature(compute_egocentric_rate)

        # These should be keyword-only (after *).
        # `env` is now first positional (canonical order); excluded here.
        keyword_only_params = [
            "distance_range",
            "n_distance_bins",
            "n_direction_bins",
            "metric",
            "method",
            "bandwidth",
            "min_occupancy",
        ]

        for param_name in keyword_only_params:
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{param_name} should be keyword-only"
                )


class TestEgocentricRateConvention:
    """End-to-end recovery of the egocentric direction and distance tuning.

    Drives the public ``compute_egocentric_rate`` on Poisson spikes from an
    object-vector cell with a known egocentric ``preferred_direction`` and
    ``preferred_distance`` and asserts the recovered peak matches both within
    one bin. This pins the shared egocentric convention (0=ahead, +pi/2=left,
    -pi/2=right) between the simulator and the analysis: a sign flip in
    ``compute_egocentric_bearing`` or a swapped left/right convention would
    fail at least one parametrization.

    The animal moves on an OU trajectory (the analysis needs many sampled
    egocentric configurations; a stationary animal collapses occupancy into a
    single bin and the recovered map is dominated by smoothing artifacts).

    Distance recovery exercises the default ``"binned"`` method, which reports
    the raw bin rate (spikes / occupancy). Graph-diffusion smoothing on the
    polar environment bleeds rate radially across distance rings and flattens
    the distance marginal, so it cannot recover ``preferred_distance``; the
    binned rate can.
    """

    N_DIRECTION_BINS = 12
    DISTANCE_RANGE = (0.0, 50.0)

    @pytest.fixture(scope="class")
    def trajectory(self):
        from neurospatial import Environment
        from neurospatial.behavior import heading_from_velocity
        from neurospatial.simulation import simulate_trajectory_ou

        samples = np.random.default_rng(0).uniform(0, 40, (3000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)
        env = env.with_metadata(units="cm")
        positions, times = simulate_trajectory_ou(
            env, duration=1500.0, speed_units="cm", seed=7
        )
        # The cell fires on the movement direction at every sample; the
        # analysis sees a heading only where the animal moves at >= 2 cm/s.
        movement = heading_from_velocity(times, positions)
        headings = heading_from_velocity(times, positions, min_speed=2.0)
        return env, positions, times, movement, headings

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "preferred_direction",
        [0.0, np.pi / 2, -np.pi / 2],
        ids=["ahead", "left", "right"],
    )
    def test_preferred_direction_recovered(self, trajectory, preferred_direction):
        from neurospatial.encoding.egocentric import compute_egocentric_rate
        from neurospatial.simulation import generate_poisson_spikes
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        env, positions, times, movement, headings = trajectory
        object_positions = np.array([[20.0, 20.0]])  # object at arena center
        model = ObjectVectorCellModel(
            direction_frame="egocentric",
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=4.0,
            preferred_direction=preferred_direction,
            direction_kappa=4.0,
            max_rate=25.0,
            baseline_rate=0.1,
            metric="euclidean",
        )
        rates = model.firing_rate(positions, headings=movement)
        # Independent seed from the trajectory RNG so spike-acceptance uniforms
        # are not correlated with the simulated state.
        spike_times = generate_poisson_spikes(rates, times, seed=99)

        result = compute_egocentric_rate(
            env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="euclidean",
            distance_range=self.DISTANCE_RANGE,
            distance_bin_size=50.0 / 10,
            direction_bin_size=2 * np.pi / self.N_DIRECTION_BINS,
        )

        # Direction convention recovered within one direction bin.
        direction_bin = 2 * np.pi / self.N_DIRECTION_BINS
        circ_dist = abs(
            np.angle(np.exp(1j * (result.preferred_direction() - preferred_direction)))
        )
        assert circ_dist < direction_bin

        # Distance tuning recovered within one distance bin. DISTANCE_RANGE
        # spans 0-50 cm over 10 bins -> 5 cm/bin; simulated peak is 10 cm.
        # Observed recovery is ~12.5 cm (the 10-15 cm bin center), so the
        # one-bin tolerance is intentional: it is tight enough that the old
        # diffusion-smoothed behavior (peak collapsed to ~2.5 cm) fails it.
        distance_bin = (self.DISTANCE_RANGE[1] - self.DISTANCE_RANGE[0]) / 10
        assert abs(result.preferred_distance() - 10.0) < distance_bin


class TestEgocentricRateNaNHandling:
    """NaN positions/headings are missing samples, treated as deleted rows."""

    @pytest.mark.parametrize("nan_field", ["positions", "headings"])
    def test_nan_samples_equal_deleted_rows(
        self, ou_2min, ou_2min_env, noise_trains, nan_field
    ):
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = ou_2min
        # Every sample of the 100 cm box is within 80 cm of its center.
        settings = {"distance_range": (0.0, 80.0), "metric": "euclidean"}
        obj = np.array([[50.0, 50.0]])
        drop = np.random.default_rng(5).choice(len(times), len(times) // 10, False)
        data = {"positions": positions.copy(), "headings": headings.copy()}
        data[nan_field][drop] = np.nan
        spikes = noise_trains(120)[0]
        result = compute_egocentric_rate(
            ou_2min_env,
            spikes,
            times,
            data["positions"],
            data["headings"],
            obj,
            **settings,
        )
        kept = np.setdiff1d(np.arange(len(times)), drop)
        deleted = compute_egocentric_rate(
            ou_2min_env,
            spikes,
            times[kept],
            positions[kept],
            headings[kept],
            obj,
            **settings,
        )
        np.testing.assert_allclose(result.occupancy, deleted.occupancy)
        np.testing.assert_allclose(result.firing_rate, deleted.firing_rate)
        assert np.isfinite(np.nanmax(result.firing_rate))


@pytest.mark.parametrize(
    ("side", "bearing"), [("left", np.pi / 2), ("right", -np.pi / 2)]
)
def test_an_object_on_the_left_is_at_plus_pi_over_2(side, bearing):
    """Egocentric bearing: 0 ahead, +pi/2 left, -pi/2 right.

    The expected bearing is computed here from first principles, the
    direction to the object minus the heading, independently of
    ``compute_egocentric_bearing`` and the simulation models.
    """
    from neurospatial.encoding.egocentric import compute_egocentric_rate
    from neurospatial.simulation import generate_poisson_spikes

    rng = np.random.default_rng(0)
    times = np.arange(0, 300, 1 / 30)
    positions = rng.uniform(0, 100, (len(times), 2))
    headings = rng.uniform(-np.pi, np.pi, len(times))
    obj = np.array([[50.0, 50.0]])
    delta = obj - positions
    to_object = np.arctan2(delta[:, 1], delta[:, 0]) - headings
    distance = np.hypot(delta[:, 0], delta[:, 1])
    rate = 0.2 + 20 * np.exp(2 * (np.cos(to_object - bearing) - 1)) * np.exp(
        -0.5 * ((distance - 20) / 5) ** 2
    )
    spikes = generate_poisson_spikes(rate, times, seed=1)

    result = compute_egocentric_rate(
        None,
        spikes,
        times,
        positions,
        headings,
        obj,
        distance_range=(0.0, 50.0),
        distance_bin_size=5.0,
    )
    error = np.angle(np.exp(1j * (result.preferred_direction() - bearing)))
    assert abs(error) <= np.pi / 12 + 1e-12, side  # within one 30 degree bin
    assert 15.0 <= result.preferred_distance() <= 25.0
