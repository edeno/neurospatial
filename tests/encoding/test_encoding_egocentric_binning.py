"""Tests for egocentric binning layer.

This module tests the binning layer for egocentric (object-vector) encoding,
through the path the encoders use: ``prepare_egocentric_binning`` followed by
``FrameBinning.count``. It computes:
1. Egocentric occupancy (time spent at each distance/direction from objects)
2. Spike counts binned by egocentric coordinates at spike time
3. Populations of neurons sharing one occupancy

The egocentric binning differs from spatial binning in that:
- Spatial binning: bins by *where the animal was*
- Egocentric binning: bins by *distance/direction to nearest object*
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._egocentric_counts import prepare_egocentric_binning
from neurospatial.encoding.egocentric import compute_egocentric_rate

GRID = {"distance_range": (0.0, 50.0), "n_distance_bins": 10, "n_direction_bins": 12}
"""A 10 x 12 polar grid out to 50 units."""


def _bin(
    spike_trains, times, positions, headings, object_positions, *, n_jobs=1, **kwargs
):
    """Counts ``(n_neurons, n_bins)``, occupancy ``(n_bins,)`` and the polar env."""
    prepared, polar_env, _ = prepare_egocentric_binning(
        times, positions, headings, object_positions, **kwargs
    )
    counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)
    return counts, occupancy, polar_env


def _counts(spikes, times, positions, headings, object_positions, **kwargs):
    """One neuron's spike counts, shape ``(n_bins,)``."""
    return _bin([spikes], times, positions, headings, object_positions, **kwargs)[0][0]


def _trajectory(data):
    return data["times"], data["positions"], data["headings"]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_positions() -> np.ndarray:
    """Sample positions for environment creation and testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, (500, 2))


@pytest.fixture
def env(sample_positions: np.ndarray) -> Environment:
    """Create a test environment."""
    return Environment.from_samples(sample_positions, bin_size=5.0)


@pytest.fixture
def trajectory_data() -> dict:
    """Create trajectory data for testing."""
    rng = np.random.default_rng(42)
    n_time = 1000
    return {
        "times": np.linspace(0, 100, n_time),
        "positions": rng.uniform(10, 90, (n_time, 2)),
        "headings": rng.uniform(-np.pi, np.pi, n_time),
    }


@pytest.fixture
def object_positions() -> np.ndarray:
    """Object positions in allocentric coordinates."""
    return np.array([[50.0, 50.0], [25.0, 75.0]])


@pytest.fixture
def spike_times() -> np.ndarray:
    """Sample spike times for single neuron."""
    rng = np.random.default_rng(42)
    return np.sort(rng.uniform(0, 100, 200))


@pytest.fixture
def spike_times_list() -> list[np.ndarray]:
    """Sample spike times for multiple neurons."""
    rng = np.random.default_rng(42)
    return [
        np.sort(rng.uniform(0, 100, 100)),  # Neuron 0
        np.sort(rng.uniform(0, 100, 150)),  # Neuron 1
        np.sort(rng.uniform(0, 100, 50)),  # Neuron 2
    ]


# =============================================================================
# Occupancy
# =============================================================================


class TestEgocentricOccupancy:
    """Time spent at each distance and direction from the nearest object."""

    def test_returns_polar_env(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Binning builds an egocentric polar environment."""
        _, _, polar_env = _bin(
            [], *_trajectory(trajectory_data), object_positions, **GRID
        )

        # Egocentric occupancy builds a distinct polar environment type.
        assert isinstance(polar_env, Environment)
        assert polar_env.coordinate_kind == "polar"

    def test_occupancy_shape_and_dtype(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Occupancy is float64 with one value per polar bin."""
        _, occupancy, polar_env = _bin(
            [], *_trajectory(trajectory_data), object_positions, **GRID
        )
        expected_n_bins = GRID["n_distance_bins"] * GRID["n_direction_bins"]
        assert isinstance(occupancy, np.ndarray)
        assert occupancy.dtype == np.float64
        assert occupancy.shape == (expected_n_bins,)
        assert len(occupancy) == polar_env.n_bins

    def test_occupancy_non_negative(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Occupancy values are non-negative."""
        _, occupancy, _ = _bin(
            [], *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximates_duration(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Total occupancy approximates recording duration."""
        times = trajectory_data["times"]
        duration = times[-1] - times[0]

        # Use large distance range to capture all samples
        _, occupancy, _ = _bin(
            [],
            *_trajectory(trajectory_data),
            object_positions,
            distance_range=(0.0, 200.0),  # Large range to capture all
            n_distance_bins=20,
            n_direction_bins=12,
        )
        # Total occupancy should be close to duration (minus one dt for n-1 intervals)
        assert abs(np.sum(occupancy) - duration) < duration * 0.1


class TestEgocentricDistanceMetric:
    """Tests for metric parameter."""

    def test_euclidean_default(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Euclidean is the default distance metric and needs no env."""
        counts, occupancy, polar_env = _bin(
            [spike_times], *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert counts.shape == (1, polar_env.n_bins)
        assert occupancy.shape == (polar_env.n_bins,)
        assert occupancy.sum() > 0

    def test_geodesic_requires_env(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Geodesic distance metric requires env parameter."""
        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            prepare_egocentric_binning(
                *_trajectory(trajectory_data),
                object_positions,
                **GRID,
                metric="geodesic",
            )

    def test_geodesic_with_env(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Geodesic distance works when env is provided."""
        counts, occupancy, polar_env = _bin(
            spike_times_list,
            *_trajectory(trajectory_data),
            object_positions,
            **GRID,
            metric="geodesic",
            env=env,
        )
        assert counts.shape == (len(spike_times_list), polar_env.n_bins)
        assert occupancy.shape == (polar_env.n_bins,)

    def test_invalid_metric_raises(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Invalid distance metric raises ValueError."""
        with pytest.raises(ValueError, match=r"invalid|Invalid"):
            prepare_egocentric_binning(
                *_trajectory(trajectory_data),
                object_positions,
                **GRID,
                metric="manhattan",  # type: ignore[arg-type]
            )


class TestEgocentricInputValidation:
    """Misaligned or malformed samples are refused."""

    def test_mismatched_times_positions_raises(self, object_positions: np.ndarray):
        """Mismatched times and positions lengths raise error."""
        times = np.linspace(0, 10, 100)
        positions = np.random.rand(90, 2)  # Wrong length
        headings = np.random.rand(100)

        with pytest.raises(ValueError, match=r"times.*positions"):
            compute_egocentric_rate(
                None,
                np.array([1.0]),
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
            )

    def test_mismatched_times_headings_raises(self, object_positions: np.ndarray):
        """Mismatched times and headings lengths raise error."""
        times = np.linspace(0, 10, 100)
        positions = np.random.rand(100, 2)
        headings = np.random.rand(90)  # Wrong length

        with pytest.raises(ValueError, match=r"times.*headings"):
            compute_egocentric_rate(
                None,
                np.array([1.0]),
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
            )

    def test_insufficient_samples_raises(self, object_positions: np.ndarray):
        """Fewer than 2 samples raises error."""
        times = np.array([0.0])  # Only 1 sample
        positions = np.array([[50.0, 50.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="At least 2 samples"):
            prepare_egocentric_binning(
                times, positions, headings, object_positions, **GRID
            )

    def test_unsorted_times_raises(self, object_positions: np.ndarray):
        """Non-monotonic times raise error."""
        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not sorted
        positions = np.random.rand(4, 2)
        headings = np.random.rand(4)

        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            prepare_egocentric_binning(
                times, positions, headings, object_positions, **GRID
            )

    @pytest.mark.parametrize(
        ("short", "message"),
        [
            ("positions", r"times has 100 samples, positions has 90"),
            ("headings", r"headings has 90 samples but times has 100"),
            ("object_positions", r"object_positions has 90 samples but times has 100"),
        ],
    )
    def test_samples_must_align_with_times(self, short: str, message: str):
        """Every per-sample array has one row per timestamp."""
        n = 100
        data = {
            "positions": np.random.default_rng(0).random((n, 2)) * 50,
            "headings": np.zeros(n),
            "object_positions": np.full((n, 1, 2), 25.0),
        }
        data[short] = data[short][:90]
        with pytest.raises(ValueError, match=message):
            prepare_egocentric_binning(
                np.linspace(0, 10, n),
                data["positions"],
                data["headings"],
                data["object_positions"],
                **GRID,
            )

    def test_position_columns_must_match_env(self, object_positions: np.ndarray):
        env = Environment.from_samples(
            np.random.default_rng(0).random((200, 2)) * 50, bin_size=5.0
        )
        positions = np.random.default_rng(1).random((100, 3)) * 50
        with pytest.raises(ValueError, match=r"positions with shape \(100, 3\)"):
            prepare_egocentric_binning(
                np.linspace(0, 10, 100),
                positions,
                np.zeros(100),
                object_positions,
                **GRID,
                env=env,
            )


# =============================================================================
# Spike counts
# =============================================================================


class TestEgocentricSpikeCounts:
    """One neuron's spike counts per polar bin."""

    def test_spike_counts_shape_and_dtype(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Spike counts are float64 (for smoothing) with one value per bin."""
        counts, _, polar_env = _bin(
            [spike_times], *_trajectory(trajectory_data), object_positions, **GRID
        )
        spike_counts = counts[0]
        expected_n_bins = GRID["n_distance_bins"] * GRID["n_direction_bins"]
        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64
        assert spike_counts.shape == (expected_n_bins,)
        assert len(spike_counts) == polar_env.n_bins

    def test_spike_counts_non_negative(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Spike counts are non-negative."""
        spike_counts = _counts(
            spike_times, *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert np.all(spike_counts >= 0)

    def test_total_spike_count_reasonable(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Total spike count is at most number of input spikes."""
        # Use large distance range to capture all spikes
        spike_counts = _counts(
            spike_times,
            *_trajectory(trajectory_data),
            object_positions,
            distance_range=(0.0, 200.0),  # Large range
            n_distance_bins=20,
            n_direction_bins=12,
        )
        assert np.sum(spike_counts) <= len(spike_times)

    def test_empty_spike_train(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Empty spike train returns zeros."""
        counts, _, polar_env = _bin(
            [np.array([])], *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert np.sum(counts) == 0
        assert counts[0].shape == (polar_env.n_bins,)


# =============================================================================
# Populations
# =============================================================================


class TestEgocentricPopulationCounts:
    """Spike counts for several neurons sharing one occupancy."""

    def test_population_shapes_and_dtypes(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Counts are (n_neurons, n_bins); occupancy is (n_bins,); both float64."""
        n_neurons = len(spike_times_list)
        expected_n_bins = GRID["n_distance_bins"] * GRID["n_direction_bins"]

        spike_counts, occupancy, polar_env = _bin(
            spike_times_list, *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert spike_counts.shape == (n_neurons, expected_n_bins)
        assert spike_counts.shape[1] == polar_env.n_bins
        assert occupancy.shape == (expected_n_bins,)
        assert spike_counts.dtype == np.float64
        assert occupancy.dtype == np.float64

    def test_each_row_matches_the_neuron_alone(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """A neuron's counts do not depend on the other neurons counted with it."""
        spike_counts_population, _, _ = _bin(
            spike_times_list, *_trajectory(trajectory_data), object_positions, **GRID
        )

        for i, spike_times in enumerate(spike_times_list):
            spike_counts_single = _counts(
                spike_times, *_trajectory(trajectory_data), object_positions, **GRID
            )
            np.testing.assert_array_equal(
                spike_counts_population[i], spike_counts_single
            )

    def test_empty_neuron_list(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Empty neuron list returns correct shapes."""
        expected_n_bins = GRID["n_distance_bins"] * GRID["n_direction_bins"]

        spike_counts, occupancy, _ = _bin(
            [], *_trajectory(trajectory_data), object_positions, **GRID
        )
        assert spike_counts.shape == (0, expected_n_bins)
        assert occupancy.shape == (expected_n_bins,)

    def test_n_jobs_parameter(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """n_jobs parameter does not change the counts."""
        spike_counts_seq, _, _ = _bin(
            spike_times_list,
            *_trajectory(trajectory_data),
            object_positions,
            **GRID,
            n_jobs=1,
        )
        spike_counts_par, _, _ = _bin(
            spike_times_list,
            *_trajectory(trajectory_data),
            object_positions,
            **GRID,
            n_jobs=2,
        )

        np.testing.assert_array_equal(spike_counts_seq, spike_counts_par)


# =============================================================================
# Test egocentric coordinate correctness
# =============================================================================


class TestEgocentricCoordinates:
    """Tests for correct egocentric coordinate computation."""

    def test_bearing_zero_ahead(self):
        """Object directly ahead should have bearing 0."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin facing East (heading=0), object at (10, 0)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[10.0, 0.0]])

        distances, bearings = _compute_object_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], 0.0)

    def test_bearing_left(self):
        """Object to the left should have bearing ~pi/2."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin facing East (heading=0), object at (0, 10)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[0.0, 10.0]])

        distances, bearings = _compute_object_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], np.pi / 2)

    def test_bearing_right(self):
        """Object to the right should have bearing ~-pi/2."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin facing East (heading=0), object at (0, -10)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[0.0, -10.0]])

        distances, bearings = _compute_object_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], -np.pi / 2)

    def test_nearest_object_selection(self):
        """Nearest object should be selected at each timepoint."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin, object1 at (5,0), object2 at (10,0)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[5.0, 0.0], [10.0, 0.0]])

        distances, _bearings = _compute_object_coords(
            positions, headings, object_positions
        )

        # Should return distance to NEAREST object (5)
        np.testing.assert_almost_equal(distances[0, 0], 5.0)


class TestCoordsToFlatBinIdxWrap:
    """Tests for the angular wrap-around at the +/-pi seam.

    A bearing of exactly +pi must land in the same direction bin as -pi:
    both describe "directly behind" the animal and the bins tile the full
    circle [-pi, pi). Without wrapping, +pi shifts to 2*pi, floors to
    n_direction_bins, and (pre-fix) clipped to the last bin while -pi maps
    to bin 0 -- a spurious discontinuity at the seam.
    """

    def test_plus_pi_and_minus_pi_same_direction_bin(self):
        """Bearing +pi and -pi map to the same direction bin (same flat bin)."""
        from neurospatial.encoding._egocentric_counts import (
            _coords_to_flat_bin_idx,
        )

        distance_range = (0.0, 50.0)
        n_distance_bins = 10
        n_direction_bins = 12

        # Same distance so the distance component is identical; only the
        # bearing differs (+pi vs -pi). Both name "directly behind".
        distances = np.array([25.0, 25.0])
        bearings = np.array([np.pi, -np.pi])

        flat = _coords_to_flat_bin_idx(
            distances,
            bearings,
            distance_range,
            n_distance_bins,
            n_direction_bins,
        )

        assert flat[0] == flat[1], (
            f"+pi mapped to bin {flat[0]} but -pi mapped to bin {flat[1]}; "
            "the +/-pi seam must be a single direction bin."
        )

    def test_plus_pi_not_in_overflow_bin(self):
        """Bearing +pi must not produce a direction bin >= n_direction_bins."""
        from neurospatial.encoding._egocentric_counts import (
            _coords_to_flat_bin_idx,
        )

        n_distance_bins = 10
        n_direction_bins = 12
        flat = _coords_to_flat_bin_idx(
            np.array([25.0]),
            np.array([np.pi]),
            (0.0, 50.0),
            n_distance_bins,
            n_direction_bins,
        )
        # -pi maps to direction bin 0 (distance bin 5 -> flat 5*12 = 60).
        # +pi must map to the same bin.
        angle_bin = flat[0] % n_direction_bins
        assert angle_bin == 0, (
            f"+pi mapped to direction bin {angle_bin}, expected 0 (same as -pi)."
        )


# =============================================================================
# Test NaN handling in egocentric coordinate computation
# =============================================================================


class TestEgocentricNaNHandling:
    """Tests for NaN handling in egocentric nearest-object selection.

    Bug: When using geodesic distance, objects outside the environment have NaN
    distances. np.argmin doesn't handle NaN properly, returning arbitrary indices.

    This test class covers:
    - Objects outside environment with geodesic distance
    - Mixed NaN/finite distance cases (some objects inside, some outside)
    - All-NaN case (all objects outside environment)
    """

    @pytest.fixture
    def env_with_hole(self) -> Environment:
        """Create environment with a hole where objects can be placed outside.

        Creates an L-shaped environment using a regular grid that excludes
        the upper-right quadrant (60-100, 60-100). This ensures predictable
        bin coverage for testing.
        """
        # Create a dense grid covering the L-shaped area
        # Lower-left quadrant (0-60, 0-60) - fully covered
        # Lower-right strip (60-100, 0-60) - covered
        # Upper-left strip (0-60, 60-100) - covered
        # Upper-right quadrant (60-100, 60-100) - HOLE (not covered)
        x_lower = np.linspace(2.5, 57.5, 12)  # Centers for 0-60
        y_lower = np.linspace(2.5, 57.5, 12)
        x_upper = np.linspace(2.5, 57.5, 12)  # x for upper-left strip
        y_upper = np.linspace(62.5, 97.5, 8)
        x_right = np.linspace(62.5, 97.5, 8)  # x for lower-right strip
        y_right = np.linspace(2.5, 57.5, 12)

        # Create grid positions
        positions = []
        # Lower-left quadrant
        for x in x_lower:
            for y in y_lower:
                positions.append([x, y])
        # Upper-left strip
        for x in x_upper:
            for y in y_upper:
                positions.append([x, y])
        # Lower-right strip
        for x in x_right:
            for y in y_right:
                positions.append([x, y])

        positions = np.array(positions)
        return Environment.from_samples(positions, bin_size=5.0)

    def test_object_outside_env_geodesic(self, env_with_hole: Environment):
        """Object outside environment should produce NaN distance with geodesic."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin (inside env), object in upper-right quadrant (outside)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Object outside the L-shaped environment (in the hole)
        object_positions = np.array([[75.0, 75.0]])

        distances, bearings = _compute_object_coords(
            positions,
            headings,
            object_positions,
            metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (object outside env)
        assert np.isnan(distances[0, 0])
        # Bearing should also be NaN (no reachable object to compute bearing to)
        assert np.isnan(bearings[0, 0])

    def test_mixed_nan_finite_distances(self, env_with_hole: Environment):
        """With mixed NaN/finite distances, should select nearest valid object."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin (inside env)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Object 1: outside env (NaN distance)
        # Object 2: inside env (finite distance)
        object_positions = np.array(
            [
                [75.0, 75.0],  # Outside (in hole)
                [30.0, 25.0],  # Inside (5 units away)
            ]
        )

        distances, bearings = _compute_object_coords(
            positions,
            headings,
            object_positions,
            metric="geodesic",
            env=env_with_hole,
        )

        # Should select object 2 (inside env, finite distance ~5)
        # NOT object 1 (outside env, NaN distance)
        assert not np.isnan(distances[0, 0]), "Should return finite distance, not NaN"
        # Distance should be approximately 5 (euclidean, but geodesic might differ slightly)
        assert distances[0, 0] < 20, f"Expected ~5, got {distances[0, 0]}"
        # Bearing should be ~0 (object 2 is directly ahead)
        assert abs(bearings[0, 0]) < 0.5, f"Expected ~0, got {bearings[0, 0]}"

    def test_all_objects_outside_env_geodesic(self, env_with_hole: Environment):
        """When all objects are outside env, should return NaN distance."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal at origin (inside env)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Both objects outside env (in the hole)
        object_positions = np.array(
            [
                [75.0, 75.0],
                [80.0, 80.0],
            ]
        )

        distances, bearings = _compute_object_coords(
            positions,
            headings,
            object_positions,
            metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (all objects outside)
        assert np.isnan(distances[0, 0]), "All objects outside: should return NaN"
        # Bearing should also be NaN (no valid nearest object)
        assert np.isnan(bearings[0, 0]), "All objects outside: bearing should be NaN"

    def test_position_outside_env_geodesic(self, env_with_hole: Environment):
        """When animal position is outside env, should return NaN distance."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Animal in the hole (outside env)
        positions = np.array([[75.0, 75.0]])
        headings = np.array([0.0])
        # Object inside env
        object_positions = np.array([[25.0, 25.0]])

        distances, _bearings = _compute_object_coords(
            positions,
            headings,
            object_positions,
            metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (animal position outside env)
        assert np.isnan(distances[0, 0]), "Position outside env: should return NaN"

    def test_mixed_valid_invalid_positions_geodesic(self, env_with_hole: Environment):
        """With some positions outside env, should handle NaN distances correctly."""
        from neurospatial.encoding._egocentric_counts import (
            _compute_object_coords,
        )

        # Time 0: inside env, Time 1: outside env
        positions = np.array(
            [
                [25.0, 25.0],  # Inside
                [75.0, 75.0],  # Outside (in hole)
            ]
        )
        headings = np.array([0.0, 0.0])
        # Object inside env
        object_positions = np.array([[30.0, 25.0]])

        distances, _bearings = _compute_object_coords(
            positions,
            headings,
            object_positions,
            metric="geodesic",
            env=env_with_hole,
        )

        # Time 0: should have finite distance
        assert not np.isnan(distances[0, 0]), (
            "Position inside: should have finite distance"
        )
        # Time 1: should have NaN distance
        assert np.isnan(distances[1, 0]), "Position outside: should have NaN distance"

    def test_occupancy_with_nan_distances(self, env_with_hole: Environment):
        """Occupancy computation should handle NaN distances gracefully."""
        n_time = 100
        rng = np.random.default_rng(42)
        # Some positions inside, some outside
        positions = np.zeros((n_time, 2))
        positions[:50] = rng.uniform(10, 40, (50, 2))  # Inside
        positions[50:] = rng.uniform(60, 90, (50, 2))  # In hole (outside)
        times = np.linspace(0, 10, n_time)
        headings = np.zeros(n_time)

        # Object in the hole (outside env)
        object_positions = np.array([[75.0, 75.0]])

        # Should not raise an error
        _, occupancy, _env = _bin(
            [],
            times,
            positions,
            headings,
            object_positions,
            **GRID,
            metric="geodesic",
            env=env_with_hole,
        )

        # Occupancy should be all zeros or NaN (object outside env)
        # because we can never compute a valid distance to the object
        assert occupancy.shape == (10 * 12,)

    def test_spike_binning_with_nan_distances(self, env_with_hole: Environment):
        """Spike binning should handle NaN distances gracefully."""
        n_time = 100
        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 40, (n_time, 2))  # All inside env
        times = np.linspace(0, 10, n_time)
        headings = np.zeros(n_time)
        spike_times = np.array([1.0, 3.0, 5.0])

        # Mix of inside/outside objects
        object_positions = np.array(
            [
                [75.0, 75.0],  # Outside (in hole)
                [30.0, 25.0],  # Inside
            ]
        )

        # Should not raise an error
        spike_counts = _counts(
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            **GRID,
            metric="geodesic",
            env=env_with_hole,
        )

        # Should have binned some spikes
        assert spike_counts.shape == (10 * 12,)
        # Some spikes should be binned (those at valid positions near valid object)
        assert np.sum(spike_counts) > 0
