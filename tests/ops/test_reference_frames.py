"""Tests for reference_frames module.

Tests for coordinate transformations between allocentric and egocentric
reference frames, and egocentric polar environment creation. Heading
computation is tested in ``tests/behavior/test_heading.py``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestModuleSetup:
    """Test module imports and structure."""


class TestAllocentricToEgocentric:
    """Tests for allocentric_to_egocentric batch function."""

    def test_batch_transform_multiple_timepoints(self):
        """Batch transform handles multiple timepoints correctly."""
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        # 3 timepoints, 2 landmarks
        landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # Shape: (2, 2)
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])  # Shape: (3, 2)
        headings = np.array([0.0, np.pi / 2, np.pi])  # Shape: (3,)

        ego = allocentric_to_egocentric(positions, headings, landmarks)

        # Output shape: (n_time, n_points, 2) = (3, 2, 2)
        assert ego.shape == (3, 2, 2)

        # At t=0: animal at origin, facing East (heading=0)
        # Landmark (10, 0) is 10 units ahead
        assert_allclose(ego[0, 0], [10.0, 0.0], atol=1e-10)
        # Landmark (0, 10) is 10 units left
        assert_allclose(ego[0, 1], [0.0, 10.0], atol=1e-10)

    def test_broadcasting_2d_points_to_3d(self):
        """2D point input broadcasts to 3D output."""
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        # Same landmarks transformed at each timepoint
        landmarks = np.array([[10.0, 0.0]])  # Shape: (1, 2)
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Shape: (2, 2)
        headings = np.array([0.0, np.pi / 2])  # Shape: (2,)

        ego = allocentric_to_egocentric(positions, headings, landmarks)

        # Output: (n_time, n_points, 2) = (2, 1, 2)
        assert ego.shape == (2, 1, 2)

        # t=0, heading=0: (10, 0) is ahead
        assert_allclose(ego[0, 0], [10.0, 0.0], atol=1e-10)

        # t=1, heading=π/2: (10, 0) is right
        assert_allclose(ego[1, 0], [0.0, -10.0], atol=1e-10)

    def test_shape_validation_error_messages(self):
        """Invalid shapes produce clear error messages."""
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        points = np.array([10.0, 0.0])  # Wrong: 1D instead of 2D
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="points"):
            allocentric_to_egocentric(positions, headings, points)

    def test_positions_headings_length_mismatch(self):
        """Positions and headings must have matching length."""
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        points = np.array([[10.0, 0.0]])
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 timepoints
        headings = np.array([0.0])  # 1 timepoint

        with pytest.raises(ValueError, match=r"Headings/positions length mismatch"):
            allocentric_to_egocentric(positions, headings, points)

    @pytest.mark.parametrize("n_rows", [1, 3])
    def test_time_varying_points_must_have_one_row_per_time(self, n_rows):
        """A (1, n_points, 2) array is not silently reused at every time."""
        from neurospatial.ops.egocentric import (
            allocentric_to_egocentric,
            compute_egocentric_bearing,
        )

        positions = np.zeros((2, 2))
        headings = np.zeros(2)
        points = np.ones((n_rows, 4, 2))
        for transform in (allocentric_to_egocentric, compute_egocentric_bearing):
            with pytest.raises(
                ValueError,
                match=rf"points time axis {n_rows} does not match positions length 2",
            ):
                transform(positions, headings, points)

    @pytest.mark.parametrize("shape", [(4, 3), (2, 4, 3)])
    def test_points_need_x_and_y(self, shape):
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        with pytest.raises(ValueError, match=r"points must have 2 columns \(x, y\)"):
            allocentric_to_egocentric(np.zeros((2, 2)), np.zeros(2), np.ones(shape))


class TestEgocentricToAllocentric:
    """Tests for egocentric_to_allocentric batch function."""

    def test_inverse_of_allocentric_to_egocentric(self):
        """egocentric_to_allocentric is the inverse of allocentric_to_egocentric."""
        from neurospatial.ops.egocentric import (
            allocentric_to_egocentric,
            egocentric_to_allocentric,
        )

        rng = np.random.default_rng(42)

        points = rng.uniform(-50, 50, size=(5, 2))
        positions = rng.uniform(-10, 10, size=(10, 2))
        headings = rng.uniform(-np.pi, np.pi, size=10)

        ego = allocentric_to_egocentric(positions, headings, points)
        recovered = egocentric_to_allocentric(positions, headings, ego)

        # Output should broadcast back to match input shape
        assert recovered.shape == (10, 5, 2)

        # Each recovered point should match original
        for t in range(10):
            assert_allclose(recovered[t], points, atol=1e-10)

    def test_invalid_ego_points_shape_raises(self):
        """A 2D ego_points array (missing time axis) raises a clear ValueError."""
        from neurospatial.ops.egocentric import egocentric_to_allocentric

        # ego_points must be 3D (n_time, n_points, 2). Pass a 2D array.
        ego_points = np.array([[10.0, 0.0], [0.0, 10.0]])  # (2, 2): ambiguous
        positions = np.array([[0.0, 0.0]])  # 1 timepoint
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="ego_points"):
            egocentric_to_allocentric(positions, headings, ego_points)

    def test_invalid_positions_shape_raises(self):
        """Positions not shaped (n_time, 2) raise a clear ValueError."""
        from neurospatial.ops.egocentric import egocentric_to_allocentric

        ego_points = np.zeros((1, 2, 2))
        positions = np.array([0.0, 0.0])  # 1D, wrong
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="positions"):
            egocentric_to_allocentric(positions, headings, ego_points)

    def test_positions_headings_length_mismatch_raises(self):
        """Headings and positions of mismatched length raise a clear ValueError."""
        from neurospatial.ops.egocentric import egocentric_to_allocentric

        ego_points = np.zeros((2, 1, 2))
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 timepoints
        headings = np.array([0.0])  # 1 timepoint

        with pytest.raises(ValueError, match=r"Headings/positions length mismatch"):
            egocentric_to_allocentric(positions, headings, ego_points)


class TestComputeEgocentricBearing:
    """Tests for compute_egocentric_bearing function."""

    def test_bearing_zero_when_target_ahead(self):
        """Bearing is 0 when target is directly ahead."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # Target at (10, 0), animal at origin facing East
        target = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(position, heading, target)
        assert_allclose(bearing, [[0.0]], atol=1e-10)

    def test_bearing_pi_half_when_target_left(self):
        """Bearing is π/2 when target is to the left."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # Target at (0, 10), animal at origin facing East
        # Target is 90 degrees left
        target = np.array([[0.0, 10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(position, heading, target)
        assert_allclose(bearing, [[np.pi / 2]], atol=1e-10)

    def test_bearing_negative_pi_half_when_target_right(self):
        """Bearing is -π/2 when target is to the right."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # Target at (0, -10), animal at origin facing East
        target = np.array([[0.0, -10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(position, heading, target)
        assert_allclose(bearing, [[-np.pi / 2]], atol=1e-10)

    def test_bearing_behind_target(self):
        """Bearing is ±π when target is behind."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # Target at (-10, 0), animal at origin facing East
        target = np.array([[-10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(position, heading, target)
        # Should be π or -π (both valid)
        assert np.abs(np.abs(bearing[0, 0]) - np.pi) < 1e-10

    def test_angle_wrapping_near_pi(self):
        """Angles wrap correctly near ±π boundary."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # Test points slightly behind, should still give angles near ±π
        target1 = np.array([[-10.0, 0.1]])  # Slightly to the left of behind
        target2 = np.array([[-10.0, -0.1]])  # Slightly to the right of behind
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing1 = compute_egocentric_bearing(position, heading, target1)
        bearing2 = compute_egocentric_bearing(position, heading, target2)

        # Both should be close to π in absolute value
        assert np.abs(bearing1[0, 0]) > 0.99 * np.pi
        assert np.abs(bearing2[0, 0]) > 0.99 * np.pi

        # bearing1 (slightly left) should be positive, bearing2 negative
        assert bearing1[0, 0] > 0  # Slightly left of behind
        assert bearing2[0, 0] < 0  # Slightly right of behind

    def test_batch_multiple_targets_and_times(self):
        """Handles multiple targets and timepoints."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        # 3 targets, 5 timepoints
        targets = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
        positions = np.zeros((5, 2))
        headings = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2])

        bearing = compute_egocentric_bearing(positions, headings, targets)

        assert bearing.shape == (5, 3)


class TestComputeEgocentricDistance:
    """Tests for compute_egocentric_distance function."""

    def test_euclidean_distance(self):
        """Euclidean distance computed correctly."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        targets = np.array([[10.0, 0.0], [0.0, 10.0]])
        position = np.array([[0.0, 0.0]])

        distances = compute_egocentric_distance(
            position, None, targets, metric="euclidean"
        )

        assert distances.shape == (1, 2)
        assert_allclose(distances, [[10.0, 10.0]], atol=1e-10)

    def test_geodesic_distance_with_environment(self):
        """Geodesic distance uses environment's distance_field."""
        from neurospatial import Environment
        from neurospatial.ops.egocentric import compute_egocentric_distance

        # Create a simple 2D grid environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Targets and position within the environment
        targets = np.array([[50.0, 50.0], [80.0, 80.0]])
        position = np.array([[10.0, 10.0]])

        distances = compute_egocentric_distance(
            position, None, targets, metric="geodesic", env=env
        )

        assert distances.shape == (1, 2)
        # Geodesic distances should be finite and positive
        assert np.all(np.isfinite(distances))
        assert np.all(distances > 0)
        # Geodesic distances should be within reasonable range of Euclidean
        # (can be slightly different due to graph discretization)
        eucl = np.linalg.norm(targets - position, axis=1)
        # Allow for discretization differences (within 20% or 10 units)
        assert np.all(np.abs(distances[0] - eucl) < np.maximum(0.2 * eucl, 10))

    def test_invalid_metric_raises(self):
        """Invalid metric raises ValueError."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="metric"):
            compute_egocentric_distance(position, None, targets, metric="invalid")

    def test_geodesic_without_env_raises(self):
        """Geodesic metric without environment raises error."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="env"):
            compute_egocentric_distance(position, None, targets, metric="geodesic")


class TestGeodesicDistanceTimeVarying:
    """Geodesic distance honors time-varying targets (shape (n_time, n_targets, 2))."""

    def _build_quasi_1d_env(self):
        """Build a 2D environment that behaves like a 1D linear track along x."""
        from neurospatial import Environment

        rng = np.random.default_rng(0)
        # Dense sampling along a thin strip y in [-0.4, 0.4] so the env is
        # effectively 1D after binning at bin_size=1.0 (one or two y-bins).
        x = np.linspace(0.0, 100.0, 3000)
        y = rng.uniform(-0.4, 0.4, size=x.size)
        return Environment.from_samples(np.column_stack([x, y]), bin_size=1.0)

    def test_moving_target_returns_per_timestep_distance(self):
        """Each timestep's target is honored individually, not just t=0."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        env = self._build_quasi_1d_env()

        # Animal stationary at (50, 0) for 3 timesteps.
        positions = np.tile(np.array([[50.0, 0.0]]), (3, 1))
        # Target moves: (10, 0) -> (50, 0) -> (90, 0).
        # Shape (n_time=3, n_targets=1, 2) triggers the time-varying path.
        targets = np.array(
            [
                [[10.0, 0.0]],
                [[50.0, 0.0]],
                [[90.0, 0.0]],
            ],
            dtype=np.float64,
        )

        distances = compute_egocentric_distance(
            positions, None, targets, metric="geodesic", env=env
        )

        assert distances.shape == (3, 1)
        # Graph distance is approximately the linear x-distance along the strip.
        # Tolerance: 1.5 * bin_size (1.0) accounts for binning discretization.
        bin_size = 1.0
        assert distances[0, 0] == pytest.approx(40.0, abs=1.5 * bin_size)
        assert distances[1, 0] == pytest.approx(0.0, abs=1.5 * bin_size)
        assert distances[2, 0] == pytest.approx(40.0, abs=1.5 * bin_size)

    def test_static_target_via_tile_matches_2d_broadcast(self):
        """Passing a static target explicitly tiled to 3D matches the 2D-broadcast path."""
        from neurospatial import Environment
        from neurospatial.ops.egocentric import compute_egocentric_distance

        rng = np.random.default_rng(1)
        sample_positions = rng.uniform(0, 100, size=(2000, 2))
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Animal moving along a diagonal trajectory.
        positions = np.column_stack(
            [np.linspace(10.0, 80.0, 6), np.linspace(20.0, 70.0, 6)]
        )
        static_targets_2d = np.array([[50.0, 50.0], [70.0, 30.0]])  # (n_targets=2, 2)

        # Reference run: 2D targets, library broadcasts internally.
        dist_2d = compute_egocentric_distance(
            positions, None, static_targets_2d, metric="geodesic", env=env
        )

        # Time-varying form with identical targets at every timestep.
        n_time = positions.shape[0]
        static_targets_3d = np.broadcast_to(
            static_targets_2d, (n_time, 2, 2)
        ).copy()  # copy so the contiguous-array assumption is exercised
        dist_3d = compute_egocentric_distance(
            positions, None, static_targets_3d, metric="geodesic", env=env
        )

        assert_allclose(dist_3d, dist_2d, equal_nan=True)

    @staticmethod
    def _reference_geodesic(positions, targets, env):
        """Reference geodesic distance: original per-cell bin_at loop.

        Mirrors the pre-optimization implementation so the hoisted version can
        be verified for exact equivalence (including NaN handling for
        out-of-environment positions and targets).
        """
        from neurospatial.ops.distance import distance_field as compute_distance_field

        n_time = len(positions)
        targets = np.asarray(targets, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        if targets.ndim == 2:
            targets_3d = np.broadcast_to(targets, (n_time, targets.shape[0], 2))
        else:
            targets_3d = targets
        n_targets = targets_3d.shape[1]

        distances = np.full((n_time, n_targets), np.nan, dtype=np.float64)
        cache: dict[int, np.ndarray] = {}
        for t in range(n_time):
            pos_bin = int(env.bin_at(positions[t].reshape(1, -1))[0])
            if pos_bin < 0:
                continue
            for i in range(n_targets):
                target_bin = int(env.bin_at(targets_3d[t, i].reshape(1, -1))[0])
                if target_bin < 0:
                    continue
                if target_bin not in cache:
                    cache[target_bin] = compute_distance_field(
                        env.connectivity, sources=[target_bin]
                    )
                distances[t, i] = cache[target_bin][pos_bin]
        return distances

    def test_geodesic_matches_reference_loop_with_out_of_env(self):
        """Batched geodesic distance equals the reference loop, incl. NaN cells."""
        from neurospatial import Environment
        from neurospatial.ops.egocentric import compute_egocentric_distance

        rng = np.random.default_rng(7)
        sample_positions = rng.uniform(0, 100, size=(2000, 2))
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Some positions inside, one far outside the environment (-> NaN row).
        positions = np.array(
            [
                [20.0, 30.0],
                [60.0, 70.0],
                [10000.0, 10000.0],  # outside env -> NaN row
                [40.0, 40.0],
            ]
        )
        # Time-varying targets, one missing at t=1 (-> NaN cell).
        targets = np.array(
            [
                [[50.0, 50.0], [70.0, 30.0]],
                [[50.0, 50.0], [np.nan, np.nan]],  # second target missing
                [[50.0, 50.0], [70.0, 30.0]],
                [[30.0, 30.0], [80.0, 80.0]],
            ],
            dtype=np.float64,
        )

        expected = self._reference_geodesic(positions, targets, env)
        actual = compute_egocentric_distance(
            positions, None, targets, metric="geodesic", env=env
        )

        assert_allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize("time_varying", [False, True])
    def test_target_outside_every_bin_is_refused(self, time_varying):
        """A target outside the environment has no geodesic path: refuse it.

        Leaving the whole target column NaN would look like a missing target.
        """
        from neurospatial.ops.egocentric import compute_egocentric_distance

        env = self._build_quasi_1d_env()
        positions = np.array([[20.0, 0.0], [30.0, 0.0]])
        targets = np.array([[50.0, 0.0], [50.0, 30.0]])  # second is 30 off-strip
        if time_varying:
            targets = np.stack([targets, targets])
        where = "at time index 0 " if time_varying else ""
        with pytest.raises(
            ValueError,
            match=rf"^Target 1 {where}at \(50, 30\) lies outside every bin(?s:.*)"
            r"nearest bin center is 29\.\d away(?s:.*)metric='euclidean'",
        ):
            compute_egocentric_distance(
                positions, None, targets, metric="geodesic", env=env
            )

    def test_position_outside_is_missing_not_refused(self):
        """An animal outside the environment gives a NaN row, as a missing sample."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        env = self._build_quasi_1d_env()
        positions = np.array([[20.0, 0.0], [20.0, 30.0], [np.nan, np.nan]])
        targets = np.array([[50.0, 0.0]])
        distances = compute_egocentric_distance(
            positions, None, targets, metric="geodesic", env=env
        )
        assert np.isfinite(distances[0, 0])
        assert np.isnan(distances[1:, 0]).all()


class TestComputeEgocentricDistanceFunctions:
    """Tests for distance utility functions."""

    def test_euclidean_multiple_timepoints(self):
        """Euclidean distance across multiple timepoints."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        # Single target, multiple animal positions
        targets = np.array([[50.0, 50.0]])
        positions = np.array([[0.0, 0.0], [25.0, 25.0], [50.0, 50.0]])

        distances = compute_egocentric_distance(
            positions, None, targets, metric="euclidean"
        )

        assert distances.shape == (3, 1)
        expected = [
            np.sqrt(50**2 + 50**2),  # Distance from origin
            np.sqrt(25**2 + 25**2),  # Distance from (25, 25)
            0.0,  # At the target
        ]
        assert_allclose(distances[:, 0], expected, atol=1e-10)


class TestAllocentricToEgocentricSignConvention:
    """Pin the rotation-matrix sign convention end-to-end.

    Animal at the origin. For heading ``h`` and allocentric target ``(tx, ty)``
    the egocentric coordinate is the rotation by ``-h``::

        ego_x = cos(h) * tx + sin(h) * ty
        ego_y = -sin(h) * tx + cos(h) * ty
    """

    @pytest.mark.parametrize(
        "heading",
        [0.0, np.pi / 2, np.pi, -np.pi / 2],
        ids=["heading_0", "heading_pi_2", "heading_pi", "heading_neg_pi_2"],
    )
    @pytest.mark.parametrize(
        "target",
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        ids=["east", "north", "west", "south"],
    )
    def test_sign_convention(self, heading, target):
        """Each cardinal heading x cardinal target matches the closed form."""
        from neurospatial.ops.egocentric import allocentric_to_egocentric

        positions = np.array([[0.0, 0.0]])
        headings = np.array([heading])
        targets = np.array([target])

        ego = allocentric_to_egocentric(positions, headings, targets)

        tx, ty = target
        expected_x = np.cos(heading) * tx + np.sin(heading) * ty
        expected_y = -np.sin(heading) * tx + np.cos(heading) * ty

        assert ego.shape == (1, 1, 2)
        assert_allclose(ego[0, 0], [expected_x, expected_y], atol=1e-10)


class TestEgocentricDistance3DTargets:
    """compute_egocentric_distance must validate a 3D targets' time axis."""

    def test_egocentric_distance_3d_targets_time_mismatch_raises(self):
        """A 3D targets array with the wrong time axis raises a clear ValueError."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        n_time = 5
        positions = np.column_stack([np.arange(n_time, dtype=float), np.zeros(n_time)])
        # Wrong leading axis: n_time + 1.
        targets = np.zeros((n_time + 1, 2, 2), dtype=float)

        with pytest.raises(ValueError, match="time axis"):
            compute_egocentric_distance(positions, None, targets)

    def test_egocentric_distance_3d_targets_valid_roundtrip(self):
        """A correctly-shaped (n_time, k, 2) targets array returns (n_time, k)."""
        from neurospatial.ops.egocentric import compute_egocentric_distance

        n_time, k = 4, 3
        positions = np.column_stack([np.arange(n_time, dtype=float), np.zeros(n_time)])
        rng = np.random.default_rng(0)
        targets = rng.normal(size=(n_time, k, 2))

        distances = compute_egocentric_distance(positions, None, targets)

        assert distances.shape == (n_time, k)
        expected = np.sqrt(
            np.sum((targets - positions[:, np.newaxis, :]) ** 2, axis=-1)
        )
        assert_allclose(distances, expected)


class TestWrapAngleAntipode:
    """_wrap_bearing keeps the antipode at +pi per the documented (-pi, pi]."""

    def test_wrap_angle_antipode_is_plus_pi(self):
        """+/- pi and odd multiples wrap to +pi; interior values are unchanged."""
        from neurospatial.ops.egocentric import _wrap_bearing

        antipodes = _wrap_bearing(np.array([np.pi, -np.pi, 3 * np.pi, -3 * np.pi]))
        assert_allclose(antipodes, np.pi)

        interior = np.array([0.3, -1.2, 0.0, 2.0])
        assert_allclose(_wrap_bearing(interior), interior)

    def test_egocentric_bearing_directly_behind_is_plus_pi(self):
        """A target directly behind the animal returns bearing +pi, not -pi."""
        from neurospatial.ops.egocentric import compute_egocentric_bearing

        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])  # facing East
        target = np.array([[-10.0, 0.0]])  # directly behind (West)

        bearing = compute_egocentric_bearing(position, heading, target)
        assert_allclose(bearing, [[np.pi]])
