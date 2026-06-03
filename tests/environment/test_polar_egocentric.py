"""Tests for from_polar_egocentric() factory method.

This module tests the egocentric polar environment factory method which creates
environments in polar coordinates (distance, angle) centered on the animal.
"""

import numpy as np
import pytest

from neurospatial import Environment


class TestFromPolarEgocentricBasic:
    """Test basic creation and properties of egocentric polar environments."""

    def test_creates_environment_with_expected_n_bins(self):
        """Test that n_bins matches expected n_distance * n_angle."""
        # Create polar environment
        distance_range = (0.0, 100.0)  # 0-100 cm
        angle_range = (-np.pi, np.pi)  # Full circle
        distance_bin_size = 10.0  # 10 bins
        angle_bin_size = np.pi / 4  # 8 bins (45 degrees each)

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        # Expected: 10 distance bins * 8 angle bins = 80 bins
        expected_n_distance = 10
        expected_n_angle = 8
        expected_n_bins = expected_n_distance * expected_n_angle

        assert env.n_bins == expected_n_bins

    def test_bin_centers_have_correct_dimensions(self):
        """Test that bin_centers has shape (n_bins, 2) with distance and angle."""
        distance_range = (0.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        assert env.bin_centers.shape[1] == 2  # 2D: (distance, angle)
        assert env.n_dims == 2

    def test_bin_centers_distances_in_range(self):
        """Test that bin_centers[:, 0] (distances) are within distance_range."""
        distance_range = (5.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        distances = env.bin_centers[:, 0]
        # Centers should be within range (with some tolerance for bin center calculation)
        assert np.all(distances >= distance_range[0])
        assert np.all(distances <= distance_range[1])

    def test_bin_centers_angles_in_range(self):
        """Test that bin_centers[:, 1] (angles) are within angle_range."""
        distance_range = (0.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        angles = env.bin_centers[:, 1]
        # Centers should be within range
        assert np.all(angles >= angle_range[0])
        assert np.all(angles <= angle_range[1])


class TestCircularConnectivity:
    """Test circular connectivity wrapping for angle dimension."""

    def test_circular_angle_true_wraps_first_last_angle_bins(self):
        """Test that circular_angle=True connects first and last angle bins."""
        distance_range = (0.0, 10.0)  # 1 distance bin
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 1 bin
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # With 1 distance bin and 4 angle bins, bin_centers should be:
        # bin 0: (5, -3π/4), bin 1: (5, -π/4), bin 2: (5, π/4), bin 3: (5, 3π/4)
        # With circular connectivity, bin 0 and bin 3 should be connected

        connectivity = env.connectivity
        # The first and last angle bins at the same distance should be connected
        # Find bins at the first angle (-3π/4) and last angle (3π/4)
        angles = env.bin_centers[:, 1]
        first_angle_bin = np.argmin(angles)
        last_angle_bin = np.argmax(angles)

        # They should be neighbors
        assert connectivity.has_edge(first_angle_bin, last_angle_bin)

    def test_circular_angle_false_does_not_wrap(self):
        """Test that circular_angle=False does not connect first and last angle bins."""
        distance_range = (0.0, 10.0)  # 1 distance bin
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 1 bin
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=False,
        )

        connectivity = env.connectivity
        angles = env.bin_centers[:, 1]
        first_angle_bin = np.argmin(angles)
        last_angle_bin = np.argmax(angles)

        # They should NOT be neighbors
        assert not connectivity.has_edge(first_angle_bin, last_angle_bin)

    def test_circular_wrapping_at_multiple_distances(self):
        """Test circular wrapping works correctly at each distance ring."""
        distance_range = (0.0, 20.0)  # 2 distance bins
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 2 bins
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Group bins by distance
        distances = env.bin_centers[:, 0]
        unique_distances = np.unique(distances)

        for d in unique_distances:
            # Get bins at this distance
            mask = np.isclose(distances, d)
            bin_indices = np.where(mask)[0]
            angles_at_d = env.bin_centers[bin_indices, 1]

            # Find first and last angle bins at this distance
            first_idx = bin_indices[np.argmin(angles_at_d)]
            last_idx = bin_indices[np.argmax(angles_at_d)]

            # They should be connected
            assert env.connectivity.has_edge(first_idx, last_idx)

    def test_circular_edges_have_required_attributes(self):
        """Test that circular connectivity edges have all required attributes."""
        distance_range = (0.0, 20.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Find a circular edge (connects first and last angle at some distance)
        distances = env.bin_centers[:, 0]

        # Get bins at first distance ring
        d_val = distances[0]
        mask = np.isclose(distances, d_val)
        bin_indices = np.where(mask)[0]
        angles_at_d = env.bin_centers[bin_indices, 1]

        first_idx = bin_indices[np.argmin(angles_at_d)]
        last_idx = bin_indices[np.argmax(angles_at_d)]

        # Verify edge exists
        assert env.connectivity.has_edge(first_idx, last_idx)

        # Verify all required attributes
        edge_data = env.connectivity[first_idx][last_idx]
        assert "distance" in edge_data
        assert "vector" in edge_data
        assert "angle_2d" in edge_data
        assert "edge_id" in edge_data

        # Verify types
        assert isinstance(edge_data["distance"], float)
        assert isinstance(edge_data["vector"], list)
        assert isinstance(edge_data["angle_2d"], float)
        assert isinstance(edge_data["edge_id"], int)

    def test_circular_angle_with_single_angle_bin(self):
        """Test that circular_angle=True with n_angle=1 doesn't create self-loops."""
        distance_range = (0.0, 20.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = 2 * np.pi  # Single angle bin spanning full circle

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Should have no self-loops
        for node in env.connectivity.nodes():
            assert not env.connectivity.has_edge(node, node)


class TestCircularWrapEdgeArcLength:
    """The angular wrap edge must carry arc length, not chord length.

    Regression: the wrap edge connects the first and last angle bins on a
    single distance ring. Both bins sit at the same radius, so the Euclidean
    norm between their centers is the chord across the *full* angular span --
    not the one-``angle_bin_size`` step the seam actually represents. The
    geodesic ``distance`` weight must instead be ``ring_radius *
    angle_bin_size`` (true arc length).
    """

    @staticmethod
    def _wrap_edge(env):
        """Return (ring_radius, first_pos, last_pos, edge_data) for innermost ring."""
        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        ring_radius = distances.min()
        ring = np.isclose(distances, ring_radius)
        ring_idx = np.flatnonzero(ring)
        first_idx = int(ring_idx[np.argmin(angles[ring_idx])])
        last_idx = int(ring_idx[np.argmax(angles[ring_idx])])
        first_pos = np.array(env.connectivity.nodes[first_idx]["pos"])
        last_pos = np.array(env.connectivity.nodes[last_idx]["pos"])
        return ring_radius, first_pos, last_pos, env.connectivity[first_idx][last_idx]

    def test_wrap_edge_distance_is_arc_length_not_chord(self):
        angle_bin_size = np.pi / 2
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        ring_radius, first_pos, last_pos, edge_data = self._wrap_edge(env)
        expected_arc = ring_radius * angle_bin_size

        assert edge_data["distance"] == pytest.approx(expected_arc)

        # Guard against a regression to the old behavior: the previous code set
        # ``distance = norm(pos_last - pos_first)`` over the (distance, angle)
        # node positions. Because both bins lie on the same distance ring, that
        # norm collapses to the angular span across the *full* circle minus one
        # step, an entirely different quantity from the one-step arc.
        old_buggy_value = float(np.linalg.norm(last_pos - first_pos))
        assert edge_data["distance"] != pytest.approx(old_buggy_value)

    def test_wrap_edge_arc_length_scales_with_ring_radius(self):
        """Outer rings have longer wrap arcs (arc = radius * angle_bin_size)."""
        angle_bin_size = np.pi / 3
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,  # 3 distance rings
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        observed = {}
        for ring_radius in np.unique(distances):
            ring_idx = np.flatnonzero(np.isclose(distances, ring_radius))
            first_idx = int(ring_idx[np.argmin(angles[ring_idx])])
            last_idx = int(ring_idx[np.argmax(angles[ring_idx])])
            observed[ring_radius] = env.connectivity[first_idx][last_idx]["distance"]

        for ring_radius, dist in observed.items():
            assert dist == pytest.approx(ring_radius * angle_bin_size)


class TestPolarStateVersionFreshness:
    """Post-construction mutations must invalidate versioned caches.

    ``EgocentricPolarEnvironment.create`` fixes polar edge geometry and
    (optionally) adds circular wrap edges *after* ``_setup_from_layout`` finalized
    ``_state_version``. Without a final version bump, any
    ``versioned_cached_property`` cached at the construction version would go
    stale relative to the mutated connectivity graph.
    """

    def test_construction_bumps_version_past_layout_setup(self):
        """The polar factory bumps the version beyond a plain grid build.

        A bare ``from_grid_mask`` env of the same geometry performs no
        post-construction mutation, so its ``_state_version`` reflects only
        ``_setup_from_layout``. The polar factory mutates after setup and must
        bump again -- so its version must be strictly greater.
        """
        grid_mask = np.ones((1, 4), dtype=bool)
        grid_edges = (np.linspace(0.0, 10.0, 2), np.linspace(-np.pi, np.pi, 5))
        grid_env = Environment.from_grid_mask(
            active_mask=grid_mask,
            grid_edges=grid_edges,
            connect_diagonal_neighbors=True,
        )

        polar_env = Environment.from_polar_egocentric(
            distance_range=(0.0, 10.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=True,
        )

        assert polar_env._state_version > grid_env._state_version

    def test_differential_operator_reflects_circular_edges(self):
        """A versioned cached property sees the wrap edges, not a stale graph.

        ``get_differential_operator`` is a ``versioned_cached_property`` of
        shape ``(n_bins, n_edges)``. Its edge (column) dimension must match the
        connectivity graph's edge count including the circular wrap edges.
        """
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 10.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=True,
        )

        n_edges = env.connectivity.number_of_edges()
        # The wrap edge is present in the graph...
        angles = env.bin_centers[:, 1]
        first_idx = int(np.argmin(angles))
        last_idx = int(np.argmax(angles))
        assert env.connectivity.has_edge(first_idx, last_idx)

        # ...and the versioned cached differential operator reflects it. The
        # operator has shape (n_bins, n_edges).
        D = env.get_differential_operator()
        assert D.shape == (env.n_bins, n_edges)
        # The edge-attribute cache (also versioned) agrees, too.
        assert len(env.get_edge_attributes()) == n_edges


class TestConnectDiagonalNeighborsParameter:
    """from_polar_egocentric exposes connect_diagonal_neighbors (default True)."""

    def test_default_connects_diagonals(self):
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        env_explicit = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            connect_diagonal_neighbors=True,
        )
        assert (
            env.connectivity.number_of_edges()
            == env_explicit.connectivity.number_of_edges()
        )

    def test_disabling_diagonals_reduces_edge_count(self):
        """An 8-connected grid has more edges than a 4-connected one."""
        diag = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=False,
            connect_diagonal_neighbors=True,
        )
        no_diag = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=False,
            connect_diagonal_neighbors=False,
        )
        assert (
            no_diag.connectivity.number_of_edges() < diag.connectivity.number_of_edges()
        )


class TestParameterValidation:
    """Test parameter validation for from_polar_egocentric()."""

    def test_raises_on_negative_distance_bin_size(self):
        """Test that negative distance_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=-10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_negative_angle_bin_size(self):
        """Test that negative angle_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=-np.pi / 4,
            )

    def test_raises_on_zero_distance_bin_size(self):
        """Test that zero distance_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=0.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_invalid_distance_range(self):
        """Test that distance_range[0] >= distance_range[1] raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(100.0, 0.0),  # Inverted range
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_invalid_angle_range(self):
        """Test that angle_range[0] >= angle_range[1] raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(np.pi, -np.pi),  # Inverted range
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_equal_distance_range(self):
        """Test that equal distance_range raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(100.0, 100.0),  # Equal bounds
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )


class TestEnvironmentMetadata:
    """Test that environment metadata is set correctly."""

    def test_name_parameter(self):
        """Test that name parameter is set correctly."""
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 100.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
            name="test_polar_env",
        )

        assert env.name == "test_polar_env"


class TestDocstringNote:
    """Test that environment has correct coordinate convention documentation."""


class TestPolarCoordinateKindFlag:
    """Regression: polar envs are a distinct type from Cartesian.

    Downstream code that computes Euclidean distance on `bin_centers`
    would silently treat `(distance, angle)` pairs as `(x, y)` and produce
    meaningless numbers. Polar environments are now a separate type
    (``EgocentricPolarEnvironment``, a sibling of ``Environment``, not a
    subclass) so the geometry is carried by the type itself and the
    Cartesian-only methods are simply unavailable.
    """

    @pytest.fixture
    def polar_env(self):
        """A small polar env: 5 distance bins, 4 angle bins."""
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    @pytest.fixture
    def cartesian_env(self):
        """A small Cartesian env for the negative-control side of every test."""
        positions = np.column_stack([np.linspace(0, 30, 16), np.linspace(0, 30, 16)])
        return Environment.from_samples(positions, bin_size=2.0)

    def test_from_polar_egocentric_returns_polar_type(self, polar_env):
        """from_polar_egocentric returns the distinct EgocentricPolarEnvironment."""
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        assert isinstance(polar_env, EgocentricPolarEnvironment)
        assert not isinstance(polar_env, Environment)
        assert polar_env._POLAR is True

    def test_cartesian_factories_default_to_cartesian(self, cartesian_env):
        """from_samples / from_polygon / from_grid_mask / etc. produce Cartesian envs."""
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        assert isinstance(cartesian_env, Environment)
        assert not isinstance(cartesian_env, EgocentricPolarEnvironment)
        assert cartesian_env._POLAR is False


class TestPolarRaisesOnCartesianAssumingMethods:
    """Methods that interpret inputs as (x, y) must reject polar envs.

    bin_at, distance_between, and distance_to(metric='euclidean') all
    compute Euclidean operations on bin_centers; running them against a
    polar env's (distance, angle) bin centers produces silently wrong
    numbers. They should fail at the boundary instead.

    Methods that operate purely on the connectivity graph (path_between,
    reachable_from, neighbors, distance_to(metric='geodesic')) are safe
    on polar envs and must keep working.
    """

    @pytest.fixture
    def polar_env(self):
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    def test_bin_at_raises_on_polar_env(self, polar_env):
        from_xy = np.array([[5.0, 5.0]])
        with pytest.raises(NotImplementedError, match=r"polar.*bin_at|bin_at.*polar"):
            polar_env.bin_at(from_xy)

    def test_distance_between_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError,
            match=r"polar.*distance_between|distance_between.*polar",
        ):
            polar_env.distance_between(np.array([0.0, 0.0]), np.array([10.0, 10.0]))

    def test_distance_to_euclidean_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError,
            match=r"polar.*distance_to.*euclidean|distance_to.*euclidean.*polar",
        ):
            polar_env.distance_to([0], metric="euclidean")

    def test_distance_to_geodesic_works_on_polar_env(self, polar_env):
        """Geodesic distance respects connectivity, so it's well-defined on polar."""
        distances = polar_env.distance_to([0], metric="geodesic")
        assert distances.shape == (polar_env.n_bins,)
        # Distance from a bin to itself is zero; everything else is positive.
        assert distances[0] == 0.0
        assert (distances[1:] >= 0).all()

    def test_neighbors_works_on_polar_env(self, polar_env):
        """neighbors only reads the connectivity graph; should not raise."""
        nbrs = polar_env.neighbors(0)
        assert isinstance(nbrs, list)

    # --- Inherited Cartesian-coordinate / Cartesian-grid ops must also raise.

    def test_interpolate_raises_on_polar_env(self, polar_env):
        """interpolate bypasses bin_at and previously returned a silent array.

        Fail-before: prior to the fix this returned geometric nonsense
        (an ndarray) instead of raising on (distance, angle) coordinates.
        """
        field = np.arange(polar_env.n_bins, dtype=float)
        points = np.array([[5.0, 0.0], [25.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*interpolate|interpolate.*polar"
        ):
            polar_env.interpolate(field, points, mode="nearest")

    def test_occupancy_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*occupancy|occupancy.*polar"
        ):
            polar_env.occupancy(times, positions)

    def test_bin_sequence_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*bin_sequence|bin_sequence.*polar"
        ):
            polar_env.bin_sequence(times, positions)

    def test_bin_sequence_with_runs_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError,
            match=r"polar.*bin_sequence_with_runs|bin_sequence_with_runs.*polar",
        ):
            polar_env.bin_sequence_with_runs(times, positions)

    def test_to_linear_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError, match=r"polar.*to_linear|to_linear.*polar"
        ):
            polar_env.to_linear(np.array([[5.0, 0.0]]))

    def test_linear_to_nd_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError, match=r"polar.*linear_to_nd|linear_to_nd.*polar"
        ):
            polar_env.linear_to_nd(np.array([5.0]))

    def test_rebin_raises_on_polar_env(self, polar_env):
        with pytest.raises(NotImplementedError, match=r"polar.*rebin|rebin.*polar"):
            polar_env.rebin(2)

    def test_subset_raises_on_polar_env(self, polar_env):
        mask = np.ones(polar_env.n_bins, dtype=bool)
        with pytest.raises(NotImplementedError, match=r"polar.*subset|subset.*polar"):
            polar_env.subset(bins=mask)

    # --- Graph operations remain valid on polar and must not regress.

    def test_path_between_works_on_polar_env(self, polar_env):
        path = polar_env.path_between(0, polar_env.n_bins - 1)
        assert isinstance(path, list)
        assert path[0] == 0 and path[-1] == polar_env.n_bins - 1

    def test_reachable_from_works_on_polar_env(self, polar_env):
        reachable = polar_env.reachable_from(0)
        assert reachable.shape == (polar_env.n_bins,)
        assert bool(reachable[0]) is True

    def test_smooth_works_on_polar_env(self, polar_env):
        field = np.zeros(polar_env.n_bins, dtype=float)
        field[0] = 1.0
        smoothed = polar_env.smooth(field, bandwidth=5.0)
        assert smoothed.shape == (polar_env.n_bins,)

    def test_repr_shows_polar_class_name(self, polar_env):
        """repr must identify the concrete polar type, not 'Environment'."""
        assert "EgocentricPolarEnvironment" in repr(polar_env)


class TestPolarSerializationRoundTrip:
    """The polar type must survive copy and to_file/from_file round-trips."""

    def test_copy_preserves_coordinate_kind(self):
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        env_copy = env.copy()
        assert isinstance(env_copy, EgocentricPolarEnvironment)
        assert env_copy._POLAR is True

    def test_to_file_from_file_preserves_coordinate_kind(self, tmp_path):
        from neurospatial.environment.polar import EgocentricPolarEnvironment
        from neurospatial.io.files import from_file, to_file

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        path = tmp_path / "polar_env"
        to_file(env, path)
        loaded = from_file(path)
        assert isinstance(loaded, EgocentricPolarEnvironment)
        assert loaded._POLAR is True


class TestPolarBoundaryGuardsExtended:
    """Regression coverage for review-found polar holes.

    Adds boundary guards for code paths the original polar-coordinate-kind
    commit missed: ``Environment.contains`` (was bypassing the polar check),
    ``Environment.plot_field`` (was silently mislabeling axes as
    X/Y), ``apply_transform`` (was silently flipping the env back to
    Cartesian), and the dict-round-trip (``from_dict`` was discarding
    ``coordinate_kind`` even though ``to_dict`` wrote it).
    """

    @pytest.fixture
    def polar_env(self):
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    def test_contains_raises_on_polar_env(self, polar_env):
        """contains is the boolean partner of bin_at and is unavailable on polar."""
        with pytest.raises(
            NotImplementedError, match=r"polar.*contains|contains.*polar"
        ):
            polar_env.contains(np.array([[5.0, 5.0]]))

    def test_plot_field_relabels_polar_axes(self, polar_env):
        """plot_field on a polar env labels axes Distance / Angle (rad), not X/Y."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        field = np.zeros(polar_env.n_bins)
        fig, ax = plt.subplots()
        try:
            polar_env.plot_field(field, ax=ax)
            assert "Distance" in ax.get_xlabel()
            assert "Angle" in ax.get_ylabel()
            assert "X Position" not in ax.get_xlabel()
            assert "Y Position" not in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_apply_transform_refuses_polar_env(self, polar_env):
        """apply_transform_to_environment must reject polar envs.

        An affine transform on (distance, angle) bin centers is not
        geometrically meaningful. The free-function path guards against a
        polar env being passed directly (the method override on
        EgocentricPolarEnvironment raises before reaching it).
        """
        from neurospatial.ops.transforms import (
            apply_transform_to_environment,
            translate,
        )

        T = translate(1.0, 1.0)
        with pytest.raises(ValueError, match=r"polar"):
            apply_transform_to_environment(polar_env, T)

    def test_apply_transform_method_refuses_polar_env(self, polar_env):
        """The apply_transform method on a polar env raises NotImplementedError."""
        from neurospatial.ops.transforms import translate

        T = translate(1.0, 1.0)
        with pytest.raises(
            NotImplementedError, match=r"polar.*apply_transform|apply_transform.*polar"
        ):
            polar_env.apply_transform(T)

    def test_from_dict_round_trip_preserves_coordinate_kind(self, polar_env):
        """to_dict / from_dict must round-trip coordinate_kind for polar envs.

        to_dict already writes the field; from_dict was discarding it,
        so an in-memory dict round-trip silently flipped polar envs to
        Cartesian even though to_file/from_file round-tripped correctly.
        """
        from neurospatial.environment.polar import EgocentricPolarEnvironment
        from neurospatial.io.files import from_dict, to_dict

        data = to_dict(polar_env)
        loaded = from_dict(data)
        assert isinstance(loaded, EgocentricPolarEnvironment)
        assert loaded._POLAR is True


class TestSubsetPreservesGraphLayout:
    """Regression for the review-found graph corruption.

    A 1-D linearized track (Graph layout) embedded in 2-D space
    populates ``active_mask``, ``grid_shape``, and ``grid_edges`` for
    the linearized 1-D representation -- so the original
    grid-fast-path check (looking only at those three attributes)
    routed graph parents through ``Environment.from_grid_mask``, which
    rebuilt them as MaskedGrid envs with 1-D bin_centers, dropping the
    2-D embedding and flipping ``is_linearized_track`` to False.
    """

    def _build_graph_env(self):
        import networkx as nx

        from neurospatial import Environment

        graph = nx.Graph()
        graph.add_node("A", pos=(0.0, 0.0))
        graph.add_node("B", pos=(10.0, 0.0))
        graph.add_node("C", pos=(10.0, 10.0))
        graph.add_edge("A", "B", distance=10.0)
        graph.add_edge("B", "C", distance=10.0)
        return Environment.from_graph(
            graph,
            edge_order=[("A", "B"), ("B", "C")],
            edge_spacing=0.0,
            bin_size=2.0,
        )

    def test_graph_subset_keeps_is_1d_and_n_dims(self):
        env = self._build_graph_env()
        assert env.is_linearized_track is True and env.n_dims == 2

        mask = np.zeros(env.n_bins, dtype=bool)
        mask[:5] = True
        sub = env.subset(bins=mask)

        assert sub.is_linearized_track is True, "graph subset must remain 1-D"
        assert sub.n_dims == 2, "graph subset must keep its 2-D embedding"

    def test_graph_subset_does_not_use_maskedgrid(self):
        """Graph subset stays on the inline SubsetLayout fallback, not MaskedGrid.

        MaskedGrid would flatten the 2-D embedding to 1-D bin_centers.
        """
        env = self._build_graph_env()
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[:5] = True
        sub = env.subset(bins=mask)
        assert sub._layout_type_used != "MaskedGrid"


class TestPolarEgocentricAngularSeam:
    """Distance-field topology across the angular ±pi seam.

    The factory takes ``distance_bin_size`` / ``angle_bin_size`` (not
    ``n_distance_bins`` / ``n_direction_bins``). To probe the angular topology
    in isolation, these tests measure graph-hop distance (a unit-weighted
    distance field): the weighted geodesic ``distance`` does not distinguish
    the two cases because the seam edge carries the full chord length as its
    weight, while the seam edge's mere existence (hop = 1) is the load-bearing
    property of ``circular_angle=True``.
    """

    @staticmethod
    def _build(circular_angle):
        # distance_bin_size=5 over (0, 50) -> 10 distance bins;
        # angle_bin_size=pi/6 over (-pi, pi) -> 12 angle bins.
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 6,
            circular_angle=circular_angle,
        )

    @staticmethod
    def _hop_distance_field(env, source_bin):
        from neurospatial.ops.distance import distance_field

        graph = env.connectivity.copy()
        for u, v in graph.edges():
            graph[u][v]["hop"] = 1.0
        return distance_field(graph, sources=[int(source_bin)], weight="hop")

    @staticmethod
    def _ring_bins(env):
        """Indices, angles, and key bins for the innermost distance ring."""
        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        ring = np.isclose(distances, distances.min())
        ring_idx = np.flatnonzero(ring)
        ring_angles = angles[ring_idx]
        source = int(ring_idx[np.argmax(ring_angles)])  # angle ~ +pi
        neg_pi = int(ring_idx[np.argmin(ring_angles)])  # angle ~ -pi
        zero = int(ring_idx[np.argmin(np.abs(ring_angles))])  # angle ~ 0
        return source, neg_pi, zero

    def test_distance_field_wraps_across_seam(self):
        """With circular_angle=True the -pi bin is closer than the 0 bin."""
        env = self._build(circular_angle=True)
        source, neg_pi, zero = self._ring_bins(env)

        field = self._hop_distance_field(env, source)

        # Source sits near +pi; the seam connects directly to the -pi bin.
        assert field[neg_pi] < field[zero]

    def test_distance_field_non_circular_does_not_wrap(self):
        """With circular_angle=False the -pi bin is farther than the 0 bin."""
        env = self._build(circular_angle=False)
        source, neg_pi, zero = self._ring_bins(env)

        field = self._hop_distance_field(env, source)

        # No seam edge: reaching -pi requires traversing the whole angular span.
        assert field[neg_pi] > field[zero]


class TestPolarTypeAndGeometry:
    """Validation slice for the distinct polar type and physical geometry."""

    @pytest.fixture
    def polar_env(self):
        # 5 distance rings (centers 5,15,25,35,45), 8 angle bins, no wrap so the
        # angular-step edges are unambiguous grid edges.
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
            circular_angle=False,
        )

    def test_polar_env_is_distinct_type(self, polar_env):
        """from_polar_egocentric returns EgocentricPolarEnvironment, not Environment."""
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        assert type(polar_env) is EgocentricPolarEnvironment
        assert not isinstance(polar_env, Environment)

        # Cartesian-only methods are absent/clearly errored, not silently disabled.
        with pytest.raises(NotImplementedError):
            polar_env.bin_at(np.array([[1.0, 1.0]]))
        with pytest.raises(NotImplementedError):
            polar_env.contains(np.array([[1.0, 1.0]]))
        with pytest.raises(NotImplementedError):
            polar_env.distance_between(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        with pytest.raises(NotImplementedError):
            polar_env.distance_to([0], metric="euclidean")

    def test_polar_edge_distances_physical(self, polar_env):
        """Two equal angular moves at different radii have arc length scaling with r.

        Before the geometry fix the edge "distance" was the Euclidean norm of
        (Delta_r, Delta_theta) = (0, pi/4), i.e. a constant ~0.785 at every
        radius -- so arcs at r=5 and r=45 were equal (a ~9x error). With the
        fix the arc length is r * Delta_theta, which scales with radius.
        """
        G = polar_env.connectivity
        # node id = distance_idx * n_angle + angle_idx ; n_angle = 8
        arc_inner = G.edges[0, 1]["distance"]  # ring r_center=5, angular step
        arc_outer = G.edges[32, 33]["distance"]  # ring r_center=45, angular step

        assert arc_inner == pytest.approx(5.0 * np.pi / 4, rel=1e-9)
        assert arc_outer == pytest.approx(45.0 * np.pi / 4, rel=1e-9)
        # Equal physical angular step -> arc scales with radius (45/5 = 9).
        assert arc_outer / arc_inner == pytest.approx(9.0, rel=1e-9)

        # Radial step (same angle, adjacent ring) is the radial distance Delta_r.
        radial = G.edges[0, 8]["distance"]
        assert radial == pytest.approx(10.0, rel=1e-9)

    def test_polar_geodesic_correct(self, polar_env):
        """A known polar geodesic matches the analytic value within tolerance.

        From the innermost ring center (r=5) to the outermost ring center
        (r=45) along the SAME angle, the shortest path traverses four radial
        edges of length Delta_r = 10 each, so the geodesic distance is 40.
        """
        # Same angle (idx 0), distance idx 0 -> distance idx 4: nodes 0 and 32.
        dist = polar_env.distance_to([0], metric="geodesic")
        # Node 32 is r=45 at the same angle as node 0 (r=5).
        assert dist[32] == pytest.approx(40.0, rel=1e-9)
        # A pure angular hop from node 0 to node 1 equals the inner arc length.
        assert dist[1] == pytest.approx(5.0 * np.pi / 4, rel=1e-9)
