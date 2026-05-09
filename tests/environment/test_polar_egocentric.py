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

    def test_circular_angle_default_is_true(self):
        """Test that circular_angle defaults to True."""
        distance_range = (0.0, 10.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        connectivity = env.connectivity
        angles = env.bin_centers[:, 1]
        first_angle_bin = np.argmin(angles)
        last_angle_bin = np.argmax(angles)

        # Default should be circular, so they should be connected
        assert connectivity.has_edge(first_angle_bin, last_angle_bin)

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

    def test_environment_is_fitted(self):
        """Test that the returned environment is fitted."""
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 100.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        )

        # Should be able to call methods that require fitted state
        assert env._is_fitted

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

    def test_connectivity_graph_exists(self):
        """Test that connectivity graph is created."""
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 100.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        )

        assert env.connectivity is not None
        assert env.connectivity.number_of_nodes() == env.n_bins


class TestDocstringNote:
    """Test that environment has correct coordinate convention documentation."""

    def test_bin_centers_first_column_is_distance(self):
        """Verify bin_centers[:, 0] represents distance (per docstring convention)."""
        distance_range = (0.0, 50.0)
        angle_range = (-np.pi / 2, np.pi / 2)  # Half circle
        distance_bin_size = 25.0  # 2 distance bins
        angle_bin_size = np.pi / 2  # 2 angle bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        # First column should be distances (centered at 12.5 and 37.5)
        distances = env.bin_centers[:, 0]
        assert np.all(distances >= 0)  # Distances are non-negative

    def test_bin_centers_second_column_is_angle(self):
        """Verify bin_centers[:, 1] represents angle (per docstring convention)."""
        distance_range = (0.0, 10.0)  # 1 distance bin
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi  # 2 angle bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        # Second column should be angles (centered at -π/2 and π/2)
        angles = env.bin_centers[:, 1]
        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)


class TestPolarCoordinateKindFlag:
    """Regression for M1 1.3: polar envs are flagged distinct from Cartesian.

    Without an explicit flag, downstream code that computes Euclidean
    distance on `bin_centers` silently treats `(distance, angle)` pairs
    as `(x, y)` and produces meaningless numbers. The new
    `coordinate_kind` attribute and `is_polar` property make the
    distinction visible to callers and let safety checks fire.
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

    def test_from_polar_egocentric_marks_env_polar(self, polar_env):
        """from_polar_egocentric should set coordinate_kind='polar'."""
        assert polar_env.coordinate_kind == "polar"
        assert polar_env.is_polar is True

    def test_cartesian_factories_default_to_cartesian(self, cartesian_env):
        """from_samples / from_polygon / from_mask / etc. produce Cartesian envs."""
        assert cartesian_env.coordinate_kind == "cartesian"
        assert cartesian_env.is_polar is False


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
        with pytest.raises(ValueError, match=r"polar.*bin_at|bin_at.*polar"):
            polar_env.bin_at(from_xy)

    def test_distance_between_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            ValueError, match=r"polar.*distance_between|distance_between.*polar"
        ):
            polar_env.distance_between(np.array([0.0, 0.0]), np.array([10.0, 10.0]))

    def test_distance_to_euclidean_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            ValueError,
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


class TestPolarSerializationRoundTrip:
    """coordinate_kind must survive copy and to_file/from_file round-trips."""

    def test_copy_preserves_coordinate_kind(self):
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        env_copy = env.copy()
        assert env_copy.coordinate_kind == "polar"
        assert env_copy.is_polar is True

    def test_to_file_from_file_preserves_coordinate_kind(self, tmp_path):
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
        assert loaded.coordinate_kind == "polar"
        assert loaded.is_polar is True


class TestPolarBoundaryGuardsExtended:
    """Regression coverage for review-found polar holes (M1 1.3 follow-up).

    Adds boundary guards for code paths the original M1 1.3 commit
    missed: ``Environment.contains`` (was bypassing the polar check),
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
        """Environment.contains is the boolean partner of bin_at and must refuse polar."""
        with pytest.raises(ValueError, match=r"polar.*contains|contains.*polar"):
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
        geometrically meaningful, and the rebuilt env would silently
        reset coordinate_kind to "cartesian" without raising. Refuse
        at the boundary.
        """
        from neurospatial.ops.transforms import (
            apply_transform_to_environment,
            translate,
        )

        T = translate(1.0, 1.0)
        with pytest.raises(ValueError, match=r"polar|coordinate_kind"):
            apply_transform_to_environment(polar_env, T)

    def test_from_dict_round_trip_preserves_coordinate_kind(self, polar_env):
        """to_dict / from_dict must round-trip coordinate_kind for polar envs.

        to_dict already writes the field; from_dict was discarding it,
        so an in-memory dict round-trip silently flipped polar envs to
        Cartesian even though to_file/from_file round-tripped correctly.
        """
        from neurospatial.io.files import from_dict, to_dict

        data = to_dict(polar_env)
        loaded = from_dict(data)
        assert loaded.coordinate_kind == "polar"
        assert loaded.is_polar is True


class TestSubsetPreservesGraphLayout:
    """Regression for the M1 1.1 review-found graph corruption.

    A 1-D linearized track (Graph layout) embedded in 2-D space
    populates ``active_mask``, ``grid_shape``, and ``grid_edges`` for
    the linearized 1-D representation -- so the original M1 1.1
    grid-fast-path check (looking only at those three attributes)
    routed graph parents through ``Environment.from_mask``, which
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
