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
