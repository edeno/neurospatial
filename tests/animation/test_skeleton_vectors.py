"""Tests for precomputed skeleton vectors implementation.

Tests the _build_skeleton_vectors helper function that replaces
per-frame Shapes layer updates with precomputed Vectors layer data.

This is part of the napari playback performance optimization to eliminate
the 5.38ms per-frame `layer.data` assignment blocking the Qt event loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.animation.skeleton import Skeleton

# Skip all tests if napari not available
pytest.importorskip("napari")


@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    from neurospatial import Environment

    positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def bodypart_data_with_skeleton():
    """Create BodypartData with skeleton for testing."""
    from neurospatial.animation.overlays import BodypartData

    # 5 frames, 3 bodyparts (head, body, tail) in a line
    n_frames = 5
    bodyparts = {
        "head": np.array([[5.0 + i, 7.0] for i in range(n_frames)]),
        "body": np.array([[4.0 + i, 5.0] for i in range(n_frames)]),
        "tail": np.array([[3.0 + i, 3.0] for i in range(n_frames)]),
    }
    skeleton = Skeleton(
        name="test",
        nodes=("head", "body", "tail"),
        edges=(("head", "body"), ("body", "tail")),
        edge_color="white",
        edge_width=2.0,
    )
    return BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors={"head": "red", "body": "green", "tail": "blue"},
    )


@pytest.fixture
def bodypart_data_no_skeleton():
    """Create BodypartData without skeleton."""
    from neurospatial.animation.overlays import BodypartData

    n_frames = 5
    bodyparts = {
        "head": np.array([[5.0 + i, 7.0] for i in range(n_frames)]),
    }
    return BodypartData(
        bodyparts=bodyparts,
        skeleton=None,  # No skeleton
        colors={"head": "red"},
    )


@pytest.fixture
def bodypart_data_empty_skeleton():
    """Create BodypartData with empty skeleton (no edges)."""
    from neurospatial.animation.overlays import BodypartData

    n_frames = 5
    bodyparts = {
        "head": np.array([[5.0 + i, 7.0] for i in range(n_frames)]),
    }
    # Skeleton with no edges (just a single node)
    skeleton = Skeleton(
        name="test",
        nodes=("head",),
        edges=(),  # Empty edges tuple
        edge_color="white",
        edge_width=2.0,
    )
    return BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,  # Skeleton with no edges
        colors={"head": "red"},
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBuildSkeletonVectorsShape:
    """Tests for output shape and structure of _build_skeleton_vectors."""

    def test_output_shape_correct(self, bodypart_data_with_skeleton, simple_env):
        """Test vectors array has shape (n_frames * n_edges, 2, 3)."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, _features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        # 5 frames * 2 edges = 10 segments
        n_frames = 5
        n_edges = 2
        expected_n_segments = n_frames * n_edges

        assert vectors.shape == (expected_n_segments, 2, 3)

    def test_features_contains_edge_names(
        self, bodypart_data_with_skeleton, simple_env
    ):
        """Test features dict contains edge_name array."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        assert "edge_name" in features
        assert len(features["edge_name"]) == vectors.shape[0]

    def test_edge_names_match_skeleton(self, bodypart_data_with_skeleton, simple_env):
        """Test edge names follow pattern 'start-end' for each skeleton edge."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        _vectors, features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        edge_names = features["edge_name"]

        # Should contain "body-head" and "body-tail" (canonical edge names)
        # Edges are canonicalized: ("head", "body") → ("body", "head") since "body" < "head"
        unique_names = set(edge_names)
        assert "body-head" in unique_names
        assert "body-tail" in unique_names


class TestBuildSkeletonVectorsTimeStamps:
    """Tests for time dimension in skeleton vectors."""

    def test_time_stamps_are_frame_indices(
        self, bodypart_data_with_skeleton, simple_env
    ):
        """Test time dimension contains frame indices 0, 1, 2, ..."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, _features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        # Extract time values from position row (first row)
        # Format is [position, direction] where position = [t, y, x]
        times_position = vectors[:, 0, 0]  # Time of position points

        # Direction row (second row) should have dt=0 (same time as position)
        dt_direction = vectors[:, 1, 0]  # Delta-time of direction vectors
        np.testing.assert_array_equal(dt_direction, np.zeros(len(dt_direction)))

        # Should contain frame indices 0-4 for 5 frames
        unique_times = np.unique(times_position)
        expected_times = np.arange(5)
        np.testing.assert_array_equal(unique_times, expected_times)

    def test_each_frame_has_all_edges(self, bodypart_data_with_skeleton, simple_env):
        """Test each frame has entries for all skeleton edges."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        times = vectors[:, 0, 0]
        edge_names = features["edge_name"]

        # Each frame should have 2 edges
        for frame in range(5):
            frame_mask = times == frame
            frame_edges = edge_names[frame_mask]
            assert len(frame_edges) == 2
            # Canonical edge names: "body" < "head", "body" < "tail"
            assert set(frame_edges) == {"body-head", "body-tail"}


class TestBuildSkeletonVectorsCoordinateTransform:
    """Tests for coordinate transformation from env to napari space."""

    def test_coordinates_are_transformed(self, bodypart_data_with_skeleton, simple_env):
        """Test coordinates use napari (row, col) convention, not env (x, y)."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
            _transform_coords_for_napari,
        )

        vectors, features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        # Get expected transformed coordinates for frame 0, body
        # Canonical edge is ("body", "head"), so start point is "body"
        body_env_coords = bodypart_data_with_skeleton.bodyparts["body"][0:1]
        body_napari = _transform_coords_for_napari(body_env_coords, simple_env)[0]

        # Find the body-head edge for frame 0 (canonical: "body" < "head")
        times = vectors[:, 0, 0]
        edge_names = features["edge_name"]

        frame0_body_head_mask = (times == 0) & (edge_names == "body-head")
        start_point = vectors[frame0_body_head_mask, 0, 1:3][0]  # [row, col]

        # Should match transformed coordinates (start point is now body)
        np.testing.assert_allclose(start_point, body_napari, rtol=1e-5)

    def test_start_and_end_points_different(
        self, bodypart_data_with_skeleton, simple_env
    ):
        """Test start and end points of each segment are different (not same point)."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, _features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        for i in range(vectors.shape[0]):
            start_coords = vectors[i, 0, 1:3]  # [row, col]
            end_coords = vectors[i, 1, 1:3]  # [row, col]
            assert not np.allclose(start_coords, end_coords), (
                f"Segment {i} has identical start and end"
            )


# =============================================================================
# Empty/None Skeleton Tests
# =============================================================================


class TestBuildSkeletonVectorsEmptySkeleton:
    """Tests for handling None or empty skeleton."""

    def test_none_skeleton_returns_empty(self, bodypart_data_no_skeleton, simple_env):
        """Test None skeleton returns empty vectors array."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, features = _build_skeleton_vectors(
            bodypart_data_no_skeleton, simple_env
        )

        assert vectors.size == 0
        assert features["edge_name"].size == 0

    def test_empty_skeleton_returns_empty(
        self, bodypart_data_empty_skeleton, simple_env
    ):
        """Test empty skeleton list returns empty vectors array."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, features = _build_skeleton_vectors(
            bodypart_data_empty_skeleton, simple_env
        )

        assert vectors.size == 0
        assert features["edge_name"].size == 0


# =============================================================================
# Data Type and Memory Tests
# =============================================================================


class TestBuildSkeletonVectorsDataTypes:
    """Tests for output data types and memory efficiency."""

    def test_vectors_dtype_is_float32(self, bodypart_data_with_skeleton, simple_env):
        """Test vectors array uses float32 for memory efficiency."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, _features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env
        )

        # Default should be float32 for memory efficiency
        assert vectors.dtype == np.float32

    def test_custom_dtype_respected(self, bodypart_data_with_skeleton, simple_env):
        """Test custom dtype parameter is respected."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )

        vectors, _features = _build_skeleton_vectors(
            bodypart_data_with_skeleton, simple_env, dtype=np.float64
        )

        assert vectors.dtype == np.float64


# =============================================================================
# NaN Handling Tests
# =============================================================================


class TestBuildSkeletonVectorsNaNHandling:
    """Tests for handling NaN values in bodypart coordinates."""

    def test_nan_endpoints_excluded(self, simple_env):
        """Test skeleton segments with NaN endpoints are excluded from output."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )
        from neurospatial.animation.overlays import BodypartData

        # Create bodypart data with NaN in frame 2
        n_frames = 5
        head_coords = np.array([[5.0 + i, 7.0] for i in range(n_frames)])
        head_coords[2] = [np.nan, np.nan]  # Frame 2 is NaN

        bodyparts = {
            "head": head_coords,
            "body": np.array([[4.0 + i, 5.0] for i in range(n_frames)]),
        }
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body"),
            edges=(("head", "body"),),
            edge_color="white",
            edge_width=2.0,
        )
        bodypart_data = BodypartData(
            bodyparts=bodyparts,
            skeleton=skeleton,
            colors=None,
        )

        vectors, _features = _build_skeleton_vectors(bodypart_data, simple_env)

        # Should have 4 segments (frame 2 excluded due to NaN)
        assert vectors.shape[0] == 4

        # Time stamps should be 0, 1, 3, 4 (no frame 2)
        times = vectors[:, 0, 0]
        assert 2 not in times


# =============================================================================
# Missing Bodypart Tests
# =============================================================================


class TestBuildSkeletonVectorsMissingBodyparts:
    """Tests for handling skeleton edges with missing bodypart names."""

    def test_missing_bodypart_edge_skipped(self, simple_env):
        """Test skeleton edge referencing missing bodypart is skipped."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )
        from neurospatial.animation.overlays import BodypartData

        n_frames = 3
        bodyparts = {
            "head": np.array([[5.0 + i, 7.0] for i in range(n_frames)]),
            "body": np.array([[4.0 + i, 5.0] for i in range(n_frames)]),
            # "tail" is missing from bodyparts dict (but defined in skeleton)
        }
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),  # tail edge exists
            edge_color="white",
            edge_width=2.0,
        )
        bodypart_data = BodypartData(
            bodyparts=bodyparts,
            skeleton=skeleton,  # tail is in skeleton but missing from bodyparts
            colors=None,
        )

        vectors, features = _build_skeleton_vectors(bodypart_data, simple_env)

        # Only body-head edges should be included (3 frames)
        # Canonical: ("head", "body") → ("body", "head") since "body" < "head"
        assert vectors.shape[0] == 3

        # All edges should be body-head (canonical form)
        assert all(name == "body-head" for name in features["edge_name"])


# =============================================================================
# Large Dataset Tests
# =============================================================================


class TestBuildSkeletonVectorsLargeDataset:
    """Tests for performance with large datasets."""

    def test_large_frame_count(self, simple_env):
        """Test function handles large frame counts efficiently."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )
        from neurospatial.animation.overlays import BodypartData

        # 1000 frames, 3 bodyparts, 2 edges
        n_frames = 1000
        bodyparts = {
            "head": np.random.rand(n_frames, 2) * 10 + 5,
            "body": np.random.rand(n_frames, 2) * 10 + 3,
            "tail": np.random.rand(n_frames, 2) * 10 + 1,
        }
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
            edge_color="white",
            edge_width=2.0,
        )
        bodypart_data = BodypartData(
            bodyparts=bodyparts,
            skeleton=skeleton,
            colors=None,
        )

        vectors, _features = _build_skeleton_vectors(bodypart_data, simple_env)

        # Should have 1000 * 2 = 2000 segments
        assert vectors.shape[0] == 2000
        assert vectors.shape == (2000, 2, 3)

    def test_memory_efficient_for_large_data(self, simple_env):
        """Test float32 dtype keeps memory usage reasonable for large datasets."""
        from neurospatial.animation.backends.napari_backend import (
            _build_skeleton_vectors,
        )
        from neurospatial.animation.overlays import BodypartData

        # 10000 frames, 8 bodyparts, 7 edges
        n_frames = 10000
        n_bodyparts = 8
        n_edges = 7

        bodypart_names = [f"part_{i}" for i in range(n_bodyparts)]
        bodyparts = {name: np.random.rand(n_frames, 2) * 20 for name in bodypart_names}
        edges = [(bodypart_names[i], bodypart_names[i + 1]) for i in range(n_edges)]
        skeleton = Skeleton(
            name="test",
            nodes=tuple(bodypart_names),
            edges=tuple(edges),
            edge_color="white",
            edge_width=2.0,
        )

        bodypart_data = BodypartData(
            bodyparts=bodyparts,
            skeleton=skeleton,
            colors=None,
        )

        vectors, _features = _build_skeleton_vectors(bodypart_data, simple_env)

        # Should have 10000 * 7 = 70000 segments
        expected_segments = n_frames * n_edges
        assert vectors.shape[0] == expected_segments

        # Memory should be reasonable (float32)
        # 70000 segments * 2 points * 3 values * 4 bytes = ~1.68 MB
        expected_bytes = expected_segments * 2 * 3 * 4  # float32 = 4 bytes
        assert vectors.nbytes == expected_bytes
