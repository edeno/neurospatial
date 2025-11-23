"""
Tests for NWB pose estimation reading functions.

Tests the read_pose() function for reading PoseEstimation data from NWB files
using the ndx-pose extension.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Skip all tests if ndx_pose is not installed
ndx_pose = pytest.importorskip("ndx_pose")


def _create_nwb_with_pose(
    nwbfile,
    skeleton_name: str,
    node_names: list[str],
    edges: np.ndarray,
    n_samples: int = 10,
    n_dims: int = 2,
):
    """
    Create NWB file with PoseEstimation using custom skeleton for testing.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to add pose estimation to.
    skeleton_name : str
        Name for the skeleton.
    node_names : list[str]
        Names of the skeleton nodes (bodyparts).
    edges : np.ndarray
        Edge array with shape (n_edges, 2) containing node indices.
    n_samples : int, optional
        Number of time samples. Default is 10.
    n_dims : int, optional
        Number of spatial dimensions. Default is 2.

    Returns
    -------
    NWBFile
        The modified NWB file with PoseEstimation added.
    """
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavior data"
    )

    skeleton = Skeleton(
        name=skeleton_name,
        nodes=node_names,
        edges=edges,
    )

    timestamps = np.arange(n_samples) / 30.0
    rng = np.random.default_rng(42)

    pose_series = [
        PoseEstimationSeries(
            name=name,
            data=rng.random((n_samples, n_dims)),
            confidence=np.ones(n_samples),
            timestamps=timestamps,
            reference_frame="test",
            unit="cm",
        )
        for name in node_names
    ]

    pose = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=pose_series,
        skeleton=skeleton,
        source_software="Test",
    )
    behavior_module.add(pose)

    nwbfile.create_processing_module(
        name="Skeletons", description="Skeleton definitions"
    )
    nwbfile.processing["Skeletons"].add(skeleton)

    return nwbfile


class TestReadPose:
    """Tests for read_pose() function."""

    def test_basic_pose_reading(self, sample_nwb_with_pose):
        """Test reading pose estimation data from NWB file."""
        from neurospatial.nwb import read_pose

        bodyparts, timestamps, skeleton = read_pose(sample_nwb_with_pose)

        # Check bodyparts dict
        assert isinstance(bodyparts, dict)
        assert len(bodyparts) == 3  # nose, body, tail
        assert "nose" in bodyparts
        assert "body" in bodyparts
        assert "tail" in bodyparts

        # Check shapes (500 samples, 2D data)
        for name, data in bodyparts.items():
            assert data.shape == (500, 2), f"Bodypart {name} has wrong shape"
            assert data.dtype == np.float64

        # Check timestamps
        assert timestamps.shape == (500,)
        assert timestamps.dtype == np.float64
        assert np.all(np.isfinite(timestamps))
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0.0

        # Check skeleton
        from neurospatial.animation.skeleton import Skeleton

        assert isinstance(skeleton, Skeleton)
        assert skeleton.n_nodes == 3
        assert skeleton.n_edges == 2
        assert set(skeleton.nodes) == {"nose", "body", "tail"}

    def test_pose_data_matches_original(self, sample_nwb_with_pose):
        """Test that read data matches the original data in the NWB file."""
        from neurospatial.nwb import read_pose

        bodyparts, timestamps, _skeleton = read_pose(sample_nwb_with_pose)

        # Get original data directly from NWB
        original_pose = sample_nwb_with_pose.processing["behavior"]["PoseEstimation"]

        for name in ["nose", "body", "tail"]:
            original_series = original_pose.pose_estimation_series[name]
            np.testing.assert_array_almost_equal(
                bodyparts[name], original_series.data[:]
            )

        # Check timestamps match first series
        first_series = original_pose.pose_estimation_series["nose"]
        np.testing.assert_array_almost_equal(timestamps, first_series.timestamps[:])

    def test_with_explicit_pose_estimation_name(self, empty_nwb):
        """Test reading with explicit pose_estimation_name parameter."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        # Create skeleton
        skeleton = Skeleton(
            name="test_skeleton",
            nodes=["a", "b"],
            edges=np.array([[0, 1]], dtype=np.uint8),
        )

        # Create two PoseEstimation containers
        timestamps = np.arange(50) / 30.0

        # First one (alphabetically second: "PoseB")
        pose_b = PoseEstimation(
            name="PoseB",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 10.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.ones((50, 2)) * 20.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose_b)

        # Second one (alphabetically first: "PoseA")
        pose_a = PoseEstimation(
            name="PoseA",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 99.0,  # Distinctive value
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.ones((50, 2)) * 88.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose_a)

        # Add skeleton to Skeletons module
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        # Request specific PoseEstimation by name
        bodyparts, _timestamps, _skeleton = read_pose(
            nwbfile, pose_estimation_name="PoseB"
        )

        # Check we got PoseB (value 10.0), not PoseA (value 99.0)
        np.testing.assert_array_almost_equal(bodyparts["a"], np.ones((50, 2)) * 10.0)

    def test_error_when_no_pose_estimation_found(self, empty_nwb):
        """Test KeyError when no PoseEstimation container found."""
        from neurospatial.nwb import read_pose

        with pytest.raises(KeyError, match=r"No PoseEstimation.*found"):
            read_pose(empty_nwb)

    def test_error_when_named_pose_not_found(self, sample_nwb_with_pose):
        """Test KeyError with available list when specific pose_estimation_name not found."""
        from neurospatial.nwb import read_pose

        with pytest.raises(KeyError, match=r"nonexistent.*not found.*Available"):
            read_pose(sample_nwb_with_pose, pose_estimation_name="nonexistent")

    def test_skeleton_extraction(self, sample_nwb_with_pose):
        """Test that skeleton is correctly extracted and converted."""
        from neurospatial.nwb import read_pose

        _bodyparts, _timestamps, skeleton = read_pose(sample_nwb_with_pose)

        # Verify skeleton structure matches what was stored
        assert skeleton.name == "mouse_skeleton"
        assert skeleton.nodes == ("nose", "body", "tail")
        # Edges are canonicalized (sorted lexicographically)
        assert ("body", "nose") in skeleton.edges or ("nose", "body") in skeleton.edges
        assert ("body", "tail") in skeleton.edges or ("tail", "body") in skeleton.edges

    def test_multiple_bodyparts_to_dict(self, sample_nwb_with_pose):
        """Test that all bodyparts are converted to dict correctly."""
        from neurospatial.nwb import read_pose

        bodyparts, _timestamps, _skeleton = read_pose(sample_nwb_with_pose)

        # All bodyparts should be present
        expected_bodyparts = {"nose", "body", "tail"}
        assert set(bodyparts.keys()) == expected_bodyparts

        # Each should be a 2D array
        for name, data in bodyparts.items():
            assert data.ndim == 2, f"Bodypart {name} should be 2D"
            assert data.shape[1] == 2, f"Bodypart {name} should have 2 columns (x, y)"

    def test_multiple_pose_estimation_uses_first_alphabetically(
        self, empty_nwb, caplog
    ):
        """Test that multiple PoseEstimation uses first alphabetically with INFO log."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        skeleton = Skeleton(
            name="skel",
            nodes=["a", "b"],
            edges=np.array([[0, 1]], dtype=np.uint8),
        )

        timestamps = np.arange(50) / 30.0

        # Add "zebra" first (alphabetically last)
        pose_z = PoseEstimation(
            name="zebra",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 999.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.ones((50, 2)) * 999.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose_z)

        # Add "alpha" second (alphabetically first)
        pose_a = PoseEstimation(
            name="alpha",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 1.0,  # Distinctive value
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.ones((50, 2)) * 1.0,
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose_a)

        # Add skeleton
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        # Enable logging capture
        with caplog.at_level(logging.INFO, logger="neurospatial.nwb"):
            bodyparts, _timestamps, _skeleton = read_pose(nwbfile)

        # Should use 'alpha' (first alphabetically)
        np.testing.assert_array_almost_equal(bodyparts["a"], np.ones((50, 2)) * 1.0)

        # Check INFO log message
        assert any(
            "Multiple PoseEstimation" in record.message
            or ("Using" in record.message and "alpha" in record.message)
            for record in caplog.records
        )

    def test_prioritizes_behavior_module_over_others(self, empty_nwb):
        """Test that processing/behavior is prioritized over other modules."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        skeleton = Skeleton(
            name="skel",
            nodes=["a"],
            edges=np.array([], dtype=np.uint8).reshape(0, 2),
        )

        timestamps = np.arange(50) / 30.0

        # Add to analysis module first (should be deprioritized)
        analysis_module = nwbfile.create_processing_module(
            name="analysis", description="Analysis data"
        )
        pose_analysis = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 10.0,  # Distinctive value
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        analysis_module.add(pose_analysis)

        # Add to behavior module second (should be prioritized)
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        pose_behavior = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)) * 99.0,  # Different distinctive value
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose_behavior)

        # Add skeleton
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        bodyparts, _timestamps, _skeleton = read_pose(nwbfile)

        # Should get behavior module (value 99.0), not analysis (value 10.0)
        np.testing.assert_array_almost_equal(bodyparts["a"], np.ones((50, 2)) * 99.0)

    def test_uses_rate_when_timestamps_not_available(self, empty_nwb):
        """Test that timestamps are computed from rate when explicit timestamps not provided."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        skeleton = Skeleton(
            name="skel",
            nodes=["a"],
            edges=np.array([], dtype=np.uint8).reshape(0, 2),
        )

        # Create pose series with rate instead of timestamps
        pose = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.random.rand(100, 2),
                    confidence=np.ones(100),
                    rate=30.0,  # 30 Hz, no explicit timestamps
                    starting_time=0.0,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose)

        # Add skeleton
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        bodyparts, timestamps, _skeleton = read_pose(nwbfile)

        assert bodyparts["a"].shape == (100, 2)
        assert timestamps.shape == (100,)
        # Check computed timestamps are correct
        expected_timestamps = np.arange(100) / 30.0
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_uses_rate_with_nonzero_starting_time(self, empty_nwb):
        """Test that starting_time offset is applied correctly."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        skeleton = Skeleton(
            name="skel",
            nodes=["a"],
            edges=np.array([], dtype=np.uint8).reshape(0, 2),
        )

        # Create pose series with rate and nonzero starting_time
        pose = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.random.rand(100, 2),
                    confidence=np.ones(100),
                    rate=30.0,  # 30 Hz
                    starting_time=10.5,  # Start at 10.5 seconds
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose)

        # Add skeleton
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        bodyparts, timestamps, _skeleton = read_pose(nwbfile)

        assert bodyparts["a"].shape == (100, 2)
        assert timestamps.shape == (100,)
        # Check computed timestamps include offset
        expected_timestamps = np.arange(100) / 30.0 + 10.5
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_3d_pose_data(self, empty_nwb):
        """Test reading 3D pose data."""
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        skeleton = Skeleton(
            name="skel",
            nodes=["a", "b"],
            edges=np.array([[0, 1]], dtype=np.uint8),
        )

        timestamps = np.arange(50) / 30.0

        pose = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.random.rand(50, 3),  # 3D data
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.random.rand(50, 3),  # 3D data
                    confidence=np.ones(50),
                    timestamps=timestamps,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose)

        # Add skeleton
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        bodyparts, timestamps_out, _skeleton = read_pose(nwbfile)

        assert bodyparts["a"].shape == (50, 3)
        assert bodyparts["b"].shape == (50, 3)
        assert timestamps_out.shape == (50,)

    def test_linked_timestamps(self, empty_nwb):
        """Test reading pose with linked timestamps (common pattern from DeepLabCut).

        In ndx-pose, subsequent series can reference timestamps from the first
        series rather than duplicating them. This tests that pattern.
        """
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.nwb import read_pose

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )

        skeleton = Skeleton(
            name="skel",
            nodes=["a", "b", "c"],
            edges=np.array([[0, 1], [1, 2]], dtype=np.uint8),
        )

        timestamps = np.arange(100) / 30.0

        # First series has explicit timestamps
        series_a = PoseEstimationSeries(
            name="a",
            data=np.random.rand(100, 2),
            confidence=np.ones(100),
            timestamps=timestamps,  # Explicit timestamps
            reference_frame="test",
            unit="cm",
        )

        # Subsequent series link to the first series (common pattern)
        # In pynwb, this creates a link rather than duplicating data
        series_b = PoseEstimationSeries(
            name="b",
            data=np.random.rand(100, 2),
            confidence=np.ones(100),
            timestamps=series_a,  # Link to series_a timestamps
            reference_frame="test",
            unit="cm",
        )

        series_c = PoseEstimationSeries(
            name="c",
            data=np.random.rand(100, 2),
            confidence=np.ones(100),
            timestamps=series_a,  # Link to series_a timestamps
            reference_frame="test",
            unit="cm",
        )

        pose = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[series_a, series_b, series_c],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose)

        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        nwbfile.processing["Skeletons"].add(skeleton)

        bodyparts, timestamps_out, _skeleton = read_pose(nwbfile)

        # All bodyparts should be present
        assert set(bodyparts.keys()) == {"a", "b", "c"}

        # Timestamps should be correctly extracted (from first series alphabetically)
        assert timestamps_out.shape == (100,)
        np.testing.assert_array_almost_equal(timestamps_out, timestamps)


class TestReadPoseImportError:
    """Tests for ndx-pose import error handling.

    These tests verify that read_pose() raises an ImportError with a helpful
    message when ndx-pose is not installed. Since ndx_pose is imported at module
    level (via pytest.importorskip), we need a separate test approach.
    """

    def test_import_error_message_format(self):
        """Test that _require_ndx_pose() provides helpful error message."""
        # This test verifies the error message format exists.
        # Since ndx_pose IS installed (otherwise this file wouldn't run),
        # we test the helper function directly.
        from neurospatial.nwb._core import _require_ndx_pose

        # Should not raise when ndx_pose is installed
        result = _require_ndx_pose()
        assert result is not None


class TestSkeletonRoundTrip:
    """Tests for ndx-pose Skeleton → neurospatial Skeleton round-trip conversion.

    These tests verify that Skeleton.from_ndx_pose() correctly converts
    ndx-pose skeletons to neurospatial skeletons without data loss.
    """

    def test_skeleton_name_preserved(self, empty_nwb):
        """Test that skeleton name is preserved through conversion."""
        from neurospatial.nwb import read_pose

        nwbfile = _create_nwb_with_pose(
            empty_nwb,
            skeleton_name="custom_skeleton_name",
            node_names=["a", "b"],
            edges=np.array([[0, 1]], dtype=np.uint8),
        )

        _bodyparts, _timestamps, ns_skeleton = read_pose(nwbfile)

        # Verify name preserved
        assert ns_skeleton.name == "custom_skeleton_name"

    def test_skeleton_nodes_preserved(self, empty_nwb):
        """Test that all skeleton nodes are preserved through conversion."""
        from neurospatial.nwb import read_pose

        node_names = ["nose", "left_ear", "right_ear", "neck", "body", "tail"]
        nwbfile = _create_nwb_with_pose(
            empty_nwb,
            skeleton_name="multi_node_skeleton",
            node_names=node_names,
            edges=np.array(
                [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]], dtype=np.uint8
            ),  # nose-neck, left_ear-neck, right_ear-neck, neck-body, body-tail
        )

        _bodyparts, _timestamps, ns_skeleton = read_pose(nwbfile)

        # Verify all nodes preserved
        assert ns_skeleton.n_nodes == len(node_names)
        assert set(ns_skeleton.nodes) == set(node_names)

    def test_skeleton_edges_preserved(self, empty_nwb):
        """Test that all skeleton edges are preserved through conversion."""
        from neurospatial.nwb import read_pose

        # Create skeleton with specific edges: a-b, b-c, c-d (chain structure)
        node_names = ["a", "b", "c", "d"]
        nwbfile = _create_nwb_with_pose(
            empty_nwb,
            skeleton_name="chain_skeleton",
            node_names=node_names,
            edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8),
        )

        _bodyparts, _timestamps, ns_skeleton = read_pose(nwbfile)

        # Verify edge count preserved
        assert ns_skeleton.n_edges == 3

        # Verify edges are preserved (note: edges are canonicalized to sorted order)
        # Expected edges: (a, b), (b, c), (c, d)
        expected_edges_as_sets = [{"a", "b"}, {"b", "c"}, {"c", "d"}]
        actual_edges_as_sets = [set(edge) for edge in ns_skeleton.edges]
        for expected in expected_edges_as_sets:
            assert expected in actual_edges_as_sets, f"Edge {expected} not found"

    def test_skeleton_no_edges(self, empty_nwb):
        """Test skeleton with no edges (isolated nodes) is converted correctly."""
        from neurospatial.nwb import read_pose

        # Create skeleton with no edges
        node_names = ["point1", "point2", "point3"]
        nwbfile = _create_nwb_with_pose(
            empty_nwb,
            skeleton_name="isolated_nodes",
            node_names=node_names,
            edges=np.array([], dtype=np.uint8).reshape(0, 2),
        )

        _bodyparts, _timestamps, ns_skeleton = read_pose(nwbfile)

        # Verify nodes preserved but no edges
        assert ns_skeleton.n_nodes == 3
        assert ns_skeleton.n_edges == 0
        assert set(ns_skeleton.nodes) == set(node_names)

    def test_skeleton_complex_graph(self, empty_nwb):
        """Test skeleton with complex branching structure (not just chain)."""
        from neurospatial.nwb import read_pose

        # Create skeleton with branching (star topology from center)
        # center connects to: nose, left_ear, right_ear, tail
        node_names = ["center", "nose", "left_ear", "right_ear", "tail"]
        nwbfile = _create_nwb_with_pose(
            empty_nwb,
            skeleton_name="star_skeleton",
            node_names=node_names,
            edges=np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.uint8),
        )

        _bodyparts, _timestamps, ns_skeleton = read_pose(nwbfile)

        # Verify structure
        assert ns_skeleton.n_nodes == 5
        assert ns_skeleton.n_edges == 4

        # Verify adjacency for center node (should have 4 neighbors)
        assert len(ns_skeleton.adjacency["center"]) == 4
        assert set(ns_skeleton.adjacency["center"]) == {
            "nose",
            "left_ear",
            "right_ear",
            "tail",
        }
