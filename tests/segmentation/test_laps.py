"""Tests for lap detection functions.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment


class TestDetectLaps:
    """Test detect_laps function."""

    def test_detect_laps_circular_track_auto(self):
        """Test lap detection on circular track with auto template."""
        # Create circular track trajectory
        n_samples = 500
        theta = np.linspace(0, 8 * np.pi, n_samples)  # 4 complete laps
        radius = 30.0
        center = 50.0
        x = center + radius * np.cos(theta)
        y = center + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        # Create environment
        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 100, n_samples)

        from neurospatial.segmentation.laps import detect_laps

        laps = detect_laps(
            trajectory_bins,
            times,
            env,
            method="auto",
            min_overlap=0.7,
            direction="both",
        )

        # Should detect ~4 laps (auto template from first 10%, then 3 more laps)
        assert len(laps) >= 2, "Should detect at least 2 laps on circular track"

        # Verify lap structure
        for lap in laps:
            assert hasattr(lap, "start_time")
            assert hasattr(lap, "end_time")
            assert hasattr(lap, "direction")
            assert hasattr(lap, "overlap_score")
            assert lap.end_time > lap.start_time
            assert 0.0 <= lap.overlap_score <= 1.0

    def test_detect_laps_direction_clockwise(self):
        """Test detection of clockwise laps only."""
        n_samples = 300
        theta = np.linspace(0, 4 * np.pi, n_samples)  # 2 laps clockwise
        radius = 30.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 60, n_samples)

        from neurospatial.segmentation.laps import detect_laps

        # Since we're going counter-clockwise (theta increases → CCW)
        # Request only clockwise should give fewer/no laps
        laps_cw = detect_laps(
            trajectory_bins, times, env, method="auto", direction="clockwise"
        )
        laps_ccw = detect_laps(
            trajectory_bins, times, env, method="auto", direction="counter-clockwise"
        )

        # Counter-clockwise should have more laps
        assert len(laps_ccw) >= len(laps_cw)

    def test_detect_laps_reference_method(self):
        """Test lap detection with user-provided reference lap."""
        n_samples = 400
        theta = np.linspace(0, 6 * np.pi, n_samples)  # 3 laps
        radius = 30.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 80, n_samples)

        # Create reference lap from first complete lap
        # First 1/3 of trajectory is roughly one lap
        reference_bins = trajectory_bins[: n_samples // 3]

        from neurospatial.segmentation.laps import detect_laps

        laps = detect_laps(
            trajectory_bins,
            times,
            env,
            method="reference",
            reference_lap=reference_bins,
            min_overlap=0.6,
        )

        # Should detect laps matching the reference
        assert len(laps) >= 1
        for lap in laps:
            assert lap.overlap_score >= 0.6

    def test_detect_laps_region_method(self):
        """Test lap detection using region crossings."""
        # Create circular trajectory
        n_samples = 300
        theta = np.linspace(0, 4 * np.pi, n_samples)
        radius = 30.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 60, n_samples)

        # Add start region at theta=0 position
        start_x = 50 + radius
        start_y = 50
        env.regions.add("start", polygon=Point(start_x, start_y).buffer(5.0))

        from neurospatial.segmentation.laps import detect_laps

        laps = detect_laps(
            trajectory_bins,
            times,
            env,
            method="region",
            start_region="start",
            min_overlap=0.0,  # Not used for region method
        )

        # Should detect laps as segments between region crossings
        assert len(laps) >= 1
        for lap in laps:
            assert lap.end_time > lap.start_time

    def test_detect_laps_overlap_threshold(self):
        """Test that min_overlap filters low-quality laps."""
        n_samples = 200
        theta = np.linspace(0, 4 * np.pi, n_samples)
        radius = 30.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 40, n_samples)

        from neurospatial.segmentation.laps import detect_laps

        # Low threshold should accept more laps
        laps_low = detect_laps(
            trajectory_bins, times, env, method="auto", min_overlap=0.5
        )
        # High threshold should accept fewer laps
        laps_high = detect_laps(
            trajectory_bins, times, env, method="auto", min_overlap=0.9
        )

        assert len(laps_low) >= len(laps_high)

    def test_detect_laps_empty_trajectory(self):
        """Test lap detection on empty trajectory."""
        env = Environment.from_samples(np.random.randn(100, 2) * 10 + 50, bin_size=5.0)
        trajectory_bins = np.array([], dtype=np.int64)
        times = np.array([], dtype=np.float64)

        from neurospatial.segmentation.laps import detect_laps

        laps = detect_laps(trajectory_bins, times, env, method="auto")

        assert len(laps) == 0

    def test_detect_laps_no_laps_found(self):
        """Test lap detection when no laps match criteria."""
        # Straight line trajectory (no laps)
        positions = np.column_stack([np.linspace(0, 100, 100), np.ones(100) * 50])
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 20, 100)

        from neurospatial.segmentation.laps import detect_laps

        laps = detect_laps(trajectory_bins, times, env, method="auto", min_overlap=0.8)

        # Straight line should not produce laps
        assert len(laps) == 0

    def test_detect_laps_parameter_order(self):
        """Test that parameters are in the expected order."""
        positions = np.random.randn(50, 2) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.segmentation.laps import detect_laps

        # Should accept positional args in this order
        laps = detect_laps(trajectory_bins, times, env)
        assert isinstance(laps, list)

    def test_detect_laps_validation_method(self):
        """Test validation of method parameter."""
        positions = np.random.randn(50, 2) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.segmentation.laps import detect_laps

        with pytest.raises(ValueError, match="method must be one of"):
            detect_laps(trajectory_bins, times, env, method="invalid")

    def test_detect_laps_validation_reference_required(self):
        """Test that reference method requires reference_lap parameter."""
        positions = np.random.randn(50, 2) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.segmentation.laps import detect_laps

        with pytest.raises(
            ValueError, match=r"reference_lap.*required when method='reference'"
        ):
            detect_laps(trajectory_bins, times, env, method="reference")

    def test_detect_laps_validation_region_required(self):
        """Test that region method requires start_region parameter."""
        positions = np.random.randn(50, 2) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.segmentation.laps import detect_laps

        with pytest.raises(
            ValueError, match=r"start_region.*required when method='region'"
        ):
            detect_laps(trajectory_bins, times, env, method="region")

    def test_detect_laps_integration_workflow(self):
        """Test complete lap detection workflow with multiple methods."""
        # Create complex trajectory with multiple laps
        n_samples = 600
        theta = np.linspace(0, 10 * np.pi, n_samples)  # 5 laps
        radius = 25.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        trajectory_bins = env.bin_at(positions)
        times = np.linspace(0, 120, n_samples)

        # Add start region
        env.regions.add("start", polygon=Point(50 + radius, 50).buffer(5.0))

        from neurospatial.segmentation.laps import detect_laps

        # Test all three methods
        laps_auto = detect_laps(trajectory_bins, times, env, method="auto")

        # Get reference from first lap
        template_size = n_samples // 5
        reference = trajectory_bins[:template_size]
        laps_ref = detect_laps(
            trajectory_bins, times, env, method="reference", reference_lap=reference
        )

        laps_region = detect_laps(
            trajectory_bins, times, env, method="region", start_region="start"
        )

        # All methods should detect laps
        assert len(laps_auto) >= 2
        assert len(laps_ref) >= 2
        assert len(laps_region) >= 2

        # Verify temporal ordering
        for laps in [laps_auto, laps_ref, laps_region]:
            for i in range(len(laps) - 1):
                assert laps[i].end_time <= laps[i + 1].start_time
