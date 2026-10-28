"""Tests for lap detection functions."""

import itertools

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment, Regions


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
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 100, n_samples)

        from neurospatial.behavior.segmentation import detect_laps

        laps = detect_laps(
            position_bins,
            times,
            env,
            method="auto",
            min_overlap=0.7,
            direction="both",
        )

        # Trajectory is exactly 4 laps. The auto template consumes the first
        # ~10% of the trajectory, so one boundary lap can be dropped; observed
        # count is 3. Pin to {3, 4} to catch both over- and under-detection.
        assert len(laps) in {3, 4}, (
            f"4-lap circular track should yield 3-4 detected laps, got {len(laps)}"
        )

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
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 60, n_samples)

        from neurospatial.behavior.segmentation import detect_laps

        # theta increases, so the trajectory is counter-clockwise. The auto
        # template is the first circuit, itself the first lap, so both laps
        # are found; requesting the opposite direction must find none.
        laps_cw = detect_laps(
            position_bins,
            times,
            env,
            method="auto",
            min_overlap=0.6,
            direction="clockwise",
        )
        laps_ccw = detect_laps(
            position_bins,
            times,
            env,
            method="auto",
            min_overlap=0.6,
            direction="counter-clockwise",
        )

        # Direction filter must discriminate: CCW motion yields a CCW lap and no
        # CW laps (a sign error in the direction test would flip these).
        assert len(laps_cw) == 0
        assert len(laps_ccw) == 2
        assert all(lap.direction == "counter-clockwise" for lap in laps_ccw)

    def test_detect_laps_reference_method(self):
        """Test lap detection with user-provided reference lap."""
        n_samples = 400
        theta = np.linspace(0, 6 * np.pi, n_samples)  # 3 laps
        radius = 30.0
        x = 50 + radius * np.cos(theta)
        y = 50 + radius * np.sin(theta)
        positions = np.column_stack([x, y])

        env = Environment.from_samples(positions, bin_size=3.0)
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 80, n_samples)

        # Create reference lap from first complete lap
        # First 1/3 of trajectory is roughly one lap
        reference_bins = position_bins[: n_samples // 3]

        from neurospatial.behavior.segmentation import detect_laps

        laps = detect_laps(
            position_bins,
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
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 60, n_samples)

        # Add start region at theta=0 position
        start_x = 50 + radius
        start_y = 50
        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(start_x, start_y).buffer(5.0))
        env = env.with_regions(region_builder)

        from neurospatial.behavior.segmentation import detect_laps

        laps = detect_laps(
            position_bins,
            times,
            env,
            method="region",
            start_region="start",
            min_overlap=0.0,  # Not used for region method
        )

        # The 2-lap (4pi) trajectory crosses the start region twice, which
        # bounds exactly one lap segment between the crossings.
        assert len(laps) == 1
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
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 40, n_samples)

        from neurospatial.behavior.segmentation import detect_laps

        # Low threshold should accept more laps
        laps_low = detect_laps(
            position_bins, times, env, method="auto", min_overlap=0.5
        )
        # High threshold should accept fewer laps
        laps_high = detect_laps(
            position_bins, times, env, method="auto", min_overlap=0.9
        )

        assert len(laps_low) >= len(laps_high)

    def test_detect_laps_empty_trajectory(self):
        """Test lap detection on empty trajectory."""
        # Deterministic grid
        x = np.linspace(40, 60, 11)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = np.array([], dtype=np.int64)
        times = np.array([], dtype=np.float64)

        from neurospatial.behavior.segmentation import detect_laps

        laps = detect_laps(position_bins, times, env, method="auto")

        assert len(laps) == 0

    def test_detect_laps_no_laps_found(self):
        """Test lap detection when no laps match criteria."""
        # Straight line trajectory (no laps)
        positions = np.column_stack([np.linspace(0, 100, 100), np.ones(100) * 50])
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 20, 100)

        from neurospatial.behavior.segmentation import detect_laps

        laps = detect_laps(position_bins, times, env, method="auto", min_overlap=0.8)

        # Straight line should not produce laps
        assert len(laps) == 0

    def test_detect_laps_validation_method(self):
        """Test validation of method parameter."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((50, 2)) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.behavior.segmentation import detect_laps

        with pytest.raises(ValueError, match="method must be one of"):
            detect_laps(position_bins, times, env, method="invalid")

    def test_detect_laps_validation_reference_required(self):
        """Test that reference method requires reference_lap parameter."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((50, 2)) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.behavior.segmentation import detect_laps

        with pytest.raises(
            ValueError, match=r"reference_lap.*required when method='reference'"
        ):
            detect_laps(position_bins, times, env, method="reference")

    def test_detect_laps_validation_region_required(self):
        """Test that region method requires start_region parameter."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((50, 2)) * 20 + 50
        env = Environment.from_samples(positions, bin_size=5.0)
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 10, 50)

        from neurospatial.behavior.segmentation import detect_laps

        with pytest.raises(
            ValueError, match=r"start_region.*required when method='region'"
        ):
            detect_laps(position_bins, times, env, method="region")

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
        position_bins = env.bin_at(positions)
        times = np.linspace(0, 120, n_samples)

        # Add start region
        region_builder = Regions(env.regions.values())
        region_builder.add("start", polygon=Point(50 + radius, 50).buffer(5.0))
        env = env.with_regions(region_builder)

        from neurospatial.behavior.segmentation import detect_laps

        # Test all three methods
        laps_auto = detect_laps(position_bins, times, env, method="auto")

        # Get reference from first lap
        template_size = n_samples // 5
        reference = position_bins[:template_size]
        laps_ref = detect_laps(
            position_bins, times, env, method="reference", reference_lap=reference
        )

        laps_region = detect_laps(
            position_bins, times, env, method="region", start_region="start"
        )

        # All methods should detect laps
        assert len(laps_auto) >= 2
        assert len(laps_ref) >= 2
        assert len(laps_region) >= 2

        # Verify temporal ordering
        for laps in [laps_auto, laps_ref, laps_region]:
            for i in range(len(laps) - 1):
                assert laps[i].end_time <= laps[i + 1].start_time


class TestLapRoundTrip:
    """Pin the lap simulator and detector to each other."""

    def test_simulate_then_detect_recovers_lap_count(self):
        """simulate_trajectory_laps -> detect_laps recovers the simulated count."""
        from neurospatial.behavior.segmentation import detect_laps
        from neurospatial.simulation.trajectory import simulate_trajectory_laps

        # Linear-corridor environment for back-and-forth laps.
        x = np.linspace(0, 100, 200)
        env = Environment.from_samples(
            np.column_stack([x, np.full_like(x, 50.0)]), bin_size=4.0
        )

        n_laps = 5
        positions, times = simulate_trajectory_laps(
            env,
            n_laps=n_laps,
            seed=7,
            sampling_frequency=30.0,
            speed_mean=15.0,
        )
        position_bins = env.bin_at(positions)

        laps = detect_laps(
            position_bins, times, env, method="auto", min_overlap=0.7, direction="both"
        )

        # The simulator's laps are one-way traversals of the corridor; a lap
        # found by detect_laps is a complete circuit, out to the far end and
        # back to the start, so 5 traversals hold 2 complete circuits.
        assert len(laps) == n_laps // 2, (
            f"Simulated {n_laps} traversals but detected {len(laps)} circuits"
        )
        # Each circuit starts and ends at the start of the corridor.
        start_x = positions[0, 0]
        for lap in laps:
            for time in (lap.start_time, lap.end_time):
                x = positions[np.searchsorted(times, time), 0]
                assert abs(x - start_x) <= 2 * 4.0  # within the start zone


class TestLapDirection:
    """Lap direction is the sign of the closed path's enclosed area."""

    @pytest.mark.parametrize("center", [(0.0, 0.0), (300.0, 300.0), (-1e4, 5e3)])
    @pytest.mark.parametrize("phase", np.arange(8) * np.pi / 4)
    @pytest.mark.parametrize(
        ("sign", "expected"), [(1, "counter-clockwise"), (-1, "clockwise")]
    )
    def test_partial_lap_direction_ignores_origin_and_start_phase(
        self, center, phase, sign, expected
    ):
        """An 85% arc keeps its turning direction wherever it is drawn."""
        from neurospatial.behavior.segmentation import _detect_lap_direction

        theta = phase + sign * np.linspace(0.0, 0.85 * 2 * np.pi, 100)
        points = np.c_[center[0] + 30 * np.cos(theta), center[1] + 30 * np.sin(theta)]
        assert _detect_lap_direction(points, np.arange(100)) == expected

    def test_region_laps_far_from_origin(self):
        """Counter-clockwise laps on a track centered at (300, 300) cm."""
        from neurospatial.behavior.segmentation import detect_laps

        theta = np.linspace(0, 6 * np.pi, 300)
        positions = np.column_stack(
            [300 + 30 * np.cos(theta), 300 + 30 * np.sin(theta)]
        )
        env = Environment.from_samples(positions, bin_size=3.0)
        regions = Regions(env.regions.values())
        regions.add("start", polygon=Point(300.0, 330.0).buffer(5.0))  # top
        env = env.with_regions(regions)
        laps = detect_laps(
            env.bin_at(positions),
            np.linspace(0, 60, 300),
            env,
            method="region",
            start_region="start",
        )
        assert len(laps) == 2
        assert [lap.direction for lap in laps] == ["counter-clockwise"] * 2


class TestRegionLaps:
    """Region laps need a trip to the far side; jitter does not split them."""

    @staticmethod
    def _circle_with_jitter():
        theta = np.linspace(0, 6 * np.pi, 241)  # 3 laps of 80 samples
        circle = np.c_[50 + 30 * np.cos(theta), 50 + 30 * np.sin(theta)]
        env = Environment.from_samples(circle, bin_size=3.0)
        regions = Regions(env.regions.values())
        # Samples 0-1, 79-81, 159-161, 239-240 fall in this region; the
        # animal enters it at samples 79, 159 and 239.
        regions.add("start", polygon=Point(80.0, 50.0).buffer(4.0))
        env = env.with_regions(regions)
        # After the entries at 79 and 159, tracking jitters out and back in
        # for one frame: entries now at samples 79, 81, 161, 163 and 243.
        order = np.r_[
            np.arange(0, 80), 78, 79, np.arange(80, 160), 158, 159, np.arange(160, 241)
        ]
        positions = circle[order]
        times = np.arange(order.size) * 0.25
        return env, env.bin_at(positions), times

    def test_boundary_jitter_does_not_split_laps(self):
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times = self._circle_with_jitter()
        laps = detect_laps(bins, times, env, method="region", start_region="start")

        assert [(lap.start_time, lap.end_time) for lap in laps] == [
            (times[79], times[161]),
            (times[161], times[243]),
        ]
        assert [lap.direction for lap in laps] == ["counter-clockwise"] * 2
        assert all(np.isnan(lap.overlap_score) for lap in laps)


class TestTemplateLaps:
    """Template laps are found whole at any speed, with exact boundaries."""

    RADIUS = 30.0
    BIN = 2.0
    FS = 20.0  # Hz
    DURATIONS = np.array([10.0, 20.0, 5.0, 15.0, 8.0])  # s per lap

    def _ring(self):
        """Counter-clockwise laps from angle 0, each at its own constant speed."""
        bounds = np.r_[0.0, np.cumsum(self.DURATIONS)]
        times = np.arange(0.0, bounds[-1] + 0.5 / self.FS, 1.0 / self.FS)
        lap = np.clip(np.searchsorted(bounds, times, side="right") - 1, 0, 4)
        theta = 2 * np.pi * (lap + (times - bounds[lap]) / self.DURATIONS[lap])
        positions = np.c_[
            50 + self.RADIUS * np.cos(theta), 50 + self.RADIUS * np.sin(theta)
        ]
        env = Environment.from_samples(positions, bin_size=self.BIN)
        return env, env.bin_at(positions), times, bounds

    def _check(self, laps, bounds):
        assert len(laps) == len(self.DURATIONS)
        speeds = 2 * np.pi * self.RADIUS / self.DURATIONS  # cm/s
        # Each boundary is the sample nearest the start line, so it is off by
        # at most the time to cross one bin, plus one sample.
        for k, lap in enumerate(laps):
            tolerance = self.BIN / speeds[k] + 1.0 / self.FS
            assert lap.start_time == pytest.approx(bounds[k], abs=tolerance)
            end_tolerance = self.BIN / speeds[min(k + 1, 4)] + 1.0 / self.FS
            if k + 1 < len(self.DURATIONS):
                assert lap.end_time == pytest.approx(bounds[k + 1], abs=end_tolerance)
            assert lap.direction == "counter-clockwise"
        assert laps[0].start_time == 0.0
        # Laps tile the recording: each ends where the next begins.
        assert all(a.end_time == b.start_time for a, b in itertools.pairwise(laps))

    def test_auto_laps_at_varying_speed(self):
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times, bounds = self._ring()
        laps = detect_laps(bins, times, env, method="auto")
        self._check(laps, bounds)
        assert laps[0].overlap_score == 1.0  # the first circuit is the template

    def test_reference_lap_from_a_fast_lap_finds_slow_laps(self):
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times, bounds = self._ring()
        fast = bins[(times >= bounds[2]) & (times < bounds[3])]  # the 5 s lap
        # The reference starts a quarter lap later; laps then start there.
        quarter = (times >= bounds[2] + 1.25) & (times < bounds[3])
        rotated = np.r_[bins[quarter], fast[: len(fast) - quarter.sum()]]
        laps = detect_laps(bins, times, env, method="reference", reference_lap=fast)
        self._check(laps, bounds)
        rotated_laps = detect_laps(
            bins, times, env, method="reference", reference_lap=rotated
        )
        # Start line a quarter turn on: four complete laps between passes.
        assert len(rotated_laps) == 4
        assert rotated_laps[0].start_time == pytest.approx(10.0 / 4, abs=0.2)

    def test_turn_back_is_not_a_lap(self):
        """Half a lap out and back through the start line is not a lap."""
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times, _ = self._ring()
        theta = np.r_[np.linspace(0, np.pi, 60), np.linspace(np.pi, -0.5, 60)]
        half = np.c_[50 + self.RADIUS * np.cos(theta), 50 + self.RADIUS * np.sin(theta)]
        laps = detect_laps(
            env.bin_at(half),
            np.arange(theta.size) / self.FS,
            env,
            method="reference",
            reference_lap=bins[times < 10.0],
        )
        assert laps == []

    @pytest.mark.parametrize("min_overlap", [0.0, -0.5, 1.5, np.nan])
    def test_min_overlap_outside_unit_interval_raises(self, min_overlap):
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times, _ = self._ring()
        with pytest.raises(ValueError, match=r"min_overlap must be in \(0, 1\]"):
            detect_laps(bins, times, env, min_overlap=min_overlap)

    def test_unknown_direction_raises(self):
        from neurospatial.behavior.segmentation import detect_laps

        env, bins, times, _ = self._ring()
        with pytest.raises(ValueError, match="direction must be one of"):
            detect_laps(bins, times, env, direction="cw")  # type: ignore[arg-type]
