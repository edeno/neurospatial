"""Tests for direction label helper functions.

These tests verify the behavior of goal_pair_direction_labels and
heading_direction_labels functions for generating direction labels
from behavioral data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial.behavior.segmentation import Trial
from neurospatial.behavioral import goal_pair_direction_labels, heading_direction_labels

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_times() -> NDArray[np.float64]:
    """Simple timestamps spanning 0-100 seconds at 10 Hz."""
    return np.linspace(0.0, 100.0, 1001)


@pytest.fixture
def basic_trials() -> list[Trial]:
    """Two simple non-overlapping successful trials."""
    return [
        Trial(
            start_time=10.0,
            end_time=20.0,
            start_region="home",
            end_region="goal_left",
            success=True,
        ),
        Trial(
            start_time=30.0,
            end_time=45.0,
            start_region="home",
            end_region="goal_right",
            success=True,
        ),
    ]


@pytest.fixture
def failed_trial() -> Trial:
    """A trial that timed out (no end region reached)."""
    return Trial(
        start_time=50.0,
        end_time=65.0,
        start_region="home",
        end_region=None,  # Failed trial - did not reach end
        success=False,
    )


@pytest.fixture
def overlapping_trials() -> list[Trial]:
    """Two trials with overlapping time ranges (later should overwrite)."""
    return [
        Trial(
            start_time=10.0,
            end_time=30.0,
            start_region="home",
            end_region="goal_left",
            success=True,
        ),
        Trial(
            start_time=20.0,  # Overlaps with first trial
            end_time=40.0,
            start_region="center",
            end_region="goal_right",
            success=True,
        ),
    ]


# -----------------------------------------------------------------------------
# Tests for goal_pair_direction_labels
# -----------------------------------------------------------------------------


class TestGoalPairDirectionLabels:
    """Tests for goal_pair_direction_labels function."""

    def test_basic_trials(
        self,
        simple_times: NDArray[np.float64],
        basic_trials: list[Trial],
    ) -> None:
        """Two trials should produce correct directional labels."""
        labels = goal_pair_direction_labels(simple_times, basic_trials)

        # Check output shape
        assert len(labels) == len(simple_times)
        assert labels.dtype == object

        # Check labels during first trial (10-20s, indices 100-200)
        # Using arrow notation: "home→goal_left"
        first_trial_mask = (simple_times >= 10.0) & (simple_times <= 20.0)
        first_trial_labels = labels[first_trial_mask]
        assert all(label == "home→goal_left" for label in first_trial_labels)

        # Check labels during second trial (30-45s, indices 300-450)
        second_trial_mask = (simple_times >= 30.0) & (simple_times <= 45.0)
        second_trial_labels = labels[second_trial_mask]
        assert all(label == "home→goal_right" for label in second_trial_labels)

    def test_outside_trials_labeled_other(
        self,
        simple_times: NDArray[np.float64],
        basic_trials: list[Trial],
    ) -> None:
        """Timepoints outside all trials should be labeled 'other'."""
        labels = goal_pair_direction_labels(simple_times, basic_trials)

        # Before first trial (0-10s)
        before_mask = simple_times < 10.0
        assert all(label == "other" for label in labels[before_mask])

        # Between trials (20-30s)
        between_mask = (simple_times > 20.0) & (simple_times < 30.0)
        assert all(label == "other" for label in labels[between_mask])

        # After last trial (45-100s)
        after_mask = simple_times > 45.0
        assert all(label == "other" for label in labels[after_mask])

    def test_failed_trial_labeled_other(
        self,
        simple_times: NDArray[np.float64],
        failed_trial: Trial,
    ) -> None:
        """Failed trials (end_region=None) should be labeled 'other'."""
        labels = goal_pair_direction_labels(simple_times, [failed_trial])

        # All timepoints should be "other" since trial failed
        assert all(label == "other" for label in labels)

        # Even during trial time window (50-65s)
        during_trial_mask = (simple_times >= 50.0) & (simple_times <= 65.0)
        assert all(label == "other" for label in labels[during_trial_mask])

    def test_overlapping_trials_later_overwrites(
        self,
        simple_times: NDArray[np.float64],
        overlapping_trials: list[Trial],
    ) -> None:
        """Later trials in the list overwrite earlier ones in overlapping regions."""
        labels = goal_pair_direction_labels(simple_times, overlapping_trials)

        # First trial only region (10-20s)
        first_only_mask = (simple_times >= 10.0) & (simple_times < 20.0)
        assert all(label == "home→goal_left" for label in labels[first_only_mask])

        # Overlap region (20-30s) should have second trial's label
        overlap_mask = (simple_times >= 20.0) & (simple_times <= 30.0)
        assert all(label == "center→goal_right" for label in labels[overlap_mask])

        # Second trial only region (30-40s)
        second_only_mask = (simple_times > 30.0) & (simple_times <= 40.0)
        assert all(label == "center→goal_right" for label in labels[second_only_mask])

    def test_empty_trials_list(
        self,
        simple_times: NDArray[np.float64],
    ) -> None:
        """Empty trials list should return all 'other' labels."""
        labels = goal_pair_direction_labels(simple_times, [])

        assert len(labels) == len(simple_times)
        assert all(label == "other" for label in labels)

    def test_empty_times_array(self) -> None:
        """Empty times array should return empty labels array."""
        times = np.array([], dtype=np.float64)
        trials = [
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="home",
                end_region="goal",
                success=True,
            )
        ]

        labels = goal_pair_direction_labels(times, trials)

        assert len(labels) == 0
        assert labels.dtype == object

    def test_label_format_uses_arrow(
        self,
        simple_times: NDArray[np.float64],
    ) -> None:
        """Labels should use arrow notation: 'start→end'."""
        trials = [
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="A",
                end_region="B",
                success=True,
            ),
        ]

        labels = goal_pair_direction_labels(simple_times, trials)

        # Check arrow notation
        trial_mask = (simple_times >= 10.0) & (simple_times <= 20.0)
        expected_label = "A→B"
        assert all(label == expected_label for label in labels[trial_mask])

    def test_boundary_inclusive(
        self,
    ) -> None:
        """Boundary timepoints should be included in the trial."""
        # Exact timestamps for boundaries
        times = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        trials = [
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="home",
                end_region="goal",
                success=True,
            ),
        ]

        labels = goal_pair_direction_labels(times, trials)

        # Boundaries should be included
        assert labels[0] == "other"  # 5.0 - before
        assert labels[1] == "home→goal"  # 10.0 - at start (inclusive)
        assert labels[2] == "home→goal"  # 15.0 - during
        assert labels[3] == "home→goal"  # 20.0 - at end (inclusive)
        assert labels[4] == "other"  # 25.0 - after

    def test_multiple_region_names(
        self,
        simple_times: NDArray[np.float64],
    ) -> None:
        """Different region names produce different labels."""
        trials = [
            Trial(
                start_time=10.0,
                end_time=20.0,
                start_region="center",
                end_region="arm1",
                success=True,
            ),
            Trial(
                start_time=30.0,
                end_time=40.0,
                start_region="center",
                end_region="arm2",
                success=True,
            ),
        ]

        labels = goal_pair_direction_labels(simple_times, trials)

        # Unique labels
        unique_labels = set(labels)
        assert "center→arm1" in unique_labels
        assert "center→arm2" in unique_labels
        assert "other" in unique_labels


# -----------------------------------------------------------------------------
# Tests for heading_direction_labels
# -----------------------------------------------------------------------------


class TestHeadingDirectionLabels:
    """Tests for heading_direction_labels function."""

    def test_straight_path_positive_x(self) -> None:
        """Movement in +x direction gives consistent heading label."""
        # Trajectory moving steadily in +x direction
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.linspace(0.0, 100.0, n_samples), np.zeros(n_samples)]
        )

        labels = heading_direction_labels(positions=positions, times=times)

        # All moving samples should have the same direction label (near 0°)
        # First sample is stationary (padded speed=0), rest should move
        moving_labels = labels[1:]  # Skip first (may be stationary due to padding)

        # Filter out any "stationary" labels that might exist
        direction_labels = [lbl for lbl in moving_labels if lbl != "stationary"]

        # All direction labels should be the same (0° heading bin)
        assert len(set(direction_labels)) == 1, (
            f"Expected single direction label for straight +x path, got {set(direction_labels)}"
        )
        # The label should contain "0" since heading is 0° (positive x)
        assert "0" in direction_labels[0]

    def test_straight_path_positive_y(self) -> None:
        """Movement in +y direction gives 90° heading label."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.zeros(n_samples), np.linspace(0.0, 100.0, n_samples)]
        )

        labels = heading_direction_labels(positions=positions, times=times)

        moving_labels = [lbl for lbl in labels[1:] if lbl != "stationary"]
        assert len(set(moving_labels)) == 1
        # The label should contain "90" since heading is 90° (positive y)
        assert "90" in moving_labels[0]

    def test_stationary_labeled_correctly(self) -> None:
        """Low speed periods are labeled 'stationary'."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        # Stationary: small random jitter around origin (well below min_speed=5.0)
        np.random.seed(42)
        positions = np.random.uniform(-0.1, 0.1, (n_samples, 2))

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        # All labels should be "stationary"
        assert all(label == "stationary" for label in labels)

    def test_min_speed_threshold(self) -> None:
        """Verify min_speed threshold is respected."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        # Speed of exactly 4.0 cm/s in +x direction (dt = 0.1s, so dx = 0.4 per sample)
        x_positions = np.linspace(0.0, 4.0, n_samples)  # 4 cm total over 10s = 0.4 cm/s
        positions = np.column_stack([x_positions, np.zeros(n_samples)])

        # With min_speed=5.0, these should all be stationary
        labels_high_thresh = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )
        # With min_speed=0.1, these should have direction labels
        labels_low_thresh = heading_direction_labels(
            positions=positions, times=times, min_speed=0.1
        )

        # High threshold: mostly stationary
        assert sum(1 for lbl in labels_high_thresh if lbl == "stationary") > 90

        # Low threshold: mostly directions
        assert sum(1 for lbl in labels_low_thresh if lbl != "stationary") > 50

    def test_precomputed_matches_computed(self) -> None:
        """Precomputed speed/heading produces same result as positions/times."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.linspace(0.0, 100.0, n_samples), np.linspace(0.0, 50.0, n_samples)]
        )

        # Compute speed and heading manually
        velocity = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]
        speed_computed = np.linalg.norm(velocity, axis=1)
        heading_computed = np.arctan2(velocity[:, 1], velocity[:, 0])
        # Pad first element
        speed = np.concatenate([[0], speed_computed])
        heading = np.concatenate([[0], heading_computed])

        # Labels from positions/times
        labels_from_positions = heading_direction_labels(
            positions=positions, times=times
        )

        # Labels from precomputed speed/heading
        labels_from_precomputed = heading_direction_labels(speed=speed, heading=heading)

        np.testing.assert_array_equal(labels_from_positions, labels_from_precomputed)

    def test_precomputed_takes_precedence(self) -> None:
        """When both provided, precomputed speed/heading takes precedence."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)

        # Positions suggest +x direction
        positions = np.column_stack(
            [np.linspace(0.0, 100.0, n_samples), np.zeros(n_samples)]
        )

        # But precomputed heading is 90° (+y direction)
        speed = np.full(n_samples, 10.0)
        heading = np.full(n_samples, np.pi / 2)  # 90 degrees

        labels = heading_direction_labels(
            positions=positions, times=times, speed=speed, heading=heading
        )

        # Should use precomputed, so labels should be 90° not 0°
        direction_labels = [lbl for lbl in labels if lbl != "stationary"]
        assert len(set(direction_labels)) == 1
        assert "90" in direction_labels[0]

    def test_error_no_inputs(self) -> None:
        """Raises ValueError if neither (positions, times) nor (speed, heading) provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            heading_direction_labels()

    def test_error_incomplete_positions(self) -> None:
        """Raises ValueError if positions provided without times."""
        positions = np.random.rand(100, 2)
        with pytest.raises(ValueError, match="positions and times"):
            heading_direction_labels(positions=positions)

    def test_error_incomplete_precomputed(self) -> None:
        """Raises ValueError if speed provided without heading."""
        speed = np.random.rand(100)
        with pytest.raises(ValueError, match="speed and heading"):
            heading_direction_labels(speed=speed)

    def test_error_mismatched_lengths(self) -> None:
        """Raises ValueError if speed and heading have different lengths."""
        speed = np.random.rand(100)
        heading = np.random.rand(50)  # Different length
        with pytest.raises(ValueError, match="same length"):
            heading_direction_labels(speed=speed, heading=heading)

    def test_n_directions_default_8(self) -> None:
        """Default n_directions=8 produces 8 bins (45° each)."""
        n_samples = 800
        times = np.linspace(0.0, 80.0, n_samples)

        # Create circular trajectory to sample all directions
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        # Large radius so speed is high
        radius = 100.0
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        labels = heading_direction_labels(positions=positions, times=times)

        # Get unique non-stationary labels
        direction_labels = {lbl for lbl in labels if lbl != "stationary"}

        # Should have 8 unique direction bins
        assert len(direction_labels) == 8

    def test_n_directions_custom(self) -> None:
        """Custom n_directions produces correct number of bins."""
        n_samples = 400
        times = np.linspace(0.0, 40.0, n_samples)

        # Create circular trajectory
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        radius = 100.0
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        # Test with n_directions=4 (90° bins)
        labels_4 = heading_direction_labels(
            positions=positions, times=times, n_directions=4
        )
        direction_labels_4 = {lbl for lbl in labels_4 if lbl != "stationary"}
        assert len(direction_labels_4) == 4

        # Test with n_directions=16 (22.5° bins)
        labels_16 = heading_direction_labels(
            positions=positions, times=times, n_directions=16
        )
        direction_labels_16 = {lbl for lbl in labels_16 if lbl != "stationary"}
        assert len(direction_labels_16) == 16

    def test_bin_boundaries_at_edges(self) -> None:
        """Verify bin boundaries are correct at exact angles."""
        n_samples = 100

        # Test exact 45° heading - should fall in 45-90° bin (with 8 bins)
        speed = np.full(n_samples, 10.0)
        heading_45 = np.full(n_samples, np.pi / 4)  # 45°

        labels = heading_direction_labels(
            speed=speed, heading=heading_45, n_directions=8
        )

        # 45° should be exactly at boundary between 0-45° and 45-90°
        # Depending on implementation (left-inclusive), it should be in one or the other
        direction_labels = {lbl for lbl in labels if lbl != "stationary"}
        assert len(direction_labels) == 1
        # Should be either "0–45°" or "45–90°" (boundary case)
        label = next(iter(direction_labels))
        assert "45" in label

    def test_negative_angles_handled(self) -> None:
        """Negative angles (−π to π) are correctly mapped to labels."""
        n_samples = 100
        speed = np.full(n_samples, 10.0)
        heading_neg = np.full(n_samples, -np.pi / 2)  # -90° = 270°

        labels = heading_direction_labels(
            speed=speed, heading=heading_neg, n_directions=8
        )

        direction_labels = {lbl for lbl in labels if lbl != "stationary"}
        assert len(direction_labels) == 1
        # -90° should map to bin around -90° or 270°
        label = next(iter(direction_labels))
        # The bin should indicate negative angle or wrapped positive
        assert "-90" in label or "270" in label

    def test_output_shape_matches_input(self) -> None:
        """Output array length matches input array length."""
        n_samples = 123  # Non-round number
        times = np.linspace(0.0, 12.3, n_samples)
        positions = np.random.rand(n_samples, 2) * 100.0

        labels = heading_direction_labels(positions=positions, times=times)

        assert len(labels) == n_samples
        assert labels.dtype == object

    def test_output_dtype_is_object(self) -> None:
        """Output array has dtype=object for string labels."""
        n_samples = 50
        speed = np.full(n_samples, 10.0)
        heading = np.zeros(n_samples)

        labels = heading_direction_labels(speed=speed, heading=heading)

        assert labels.dtype == object

    def test_empty_arrays(self) -> None:
        """Empty input arrays return empty output array."""
        times = np.array([], dtype=np.float64)
        positions = np.zeros((0, 2), dtype=np.float64)

        labels = heading_direction_labels(positions=positions, times=times)

        assert len(labels) == 0
        assert labels.dtype == object

    def test_single_timepoint(self) -> None:
        """Single timepoint returns 'stationary' (no velocity can be computed)."""
        times = np.array([0.0])
        positions = np.array([[10.0, 20.0]])

        labels = heading_direction_labels(positions=positions, times=times)

        assert len(labels) == 1
        assert labels[0] == "stationary"  # Can't compute velocity from single point
