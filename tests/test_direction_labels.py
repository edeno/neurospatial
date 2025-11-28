"""Tests for direction label helper functions.

These tests verify the behavior of goal_pair_direction_labels and
heading_direction_labels functions for generating direction labels
from behavioral data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial.behavioral import goal_pair_direction_labels
from neurospatial.segmentation import Trial

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
