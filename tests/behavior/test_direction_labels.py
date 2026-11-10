"""Tests for direction label helper functions.

These tests verify the behavior of heading_direction_labels and
headings_to_direction_labels for generating direction labels from behavioral
data. Labels from laps and trials are in test_direction_label_bridges.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.behavior import heading_from_velocity
from neurospatial.behavior.navigation import (
    heading_direction_labels,
    headings_to_direction_labels,
)
from neurospatial.behavior.trajectory import compute_speed

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _sector_labels(headings, n_directions):
    """Label of the half-open sector [c - w/2, c + w/2) holding each heading.

    Sectors are centred on multiples of w = 360° / n_directions, named by
    their centre in degrees within (-180°, 180°].
    """
    width = 2 * np.pi / n_directions
    index = np.floor(np.asarray(headings) / width + 0.5).astype(int) % n_directions
    degrees = index * 360.0 / n_directions
    degrees = np.where(degrees > 180.0, degrees - 360.0, degrees)
    return [f"{d:g}°" for d in degrees]


def _circling(n_samples, duration):
    """Counter-clockwise circling of radius 100 cm, one revolution.

    The angles are offset by 0.3 of a sample so that no sample's direction
    of movement lies on a sector edge (with 400 or 800 samples and 4, 8 or
    16 sectors, an offset of half a sample would put some exactly on one).
    Away from the two ends the Gaussian line
    fit is symmetric, so the direction is exactly the tangent, angle + 90°;
    at the ends it lags by under 0.01 rad, near 90°, a sector centre.
    """
    times = np.linspace(0.0, duration, n_samples)
    angles = 2 * np.pi * (np.arange(n_samples) + 0.3) / n_samples
    positions = 100.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    return times, positions, angles + np.pi / 2


@pytest.fixture
def rng():
    """Create a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


# -----------------------------------------------------------------------------
# Tests for heading_direction_labels and headings_to_direction_labels
# -----------------------------------------------------------------------------


class TestHeadingDirectionLabels:
    """Tests for heading_direction_labels and headings_to_direction_labels."""

    def test_straight_path_positive_x(self) -> None:
        """Movement in +x direction gives consistent heading label."""
        # Trajectory moving steadily in +x direction
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.linspace(0.0, 100.0, n_samples), np.zeros(n_samples)]
        )

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        # All moving samples should have the same direction label (near 0°)
        # First sample is stationary (padded speed=0), rest should move
        moving_labels = labels[1:]  # Skip first (may be stationary due to padding)

        # Filter out any "stationary" labels that might exist
        direction_labels = [lbl for lbl in moving_labels if lbl != "stationary"]

        # East (+x) is the sector centred on 0°.
        assert set(direction_labels) == {"0°"}

    def test_first_sample_undefined_heading_not_labeled_east(self) -> None:
        """The first sample is labeled by its own motion, never a padded East.

        The first sample's velocity is its forward difference. Old code padded
        ``heading=0``, which with ``min_speed=0`` spuriously labeled it as East
        (the 0 deg sector).
        """
        n_samples = 50
        times = np.linspace(0.0, 5.0, n_samples)
        # Steady movement in the +y direction (true heading ~90 deg), so the
        # genuine direction label for moving samples is the 90 deg sector, NOT
        # the 0 deg (East) sector.
        positions = np.column_stack(
            [np.zeros(n_samples), np.linspace(0.0, 100.0, n_samples)]
        )

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=0.0
        )

        # The undefined first sample must not be assigned a spurious East label.
        # Every sample, the first included, reflects the +y (90 deg) heading.
        assert set(labels) == {"90°"}, labels

    def test_straight_path_positive_y(self) -> None:
        """Movement in +y direction gives 90° heading label."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.zeros(n_samples), np.linspace(0.0, 100.0, n_samples)]
        )

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        moving_labels = [lbl for lbl in labels[1:] if lbl != "stationary"]
        # North (+y) is the sector centred on 90°.
        assert set(moving_labels) == {"90°"}

    def test_stationary_labeled_correctly(self, rng) -> None:
        """Low speed periods are labeled 'stationary'."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        # Stationary: small random jitter around origin (well below min_speed=5.0)
        positions = rng.uniform(-0.1, 0.1, (n_samples, 2))

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        # All labels should be "stationary"
        assert all(label == "stationary" for label in labels)

    def test_min_min_speed(self) -> None:
        """A constant 0.4 cm/s run East is moving above 0.4 cm/s, still below.

        4 cm in 10 s: the line fit's speed is exactly 0.4 cm/s at every
        sample, so the gate (moving at speed >= min_speed) labels every
        sample "0°" at or below 0.39 cm/s and "stationary" at or above
        0.41 cm/s.
        """
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        x_positions = np.linspace(0.0, 4.0, n_samples)
        positions = np.column_stack([x_positions, np.zeros(n_samples)])

        for min_speed in (0.1, 0.39):
            labels = heading_direction_labels(
                positions=positions, times=times, min_speed=min_speed
            )
            assert labels.tolist() == ["0°"] * n_samples
        for min_speed in (0.41, 5.0):
            labels = heading_direction_labels(
                positions=positions, times=times, min_speed=min_speed
            )
            assert labels.tolist() == ["stationary"] * n_samples

    def test_precomputed_matches_computed(self) -> None:
        """Precomputed speed/heading produces same result as positions/times."""
        n_samples = 100
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [np.linspace(0.0, 100.0, n_samples), np.linspace(0.0, 50.0, n_samples)]
        )

        # The same smoothed speed and heading the trajectory path derives
        speed = compute_speed(times, positions)
        heading = heading_from_velocity(times, positions)

        # Labels from positions/times
        labels_from_positions = heading_direction_labels(
            positions=positions,
            times=times,
            min_speed=5.0,
        )

        # Labels from precomputed speed/heading
        labels_from_precomputed = headings_to_direction_labels(
            heading, speed, min_speed=5.0
        )

        np.testing.assert_array_equal(labels_from_positions, labels_from_precomputed)

    def test_error_mismatched_lengths(self, rng) -> None:
        """Raises ValueError if headings and speeds have different lengths."""
        speeds = rng.random(100)
        headings = rng.random(50)  # Different length
        with pytest.raises(ValueError, match="50 headings and 100 speeds"):
            headings_to_direction_labels(headings, speeds, min_speed=5.0)

    def test_n_directions_default_8(self) -> None:
        """Circling labels each sample by the 45° sector of its tangent."""
        times, positions, tangent = _circling(800, 80.0)  # 7.9 cm/s

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        assert labels.tolist() == _sector_labels(tangent, 8)
        assert len(set(labels.tolist())) == 8

    def test_n_directions_custom(self) -> None:
        """Circling labels each sample by its tangent's 90° or 22.5° sector."""
        times, positions, tangent = _circling(400, 40.0)

        # Test with n_directions=4 (90° bins)
        labels_4 = heading_direction_labels(
            positions=positions,
            times=times,
            n_directions=4,
            min_speed=5.0,
        )
        assert labels_4.tolist() == _sector_labels(tangent, 4)
        assert len(set(labels_4.tolist())) == 4

        # Test with n_directions=16 (22.5° bins)
        labels_16 = heading_direction_labels(
            positions=positions,
            times=times,
            n_directions=16,
            min_speed=5.0,
        )
        assert labels_16.tolist() == _sector_labels(tangent, 16)
        assert len(set(labels_16.tolist())) == 16

    def test_bin_boundaries_at_edges(self) -> None:
        """Verify bin boundaries are correct at exact angles."""
        n_samples = 100

        # Sectors are centred on multiples of 45° and half-open
        # [centre - 22.5°, centre + 22.5°): 45° is a centre, 22.5° is the
        # clockwise edge of the 45° sector.
        speed = np.full(n_samples, 10.0)
        for heading, expected in [(np.pi / 4, "45°"), (np.pi / 8, "45°")]:
            labels = headings_to_direction_labels(
                np.full(n_samples, heading),
                speed,
                n_directions=8,
                min_speed=5.0,
            )
            assert set(labels) == {expected}

    def test_negative_angles_handled(self) -> None:
        """Negative angles (−π to π) are correctly mapped to labels."""
        n_samples = 100
        speed = np.full(n_samples, 10.0)
        heading_neg = np.full(n_samples, -np.pi / 2)  # -90° = 270°

        labels = headings_to_direction_labels(
            heading_neg,
            speed,
            n_directions=8,
            min_speed=5.0,
        )

        # South is the sector centred on -90°.
        assert set(labels) == {"-90°"}

    def test_output_shape_matches_input(self, rng) -> None:
        """Output array length matches input array length."""
        n_samples = 123  # Non-round number
        times = np.linspace(0.0, 12.3, n_samples)
        positions = rng.random((n_samples, 2)) * 100.0

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        assert len(labels) == n_samples
        assert labels.dtype == object

    def test_empty_arrays(self) -> None:
        """Empty trajectory input raises: there is no observation to label."""
        times = np.array([], dtype=np.float64)
        positions = np.zeros((0, 2), dtype=np.float64)

        with pytest.raises(ValueError, match="heading_direction_labels"):
            heading_direction_labels(positions=positions, times=times, min_speed=5.0)

    def test_single_timepoint(self) -> None:
        """A single timepoint has no observed movement, so its label is None."""
        times = np.array([0.0])
        positions = np.array([[10.0, 20.0]])

        labels = heading_direction_labels(
            positions=positions, times=times, min_speed=5.0
        )

        assert labels.dtype == object
        assert labels.tolist() == [None]
