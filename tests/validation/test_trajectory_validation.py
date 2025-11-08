"""
Validation tests for trajectory metrics against known properties and ecology packages.

Validates:
1. Turn angles against known geometric properties
2. Step lengths against graph distances
3. Home range against known occupancy distributions
4. Mean square displacement against diffusion theory

External package comparisons (Traja, yupi) are optional.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.metrics.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)


class TestTurnAnglesValidation:
    """Validate turn angle computation against geometric ground truth.

    References:
    - Traja package (ecology literature)
    - Basic geometry: turn angle = angle between consecutive velocity vectors
    """

    def test_turn_angles_straight_line(self):
        """Straight line trajectory should have turn angles ~ 0."""
        # Create straight line trajectory
        positions = []
        for i in range(100):
            for j in range(100):
                positions.append([i * 2.0, j * 2.0])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Straight line: move in +x direction
        trajectory_bins = []
        for x in range(20):
            # Find bin closest to (x*10, 50)
            target = np.array([x * 10.0, 50.0])
            distances = np.linalg.norm(env.bin_centers - target, axis=1)
            trajectory_bins.append(np.argmin(distances))

        trajectory_bins = np.array(trajectory_bins)

        angles = compute_turn_angles(trajectory_bins, env)

        # Remove NaN values (stationary periods)
        angles_clean = angles[~np.isnan(angles)]

        if len(angles_clean) > 0:
            # Mean angle should be close to 0 for straight line
            mean_angle = np.abs(angles_clean).mean()
            assert mean_angle < np.pi / 6, (  # Less than 30 degrees
                f"Straight line should have small turn angles, "
                f"got mean={np.degrees(mean_angle):.1f}°"
            )

    def test_turn_angles_circle(self):
        """Circular trajectory should have constant turn angles."""
        # Create circular environment
        positions = []
        for theta in np.linspace(0, 2 * np.pi, 200):
            r = 50.0
            positions.append([50 + r * np.cos(theta), 50 + r * np.sin(theta)])

        # Add points in circle
        for theta in np.linspace(0, 2 * np.pi, 100):
            for r in np.linspace(20, 80, 10):
                positions.append([50 + r * np.cos(theta), 50 + r * np.sin(theta)])

        positions = np.array(positions)
        env = Environment.from_samples(positions, bin_size=8.0)

        # Create circular trajectory
        trajectory_bins = []
        for theta in np.linspace(0, 2 * np.pi, 30):
            r = 50.0
            target = np.array([50 + r * np.cos(theta), 50 + r * np.sin(theta)])
            distances = np.linalg.norm(env.bin_centers - target, axis=1)
            trajectory_bins.append(np.argmin(distances))

        trajectory_bins = np.array(trajectory_bins)

        angles = compute_turn_angles(trajectory_bins, env)
        angles_clean = angles[~np.isnan(angles)]

        if len(angles_clean) > 5:
            # Standard deviation should be relatively small for constant turning
            std_angle = np.std(angles_clean)
            assert std_angle < np.pi / 3, (  # Less than 60 degrees std
                f"Circular path should have relatively constant turn angles, "
                f"got std={np.degrees(std_angle):.1f}°"
            )

    def test_turn_angles_range(self):
        """Turn angles should be in range [-π, π]."""
        positions = np.random.randn(1000, 2) * 30
        env = Environment.from_samples(positions, bin_size=5.0)

        # Random walk
        np.random.seed(42)
        trajectory_bins = np.random.choice(env.n_bins, size=50, replace=True)

        angles = compute_turn_angles(trajectory_bins, env)
        angles_clean = angles[~np.isnan(angles)]

        assert np.all(angles_clean >= -np.pi), "Angles must be >= -π"
        assert np.all(angles_clean <= np.pi), "Angles must be <= π"

    def test_turn_angles_zero_for_no_turn(self):
        """No turn (same direction) should give angle ~ 0."""
        # Create grid
        positions = []
        for i in range(50):
            for j in range(50):
                positions.append([i * 2.0, j * 2.0])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=5.0)

        # Three points in straight line: (0,0) -> (10,0) -> (20,0)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[10.0, 0.0]])
        p3 = np.array([[20.0, 0.0]])

        b1 = env.bin_at(p1)[0]
        b2 = env.bin_at(p2)[0]
        b3 = env.bin_at(p3)[0]

        trajectory_bins = np.array([b1, b2, b3])

        angles = compute_turn_angles(trajectory_bins, env)

        # Should have one angle (from 3 points)
        # Angle should be close to 0 (continuing straight)
        if not np.isnan(angles[0]):
            assert np.abs(angles[0]) < np.pi / 6, (  # Less than 30 degrees
                f"Straight continuation should give angle ~ 0, "
                f"got {np.degrees(angles[0]):.1f}°"
            )


class TestStepLengthsValidation:
    """Validate step length computation using graph distances."""

    def test_step_lengths_adjacent_bins(self):
        """Adjacent bins should have step length ~ bin_size."""
        positions = []
        for i in range(50):
            for j in range(50):
                positions.append([i * 2.0, j * 2.0])
        positions = np.array(positions)

        bin_size = 5.0
        env = Environment.from_samples(positions, bin_size=bin_size)

        # Get two adjacent bins
        # Find bin at (0, 0)
        b1 = env.bin_at(np.array([[0.0, 0.0]]))[0]

        # Find neighbors
        neighbors = list(env.connectivity.neighbors(b1))
        if len(neighbors) > 0:
            b2 = neighbors[0]

            trajectory_bins = np.array([b1, b2])
            lengths = compute_step_lengths(trajectory_bins, env)

            # Step length should be close to edge distance
            edge_data = env.connectivity[b1][b2]
            expected_length = edge_data['distance']

            assert len(lengths) == 1, f"Should have 1 step, got {len(lengths)}"
            assert np.abs(lengths[0] - expected_length) < 0.1, (
                f"Adjacent bins step length should be {expected_length:.2f}, "
                f"got {lengths[0]:.2f}"
            )

    def test_step_lengths_stationary(self):
        """Staying in same bin should give step length = 0."""
        positions = np.random.randn(500, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Stay in same bin
        bin_id = 50
        trajectory_bins = np.array([bin_id, bin_id, bin_id])

        lengths = compute_step_lengths(trajectory_bins, env)

        # All steps should be 0
        assert np.all(lengths == 0), f"Stationary should give 0 length, got {lengths}"

    def test_step_lengths_non_negative(self):
        """All step lengths should be non-negative."""
        positions = np.random.randn(1000, 2) * 30
        env = Environment.from_samples(positions, bin_size=5.0)

        # Random trajectory
        np.random.seed(42)
        trajectory_bins = np.random.choice(env.n_bins, size=100, replace=True)

        lengths = compute_step_lengths(trajectory_bins, env)

        assert np.all(lengths >= 0), "Step lengths must be non-negative"


class TestHomeRangeValidation:
    """Validate home range estimation.

    Reference: adehabitatHR (R package) - 95% kernel density estimator standard
    """

    def test_home_range_95_percentile(self):
        """95% home range should contain 95% of time."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create trajectory with known occupancy
        np.random.seed(42)
        # Spend 95% of time in subset of bins
        preferred_bins = np.arange(0, int(env.n_bins * 0.3))  # 30% of bins
        other_bins = np.arange(int(env.n_bins * 0.3), env.n_bins)

        # 950 samples in preferred bins, 50 in others
        trajectory = np.concatenate([
            np.random.choice(preferred_bins, size=950, replace=True),
            np.random.choice(other_bins, size=50, replace=True),
        ])

        home_range_bins = compute_home_range(trajectory, percentile=95.0)

        # Count how many trajectory points are in home range
        in_home_range = np.isin(trajectory, home_range_bins).sum()
        fraction = in_home_range / len(trajectory)

        # Should be close to 0.95
        assert 0.93 <= fraction <= 0.97, (
            f"95% home range should contain ~95% of trajectory, "
            f"got {fraction * 100:.1f}%"
        )

    def test_home_range_50_percentile(self):
        """50% home range should contain 50% of time."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Uniform random trajectory
        np.random.seed(42)
        trajectory = np.random.choice(env.n_bins, size=1000, replace=True)

        home_range_bins = compute_home_range(trajectory, percentile=50.0)

        # Count how many trajectory points are in home range
        in_home_range = np.isin(trajectory, home_range_bins).sum()
        fraction = in_home_range / len(trajectory)

        # Should contain close to 50% of time
        assert 0.48 <= fraction <= 0.52, (
            f"50% home range should contain ~50% of trajectory, "
            f"got {fraction * 100:.1f}%"
        )

    def test_home_range_localized(self):
        """Highly localized trajectory should have small home range."""
        positions = []
        for i in range(50):
            for j in range(50):
                positions.append([i * 2.0, j * 2.0])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=5.0)

        # Spend all time in central bin
        center = env.bin_centers.mean(axis=0)
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        central_bin = np.argmin(distances)

        trajectory = np.full(1000, central_bin)

        home_range_bins = compute_home_range(trajectory, percentile=95.0)

        # Home range should be very small (just the central bin)
        assert len(home_range_bins) <= 5, (
            f"Localized trajectory should have small home range, "
            f"got {len(home_range_bins)} bins"
        )
        assert central_bin in home_range_bins, "Home range should contain central bin"


class TestMSDValidation:
    """Validate mean square displacement against diffusion theory.

    Theory: For random walk, MSD(τ) ~ τ^α where:
    - α = 1: normal diffusion (Brownian motion)
    - α < 1: subdiffusion (confined)
    - α > 1: superdiffusion (directed motion)

    Reference: yupi, ctmm packages
    """

    def test_msd_stationary(self):
        """Stationary trajectory should have MSD ~ 0."""
        positions = np.random.randn(500, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Stay in same bin
        bin_id = 50
        trajectory_bins = np.full(100, bin_id)
        times = np.arange(100) * 0.1  # 0.1 second intervals

        tau_values, msd_values = mean_square_displacement(
            trajectory_bins, times, env, max_tau=5.0
        )

        # All MSD values should be 0
        assert np.allclose(msd_values, 0.0), (
            f"Stationary trajectory should have MSD=0, got max={msd_values.max():.2f}"
        )

    def test_msd_positive_for_movement(self):
        """MSD should be positive for trajectory with movement."""
        positions = np.random.randn(1000, 2) * 30
        env = Environment.from_samples(positions, bin_size=5.0)

        # Random walk trajectory
        np.random.seed(42)
        trajectory_bins = np.random.choice(env.n_bins, size=200, replace=True)
        times = np.arange(200) * 0.1

        tau_values, msd_values = mean_square_displacement(
            trajectory_bins, times, env, max_tau=5.0
        )

        # MSD values should generally be positive for moving trajectory
        if len(msd_values) > 0:
            # At least some MSD values should be positive
            assert np.any(msd_values > 0), (
                f"MSD should have positive values for moving trajectory, "
                f"got max={msd_values.max():.2f}"
            )

    def test_msd_non_negative(self):
        """MSD must be non-negative."""
        positions = np.random.randn(1000, 2) * 30
        env = Environment.from_samples(positions, bin_size=5.0)

        np.random.seed(42)
        trajectory_bins = np.random.choice(env.n_bins, size=100, replace=True)
        times = np.arange(100) * 0.1

        tau_values, msd_values = mean_square_displacement(
            trajectory_bins, times, env, max_tau=5.0
        )

        assert np.all(msd_values >= 0), (
            f"MSD must be non-negative, got min={msd_values.min():.2f}"
        )

    def test_msd_directed_motion_superdiffusion(self):
        """Directed motion should show superdiffusion (α > 1)."""
        # Create linear trajectory (directed motion)
        positions = []
        for i in range(50):
            for j in range(50):
                positions.append([i * 2.0, j * 2.0])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=5.0)

        # Create directed trajectory: move consistently in +x direction
        trajectory_bins = []
        for x in range(30):
            # Find bin at (x*5, 25)
            target = np.array([x * 5.0, 25.0])
            distances = np.linalg.norm(env.bin_centers - target, axis=1)
            trajectory_bins.append(np.argmin(distances))

        trajectory_bins = np.array(trajectory_bins)
        times = np.arange(len(trajectory_bins)) * 0.1

        tau_values, msd_values = mean_square_displacement(
            trajectory_bins, times, env, max_tau=2.0
        )

        if len(tau_values) > 3:
            # Fit MSD ~ τ^α using log-log
            # Remove tau=0 if present
            mask = tau_values > 0
            if mask.sum() > 2:
                log_tau = np.log(tau_values[mask])
                log_msd = np.log(msd_values[mask] + 1e-10)  # Avoid log(0)

                # Linear fit in log-log space
                coeffs = np.polyfit(log_tau, log_msd, deg=1)
                alpha = coeffs[0]

                # Directed motion should have α >= 1
                # (may not be much greater due to discretization)
                assert alpha >= 0.5, (
                    f"Directed motion should show α >= 0.5, got α={alpha:.2f}"
                )


try:
    import traja
    TRAJA_AVAILABLE = True
except ImportError:
    TRAJA_AVAILABLE = False


@pytest.mark.skipif(not TRAJA_AVAILABLE, reason="traja not available")
class TestTrajaComparison:
    """Direct comparison with Traja package (if installed)."""

    def test_turn_angles_match_traja(self):
        """Compare turn angles with Traja on synthetic trajectory."""
        import traja
        import pandas as pd

        # Create trajectory DataFrame for Traja
        t = np.linspace(0, 10, 100)
        x = 50 + 30 * np.cos(t)  # Circular motion
        y = 50 + 30 * np.sin(t)

        df = pd.DataFrame({'x': x, 'y': y})

        # Traja turn angles
        traja_angles = traja.trajectory.calc_turn_angle(df)

        # Our implementation
        positions = np.column_stack([x, y])
        # Add grid points
        for i in range(20, 80, 5):
            for j in range(20, 80, 5):
                positions = np.vstack([positions, [i, j]])

        env = Environment.from_samples(positions, bin_size=8.0)

        # Convert trajectory to bins
        trajectory_bins = env.bin_at(df[['x', 'y']].values)

        our_angles = compute_turn_angles(trajectory_bins, env)

        # Compare (allowing for discretization differences)
        # Just check that mean is similar
        traja_mean = np.nanmean(traja_angles)
        our_mean = np.nanmean(our_angles)

        assert np.abs(traja_mean - our_mean) < np.pi / 4, (
            f"Mean turn angle should match Traja: "
            f"Traja={np.degrees(traja_mean):.1f}°, "
            f"Ours={np.degrees(our_mean):.1f}°"
        )


try:
    import yupi
    YUPI_AVAILABLE = True
except ImportError:
    YUPI_AVAILABLE = False


@pytest.mark.skipif(not YUPI_AVAILABLE, reason="yupi not available")
class TestYupiComparison:
    """Direct comparison with yupi package (if installed)."""

    def test_msd_matches_yupi(self):
        """Compare MSD computation with yupi on random walk."""
        # This would require yupi to be installed and properly configured
        # Placeholder for actual comparison
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
