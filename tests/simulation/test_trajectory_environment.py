"""Test that trajectory simulation works with proper environments."""

import numpy as np

from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_ou


def test_ou_trajectory_explores_sparse_environment_fails():
    """Demonstrate that sparse environments trap OU trajectories.

    This test DOCUMENTS the known issue: sparse environments (from just
    corner points) create disconnected bins that trap the OU process.
    """
    # Create SPARSE environment (only corners - WRONG approach)
    arena_size = 80.0
    sparse_data = np.array([[0, 0], [arena_size, arena_size]])  # Only 2 points!
    sparse_env = Environment.from_samples(sparse_data, bin_size=5.0)
    sparse_env.units = "cm"

    # Try to simulate trajectory
    positions, _ = simulate_trajectory_ou(
        sparse_env,
        duration=10.0,
        dt=0.1,
        speed_mean=7.5,
        seed=42,
    )

    # Trajectory gets STUCK - moves less than 1cm over 10 seconds!
    position_range = np.ptp(positions, axis=0)
    assert position_range.max() < 1.0, (
        "Sparse environment traps trajectory (expected behavior to document)"
    )


def test_ou_trajectory_explores_grid_environment_succeeds():
    """Demonstrate that grid environments enable proper OU exploration.

    This test shows the CORRECT approach: create a full grid that covers
    the arena continuously.
    """
    # Create GRID environment (proper approach)
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)  # 17 points = 5cm bins
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])  # 289 grid points

    grid_env = Environment.from_samples(grid_data, bin_size=5.0)
    grid_env.units = "cm"

    # Simulate trajectory
    positions, times = simulate_trajectory_ou(
        grid_env,
        duration=10.0,
        dt=0.1,
        speed_mean=7.5,
        seed=42,
    )

    # Trajectory EXPLORES properly - moves >10cm in each dimension
    position_range = np.ptp(positions, axis=0)
    assert position_range.min() > 10.0, (
        f"Trajectory should explore >10cm, got {position_range}"
    )

    # Mean speed should be close to requested speed_mean
    velocities = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = speeds.mean()

    # Should be within 50% of requested speed (OU process has variability)
    assert 0.5 * 7.5 < mean_speed < 1.5 * 7.5, (
        f"Mean speed {mean_speed:.2f} should be near 7.5 cm/s"
    )


def test_grid_environment_coverage():
    """Verify grid environment has proper coverage and connectivity."""
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)

    # Should have full coverage (17x17 = 289 bins)
    assert env.n_bins == 289, f"Expected 289 bins, got {env.n_bins}"

    # Should cover full arena extent
    ranges = env.dimension_ranges
    assert ranges[0][0] <= 0 and ranges[0][1] >= arena_size
    assert ranges[1][0] <= 0 and ranges[1][1] >= arena_size

    # Check a point in the middle is contained
    middle_point = np.array([[arena_size / 2, arena_size / 2]])
    assert env.contains(middle_point)[0], "Middle point should be in environment"
