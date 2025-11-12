"""Test rotational OU with periodic boundaries to isolate boundary effects."""

import numpy as np

from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_ou


def test_rotational_ou_periodic_boundaries():
    """Test rotational OU with periodic boundaries (no reflections)."""
    # Create grid environment
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)
    env.units = "cm"

    # Simulate with PERIODIC boundaries (no reflections)
    duration = 60.0  # Longer to reach steady state
    dt = 0.1
    seed = 42

    positions, times = simulate_trajectory_ou(
        env,
        duration=duration,
        dt=dt,
        speed_mean=7.5,
        rotational_velocity_std=120 * (np.pi / 180),  # 120 deg/s
        rotational_velocity_coherence_time=0.08,
        boundary_mode="periodic",  # KEY: No reflections!
        seed=seed,
    )

    # Compute diagnostics (skip first 5 seconds for steady state)
    start_idx = int(5.0 / dt)
    positions_ss = positions[start_idx:]
    _times_ss = times[start_idx:]

    velocities = np.diff(positions_ss, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)

    # Compute heading angles
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_changes = np.diff(headings)
    # Wrap to [-pi, pi]
    heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))

    # Compute turn rates (deg/s)
    turn_rates = heading_changes / dt * (180 / np.pi)

    print("\n=== Rotational OU with Periodic Boundaries (Steady State) ===")
    print(f"Duration: {duration}s, dt: {dt}s, seed: {seed}")
    print(f"Analyzing last {duration - 5:.0f}s (after steady state)")

    print("\nSpeed Statistics:")
    print(f"  Mean speed: {speeds.mean():.2f} cm/s (target: 7.5)")
    print(f"  Speed std: {speeds.std():.2f} cm/s")

    print("\nHeading Statistics:")
    print(f"  Heading std: {headings.std() * (180 / np.pi):.1f} deg")
    print("  (Uniform distribution would be 180/sqrt(12) = 103.9 deg)")

    print("\nTurn Rate Statistics:")
    print(f"  Turn rate std: {turn_rates.std():.1f} deg/s (expected: ~120)")
    print(f"  Mean |turn rate|: {np.abs(turn_rates).mean():.1f} deg/s")
    print(f"  Turn rate range: [{turn_rates.min():.1f}, {turn_rates.max():.1f}] deg/s")

    print("\nSpatial Coverage (with wrapping):")
    # Can't use simple range with periodic boundaries, but check motion
    distances = np.linalg.norm(np.diff(positions_ss, axis=0), axis=1)
    print(f"  Mean step distance: {distances.mean():.2f} cm (expected: {7.5 * dt:.2f})")
    print(
        f"  Total distance: {distances.sum():.1f} cm (expected: {7.5 * (len(positions_ss) - 1) * dt:.1f})"
    )

    # Check if turn rate std matches expectation
    if 80 < turn_rates.std() < 160:
        print("\n✓ Turn rate std is reasonable (within 80-160 deg/s)")
    else:
        print("\n✗ Turn rate std is outside expected range (80-160 deg/s)")


if __name__ == "__main__":
    test_rotational_ou_periodic_boundaries()
