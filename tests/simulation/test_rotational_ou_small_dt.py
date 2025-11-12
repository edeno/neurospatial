"""Test rotational OU with smaller dt for numerical stability."""

import numpy as np

from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_ou


def test_rotational_ou_small_dt():
    """Test with dt=0.01s for numerical stability."""
    # Create grid environment
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)
    env.units = "cm"

    # Simulate with SMALL dt for stability
    duration = 60.0
    dt = 0.01  # 10ms instead of 100ms - KEY FIX!
    seed = 42

    print(f"\n=== Testing with dt={dt}s ===")
    print(f"theta * dt = (1/0.08) * {dt} = {(1 / 0.08) * dt:.3f}")
    print(f"{'STABLE' if (1 / 0.08) * dt < 1 else 'UNSTABLE'} (need < 1)")

    positions, _ = simulate_trajectory_ou(
        env,
        duration=duration,
        dt=dt,
        speed_mean=7.5,
        rotational_velocity_std=120 * (np.pi / 180),
        rotational_velocity_coherence_time=0.08,
        boundary_mode="reflect",
        seed=seed,
    )

    # Compute diagnostics (skip first 5 seconds)
    start_idx = int(5.0 / dt)
    positions_ss = positions[start_idx:]

    # Since positions use reflect mode, we can compute velocities from positions
    velocities = np.diff(positions_ss, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)

    # Compute heading angles
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_changes = np.diff(headings)
    heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))

    turn_rates = heading_changes / dt * (180 / np.pi)

    print("\nSpeed Statistics (steady state):")
    print(f"  Mean: {speeds.mean():.2f} cm/s (target: 7.5)")
    print(f"  Std: {speeds.std():.3f} cm/s")

    print("\nTurn Rate Statistics:")
    print(f"  Std: {turn_rates.std():.1f} deg/s (expected: ~120)")
    print(f"  Mean |turn rate|: {np.abs(turn_rates).mean():.1f} deg/s")

    print("\nSpatial Coverage:")
    position_range = np.ptp(positions_ss, axis=0)
    print(f"  X range: {position_range[0]:.1f} cm")
    print(f"  Y range: {position_range[1]:.1f} cm")

    # Check region coverage
    y_edges = [0, 16, 32, 48, 64, 80]
    region_counts = np.zeros(5)
    for pos in positions_ss:
        for i in range(5):
            if y_edges[i] <= pos[1] < y_edges[i + 1]:
                region_counts[i] += 1
                break

    region_pcts = region_counts / len(positions_ss) * 100
    print("\nRegion Coverage (Y-axis):")
    for i, pct in enumerate(region_pcts):
        print(f"  Region {i} ({y_edges[i]:.0f}-{y_edges[i + 1]:.0f} cm): {pct:.1f}%")

    balance_ratio = (
        region_pcts.max() / region_pcts.min() if region_pcts.min() > 0 else float("inf")
    )
    print(f"\n  Balance ratio: {balance_ratio:.2f} (ideal: 1.0)")

    # Success criteria
    success = []
    if 80 < turn_rates.std() < 160:
        print("\n✓ Turn rate std is reasonable")
        success.append(True)
    else:
        print("\n✗ Turn rate std is outside expected range")
        success.append(False)

    if balance_ratio < 3.0:
        print("✓ Coverage is relatively uniform (balance < 3.0)")
        success.append(True)
    else:
        print(f"✗ Coverage is non-uniform (balance = {balance_ratio:.2f})")
        success.append(False)

    if all(success):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n✗ Some tests failed")


if __name__ == "__main__":
    test_rotational_ou_small_dt()
