"""Diagnostic tests to understand rotational velocity OU behavior."""

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_ou


def test_rotational_velocity_diagnostics():
    """Diagnose rotational velocity behavior with detailed logging."""
    # Create grid environment
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)
    env.units = "cm"

    # Simulate with diagnostics
    duration = 10.0
    dt = 0.1
    seed = 42

    # Run simulation to get positions
    positions, times = simulate_trajectory_ou(
        env,
        duration=duration,
        dt=dt,
        speed_mean=7.5,
        rotational_velocity_std=120 * (np.pi / 180),  # 120 deg/s
        rotational_velocity_coherence_time=0.08,
        boundary_mode="reflect",
        seed=seed,
    )

    # Compute diagnostics
    velocities = np.diff(positions, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)

    # Compute heading angles
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_changes = np.diff(headings)
    # Wrap to [-pi, pi]
    heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))

    # Compute turn rates (deg/s)
    turn_rates = heading_changes / dt * (180 / np.pi)

    print("\n=== Rotational Velocity OU Diagnostics ===")
    print(f"Duration: {duration}s, dt: {dt}s, seed: {seed}")
    print("\nSpeed Statistics:")
    print(f"  Mean speed: {speeds.mean():.2f} cm/s (target: 7.5)")
    print(f"  Speed std: {speeds.std():.2f} cm/s")
    print(f"  Speed range: [{speeds.min():.2f}, {speeds.max():.2f}]")

    print("\nHeading Statistics:")
    print(
        f"  Heading range: [{headings.min() * (180 / np.pi):.1f}, {headings.max() * (180 / np.pi):.1f}] deg"
    )
    print(f"  Heading std: {headings.std() * (180 / np.pi):.1f} deg")

    print("\nTurn Rate Statistics:")
    print(f"  Mean |turn rate|: {np.abs(turn_rates).mean():.1f} deg/s")
    print(f"  Turn rate std: {turn_rates.std():.1f} deg/s")
    print(f"  Turn rate range: [{turn_rates.min():.1f}, {turn_rates.max():.1f}] deg/s")
    print("  Expected std: 120 deg/s")

    print("\nSpatial Coverage:")
    position_range = np.ptp(positions, axis=0)
    print(
        f"  X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] → {position_range[0]:.1f} cm"
    )
    print(
        f"  Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] → {position_range[1]:.1f} cm"
    )

    # Check for persistent straight motion
    window = 10  # 1 second windows
    mean_abs_turn_rates = []
    for i in range(0, len(turn_rates) - window, window):
        mean_abs_turn_rates.append(np.abs(turn_rates[i : i + window]).mean())

    low_turn_windows = sum(1 for rate in mean_abs_turn_rates if rate < 10)
    print("\nPersistent Straight Motion:")
    print(
        f"  Windows with <10 deg/s mean turn rate: {low_turn_windows}/{len(mean_abs_turn_rates)}"
    )

    # Create diagnostic plots
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], "b-", alpha=0.5, linewidth=0.5)
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "ro", markersize=10, label="End")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Trajectory (10s)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: Turn rates over time
    ax = axes[0, 1]
    ax.plot(times[1:-1], turn_rates, "b-", alpha=0.5, linewidth=0.5)
    ax.axhline(120, color="r", linestyle="--", label="Target std")
    ax.axhline(-120, color="r", linestyle="--")
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Turn Rate (deg/s)")
    ax.set_title("Rotational Velocity Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Speed over time
    ax = axes[1, 0]
    ax.plot(times[1:], speeds, "b-", alpha=0.5, linewidth=0.5)
    ax.axhline(7.5, color="r", linestyle="--", label="Target mean")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title("Speed Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Heading distribution
    ax = axes[1, 1]
    ax.hist(headings * (180 / np.pi), bins=36, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Heading (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Heading Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("/tmp/rotational_ou_diagnostics.png", dpi=150, bbox_inches="tight")
    print("\nDiagnostic plot saved to: /tmp/rotational_ou_diagnostics.png")

    # Key insight: Check if turn rates are too small
    if np.abs(turn_rates).mean() < 20:
        print("\n⚠️  WARNING: Mean turn rate is very low (<20 deg/s)")
        print("   This suggests rotational velocity might not be exploring properly.")
        print(
            "   Expected: Mean |turn rate| should be ~50-80 deg/s for uniform exploration"
        )


if __name__ == "__main__":
    test_rotational_velocity_diagnostics()
