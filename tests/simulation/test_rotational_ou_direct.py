"""Test rotational OU by instrumenting the simulation directly."""

import numpy as np

from neurospatial import Environment


def test_rotational_ou_direct_instrumentation():
    """Directly instrument the OU simulation to track rotational velocity."""
    # Create grid environment
    arena_size = 80.0
    x = np.linspace(0, arena_size, 17)
    y = np.linspace(0, arena_size, 17)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)
    env.units = "cm"

    # Replicate the OU simulation with instrumentation
    duration = 60.0
    dt = 0.1
    seed = 42
    speed_mean = 7.5
    rotational_velocity_std = 120 * (np.pi / 180)
    rotational_velocity_coherence_time = 0.08

    # Setup
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    n_dims = 2

    # Find valid start position
    valid_bins = np.where(env.active_mask)[0]
    start_bin = rng.choice(valid_bins)
    position = env.bin_centers[start_bin].copy()

    # Initialize velocity
    velocity = rng.standard_normal(n_dims)
    velocity = velocity / np.linalg.norm(velocity) * speed_mean

    # OU parameters
    rotational_velocity = 0.0
    theta_rot = 1.0 / rotational_velocity_coherence_time
    sigma_rot = rotational_velocity_std * np.sqrt(2 * theta_rot / dt)

    # Track variables
    rotational_velocities = []
    speeds = []
    heading_angles = []

    # Simulate (skip first 5s for steady state)
    start_idx = int(5.0 / dt)

    for i in range(n_steps):
        # Update rotational velocity using OU
        dw_rot = rng.standard_normal() * np.sqrt(dt)
        rotational_velocity = (
            rotational_velocity
            - theta_rot * rotational_velocity * dt
            + sigma_rot * dw_rot
        )

        # Rotate velocity vector
        dtheta = rotational_velocity * dt
        cos_dtheta = np.cos(dtheta)
        sin_dtheta = np.sin(dtheta)
        velocity = np.array(
            [
                velocity[0] * cos_dtheta - velocity[1] * sin_dtheta,
                velocity[0] * sin_dtheta + velocity[1] * cos_dtheta,
            ]
        )

        # Maintain constant speed
        speed = np.linalg.norm(velocity)
        if speed > 0:
            velocity = velocity * (speed_mean / speed)

        # Track after steady state
        if i >= start_idx:
            rotational_velocities.append(rotational_velocity)
            speeds.append(np.linalg.norm(velocity))
            heading = np.arctan2(velocity[1], velocity[0])
            heading_angles.append(heading)

        # Update position (with wrapping)
        position = position + velocity * dt
        for dim in range(n_dims):
            range_min, range_max = env.dimension_ranges[dim]
            range_size = range_max - range_min
            position[dim] = range_min + (position[dim] - range_min) % range_size

    # Convert to arrays
    rotational_velocities = np.array(rotational_velocities) * (
        180 / np.pi
    )  # Convert to deg/s
    speeds = np.array(speeds)
    heading_angles = np.array(heading_angles) * (180 / np.pi)  # Convert to degrees

    print("\n=== Direct Instrumentation of Rotational OU ===")
    print(f"Duration: {duration}s, dt: {dt}s, seed: {seed}")
    print(f"Analyzing last {duration - 5:.0f}s (after steady state)")

    print("\nRotational Velocity Statistics:")
    print(f"  Mean: {rotational_velocities.mean():.2f} deg/s (expected: ~0)")
    print(f"  Std: {rotational_velocities.std():.1f} deg/s (expected: 120)")
    print(
        f"  Range: [{rotational_velocities.min():.1f}, {rotational_velocities.max():.1f}] deg/s"
    )

    print("\nSpeed Statistics:")
    print(f"  Mean: {speeds.mean():.2f} cm/s (expected: 7.5)")
    print(f"  Std: {speeds.std():.4f} cm/s (expected: ~0)")
    print(f"  Range: [{speeds.min():.2f}, {speeds.max():.2f}] cm/s")

    print("\nHeading Statistics:")
    print(f"  Std: {heading_angles.std():.1f} deg")
    print("  (Uniform distribution: 103.9 deg)")

    # Check if rotational velocity std matches expectation
    if 100 < rotational_velocities.std() < 140:
        print("\n✓ Rotational velocity std is correct (100-140 deg/s)")
    else:
        print("\n✗ Rotational velocity std is wrong (expected 100-140 deg/s)")

    # Check if speed is constant
    if speeds.std() < 0.01:
        print("✓ Speed is constant (std < 0.01)")
    else:
        print(f"✗ Speed varies (std = {speeds.std():.4f})")


if __name__ == "__main__":
    test_rotational_ou_direct_instrumentation()
