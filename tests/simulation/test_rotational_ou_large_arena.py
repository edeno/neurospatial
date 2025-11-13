"""Test rotational OU in a very large arena to minimize boundary effects."""

import numpy as np

from neurospatial import Environment


def test_rotational_ou_large_arena():
    """Test in large arena where boundaries are rarely hit."""
    # Create LARGE arena: 400x400 cm (4m x 4m)
    arena_size = 400.0
    x = np.linspace(0, arena_size, 81)  # 5cm bins
    y = np.linspace(0, arena_size, 81)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(grid_data, bin_size=5.0)
    env.units = "cm"

    # Replicate OU simulation directly
    duration = 60.0
    dt = 0.01  # Small dt for stability
    seed = 42
    speed_mean = 7.5  # cm/s
    rotational_velocity_std = 120 * (np.pi / 180)  # rad/s
    rotational_velocity_coherence_time = 0.08  # s

    # Setup
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)

    # Start in center of arena to maximize distance from boundaries
    position = np.array([arena_size / 2, arena_size / 2])

    # Initialize velocity
    velocity = rng.standard_normal(2)
    velocity = velocity / np.linalg.norm(velocity) * speed_mean

    # OU parameters
    rotational_velocity = 0.0
    theta_rot = 1.0 / rotational_velocity_coherence_time
    sigma_rot = rotational_velocity_std * np.sqrt(2 * theta_rot / dt)

    print("\n=== Large Arena Test (minimize boundary effects) ===")
    print(f"Arena size: {arena_size}x{arena_size} cm")
    print(f"Start position: center at ({arena_size / 2}, {arena_size / 2}) cm")
    print(f"Max travel distance: {speed_mean * duration:.0f} cm")
    print(f"Distance to nearest wall: {arena_size / 2:.0f} cm")
    print(
        f"dt={dt}s, theta*dt={(1 / rotational_velocity_coherence_time) * dt:.3f} (stable: <1)"
    )

    # Track variables
    rotational_velocities = []
    positions_list = []
    boundary_hits = 0

    # Simulate
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

        # Check for boundary (before updating position)
        proposed_position = position + velocity * dt
        if not env.contains(proposed_position):
            boundary_hits += 1
            # Simple reflection: reverse velocity component perpendicular to boundary
            if proposed_position[0] < 0 or proposed_position[0] > arena_size:
                velocity[0] = -velocity[0]
            if proposed_position[1] < 0 or proposed_position[1] > arena_size:
                velocity[1] = -velocity[1]

        # Update position
        position = position + velocity * dt

        # Track (skip first 5s for steady state)
        if i >= int(5.0 / dt):
            rotational_velocities.append(rotational_velocity)
            positions_list.append(position.copy())

    # Convert to arrays
    rotational_velocities = np.array(rotational_velocities) * (180 / np.pi)  # deg/s
    positions_array = np.array(positions_list)

    print("\n Boundary Interactions:")
    print(f"  Total boundary hits: {boundary_hits}")
    print(f"  Hit rate: {boundary_hits / n_steps * 100:.1f}% of steps")

    print("\nRotational Velocity Statistics (steady state):")
    print(f"  Mean: {rotational_velocities.mean():.2f} deg/s (expected: ~0)")
    print(f"  Std: {rotational_velocities.std():.1f} deg/s (expected: 120)")
    print(
        f"  Range: [{rotational_velocities.min():.1f}, {rotational_velocities.max():.1f}] deg/s"
    )

    print("\nSpatial Coverage:")
    position_range = np.ptp(positions_array, axis=0)
    print(f"  X range: {position_range[0]:.1f} cm")
    print(f"  Y range: {position_range[1]:.1f} cm")
    print(f"  Mean range: {position_range.mean():.1f} cm")

    # Check if rotational velocity std is correct
    if 100 < rotational_velocities.std() < 140:
        print("\n✓ Rotational velocity std is correct (100-140 deg/s)")
        return True
    else:
        print("\n✗ Rotational velocity std is wrong")
        print(f"  Measured: {rotational_velocities.std():.1f} deg/s")
        print("  Expected: 120 deg/s")
        print(f"  Ratio: {rotational_velocities.std() / 120:.2f}x")
        return False


if __name__ == "__main__":
    test_rotational_ou_large_arena()
