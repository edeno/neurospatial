"""Debug the OU formula by printing intermediate values."""

import numpy as np


def test_ou_formula_step_by_step():
    """Step through OU updates manually to verify formula."""
    # Parameters
    rotational_velocity_std = 120 * (np.pi / 180)  # 2.094 rad/s
    coherence_time = 0.08  # s
    dt = 0.01  # s
    seed = 42

    # Compute OU parameters
    theta = 1.0 / coherence_time
    sigma = rotational_velocity_std * np.sqrt(2 * theta / dt)

    print("=== OU Formula Debug ===")
    print("\nInput Parameters:")
    print(
        f"  rotational_velocity_std: {rotational_velocity_std:.4f} rad/s = {rotational_velocity_std * 180 / np.pi:.1f} deg/s"
    )
    print(f"  coherence_time: {coherence_time} s")
    print(f"  dt: {dt} s")

    print("\nComputed OU Parameters:")
    print(f"  theta = 1/coherence_time = {theta:.2f}")
    print(f"  theta * dt = {theta * dt:.3f} (must be < 1 for stability)")
    print(
        f"  sigma = std * sqrt(2*theta/dt) = {sigma:.4f} rad/s = {sigma * 180 / np.pi:.1f} deg/s"
    )

    print("\nExpected Steady-State:")
    theoretical_var = (sigma * np.sqrt(dt)) ** 2 / (2 * theta)
    theoretical_std = np.sqrt(theoretical_var)
    print(f"  Var(ω) = (sigma*sqrt(dt))²/(2*theta) = {theoretical_var:.4f} (rad/s)²")
    print(
        f"  Std(ω) = {theoretical_std:.4f} rad/s = {theoretical_std * 180 / np.pi:.1f} deg/s"
    )

    # Run simulation
    rng = np.random.default_rng(seed)
    n_steps = 60000  # 60s at dt=0.01s
    omega = 0.0  # Initial rotational velocity
    omega_history = []

    print(f"\nRunning {n_steps} steps...")
    for i in range(n_steps):
        # OU update: dω = -θ ω dt + σ dW
        dw = rng.standard_normal() * np.sqrt(dt)
        d_omega = -theta * omega * dt + sigma * dw
        omega = omega + d_omega

        # Track (skip first 5s for steady state)
        if i >= 5000:
            omega_history.append(omega)

    omega_history = np.array(omega_history) * (180 / np.pi)  # Convert to deg/s

    print("\nMeasured Steady-State (last 55s):")
    print(f"  Mean: {omega_history.mean():.2f} deg/s (expected: ~0)")
    print(f"  Std: {omega_history.std():.1f} deg/s (expected: 120)")
    print(f"  Min: {omega_history.min():.1f} deg/s")
    print(f"  Max: {omega_history.max():.1f} deg/s")

    # Analyze noise term
    print("\nNoise Term Analysis:")
    print(
        f"  sigma * sqrt(dt) = {sigma * np.sqrt(dt):.4f} rad/s = {sigma * np.sqrt(dt) * 180 / np.pi:.1f} deg/s"
    )
    print("  This is the std of noise added per step")

    # Check if close
    ratio = omega_history.std() / 120.0
    if 0.9 < ratio < 1.1:
        print("\n✓✓✓ OU FORMULA IS CORRECT ✓✓✓")
        print(f"  Measured std / Expected std = {ratio:.3f}")
    else:
        print("\n✗✗✗ OU FORMULA IS WRONG ✗✗✗")
        print(f"  Measured std / Expected std = {ratio:.3f}")
        print("  Something is wrong with the implementation!")


if __name__ == "__main__":
    test_ou_formula_step_by_step()
