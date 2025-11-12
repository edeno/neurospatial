"""Test different OU formula variants to find correct discretization."""

import numpy as np


def test_variant_1_continuous_formula():
    """Variant 1: sigma = std * sqrt(2*theta/dt), noise = N(0, sqrt(dt))"""
    print("\n=== VARIANT 1: Continuous-time formula ===")
    print("sigma = std * sqrt(2*theta/dt)")
    print("noise = rng.standard_normal() * sqrt(dt)")

    std = 120 * (np.pi / 180)  # 2.094 rad/s
    coherence_time = 0.08
    dt = 0.01

    theta = 1.0 / coherence_time
    sigma = std * np.sqrt(2 * theta / dt)

    print(f"  theta = {theta:.2f}, dt = {dt}")
    print(f"  sigma = {sigma:.2f} rad/s = {sigma * 180 / np.pi:.1f} deg/s")

    # Simulate
    rng = np.random.default_rng(42)
    omega = 0.0
    omega_history = []

    for i in range(60000):
        dw = rng.standard_normal() * np.sqrt(dt)
        omega = omega - theta * omega * dt + sigma * dw
        if i >= 5000:
            omega_history.append(omega)

    omega_history = np.array(omega_history) * (180 / np.pi)
    print(f"  Measured std: {omega_history.std():.1f} deg/s")
    print("  Expected std: 120.0 deg/s")
    print(f"  Ratio: {omega_history.std() / 120:.2f}x")

    return omega_history.std()


def test_variant_2_discrete_formula():
    """Variant 2: sigma = std * sqrt(2*theta), noise = N(0, sqrt(dt))"""
    print("\n=== VARIANT 2: Discrete-time formula (no dt in sigma) ===")
    print("sigma = std * sqrt(2*theta)")
    print("noise = rng.standard_normal() * sqrt(dt)")

    std = 120 * (np.pi / 180)  # 2.094 rad/s
    coherence_time = 0.08
    dt = 0.01

    theta = 1.0 / coherence_time
    sigma = std * np.sqrt(2 * theta)  # NO /dt here!

    print(f"  theta = {theta:.2f}, dt = {dt}")
    print(f"  sigma = {sigma:.2f} rad/s = {sigma * 180 / np.pi:.1f} deg/s")

    # Simulate
    rng = np.random.default_rng(42)
    omega = 0.0
    omega_history = []

    for i in range(60000):
        dw = rng.standard_normal() * np.sqrt(dt)
        omega = omega - theta * omega * dt + sigma * dw
        if i >= 5000:
            omega_history.append(omega)

    omega_history = np.array(omega_history) * (180 / np.pi)
    print(f"  Measured std: {omega_history.std():.1f} deg/s")
    print("  Expected std: 120.0 deg/s")
    print(f"  Ratio: {omega_history.std() / 120:.2f}x")

    return omega_history.std()


def test_variant_3_no_sqrt_dt():
    """Variant 3: sigma = std * sqrt(2*theta/dt), noise = N(0, 1) WITHOUT sqrt(dt)"""
    print("\n=== VARIANT 3: No sqrt(dt) in noise term ===")
    print("sigma = std * sqrt(2*theta/dt)")
    print("noise = rng.standard_normal()  # NO sqrt(dt)!")

    std = 120 * (np.pi / 180)  # 2.094 rad/s
    coherence_time = 0.08
    dt = 0.01

    theta = 1.0 / coherence_time
    sigma = std * np.sqrt(2 * theta / dt)

    print(f"  theta = {theta:.2f}, dt = {dt}")
    print(f"  sigma = {sigma:.2f} rad/s = {sigma * 180 / np.pi:.1f} deg/s")

    # Simulate
    rng = np.random.default_rng(42)
    omega = 0.0
    omega_history = []

    for i in range(60000):
        dw = rng.standard_normal()  # NO sqrt(dt) scaling!
        omega = omega - theta * omega * dt + sigma * dw
        if i >= 5000:
            omega_history.append(omega)

    omega_history = np.array(omega_history) * (180 / np.pi)
    print(f"  Measured std: {omega_history.std():.1f} deg/s")
    print("  Expected std: 120.0 deg/s")
    print(f"  Ratio: {omega_history.std() / 120:.2f}x")

    return omega_history.std()


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING OU DISCRETIZATION VARIANTS")
    print("=" * 70)

    std1 = test_variant_1_continuous_formula()
    std2 = test_variant_2_discrete_formula()
    std3 = test_variant_3_no_sqrt_dt()

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Variant 1 (continuous, with sqrt(dt)): {std1:.1f} deg/s")
    print(f"Variant 2 (discrete, with sqrt(dt)):   {std2:.1f} deg/s")
    print(f"Variant 3 (continuous, NO sqrt(dt)):   {std3:.1f} deg/s")
    print("\nTarget: 120.0 deg/s")

    # Find winner
    errors = [abs(std1 - 120), abs(std2 - 120), abs(std3 - 120)]
    winner = errors.index(min(errors)) + 1

    print(f"\n✓✓✓ WINNER: Variant {winner} ✓✓✓")
    if winner == 1:
        print("Use: sigma = std * sqrt(2*theta/dt) with dw = N(0, sqrt(dt))")
    elif winner == 2:
        print("Use: sigma = std * sqrt(2*theta) with dw = N(0, sqrt(dt))")
    else:
        print("Use: sigma = std * sqrt(2*theta/dt) with dw = N(0, 1)")
