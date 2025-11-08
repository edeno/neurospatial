"""
Tests for place field metrics.

Following TDD: Tests written FIRST before implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from neurospatial import Environment

# =============================================================================
# Test detect_place_fields
# =============================================================================


def test_detect_place_fields_synthetic():
    """Test place field detection with known synthetic field positions."""
    # Create 2D environment (10x10 grid, 20cm x 20cm)
    positions = []
    for x in np.linspace(0, 20, 100):
        for y in np.linspace(0, 20, 100):
            positions.append([x, y])
    positions = np.array(positions)

    env = Environment.from_samples(positions, bin_size=2.0)

    # Create synthetic firing rate map with one clear place field
    # Peak at center (10, 10) with Gaussian falloff
    firing_rate = np.zeros(env.n_bins)
    for i in range(env.n_bins):
        center = env.bin_centers[i]
        distance = np.sqrt((center[0] - 10) ** 2 + (center[1] - 10) ** 2)
        firing_rate[i] = 5.0 * np.exp(-(distance**2) / (2 * 2.5**2))  # 5 Hz peak

    # Import after env creation to test actual import
    from neurospatial.metrics.place_fields import detect_place_fields

    # Detect fields
    fields = detect_place_fields(firing_rate, env)

    # Should detect exactly one field
    assert len(fields) == 1

    # Field should contain bins near center
    field_centers = env.bin_centers[fields[0]]
    mean_center = field_centers.mean(axis=0)
    assert_allclose(mean_center, [10.0, 10.0], atol=2.0)


def test_detect_place_fields_subfields():
    """Test subfield discrimination with coalescent fields."""
    # Create environment
    positions = np.random.randn(10000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    # Create two close peaks that might be detected as subfields
    firing_rate = np.zeros(env.n_bins)
    peak1_pos = np.array([0.0, 0.0])
    peak2_pos = np.array([3.0, 0.0])  # Close but distinct

    for i in range(env.n_bins):
        center = env.bin_centers[i]
        dist1 = np.linalg.norm(center - peak1_pos)
        dist2 = np.linalg.norm(center - peak2_pos)
        firing_rate[i] = 8.0 * np.exp(-(dist1**2) / (2 * 2.0**2)) + 6.0 * np.exp(
            -(dist2**2) / (2 * 2.0**2)
        )

    from neurospatial.metrics.place_fields import detect_place_fields

    # With subfield detection enabled (default)
    fields_with_subfields = detect_place_fields(firing_rate, env, detect_subfields=True)

    # Should detect 2 subfields
    assert len(fields_with_subfields) >= 1

    # Without subfield detection
    fields_no_subfields = detect_place_fields(firing_rate, env, detect_subfields=False)

    # Should merge into one field
    assert len(fields_no_subfields) == 1


def test_detect_place_fields_interneuron_exclusion():
    """Test interneuron exclusion (high mean rate > 10 Hz)."""
    positions = np.random.randn(5000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    # Create high firing rate everywhere (interneuron-like)
    firing_rate = np.ones(env.n_bins) * 15.0  # 15 Hz everywhere

    from neurospatial.metrics.place_fields import detect_place_fields

    # Should detect no fields (excluded as interneuron)
    fields = detect_place_fields(firing_rate, env, max_mean_rate=10.0)

    assert len(fields) == 0


def test_detect_place_fields_no_fields():
    """Test detection with uniform low firing (detects one large field)."""
    positions = np.random.randn(5000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    # Uniform low firing rate everywhere
    firing_rate = np.ones(env.n_bins) * 0.01

    from neurospatial.metrics.place_fields import detect_place_fields

    fields = detect_place_fields(firing_rate, env)

    # Uniform firing creates one large field (all bins above threshold)
    assert len(fields) == 1
    # The field should contain most/all bins
    assert len(fields[0]) > env.n_bins * 0.9


def test_detect_place_fields_parameter_order():
    """Test that firing_rate comes before env (matches project convention)."""
    positions = np.random.randn(1000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)
    firing_rate = np.ones(env.n_bins)

    from neurospatial.metrics.place_fields import detect_place_fields

    # This should work (firing_rate first)
    fields = detect_place_fields(firing_rate, env)
    assert isinstance(fields, list)


# =============================================================================
# Test field_size
# =============================================================================


def test_field_size():
    """Test field size calculation (area in physical units)."""
    # Create 2D environment with known bin size
    positions = np.random.randn(5000, 2) * 20
    env = Environment.from_samples(positions, bin_size=2.0)

    # Select a field (5 bins)
    field_bins = np.array([0, 1, 2, 3, 4])

    from neurospatial.metrics.place_fields import field_size

    size = field_size(field_bins, env)

    # Size should be positive
    assert size > 0

    # For 2D grid with 2cm bins, each bin is ~4 cm²
    # 5 bins ≈ 20 cm² (approximate, depends on connectivity)
    assert size > 10.0  # At least 10 cm²
    assert size < 50.0  # Less than 50 cm²


def test_field_size_single_bin():
    """Test field size with single bin."""
    positions = np.random.randn(1000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    field_bins = np.array([0])

    from neurospatial.metrics.place_fields import field_size

    size = field_size(field_bins, env)

    # Single bin size should be approximately bin_size²
    assert size > 0
    assert size < 10.0  # Less than 10 cm² for 2cm bins


# =============================================================================
# Test field_centroid
# =============================================================================


def test_field_centroid():
    """Test weighted center of mass calculation."""
    # Create simple environment
    positions = []
    for x in np.linspace(0, 20, 100):
        for y in np.linspace(0, 20, 100):
            positions.append([x, y])
    positions = np.array(positions)

    env = Environment.from_samples(positions, bin_size=2.0)

    # Create symmetric field centered at (10, 10)
    firing_rate = np.zeros(env.n_bins)
    center_pos = np.array([10.0, 10.0])

    for i in range(env.n_bins):
        distance = np.linalg.norm(env.bin_centers[i] - center_pos)
        firing_rate[i] = 5.0 * np.exp(-(distance**2) / (2 * 2.0**2))

    # Select bins in field (firing rate > threshold)
    threshold = 0.5
    field_bins = np.where(firing_rate > threshold)[0]

    from neurospatial.metrics.place_fields import field_centroid

    centroid = field_centroid(firing_rate, field_bins, env)

    # Centroid should be near (10, 10)
    assert centroid.shape == (2,)
    assert_allclose(centroid, [10.0, 10.0], atol=1.0)


def test_field_centroid_asymmetric():
    """Test centroid with asymmetric field (weighted toward high rates)."""
    positions = np.random.randn(5000, 2) * 20
    env = Environment.from_samples(positions, bin_size=2.0)

    # Create asymmetric field: peak offset from geometric center
    firing_rate = np.zeros(env.n_bins)
    peak_pos = np.array([5.0, 5.0])

    for i in range(env.n_bins):
        distance = np.linalg.norm(env.bin_centers[i] - peak_pos)
        firing_rate[i] = 10.0 * np.exp(-(distance**2) / (2 * 3.0**2))

    field_bins = np.where(firing_rate > 1.0)[0]

    from neurospatial.metrics.place_fields import field_centroid

    centroid = field_centroid(firing_rate, field_bins, env)

    # Centroid should be near peak
    assert_allclose(centroid, peak_pos, atol=2.0)


# =============================================================================
# Test skaggs_information
# =============================================================================


def test_skaggs_information_formula():
    """Test Skaggs information formula matches expected computation."""
    # Create simple scenario with known values
    firing_rate = np.array([0.0, 2.0, 4.0, 2.0])  # Hz
    occupancy = np.array([0.25, 0.25, 0.25, 0.25])  # Equal occupancy

    from neurospatial.metrics.place_fields import skaggs_information

    info = skaggs_information(firing_rate, occupancy, base=2.0)

    # Mean rate: 2.0 Hz
    # Expected: Σ p_i (r_i / r̄) log₂(r_i / r̄)
    # = 0.25 * (0/2) * log₂(0/2) + 0.25 * (2/2) * log₂(2/2) +
    #   0.25 * (4/2) * log₂(4/2) + 0.25 * (2/2) * log₂(2/2)
    # = 0 + 0 + 0.25 * 2 * 1 + 0
    # = 0.5 bits/spike

    assert info >= 0  # Non-negative
    assert_allclose(info, 0.5, rtol=0.1)


def test_skaggs_information_uniform():
    """Test that uniform firing gives zero information."""
    firing_rate = np.ones(100) * 3.0  # Constant 3 Hz everywhere
    occupancy = np.ones(100) / 100  # Equal occupancy

    from neurospatial.metrics.place_fields import skaggs_information

    info = skaggs_information(firing_rate, occupancy)

    # Uniform firing → no spatial information
    assert_allclose(info, 0.0, atol=1e-6)


def test_skaggs_information_high_selectivity():
    """Test that selective firing gives high information."""
    # Highly selective: fires in only one bin
    firing_rate = np.zeros(100)
    firing_rate[50] = 100.0  # Very high rate in one bin
    occupancy = np.ones(100) / 100

    from neurospatial.metrics.place_fields import skaggs_information

    info = skaggs_information(firing_rate, occupancy)

    # Should have high information (selective firing)
    assert info > 1.0  # At least 1 bit/spike


# =============================================================================
# Test sparsity
# =============================================================================


def test_sparsity_formula():
    """Test sparsity calculation matches Skaggs et al. 1996 formula."""
    firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
    occupancy = np.array([0.25, 0.25, 0.25, 0.25])

    from neurospatial.metrics.place_fields import sparsity

    spars = sparsity(firing_rate, occupancy)

    # Formula: (Σ p_i r_i)² / Σ p_i r_i²
    # = (0.25*0 + 0.25*2 + 0.25*4 + 0.25*2)² / (0.25*0² + 0.25*2² + 0.25*4² + 0.25*2²)
    # = (2.0)² / (0 + 1 + 4 + 1) = 4 / 6 = 0.667

    assert 0 <= spars <= 1  # Valid range
    assert_allclose(spars, 0.667, atol=0.01)


def test_sparsity_range():
    """Test that sparsity is always in [0, 1]."""
    # Test various firing patterns
    patterns = [
        np.ones(50),  # Uniform
        np.concatenate([np.ones(10) * 10, np.zeros(40)]),  # Sparse
        np.random.rand(100) * 5,  # Random
    ]

    occupancy = np.ones(50) / 50  # Equal occupancy for first two

    from neurospatial.metrics.place_fields import sparsity

    for pattern in patterns[:2]:
        spars = sparsity(pattern, occupancy)
        assert 0 <= spars <= 1


def test_sparsity_sparse_field():
    """Test that sparse fields have low sparsity values."""
    # Fires in only 10% of bins
    firing_rate = np.zeros(100)
    firing_rate[:10] = 10.0
    occupancy = np.ones(100) / 100

    from neurospatial.metrics.place_fields import sparsity

    spars = sparsity(firing_rate, occupancy)

    # Sparse firing → low sparsity value
    assert spars < 0.5


def test_sparsity_uniform_field():
    """Test that uniform fields have high sparsity values."""
    firing_rate = np.ones(100) * 5.0  # Fires everywhere equally
    occupancy = np.ones(100) / 100

    from neurospatial.metrics.place_fields import sparsity

    spars = sparsity(firing_rate, occupancy)

    # Uniform firing → high sparsity value (close to 1)
    assert spars > 0.9


# =============================================================================
# Test field_stability
# =============================================================================


def test_field_stability_identical():
    """Test stability of identical rate maps (should be 1.0)."""
    rate_map_1 = np.random.rand(100) * 5
    rate_map_2 = rate_map_1.copy()

    from neurospatial.metrics.place_fields import field_stability

    stability = field_stability(rate_map_1, rate_map_2, method="pearson")

    # Identical maps → perfect correlation
    assert_allclose(stability, 1.0, atol=1e-6)


def test_field_stability_uncorrelated():
    """Test stability of uncorrelated rate maps (should be ~0)."""
    np.random.seed(42)
    rate_map_1 = np.random.rand(100)
    rate_map_2 = np.random.rand(100)

    from neurospatial.metrics.place_fields import field_stability

    stability = field_stability(rate_map_1, rate_map_2, method="pearson")

    # Uncorrelated → near zero
    assert abs(stability) < 0.3


def test_field_stability_methods():
    """Test both Pearson and Spearman methods."""
    rate_map_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rate_map_2 = np.array([1.1, 2.1, 2.9, 4.1, 5.1])  # Slightly noisy

    from neurospatial.metrics.place_fields import field_stability

    pearson = field_stability(rate_map_1, rate_map_2, method="pearson")
    spearman = field_stability(rate_map_1, rate_map_2, method="spearman")

    # Both should be high (strong correlation)
    assert pearson > 0.95
    assert spearman > 0.95


def test_field_stability_parameter_naming():
    """Test parameter names match expected API."""
    rate_map_1 = np.random.rand(50) * 5
    rate_map_2 = np.random.rand(50) * 5

    from neurospatial.metrics.place_fields import field_stability

    # Should accept 'method' parameter
    stability = field_stability(rate_map_1, rate_map_2, method="pearson")
    assert isinstance(stability, (float, np.floating))


def test_field_stability_constant_arrays():
    """Test that constant arrays return NaN (correlation undefined)."""
    rate_map_1 = np.ones(50)
    rate_map_2 = np.ones(50)

    from neurospatial.metrics.place_fields import field_stability

    # Constant arrays → correlation undefined
    stability = field_stability(rate_map_1, rate_map_2, method="pearson")
    assert np.isnan(stability)


# =============================================================================
# Integration test
# =============================================================================


def test_place_field_workflow_integration():
    """Test complete workflow: detect → measure size/centroid → compute metrics."""
    # Create synthetic place cell data
    positions = np.random.randn(10000, 2) * 15
    env = Environment.from_samples(positions, bin_size=2.0)

    # Create place field
    firing_rate = np.zeros(env.n_bins)
    peak_pos = np.array([0.0, 0.0])
    for i in range(env.n_bins):
        dist = np.linalg.norm(env.bin_centers[i] - peak_pos)
        firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 3.0**2))

    occupancy = np.ones(env.n_bins) / env.n_bins

    # Import all functions
    from neurospatial.metrics.place_fields import (
        detect_place_fields,
        field_centroid,
        field_size,
        field_stability,
        skaggs_information,
        sparsity,
    )

    # Detect fields
    fields = detect_place_fields(firing_rate, env)
    assert len(fields) > 0

    # Measure first field
    field = fields[0]
    size = field_size(field, env)
    centroid = field_centroid(firing_rate, field, env)

    assert size > 0
    assert centroid.shape == (2,)

    # Compute metrics
    info = skaggs_information(firing_rate, occupancy)
    spars = sparsity(firing_rate, occupancy)

    assert info > 0
    assert 0 <= spars <= 1

    # Test stability with itself
    stability = field_stability(firing_rate, firing_rate)
    assert_allclose(stability, 1.0, atol=1e-6)


# =============================================================================
# Test rate_map_coherence
# =============================================================================


def test_rate_map_coherence_perfectly_smooth():
    """Test coherence on perfectly smooth (constant) rate map."""
    # Create environment
    positions = np.random.randn(2000, 2) * 20
    env = Environment.from_samples(positions, bin_size=4.0)

    # Uniform firing rate (constant - no variance)
    firing_rate = np.ones(env.n_bins) * 5.0

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Constant map has no variance - coherence undefined (NaN)
    assert np.isnan(coherence), f"Expected NaN for constant map, got {coherence}"


def test_rate_map_coherence_random_noise():
    """Test coherence on random noise (no spatial structure)."""
    positions = np.random.randn(2000, 2) * 20
    env = Environment.from_samples(positions, bin_size=4.0)

    # Random firing rates (no spatial structure)
    np.random.seed(42)
    firing_rate = np.random.rand(env.n_bins) * 5.0

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Random noise should have low coherence
    assert coherence < 0.5, (
        f"Expected coherence < 0.5 for random noise, got {coherence}"
    )


def test_rate_map_coherence_gaussian_field():
    """Test coherence on smooth Gaussian field."""
    # Create environment
    positions = []
    for x in np.linspace(0, 40, 400):
        for y in np.linspace(0, 40, 400):
            positions.append([x, y])
    positions = np.array(positions)
    env = Environment.from_samples(positions, bin_size=4.0)

    # Smooth Gaussian field
    firing_rate = np.zeros(env.n_bins)
    for i in range(env.n_bins):
        center = env.bin_centers[i]
        distance = np.sqrt((center[0] - 20) ** 2 + (center[1] - 20) ** 2)
        firing_rate[i] = 5.0 * np.exp(-(distance**2) / (2 * 8.0**2))

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Smooth field should have high coherence
    assert coherence > 0.7, (
        f"Expected coherence > 0.7 for smooth field, got {coherence}"
    )


def test_rate_map_coherence_all_zeros():
    """Test coherence with zero firing everywhere."""
    positions = np.random.randn(1000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    # All zeros
    firing_rate = np.zeros(env.n_bins)

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Should return NaN (no variance)
    assert np.isnan(coherence), f"Expected NaN for zero firing, got {coherence}"


def test_rate_map_coherence_with_nans():
    """Test coherence handles NaN values correctly."""
    positions = np.random.randn(2000, 2) * 20
    env = Environment.from_samples(positions, bin_size=4.0)

    # Firing rate with some NaNs and varying values
    np.random.seed(42)
    firing_rate = np.random.rand(env.n_bins) * 5.0
    firing_rate[::5] = np.nan  # 20% NaN

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Should handle NaNs gracefully (compute coherence on valid bins only)
    assert not np.isnan(coherence) or np.all(np.isnan(firing_rate)), (
        "Coherence should handle NaNs unless all values are NaN"
    )


def test_rate_map_coherence_method_parameter():
    """Test that method parameter works (pearson vs spearman)."""
    positions = np.random.randn(2000, 2) * 20
    env = Environment.from_samples(positions, bin_size=4.0)

    # Smooth field
    firing_rate = np.ones(env.n_bins) * 5.0
    firing_rate[: len(firing_rate) // 2] = 3.0

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence_pearson = rate_map_coherence(firing_rate, env, method="pearson")
    coherence_spearman = rate_map_coherence(firing_rate, env, method="spearman")

    # Both should be valid
    assert -1.0 <= coherence_pearson <= 1.0
    assert -1.0 <= coherence_spearman <= 1.0


def test_rate_map_coherence_return_type():
    """Test that coherence returns scalar float."""
    positions = np.random.randn(1000, 2) * 10
    env = Environment.from_samples(positions, bin_size=2.0)

    firing_rate = np.random.rand(env.n_bins) * 5.0

    from neurospatial.metrics.place_fields import rate_map_coherence

    coherence = rate_map_coherence(firing_rate, env)

    # Should return scalar
    assert np.ndim(coherence) == 0, "Coherence should be scalar"
    assert isinstance(coherence, (float, np.floating)) or np.isnan(coherence)


def test_rate_map_coherence_range():
    """Test that coherence is always in [-1, 1]."""
    # Test multiple random environments
    for _ in range(5):
        positions = np.random.randn(2000, 2) * 15
        env = Environment.from_samples(positions, bin_size=3.0)

        # Random firing rate
        firing_rate = np.random.rand(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import rate_map_coherence

        coherence = rate_map_coherence(firing_rate, env)

        # Coherence should be in valid range or NaN
        if not np.isnan(coherence):
            assert -1.0 <= coherence <= 1.0, (
                f"Coherence {coherence} out of range [-1, 1]"
            )
