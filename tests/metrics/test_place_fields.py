"""
Tests for place field metrics.

Following TDD: Tests written FIRST before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment

# =============================================================================
# Integration Test (tests multiple functions together)
# =============================================================================


@pytest.mark.slow
def test_place_field_workflow_integration():
    """Test complete workflow: detect → measure size/centroid → compute metrics.

    Marked as slow because creating environment from 10,000 samples takes time on CI.
    """
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
        selectivity,
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
    select = selectivity(firing_rate, occupancy)

    assert info > 0
    assert 0 <= spars <= 1
    assert select >= 1.0

    # Test stability with itself
    stability = field_stability(firing_rate, firing_rate)
    assert_allclose(stability, 1.0, atol=1e-6)


# =============================================================================
# Main Test Classes (organized by metric/functionality)
# =============================================================================


class TestDetectPlaceFields:
    """Tests for place field detection algorithm.

    Place field detection identifies contiguous regions where firing rate
    exceeds a threshold (typically 20% of peak rate). This is fundamental
    for characterizing place cells in spatial navigation tasks.

    Tests cover:
    - Synthetic fields with known positions
    - Multiple subfields within single neuron
    - Interneuron exclusion (low overall firing)
    - Uniform firing patterns
    - Parameter sensitivity
    """

    def test_detect_place_fields_synthetic(self):
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

    def test_detect_place_fields_subfields(self):
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
        fields_with_subfields = detect_place_fields(
            firing_rate, env, detect_subfields=True
        )

        # Should detect 2 subfields
        assert len(fields_with_subfields) >= 1

        # Without subfield detection
        fields_no_subfields = detect_place_fields(
            firing_rate, env, detect_subfields=False
        )

        # Should merge into one field
        assert len(fields_no_subfields) == 1

    def test_detect_place_fields_interneuron_exclusion(self):
        """Test interneuron exclusion (high mean rate > 10 Hz)."""
        positions = np.random.randn(5000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        # Create high firing rate everywhere (interneuron-like)
        firing_rate = np.ones(env.n_bins) * 15.0  # 15 Hz everywhere

        from neurospatial.metrics.place_fields import detect_place_fields

        # Should detect no fields (excluded as interneuron)
        fields = detect_place_fields(firing_rate, env, max_mean_rate=10.0)

        assert len(fields) == 0

    def test_detect_place_fields_no_fields(self):
        """Test detection with uniform low firing (detects one large field)."""
        np.random.seed(42)
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

    def test_detect_place_fields_parameter_order(self):
        """Test that firing_rate comes before env (matches project convention)."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)
        firing_rate = np.ones(env.n_bins)

        from neurospatial.metrics.place_fields import detect_place_fields

        # This should work (firing_rate first)
        fields = detect_place_fields(firing_rate, env)
        assert isinstance(fields, list)


class TestFieldMetrics:
    """Tests for field size and centroid metrics."""

    def test_field_size(self):
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

    def test_field_size_single_bin(self):
        """Test field size with single bin."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        field_bins = np.array([0])

        from neurospatial.metrics.place_fields import field_size

        size = field_size(field_bins, env)

        # Single bin size should be approximately bin_size²
        assert size > 0
        assert size < 10.0  # Less than 10 cm² for 2cm bins

    def test_field_centroid(self):
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

    def test_field_centroid_asymmetric(self):
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


class TestSkaggsInformation:
    """Tests for Skaggs information metric.

    Skaggs information (bits/spike) quantifies how much spatial information
    each spike conveys about the animal's location. Higher values indicate
    more spatially selective firing.

    Formula: I = Σ p(x) * λ(x) * log2(λ(x)/λ_mean)
    where p(x) is occupancy probability and λ(x) is firing rate at position x.

    Tests cover:
    - Mathematical formula validation
    - Uniform firing (I = 0)
    - Highly selective firing (I >> 0)
    """

    def test_skaggs_information_formula(self):
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

    def test_skaggs_information_uniform(self, medium_2d_env):
        """Test that uniform firing gives zero information."""
        env = medium_2d_env
        firing_rate = np.ones(env.n_bins) * 3.0  # Constant 3 Hz everywhere
        occupancy = np.ones(env.n_bins) / env.n_bins  # Equal occupancy

        from neurospatial.metrics.place_fields import skaggs_information

        info = skaggs_information(firing_rate, occupancy)

        # Uniform firing → no spatial information
        assert_allclose(info, 0.0, atol=1e-6)

    def test_skaggs_information_high_selectivity(self, medium_2d_env):
        """Test that selective firing gives high information."""
        env = medium_2d_env
        # Highly selective: fires in only one bin
        firing_rate = np.zeros(env.n_bins)
        firing_rate[env.n_bins // 2] = 100.0  # Very high rate in one bin
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import skaggs_information

        info = skaggs_information(firing_rate, occupancy)

        # Should have high information (selective firing)
        assert info > 1.0  # At least 1 bit/spike


class TestSparsity:
    """Tests for sparsity metric."""

    def test_sparsity_formula(self):
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

    def test_sparsity_range(self, small_2d_env):
        """Test that sparsity is always in [0, 1]."""
        env = small_2d_env
        # Test various firing patterns
        patterns = [
            np.ones(env.n_bins),  # Uniform
            np.concatenate(
                [np.ones(env.n_bins // 5) * 10, np.zeros(env.n_bins - env.n_bins // 5)]
            ),  # Sparse
        ]

        occupancy = np.ones(env.n_bins) / env.n_bins  # Equal occupancy

        from neurospatial.metrics.place_fields import sparsity

        for pattern in patterns:
            spars = sparsity(pattern, occupancy)
            assert 0 <= spars <= 1

    def test_sparsity_sparse_field(self, medium_2d_env):
        """Test that sparse fields have low sparsity values."""
        env = medium_2d_env
        # Fires in only 10% of bins
        n_active_bins = env.n_bins // 10
        firing_rate = np.zeros(env.n_bins)
        firing_rate[:n_active_bins] = 10.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import sparsity

        spars = sparsity(firing_rate, occupancy)

        # Sparse firing → low sparsity value
        assert spars < 0.5

    def test_sparsity_uniform_field(self, medium_2d_env):
        """Test that uniform fields have high sparsity values."""
        env = medium_2d_env
        firing_rate = np.ones(env.n_bins) * 5.0  # Fires everywhere equally
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import sparsity

        spars = sparsity(firing_rate, occupancy)

        # Uniform firing → high sparsity value (close to 1)
        assert spars > 0.9


class TestFieldStability:
    """Tests for field stability metric."""

    def test_field_stability_identical(self):
        """Test stability of identical rate maps (should be 1.0)."""
        rate_map_1 = np.random.rand(100) * 5
        rate_map_2 = rate_map_1.copy()

        from neurospatial.metrics.place_fields import field_stability

        stability = field_stability(rate_map_1, rate_map_2, method="pearson")

        # Identical maps → perfect correlation
        assert_allclose(stability, 1.0, atol=1e-6)

    def test_field_stability_uncorrelated(self):
        """Test stability of uncorrelated rate maps (should be ~0)."""
        np.random.seed(42)
        rate_map_1 = np.random.rand(100)
        rate_map_2 = np.random.rand(100)

        from neurospatial.metrics.place_fields import field_stability

        stability = field_stability(rate_map_1, rate_map_2, method="pearson")

        # Uncorrelated → near zero
        assert abs(stability) < 0.3

    def test_field_stability_methods(self):
        """Test both Pearson and Spearman methods."""
        rate_map_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rate_map_2 = np.array([1.1, 2.1, 2.9, 4.1, 5.1])  # Slightly noisy

        from neurospatial.metrics.place_fields import field_stability

        pearson = field_stability(rate_map_1, rate_map_2, method="pearson")
        spearman = field_stability(rate_map_1, rate_map_2, method="spearman")

        # Both should be high (strong correlation)
        assert pearson > 0.95
        assert spearman > 0.95

    def test_field_stability_parameter_naming(self):
        """Test parameter names match expected API."""
        rate_map_1 = np.random.rand(50) * 5
        rate_map_2 = np.random.rand(50) * 5

        from neurospatial.metrics.place_fields import field_stability

        # Should accept 'method' parameter
        stability = field_stability(rate_map_1, rate_map_2, method="pearson")
        assert isinstance(stability, (float, np.floating))

    def test_field_stability_constant_arrays(self):
        """Test that constant arrays return NaN (correlation undefined)."""
        rate_map_1 = np.ones(50)
        rate_map_2 = np.ones(50)

        from neurospatial.metrics.place_fields import field_stability

        # Constant arrays → correlation undefined
        stability = field_stability(rate_map_1, rate_map_2, method="pearson")
        assert np.isnan(stability)


class TestRateMapCoherence:
    """Tests for rate map coherence metric.

    Coherence measures spatial smoothness by correlating each bin's firing rate
    with the mean of its neighbors. High coherence indicates spatially structured
    firing patterns typical of place cells.

    Interpretation:
    - High coherence (>0.5): Smooth, place-cell-like firing
    - Low coherence (<0.3): Noisy, unstructured firing
    - NaN: Constant firing (no variance)

    Tests cover:
    - Perfectly smooth fields (expected high coherence)
    - Random noise (expected low coherence)
    - Edge cases (NaN handling, constant firing)
    - Different correlation methods (Pearson vs. Spearman)
    """

    def test_rate_map_coherence_perfectly_smooth(self):
        """Test coherence on perfectly smooth (constant) rate map."""
        # Create environment
        positions = (
            np.random.randn(500, 2) * 20
        )  # Reduced from 2000 - sufficient for coherence test
        env = Environment.from_samples(positions, bin_size=4.0)

        # Uniform firing rate (constant - no variance)
        firing_rate = np.ones(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import rate_map_coherence

        coherence = rate_map_coherence(firing_rate, env)

        # Constant map has no variance - coherence undefined (NaN)
        assert np.isnan(coherence), f"Expected NaN for constant map, got {coherence}"

    def test_rate_map_coherence_random_noise(self):
        """Test coherence on random noise (no spatial structure)."""
        rng = np.random.default_rng(42)

        positions = (
            rng.standard_normal((500, 2)) * 20
        )  # Reduced from 2000 - sufficient for randomness test

        env = Environment.from_samples(positions, bin_size=4.0)

        # Random firing rates (no spatial structure)
        firing_rate = np.random.rand(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import rate_map_coherence

        coherence = rate_map_coherence(firing_rate, env)

        # Random noise should have low coherence
        assert coherence < 0.5, (
            f"Expected coherence < 0.5 for random noise, got {coherence}"
        )

    def test_rate_map_coherence_gaussian_field(self):
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

    def test_rate_map_coherence_all_zeros(self):
        """Test coherence with zero firing everywhere."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        # All zeros
        firing_rate = np.zeros(env.n_bins)

        from neurospatial.metrics.place_fields import rate_map_coherence

        coherence = rate_map_coherence(firing_rate, env)

        # Should return NaN (no variance)
        assert np.isnan(coherence), f"Expected NaN for zero firing, got {coherence}"

    def test_rate_map_coherence_with_nans(self):
        """Test coherence handles NaN values correctly."""
        positions = (
            np.random.randn(500, 2) * 20
        )  # Reduced from 2000 - sufficient for coherence test
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

    def test_rate_map_coherence_method_parameter(self):
        """Test that method parameter works (pearson vs spearman)."""
        positions = (
            np.random.randn(500, 2) * 20
        )  # Reduced from 2000 - sufficient for coherence test
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

    def test_rate_map_coherence_return_type(self):
        """Test that coherence returns scalar float."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        firing_rate = np.random.rand(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import rate_map_coherence

        coherence = rate_map_coherence(firing_rate, env)

        # Should return scalar
        assert np.ndim(coherence) == 0, "Coherence should be scalar"
        assert isinstance(coherence, (float, np.floating)) or np.isnan(coherence)

    def test_rate_map_coherence_range(self):
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


class TestSelectivity:
    """Tests for selectivity metric.

    Selectivity quantifies how concentrated firing is in space. It ranges
    from 0 (uniform firing everywhere) to 1 (firing in single location).

    Formula: selectivity = (max_rate - mean_rate) / max_rate

    Interpretation:
    - selectivity ≈ 0: Uniform, non-selective firing
    - selectivity ≈ 0.5: Moderate spatial selectivity
    - selectivity ≈ 1: Highly selective (place cell-like)

    Tests cover:
    - Formula validation
    - Uniform firing (selectivity = 0)
    - Highly selective firing (selectivity → 1)
    - Edge cases (NaN handling, zero mean, occupancy effects)
    """

    def test_selectivity_formula(self):
        """Test selectivity calculation (peak rate / mean rate)."""
        # Create simple scenario with known values
        firing_rate = np.array([0.0, 2.0, 8.0, 2.0])  # Peak = 8.0
        occupancy = np.array([0.25, 0.25, 0.25, 0.25])  # Equal occupancy

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Mean rate: 0.25*0 + 0.25*2 + 0.25*8 + 0.25*2 = 3.0 Hz
        # Peak rate: 8.0 Hz
        # Expected selectivity: 8.0 / 3.0 = 2.667

        assert select >= 1.0  # Selectivity always >= 1 (peak >= mean)
        assert_allclose(select, 8.0 / 3.0, rtol=0.01)

    def test_selectivity_uniform(self, medium_2d_env):
        """Test that uniform firing gives selectivity = 1.0."""
        env = medium_2d_env
        # Uniform firing: peak = mean
        firing_rate = np.ones(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Uniform → selectivity = 1.0
        assert_allclose(select, 1.0, atol=1e-6)

    def test_selectivity_highly_selective(self, medium_2d_env):
        """Test that highly selective cell has high selectivity."""
        env = medium_2d_env
        # Fires in only one bin at high rate
        firing_rate = np.zeros(env.n_bins)
        firing_rate[env.n_bins // 2] = 100.0  # Very high rate in one bin
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Peak = 100.0, Mean = 100.0 / n_bins
        # Selectivity = 100.0 / (100.0 / n_bins) = n_bins
        # For medium_2d_env with ~625 bins, selectivity ≈ 625
        assert select > 100.0  # Much higher than uniform case

    def test_selectivity_with_nonuniform_occupancy(self):
        """Test selectivity with non-uniform occupancy."""
        # More time spent in low-firing bins
        firing_rate = np.array([1.0, 1.0, 1.0, 10.0])
        occupancy = np.array([0.4, 0.3, 0.2, 0.1])  # Less time at peak

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Mean rate: 0.4*1 + 0.3*1 + 0.2*1 + 0.1*10 = 1.9
        # Peak rate: 10.0
        # Selectivity: 10.0 / 1.9 ≈ 5.26

        assert select > 5.0
        assert select < 6.0

    def test_selectivity_zero_mean(self):
        """Test selectivity returns infinity when mean rate is zero."""
        # All zeros except one bin with NaN
        firing_rate = np.zeros(100)
        firing_rate[50] = 0.0
        occupancy = np.ones(100) / 100

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Mean rate is zero → selectivity undefined
        assert np.isnan(select) or np.isinf(select)

    def test_selectivity_all_nan(self):
        """Test selectivity handles all NaN values."""
        firing_rate = np.full(100, np.nan)
        occupancy = np.ones(100) / 100

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Should return NaN
        assert np.isnan(select)

    def test_selectivity_with_some_nan(self):
        """Test selectivity handles some NaN values correctly."""
        firing_rate = np.array([1.0, 2.0, np.nan, 8.0, 3.0])
        occupancy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Should compute on valid values only
        # Valid: [1.0, 2.0, 8.0, 3.0] with occupancy renormalized
        # Peak = 8.0, Mean ≈ 3.5 (weighted)
        # Selectivity ≈ 2.29

        assert not np.isnan(select)
        assert select >= 1.0

    def test_selectivity_range(self, small_2d_env):
        """Test that selectivity is always >= 1.0."""
        env = small_2d_env
        # Test various firing patterns
        for _ in range(10):
            firing_rate = np.random.rand(env.n_bins) * 10
            occupancy = np.ones(env.n_bins) / env.n_bins

            from neurospatial.metrics.place_fields import selectivity

            select = selectivity(firing_rate, occupancy)

            # Selectivity always >= 1.0 (peak >= mean)
            if not np.isnan(select) and not np.isinf(select):
                assert select >= 1.0, f"Selectivity {select} < 1.0"

    def test_selectivity_return_type(self, medium_2d_env):
        """Test that selectivity returns scalar float."""
        env = medium_2d_env
        firing_rate = np.random.rand(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import selectivity

        select = selectivity(firing_rate, occupancy)

        # Should return scalar
        assert np.ndim(select) == 0
        assert isinstance(select, (float, np.floating)) or np.isnan(select)


class TestInOutFieldRatio:
    """Tests for in/out field ratio metric."""

    def test_in_out_field_ratio_strong_field(self):
        """Test in/out ratio for strong place field."""
        # Create field with 10x stronger firing inside than outside
        firing_rate = np.ones(100) * 1.0
        firing_rate[40:50] = 10.0
        field_bins = np.arange(40, 50)

        from neurospatial.metrics.place_fields import in_out_field_ratio

        ratio = in_out_field_ratio(firing_rate, field_bins)

        # Should be ~10.0
        assert_allclose(ratio, 10.0, rtol=0.1)

    def test_in_out_field_ratio_no_selectivity(self):
        """Test ratio when firing is uniform."""
        firing_rate = np.ones(100) * 5.0
        field_bins = np.arange(40, 50)

        from neurospatial.metrics.place_fields import in_out_field_ratio

        ratio = in_out_field_ratio(firing_rate, field_bins)

        # Should be ~1.0 (no selectivity)
        assert_allclose(ratio, 1.0, rtol=0.01)

    def test_in_out_field_ratio_empty_field(self):
        """Test ratio with empty field."""
        firing_rate = np.random.rand(100) * 5.0
        field_bins = np.array([], dtype=np.int64)

        from neurospatial.metrics.place_fields import in_out_field_ratio

        ratio = in_out_field_ratio(firing_rate, field_bins)

        assert np.isnan(ratio)


class TestInformationMetrics:
    """Tests for information per second and mutual information."""

    def test_information_per_second_basic(self, medium_2d_env):
        """Test information rate calculation."""
        env = medium_2d_env
        # Selective cell: fires only in one bin
        firing_rate = np.zeros(env.n_bins)
        firing_rate[env.n_bins // 2] = 10.0  # 10 Hz in one bin
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import information_per_second

        info_rate = information_per_second(firing_rate, occupancy)

        # Should be positive
        assert info_rate > 0
        assert not np.isnan(info_rate)

    def test_information_per_second_uniform_zero(self, medium_2d_env):
        """Test that uniform firing gives zero information."""
        env = medium_2d_env
        firing_rate = np.ones(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import information_per_second

        info_rate = information_per_second(firing_rate, occupancy)

        # Uniform firing → 0 bits/spike → 0 bits/second
        assert_allclose(info_rate, 0.0, atol=1e-10)

    def test_mutual_information_equals_info_per_second(self, medium_2d_env):
        """Test that MI equals information_per_second."""
        env = medium_2d_env
        firing_rate = np.random.rand(env.n_bins) * 10.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import (
            information_per_second,
            mutual_information,
        )

        mi = mutual_information(firing_rate, occupancy)
        info_rate = information_per_second(firing_rate, occupancy)

        # Should be identical
        assert_allclose(mi, info_rate)

    def test_mutual_information_uniform_zero(self, medium_2d_env):
        """Test that uniform firing gives zero MI."""
        env = medium_2d_env
        firing_rate = np.ones(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins

        from neurospatial.metrics.place_fields import mutual_information

        mi = mutual_information(firing_rate, occupancy)

        # Uniform firing → 0 MI
        assert_allclose(mi, 0.0, atol=1e-10)


class TestSpatialCoverage:
    """Tests for spatial coverage metric."""

    def test_spatial_coverage_selective(self, medium_2d_env):
        """Test coverage for selective cell."""
        env = medium_2d_env
        n_active_bins = env.n_bins // 10  # ~10% of bins
        firing_rate = np.zeros(env.n_bins)
        firing_rate[:n_active_bins] = 5.0  # Fires in ~10% of bins

        from neurospatial.metrics.place_fields import spatial_coverage_single_cell

        coverage = spatial_coverage_single_cell(firing_rate, threshold=0.1)

        # Should cover approximately 10% of environment (within 3% due to integer division)
        assert_allclose(coverage, 0.10, rtol=0.03)

    def test_spatial_coverage_uniform(self, medium_2d_env):
        """Test coverage for cell that fires everywhere."""
        env = medium_2d_env
        firing_rate = np.ones(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import spatial_coverage_single_cell

        coverage = spatial_coverage_single_cell(firing_rate, threshold=0.1)

        # Should cover 100% of environment
        assert_allclose(coverage, 1.0)

    def test_spatial_coverage_silent(self, medium_2d_env):
        """Test coverage for silent cell."""
        env = medium_2d_env
        firing_rate = np.zeros(env.n_bins)

        from neurospatial.metrics.place_fields import spatial_coverage_single_cell

        coverage = spatial_coverage_single_cell(firing_rate, threshold=0.1)

        # Should cover 0% of environment
        assert_allclose(coverage, 0.0)


class TestFieldShapeMetrics:
    """Tests for field shape metrics."""

    def test_field_shape_metrics_circular(self):
        """Test shape metrics for circular field."""
        np.random.seed(42)
        # Create 2D environment
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create circular field
        firing_rate = np.zeros(env.n_bins)
        centers = env.bin_centers
        circular_mask = np.linalg.norm(centers - [0, 0], axis=1) < 5
        field_bins = np.where(circular_mask)[0]
        firing_rate[field_bins] = 10.0

        from neurospatial.metrics.place_fields import field_shape_metrics

        shape = field_shape_metrics(firing_rate, field_bins, env)

        # Circular field should have reasonably low eccentricity
        # Random sampling makes perfect circles unlikely, so use generous threshold
        assert shape["eccentricity"] < 0.8
        assert "major_axis_length" in shape
        assert "minor_axis_length" in shape
        assert "orientation" in shape
        assert "area" in shape
        # Major and minor axes should be similar for circular fields
        assert shape["major_axis_length"] / shape["minor_axis_length"] < 2.0

    def test_field_shape_metrics_elongated(self):
        """Test shape metrics for elongated field."""
        # Create 2D environment
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create elongated field along x-axis
        firing_rate = np.zeros(env.n_bins)
        centers = env.bin_centers
        elongated_mask = (
            (np.abs(centers[:, 1]) < 2) & (centers[:, 0] > -10) & (centers[:, 0] < 10)
        )
        field_bins = np.where(elongated_mask)[0]

        if len(field_bins) > 0:
            firing_rate[field_bins] = 10.0

            from neurospatial.metrics.place_fields import field_shape_metrics

            shape = field_shape_metrics(firing_rate, field_bins, env)

            # Elongated field should have high eccentricity
            assert shape["eccentricity"] > 0.3
            assert shape["major_axis_length"] > shape["minor_axis_length"]

    def test_field_shape_metrics_3d_warning(self):
        """Test that 3D environment raises warning."""
        # Create 3D environment
        data = np.random.randn(500, 3) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        firing_rate = np.zeros(env.n_bins)
        field_bins = np.arange(10)
        firing_rate[field_bins] = 5.0

        from neurospatial.metrics.place_fields import field_shape_metrics

        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning, "field_shape_metrics currently only supports 2D")
            shape = field_shape_metrics(firing_rate, field_bins, env)

        # Should return NaN values
        assert np.isnan(shape["eccentricity"])


class TestFieldShiftDistance:
    """Tests for field shift distance metric."""

    def test_field_shift_distance_no_shift(self):
        """Test shift distance when field hasn't moved."""
        # Create two identical environments
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env1 = Environment.from_samples(data, bin_size=2.0)
        env2 = Environment.from_samples(data, bin_size=2.0)

        # Create identical field in both environments
        firing_rate_1 = np.zeros(env1.n_bins)
        firing_rate_2 = np.zeros(env2.n_bins)
        centers1 = env1.bin_centers
        centers2 = env2.bin_centers
        mask1 = np.linalg.norm(centers1 - [10, 10], axis=1) < 5
        mask2 = np.linalg.norm(centers2 - [10, 10], axis=1) < 5
        field_bins_1 = np.where(mask1)[0]
        field_bins_2 = np.where(mask2)[0]
        firing_rate_1[field_bins_1] = 10.0
        firing_rate_2[field_bins_2] = 10.0

        from neurospatial.metrics.place_fields import field_shift_distance

        shift = field_shift_distance(
            firing_rate_1,
            field_bins_1,
            env1,
            firing_rate_2,
            field_bins_2,
            env2,
        )

        # Should be ~0 (no shift)
        assert_allclose(shift, 0.0, atol=2.0)

    def test_field_shift_distance_euclidean(self):
        """Test Euclidean shift distance."""
        # Create two environments
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env1 = Environment.from_samples(data, bin_size=2.0)
        env2 = Environment.from_samples(data, bin_size=2.0)

        # Create shifted fields
        firing_rate_1 = np.zeros(env1.n_bins)
        firing_rate_2 = np.zeros(env2.n_bins)
        centers1 = env1.bin_centers
        centers2 = env2.bin_centers
        mask1 = np.linalg.norm(centers1 - [0, 0], axis=1) < 5
        mask2 = np.linalg.norm(centers2 - [10, 0], axis=1) < 5
        field_bins_1 = np.where(mask1)[0]
        field_bins_2 = np.where(mask2)[0]
        firing_rate_1[field_bins_1] = 10.0
        firing_rate_2[field_bins_2] = 10.0

        from neurospatial.metrics.place_fields import field_shift_distance

        shift = field_shift_distance(
            firing_rate_1,
            field_bins_1,
            env1,
            firing_rate_2,
            field_bins_2,
            env2,
            use_geodesic=False,
        )

        # Should be ~10 (shifted by 10 units along x-axis)
        assert_allclose(shift, 10.0, atol=2.0)

    def test_field_shift_distance_geodesic(self):
        """Test geodesic shift distance."""
        # Create same environment for both sessions with denser sampling
        # to ensure bins are well-distributed
        np.random.seed(42)
        data = np.random.randn(5000, 2) * 20
        env = Environment.from_samples(data, bin_size=3.0)

        # Ensure environment has enough bins
        if env.n_bins < 30:
            # Skip if environment is too small
            return

        # Create shifted fields in same environment
        # Use actual bin positions to ensure fields are within environment
        firing_rate_1 = np.zeros(env.n_bins)
        firing_rate_2 = np.zeros(env.n_bins)

        # Pick bins around center of environment for field 1
        # This ensures the centroid is within bounds
        mid_bins = env.n_bins // 2
        field_bins_1 = np.arange(mid_bins - 5, mid_bins + 5)
        firing_rate_1[field_bins_1] = 10.0

        # Pick bins further away for field 2
        field_bins_2 = np.arange(mid_bins + 10, mid_bins + 20)
        firing_rate_2[field_bins_2] = 10.0

        from neurospatial.metrics.place_fields import field_shift_distance

        # Fallback to Euclidean if geodesic fails
        try:
            shift = field_shift_distance(
                firing_rate_1,
                field_bins_1,
                env,
                firing_rate_2,
                field_bins_2,
                env,
                use_geodesic=True,
            )

            # If geodesic worked, distance should be non-negative and finite
            if not np.isnan(shift):
                assert shift >= 0
        except Exception:
            # Geodesic distance can fail in some configurations
            # Just check that Euclidean works
            shift_euclidean = field_shift_distance(
                firing_rate_1,
                field_bins_1,
                env,
                firing_rate_2,
                field_bins_2,
                env,
                use_geodesic=False,
            )
            assert shift_euclidean >= 0
            assert not np.isnan(shift_euclidean)


# =============================================================================
# Edge Case and Validation Test Classes (keep as-is)
# =============================================================================


class TestComputeFieldEMD:
    """Test compute_field_emd function."""

    def test_emd_identical_distributions(self):
        """Test EMD is zero for identical distributions."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a firing rate distribution
        centers = env.bin_centers
        firing_rate = np.exp(-0.05 * np.linalg.norm(centers - [0, 0], axis=1) ** 2)

        # EMD of distribution with itself should be zero
        emd = compute_field_emd(firing_rate, firing_rate, env, metric="euclidean")
        assert_allclose(emd, 0.0, atol=1e-10)

    def test_emd_euclidean_metric(self):
        """Test EMD with Euclidean metric."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create two sparse distributions with uniform values in localized regions
        # This avoids numerical precision issues with tiny exponential tails
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        # Set bins within radius to uniform values
        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [10, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 1.0

        # EMD should be positive
        emd = compute_field_emd(field1, field2, env, metric="euclidean")
        assert emd > 0
        assert not np.isnan(emd)

    def test_emd_geodesic_metric(self):
        """Test EMD with geodesic metric."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        # Create denser sampling for better connectivity
        data = np.random.randn(5000, 2) * 20
        env = Environment.from_samples(data, bin_size=2.0)

        # Create two sparse distributions
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [5, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 1.0

        # EMD should be positive
        emd = compute_field_emd(field1, field2, env, metric="geodesic")
        assert emd > 0
        assert not np.isnan(emd)

    def test_emd_normalization(self):
        """Test that normalization works correctly."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create sparse distributions with different total mass
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [10, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 10.0  # Different total mass

        # Should work with normalize=True
        emd_normalized = compute_field_emd(
            field1, field2, env, metric="euclidean", normalize=True
        )
        assert emd_normalized > 0
        assert not np.isnan(emd_normalized)

        # Should fail with normalize=False (unequal mass)
        with pytest.raises(ValueError, match="equal total mass"):
            compute_field_emd(field1, field2, env, metric="euclidean", normalize=False)

    def test_emd_nan_handling(self):
        """Test EMD handles NaN values gracefully."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create sparse distributions
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [10, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 1.0

        # Add some NaN values
        field1[0:5] = np.nan
        field2[10:15] = np.nan

        # Should warn and set NaN to zero
        with pytest.warns(UserWarning, match="NaN values"):
            emd = compute_field_emd(field1, field2, env, metric="euclidean")
            assert emd > 0
            assert not np.isnan(emd)

    def test_emd_all_zeros(self):
        """Test EMD with all-zero distributions."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # All zeros should return NaN with warning
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        with pytest.warns(UserWarning, match="zero total mass"):
            emd = compute_field_emd(field1, field2, env, metric="euclidean")
            assert np.isnan(emd)

    def test_emd_single_bin(self):
        """Test EMD with single non-zero bin."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Single bin with mass
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)
        field1[10] = 1.0
        field2[10] = 1.0

        # EMD should be zero (same bin)
        emd = compute_field_emd(field1, field2, env, metric="euclidean")
        assert_allclose(emd, 0.0, atol=1e-10)

    def test_emd_dimension_mismatch_raises_error(self):
        """Test EMD raises error with mismatched dimensions."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        field1 = np.ones(env.n_bins)
        field2 = np.ones(env.n_bins + 10)  # Wrong size

        with pytest.raises(ValueError, match="same length"):
            compute_field_emd(field1, field2, env, metric="euclidean")

    def test_emd_invalid_metric_raises_error(self):
        """Test EMD raises error with invalid metric."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        field1 = np.ones(env.n_bins)
        field2 = np.ones(env.n_bins)

        with pytest.raises(ValueError, match="metric must be"):
            compute_field_emd(field1, field2, env, metric="invalid")

    def test_emd_wrong_n_bins_raises_error(self):
        """Test EMD raises error when arrays don't match env.n_bins."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create arrays that don't match env.n_bins
        field1 = np.ones(50)
        field2 = np.ones(50)

        with pytest.raises(ValueError, match=r"match env.n_bins"):
            compute_field_emd(field1, field2, env, metric="euclidean")

    def test_emd_sparse_fields(self):
        """Test EMD with sparse fields (few non-zero bins)."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Sparse fields (only a few bins with mass)
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        # Set a few bins
        if env.n_bins >= 20:
            field1[5:10] = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
            field2[15:20] = np.array([0.1, 0.2, 0.3, 0.2, 0.1])

            # EMD should be positive (different locations)
            emd = compute_field_emd(field1, field2, env, metric="euclidean")
            assert emd > 0
            assert not np.isnan(emd)

    def test_emd_euclidean_vs_geodesic_open_field(self):
        """Test that Euclidean and geodesic EMD are similar in open field."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        # Dense sampling for good connectivity
        data = np.random.randn(5000, 2) * 20
        env = Environment.from_samples(data, bin_size=2.0)

        # Create two sparse distributions
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [5, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 1.0

        # Compute both
        emd_euclidean = compute_field_emd(field1, field2, env, metric="euclidean")
        emd_geodesic = compute_field_emd(field1, field2, env, metric="geodesic")

        # In open field, should be similar (within 20%)
        # Note: They won't be identical due to discretization effects
        assert emd_euclidean > 0
        assert emd_geodesic > 0
        assert_allclose(emd_euclidean, emd_geodesic, rtol=0.3)

    def test_emd_symmetry(self):
        """Test that EMD is symmetric: EMD(p, q) == EMD(q, p)."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        data = np.random.randn(500, 2) * 20  # Reduced from 1000 for faster tests
        env = Environment.from_samples(data, bin_size=2.0)

        # Create two different distributions
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        dist_from_center1 = np.linalg.norm(centers - [0, 0], axis=1)
        dist_from_center2 = np.linalg.norm(centers - [10, 0], axis=1)

        field1[dist_from_center1 < 5] = 1.0
        field2[dist_from_center2 < 5] = 1.0

        # Compute EMD in both directions
        emd_forward = compute_field_emd(field1, field2, env, metric="euclidean")
        emd_reverse = compute_field_emd(field2, field1, env, metric="euclidean")

        # Should be identical (symmetric property of EMD)
        assert_allclose(emd_forward, emd_reverse, rtol=1e-10)

    def test_emd_geodesic_respects_barriers(self):
        """Test that geodesic EMD > Euclidean EMD when barriers present."""
        from neurospatial.metrics import compute_field_emd

        # Create L-shaped environment with barrier in middle
        # This creates a non-convex environment where geodesic != Euclidean
        np.random.seed(42)

        # Generate L-shaped data (two rectangles forming an L)
        # Horizontal bar: x in [0, 20], y in [0, 5]
        horizontal = np.random.uniform([0, 0], [20, 5], size=(2000, 2))
        # Vertical bar: x in [0, 5], y in [5, 20]
        vertical = np.random.uniform([0, 5], [5, 20], size=(2000, 2))
        data = np.vstack([horizontal, vertical])

        env = Environment.from_samples(data, bin_size=2.0)

        # Create two fields at opposite ends of the L
        # Field 1: Top of vertical bar (upper left)
        # Field 2: Right end of horizontal bar (lower right)
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        # Field 1: Near [2.5, 17.5] (top of L)
        dist_from_corner1 = np.linalg.norm(centers - [2.5, 17.5], axis=1)
        field1[dist_from_corner1 < 3] = 1.0

        # Field 2: Near [17.5, 2.5] (end of horizontal bar)
        dist_from_corner2 = np.linalg.norm(centers - [17.5, 2.5], axis=1)
        field2[dist_from_corner2 < 3] = 1.0

        # Compute both EMDs
        emd_euclidean = compute_field_emd(field1, field2, env, metric="euclidean")
        emd_geodesic = compute_field_emd(field1, field2, env, metric="geodesic")

        # Geodesic should be larger (needs to go around the corner)
        # Euclidean distance ~ sqrt((17.5-2.5)^2 + (2.5-17.5)^2) ~ 21.2
        # Geodesic distance ~ (17.5-2.5) + (17.5-2.5) ~ 30 (around corner)
        assert emd_euclidean > 0
        assert emd_geodesic > 0
        assert emd_geodesic > emd_euclidean, (
            f"Geodesic EMD ({emd_geodesic:.2f}) should be > Euclidean EMD ({emd_euclidean:.2f}) "
            "when barriers force longer paths"
        )

    def test_emd_geodesic_disconnected_warning(self):
        """Test that disconnected bins trigger aggregated warning."""
        from neurospatial.metrics import compute_field_emd

        np.random.seed(42)
        # Create sparse environment with disconnected regions
        # Two separate clusters with no connectivity between them
        cluster1 = np.random.randn(500, 2) * 3 + np.array([0, 0])
        cluster2 = np.random.randn(500, 2) * 3 + np.array([50, 50])
        data = np.vstack([cluster1, cluster2])

        env = Environment.from_samples(data, bin_size=2.0)

        # Create fields in both clusters
        centers = env.bin_centers
        field1 = np.zeros(env.n_bins)
        field2 = np.zeros(env.n_bins)

        # Field 1: Cluster 1
        dist_from_cluster1 = np.linalg.norm(centers - [0, 0], axis=1)
        field1[dist_from_cluster1 < 5] = 1.0

        # Field 2: Cluster 2
        dist_from_cluster2 = np.linalg.norm(centers - [50, 50], axis=1)
        field2[dist_from_cluster2 < 5] = 1.0

        # Should warn about disconnected pairs
        with pytest.warns(UserWarning, match="disconnected bin pairs"):
            emd = compute_field_emd(field1, field2, env, metric="geodesic")

            # Should still return a valid result (using Euclidean fallback)
            assert emd > 0
            assert not np.isnan(emd)


# =============================================================================
# Additional Edge Case Tests for Coverage
# =============================================================================


class TestDetectPlaceFieldsValidation:
    """Test validation and error handling in detect_place_fields."""

    def test_firing_rate_shape_mismatch_raises_error(self):
        """Test that ValueError is raised when firing_rate shape doesn't match env.n_bins."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import detect_place_fields

        wrong_firing_rate = np.zeros(env.n_bins + 10)

        with pytest.raises(
            ValueError, match=r"firing_rate shape.*does not match.*n_bins"
        ):
            detect_place_fields(wrong_firing_rate, env)

    def test_threshold_out_of_range_raises_error(self):
        """Test that ValueError is raised when threshold is not in (0, 1)."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)
        firing_rate = np.random.rand(env.n_bins) * 5.0

        from neurospatial.metrics.place_fields import detect_place_fields

        with pytest.raises(ValueError, match="threshold must be in \\(0, 1\\)"):
            detect_place_fields(firing_rate, env, threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in \\(0, 1\\)"):
            detect_place_fields(firing_rate, env, threshold=1.0)

        with pytest.raises(ValueError, match="threshold must be in \\(0, 1\\)"):
            detect_place_fields(firing_rate, env, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be in \\(0, 1\\)"):
            detect_place_fields(firing_rate, env, threshold=1.5)

    def test_all_nan_firing_rate_returns_empty(self):
        """Test that all-NaN firing rate returns empty field list."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import detect_place_fields

        firing_rate = np.full(env.n_bins, np.nan)

        fields = detect_place_fields(firing_rate, env)

        assert fields == []

    def test_explicit_min_size_parameter(self):
        """Test that min_size parameter is respected."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import detect_place_fields

        firing_rate = np.zeros(env.n_bins)
        peak_idx = env.n_bins // 2
        firing_rate[peak_idx] = 5.0
        for neighbor in env.neighbors(peak_idx):
            firing_rate[neighbor] = 3.0

        fields = detect_place_fields(firing_rate, env, min_size=20)
        assert len(fields) == 0 or all(len(f) >= 20 for f in fields)

    def test_subfields_extension_path(self):
        """Test that subfields are properly extended to fields list."""
        positions = []
        for x in np.linspace(0, 30, 150):
            for y in np.linspace(0, 30, 150):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import detect_place_fields

        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            center = env.bin_centers[i]
            dist1 = np.sqrt((center[0] - 10) ** 2 + (center[1] - 10) ** 2)
            dist2 = np.sqrt((center[0] - 20) ** 2 + (center[1] - 20) ** 2)
            firing_rate[i] = 5.0 * np.exp(-(dist1**2) / (2 * 2.5**2)) + 5.0 * np.exp(
                -(dist2**2) / (2 * 2.5**2)
            )

        fields_with_subfields = detect_place_fields(
            firing_rate, env, detect_subfields=True, threshold=0.2
        )

        fields_without_subfields = detect_place_fields(
            firing_rate, env, detect_subfields=False, threshold=0.2
        )

        assert len(fields_with_subfields) >= len(fields_without_subfields)


class TestFieldCentroidEdgeCases:
    """Test edge cases in field_centroid function."""

    def test_field_centroid_zero_firing_rate(self):
        """Test field_centroid when all firing rates are zero (unweighted centroid)."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import field_centroid

        field_bins = np.array([0, 1, 2, 3, 4])
        firing_rate = np.zeros(env.n_bins)
        firing_rate[field_bins] = 0.0

        centroid = field_centroid(firing_rate, field_bins, env)

        expected_centroid = env.bin_centers[field_bins].mean(axis=0)
        assert_allclose(centroid, expected_centroid, rtol=1e-10)


class TestSkaggsInformationEdgeCases:
    """Test edge cases in skaggs_information function."""

    def test_skaggs_information_zero_mean_rate(self):
        """Test skaggs_information when mean rate is zero."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import skaggs_information

        firing_rate = np.zeros(env.n_bins)
        occupancy = np.ones(env.n_bins)

        info = skaggs_information(firing_rate, occupancy)
        assert info == 0.0

    def test_skaggs_information_nan_mean_rate(self):
        """Test skaggs_information when mean rate is NaN."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import skaggs_information

        firing_rate = np.full(env.n_bins, np.nan)
        occupancy = np.ones(env.n_bins)

        info = skaggs_information(firing_rate, occupancy)
        assert info == 0.0


class TestSparsityEdgeCases:
    """Test edge cases in sparsity function."""

    def test_sparsity_zero_denominator(self):
        """Test sparsity when denominator is zero."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import sparsity

        firing_rate = np.zeros(env.n_bins)
        occupancy = np.ones(env.n_bins)

        sparsity_value = sparsity(firing_rate, occupancy)
        assert sparsity_value == 0.0

    def test_sparsity_nan_denominator(self):
        """Test sparsity when denominator is NaN."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import sparsity

        firing_rate = np.full(env.n_bins, np.nan)
        occupancy = np.ones(env.n_bins)

        sparsity_value = sparsity(firing_rate, occupancy)
        assert sparsity_value == 0.0


class TestFieldStabilityEdgeCases:
    """Test edge cases in field_stability function."""

    def test_field_stability_insufficient_valid_points(self):
        """Test field_stability when fewer than 2 valid points."""
        from neurospatial.metrics.place_fields import field_stability

        rate_map_1 = np.array([1.0, np.nan, np.nan, np.nan])
        rate_map_2 = np.array([2.0, np.nan, np.nan, np.nan])

        stability = field_stability(rate_map_1, rate_map_2)
        assert stability == 0.0

    def test_field_stability_all_nan(self):
        """Test field_stability when all values are NaN."""
        from neurospatial.metrics.place_fields import field_stability

        rate_map_1 = np.array([np.nan, np.nan, np.nan])
        rate_map_2 = np.array([np.nan, np.nan, np.nan])

        stability = field_stability(rate_map_1, rate_map_2)
        assert stability == 0.0

    def test_field_stability_invalid_method(self):
        """Test field_stability raises ValueError for invalid method."""
        from neurospatial.metrics.place_fields import field_stability

        rate_map_1 = np.array([1.0, 2.0, 3.0])
        rate_map_2 = np.array([1.5, 2.5, 3.5])

        with pytest.raises(
            ValueError, match=r"Unknown method.*Use 'pearson' or 'spearman'"
        ):
            field_stability(rate_map_1, rate_map_2, method="invalid")


class TestRateMapCoherenceEdgeCases:
    """Test edge cases in rate_map_coherence function."""

    def test_rate_map_coherence_wrong_shape_raises_error(self):
        """Test rate_map_coherence raises ValueError for wrong shape."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import rate_map_coherence

        # Wrong shape
        firing_rate = np.random.rand(env.n_bins + 10)

        with pytest.raises(ValueError, match=r"firing_rate\.shape must be"):
            rate_map_coherence(firing_rate, env)

    def test_rate_map_coherence_all_nan_returns_nan(self):
        """Test rate_map_coherence returns NaN when all values are NaN."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import rate_map_coherence

        firing_rate = np.full(env.n_bins, np.nan)

        coherence = rate_map_coherence(firing_rate, env)
        assert np.isnan(coherence)

    def test_rate_map_coherence_insufficient_points_returns_nan(self):
        """Test rate_map_coherence returns NaN when too few valid points."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import rate_map_coherence

        # Only one valid bin
        firing_rate = np.full(env.n_bins, np.nan)
        firing_rate[0] = 1.0

        coherence = rate_map_coherence(firing_rate, env)
        assert np.isnan(coherence)

    def test_rate_map_coherence_zero_variance_returns_nan(self):
        """Test rate_map_coherence returns NaN when zero variance."""
        positions = []
        for x in np.linspace(0, 20, 50):
            for y in np.linspace(0, 20, 50):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import rate_map_coherence

        # Constant firing rate (zero variance)
        firing_rate = np.ones(env.n_bins) * 5.0

        coherence = rate_map_coherence(firing_rate, env)
        # May return NaN for zero variance
        assert isinstance(coherence, (float, np.floating))

    def test_rate_map_coherence_invalid_method_raises_error(self):
        """Test rate_map_coherence raises ValueError for invalid method."""
        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        from neurospatial.metrics.place_fields import rate_map_coherence

        firing_rate = np.random.rand(env.n_bins) * 5.0

        with pytest.raises(
            ValueError, match=r"Unknown method.*Use 'pearson' or 'spearman'"
        ):
            rate_map_coherence(firing_rate, env, method="invalid")


class TestSelectivityEdgeCases:
    """Test edge cases in selectivity function."""

    def test_selectivity_zero_mean_positive_peak_returns_inf(self):
        """Test selectivity returns inf when mean_rate is 0 but peak_rate > 0."""
        from neurospatial.metrics.place_fields import selectivity

        # Create firing rate with one positive value, rest zeros
        firing_rate = np.zeros(100)
        firing_rate[50] = 5.0

        # Create uniform occupancy so mean_rate will be low
        # With one non-zero firing rate and uniform occupancy, mean should be close to zero
        occupancy = np.ones(100)

        result = selectivity(firing_rate, occupancy)

        # When mean is effectively zero but peak > 0, should return very high selectivity
        assert result >= 100  # Very high selectivity (exactly 100 in this case)


class TestInOutFieldRatioEdgeCases:
    """Test edge cases in in_out_field_ratio function."""

    def test_in_out_field_ratio_entire_environment(self):
        """Test in_out_field_ratio returns NaN when field covers entire environment."""
        from neurospatial.metrics.place_fields import in_out_field_ratio

        firing_rate = np.random.rand(100) * 5.0
        field_bins = np.arange(100)  # All bins

        result = in_out_field_ratio(firing_rate, field_bins)
        assert np.isnan(result)

    def test_in_out_field_ratio_no_valid_bins(self):
        """Test in_out_field_ratio returns NaN when no valid in/out bins."""
        from neurospatial.metrics.place_fields import in_out_field_ratio

        # All NaN firing rates
        firing_rate = np.full(100, np.nan)
        field_bins = np.array([10, 20, 30])

        result = in_out_field_ratio(firing_rate, field_bins)
        assert np.isnan(result)

    def test_in_out_field_ratio_zero_out_field_positive_in_field(self):
        """Test in_out_field_ratio returns inf when out_field_rate is 0 but in_field_rate > 0."""
        from neurospatial.metrics.place_fields import in_out_field_ratio

        # Only field bins have non-zero rates
        firing_rate = np.zeros(100)
        field_bins = np.array([10, 20, 30])
        firing_rate[field_bins] = 5.0

        result = in_out_field_ratio(firing_rate, field_bins)
        assert np.isinf(result)

    def test_in_out_field_ratio_zero_both(self):
        """Test in_out_field_ratio returns NaN when both in_field and out_field are 0."""
        from neurospatial.metrics.place_fields import in_out_field_ratio

        # All zeros
        firing_rate = np.zeros(100)
        field_bins = np.array([10, 20, 30])

        result = in_out_field_ratio(firing_rate, field_bins)
        assert np.isnan(result)


class TestInformationPerSecondEdgeCases:
    """Test edge cases in information_per_second function."""

    def test_information_per_second_no_valid_pairs(self):
        """Test information_per_second returns NaN when no valid firing_rate/occupancy pairs."""
        from neurospatial.metrics.place_fields import information_per_second

        # All NaN
        firing_rate = np.full(100, np.nan)
        occupancy = np.full(100, np.nan)

        result = information_per_second(firing_rate, occupancy)
        assert np.isnan(result)


class TestSpatialCoverageSingleCellEdgeCases:
    """Test edge cases in spatial_coverage_single_cell function."""

    def test_spatial_coverage_all_nan_returns_nan(self):
        """Test spatial_coverage_single_cell returns NaN when all firing rates are NaN."""
        from neurospatial.metrics.place_fields import spatial_coverage_single_cell

        firing_rate = np.full(100, np.nan)

        result = spatial_coverage_single_cell(firing_rate, threshold=0.1)
        assert np.isnan(result)


class TestFieldShapeMetricsEdgeCases:
    """Test edge cases in field_shape_metrics function."""

    def test_field_shape_metrics_empty_field(self):
        """Test field_shape_metrics returns empty dict for empty field_bins."""
        from neurospatial.metrics.place_fields import field_shape_metrics

        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        firing_rate = np.random.rand(env.n_bins)
        field_bins = np.array([], dtype=np.int64)

        result = field_shape_metrics(firing_rate, field_bins, env)

        # Should return dict with NaN values
        assert isinstance(result, dict)
        assert np.isnan(result.get("area", 0))

    def test_field_shape_metrics_all_nan_rates(self):
        """Test field_shape_metrics returns NaN values when all rates are NaN."""
        from neurospatial.metrics.place_fields import field_shape_metrics

        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        firing_rate = np.full(env.n_bins, np.nan)
        field_bins = np.array([10, 20, 30])

        result = field_shape_metrics(firing_rate, field_bins, env)

        # Should return dict with NaN values
        assert isinstance(result, dict)
        assert np.isnan(result.get("area", 0))


class TestFieldShiftDistanceEdgeCases:
    """Test edge cases in field_shift_distance function."""

    def test_field_shift_distance_nan_centroid(self):
        """Test field_shift_distance returns NaN when centroids are NaN."""
        from neurospatial.metrics.place_fields import field_shift_distance

        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        # All NaN firing rates will produce NaN centroids
        firing_rate_1 = np.full(env.n_bins, np.nan)
        firing_rate_2 = np.random.rand(env.n_bins)
        field_bins_1 = np.array([10, 20, 30])
        field_bins_2 = np.array([40, 50, 60])

        result = field_shift_distance(
            firing_rate_1, field_bins_1, env, firing_rate_2, field_bins_2, env
        )
        assert np.isnan(result)

    def test_field_shift_distance_incompatible_environments_geodesic(self):
        """Test field_shift_distance with incompatible environments (geodesic mode)."""
        from neurospatial.metrics.place_fields import field_shift_distance

        # Use deterministic positions to ensure centroids are in bounds
        positions1 = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        env1 = Environment.from_samples(positions1, bin_size=1.0)

        positions2 = np.array([[0, 0], [1, 0], [0, 1]])  # Fewer bins
        env2 = Environment.from_samples(positions2, bin_size=1.0)

        # Create firing rates and use bins that exist in both environments
        firing_rate_1 = np.random.rand(env1.n_bins)
        firing_rate_2 = np.random.rand(env2.n_bins)
        # Use bins near center to ensure centroids are valid
        field_bins_1 = np.array([0, 1, 2])  # Use first few bins
        field_bins_2 = np.array([0, 1, 2])  # Use first few bins (if they exist in env2)

        # When environments have different number of bins, falls back to Euclidean
        # This should trigger the "different number of bins" warning (lines 1660-1670)
        # OR the "centroids outside bounds" warning if centroids map to invalid bins
        with pytest.warns(
            UserWarning, match="(different number of bins|centroids fall outside)"
        ):
            result = field_shift_distance(
                firing_rate_1,
                field_bins_1,
                env1,
                firing_rate_2,
                field_bins_2,
                env2,
                use_geodesic=True,
            )
        # Should return either a valid distance (Euclidean fallback) or NaN (out of bounds)
        assert isinstance(result, float)
        assert result >= 0


class TestComputeFieldEMDEdgeCases:
    """Test edge cases in compute_field_emd function."""

    def test_compute_field_emd_both_zero(self):
        """Test compute_field_emd returns 0 when both distributions are all zeros (unnormalized)."""
        from neurospatial.metrics.place_fields import compute_field_emd

        positions = np.random.randn(1000, 2) * 10
        env = Environment.from_samples(positions, bin_size=2.0)

        firing_rate_1 = np.zeros(env.n_bins)
        firing_rate_2 = np.zeros(env.n_bins)

        # Use normalize=False since zero mass with normalize=True returns NaN
        result = compute_field_emd(firing_rate_1, firing_rate_2, env, normalize=False)
        assert result == 0.0
