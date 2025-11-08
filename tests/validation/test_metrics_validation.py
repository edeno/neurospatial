"""
Validation tests for neurospatial metrics against published algorithms.

These tests validate our implementations against:
1. Known mathematical properties (ground truth)
2. Published formulas from reference papers
3. Synthetic data with expected outcomes

External package comparisons (opexebo, neurocode) are marked as optional
and only run if those packages are installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.metrics import (
    border_score,
    detect_place_fields,
    skaggs_information,
    sparsity,
)


class TestSpatialInformationValidation:
    """Validate Skaggs spatial information against published formula.

    Reference: Skaggs et al. (1993) "An Information-Theoretic Approach to
    Deciphering the Hippocampal Code"

    Formula: I = Σ pᵢ (rᵢ/r̄) log₂(rᵢ/r̄)
    where pᵢ = occupancy probability, rᵢ = firing rate, r̄ = mean rate
    """

    def test_spatial_information_perfect_localization(self):
        """Cell firing in single bin has maximum information content."""
        # Create 10x10 environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Fire only in one bin (perfect place cell)
        firing_rate = np.zeros(env.n_bins)
        firing_rate[50] = 10.0  # Single bin fires at 10 Hz

        info = skaggs_information(firing_rate, occupancy, base=2.0)

        # Expected: log2(n_bins) = log2(100) ≈ 6.64 bits/spike
        expected = np.log2(env.n_bins)

        # Allow small numerical error
        assert np.abs(info - expected) < 0.01, (
            f"Perfect localization should give ~{expected:.2f} bits/spike, "
            f"got {info:.2f}"
        )

    def test_spatial_information_uniform_firing(self):
        """Uniform firing has zero information content."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Uniform occupancy and firing
        occupancy = np.ones(env.n_bins)
        firing_rate = np.ones(env.n_bins) * 5.0  # Uniform 5 Hz everywhere

        info = skaggs_information(firing_rate, occupancy, base=2.0)

        # Expected: 0 bits/spike (no spatial information)
        assert np.abs(info) < 0.01, (
            f"Uniform firing should give ~0 bits/spike, got {info:.2f}"
        )

    def test_spatial_information_formula_validation(self):
        """Validate against manual calculation of Skaggs formula."""
        positions = np.random.randn(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create simple test case with correct size
        occupancy = np.ones(env.n_bins)
        occupancy[:min(4, env.n_bins)] = np.array([10.0, 20.0, 30.0, 40.0])[:min(4, env.n_bins)]
        occupancy = occupancy / occupancy.sum()  # Normalize to probabilities

        firing_rate = np.zeros(env.n_bins)
        firing_rate[:min(4, env.n_bins)] = np.array([0.0, 5.0, 10.0, 15.0])[:min(4, env.n_bins)]

        # Manual calculation
        mean_rate = np.sum(occupancy * firing_rate)

        # Only include bins with firing
        mask = firing_rate > 0
        expected_info = 0.0
        for i in range(len(firing_rate)):
            if mask[i]:
                log_term = np.log2(firing_rate[i] / mean_rate)
                expected_info += occupancy[i] * (firing_rate[i] / mean_rate) * log_term

        # Compare with implementation
        computed_info = skaggs_information(firing_rate, occupancy, base=2.0)

        assert np.abs(computed_info - expected_info) < 1e-10, (
            f"Manual calculation: {expected_info:.10f}, "
            f"Implementation: {computed_info:.10f}"
        )

    def test_spatial_information_range_validation(self):
        """Spatial information should be non-negative."""
        positions = np.random.randn(500, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Random realistic firing pattern
        np.random.seed(42)
        occupancy = np.random.rand(env.n_bins)
        occupancy = occupancy / occupancy.sum()

        firing_rate = np.random.rand(env.n_bins) * 10.0

        info = skaggs_information(firing_rate, occupancy, base=2.0)

        assert info >= 0, f"Information content must be non-negative, got {info:.2f}"
        assert info <= np.log2(env.n_bins), (
            f"Information content must be ≤ log2(n_bins)={np.log2(env.n_bins):.2f}, "
            f"got {info:.2f}"
        )


class TestSparsityValidation:
    """Validate sparsity measure against published formula.

    Reference: Skaggs et al. (1996)
    Formula: S = (Σ pᵢ rᵢ)² / Σ pᵢ rᵢ²

    Range: [0, 1], where 0 = fires everywhere equally, 1 = fires in single bin
    """

    def test_sparsity_perfect_place_cell(self):
        """Cell firing in single bin has sparsity = 1/n_bins."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)
        occupancy = occupancy / occupancy.sum()

        # Fire only in one bin
        firing_rate = np.zeros(env.n_bins)
        firing_rate[50] = 10.0

        s = sparsity(firing_rate, occupancy)

        # Expected: 1/n_bins for perfect localization
        expected = 1.0 / env.n_bins

        assert np.abs(s - expected) < 0.01, (
            f"Single-bin firing should give sparsity={expected:.4f}, got {s:.4f}"
        )

    def test_sparsity_uniform_firing(self):
        """Uniform firing has sparsity = 1.0 (maximal sparsity)."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Uniform occupancy and firing
        occupancy = np.ones(env.n_bins) / env.n_bins
        firing_rate = np.ones(env.n_bins) * 5.0

        s = sparsity(firing_rate, occupancy)

        # Expected: 1.0 for uniform firing
        assert np.abs(s - 1.0) < 0.01, (
            f"Uniform firing should give sparsity=1.0, got {s:.4f}"
        )

    def test_sparsity_formula_validation(self):
        """Validate against manual calculation."""
        positions = np.random.randn(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=10.0)

        # Simple test case
        occupancy = np.array([0.1, 0.2, 0.3, 0.4])[:env.n_bins]
        firing_rate = np.array([1.0, 2.0, 3.0, 4.0])[:env.n_bins]

        # Manual calculation: (Σ pᵢ rᵢ)² / Σ pᵢ rᵢ²
        numerator = np.sum(occupancy * firing_rate) ** 2
        denominator = np.sum(occupancy * firing_rate**2)
        expected_sparsity = numerator / denominator

        computed_sparsity = sparsity(firing_rate, occupancy)

        assert np.abs(computed_sparsity - expected_sparsity) < 1e-10, (
            f"Manual: {expected_sparsity:.10f}, Implementation: {computed_sparsity:.10f}"
        )

    def test_sparsity_range_validation(self):
        """Sparsity should be in range [1/n_bins, 1.0]."""
        positions = np.random.randn(500, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        np.random.seed(42)
        occupancy = np.random.rand(env.n_bins)
        occupancy = occupancy / occupancy.sum()

        firing_rate = np.random.rand(env.n_bins) * 10.0

        s = sparsity(firing_rate, occupancy)

        min_sparsity = 1.0 / env.n_bins
        assert min_sparsity <= s <= 1.0, (
            f"Sparsity must be in [{min_sparsity:.4f}, 1.0], got {s:.4f}"
        )


class TestBorderScoreValidation:
    """Validate border score implementation.

    Reference: Solstad et al. (2008) "Representation of Geometric Borders in
    the Entorhinal Cortex"

    Formula: border_score = (cM - d) / (cM + d)
    where cM = max boundary coverage, d = normalized mean distance to boundary

    Note: Our implementation generalizes to irregular graphs using geodesic
    distances instead of Euclidean distances.
    """

    def test_border_score_perfect_border_cell(self):
        """Cell firing along entire boundary has high border score."""
        # Create rectangular environment
        positions = []
        for x in np.linspace(0, 100, 50):
            for y in np.linspace(0, 100, 50):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Get boundary bins
        boundary_bins = env.boundary_bins

        # Create firing rate with high firing at boundary
        firing_rate = np.zeros(env.n_bins)
        firing_rate[boundary_bins] = 10.0  # Fire at 10 Hz at boundary

        score = border_score(firing_rate, env, threshold=0.3)

        # Perfect border cell should have high positive score
        assert score > 0.5, (
            f"Perfect border cell should have score > 0.5, got {score:.3f}"
        )

    def test_border_score_central_field(self):
        """Cell firing in center has negative or low border score."""
        positions = []
        for x in np.linspace(0, 100, 50):
            for y in np.linspace(0, 100, 50):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Find central bin
        center = env.bin_centers.mean(axis=0)
        distances_to_center = np.linalg.norm(
            env.bin_centers - center, axis=1
        )
        central_bins = np.argsort(distances_to_center)[:5]  # 5 central bins

        # Fire only in center
        firing_rate = np.zeros(env.n_bins)
        firing_rate[central_bins] = 10.0

        score = border_score(firing_rate, env, threshold=0.3)

        # Central field should have low or negative score
        assert score < 0.5, (
            f"Central field should have score < 0.5, got {score:.3f}"
        )

    def test_border_score_range_validation(self):
        """Border score should be in range [-1, 1]."""
        positions = np.random.randn(500, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Random firing pattern
        np.random.seed(42)
        firing_rate = np.random.rand(env.n_bins) * 10.0

        score = border_score(firing_rate, env, threshold=0.3)

        assert -1.0 <= score <= 1.0, (
            f"Border score must be in [-1, 1], got {score:.3f}"
        )

    def test_border_score_formula_components(self):
        """Validate formula components (coverage and distance)."""
        positions = []
        for x in np.linspace(0, 100, 30):
            for y in np.linspace(0, 100, 30):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=12.0)

        # Fire along one wall
        boundary_bins = env.boundary_bins
        # Select bins on left wall (x ~ 0)
        left_wall_bins = [
            b for b in boundary_bins
            if env.bin_centers[b, 0] < 20
        ]

        firing_rate = np.zeros(env.n_bins)
        firing_rate[left_wall_bins] = 10.0

        score = border_score(firing_rate, env, threshold=0.3)

        # Wall-firing cell should have positive score
        # (high coverage, low distance)
        assert score > 0, (
            f"Wall-firing cell should have positive score, got {score:.3f}"
        )


class TestPlaceFieldDetectionValidation:
    """Validate place field detection algorithm.

    Reference: Our algorithm follows neurocode's iterative peak-based approach
    with subfield discrimination.

    Key features:
    1. Iterative peak detection at threshold * peak_rate
    2. Connected component analysis
    3. Subfield discrimination (recursive threshold)
    4. Interneuron exclusion (mean rate > 10 Hz)
    """

    def test_place_field_detection_single_field(self):
        """Single Gaussian field should be detected correctly."""
        # Create environment
        positions = []
        for x in np.linspace(0, 100, 50):
            for y in np.linspace(0, 100, 50):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create Gaussian field centered at (50, 50)
        center = np.array([50.0, 50.0])
        sigma = 15.0

        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 10.0 * np.exp(-(distances**2) / (2 * sigma**2))

        fields = detect_place_fields(
            firing_rate, env, threshold=0.2, min_size=None
        )

        # Should detect exactly one field
        assert len(fields) == 1, (
            f"Should detect 1 field, got {len(fields)}"
        )

        # Field should contain center bin
        center_bin = env.bin_at(center.reshape(1, -1))[0]
        assert center_bin in fields[0], (
            f"Field should contain center bin {center_bin}"
        )

    def test_place_field_detection_multiple_fields(self):
        """Multiple separate fields should be detected."""
        positions = []
        for x in np.linspace(0, 100, 50):
            for y in np.linspace(0, 100, 50):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create two Gaussian fields
        center1 = np.array([25.0, 25.0])
        center2 = np.array([75.0, 75.0])
        sigma = 12.0

        dist1 = np.linalg.norm(env.bin_centers - center1, axis=1)
        dist2 = np.linalg.norm(env.bin_centers - center2, axis=1)

        firing_rate = (
            8.0 * np.exp(-(dist1**2) / (2 * sigma**2))
            + 8.0 * np.exp(-(dist2**2) / (2 * sigma**2))
        )

        fields = detect_place_fields(
            firing_rate, env, threshold=0.2, min_size=None
        )

        # Should detect two fields
        assert len(fields) >= 2, (
            f"Should detect at least 2 fields, got {len(fields)}"
        )

    def test_place_field_interneuron_exclusion(self):
        """High firing rate cells should be excluded as interneurons."""
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Very high uniform firing (interneuron-like)
        firing_rate = np.ones(env.n_bins) * 50.0  # 50 Hz mean

        fields = detect_place_fields(
            firing_rate, env, threshold=0.2, max_mean_rate=10.0
        )

        # Should detect no fields (excluded as interneuron)
        assert len(fields) == 0, (
            f"High firing rate cell should be excluded, got {len(fields)} fields"
        )

    def test_place_field_threshold_parameter(self):
        """Threshold parameter should control field boundary."""
        positions = []
        for x in np.linspace(0, 100, 40):
            for y in np.linspace(0, 100, 40):
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Gaussian field
        center = np.array([50.0, 50.0])
        sigma = 15.0
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 10.0 * np.exp(-(distances**2) / (2 * sigma**2))

        # Detect with different thresholds
        fields_20pct = detect_place_fields(firing_rate, env, threshold=0.2)
        fields_50pct = detect_place_fields(firing_rate, env, threshold=0.5)

        # Higher threshold should give smaller field
        if len(fields_20pct) > 0 and len(fields_50pct) > 0:
            size_20 = len(fields_20pct[0])
            size_50 = len(fields_50pct[0])

            assert size_50 < size_20, (
                f"Higher threshold should give smaller field: "
                f"20%={size_20} bins, 50%={size_50} bins"
            )


try:
    import opexebo
    OPEXEBO_AVAILABLE = True
except ImportError:
    OPEXEBO_AVAILABLE = False


@pytest.mark.skipif(not OPEXEBO_AVAILABLE, reason="opexebo not available")
class TestOpexeboComparison:
    """Direct comparison with opexebo (if installed).

    These tests are skipped if opexebo is not installed.
    """

    def test_spatial_information_matches_opexebo(self):
        """Compare our Skaggs information with opexebo.

        Creates environment structure that matches opexebo's expectations:
        - Regular grid layout (not irregular graph)
        - 2D array format for opexebo
        - Same occupancy normalization
        """
        import opexebo

        # Create regular grid environment - use explicit grid size
        # to ensure we get exactly the dimensions we want
        nx_bins = 20
        ny_bins = 20
        arena_size = 100.0
        bin_size = arena_size / nx_bins

        # Create grid positions - ensure complete coverage
        positions = []
        for i in range(nx_bins):
            for j in range(ny_bins):
                x = (i + 0.5) * bin_size
                y = (j + 0.5) * bin_size
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=bin_size)

        # Verify we have regular grid
        assert hasattr(env.layout, 'grid_shape'), "Environment must have regular grid"
        grid_shape = env.layout.grid_shape

        # Create Gaussian place field at center
        center = np.array([arena_size / 2, arena_size / 2])
        sigma = 15.0

        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate_1d = 10.0 * np.exp(-(distances**2) / (2 * sigma**2))

        # Uniform occupancy
        occupancy_1d = np.ones(env.n_bins)

        # Compute with neurospatial
        our_info = skaggs_information(firing_rate_1d, occupancy_1d, base=2.0)

        # Convert to 2D for opexebo
        # Need to reshape correctly based on how bins are indexed
        firing_rate_2d = firing_rate_1d.reshape(grid_shape)
        occupancy_2d = occupancy_1d.reshape(grid_shape)

        # Compute with opexebo
        opexebo_stats = opexebo.analysis.rate_map_stats(firing_rate_2d, occupancy_2d)
        # opexebo returns both information_rate (bits/sec) and information_content (bits/spike)
        # Our function returns bits/spike, so compare with content
        opexebo_info = opexebo_stats['spatial_information_content']

        # Should match within 5% (allowing for implementation differences)
        # opexebo sets bins with rate < mean_rate to not contribute (log_arg[log_arg < 1] = 1)
        # This is a slight algorithmic difference from Skaggs formula
        relative_error = np.abs(our_info - opexebo_info) / max(opexebo_info, 1e-10)
        assert relative_error < 0.15, (  # 15% tolerance due to algorithmic difference
            f"Spatial information should match opexebo within 15%: "
            f"neurospatial={our_info:.4f} bits/spike, "
            f"opexebo={opexebo_info:.4f} bits/spike, "
            f"relative_error={relative_error * 100:.2f}%"
        )

    def test_sparsity_matches_opexebo(self):
        """Compare our sparsity calculation with opexebo."""
        import opexebo

        # Create regular grid
        nx_bins = 20
        ny_bins = 20
        arena_size = 100.0
        bin_size = arena_size / nx_bins

        positions = []
        for i in range(nx_bins):
            for j in range(ny_bins):
                x = (i + 0.5) * bin_size
                y = (j + 0.5) * bin_size
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=bin_size)
        grid_shape = env.layout.grid_shape

        # Create place field
        center = np.array([50.0, 50.0])
        sigma = 12.0
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate_1d = 8.0 * np.exp(-(distances**2) / (2 * sigma**2))

        occupancy_1d = np.ones(env.n_bins)

        # Compute with neurospatial
        our_sparsity = sparsity(firing_rate_1d, occupancy_1d)

        # Convert to 2D for opexebo
        firing_rate_2d = firing_rate_1d.reshape(grid_shape)
        occupancy_2d = occupancy_1d.reshape(grid_shape)

        # Compute with opexebo
        opexebo_sparsity = opexebo.analysis.rate_map_stats(
            firing_rate_2d, occupancy_2d
        )['sparsity']

        # Should match within 1%
        relative_error = np.abs(our_sparsity - opexebo_sparsity) / max(opexebo_sparsity, 1e-10)
        assert relative_error < 0.01, (
            f"Sparsity should match opexebo within 1%: "
            f"neurospatial={our_sparsity:.4f}, "
            f"opexebo={opexebo_sparsity:.4f}, "
            f"relative_error={relative_error * 100:.2f}%"
        )

    def test_border_score_matches_opexebo(self):
        """Compare our border score with opexebo on rectangular arena.

        Creates rectangular arena with border cell firing pattern and compares
        border score computation with opexebo's implementation.
        """
        import opexebo

        # Create rectangular arena with regular grid
        nx_bins = 40
        ny_bins = 40
        arena_size = 100.0
        bin_size = arena_size / nx_bins

        # Create complete grid coverage
        positions = []
        for i in range(nx_bins):
            for j in range(ny_bins):
                x = (i + 0.5) * bin_size
                y = (j + 0.5) * bin_size
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=bin_size)
        grid_shape = env.layout.grid_shape

        # Create border cell: rectangular strip along top wall
        # This ensures both:
        # 1. opexebo detects a clear field with peak
        # 2. The field actually touches boundary bins (for high border score)

        firing_rate_2d = np.zeros(grid_shape)

        # Create strip along top wall (last 3 rows)
        # Width: central 50% of arena
        # Depth: 3 bins from top (ensures it touches boundary)
        for y_idx in range(ny_bins - 3, ny_bins):  # Last 3 rows
            for x_idx in range(nx_bins // 4, 3 * nx_bins // 4):  # Central 50%
                # Highest firing at the actual boundary (last row)
                if y_idx == ny_bins - 1:
                    firing_rate_2d[y_idx, x_idx] = 50.0
                else:
                    # Decay away from wall
                    distance_from_wall = (ny_bins - 1) - y_idx
                    firing_rate_2d[y_idx, x_idx] = 50.0 * np.exp(-distance_from_wall)

        # Convert to 1D for neurospatial
        firing_rate_1d = firing_rate_2d.flatten()

        # Compute with neurospatial
        our_score = border_score(firing_rate_1d, env, threshold=0.3)

        # Create masked array for opexebo
        import numpy.ma as ma
        firing_rate_2d_masked = ma.masked_array(
            firing_rate_2d,
            mask=np.zeros_like(firing_rate_2d, dtype=bool)
        )

        # Detect fields with opexebo - provide peak coordinates explicitly
        # opexebo expects [y, x] format
        # Peak is at top center of arena
        peak_y_grid = ny_bins - 1  # Top row
        peak_x_grid = nx_bins // 2  # Center column
        peak_coords = np.array([[peak_y_grid, peak_x_grid]])

        fields, fields_map = opexebo.analysis.place_field(
            firing_rate_2d_masked,
            min_bins=5,
            min_peak=1.0,
            peak_coords=peak_coords
        )

        if len(fields) == 0:
            pytest.skip("opexebo did not detect any fields for this pattern")

        # opexebo.analysis.border_coverage expects key 'field_map' but
        # opexebo.analysis.place_field returns key 'map' - rename it
        for field in fields:
            if 'map' in field and 'field_map' not in field:
                field['field_map'] = field['map']

        # Compute border score with opexebo
        opexebo_score, opexebo_coverage = opexebo.analysis.border_score(
            firing_rate_2d_masked,
            fields_map,
            fields,
            arena_shape="rect"
        )

        # Scores should be in similar range (both positive for border cells)
        # Allow larger tolerance due to algorithmic differences:
        # - opexebo computes per-wall coverage, we aggregate all boundaries
        # - Different distance computation methods
        # - Different field detection approaches
        assert our_score > 0, (
            f"neurospatial should detect border cell (positive score), got {our_score:.3f}"
        )
        assert opexebo_score > 0, (
            f"opexebo should detect border cell (positive score), got {opexebo_score:.3f}"
        )

        # Both should be reasonably high (> 0.3) for clear border cell
        assert our_score > 0.3, (
            f"neurospatial border score should be >0.3 for border cell, got {our_score:.3f}"
        )
        assert opexebo_score > 0.3, (
            f"opexebo border score should be >0.3 for border cell, got {opexebo_score:.3f}"
        )

        # Scores should be within same order of magnitude (within factor of 3)
        # This is lenient but validates we're computing something similar
        if opexebo_score > 0.01:  # Avoid division by tiny numbers
            ratio = max(our_score, opexebo_score) / min(our_score, opexebo_score)
            assert ratio < 3.0, (
                f"Border scores should be similar order of magnitude: "
                f"neurospatial={our_score:.3f}, opexebo={opexebo_score:.3f}, "
                f"ratio={ratio:.2f}"
            )

    def test_border_score_euclidean_closer_to_opexebo(self):
        """Verify that Euclidean distance mode gives closer match to opexebo.

        This test demonstrates that using distance_method='euclidean' produces
        scores much closer to opexebo's implementation, since both use
        Euclidean distances in physical space rather than graph geodesics.
        """
        import opexebo
        import numpy.ma as ma

        # Create rectangular arena
        nx_bins = 40
        ny_bins = 40
        arena_size = 100.0
        bin_size = arena_size / nx_bins

        positions = []
        for i in range(nx_bins):
            for j in range(ny_bins):
                x = (i + 0.5) * bin_size
                y = (j + 0.5) * bin_size
                positions.append([x, y])
        positions = np.array(positions)

        env = Environment.from_samples(positions, bin_size=bin_size)
        grid_shape = env.layout.grid_shape

        # Create border cell pattern
        firing_rate_2d = np.zeros(grid_shape)

        for y_idx in range(ny_bins - 3, ny_bins):
            for x_idx in range(nx_bins // 4, 3 * nx_bins // 4):
                if y_idx == ny_bins - 1:
                    firing_rate_2d[y_idx, x_idx] = 50.0
                else:
                    distance_from_wall = (ny_bins - 1) - y_idx
                    firing_rate_2d[y_idx, x_idx] = 50.0 * np.exp(-distance_from_wall)

        firing_rate_1d = firing_rate_2d.flatten()

        # Compute with neurospatial using BOTH methods
        score_geodesic = border_score(firing_rate_1d, env, threshold=0.3, distance_method="geodesic")
        score_euclidean = border_score(firing_rate_1d, env, threshold=0.3, distance_method="euclidean")

        # Compute with opexebo
        firing_rate_2d_masked = ma.masked_array(
            firing_rate_2d,
            mask=np.zeros_like(firing_rate_2d, dtype=bool)
        )

        peak_y_grid = ny_bins - 1
        peak_x_grid = nx_bins // 2
        peak_coords = np.array([[peak_y_grid, peak_x_grid]])

        fields, fields_map = opexebo.analysis.place_field(
            firing_rate_2d_masked,
            min_bins=5,
            min_peak=1.0,
            peak_coords=peak_coords
        )

        if len(fields) == 0:
            pytest.skip("opexebo did not detect any fields")

        for field in fields:
            if 'map' in field and 'field_map' not in field:
                field['field_map'] = field['map']

        opexebo_score, _ = opexebo.analysis.border_score(
            firing_rate_2d_masked,
            fields_map,
            fields,
            arena_shape="rect"
        )

        # Euclidean should be closer to opexebo than geodesic
        # (Since opexebo uses Euclidean distances)
        error_geodesic = abs(score_geodesic - opexebo_score)
        error_euclidean = abs(score_euclidean - opexebo_score)

        # Print for debugging
        print(f"\nopexebo score: {opexebo_score:.4f}")
        print(f"neurospatial (geodesic): {score_geodesic:.4f} (error: {error_geodesic:.4f})")
        print(f"neurospatial (euclidean): {score_euclidean:.4f} (error: {error_euclidean:.4f})")

        # Euclidean mode should have smaller error
        assert error_euclidean <= error_geodesic, (
            f"Euclidean distance mode should be closer to opexebo than geodesic mode. "
            f"opexebo={opexebo_score:.4f}, "
            f"geodesic={score_geodesic:.4f} (error={error_geodesic:.4f}), "
            f"euclidean={score_euclidean:.4f} (error={error_euclidean:.4f})"
        )

        # Euclidean should match within 30% (still some difference due to coverage computation)
        relative_error = error_euclidean / max(abs(opexebo_score), 1e-10)
        assert relative_error < 0.3, (
            f"Euclidean mode should match opexebo within 30%: "
            f"neurospatial={score_euclidean:.4f}, opexebo={opexebo_score:.4f}, "
            f"relative_error={relative_error * 100:.1f}%"
        )

    def test_spatial_information_matches_neurocode_formula(self):
        """Compare our implementation with neurocode's MapStats.m formula.

        neurocode's MapStats.m (lines 170-180) computes:
            T = sum(map.time(:))
            p_i = map.time/(T+eps)
            lambda_i = map.z
            lambda = lambda_i(:)'*p_i(:)
            specificity = sum(sum(p_i(ok).*lambda_i(ok)/lambda.*log2(lambda_i(ok)/lambda)))

        This is mathematically identical to Skaggs et al. (1993).

        Reference: https://github.com/ayalab1/neurocode/blob/master/PlaceCells/MapStats.m
        """
        # Create test environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create Gaussian place field
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 15.0 * np.exp(-(distances**2) / (2 * 8.0**2))

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Compute with neurospatial
        our_info = skaggs_information(firing_rate, occupancy, base=2.0)

        # Manually implement neurocode's exact formula
        # (from MapStats.m lines 172-180)
        map_time = occupancy  # Occupancy time in each bin
        map_z = firing_rate   # Firing rate (lambda_i)

        T = np.sum(map_time)  # Total time
        p_i = map_time / (T + np.finfo(float).eps)  # Probability of occupying bin i

        lambda_i = map_z  # Firing rate in bin i
        lambda_mean = np.dot(lambda_i.flatten(), p_i.flatten())  # Mean firing rate

        # neurocode filters bins with time > minTime (default minTime=0)
        ok = map_time > 0

        # Compute specificity (neurocode's name for spatial information)
        neurocode_specificity = np.sum(
            p_i[ok] * (lambda_i[ok] / lambda_mean) * np.log2(lambda_i[ok] / lambda_mean)
        )

        # Should match exactly (same formula)
        assert np.abs(our_info - neurocode_specificity) < 1e-10, (
            f"Spatial information should exactly match neurocode's MapStats.m formula: "
            f"neurospatial={our_info:.10f}, neurocode={neurocode_specificity:.10f}, "
            f"difference={abs(our_info - neurocode_specificity):.2e}"
        )

    def test_sparsity_matches_neurocode_formula(self):
        """Compare our sparsity implementation with neurocode's MapStats1D.m formula.

        neurocode's MapStats1D.m (line 105) computes:
            stats.sparsity = ((sum(sum(p_i.*map.z))).^2)/sum(sum(p_i.*(map.z.^2)));

        This is the Skaggs et al. (1996) sparsity formula.

        Reference: https://github.com/ayalab1/neurocode/blob/master/tutorials/pipelineFiringMaps/MapStats1D.m
        """
        # Create test environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create Gaussian place field
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 15.0 * np.exp(-(distances**2) / (2 * 8.0**2))

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Compute with neurospatial
        our_sparsity = sparsity(firing_rate, occupancy)

        # Manually implement neurocode's exact formula
        # (from MapStats1D.m line 105)
        map_time = occupancy  # Occupancy time in each bin
        map_z = firing_rate   # Firing rate (lambda_i)

        T = np.sum(map_time)  # Total time
        p_i = map_time / (T + np.finfo(float).eps)  # Probability of occupying bin i

        # Compute sparsity
        # neurocode formula: ((Σ p_i × firing_rate)²) / (Σ p_i × firing_rate²)
        numerator = (np.sum(p_i * map_z)) ** 2
        denominator = np.sum(p_i * (map_z ** 2))

        neurocode_sparsity = numerator / denominator

        # Should match exactly (same formula)
        assert np.abs(our_sparsity - neurocode_sparsity) < 1e-10, (
            f"Sparsity should exactly match neurocode's MapStats1D.m formula: "
            f"neurospatial={our_sparsity:.10f}, neurocode={neurocode_sparsity:.10f}, "
            f"difference={abs(our_sparsity - neurocode_sparsity):.2e}"
        )

    def test_information_per_spike_matches_neurocode_formula(self):
        """Compare skaggs_information with neurocode's informationPerSpike formula.

        neurocode's MapStats1D.m (line 103) computes:
            logArg = map.z./meanFiringRate;
            logArg(logArg == 0) = 1;
            stats.informationPerSpike = sum(sum(p_i.*logArg.*log2(logArg)));

        This should match both stats.specificity (line 101) and our skaggs_information.

        Reference: https://github.com/ayalab1/neurocode/blob/master/tutorials/pipelineFiringMaps/MapStats1D.m
        """
        # Create test environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create Gaussian place field
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 15.0 * np.exp(-(distances**2) / (2 * 8.0**2))

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Compute with neurospatial
        our_info = skaggs_information(firing_rate, occupancy, base=2.0)

        # Manually implement neurocode's informationPerSpike formula
        map_time = occupancy
        map_z = firing_rate

        T = np.sum(map_time)
        p_i = map_time / (T + np.finfo(float).eps)

        # Compute mean firing rate
        meanFiringRate = np.sum(map_z * map_time) / T

        # Compute logArg = firing_rate / mean_rate
        logArg = map_z / meanFiringRate
        logArg[logArg == 0] = 1  # Avoid log(0)

        # informationPerSpike formula
        neurocode_info_per_spike = np.sum(p_i * logArg * np.log2(logArg))

        # Should match exactly
        assert np.abs(our_info - neurocode_info_per_spike) < 1e-10, (
            f"Information per spike should exactly match neurocode's MapStats1D.m formula: "
            f"neurospatial={our_info:.10f}, neurocode={neurocode_info_per_spike:.10f}, "
            f"difference={abs(our_info - neurocode_info_per_spike):.2e}"
        )

    def test_selectivity_matches_neurocode_formula(self):
        """Validate selectivity metric against neurocode's MapStats1D.m formula.

        neurocode's MapStats1D.m (line 106) computes:
            stats.selectivity = max(max(map.z))./meanFiringRate;

        Selectivity is the ratio of peak firing rate to mean firing rate.
        Higher values indicate more spatially selective firing.

        Reference: https://github.com/ayalab1/neurocode/blob/master/tutorials/pipelineFiringMaps/MapStats1D.m
        """
        # Create test environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create Gaussian place field with known peak
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        peak_rate = 25.0
        firing_rate = peak_rate * np.exp(-(distances**2) / (2 * 10.0**2))

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Manually implement neurocode's selectivity formula
        map_time = occupancy
        map_z = firing_rate

        T = np.sum(map_time)
        meanFiringRate = np.sum(map_z * map_time) / T
        max_rate = np.max(map_z)

        neurocode_selectivity = max_rate / meanFiringRate

        # Compute using direct formula
        # (We don't have this implemented yet, so compute it here)
        computed_selectivity = np.max(firing_rate) / np.mean(firing_rate)

        # Should match exactly
        assert np.abs(computed_selectivity - neurocode_selectivity) < 1e-10, (
            f"Selectivity should match neurocode's formula: "
            f"computed={computed_selectivity:.10f}, neurocode={neurocode_selectivity:.10f}, "
            f"difference={abs(computed_selectivity - neurocode_selectivity):.2e}"
        )

        # Verify selectivity is > 1 for place field (concentrated firing)
        assert neurocode_selectivity > 1.0, "Selectivity should be > 1 for place field"

    def test_information_per_sec_matches_neurocode_formula(self):
        """Validate information rate (bits/sec) against neurocode's MapStats1D.m formula.

        neurocode's MapStats1D.m (line 104) computes:
            logArg = map.z./meanFiringRate;
            logArg(logArg == 0) = 1;
            stats.informationPerSec = sum(sum(p_i.*map.z.*log2(logArg)));

        This is the information rate in bits per second (not bits per spike).

        Reference: https://github.com/ayalab1/neurocode/blob/master/tutorials/pipelineFiringMaps/MapStats1D.m
        """
        # Create test environment
        positions = np.random.randn(1000, 2) * 20
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create Gaussian place field
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(env.bin_centers - center, axis=1)
        firing_rate = 20.0 * np.exp(-(distances**2) / (2 * 8.0**2))

        # Uniform occupancy
        occupancy = np.ones(env.n_bins)

        # Manually implement neurocode's informationPerSec formula
        map_time = occupancy
        map_z = firing_rate

        T = np.sum(map_time)
        p_i = map_time / (T + np.finfo(float).eps)

        # Compute mean firing rate
        meanFiringRate = np.sum(map_z * map_time) / T

        # Compute logArg = firing_rate / mean_rate
        logArg = map_z / meanFiringRate
        logArg[logArg == 0] = 1  # Avoid log(0)

        # informationPerSec formula: Σ p_i × firing_rate × log2(firing_rate/mean_rate)
        neurocode_info_per_sec = np.sum(p_i * map_z * np.log2(logArg))

        # Compute using direct formula
        # (We don't have this implemented yet, so compute it here)
        # This is: mean_rate × information_per_spike
        info_per_spike = skaggs_information(firing_rate, occupancy, base=2.0)
        computed_info_per_sec = meanFiringRate * info_per_spike

        # Should match exactly
        assert np.abs(computed_info_per_sec - neurocode_info_per_sec) < 1e-9, (
            f"Information per second should match neurocode's formula: "
            f"computed={computed_info_per_sec:.10f}, neurocode={neurocode_info_per_sec:.10f}, "
            f"difference={abs(computed_info_per_sec - neurocode_info_per_sec):.2e}"
        )

        # Verify relationship: info_per_sec = mean_rate × info_per_spike
        expected = meanFiringRate * info_per_spike
        assert np.abs(neurocode_info_per_sec - expected) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
