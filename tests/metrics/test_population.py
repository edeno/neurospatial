"""Tests for population-level place field metrics."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.metrics.population import (
    count_place_cells,
    field_density_map,
    field_overlap,
    population_coverage,
    population_vector_correlation,
)


class TestPopulationCoverage:
    """Test population_coverage function."""

    def test_population_coverage_basic(self) -> None:
        """Test basic population coverage calculation."""
        n_bins = 100
        # Two fields covering different bins
        all_place_fields = [
            [np.array([0, 1, 2, 3, 4])],  # Cell 1: 5 bins
            [np.array([10, 11, 12])],  # Cell 2: 3 bins
        ]
        coverage = population_coverage(all_place_fields, n_bins)
        # 8 unique bins out of 100 = 0.08
        assert coverage == pytest.approx(0.08)

    def test_population_coverage_overlap(self) -> None:
        """Test coverage with overlapping fields."""
        n_bins = 50
        # Two cells with overlapping fields
        all_place_fields = [
            [np.array([0, 1, 2, 3])],  # Cell 1: bins 0-3
            [np.array([2, 3, 4, 5])],  # Cell 2: bins 2-5 (overlap at 2,3)
        ]
        coverage = population_coverage(all_place_fields, n_bins)
        # 6 unique bins (0,1,2,3,4,5) out of 50
        assert coverage == pytest.approx(6 / 50)

    def test_population_coverage_no_fields(self) -> None:
        """Test coverage when no fields detected."""
        n_bins = 100
        all_place_fields = [[], []]  # No fields for any cells
        coverage = population_coverage(all_place_fields, n_bins)
        assert coverage == 0.0

    def test_population_coverage_multiple_fields_per_cell(self) -> None:
        """Test coverage when cells have multiple fields."""
        n_bins = 100
        all_place_fields = [
            [np.array([0, 1]), np.array([10, 11])],  # Cell 1: 2 fields
            [np.array([20, 21, 22])],  # Cell 2: 1 field
        ]
        coverage = population_coverage(all_place_fields, n_bins)
        # 7 unique bins out of 100
        assert coverage == pytest.approx(0.07)

    def test_population_coverage_full(self) -> None:
        """Test coverage when all bins are covered."""
        n_bins = 10
        # Single cell covering all bins
        all_place_fields = [
            [np.arange(n_bins)],
        ]
        coverage = population_coverage(all_place_fields, n_bins)
        assert coverage == pytest.approx(1.0)


class TestFieldDensityMap:
    """Test field_density_map function."""

    def test_field_density_map_no_overlap(self) -> None:
        """Test density map with non-overlapping fields."""
        n_bins = 10
        all_place_fields = [
            [np.array([0, 1, 2])],  # Cell 1
            [np.array([5, 6, 7])],  # Cell 2
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density.shape == (n_bins,)
        # Each field bin should have count 1
        assert density[0] == 1
        assert density[1] == 1
        assert density[2] == 1
        assert density[3] == 0
        assert density[5] == 1
        # Non-field bins should be 0
        assert density[4] == 0
        assert density[8] == 0

    def test_field_density_map_overlap(self) -> None:
        """Test density map with overlapping fields."""
        n_bins = 10
        all_place_fields = [
            [np.array([2, 3, 4])],  # Cell 1
            [np.array([3, 4, 5])],  # Cell 2 (overlaps at 3, 4)
            [np.array([4, 5, 6])],  # Cell 3 (overlaps at 4, 5)
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density[2] == 1  # Only cell 1
        assert density[3] == 2  # Cells 1, 2
        assert density[4] == 3  # Cells 1, 2, 3
        assert density[5] == 2  # Cells 2, 3
        assert density[6] == 1  # Only cell 3
        assert density[0] == 0  # No fields

    def test_field_density_map_multiple_fields(self) -> None:
        """Test density map when cells have multiple fields."""
        n_bins = 20
        all_place_fields = [
            [np.array([0, 1]), np.array([10, 11])],  # Cell 1: 2 fields
            [np.array([1, 2])],  # Cell 2: overlaps first field
        ]
        density = field_density_map(all_place_fields, n_bins)
        assert density[0] == 1  # Cell 1 field 1
        assert density[1] == 2  # Both cells
        assert density[2] == 1  # Cell 2 only
        assert density[10] == 1  # Cell 1 field 2
        assert density[11] == 1  # Cell 1 field 2


class TestCountPlaceCells:
    """Test count_place_cells function."""

    def test_count_place_cells_above_threshold(self) -> None:
        """Test counting cells above information threshold."""
        spatial_information = np.array([0.2, 0.8, 1.5, 0.3, 1.2])
        count = count_place_cells(spatial_information, threshold=0.5)
        # 3 cells have information > 0.5
        assert count == 3

    def test_count_place_cells_default_threshold(self) -> None:
        """Test default threshold of 0.5 bits/spike."""
        spatial_information = np.array([0.3, 0.5, 0.6, 0.4, 1.0])
        count = count_place_cells(spatial_information)
        # Default threshold is 0.5, so count cells with > 0.5
        # Values: 0.6, 1.0 are above threshold
        assert count == 2

    def test_count_place_cells_none_qualify(self) -> None:
        """Test when no cells exceed threshold."""
        spatial_information = np.array([0.1, 0.2, 0.3])
        count = count_place_cells(spatial_information, threshold=0.5)
        assert count == 0

    def test_count_place_cells_all_qualify(self) -> None:
        """Test when all cells exceed threshold."""
        spatial_information = np.array([1.0, 1.5, 2.0, 0.8])
        count = count_place_cells(spatial_information, threshold=0.5)
        assert count == 4

    def test_count_place_cells_boundary(self) -> None:
        """Test behavior at threshold boundary."""
        spatial_information = np.array([0.5, 0.50001, 0.49999])
        count = count_place_cells(spatial_information, threshold=0.5)
        # Only values strictly greater than threshold
        assert count == 1  # Only 0.50001

    def test_count_place_cells_nan_handling(self) -> None:
        """Test handling of NaN values in spatial information."""
        spatial_information = np.array([0.8, np.nan, 1.2, 0.3, np.nan])
        count = count_place_cells(spatial_information, threshold=0.5)
        # NaN values should be excluded from count
        assert count == 2  # 0.8 and 1.2


class TestFieldOverlap:
    """Test field_overlap function."""

    def test_field_overlap_identical(self) -> None:
        """Test overlap of identical fields (Jaccard = 1.0)."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([0, 1, 2, 3])
        overlap = field_overlap(field1, field2)
        assert overlap == pytest.approx(1.0)

    def test_field_overlap_disjoint(self) -> None:
        """Test overlap of completely disjoint fields (Jaccard = 0.0)."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([10, 11, 12])
        overlap = field_overlap(field1, field2)
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_partial(self) -> None:
        """Test partial overlap between fields."""
        field1 = np.array([0, 1, 2, 3])
        field2 = np.array([2, 3, 4, 5])
        overlap = field_overlap(field1, field2)
        # Intersection: {2, 3} = 2 bins
        # Union: {0, 1, 2, 3, 4, 5} = 6 bins
        # Jaccard: 2/6 = 0.333...
        assert overlap == pytest.approx(2 / 6)

    def test_field_overlap_subset(self) -> None:
        """Test when one field is subset of another."""
        field1 = np.array([1, 2])
        field2 = np.array([0, 1, 2, 3, 4])
        overlap = field_overlap(field1, field2)
        # Intersection: {1, 2} = 2
        # Union: {0, 1, 2, 3, 4} = 5
        # Jaccard: 2/5 = 0.4
        assert overlap == pytest.approx(2 / 5)

    def test_field_overlap_empty_fields(self) -> None:
        """Test overlap when one or both fields are empty."""
        field1 = np.array([0, 1, 2])
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(field1, field2)
        # Empty intersection and union = field1
        # By convention, Jaccard = 0 when one set is empty
        assert overlap == pytest.approx(0.0)

    def test_field_overlap_both_empty(self) -> None:
        """Test overlap when both fields are empty."""
        field1 = np.array([], dtype=np.int64)
        field2 = np.array([], dtype=np.int64)
        overlap = field_overlap(field1, field2)
        # By convention, Jaccard(∅, ∅) = 0 (though sometimes defined as 1)
        # We use 0 to avoid division by zero issues
        assert overlap == pytest.approx(0.0)


class TestPopulationVectorCorrelation:
    """Test population_vector_correlation function."""

    def test_population_vector_correlation_shape(self) -> None:
        """Test correlation matrix has correct shape."""
        # 3 cells, 50 bins
        rng = np.random.default_rng(42)
        population_matrix = rng.random((3, 50))
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be 3x3 symmetric matrix
        assert corr_matrix.shape == (3, 3)
        # Check symmetry
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, rtol=1e-10)

    def test_population_vector_correlation_diagonal(self) -> None:
        """Test diagonal elements are 1.0 (self-correlation)."""
        rng = np.random.default_rng(42)
        population_matrix = rng.random((5, 100))
        corr_matrix = population_vector_correlation(population_matrix)
        # Diagonal should be all 1.0
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, rtol=1e-10)

    def test_population_vector_correlation_identical_cells(self) -> None:
        """Test correlation of identical firing patterns."""
        # Two identical cells
        rng = np.random.default_rng(42)
        pattern = rng.random(50)
        population_matrix = np.vstack([pattern, pattern])
        corr_matrix = population_vector_correlation(population_matrix)
        # Off-diagonal should be 1.0 (perfect correlation)
        assert corr_matrix[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert corr_matrix[1, 0] == pytest.approx(1.0, abs=1e-10)

    def test_population_vector_correlation_orthogonal(self) -> None:
        """Test correlation of uncorrelated cells."""
        # Create orthogonal patterns (correlation ~ 0)
        rng = np.random.default_rng(42)
        n_bins = 1000  # Large number for better approximation
        cell1 = rng.standard_normal(n_bins)
        cell2 = rng.standard_normal(n_bins)
        population_matrix = np.vstack([cell1, cell2])
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be close to 0 (not exactly due to randomness)
        assert abs(corr_matrix[0, 1]) < 0.1

    def test_population_vector_correlation_anticorrelated(self) -> None:
        """Test negative correlation (anticorrelated patterns)."""
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Negatively correlated pattern
        anti_pattern = -pattern
        population_matrix = np.vstack([pattern, anti_pattern])
        corr_matrix = population_vector_correlation(population_matrix)
        # Should be -1.0 (perfect anticorrelation)
        assert corr_matrix[0, 1] == pytest.approx(-1.0, abs=1e-10)

    def test_population_vector_correlation_single_cell(self) -> None:
        """Test correlation matrix for single cell."""
        rng = np.random.default_rng(42)
        population_matrix = rng.random((1, 50))
        corr_matrix = population_vector_correlation(population_matrix)
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix[0, 0] == pytest.approx(1.0)

    def test_population_vector_correlation_constant_cells(self) -> None:
        """Test handling of cells with constant firing (zero variance)."""
        # Cell with constant firing rate (correlation undefined)
        rng = np.random.default_rng(42)
        constant_cell = np.ones(50)
        varying_cell = rng.random(50)
        population_matrix = np.vstack([constant_cell, varying_cell])
        corr_matrix = population_vector_correlation(population_matrix)
        # Correlation with constant cell should be NaN
        assert np.isnan(corr_matrix[0, 1])
        assert np.isnan(corr_matrix[1, 0])
        # Self-correlation of constant cell is still 1.0 by definition
        assert corr_matrix[0, 0] == pytest.approx(1.0)


class TestPopulationMetricsIntegration:
    """Integration tests combining multiple population metrics."""

    def test_full_population_workflow(self) -> None:
        """Test complete workflow with all population metrics."""
        n_bins = 100
        n_cells = 5

        # Create synthetic place fields
        all_place_fields = [
            [np.array([10, 11, 12, 13])],  # Cell 0
            [np.array([20, 21, 22])],  # Cell 1
            [np.array([12, 13, 14, 15])],  # Cell 2 (overlaps cell 0)
            [],  # Cell 3 (no field)
            [np.array([50, 51]), np.array([60, 61])],  # Cell 4 (2 fields)
        ]

        # Create synthetic spatial information
        spatial_info = np.array([1.2, 0.8, 1.5, 0.1, 1.0])

        # Create synthetic firing rate matrix
        rng = np.random.default_rng(42)
        population_matrix = rng.random((n_cells, n_bins))

        # Test 1: Population coverage
        coverage = population_coverage(all_place_fields, n_bins)
        # Manual count: 10,11,12,13 + 20,21,22 + 14,15 + 50,51 + 60,61 = 13 unique bins
        assert coverage == pytest.approx(13 / 100)

        # Test 2: Field density map
        density = field_density_map(all_place_fields, n_bins)
        assert density[12] == 2  # Overlap between cells 0 and 2
        assert density[13] == 2  # Overlap between cells 0 and 2
        assert density[10] == 1  # Only cell 0
        assert density[30] == 0  # No fields

        # Test 3: Count place cells
        count = count_place_cells(spatial_info, threshold=0.5)
        assert count == 4  # Cells 0,1,2,4 exceed 0.5

        # Test 4: Field overlap
        overlap_01 = field_overlap(all_place_fields[0][0], all_place_fields[1][0])
        assert overlap_01 == pytest.approx(0.0)  # Disjoint

        overlap_02 = field_overlap(all_place_fields[0][0], all_place_fields[2][0])
        # Intersection: {12, 13} = 2
        # Union: {10,11,12,13,14,15} = 6
        assert overlap_02 == pytest.approx(2 / 6)

        # Test 5: Population vector correlation
        corr_matrix = population_vector_correlation(population_matrix)
        assert corr_matrix.shape == (n_cells, n_cells)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0)
        # Matrix should be symmetric
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, rtol=1e-10)
