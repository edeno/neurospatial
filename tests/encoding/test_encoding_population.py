"""Tests for neurospatial.encoding.population module.

This module tests that population-level metrics are correctly re-exported
from the encoding.population module.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Import Tests - Direct from encoding.population
# =============================================================================


class TestImportFromEncodingPopulation:
    """Test imports from neurospatial.encoding.population."""

    def test_import_PopulationCoverageResult(self) -> None:
        """Test PopulationCoverageResult can be imported from encoding.population."""
        from neurospatial.encoding.population import PopulationCoverageResult

        assert PopulationCoverageResult is not None

    def test_import_population_coverage(self) -> None:
        """Test population_coverage can be imported from encoding.population."""
        from neurospatial.encoding.population import population_coverage

        assert callable(population_coverage)

    def test_import_plot_population_coverage(self) -> None:
        """Test plot_population_coverage can be imported from encoding.population."""
        from neurospatial.encoding.population import plot_population_coverage

        assert callable(plot_population_coverage)

    def test_import_field_density_map(self) -> None:
        """Test field_density_map can be imported from encoding.population."""
        from neurospatial.encoding.population import field_density_map

        assert callable(field_density_map)

    def test_import_count_place_cells(self) -> None:
        """Test count_place_cells can be imported from encoding.population."""
        from neurospatial.encoding.population import count_place_cells

        assert callable(count_place_cells)

    def test_import_field_overlap(self) -> None:
        """Test field_overlap can be imported from encoding.population."""
        from neurospatial.encoding.population import field_overlap

        assert callable(field_overlap)

    def test_import_population_vector_correlation(self) -> None:
        """Test population_vector_correlation can be imported from encoding.population."""
        from neurospatial.encoding.population import population_vector_correlation

        assert callable(population_vector_correlation)


# =============================================================================
# Import Tests - From encoding/__init__.py
# =============================================================================


class TestImportFromEncoding:
    """Test imports from neurospatial.encoding (top-level)."""

    def test_import_PopulationCoverageResult_from_encoding(self) -> None:
        """Test PopulationCoverageResult can be imported from encoding."""
        from neurospatial.encoding import PopulationCoverageResult

        assert PopulationCoverageResult is not None

    def test_import_population_coverage_from_encoding(self) -> None:
        """Test population_coverage can be imported from encoding."""
        from neurospatial.encoding import population_coverage

        assert callable(population_coverage)

    def test_import_plot_population_coverage_from_encoding(self) -> None:
        """Test plot_population_coverage can be imported from encoding."""
        from neurospatial.encoding import plot_population_coverage

        assert callable(plot_population_coverage)

    def test_import_field_density_map_from_encoding(self) -> None:
        """Test field_density_map can be imported from encoding."""
        from neurospatial.encoding import field_density_map

        assert callable(field_density_map)

    def test_import_count_place_cells_from_encoding(self) -> None:
        """Test count_place_cells can be imported from encoding."""
        from neurospatial.encoding import count_place_cells

        assert callable(count_place_cells)

    def test_import_field_overlap_from_encoding(self) -> None:
        """Test field_overlap can be imported from encoding."""
        from neurospatial.encoding import field_overlap

        assert callable(field_overlap)

    def test_import_population_vector_correlation_from_encoding(self) -> None:
        """Test population_vector_correlation can be imported from encoding."""
        from neurospatial.encoding import population_vector_correlation

        assert callable(population_vector_correlation)


# =============================================================================
# Module Structure Tests
# =============================================================================


class TestModuleStructure:
    """Test that encoding.population module is correctly structured."""

    def test_module_has_all_attribute(self) -> None:
        """Test that encoding.population has __all__ defined."""
        import neurospatial.encoding.population as population_module

        assert hasattr(population_module, "__all__")

    def test_all_exports_correct_count(self) -> None:
        """Test that __all__ has the expected number of exports."""
        import neurospatial.encoding.population as population_module

        # 7 symbols: 1 dataclass + 6 functions
        assert len(population_module.__all__) == 7

    def test_all_exports_are_importable(self) -> None:
        """Test that all exports in __all__ are actually importable."""
        population_module = importlib.import_module("neurospatial.encoding.population")

        for name in population_module.__all__:
            assert hasattr(population_module, name), f"Missing export: {name}"


# =============================================================================
# Re-export Identity Tests
# =============================================================================


class TestReExportIdentity:
    """Test that re-exports are identical to original implementations."""

    def test_PopulationCoverageResult_is_same_object(self) -> None:
        """Test PopulationCoverageResult is the same object as in metrics."""
        from neurospatial.encoding.population import PopulationCoverageResult
        from neurospatial.encoding.population import (
            PopulationCoverageResult as OriginalPopulationCoverageResult,
        )

        assert PopulationCoverageResult is OriginalPopulationCoverageResult

    def test_population_coverage_is_same_function(self) -> None:
        """Test population_coverage is the same function as in metrics."""
        from neurospatial.encoding.population import population_coverage
        from neurospatial.encoding.population import (
            population_coverage as original_population_coverage,
        )

        assert population_coverage is original_population_coverage

    def test_plot_population_coverage_is_same_function(self) -> None:
        """Test plot_population_coverage is the same function as in metrics."""
        from neurospatial.encoding.population import plot_population_coverage
        from neurospatial.encoding.population import (
            plot_population_coverage as original_plot_population_coverage,
        )

        assert plot_population_coverage is original_plot_population_coverage

    def test_field_density_map_is_same_function(self) -> None:
        """Test field_density_map is the same function as in metrics."""
        from neurospatial.encoding.population import field_density_map
        from neurospatial.encoding.population import (
            field_density_map as original_field_density_map,
        )

        assert field_density_map is original_field_density_map

    def test_count_place_cells_is_same_function(self) -> None:
        """Test count_place_cells is the same function as in metrics."""
        from neurospatial.encoding.population import count_place_cells
        from neurospatial.encoding.population import (
            count_place_cells as original_count_place_cells,
        )

        assert count_place_cells is original_count_place_cells

    def test_field_overlap_is_same_function(self) -> None:
        """Test field_overlap is the same function as in metrics."""
        from neurospatial.encoding.population import field_overlap
        from neurospatial.encoding.population import (
            field_overlap as original_field_overlap,
        )

        assert field_overlap is original_field_overlap

    def test_population_vector_correlation_is_same_function(self) -> None:
        """Test population_vector_correlation is the same function as in metrics."""
        from neurospatial.encoding.population import population_vector_correlation
        from neurospatial.encoding.population import (
            population_vector_correlation as original_population_vector_correlation,
        )

        assert population_vector_correlation is original_population_vector_correlation


# =============================================================================
# Functionality Tests
# =============================================================================


class TestFunctionality:
    """Test that re-exported functions work correctly."""

    def test_field_density_map_basic(self) -> None:
        """Test field_density_map computes correct density."""
        from neurospatial.encoding.population import field_density_map

        # Three cells with overlapping fields
        all_fields: list[list[NDArray[np.int64]]] = [
            [np.array([2, 3, 4], dtype=np.int64)],  # Cell 1
            [np.array([3, 4, 5], dtype=np.int64)],  # Cell 2 (overlaps at 3, 4)
            [np.array([4, 5, 6], dtype=np.int64)],  # Cell 3 (overlaps at 4, 5)
        ]
        density = field_density_map(all_fields, n_bins=10)

        assert density[4] == 3  # Bin 4 has 3 overlapping fields
        assert density[3] == 2  # Bin 3 has 2 overlapping fields
        assert density[0] == 0  # Bin 0 has no fields

    def test_count_place_cells_basic(self) -> None:
        """Test count_place_cells counts correctly."""
        from neurospatial.encoding.population import count_place_cells

        # Population with mix of place cells and non-selective cells
        spatial_info = np.array([0.2, 0.8, 1.5, 0.3, 1.2, 0.1])
        n_place_cells = count_place_cells(spatial_info, threshold=0.5)

        assert n_place_cells == 3  # 3 cells exceed 0.5 bits/spike

    def test_field_overlap_identical(self) -> None:
        """Test field_overlap returns 1.0 for identical fields."""
        from neurospatial.encoding.population import field_overlap

        field1 = np.array([0, 1, 2, 3], dtype=np.int64)
        field2 = np.array([0, 1, 2, 3], dtype=np.int64)
        overlap = field_overlap(field1, field2)

        assert np.isclose(overlap, 1.0)

    def test_field_overlap_partial(self) -> None:
        """Test field_overlap returns correct value for partial overlap."""
        from neurospatial.encoding.population import field_overlap

        field1 = np.array([0, 1, 2, 3], dtype=np.int64)
        field2 = np.array([2, 3, 4, 5], dtype=np.int64)
        # Intersection: {2, 3}, Union: {0,1,2,3,4,5} -> 2/6 = 0.333...
        overlap = field_overlap(field1, field2)

        assert np.isclose(overlap, 2 / 6)

    def test_field_overlap_disjoint(self) -> None:
        """Test field_overlap returns 0.0 for disjoint fields."""
        from neurospatial.encoding.population import field_overlap

        field1 = np.array([0, 1, 2], dtype=np.int64)
        field2 = np.array([5, 6, 7], dtype=np.int64)
        overlap = field_overlap(field1, field2)

        assert overlap == 0.0

    def test_field_overlap_empty(self) -> None:
        """Test field_overlap handles empty fields."""
        from neurospatial.encoding.population import field_overlap

        field1 = np.array([], dtype=np.int64)
        field2 = np.array([0, 1, 2], dtype=np.int64)
        overlap = field_overlap(field1, field2)

        assert overlap == 0.0

    def test_population_vector_correlation_shape(self) -> None:
        """Test population_vector_correlation returns correct shape."""
        from neurospatial.encoding.population import population_vector_correlation

        np.random.seed(42)
        population_matrix = np.random.rand(5, 50)
        corr_matrix = population_vector_correlation(population_matrix)

        assert corr_matrix.shape == (5, 5)

    def test_population_vector_correlation_diagonal(self) -> None:
        """Test population_vector_correlation has 1.0 on diagonal."""
        from neurospatial.encoding.population import population_vector_correlation

        np.random.seed(42)
        population_matrix = np.random.rand(3, 50)
        corr_matrix = population_vector_correlation(population_matrix)

        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])

    def test_population_vector_correlation_symmetric(self) -> None:
        """Test population_vector_correlation returns symmetric matrix."""
        from neurospatial.encoding.population import population_vector_correlation

        np.random.seed(42)
        population_matrix = np.random.rand(4, 100)
        corr_matrix = population_vector_correlation(population_matrix)

        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
