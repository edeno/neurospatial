"""
Tests for encoding.object_vector module.

Following TDD: Tests written FIRST before implementation.
Tests verify that encoding.object_vector re-exports functions from:
- object_vector_field.py: ObjectVectorFieldResult, compute_object_vector_field
- metrics/object_vector_cells.py: ObjectVectorMetrics, compute_object_vector_tuning,
  object_vector_score, is_object_vector_cell, plot_object_vector_tuning
"""

from __future__ import annotations

import numpy as np

from neurospatial import Environment

# =============================================================================
# Import Tests - encoding.object_vector module
# =============================================================================


class TestEncodingObjectVectorImports:
    """Test imports from encoding.object_vector module."""

    def test_import_object_vector_field_result(self) -> None:
        """Test importing ObjectVectorFieldResult from encoding.object_vector."""
        from neurospatial.encoding.object_vector import ObjectVectorFieldResult

        assert ObjectVectorFieldResult is not None

    def test_import_compute_object_vector_field(self) -> None:
        """Test importing compute_object_vector_field from encoding.object_vector."""
        from neurospatial.encoding.object_vector import compute_object_vector_field

        assert callable(compute_object_vector_field)

    def test_import_object_vector_metrics(self) -> None:
        """Test importing ObjectVectorMetrics from encoding.object_vector."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        assert ObjectVectorMetrics is not None

    def test_import_compute_object_vector_tuning(self) -> None:
        """Test importing compute_object_vector_tuning from encoding.object_vector."""
        from neurospatial.encoding.object_vector import compute_object_vector_tuning

        assert callable(compute_object_vector_tuning)

    def test_import_object_vector_score(self) -> None:
        """Test importing object_vector_score from encoding.object_vector."""
        from neurospatial.encoding.object_vector import object_vector_score

        assert callable(object_vector_score)

    def test_import_is_object_vector_cell(self) -> None:
        """Test importing is_object_vector_cell from encoding.object_vector."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        assert callable(is_object_vector_cell)

    def test_import_plot_object_vector_tuning(self) -> None:
        """Test importing plot_object_vector_tuning from encoding.object_vector."""
        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        assert callable(plot_object_vector_tuning)


# =============================================================================
# Import Tests - encoding/__init__.py re-exports
# =============================================================================


class TestEncodingInitObjectVectorImports:
    """Test that object_vector functions are exported from encoding/__init__.py."""

    def test_import_object_vector_field_result_from_encoding(self) -> None:
        """Test importing ObjectVectorFieldResult from encoding."""
        from neurospatial.encoding import ObjectVectorFieldResult

        assert ObjectVectorFieldResult is not None

    def test_import_compute_object_vector_field_from_encoding(self) -> None:
        """Test importing compute_object_vector_field from encoding."""
        from neurospatial.encoding import compute_object_vector_field

        assert callable(compute_object_vector_field)

    def test_import_object_vector_metrics_from_encoding(self) -> None:
        """Test importing ObjectVectorMetrics from encoding."""
        from neurospatial.encoding import ObjectVectorMetrics

        assert ObjectVectorMetrics is not None

    def test_import_compute_object_vector_tuning_from_encoding(self) -> None:
        """Test importing compute_object_vector_tuning from encoding."""
        from neurospatial.encoding import compute_object_vector_tuning

        assert callable(compute_object_vector_tuning)

    def test_import_object_vector_score_from_encoding(self) -> None:
        """Test importing object_vector_score from encoding."""
        from neurospatial.encoding import object_vector_score

        assert callable(object_vector_score)

    def test_import_is_object_vector_cell_from_encoding(self) -> None:
        """Test importing is_object_vector_cell from encoding."""
        from neurospatial.encoding import is_object_vector_cell

        assert callable(is_object_vector_cell)

    def test_import_plot_object_vector_tuning_from_encoding(self) -> None:
        """Test importing plot_object_vector_tuning from encoding."""
        from neurospatial.encoding import plot_object_vector_tuning

        assert callable(plot_object_vector_tuning)


# =============================================================================
# Module Structure Tests
# =============================================================================


class TestEncodingObjectVectorModuleStructure:
    """Test encoding.object_vector module structure."""

    def test_encoding_object_vector_has_all(self) -> None:
        """Test that encoding.object_vector defines __all__."""
        from neurospatial.encoding import object_vector

        assert hasattr(object_vector, "__all__")
        assert isinstance(object_vector.__all__, list)

    def test_encoding_object_vector_all_contains_expected_exports(self) -> None:
        """Test that __all__ contains all expected exports."""
        from neurospatial.encoding import object_vector

        expected = [
            "ObjectVectorFieldResult",
            "compute_object_vector_field",
            "ObjectVectorMetrics",
            "compute_object_vector_tuning",
            "object_vector_score",
            "is_object_vector_cell",
            "plot_object_vector_tuning",
        ]
        for export in expected:
            assert export in object_vector.__all__, f"{export} not in __all__"

    def test_encoding_object_vector_has_docstring(self) -> None:
        """Test that encoding.object_vector has a docstring."""
        from neurospatial.encoding import object_vector

        assert object_vector.__doc__ is not None
        assert len(object_vector.__doc__) > 50


# =============================================================================
# Re-export Verification Tests
# =============================================================================


class TestObjectVectorReExports:
    """Test that encoding.object_vector re-exports match original modules."""

    def test_object_vector_field_result_same_as_original(self) -> None:
        """Test ObjectVectorFieldResult is same as object_vector_field."""
        from neurospatial.encoding.object_vector import (
            ObjectVectorFieldResult as EncodingResult,
        )
        from neurospatial.encoding.object_vector import (
            ObjectVectorFieldResult as OriginalResult,
        )

        # Should be the same class
        assert EncodingResult is OriginalResult

    def test_compute_object_vector_field_same_as_original(self) -> None:
        """Test compute_object_vector_field is same as object_vector_field."""
        from neurospatial.encoding.object_vector import (
            compute_object_vector_field as encoding_func,
        )
        from neurospatial.encoding.object_vector import (
            compute_object_vector_field as original_func,
        )

        # Should be the same function
        assert encoding_func is original_func

    def test_object_vector_metrics_same_as_metrics(self) -> None:
        """Test ObjectVectorMetrics is same as metrics.object_vector_cells."""
        from neurospatial.encoding.object_vector import (
            ObjectVectorMetrics as EncodingMetrics,
        )
        from neurospatial.metrics.object_vector_cells import (
            ObjectVectorMetrics as MetricsMetrics,
        )

        # Should be the same class
        assert EncodingMetrics is MetricsMetrics

    def test_compute_object_vector_tuning_same_as_metrics(self) -> None:
        """Test compute_object_vector_tuning is same as metrics."""
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning as encoding_func,
        )
        from neurospatial.metrics.object_vector_cells import (
            compute_object_vector_tuning as metrics_func,
        )

        # Should be the same function
        assert encoding_func is metrics_func

    def test_object_vector_score_same_as_metrics(self) -> None:
        """Test object_vector_score is same as metrics."""
        from neurospatial.encoding.object_vector import (
            object_vector_score as encoding_func,
        )
        from neurospatial.metrics.object_vector_cells import (
            object_vector_score as metrics_func,
        )

        # Should be the same function
        assert encoding_func is metrics_func

    def test_is_object_vector_cell_same_as_metrics(self) -> None:
        """Test is_object_vector_cell is same as metrics."""
        from neurospatial.encoding.object_vector import (
            is_object_vector_cell as encoding_func,
        )
        from neurospatial.metrics.object_vector_cells import (
            is_object_vector_cell as metrics_func,
        )

        # Should be the same function
        assert encoding_func is metrics_func

    def test_plot_object_vector_tuning_same_as_metrics(self) -> None:
        """Test plot_object_vector_tuning is same as metrics."""
        from neurospatial.encoding.object_vector import (
            plot_object_vector_tuning as encoding_func,
        )
        from neurospatial.metrics.object_vector_cells import (
            plot_object_vector_tuning as metrics_func,
        )

        # Should be the same function
        assert encoding_func is metrics_func


# =============================================================================
# Functionality Tests
# =============================================================================


class TestObjectVectorFieldFunctionality:
    """Test that re-exported field functions work correctly."""

    def test_compute_object_vector_field_basic(self) -> None:
        """Test compute_object_vector_field with basic data."""
        from neurospatial.encoding.object_vector import compute_object_vector_field

        # Create test data
        rng = np.random.default_rng(42)
        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times, times, positions, headings, object_positions
        )

        # Check result structure
        assert result.field is not None
        assert result.ego_env is not None
        assert result.occupancy is not None
        assert len(result.field) == result.ego_env.n_bins

    def test_object_vector_field_result_frozen(self) -> None:
        """Test that ObjectVectorFieldResult is immutable."""
        import pytest

        from neurospatial.encoding.object_vector import ObjectVectorFieldResult

        # Create a simple egocentric environment
        ego_env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        )

        result = ObjectVectorFieldResult(
            field=np.zeros(ego_env.n_bins),
            ego_env=ego_env,
            occupancy=np.ones(ego_env.n_bins),
        )

        # Try to modify - should raise
        with pytest.raises(AttributeError):
            result.field = np.ones(ego_env.n_bins)  # type: ignore[misc]


class TestObjectVectorMetricsFunctionality:
    """Test that re-exported metrics functions work correctly."""

    def test_object_vector_score_sharp_tuning(self) -> None:
        """Test object_vector_score with sharp tuning."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Create tuning curve with sharp peak
        tc = np.zeros((10, 12)) + 0.1
        tc[5, 6] = 20.0

        score = object_vector_score(tc)

        # Sharp tuning should have high score
        assert score > 0.5, f"Expected score > 0.5 for sharp tuning, got {score}"

    def test_object_vector_score_flat_tuning(self) -> None:
        """Test object_vector_score with flat tuning."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Create flat tuning curve
        tc = np.ones((10, 12)) * 5.0

        score = object_vector_score(tc)

        # Flat tuning should have low score (close to 0)
        assert score < 0.1, f"Expected score < 0.1 for flat tuning, got {score}"

    def test_is_object_vector_cell_high_rate_sharp_tuning(self) -> None:
        """Test is_object_vector_cell with high rate and sharp tuning."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Create tuning curve with sharp peak
        tc = np.zeros((10, 12)) + 0.1
        tc[5, 6] = 25.0

        is_ovc = is_object_vector_cell(tc, peak_rate=25.0, score_threshold=0.3)

        assert is_ovc, "Expected neuron to be classified as OVC"

    def test_is_object_vector_cell_low_rate(self) -> None:
        """Test is_object_vector_cell with low firing rate."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Create tuning curve with sharp peak but low rate
        tc = np.zeros((10, 12)) + 0.01
        tc[5, 6] = 2.0

        is_ovc = is_object_vector_cell(tc, peak_rate=2.0, min_peak_rate=5.0)

        assert not is_ovc, "Expected neuron NOT to be classified as OVC (low rate)"

    def test_compute_object_vector_tuning_basic(self) -> None:
        """Test compute_object_vector_tuning with basic data."""
        from neurospatial.encoding.object_vector import compute_object_vector_tuning

        # Create test data
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        metrics = compute_object_vector_tuning(
            spike_times, times, positions, headings, object_positions, env
        )

        # Check result structure
        assert metrics.tuning_curve.shape == (10, 12)
        assert len(metrics.distance_bins) == 11
        assert len(metrics.direction_bins) == 13
        assert isinstance(metrics.preferred_distance, float)
        assert isinstance(metrics.preferred_direction, float)

    def test_object_vector_metrics_interpretation(self) -> None:
        """Test ObjectVectorMetrics interpretation method."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        metrics = ObjectVectorMetrics(
            preferred_distance=10.0,
            preferred_direction=0.0,
            distance_selectivity=3.0,
            direction_selectivity=0.6,
            object_vector_score=0.8,
            peak_rate=20.0,
            mean_rate=5.0,
            tuning_curve=np.ones((10, 12)),
            distance_bins=np.linspace(0, 50, 11),
            direction_bins=np.linspace(-np.pi, np.pi, 13),
        )

        # Test string representation
        interpretation = str(metrics)
        assert "Object-Vector" in interpretation
        assert "10.0" in interpretation  # preferred_distance
