"""Tests for object-vector cell metrics module.

These tests verify:
1. Module can be imported
2. __all__ exports are correct
3. ObjectVectorMetrics dataclass works correctly
4. compute_object_vector_tuning() bins spikes correctly
5. object_vector_score() computes correct selectivity
6. is_object_vector_cell() classifies cells correctly
7. plot_object_vector_tuning() creates correct visualization
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleSetup:
    """Tests for object_vector_cells metrics module setup."""

    def test_module_imports(self) -> None:
        """Test that object_vector_cells metrics module can be imported."""
        from neurospatial.encoding import object_vector as object_vector_cells

        assert object_vector_cells is not None

    def test_module_has_docstring(self) -> None:
        """Test that module has a docstring."""
        from neurospatial.encoding import object_vector as object_vector_cells

        assert object_vector_cells.__doc__ is not None
        assert len(object_vector_cells.__doc__) > 100  # Should be substantial

    def test_module_docstring_mentions_object_vector(self) -> None:
        """Test that module docstring mentions object-vector cells."""
        from neurospatial.encoding import object_vector as object_vector_cells

        docstring = object_vector_cells.__doc__
        assert docstring is not None
        assert "object" in docstring.lower() or "vector" in docstring.lower()

    def test_module_has_all_attribute(self) -> None:
        """Test that module has __all__ defined."""
        from neurospatial.encoding import object_vector as object_vector_cells

        assert hasattr(object_vector_cells, "__all__")
        assert isinstance(object_vector_cells.__all__, list)

    def test_module_all_contains_expected_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from neurospatial.encoding import object_vector as object_vector_cells

        expected = [
            "ObjectVectorMetrics",
            "compute_object_vector_tuning",
            "object_vector_score",
            "is_object_vector_cell",
            "plot_object_vector_tuning",
        ]
        for export in expected:
            assert export in object_vector_cells.__all__

    def test_module_docstring_has_references(self) -> None:
        """Test that module docstring includes scientific references."""
        from neurospatial.encoding import object_vector as object_vector_cells

        docstring = object_vector_cells.__doc__
        assert docstring is not None
        assert "References" in docstring or "Hoydal" in docstring


class TestObjectVectorMetricsDataclass:
    """Tests for ObjectVectorMetrics frozen dataclass."""

    def test_dataclass_can_be_imported(self) -> None:
        """Test that ObjectVectorMetrics can be imported."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        assert ObjectVectorMetrics is not None

    def test_dataclass_is_frozen(self) -> None:
        """Test that ObjectVectorMetrics is a frozen dataclass."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        # Create instance
        metrics = ObjectVectorMetrics(
            preferred_distance=10.0,
            preferred_direction=0.0,
            distance_selectivity=2.5,
            direction_selectivity=0.6,
            object_vector_score=0.8,
            peak_rate=15.0,
            mean_rate=3.0,
            tuning_curve=np.zeros((10, 12)),
            distance_bins=np.arange(11),
            direction_bins=np.linspace(-np.pi, np.pi, 13),
        )

        # Try to modify - should raise
        with pytest.raises(AttributeError):
            metrics.preferred_distance = 20.0  # type: ignore[misc]

    def test_dataclass_has_all_fields(self) -> None:
        """Test that ObjectVectorMetrics has all required fields."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        metrics = ObjectVectorMetrics(
            preferred_distance=10.0,
            preferred_direction=np.pi / 4,
            distance_selectivity=3.0,
            direction_selectivity=0.7,
            object_vector_score=0.85,
            peak_rate=20.0,
            mean_rate=5.0,
            tuning_curve=np.ones((10, 12)),
            distance_bins=np.linspace(0, 50, 11),
            direction_bins=np.linspace(-np.pi, np.pi, 13),
        )

        assert metrics.preferred_distance == pytest.approx(10.0)
        assert metrics.preferred_direction == pytest.approx(np.pi / 4)
        assert metrics.distance_selectivity == pytest.approx(3.0)
        assert metrics.direction_selectivity == pytest.approx(0.7)
        assert metrics.object_vector_score == pytest.approx(0.85)
        assert metrics.peak_rate == pytest.approx(20.0)
        assert metrics.mean_rate == pytest.approx(5.0)
        assert metrics.tuning_curve.shape == (10, 12)
        assert len(metrics.distance_bins) == 11
        assert len(metrics.direction_bins) == 13

    def test_interpretation_method_exists(self) -> None:
        """Test that ObjectVectorMetrics has interpretation() method."""
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

        interp = metrics.interpretation()
        assert isinstance(interp, str)
        assert len(interp) > 50

    def test_interpretation_shows_preferred_distance(self) -> None:
        """Test that interpretation shows preferred distance."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        metrics = ObjectVectorMetrics(
            preferred_distance=15.5,
            preferred_direction=np.pi / 2,
            distance_selectivity=4.0,
            direction_selectivity=0.7,
            object_vector_score=0.9,
            peak_rate=25.0,
            mean_rate=5.0,
            tuning_curve=np.ones((10, 12)),
            distance_bins=np.linspace(0, 50, 11),
            direction_bins=np.linspace(-np.pi, np.pi, 13),
        )

        interp = metrics.interpretation()
        assert "15.5" in interp or "distance" in interp.lower()

    def test_interpretation_shows_preferred_direction_degrees(self) -> None:
        """Test that interpretation shows preferred direction in degrees."""
        from neurospatial.encoding.object_vector import ObjectVectorMetrics

        metrics = ObjectVectorMetrics(
            preferred_distance=10.0,
            preferred_direction=np.pi / 2,  # 90 degrees
            distance_selectivity=3.0,
            direction_selectivity=0.7,
            object_vector_score=0.85,
            peak_rate=20.0,
            mean_rate=5.0,
            tuning_curve=np.ones((10, 12)),
            distance_bins=np.linspace(0, 50, 11),
            direction_bins=np.linspace(-np.pi, np.pi, 13),
        )

        interp = metrics.interpretation()
        # Should show ~90 degrees
        assert "90" in interp or "direction" in interp.lower()

    def test_str_method_returns_interpretation(self) -> None:
        """Test that __str__() returns interpretation()."""
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

        assert str(metrics) == metrics.interpretation()

    def test_exported_from_metrics_init(self) -> None:
        """Test that ObjectVectorMetrics is exported from neurospatial.metrics."""
        from neurospatial.metrics import ObjectVectorMetrics

        assert ObjectVectorMetrics is not None


class TestComputeObjectVectorTuning:
    """Tests for compute_object_vector_tuning() function."""

    def test_function_exists(self) -> None:
        """Test that compute_object_vector_tuning can be imported."""
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        assert callable(compute_object_vector_tuning)

    def test_returns_object_vector_metrics(self) -> None:
        """Test that function returns ObjectVectorMetrics instance."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            ObjectVectorMetrics,
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory data
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        # Object position
        object_positions = np.array([[50.0, 50.0]])

        # Random spike times
        spike_times = rng.choice(times, size=100, replace=False)

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
        )

        assert isinstance(metrics, ObjectVectorMetrics)

    def test_tuning_curve_shape(self) -> None:
        """Test that tuning curve has expected shape."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory data
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
            n_distance_bins=15,
            n_direction_bins=18,
        )

        assert metrics.tuning_curve.shape == (15, 18)
        assert len(metrics.distance_bins) == 16  # n_bins + 1 for edges
        assert len(metrics.direction_bins) == 19  # n_bins + 1 for edges

    def test_bins_spikes_by_nearest_object(self) -> None:
        """Test that spikes are binned by distance/direction to nearest object."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory where animal is always 10 units from object, ahead
        n_time = 1000
        times = np.linspace(0, 100, n_time)
        object_positions = np.array([[50.0, 50.0]])

        # Position animal 10 units away, with object ahead (heading toward object)
        angles = np.linspace(0, 2 * np.pi, n_time)
        positions = np.column_stack(
            [50.0 + 10.0 * np.cos(angles), 50.0 + 10.0 * np.sin(angles)]
        )
        # Headings pointing toward object center
        headings = angles - np.pi  # Opposite of position angle

        # Spikes uniformly distributed
        spike_times = times[::10]

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
            max_distance=50.0,
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Peak should be around distance=10 and direction=0 (ahead)
        assert metrics.preferred_distance == pytest.approx(10.0, rel=0.3)

    def test_occupancy_normalization(self) -> None:
        """Test that tuning curve is normalized by occupancy."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create data with known occupancy pattern
        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = times[::10]  # Regular spikes

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
        )

        # Firing rate should be finite and reasonable
        assert np.all(
            np.isfinite(metrics.tuning_curve) | np.isnan(metrics.tuning_curve)
        )

    def test_min_occupancy_threshold(self) -> None:
        """Test that min_occupancy_seconds filters low-occupancy bins."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Very sparse data
        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = times[::20]

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
            min_occupancy_seconds=0.5,  # High threshold
        )

        # Many bins should be NaN due to low occupancy
        assert np.any(np.isnan(metrics.tuning_curve))

    def test_multiple_objects_nearest(self) -> None:
        """Test that nearest object is used by default."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Multiple objects
        object_positions = np.array([[25.0, 50.0], [75.0, 50.0]])

        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = rng.choice(times, size=100, replace=False)

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
        )

        # Should produce valid metrics
        assert isinstance(metrics.preferred_distance, float)
        assert isinstance(metrics.preferred_direction, float)

    def test_validation_empty_spike_times(self) -> None:
        """Test validation when no spikes provided."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])

        spike_times = np.array([])  # No spikes

        with pytest.raises(ValueError, match="spike"):
            compute_object_vector_tuning(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
                env=env,
            )

    def test_validation_times_positions_mismatch(self) -> None:
        """Test validation when times and positions have different lengths."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (50, 2))  # Wrong length
        headings = rng.uniform(-np.pi, np.pi, 100)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length"):
            compute_object_vector_tuning(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
                env=env,
            )

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import compute_object_vector_tuning

        assert callable(compute_object_vector_tuning)


class TestObjectVectorScore:
    """Tests for object_vector_score() function."""

    def test_function_exists(self) -> None:
        """Test that object_vector_score can be imported."""
        from neurospatial.encoding.object_vector import object_vector_score

        assert callable(object_vector_score)

    def test_returns_float(self) -> None:
        """Test that function returns a float score."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Sharp tuning curve - high distance and direction selectivity
        n_dist, n_dir = 10, 12
        tuning_curve = np.zeros((n_dist, n_dir))
        tuning_curve[3, 6] = 20.0  # Peak at one location

        score = object_vector_score(tuning_curve)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_selectivity_high_score(self) -> None:
        """Test that sharp tuning produces high score."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Very sharp tuning (single peak)
        n_dist, n_dir = 10, 12
        tuning_curve = np.zeros((n_dist, n_dir)) + 0.1  # Low baseline
        tuning_curve[5, 6] = 20.0  # Sharp peak

        score = object_vector_score(tuning_curve)
        assert score > 0.5  # Should be high

    def test_uniform_tuning_low_score(self) -> None:
        """Test that uniform tuning produces low score."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Uniform tuning (no selectivity)
        n_dist, n_dir = 10, 12
        tuning_curve = np.ones((n_dist, n_dir)) * 5.0

        score = object_vector_score(tuning_curve)
        assert score < 0.3  # Should be low

    def test_distance_selectivity_computation(self) -> None:
        """Test that distance selectivity is computed as peak/mean ratio."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Tuning with 10x peak/mean ratio
        n_dist, n_dir = 10, 12
        tuning_curve = np.ones((n_dist, n_dir))
        tuning_curve[5, :] = 10.0  # 10x at one distance

        score = object_vector_score(tuning_curve)
        assert score > 0.0  # Has some selectivity

    def test_direction_selectivity_computation(self) -> None:
        """Test that direction selectivity uses mean resultant length."""
        from neurospatial.encoding.object_vector import object_vector_score

        # Sharp direction tuning (single direction)
        n_dist, n_dir = 10, 12
        tuning_curve = np.zeros((n_dist, n_dir)) + 0.1
        tuning_curve[:, 6] = 10.0  # Strong at one direction

        score = object_vector_score(tuning_curve)
        assert score > 0.3  # Should have direction selectivity

    def test_max_distance_selectivity_parameter(self) -> None:
        """Test that max_distance_selectivity parameter works."""
        from neurospatial.encoding.object_vector import object_vector_score

        n_dist, n_dir = 10, 12
        tuning_curve = np.ones((n_dist, n_dir))
        tuning_curve[5, 6] = 20.0

        # Default max_distance_selectivity
        score_default = object_vector_score(tuning_curve)

        # Higher max (more conservative)
        score_high = object_vector_score(tuning_curve, max_distance_selectivity=20.0)

        # Higher max should give lower normalized score
        assert score_high <= score_default

    def test_validation_max_distance_selectivity(self) -> None:
        """Test that max_distance_selectivity > 1 is required."""
        from neurospatial.encoding.object_vector import object_vector_score

        tuning_curve = np.ones((10, 12))

        with pytest.raises(ValueError, match="max_distance_selectivity"):
            object_vector_score(tuning_curve, max_distance_selectivity=0.5)

    def test_handles_nan_values(self) -> None:
        """Test that NaN values in tuning curve are handled."""
        from neurospatial.encoding.object_vector import object_vector_score

        n_dist, n_dir = 10, 12
        tuning_curve = np.ones((n_dist, n_dir)) * 5.0
        tuning_curve[0, 0] = np.nan  # Some NaN

        # Should not raise
        score = object_vector_score(tuning_curve)
        assert np.isfinite(score) or np.isnan(score)

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import object_vector_score

        assert callable(object_vector_score)


class TestIsObjectVectorCell:
    """Tests for is_object_vector_cell() classifier function."""

    def test_function_exists(self) -> None:
        """Test that is_object_vector_cell can be imported."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        assert callable(is_object_vector_cell)

    def test_returns_bool(self) -> None:
        """Test that function returns a boolean."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Create simple tuning curve
        tuning_curve = np.ones((10, 12)) * 5.0
        tuning_curve[5, 6] = 20.0

        result = is_object_vector_cell(tuning_curve, peak_rate=20.0)
        assert isinstance(result, bool)

    def test_classifies_high_score_high_rate_as_ovc(self) -> None:
        """Test that high score + high rate is classified as OVC."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Sharp tuning with high peak rate
        n_dist, n_dir = 10, 12
        tuning_curve = np.zeros((n_dist, n_dir)) + 0.1
        tuning_curve[5, 6] = 25.0

        result = is_object_vector_cell(
            tuning_curve, peak_rate=25.0, score_threshold=0.3, min_peak_rate=5.0
        )
        # Should classify as OVC given sharp tuning
        assert result is True

    def test_rejects_low_score(self) -> None:
        """Test that low score is not classified as OVC."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Uniform tuning (low selectivity)
        tuning_curve = np.ones((10, 12)) * 10.0

        result = is_object_vector_cell(
            tuning_curve, peak_rate=10.0, score_threshold=0.5, min_peak_rate=5.0
        )
        assert result is False

    def test_rejects_low_peak_rate(self) -> None:
        """Test that low peak rate is not classified as OVC."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        # Sharp tuning but low rate
        n_dist, n_dir = 10, 12
        tuning_curve = np.zeros((n_dist, n_dir)) + 0.1
        tuning_curve[5, 6] = 1.0  # Low peak

        result = is_object_vector_cell(
            tuning_curve, peak_rate=1.0, score_threshold=0.3, min_peak_rate=5.0
        )
        assert result is False  # Peak rate too low

    def test_threshold_parameters(self) -> None:
        """Test that score_threshold and min_peak_rate parameters work."""
        from neurospatial.encoding.object_vector import is_object_vector_cell

        n_dist, n_dir = 10, 12
        tuning_curve = np.ones((n_dist, n_dir)) * 2.0
        tuning_curve[5, 6] = 10.0

        # With very strict thresholds
        result_strict = is_object_vector_cell(
            tuning_curve, peak_rate=10.0, score_threshold=0.9, min_peak_rate=15.0
        )

        # Strict should be harder to pass
        assert not result_strict  # Should fail strict thresholds

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import is_object_vector_cell

        assert callable(is_object_vector_cell)


class TestPlotObjectVectorTuning:
    """Tests for plot_object_vector_tuning() visualization function."""

    def test_function_exists(self) -> None:
        """Test that plot_object_vector_tuning can be imported."""
        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        assert callable(plot_object_vector_tuning)

    def test_returns_axes_object(self) -> None:
        """Test that function returns matplotlib Axes."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        ax = plot_object_vector_tuning(
            tuning_curve, distance_bins=distance_bins, direction_bins=direction_bins
        )
        assert isinstance(ax, Axes)

        plt.close("all")

    def test_polar_projection_default(self) -> None:
        """Test that polar heatmap is created by default."""
        import matplotlib.pyplot as plt
        from matplotlib.projections.polar import PolarAxes

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        ax = plot_object_vector_tuning(
            tuning_curve, distance_bins=distance_bins, direction_bins=direction_bins
        )
        assert isinstance(ax, PolarAxes)

        plt.close("all")

    def test_uses_provided_axes(self) -> None:
        """Test that function uses provided axes."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        _fig, ax_input = plt.subplots(subplot_kw={"projection": "polar"})
        ax_output = plot_object_vector_tuning(
            tuning_curve,
            distance_bins=distance_bins,
            direction_bins=direction_bins,
            ax=ax_input,
        )

        assert ax_output is ax_input

        plt.close("all")

    def test_marks_peak_location(self) -> None:
        """Test that peak location is marked when show_peak=True."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.zeros((10, 12))
        tuning_curve[5, 6] = 1.0  # Clear peak
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        ax = plot_object_vector_tuning(
            tuning_curve,
            distance_bins=distance_bins,
            direction_bins=direction_bins,
            show_peak=True,
        )

        # Should have some marker for peak
        # Check for scatter points or line markers
        assert len(ax.collections) > 0 or len(ax.lines) > 0

        plt.close("all")

    def test_colorbar_options(self) -> None:
        """Test that colorbar can be added."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        fig, ax_input = plt.subplots(subplot_kw={"projection": "polar"})
        plot_object_vector_tuning(
            tuning_curve,
            distance_bins=distance_bins,
            direction_bins=direction_bins,
            ax=ax_input,
            add_colorbar=True,
        )

        # Figure should have more than one axes if colorbar was added
        # Or check that a colorbar was created somehow
        # This is a basic check that it doesn't crash
        assert fig is not None

        plt.close("all")

    def test_custom_cmap(self) -> None:
        """Test that custom colormap is applied."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 11)
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        ax = plot_object_vector_tuning(
            tuning_curve,
            distance_bins=distance_bins,
            direction_bins=direction_bins,
            cmap="hot",
        )

        # Should not raise with custom cmap
        assert ax is not None

        plt.close("all")

    def test_validation_tuning_curve_shape(self) -> None:
        """Test that tuning curve shape is validated against bins."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.object_vector import plot_object_vector_tuning

        tuning_curve = np.random.default_rng(42).random((10, 12))
        distance_bins = np.linspace(0, 50, 5)  # Wrong size
        direction_bins = np.linspace(-np.pi, np.pi, 13)

        with pytest.raises(ValueError, match="shape"):
            plot_object_vector_tuning(
                tuning_curve,
                distance_bins=distance_bins,
                direction_bins=direction_bins,
            )

        plt.close("all")

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import plot_object_vector_tuning

        assert callable(plot_object_vector_tuning)


class TestRecoverGroundTruthFromSimulation:
    """Tests for recovering ground truth parameters from simulated cells."""

    def test_recover_preferred_distance(self) -> None:
        """Test that preferred distance can be recovered from simulated OVC."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)

        # Create OVC with known parameters
        object_positions = np.array([[50.0, 50.0]])
        true_distance = 15.0

        ovc = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=true_distance,
            distance_width=5.0,
            max_rate=20.0,
        )

        # Generate trajectory covering environment
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(5, 95, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        # Compute firing rates
        rates = ovc.firing_rate(positions)

        # Generate spikes (Poisson)
        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        # Skip if not enough spikes
        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        # Compute metrics
        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
            max_distance=50.0,
        )

        # Should recover approximate preferred distance
        assert metrics.preferred_distance == pytest.approx(true_distance, rel=0.3)

    def test_recover_preferred_direction(self) -> None:
        """Test that preferred direction can be recovered from simulated OVC."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
        )
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)

        # Create OVC with direction tuning
        object_positions = np.array([[50.0, 50.0]])
        true_direction = np.pi / 4  # 45 degrees left

        ovc = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=15.0,
            distance_width=5.0,
            preferred_direction=true_direction,
            direction_kappa=4.0,
            max_rate=20.0,
        )

        # Generate trajectory
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(5, 95, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        # Compute firing rates
        rates = ovc.firing_rate(positions, headings=headings)

        # Generate spikes
        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        # Compute metrics
        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
        )

        # Should recover approximate preferred direction
        # Allow for some error due to binning and noise
        angle_diff = np.abs(metrics.preferred_direction - true_direction)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Circular
        assert angle_diff < np.pi / 3  # Within 60 degrees

    def test_simulated_ovc_classified_as_ovc(self) -> None:
        """Test that simulated OVC is classified as OVC."""
        from neurospatial import Environment
        from neurospatial.encoding.object_vector import (
            compute_object_vector_tuning,
            is_object_vector_cell,
        )
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)

        # Create strong OVC
        object_positions = np.array([[50.0, 50.0]])
        ovc = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=15.0,
            distance_width=3.0,  # Sharp tuning
            preferred_direction=0.0,
            direction_kappa=6.0,  # Sharp direction
            max_rate=30.0,
        )

        # Generate rich trajectory
        n_time = 10000
        times = np.linspace(0, 1000, n_time)
        positions = rng.uniform(5, 95, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        rates = ovc.firing_rate(positions, headings=headings)

        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        if len(spike_times) < 100:
            pytest.skip("Not enough spikes generated")

        metrics = compute_object_vector_tuning(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            env=env,
        )

        # Should be classified as OVC
        result = is_object_vector_cell(
            metrics.tuning_curve,
            peak_rate=metrics.peak_rate,
            score_threshold=0.3,
            min_peak_rate=5.0,
        )
        assert result is True
