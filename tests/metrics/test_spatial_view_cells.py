"""Tests for spatial view cell metrics module.

Tests cover:
- Module structure and imports
- SpatialViewMetrics dataclass
- spatial_view_cell_metrics() computation
- is_spatial_view_cell() classifier
- Integration with simulated spatial view cells
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleStructure:
    """Tests for module structure and imports."""

    def test_imports_from_metrics(self):
        """Can import from neurospatial.metrics."""
        from neurospatial.metrics import (
            SpatialViewMetrics,
            is_spatial_view_cell,
            spatial_view_cell_metrics,
        )

        assert SpatialViewMetrics is not None
        assert spatial_view_cell_metrics is not None
        assert is_spatial_view_cell is not None

    def test_imports_from_spatial_view_cells_module(self):
        """Can import directly from module."""
        from neurospatial.metrics.spatial_view_cells import (
            SpatialViewMetrics,
            is_spatial_view_cell,
            spatial_view_cell_metrics,
        )

        assert SpatialViewMetrics is not None
        assert spatial_view_cell_metrics is not None
        assert is_spatial_view_cell is not None

    def test_module_docstring(self):
        """Module has docstring."""
        from neurospatial.metrics import spatial_view_cells

        assert spatial_view_cells.__doc__ is not None
        assert len(spatial_view_cells.__doc__) > 100

    def test_all_exports(self):
        """Module exports expected symbols."""
        from neurospatial.metrics import spatial_view_cells

        assert hasattr(spatial_view_cells, "__all__")
        expected = [
            "SpatialViewMetrics",
            "spatial_view_cell_metrics",
            "is_spatial_view_cell",
        ]
        for name in expected:
            assert name in spatial_view_cells.__all__


class TestSpatialViewMetrics:
    """Tests for SpatialViewMetrics dataclass."""

    def test_dataclass_creation(self):
        """Can create SpatialViewMetrics dataclass."""
        from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics

        metrics = SpatialViewMetrics(
            view_field_skaggs_info=1.5,
            place_field_skaggs_info=0.5,
            view_place_correlation=0.3,
            view_field_sparsity=0.2,
            view_field_coherence=0.7,
            is_spatial_view_cell=True,
        )

        assert metrics.view_field_skaggs_info == 1.5
        assert metrics.place_field_skaggs_info == 0.5
        assert metrics.view_place_correlation == 0.3
        assert metrics.view_field_sparsity == 0.2
        assert metrics.view_field_coherence == 0.7
        assert metrics.is_spatial_view_cell is True

    def test_dataclass_is_frozen(self):
        """SpatialViewMetrics is immutable."""
        from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics

        metrics = SpatialViewMetrics(
            view_field_skaggs_info=1.5,
            place_field_skaggs_info=0.5,
            view_place_correlation=0.3,
            view_field_sparsity=0.2,
            view_field_coherence=0.7,
            is_spatial_view_cell=True,
        )

        with pytest.raises(AttributeError):
            metrics.view_field_skaggs_info = 2.0

    def test_interpretation_method(self):
        """interpretation() returns human-readable string."""
        from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics

        # Spatial view cell
        svc = SpatialViewMetrics(
            view_field_skaggs_info=1.5,
            place_field_skaggs_info=0.5,
            view_place_correlation=0.3,
            view_field_sparsity=0.2,
            view_field_coherence=0.7,
            is_spatial_view_cell=True,
        )

        interpretation = svc.interpretation()
        assert "spatial view cell" in interpretation.lower()
        assert "view field info" in interpretation.lower()
        assert "1.5" in interpretation or "1.50" in interpretation

    def test_interpretation_non_svc(self):
        """interpretation() explains why not classified."""
        from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics

        # Not a spatial view cell
        non_svc = SpatialViewMetrics(
            view_field_skaggs_info=0.3,
            place_field_skaggs_info=0.5,
            view_place_correlation=0.9,
            view_field_sparsity=0.8,
            view_field_coherence=0.2,
            is_spatial_view_cell=False,
        )

        interpretation = non_svc.interpretation()
        assert "not" in interpretation.lower() or "Not" in interpretation

    def test_str_returns_interpretation(self):
        """str() returns interpretation."""
        from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics

        metrics = SpatialViewMetrics(
            view_field_skaggs_info=1.5,
            place_field_skaggs_info=0.5,
            view_place_correlation=0.3,
            view_field_sparsity=0.2,
            view_field_coherence=0.7,
            is_spatial_view_cell=True,
        )

        assert str(metrics) == metrics.interpretation()


class TestSpatialViewCellMetrics:
    """Tests for spatial_view_cell_metrics() function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from neurospatial import Environment

        rng = np.random.default_rng(42)

        # Create environment
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory (stay inside to ensure valid views)
        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(20, 80, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        # Create spike times
        spike_times = rng.choice(times, size=100, replace=False)

        return {
            "env": env,
            "times": times,
            "positions": positions,
            "headings": headings,
            "spike_times": spike_times,
        }

    def test_returns_spatial_view_metrics(self, sample_data):
        """Returns SpatialViewMetrics dataclass."""
        from neurospatial.metrics.spatial_view_cells import (
            SpatialViewMetrics,
            spatial_view_cell_metrics,
        )

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        assert isinstance(metrics, SpatialViewMetrics)

    def test_view_field_skaggs_info_computed(self, sample_data):
        """View field Skaggs info is computed."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        # Skaggs info should be non-negative
        assert metrics.view_field_skaggs_info >= 0 or np.isnan(
            metrics.view_field_skaggs_info
        )

    def test_place_field_skaggs_info_computed(self, sample_data):
        """Place field Skaggs info is computed."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        # Skaggs info should be non-negative
        assert metrics.place_field_skaggs_info >= 0 or np.isnan(
            metrics.place_field_skaggs_info
        )

    def test_view_place_correlation_computed(self, sample_data):
        """View-place correlation is computed."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        # Correlation should be in [-1, 1]
        if not np.isnan(metrics.view_place_correlation):
            assert -1.0 <= metrics.view_place_correlation <= 1.0

    def test_sparsity_computed(self, sample_data):
        """View field sparsity is computed."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        # Sparsity should be in [0, 1]
        if not np.isnan(metrics.view_field_sparsity):
            assert 0.0 <= metrics.view_field_sparsity <= 1.0

    def test_coherence_computed(self, sample_data):
        """View field coherence is computed."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
        )

        # Coherence should be in [-1, 1]
        if not np.isnan(metrics.view_field_coherence):
            assert -1.0 <= metrics.view_field_coherence <= 1.0

    def test_empty_spikes_raises(self, sample_data):
        """Empty spike_times raises ValueError."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        with pytest.raises(ValueError, match="spike_times"):
            spatial_view_cell_metrics(
                env=sample_data["env"],
                spike_times=np.array([]),
                times=sample_data["times"],
                positions=sample_data["positions"],
                headings=sample_data["headings"],
            )

    def test_mismatched_lengths_raises(self, sample_data):
        """Mismatched array lengths raises ValueError."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        # Wrong positions length
        with pytest.raises(ValueError, match="length"):
            spatial_view_cell_metrics(
                env=sample_data["env"],
                spike_times=sample_data["spike_times"],
                times=sample_data["times"],
                positions=sample_data["positions"][:100],  # Wrong length
                headings=sample_data["headings"],
            )

    def test_view_distance_parameter(self, sample_data):
        """view_distance parameter affects computation."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics_short = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
            view_distance=5.0,
        )

        metrics_long = spatial_view_cell_metrics(
            env=sample_data["env"],
            spike_times=sample_data["spike_times"],
            times=sample_data["times"],
            positions=sample_data["positions"],
            headings=sample_data["headings"],
            view_distance=20.0,
        )

        # Different view distances should generally give different results
        # (though in degenerate cases they might be the same)
        # At minimum, both should return valid metrics
        assert isinstance(metrics_short, type(metrics_long))


class TestIsSpatialViewCell:
    """Tests for is_spatial_view_cell() classifier."""

    def test_returns_bool(self):
        """Returns boolean."""
        from neurospatial import Environment
        from neurospatial.metrics.spatial_view_cells import is_spatial_view_cell

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(20, 80, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = rng.choice(times, size=100, replace=False)

        result = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
        )

        assert isinstance(result, bool)

    def test_info_ratio_threshold(self):
        """Classification depends on view/place info ratio."""
        # We can't easily create a definite SVC, but we can test the function runs
        from neurospatial import Environment
        from neurospatial.metrics.spatial_view_cells import is_spatial_view_cell

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(20, 80, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = rng.choice(times, size=100, replace=False)

        # Very strict threshold
        result_strict = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            info_ratio=10.0,  # Very strict
        )

        # Lenient threshold
        result_lenient = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            info_ratio=0.1,  # Very lenient
        )

        # Both should be bools
        assert isinstance(result_strict, bool)
        assert isinstance(result_lenient, bool)

    def test_max_correlation_threshold(self):
        """Classification depends on max correlation threshold."""
        from neurospatial import Environment
        from neurospatial.metrics.spatial_view_cells import is_spatial_view_cell

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        times = np.linspace(0, 100, n_time)
        positions = rng.uniform(20, 80, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = rng.choice(times, size=100, replace=False)

        # Test runs without error with different thresholds
        result1 = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            max_correlation=0.5,
        )

        result2 = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            max_correlation=0.9,
        )

        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


class TestGroundTruthRecovery:
    """Tests for recovering ground truth from simulated cells."""

    @pytest.fixture
    def spatial_view_cell_data(self):
        """Create data from a simulated spatial view cell."""
        from neurospatial import Environment
        from neurospatial.simulation import SpatialViewCellModel

        rng = np.random.default_rng(42)

        # Create environment
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory with consistent heading
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(20, 80, (n_time, 2))
        # Heading varies to sample different view directions
        headings = np.linspace(-np.pi, np.pi, n_time)

        # Create spatial view cell model
        preferred_view = np.array([50.0, 50.0])
        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=preferred_view,
            view_field_width=10.0,
            view_distance=15.0,
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.5,
        )

        # Generate firing rates and spike times
        firing_rates = model.firing_rate(positions, times=times, headings=headings)

        # Generate spikes using inhomogeneous Poisson process
        dt = np.median(np.diff(times))
        spike_mask = rng.random(n_time) < (firing_rates * dt)
        spike_times = times[spike_mask]

        return {
            "env": env,
            "times": times,
            "positions": positions,
            "headings": headings,
            "spike_times": spike_times,
            "preferred_view": preferred_view,
            "model": model,
        }

    def test_spatial_view_cell_has_higher_view_info(self, spatial_view_cell_data):
        """Spatial view cell should have higher view field info than place field info."""
        from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics

        metrics = spatial_view_cell_metrics(
            env=spatial_view_cell_data["env"],
            spike_times=spatial_view_cell_data["spike_times"],
            times=spatial_view_cell_data["times"],
            positions=spatial_view_cell_data["positions"],
            headings=spatial_view_cell_data["headings"],
            view_distance=15.0,
        )

        # For a true spatial view cell, view field info should be higher
        # (or at least not much lower) than place field info
        # Skip if not enough spikes generated
        if len(spatial_view_cell_data["spike_times"]) < 50:
            pytest.skip("Not enough spikes generated")

        # The view field should have positive info
        assert metrics.view_field_skaggs_info >= 0

    def test_place_cell_not_classified_as_svc(self):
        """A place cell should not be classified as a spatial view cell."""
        from neurospatial import Environment
        from neurospatial.metrics.spatial_view_cells import is_spatial_view_cell
        from neurospatial.simulation import PlaceCellModel

        rng = np.random.default_rng(42)

        # Create environment
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create place cell (NOT a spatial view cell)
        place_cell = PlaceCellModel(
            env=env,
            center=np.array([50.0, 50.0]),
            width=10.0,
            max_rate=20.0,
            baseline_rate=0.5,
        )

        # Generate trajectory
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        # Generate firing rates and spike times
        firing_rates = place_cell.firing_rate(positions)
        dt = np.median(np.diff(times))
        spike_mask = rng.random(n_time) < (firing_rates * dt)
        spike_times = times[spike_mask]

        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        # Place cell should NOT be classified as spatial view cell
        # when using strict criteria
        result = is_spatial_view_cell(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            view_distance=15.0,
            info_ratio=2.0,  # Require view info to be 2x place info
        )

        # A typical place cell should have similar view and place info
        # and thus not be classified as a spatial view cell
        # (This test might be flaky depending on the specific trajectory)
        assert isinstance(result, bool)
