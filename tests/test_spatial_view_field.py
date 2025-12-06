"""Tests for spatial_view_field module.

These tests verify:
1. Module can be imported
2. __all__ exports are correct
3. SpatialViewFieldResult dataclass works correctly
4. compute_spatial_view_field() computes fields correctly
5. Different smoothing methods work correctly
6. NaN handling for viewing outside environment
7. Ground truth recovery from simulated spatial view cells
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleSetup:
    """Tests for spatial_view_field module setup."""

    def test_module_imports(self) -> None:
        """Test that spatial_view_field module can be imported."""
        from neurospatial import spatial_view_field

        assert spatial_view_field is not None

    def test_module_has_docstring(self) -> None:
        """Test that module has a docstring."""
        from neurospatial import spatial_view_field

        assert spatial_view_field.__doc__ is not None
        assert len(spatial_view_field.__doc__) > 100  # Should be substantial

    def test_module_docstring_mentions_spatial_view(self) -> None:
        """Test that module docstring mentions spatial view field."""
        from neurospatial import spatial_view_field

        docstring = spatial_view_field.__doc__
        assert docstring is not None
        assert "spatial" in docstring.lower() or "view" in docstring.lower()

    def test_module_has_all_attribute(self) -> None:
        """Test that module has __all__ defined."""
        from neurospatial import spatial_view_field

        assert hasattr(spatial_view_field, "__all__")
        assert isinstance(spatial_view_field.__all__, list)

    def test_module_all_contains_expected_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from neurospatial import spatial_view_field

        expected = [
            "SpatialViewFieldResult",
            "compute_spatial_view_field",
        ]
        for export in expected:
            assert export in spatial_view_field.__all__


class TestSpatialViewFieldResultDataclass:
    """Tests for SpatialViewFieldResult frozen dataclass."""

    def test_dataclass_can_be_imported(self) -> None:
        """Test that SpatialViewFieldResult can be imported."""
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult

        assert SpatialViewFieldResult is not None

    def test_dataclass_is_frozen(self) -> None:
        """Test that SpatialViewFieldResult is a frozen dataclass."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult

        # Create environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create instance
        result = SpatialViewFieldResult(
            field=np.zeros(env.n_bins),
            env=env,
            view_occupancy=np.ones(env.n_bins),
        )

        # Try to modify - should raise
        with pytest.raises(AttributeError):
            result.field = np.ones(env.n_bins)  # type: ignore[misc]

    def test_dataclass_has_all_fields(self) -> None:
        """Test that SpatialViewFieldResult has all required fields."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        field = np.random.default_rng(42).random(env.n_bins)
        view_occupancy = np.random.default_rng(42).random(env.n_bins)

        result = SpatialViewFieldResult(
            field=field,
            env=env,
            view_occupancy=view_occupancy,
        )

        assert result.field is not None
        assert result.env is not None
        assert result.view_occupancy is not None
        assert len(result.field) == env.n_bins
        assert len(result.view_occupancy) == env.n_bins


class TestComputeSpatialViewField:
    """Tests for compute_spatial_view_field() function."""

    def test_function_exists(self) -> None:
        """Test that compute_spatial_view_field can be imported."""
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        assert callable(compute_spatial_view_field)

    def test_returns_spatial_view_field_result(self) -> None:
        """Test that function returns SpatialViewFieldResult instance."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import (
            SpatialViewFieldResult,
            compute_spatial_view_field,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create trajectory data
        n_time = 1000
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        # Random spike times
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
        )

        assert isinstance(result, SpatialViewFieldResult)

    def test_field_shape_matches_env(self) -> None:
        """Test that field has same number of bins as environment."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
        )

        assert len(result.field) == env.n_bins
        assert result.env.n_bins == env.n_bins

    def test_view_occupancy_computed(self) -> None:
        """Test that view_occupancy is computed and has correct shape."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
        )

        # View occupancy should be non-negative
        assert np.all(result.view_occupancy >= 0)
        # Some bins should have view occupancy
        assert np.sum(result.view_occupancy) > 0

    def test_handles_viewing_outside_environment(self) -> None:
        """Test that NaN viewed locations are handled correctly."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        # Create a small environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(40, 60, (500, 2))  # Small central region
        env = Environment.from_samples(samples, bin_size=2.0)

        n_time = 1000
        times = np.linspace(0, 100, n_time)

        # Positions inside environment but viewing outside
        positions = np.full((n_time, 2), 50.0)  # Center
        headings = np.zeros(n_time)  # Looking right, outside env

        spike_times = rng.choice(times, size=50, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            view_distance=100.0,  # Far outside
            gaze_model="fixed_distance",
        )

        # Should still return valid result
        assert result.field is not None
        assert len(result.field) == env.n_bins

    def test_binned_method(self) -> None:
        """Test field computation with binned method."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            smoothing_method="binned",
        )

        # Field should be valid
        assert np.all(np.isfinite(result.field) | np.isnan(result.field))

    def test_diffusion_kde_method(self) -> None:
        """Test field computation with diffusion_kde method."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Field should be valid
        assert np.all(np.isfinite(result.field) | np.isnan(result.field))

    def test_gaussian_kde_method(self) -> None:
        """Test field computation with gaussian_kde method."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 500  # Smaller for speed
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 50, n_time)

        spike_times = rng.choice(times, size=50, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            smoothing_method="gaussian_kde",
            bandwidth=5.0,
        )

        # Field should be valid
        assert np.all(np.isfinite(result.field) | np.isnan(result.field))

    def test_min_occupancy_filtering(self) -> None:
        """Test that low-occupancy bins are set to NaN."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 100  # Short trajectory
        times = np.linspace(0, 10, n_time)

        # Concentrated positions with consistent viewing
        positions = rng.uniform(45, 55, (n_time, 2))
        headings = np.zeros(n_time)  # All looking same direction

        spike_times = rng.choice(times, size=10, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            min_occupancy_seconds=0.5,  # High threshold
        )

        # Many bins should be NaN due to low view occupancy
        assert np.any(np.isnan(result.field))

    def test_gaze_model_fixed_distance(self) -> None:
        """Test field computation with fixed_distance gaze model."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(20, 80, (n_time, 2))  # Keep views inside
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        assert result.field is not None

    def test_gaze_model_ray_cast(self) -> None:
        """Test field computation with ray_cast gaze model."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 500  # Smaller for speed
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 50, n_time)

        spike_times = rng.choice(times, size=50, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            gaze_model="ray_cast",
        )

        assert result.field is not None

    def test_gaze_offsets_parameter(self) -> None:
        """Test field computation with gaze offsets (eye tracking)."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(20, 80, (n_time, 2))  # Keep views inside
        headings = np.zeros(n_time)  # All facing same direction
        gaze_offsets = rng.uniform(-0.2, 0.2, n_time)  # Small eye movements
        times = np.linspace(0, 100, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            gaze_offsets=gaze_offsets,
            view_distance=10.0,
        )

        assert result.field is not None
        assert len(result.field) == env.n_bins


class TestValidation:
    """Tests for input validation."""

    def test_validation_empty_spike_times(self) -> None:
        """Test that empty spike times raises ValueError."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        spike_times = np.array([])  # No spikes

        with pytest.raises(ValueError, match="spike"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
            )

    def test_validation_times_positions_mismatch(self) -> None:
        """Test validation when times and positions have different lengths."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (50, 2))  # Wrong length
        headings = rng.uniform(-np.pi, np.pi, 100)
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
            )

    def test_validation_times_headings_mismatch(self) -> None:
        """Test validation when times and headings have different lengths."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (100, 2))
        headings = rng.uniform(-np.pi, np.pi, 50)  # Wrong length
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
            )

    def test_validation_invalid_method(self) -> None:
        """Test validation for invalid smoothing method."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="smoothing_method"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                smoothing_method="invalid_method",  # type: ignore[arg-type]
            )

    def test_validation_invalid_gaze_model(self) -> None:
        """Test validation for invalid gaze model."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="gaze"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                gaze_model="invalid_gaze",  # type: ignore[arg-type]
            )

    def test_validation_gaze_offsets_length_mismatch(self) -> None:
        """Test validation when gaze_offsets has wrong length."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (100, 2))
        headings = rng.uniform(-np.pi, np.pi, 100)
        gaze_offsets = rng.uniform(-0.1, 0.1, 50)  # Wrong length
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="gaze_offsets"):
            compute_spatial_view_field(
                env=env,
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                gaze_offsets=gaze_offsets,
            )


class TestRecoverGroundTruthFromSimulation:
    """Tests for recovering ground truth from simulated spatial view cells."""

    def test_view_field_peaks_at_preferred_view_location(self) -> None:
        """Test that view field peaks at preferred view location."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial_view import compute_spatial_view_field
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create spatial view cell with known preferred location
        preferred_view = np.array([70.0, 70.0])

        svc = SpatialViewCellModel(
            env=env,
            preferred_view_location=preferred_view,
            view_field_width=10.0,
            view_distance=15.0,
            gaze_model="fixed_distance",
            max_rate=20.0,
        )

        # Generate trajectory with varied views
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        # Compute firing rates (headings must be passed as keyword argument)
        rates = svc.firing_rate(positions, headings=headings)

        # Generate spikes (Poisson)
        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        # Compute view field
        result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            view_distance=15.0,
            gaze_model="fixed_distance",
        )

        # Find peak location in field
        valid_mask = np.isfinite(result.field)
        if not np.any(valid_mask):
            pytest.skip("No valid bins in field")

        peak_idx = np.nanargmax(result.field)
        peak_location = env.bin_centers[peak_idx]

        # Peak should be near preferred view location
        distance_to_preferred = np.linalg.norm(peak_location - preferred_view)
        assert distance_to_preferred < 20.0  # Within 20 units

    def test_view_field_differs_from_place_field_for_svc(self) -> None:
        """Test that view field differs from place field for spatial view cells."""
        from neurospatial import Environment
        from neurospatial.encoding.place import compute_place_field
        from neurospatial.encoding.spatial_view import compute_spatial_view_field
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # Create environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(samples, bin_size=5.0)

        # Create SVC with preferred view location far from animal positions
        preferred_view = np.array([80.0, 80.0])

        svc = SpatialViewCellModel(
            env=env,
            preferred_view_location=preferred_view,
            view_field_width=8.0,
            view_distance=15.0,
            gaze_model="fixed_distance",
            max_rate=20.0,
        )

        # Generate trajectory
        n_time = 5000
        times = np.linspace(0, 500, n_time)
        positions = rng.uniform(10, 90, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)

        rates = svc.firing_rate(positions, headings=headings)

        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        if len(spike_times) < 100:
            pytest.skip("Not enough spikes generated")

        # Compute both fields
        view_result = compute_spatial_view_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            view_distance=15.0,
            gaze_model="fixed_distance",
        )

        place_field = compute_place_field(
            env=env,
            spike_times=spike_times,
            times=times,
            positions=positions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # View field peak location
        valid_view = np.isfinite(view_result.field)
        valid_place = np.isfinite(place_field)

        if not np.any(valid_view) or not np.any(valid_place):
            pytest.skip("Not enough valid bins")

        view_peak_idx = np.nanargmax(view_result.field)
        place_peak_idx = np.nanargmax(place_field)

        view_peak_loc = env.bin_centers[view_peak_idx]
        place_peak_loc = env.bin_centers[place_peak_idx]

        # View field peak should be near preferred_view (80, 80)
        # Place field peak should be different (based on animal positions)
        view_dist_to_preferred = np.linalg.norm(view_peak_loc - preferred_view)
        place_dist_to_preferred = np.linalg.norm(place_peak_loc - preferred_view)

        # View field should be closer to preferred view location
        assert view_dist_to_preferred < place_dist_to_preferred


class TestExportsAndIntegration:
    """Tests for module exports and integration."""

    def test_exported_from_neurospatial(self) -> None:
        """Test that compute_spatial_view_field is exported from neurospatial."""
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        assert callable(compute_spatial_view_field)

    def test_result_exported_from_neurospatial(self) -> None:
        """Test that SpatialViewFieldResult is exported from neurospatial."""
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult

        assert SpatialViewFieldResult is not None
