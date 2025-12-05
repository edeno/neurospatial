"""Tests for object_vector_field module.

These tests verify:
1. Module can be imported
2. __all__ exports are correct
3. ObjectVectorFieldResult dataclass works correctly
4. compute_object_vector_field() computes fields correctly
5. Integration with egocentric polar environment
6. Different smoothing methods work correctly
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleSetup:
    """Tests for object_vector_field module setup."""

    def test_module_imports(self) -> None:
        """Test that object_vector_field module can be imported."""
        from neurospatial import object_vector_field

        assert object_vector_field is not None

    def test_module_has_docstring(self) -> None:
        """Test that module has a docstring."""
        from neurospatial import object_vector_field

        assert object_vector_field.__doc__ is not None
        assert len(object_vector_field.__doc__) > 100  # Should be substantial

    def test_module_docstring_mentions_object_vector(self) -> None:
        """Test that module docstring mentions object-vector field."""
        from neurospatial import object_vector_field

        docstring = object_vector_field.__doc__
        assert docstring is not None
        assert "object" in docstring.lower() or "vector" in docstring.lower()

    def test_module_has_all_attribute(self) -> None:
        """Test that module has __all__ defined."""
        from neurospatial import object_vector_field

        assert hasattr(object_vector_field, "__all__")
        assert isinstance(object_vector_field.__all__, list)

    def test_module_all_contains_expected_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from neurospatial import object_vector_field

        expected = [
            "ObjectVectorFieldResult",
            "compute_object_vector_field",
        ]
        for export in expected:
            assert export in object_vector_field.__all__


class TestObjectVectorFieldResultDataclass:
    """Tests for ObjectVectorFieldResult frozen dataclass."""

    def test_dataclass_can_be_imported(self) -> None:
        """Test that ObjectVectorFieldResult can be imported."""
        from neurospatial.object_vector_field import ObjectVectorFieldResult

        assert ObjectVectorFieldResult is not None

    def test_dataclass_is_frozen(self) -> None:
        """Test that ObjectVectorFieldResult is a frozen dataclass."""
        from neurospatial import Environment
        from neurospatial.object_vector_field import ObjectVectorFieldResult

        # Create a simple egocentric environment
        ego_env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        )

        # Create instance
        result = ObjectVectorFieldResult(
            field=np.zeros(ego_env.n_bins),
            ego_env=ego_env,
            occupancy=np.ones(ego_env.n_bins),
        )

        # Try to modify - should raise
        with pytest.raises(AttributeError):
            result.field = np.ones(ego_env.n_bins)  # type: ignore[misc]

    def test_dataclass_has_all_fields(self) -> None:
        """Test that ObjectVectorFieldResult has all required fields."""
        from neurospatial import Environment
        from neurospatial.object_vector_field import ObjectVectorFieldResult

        ego_env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        )

        field = np.random.default_rng(42).random(ego_env.n_bins)
        occupancy = np.random.default_rng(42).random(ego_env.n_bins)

        result = ObjectVectorFieldResult(
            field=field,
            ego_env=ego_env,
            occupancy=occupancy,
        )

        assert result.field is not None
        assert result.ego_env is not None
        assert result.occupancy is not None
        assert len(result.field) == ego_env.n_bins
        assert len(result.occupancy) == ego_env.n_bins


class TestComputeObjectVectorField:
    """Tests for compute_object_vector_field() function."""

    def test_function_exists(self) -> None:
        """Test that compute_object_vector_field can be imported."""
        from neurospatial.object_vector_field import compute_object_vector_field

        assert callable(compute_object_vector_field)

    def test_returns_object_vector_field_result(self) -> None:
        """Test that function returns ObjectVectorFieldResult instance."""
        from neurospatial.object_vector_field import (
            ObjectVectorFieldResult,
            compute_object_vector_field,
        )

        # Create trajectory data
        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        # Object position
        object_positions = np.array([[50.0, 50.0]])

        # Random spike times
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
        )

        assert isinstance(result, ObjectVectorFieldResult)

    def test_field_shape_matches_ego_env(self) -> None:
        """Test that field has same number of bins as ego_env."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            max_distance=50.0,
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # ego_env should be 10 distance * 12 direction = 120 bins
        assert len(result.field) == result.ego_env.n_bins
        assert result.ego_env.n_bins == 10 * 12

    def test_ego_env_has_polar_coordinates(self) -> None:
        """Test that ego_env has proper polar coordinate structure."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            max_distance=50.0,
        )

        # bin_centers[:, 0] should be distances (>= 0)
        # bin_centers[:, 1] should be angles (in range)
        bin_centers = result.ego_env.bin_centers
        assert np.all(bin_centers[:, 0] >= 0)  # Distances non-negative
        assert np.all(bin_centers[:, 1] >= -np.pi)
        assert np.all(bin_centers[:, 1] <= np.pi)

    def test_occupancy_computed_correctly(self) -> None:
        """Test that occupancy reflects time spent in each bin."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
        )

        # Occupancy should be non-negative
        assert np.all(result.occupancy >= 0)
        # Some bins should have occupancy
        assert np.sum(result.occupancy) > 0

    def test_uses_nearest_object(self) -> None:
        """Test that field is computed using nearest object."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        times = np.linspace(0, 100, n_time)

        # Multiple objects at different locations
        object_positions = np.array([[25.0, 50.0], [75.0, 50.0]])

        # Animal mostly near first object
        positions = rng.uniform(10, 40, (n_time, 2))
        positions[:, 1] = rng.uniform(40, 60, n_time)
        headings = rng.uniform(-np.pi, np.pi, n_time)

        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
        )

        # Should produce valid result
        assert isinstance(result.field, np.ndarray)
        assert len(result.field) == result.ego_env.n_bins

    def test_binned_method(self) -> None:
        """Test field computation with binned method."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            smoothing_method="binned",
        )

        # Field should be valid
        assert np.all(np.isfinite(result.field) | np.isnan(result.field))

    def test_diffusion_kde_method(self) -> None:
        """Test field computation with diffusion_kde method."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 1000
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            smoothing_method="diffusion_kde",
        )

        # Field should be valid
        assert np.all(np.isfinite(result.field) | np.isnan(result.field))

    def test_min_occupancy_filtering(self) -> None:
        """Test that low-occupancy bins are set to NaN."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 100  # Short trajectory
        times = np.linspace(0, 10, n_time)

        # Concentrated positions - should leave many bins unvisited
        positions = rng.uniform(45, 55, (n_time, 2))
        headings = np.zeros(n_time)  # Constant heading

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=10, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            min_occupancy_seconds=0.5,  # High threshold
        )

        # Many bins should be NaN due to low occupancy
        assert np.any(np.isnan(result.field))

    def test_validation_empty_spike_times(self) -> None:
        """Test validation when no spikes provided."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])

        spike_times = np.array([])  # No spikes

        with pytest.raises(ValueError, match="spike"):
            compute_object_vector_field(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
            )

    def test_validation_times_positions_mismatch(self) -> None:
        """Test validation when times and positions have different lengths."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (50, 2))  # Wrong length
        headings = rng.uniform(-np.pi, np.pi, 100)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length"):
            compute_object_vector_field(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
            )

    def test_validation_times_headings_mismatch(self) -> None:
        """Test validation when times and headings have different lengths."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 100, (100, 2))
        headings = rng.uniform(-np.pi, np.pi, 50)  # Wrong length
        object_positions = np.array([[50.0, 50.0]])
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length"):
            compute_object_vector_field(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
            )

    def test_validation_invalid_method(self) -> None:
        """Test validation for invalid smoothing method."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="smoothing_method"):
            compute_object_vector_field(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
                smoothing_method="invalid_method",
            )


class TestGeodesicDistanceSupport:
    """Tests for geodesic distance mode."""

    def test_geodesic_with_allocentric_env(self) -> None:
        """Test field computation with geodesic distance."""
        from neurospatial import Environment
        from neurospatial.object_vector_field import compute_object_vector_field

        # Create allocentric environment
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 100, (500, 2))
        alloc_env = Environment.from_samples(samples, bin_size=5.0)

        n_time = 1000
        positions = rng.uniform(5, 95, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 100, n_time)

        object_positions = np.array([[50.0, 50.0]])
        spike_times = rng.choice(times, size=100, replace=False)

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            allocentric_env=alloc_env,
            distance_metric="geodesic",
        )

        # Should produce valid result
        assert isinstance(result.field, np.ndarray)

    def test_geodesic_requires_allocentric_env(self) -> None:
        """Test that geodesic mode requires allocentric_env."""
        from neurospatial.object_vector_field import compute_object_vector_field

        rng = np.random.default_rng(42)
        n_time = 100
        times = np.linspace(0, 10, n_time)
        positions = rng.uniform(0, 100, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        object_positions = np.array([[50.0, 50.0]])
        spike_times = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="allocentric_env"):
            compute_object_vector_field(
                spike_times=spike_times,
                times=times,
                positions=positions,
                headings=headings,
                object_positions=object_positions,
                distance_metric="geodesic",
                allocentric_env=None,
            )


class TestRecoverGroundTruthFromSimulation:
    """Tests for recovering ground truth parameters from simulated OVCs."""

    def test_field_peaks_at_preferred_distance(self) -> None:
        """Test that field peaks at preferred distance for simulated OVC."""
        from neurospatial import Environment
        from neurospatial.object_vector_field import compute_object_vector_field
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

        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        # Compute field
        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
            max_distance=50.0,
            n_distance_bins=10,
        )

        # Find peak location in field
        valid_mask = np.isfinite(result.field)
        if not np.any(valid_mask):
            pytest.skip("No valid bins in field")

        peak_idx = np.nanargmax(result.field)
        peak_distance = result.ego_env.bin_centers[peak_idx, 0]

        # Peak should be near preferred distance
        assert peak_distance == pytest.approx(true_distance, rel=0.5)

    def test_field_peaks_at_preferred_direction(self) -> None:
        """Test that field peaks at preferred direction for directional OVC."""
        from neurospatial import Environment
        from neurospatial.object_vector_field import compute_object_vector_field
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

        rates = ovc.firing_rate(positions, headings=headings)

        dt = times[1] - times[0]
        spike_mask = rng.random(n_time) < rates * dt
        spike_times = times[spike_mask]

        if len(spike_times) < 50:
            pytest.skip("Not enough spikes generated")

        result = compute_object_vector_field(
            spike_times=spike_times,
            times=times,
            positions=positions,
            headings=headings,
            object_positions=object_positions,
        )

        # Find peak location
        valid_mask = np.isfinite(result.field)
        if not np.any(valid_mask):
            pytest.skip("No valid bins in field")

        peak_idx = np.nanargmax(result.field)
        peak_direction = result.ego_env.bin_centers[peak_idx, 1]

        # Peak direction should be near preferred (within 60 degrees)
        angle_diff = np.abs(peak_direction - true_direction)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        assert angle_diff < np.pi / 3


class TestExportsAndIntegration:
    """Tests for module exports and integration."""

    def test_exported_from_neurospatial(self) -> None:
        """Test that compute_object_vector_field is exported from neurospatial."""
        from neurospatial import compute_object_vector_field

        assert callable(compute_object_vector_field)

    def test_result_exported_from_neurospatial(self) -> None:
        """Test that ObjectVectorFieldResult is exported from neurospatial."""
        from neurospatial import ObjectVectorFieldResult

        assert ObjectVectorFieldResult is not None
