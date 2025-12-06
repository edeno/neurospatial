"""Tests for neurospatial.encoding.place module (TDD RED phase).

These tests verify that all place field functions are importable from the new
encoding.place location after the module reorganization.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


# ==============================================================================
# Test imports from encoding.place
# ==============================================================================
class TestEncodingPlaceImports:
    """Test all imports from neurospatial.encoding.place."""

    # --- Classes ---

    def test_import_directional_place_fields(self) -> None:
        """DirectionalPlaceFields should be importable from encoding.place."""
        from neurospatial.encoding.place import DirectionalPlaceFields

        assert DirectionalPlaceFields is not None

    # --- Spike field functions (from spike_field.py) ---

    def test_import_compute_place_field(self) -> None:
        """compute_place_field should be importable from encoding.place."""
        from neurospatial.encoding.place import compute_place_field

        assert callable(compute_place_field)

    def test_import_compute_directional_place_fields(self) -> None:
        """compute_directional_place_fields should be importable from encoding.place."""
        from neurospatial.encoding.place import compute_directional_place_fields

        assert callable(compute_directional_place_fields)

    def test_import_spikes_to_field(self) -> None:
        """spikes_to_field should be importable from encoding.place."""
        from neurospatial.encoding.place import spikes_to_field

        assert callable(spikes_to_field)

    # --- Place field metrics (from metrics/place_fields.py) ---

    def test_import_detect_place_fields(self) -> None:
        """detect_place_fields should be importable from encoding.place."""
        from neurospatial.encoding.place import detect_place_fields

        assert callable(detect_place_fields)

    def test_import_skaggs_information(self) -> None:
        """skaggs_information should be importable from encoding.place."""
        from neurospatial.encoding.place import skaggs_information

        assert callable(skaggs_information)

    def test_import_sparsity(self) -> None:
        """sparsity should be importable from encoding.place."""
        from neurospatial.encoding.place import sparsity

        assert callable(sparsity)

    def test_import_selectivity(self) -> None:
        """selectivity should be importable from encoding.place."""
        from neurospatial.encoding.place import selectivity

        assert callable(selectivity)

    def test_import_field_centroid(self) -> None:
        """field_centroid should be importable from encoding.place."""
        from neurospatial.encoding.place import field_centroid

        assert callable(field_centroid)

    def test_import_field_size(self) -> None:
        """field_size should be importable from encoding.place."""
        from neurospatial.encoding.place import field_size

        assert callable(field_size)

    def test_import_field_stability(self) -> None:
        """field_stability should be importable from encoding.place."""
        from neurospatial.encoding.place import field_stability

        assert callable(field_stability)

    def test_import_field_shape_metrics(self) -> None:
        """field_shape_metrics should be importable from encoding.place."""
        from neurospatial.encoding.place import field_shape_metrics

        assert callable(field_shape_metrics)

    def test_import_field_shift_distance(self) -> None:
        """field_shift_distance should be importable from encoding.place."""
        from neurospatial.encoding.place import field_shift_distance

        assert callable(field_shift_distance)

    def test_import_in_out_field_ratio(self) -> None:
        """in_out_field_ratio should be importable from encoding.place."""
        from neurospatial.encoding.place import in_out_field_ratio

        assert callable(in_out_field_ratio)

    def test_import_information_per_second(self) -> None:
        """information_per_second should be importable from encoding.place."""
        from neurospatial.encoding.place import information_per_second

        assert callable(information_per_second)

    def test_import_mutual_information(self) -> None:
        """mutual_information should be importable from encoding.place."""
        from neurospatial.encoding.place import mutual_information

        assert callable(mutual_information)

    def test_import_rate_map_coherence(self) -> None:
        """rate_map_coherence should be importable from encoding.place."""
        from neurospatial.encoding.place import rate_map_coherence

        assert callable(rate_map_coherence)

    def test_import_spatial_coverage_single_cell(self) -> None:
        """spatial_coverage_single_cell should be importable from encoding.place."""
        from neurospatial.encoding.place import spatial_coverage_single_cell

        assert callable(spatial_coverage_single_cell)

    def test_import_compute_field_emd(self) -> None:
        """compute_field_emd should be importable from encoding.place."""
        from neurospatial.encoding.place import compute_field_emd

        assert callable(compute_field_emd)


class TestEncodingPlaceFromEncodingInit:
    """Test all imports from neurospatial.encoding (via __init__.py)."""

    def test_import_compute_place_field_from_encoding(self) -> None:
        """compute_place_field should be importable from encoding __init__."""
        from neurospatial.encoding import compute_place_field

        assert callable(compute_place_field)

    def test_import_directional_place_fields_from_encoding(self) -> None:
        """DirectionalPlaceFields should be importable from encoding __init__."""
        from neurospatial.encoding import DirectionalPlaceFields

        assert DirectionalPlaceFields is not None

    def test_import_skaggs_information_from_encoding(self) -> None:
        """skaggs_information should be importable from encoding __init__."""
        from neurospatial.encoding import skaggs_information

        assert callable(skaggs_information)

    def test_import_detect_place_fields_from_encoding(self) -> None:
        """detect_place_fields should be importable from encoding __init__."""
        from neurospatial.encoding import detect_place_fields

        assert callable(detect_place_fields)


# ==============================================================================
# Test basic functionality
# ==============================================================================
class TestEncodingPlaceFunctionality:
    """Test basic functionality of encoding.place functions."""

    @pytest.fixture
    def env_and_data(self):
        """Create test environment and trajectory data."""
        np.random.seed(42)
        positions = np.random.uniform(0, 100, (500, 2))
        times = np.linspace(0, 50, 500)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.random.uniform(0, 50, 25)
        return env, positions, times, spike_times

    def test_compute_place_field_runs(self, env_and_data) -> None:
        """compute_place_field should run and return correct shape."""
        from neurospatial.encoding.place import compute_place_field

        env, positions, times, spike_times = env_and_data
        result = compute_place_field(env, spike_times, times, positions, bandwidth=10.0)
        assert result.shape == (env.n_bins,)
        assert np.any(np.isfinite(result))

    def test_spikes_to_field_runs(self, env_and_data) -> None:
        """spikes_to_field should run and return correct shape."""
        from neurospatial.encoding.place import spikes_to_field

        env, positions, times, spike_times = env_and_data
        result = spikes_to_field(env, spike_times, times, positions)
        assert result.shape == (env.n_bins,)

    def test_detect_place_fields_runs(self, env_and_data) -> None:
        """detect_place_fields should run and return list."""
        from neurospatial.encoding.place import detect_place_fields

        env, _positions, _times, _spike_times = env_and_data
        # Create a simple Gaussian firing field
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            dist = np.linalg.norm(env.bin_centers[i] - np.array([50, 50]))
            firing_rate[i] = 5.0 * np.exp(-(dist**2) / (2 * 20.0**2))

        fields = detect_place_fields(firing_rate, env)
        assert isinstance(fields, list)

    def test_skaggs_information_runs(self, env_and_data) -> None:
        """skaggs_information should run and return float."""
        from neurospatial.encoding.place import skaggs_information

        env, _positions, _times, _spike_times = env_and_data
        firing_rate = np.random.rand(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        info = skaggs_information(firing_rate, occupancy)
        assert isinstance(info, float)

    def test_sparsity_runs(self, env_and_data) -> None:
        """sparsity should run and return float."""
        from neurospatial.encoding.place import sparsity

        env, _positions, _times, _spike_times = env_and_data
        firing_rate = np.random.rand(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        spars = sparsity(firing_rate, occupancy)
        assert isinstance(spars, float)
        assert 0.0 <= spars <= 1.0

    def test_selectivity_runs(self, env_and_data) -> None:
        """selectivity should run and return float."""
        from neurospatial.encoding.place import selectivity

        env, _positions, _times, _spike_times = env_and_data
        firing_rate = np.random.rand(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        select = selectivity(firing_rate, occupancy)
        assert isinstance(select, float)
        assert select >= 1.0  # selectivity >= 1 by definition

    def test_field_size_runs(self, env_and_data) -> None:
        """field_size should run and return float."""
        from neurospatial.encoding.place import field_size

        env, _positions, _times, _spike_times = env_and_data
        field_bins = np.array([0, 1, 2, 3, 4])
        size = field_size(field_bins, env)
        assert isinstance(size, float)
        assert size > 0

    def test_field_centroid_runs(self, env_and_data) -> None:
        """field_centroid should run and return array."""
        from neurospatial.encoding.place import field_centroid

        env, _positions, _times, _spike_times = env_and_data
        firing_rate = np.random.rand(env.n_bins) * 5.0
        field_bins = np.array([0, 1, 2, 3, 4])
        centroid = field_centroid(firing_rate, field_bins, env)
        assert centroid.shape == (2,)

    def test_field_stability_runs(self) -> None:
        """field_stability should run and return float."""
        from neurospatial.encoding.place import field_stability

        rate_map_1 = np.random.rand(100) * 5.0
        rate_map_2 = rate_map_1 + np.random.randn(100) * 0.1
        stability = field_stability(rate_map_1, rate_map_2)
        assert isinstance(stability, float)

    def test_rate_map_coherence_runs(self, env_and_data) -> None:
        """rate_map_coherence should run and return float."""
        from neurospatial.encoding.place import rate_map_coherence

        env, _positions, _times, _spike_times = env_and_data
        # Create smooth field for coherence
        firing_rate = np.zeros(env.n_bins)
        for i in range(env.n_bins):
            dist = np.linalg.norm(env.bin_centers[i] - np.array([50, 50]))
            firing_rate[i] = 5.0 * np.exp(-(dist**2) / (2 * 20.0**2))
        coherence = rate_map_coherence(firing_rate, env)
        assert isinstance(coherence, float)

    def test_directional_place_fields_runs(self, env_and_data) -> None:
        """compute_directional_place_fields should run and return DirectionalPlaceFields."""
        from neurospatial.encoding.place import (
            DirectionalPlaceFields,
            compute_directional_place_fields,
        )

        env, positions, times, spike_times = env_and_data
        direction_labels = np.array(
            ["forward"] * 250 + ["backward"] * 250, dtype=object
        )
        result = compute_directional_place_fields(
            env, spike_times, times, positions, direction_labels, bandwidth=10.0
        )
        assert isinstance(result, DirectionalPlaceFields)
        assert "forward" in result.fields
        assert "backward" in result.fields
