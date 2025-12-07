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

# Functions that should be importable from encoding.place
ENCODING_PLACE_FUNCTIONS = [
    "compute_place_field",
    "compute_directional_place_fields",
    "spikes_to_rate_map",
    "detect_place_fields",
    "skaggs_information",
    "sparsity",
    "selectivity",
    "rate_map_centroid",
    "field_size",
    "field_stability",
    "field_shape_metrics",
    "field_shift_distance",
    "in_out_field_ratio",
    "information_per_second",
    "mutual_information",
    "rate_map_coherence",
    "spatial_coverage_single_cell",
    "compute_field_emd",
]

# Classes that should be importable from encoding.place
ENCODING_PLACE_CLASSES = [
    "DirectionalPlaceFields",
]

# Items that should be importable from encoding __init__
ENCODING_INIT_FUNCTIONS = [
    "compute_place_field",
    "skaggs_information",
    "detect_place_fields",
]

ENCODING_INIT_CLASSES = [
    "DirectionalPlaceFields",
]


class TestEncodingPlaceImports:
    """Test all imports from neurospatial.encoding.place."""

    @pytest.mark.parametrize("func_name", ENCODING_PLACE_FUNCTIONS)
    def test_function_importable_from_encoding_place(self, func_name: str) -> None:
        """Verify {func_name} is importable from encoding.place."""
        from neurospatial.encoding import place

        func = getattr(place, func_name, None)
        assert func is not None, f"{func_name} not found in encoding.place"
        assert callable(func), f"{func_name} should be callable"

    @pytest.mark.parametrize("class_name", ENCODING_PLACE_CLASSES)
    def test_class_importable_from_encoding_place(self, class_name: str) -> None:
        """Verify {class_name} is importable from encoding.place."""
        from neurospatial.encoding import place

        cls = getattr(place, class_name, None)
        assert cls is not None, f"{class_name} not found in encoding.place"


class TestEncodingPlaceFromEncodingInit:
    """Test all imports from neurospatial.encoding (via __init__.py)."""

    @pytest.mark.parametrize("func_name", ENCODING_INIT_FUNCTIONS)
    def test_function_importable_from_encoding_init(self, func_name: str) -> None:
        """Verify {func_name} is importable from encoding __init__."""
        from neurospatial import encoding

        func = getattr(encoding, func_name, None)
        assert func is not None, f"{func_name} not found in encoding __init__"
        assert callable(func), f"{func_name} should be callable"

    @pytest.mark.parametrize("class_name", ENCODING_INIT_CLASSES)
    def test_class_importable_from_encoding_init(self, class_name: str) -> None:
        """Verify {class_name} is importable from encoding __init__."""
        from neurospatial import encoding

        cls = getattr(encoding, class_name, None)
        assert cls is not None, f"{class_name} not found in encoding __init__"


# Keep original specific import tests for coverage verification
class TestEncodingPlaceSpecificImports:
    """Verify specific import statements work (for IDE/static analysis)."""

    def test_direct_import_compute_place_field(self) -> None:
        """Direct import statement works."""
        from neurospatial.encoding.place import compute_place_field

        assert callable(compute_place_field)

    def test_direct_import_detect_place_fields(self) -> None:
        """Direct import statement works."""
        from neurospatial.encoding import detect_place_fields

        assert callable(detect_place_fields)


# ==============================================================================
# Test basic functionality
# ==============================================================================
class TestEncodingPlaceFunctionality:
    """Test basic functionality of encoding.place functions."""

    @pytest.fixture
    def env_and_data(self):
        """Create test environment and trajectory data with RNG.

        Returns
        -------
        tuple
            (env, positions, times, spike_times, rng) where rng is the
            seeded random number generator for reproducible tests.
        """
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        times = np.linspace(0, 50, 500)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = rng.uniform(0, 50, 25)
        return env, positions, times, spike_times, rng

    def test_compute_place_field_runs(self, env_and_data) -> None:
        """compute_place_field should run and return correct shape."""
        from neurospatial.encoding.place import compute_place_field

        env, positions, times, spike_times, _rng = env_and_data
        result = compute_place_field(env, spike_times, times, positions, bandwidth=10.0)
        assert result.shape == (env.n_bins,)
        assert np.any(np.isfinite(result))

    def test_spikes_to_rate_map_runs(self, env_and_data) -> None:
        """spikes_to_rate_map should run and return correct shape."""
        from neurospatial.encoding.place import spikes_to_rate_map

        env, positions, times, spike_times, _rng = env_and_data
        result = spikes_to_rate_map(env, spike_times, times, positions)
        assert result.shape == (env.n_bins,)

    def test_detect_place_fields_runs(self, env_and_data) -> None:
        """detect_place_fields should run and return list."""
        from neurospatial.encoding.place import detect_place_fields

        env, _positions, _times, _spike_times, _rng = env_and_data
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

        env, _positions, _times, _spike_times, rng = env_and_data
        firing_rate = rng.random(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        info = skaggs_information(firing_rate, occupancy)
        assert isinstance(info, float)

    def test_sparsity_runs(self, env_and_data) -> None:
        """sparsity should run and return float."""
        from neurospatial.encoding.place import sparsity

        env, _positions, _times, _spike_times, rng = env_and_data
        firing_rate = rng.random(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        spars = sparsity(firing_rate, occupancy)
        assert isinstance(spars, float)
        assert 0.0 <= spars <= 1.0

    def test_selectivity_runs(self, env_and_data) -> None:
        """selectivity should run and return float."""
        from neurospatial.encoding.place import selectivity

        env, _positions, _times, _spike_times, rng = env_and_data
        firing_rate = rng.random(env.n_bins) * 5.0
        occupancy = np.ones(env.n_bins) / env.n_bins
        select = selectivity(firing_rate, occupancy)
        assert isinstance(select, float)
        assert select >= 1.0  # selectivity >= 1 by definition

    def test_field_size_runs(self, env_and_data) -> None:
        """field_size should run and return float."""
        from neurospatial.encoding.place import field_size

        env, _positions, _times, _spike_times, _rng = env_and_data
        field_bins = np.array([0, 1, 2, 3, 4])
        size = field_size(field_bins, env)
        assert isinstance(size, float)
        assert size > 0

    def test_rate_map_centroid_runs(self, env_and_data) -> None:
        """rate_map_centroid should run and return array."""
        from neurospatial.encoding.place import rate_map_centroid

        env, _positions, _times, _spike_times, rng = env_and_data
        firing_rate = rng.random(env.n_bins) * 5.0
        field_bins = np.array([0, 1, 2, 3, 4])
        centroid = rate_map_centroid(firing_rate, field_bins, env)
        assert centroid.shape == (2,)

    def test_field_stability_runs(self) -> None:
        """field_stability should run and return float."""
        from neurospatial.encoding.place import field_stability

        rng = np.random.default_rng(42)
        rate_map_1 = rng.random(100) * 5.0
        rate_map_2 = rate_map_1 + rng.standard_normal(100) * 0.1
        stability = field_stability(rate_map_1, rate_map_2)
        assert isinstance(stability, float)

    def test_rate_map_coherence_runs(self, env_and_data) -> None:
        """rate_map_coherence should run and return float."""
        from neurospatial.encoding.place import rate_map_coherence

        env, _positions, _times, _spike_times, _rng = env_and_data
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

        env, positions, times, spike_times, _rng = env_and_data
        direction_labels = np.array(
            ["forward"] * 250 + ["backward"] * 250, dtype=object
        )
        result = compute_directional_place_fields(
            env, spike_times, times, positions, direction_labels, bandwidth=10.0
        )
        assert isinstance(result, DirectionalPlaceFields)
        assert "forward" in result.fields
        assert "backward" in result.fields
