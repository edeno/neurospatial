"""Tests for neurospatial.encoding.place module (TDD RED phase).

These tests verify that all place field functions are importable from the correct
encoding module locations after the module reorganization.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

# ==============================================================================
# Test imports from encoding.place (only functions that remain in place.py)
# ==============================================================================

# Functions that should be importable from encoding.place
ENCODING_PLACE_FUNCTIONS = [
    "compute_place_field",
    "skaggs_information",
]

# Functions moved to encoding.spatial
ENCODING_SPATIAL_FUNCTIONS = [
    "compute_directional_place_fields",
    "detect_place_fields",
]

# Classes moved to encoding.spatial
ENCODING_SPATIAL_CLASSES = [
    "DirectionalPlaceFields",
]

# Functions moved to encoding._metrics
ENCODING_METRICS_FUNCTIONS = [
    "sparsity",
    "selectivity",
    "information_per_second",
    "mutual_information",
    "spatial_coverage_single_cell",
]

# Functions moved to encoding._field_metrics
ENCODING_FIELD_METRICS_FUNCTIONS = [
    "rate_map_centroid",
    "field_size",
    "field_stability",
    "field_shape_metrics",
    "field_shift_distance",
    "in_out_field_ratio",
    "rate_map_coherence",
    "compute_field_emd",
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


class TestEncodingSpatialImports:
    """Test all imports from neurospatial.encoding.spatial."""

    @pytest.mark.parametrize("func_name", ENCODING_SPATIAL_FUNCTIONS)
    def test_function_importable_from_encoding_spatial(self, func_name: str) -> None:
        """Verify {func_name} is importable from encoding.spatial."""
        from neurospatial.encoding import spatial

        func = getattr(spatial, func_name, None)
        assert func is not None, f"{func_name} not found in encoding.spatial"
        assert callable(func), f"{func_name} should be callable"

    @pytest.mark.parametrize("class_name", ENCODING_SPATIAL_CLASSES)
    def test_class_importable_from_encoding_spatial(self, class_name: str) -> None:
        """Verify {class_name} is importable from encoding.spatial."""
        from neurospatial.encoding import spatial

        cls = getattr(spatial, class_name, None)
        assert cls is not None, f"{class_name} not found in encoding.spatial"


class TestEncodingMetricsImports:
    """Test all imports from neurospatial.encoding._metrics."""

    @pytest.mark.parametrize("func_name", ENCODING_METRICS_FUNCTIONS)
    def test_function_importable_from_encoding_metrics(self, func_name: str) -> None:
        """Verify {func_name} is importable from encoding._metrics."""
        from neurospatial.encoding import _metrics

        func = getattr(_metrics, func_name, None)
        assert func is not None, f"{func_name} not found in encoding._metrics"
        assert callable(func), f"{func_name} should be callable"


class TestEncodingFieldMetricsImports:
    """Test all imports from neurospatial.encoding._field_metrics."""

    @pytest.mark.parametrize("func_name", ENCODING_FIELD_METRICS_FUNCTIONS)
    def test_function_importable_from_encoding_field_metrics(
        self, func_name: str
    ) -> None:
        """Verify {func_name} is importable from encoding._field_metrics."""
        from neurospatial.encoding import _field_metrics

        func = getattr(_field_metrics, func_name, None)
        assert func is not None, f"{func_name} not found in encoding._field_metrics"
        assert callable(func), f"{func_name} should be callable"


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
        spike_times = rng.uniform(0, 50, 30)
        return env, positions, times, spike_times, rng

    def test_compute_place_field_runs(self, env_and_data):
        """compute_place_field runs without error."""
        from neurospatial.encoding.place import compute_place_field

        env, positions, times, spike_times, _rng = env_and_data
        result = compute_place_field(env, spike_times, times, positions, bandwidth=10.0)
        assert result.shape == (env.n_bins,)

    def test_binned_rate_map_runs(self, env_and_data):
        """_binned_rate_map runs without error."""
        from neurospatial.encoding.place import _binned_rate_map

        env, positions, times, spike_times, _rng = env_and_data
        result = _binned_rate_map(env, spike_times, times, positions)
        assert result.shape == (env.n_bins,)

    def test_detect_place_fields_runs(self, env_and_data):
        """detect_place_fields runs without error."""
        from neurospatial.encoding.place import compute_place_field
        from neurospatial.encoding.spatial import detect_place_fields

        env, positions, times, spike_times, _rng = env_and_data
        firing_rate = compute_place_field(
            env, spike_times, times, positions, bandwidth=10.0
        )
        fields = detect_place_fields(firing_rate, env)
        assert isinstance(fields, list)

    def test_skaggs_information_runs(self, env_and_data):
        """skaggs_information runs without error."""
        from neurospatial.encoding.place import compute_place_field, skaggs_information

        env, positions, times, spike_times, _rng = env_and_data
        firing_rate = compute_place_field(
            env, spike_times, times, positions, bandwidth=10.0
        )
        occupancy = env.occupancy(times, positions, return_seconds=True)
        info = skaggs_information(firing_rate, occupancy)
        assert isinstance(info, float)
