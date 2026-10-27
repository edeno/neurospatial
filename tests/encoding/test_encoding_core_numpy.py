"""Tests for neurospatial.encoding._core_numpy module.

This module tests the NumPy core array operations:
- compute_firing_rates_batch: Convert spike_counts + occupancy → firing_rates

Design requirements:
- Core Rate/Metrics Layer operates on dense arrays with shapes like (n_neurons, n_bins)
- Batch functions operate on (n_neurons, n_bins) arrays
"""

from __future__ import annotations

# ==============================================================================
# Test module imports
# ==============================================================================


class TestCoreNumpyImports:
    """Test that all expected items are importable from _core_numpy module."""

    def test_module_importable(self) -> None:
        """_core_numpy module should be importable."""
        import neurospatial.encoding._core_numpy  # noqa: F401

    def test_compute_firing_rates_batch_importable(self) -> None:
        """compute_firing_rates_batch should be importable from encoding._core_numpy."""
        from neurospatial.encoding._core_numpy import compute_firing_rates_batch

        assert callable(compute_firing_rates_batch)

    def test_all_exports_defined(self) -> None:
        """__all__ should be defined and contain expected exports."""
        from neurospatial.encoding._core_numpy import __all__

        assert "compute_firing_rates_batch" in __all__


# ==============================================================================
# Test compute_firing_rates_batch
# ==============================================================================


class TestComputeFiringRatesBatch:
    """Tests for compute_firing_rates_batch function."""

    def test_function_has_docstring(self) -> None:
        """compute_firing_rates_batch should have a docstring."""
        from neurospatial.encoding._core_numpy import compute_firing_rates_batch

        assert compute_firing_rates_batch.__doc__ is not None
        assert len(compute_firing_rates_batch.__doc__) > 50


# ==============================================================================
# Test function signature requirements
# ==============================================================================


class TestFunctionSignatures:
    """Test that functions have the expected parameter signatures."""

    def test_compute_firing_rates_batch_signature(self) -> None:
        """compute_firing_rates_batch should have expected parameters."""
        import inspect

        from neurospatial.encoding._core_numpy import compute_firing_rates_batch

        sig = inspect.signature(compute_firing_rates_batch)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "spike_counts" in params
        assert "occupancy" in params


# ==============================================================================
# Test type annotations
# ==============================================================================


class TestTypeAnnotations:
    """Test that functions have proper type annotations."""

    def test_compute_firing_rates_batch_has_annotations(self) -> None:
        """compute_firing_rates_batch should have type annotations."""
        from neurospatial.encoding._core_numpy import compute_firing_rates_batch

        annotations = compute_firing_rates_batch.__annotations__
        assert "spike_counts" in annotations
        assert "occupancy" in annotations
        assert "return" in annotations
