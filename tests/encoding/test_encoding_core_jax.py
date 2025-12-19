"""Tests for neurospatial.encoding._core_jax module.

This module tests the JAX core array operations stubs:
- compute_firing_rate_single: Convert spike_counts + occupancy → firing_rate
- compute_firing_rates_batch: Batch version for multiple neurons
- smooth_rate_map_single: Apply smoothing to a single rate map
- smooth_rate_maps_batch: Batch version for multiple rate maps

TDD approach: Tests written first, implementation follows.

Design requirements (from PLAN.md):
- Core Rate/Metrics Layer operates on dense arrays with shapes like (n_neurons, n_bins)
- Single-neuron functions operate on (n_bins,) arrays
- Batch functions operate on (n_neurons, n_bins) arrays
- All functions should raise NotImplementedError initially (stubs)
- JAX module mirrors _core_numpy.py interface exactly

Note: These tests are skipped if JAX is not available on the platform.
"""

from __future__ import annotations

import pytest

from neurospatial.encoding._backend import is_jax_available

# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(
    not is_jax_available(),
    reason="JAX is not available on this platform",
)


# ==============================================================================
# Test module imports
# ==============================================================================


class TestCoreJaxImports:
    """Test that all expected items are importable from _core_jax module."""

    def test_module_importable(self) -> None:
        """_core_jax module should be importable."""
        import neurospatial.encoding._core_jax  # noqa: F401

    def test_compute_firing_rate_single_importable(self) -> None:
        """compute_firing_rate_single should be importable from encoding._core_jax."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        assert callable(compute_firing_rate_single)

    def test_compute_firing_rates_batch_importable(self) -> None:
        """compute_firing_rates_batch should be importable from encoding._core_jax."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        assert callable(compute_firing_rates_batch)

    def test_smooth_rate_map_single_importable(self) -> None:
        """smooth_rate_map_single should be importable from encoding._core_jax."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        assert callable(smooth_rate_map_single)

    def test_smooth_rate_maps_batch_importable(self) -> None:
        """smooth_rate_maps_batch should be importable from encoding._core_jax."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        assert callable(smooth_rate_maps_batch)

    def test_all_exports_defined(self) -> None:
        """__all__ should be defined and contain expected exports."""
        from neurospatial.encoding._core_jax import __all__

        assert "compute_firing_rate_single" in __all__
        assert "compute_firing_rates_batch" in __all__
        assert "smooth_rate_map_single" in __all__
        assert "smooth_rate_maps_batch" in __all__


# ==============================================================================
# Test compute_firing_rate_single stub
# ==============================================================================


class TestComputeFiringRateSingleStub:
    """Tests for compute_firing_rate_single stub function."""

    def test_raises_not_implemented_error(self) -> None:
        """compute_firing_rate_single should raise NotImplementedError (stub)."""
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import compute_firing_rate_single

        spike_counts = jnp.array([0, 1, 2, 0, 1], dtype=jnp.float64)
        occupancy = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            compute_firing_rate_single(spike_counts, occupancy)

    def test_function_has_docstring(self) -> None:
        """compute_firing_rate_single should have a docstring."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        assert compute_firing_rate_single.__doc__ is not None
        assert len(compute_firing_rate_single.__doc__) > 50


# ==============================================================================
# Test compute_firing_rates_batch stub
# ==============================================================================


class TestComputeFiringRatesBatchStub:
    """Tests for compute_firing_rates_batch stub function."""

    def test_raises_not_implemented_error(self) -> None:
        """compute_firing_rates_batch should raise NotImplementedError (stub)."""
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        # Shape: (n_neurons, n_bins)
        spike_counts = jnp.array([[0, 1, 2, 0, 1], [1, 0, 1, 2, 0]], dtype=jnp.float64)
        occupancy = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            compute_firing_rates_batch(spike_counts, occupancy)

    def test_function_has_docstring(self) -> None:
        """compute_firing_rates_batch should have a docstring."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        assert compute_firing_rates_batch.__doc__ is not None
        assert len(compute_firing_rates_batch.__doc__) > 50


# ==============================================================================
# Test smooth_rate_map_single stub
# ==============================================================================


class TestSmoothRateMapSingleStub:
    """Tests for smooth_rate_map_single stub function."""

    def test_raises_not_implemented_error(self) -> None:
        """smooth_rate_map_single should raise NotImplementedError (stub)."""
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import smooth_rate_map_single

        firing_rate = jnp.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=jnp.float64)
        # Mock adjacency matrix (sparse)
        adjacency = jnp.eye(5, dtype=jnp.float64)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            smooth_rate_map_single(
                firing_rate,
                adjacency,
                bandwidth=5.0,
                method="diffusion_kde",
            )

    def test_function_has_docstring(self) -> None:
        """smooth_rate_map_single should have a docstring."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        assert smooth_rate_map_single.__doc__ is not None
        assert len(smooth_rate_map_single.__doc__) > 50


# ==============================================================================
# Test smooth_rate_maps_batch stub
# ==============================================================================


class TestSmoothRateMapsBatchStub:
    """Tests for smooth_rate_maps_batch stub function."""

    def test_raises_not_implemented_error(self) -> None:
        """smooth_rate_maps_batch should raise NotImplementedError (stub)."""
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        # Shape: (n_neurons, n_bins)
        firing_rates = jnp.array(
            [[0.0, 1.0, 2.0, 1.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0]],
            dtype=jnp.float64,
        )
        # Mock adjacency matrix (sparse)
        adjacency = jnp.eye(5, dtype=jnp.float64)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            smooth_rate_maps_batch(
                firing_rates,
                adjacency,
                bandwidth=5.0,
                method="diffusion_kde",
            )

    def test_function_has_docstring(self) -> None:
        """smooth_rate_maps_batch should have a docstring."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        assert smooth_rate_maps_batch.__doc__ is not None
        assert len(smooth_rate_maps_batch.__doc__) > 50


# ==============================================================================
# Test function signature requirements
# ==============================================================================


class TestFunctionSignatures:
    """Test that functions have the expected parameter signatures."""

    def test_compute_firing_rate_single_signature(self) -> None:
        """compute_firing_rate_single should have expected parameters."""
        import inspect

        from neurospatial.encoding._core_jax import compute_firing_rate_single

        sig = inspect.signature(compute_firing_rate_single)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "spike_counts" in params
        assert "occupancy" in params

    def test_compute_firing_rates_batch_signature(self) -> None:
        """compute_firing_rates_batch should have expected parameters."""
        import inspect

        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        sig = inspect.signature(compute_firing_rates_batch)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "spike_counts" in params
        assert "occupancy" in params

    def test_smooth_rate_map_single_signature(self) -> None:
        """smooth_rate_map_single should have expected parameters."""
        import inspect

        from neurospatial.encoding._core_jax import smooth_rate_map_single

        sig = inspect.signature(smooth_rate_map_single)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "firing_rate" in params
        assert "adjacency" in params
        # Optional parameters
        assert "bandwidth" in params
        assert "method" in params

    def test_smooth_rate_maps_batch_signature(self) -> None:
        """smooth_rate_maps_batch should have expected parameters."""
        import inspect

        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        sig = inspect.signature(smooth_rate_maps_batch)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "firing_rates" in params
        assert "adjacency" in params
        # Optional parameters
        assert "bandwidth" in params
        assert "method" in params


# ==============================================================================
# Test type annotations
# ==============================================================================


class TestTypeAnnotations:
    """Test that functions have proper type annotations."""

    def test_compute_firing_rate_single_has_annotations(self) -> None:
        """compute_firing_rate_single should have type annotations."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        annotations = compute_firing_rate_single.__annotations__
        assert "spike_counts" in annotations
        assert "occupancy" in annotations
        assert "return" in annotations

    def test_compute_firing_rates_batch_has_annotations(self) -> None:
        """compute_firing_rates_batch should have type annotations."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        annotations = compute_firing_rates_batch.__annotations__
        assert "spike_counts" in annotations
        assert "occupancy" in annotations
        assert "return" in annotations

    def test_smooth_rate_map_single_has_annotations(self) -> None:
        """smooth_rate_map_single should have type annotations."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        annotations = smooth_rate_map_single.__annotations__
        assert "firing_rate" in annotations
        assert "adjacency" in annotations
        assert "return" in annotations

    def test_smooth_rate_maps_batch_has_annotations(self) -> None:
        """smooth_rate_maps_batch should have type annotations."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        annotations = smooth_rate_maps_batch.__annotations__
        assert "firing_rates" in annotations
        assert "adjacency" in annotations
        assert "return" in annotations


# ==============================================================================
# Test JAX-specific behavior
# ==============================================================================


class TestJaxSpecificBehavior:
    """Test JAX-specific requirements for the core module."""

    def test_module_uses_jax_array_type_hint(self) -> None:
        """Functions should use JAX-compatible type hints (jax.Array)."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        # Check that the return annotation mentions Array (JAX's array type)
        # This will be jax.Array in the actual implementation
        annotations = compute_firing_rate_single.__annotations__
        return_type = str(annotations.get("return", ""))
        # The type hint should include Array (from jax or as a string)
        assert "Array" in return_type or "NDArray" in return_type
