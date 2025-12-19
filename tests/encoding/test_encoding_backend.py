"""Tests for neurospatial.encoding._backend module.

This module tests the backend selection infrastructure:
- get_backend: Select computation backend (numpy, jax, auto)
- is_jax_available: Check if JAX is available on current platform
- SUPPORTED_BACKENDS: List of valid backend names

TDD approach: Tests written first, implementation follows.

Design requirements (from PLAN.md):
- "numpy" (default) works everywhere, including Windows
- "jax" requires JAX installation (Linux/macOS only)
- "auto" uses JAX if available, falls back to NumPy silently on Windows
  or if JAX not installed
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# ==============================================================================
# Helper function (must be defined before used in decorators)
# ==============================================================================


def _has_jax() -> bool:
    """Check if JAX is available on this platform."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


# ==============================================================================
# Test SUPPORTED_BACKENDS constant
# ==============================================================================


class TestSupportedBackends:
    """Tests for SUPPORTED_BACKENDS constant."""

    def test_supported_backends_is_tuple(self) -> None:
        """SUPPORTED_BACKENDS should be an immutable tuple."""
        from neurospatial.encoding._backend import SUPPORTED_BACKENDS

        assert isinstance(SUPPORTED_BACKENDS, tuple)

    def test_supported_backends_contains_numpy(self) -> None:
        """SUPPORTED_BACKENDS should contain 'numpy'."""
        from neurospatial.encoding._backend import SUPPORTED_BACKENDS

        assert "numpy" in SUPPORTED_BACKENDS

    def test_supported_backends_contains_jax(self) -> None:
        """SUPPORTED_BACKENDS should contain 'jax'."""
        from neurospatial.encoding._backend import SUPPORTED_BACKENDS

        assert "jax" in SUPPORTED_BACKENDS

    def test_supported_backends_contains_auto(self) -> None:
        """SUPPORTED_BACKENDS should contain 'auto'."""
        from neurospatial.encoding._backend import SUPPORTED_BACKENDS

        assert "auto" in SUPPORTED_BACKENDS


# ==============================================================================
# Test is_jax_available function
# ==============================================================================


class TestIsJaxAvailable:
    """Tests for is_jax_available function."""

    def test_returns_bool(self) -> None:
        """is_jax_available should return a boolean."""
        from neurospatial.encoding._backend import is_jax_available

        result = is_jax_available()
        assert isinstance(result, bool)

    @patch("sys.platform", "win32")
    def test_returns_false_on_windows(self) -> None:
        """is_jax_available should return False on Windows.

        JAX does not officially support Windows, so we always return False
        on Windows regardless of whether JAX is installed.
        """

        # Need to reimport to pick up the mocked platform
        import importlib

        import neurospatial.encoding._backend as backend_module

        importlib.reload(backend_module)

        assert backend_module.is_jax_available() is False

    @patch("sys.platform", "darwin")
    def test_returns_true_on_macos_if_jax_installed(self) -> None:
        """is_jax_available should return True on macOS if JAX is installed."""
        if not _has_jax():
            pytest.skip("JAX not installed")

        import importlib

        import neurospatial.encoding._backend as backend_module

        importlib.reload(backend_module)

        assert backend_module.is_jax_available() is True

    @patch("sys.platform", "linux")
    def test_returns_true_on_linux_if_jax_installed(self) -> None:
        """is_jax_available should return True on Linux if JAX is installed."""
        if not _has_jax():
            pytest.skip("JAX not installed")

        import importlib

        import neurospatial.encoding._backend as backend_module

        importlib.reload(backend_module)

        assert backend_module.is_jax_available() is True


# ==============================================================================
# Test get_backend function
# ==============================================================================


class TestGetBackend:
    """Tests for get_backend function."""

    def test_numpy_backend_returns_numpy(self) -> None:
        """get_backend('numpy') should return numpy module."""
        import numpy as np

        from neurospatial.encoding._backend import get_backend

        backend = get_backend("numpy")
        assert backend is np

    def test_numpy_backend_always_works(self) -> None:
        """get_backend('numpy') should work on any platform."""
        import numpy as np

        from neurospatial.encoding._backend import get_backend

        # Even on Windows or without JAX, numpy should work
        backend = get_backend("numpy")
        assert backend is np

    @pytest.mark.skipif(
        not _has_jax(), reason="JAX not installed or not available on this platform"
    )
    def test_jax_backend_returns_jax_numpy(self) -> None:
        """get_backend('jax') should return jax.numpy module when available."""
        import jax.numpy as jnp

        from neurospatial.encoding._backend import get_backend

        backend = get_backend("jax")
        assert backend is jnp

    def test_jax_backend_raises_when_unavailable(self) -> None:
        """get_backend('jax') should raise ImportError when JAX unavailable."""
        from neurospatial.encoding._backend import get_backend, is_jax_available

        if is_jax_available():
            pytest.skip("JAX is available, can't test unavailable path")

        with pytest.raises(ImportError, match="JAX"):
            get_backend("jax")

    def test_auto_backend_returns_numpy_or_jax(self) -> None:
        """get_backend('auto') should return numpy or jax.numpy."""
        import numpy as np

        from neurospatial.encoding._backend import get_backend, is_jax_available

        backend = get_backend("auto")

        if is_jax_available():
            import jax.numpy as jnp

            assert backend is jnp
        else:
            assert backend is np

    def test_auto_backend_returns_numpy_on_windows(self) -> None:
        """get_backend('auto') should return numpy on Windows."""
        import importlib

        import numpy as np

        import neurospatial.encoding._backend as backend_module

        # Mock Windows platform
        with patch.object(sys, "platform", "win32"):
            importlib.reload(backend_module)
            backend = backend_module.get_backend("auto")
            assert backend is np

    def test_invalid_backend_raises_valueerror(self) -> None:
        """get_backend should raise ValueError for invalid backend names."""
        from neurospatial.encoding._backend import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")

    def test_invalid_backend_suggests_valid_options(self) -> None:
        """Error message should suggest valid backend options."""
        from neurospatial.encoding._backend import get_backend

        with pytest.raises(ValueError) as exc_info:
            get_backend("invalid")

        error_msg = str(exc_info.value)
        assert "numpy" in error_msg
        assert "jax" in error_msg
        assert "auto" in error_msg


# ==============================================================================
# Test get_backend_name function
# ==============================================================================


class TestGetBackendName:
    """Tests for get_backend_name function."""

    def test_numpy_returns_numpy_string(self) -> None:
        """get_backend_name('numpy') should return 'numpy'."""
        from neurospatial.encoding._backend import get_backend_name

        assert get_backend_name("numpy") == "numpy"

    @pytest.mark.skipif(
        not _has_jax(), reason="JAX not installed or not available on this platform"
    )
    def test_jax_returns_jax_string(self) -> None:
        """get_backend_name('jax') should return 'jax'."""
        from neurospatial.encoding._backend import get_backend_name

        assert get_backend_name("jax") == "jax"

    def test_auto_resolves_to_actual_backend(self) -> None:
        """get_backend_name('auto') should return actual backend name used."""
        from neurospatial.encoding._backend import get_backend_name, is_jax_available

        result = get_backend_name("auto")

        if is_jax_available():
            assert result == "jax"
        else:
            assert result == "numpy"


# ==============================================================================
# Test module imports
# ==============================================================================


class TestBackendImports:
    """Test that all expected items are importable from _backend module."""

    def test_get_backend_importable(self) -> None:
        """get_backend should be importable from encoding._backend."""
        from neurospatial.encoding._backend import get_backend

        assert callable(get_backend)

    def test_is_jax_available_importable(self) -> None:
        """is_jax_available should be importable from encoding._backend."""
        from neurospatial.encoding._backend import is_jax_available

        assert callable(is_jax_available)

    def test_supported_backends_importable(self) -> None:
        """SUPPORTED_BACKENDS should be importable from encoding._backend."""
        from neurospatial.encoding._backend import SUPPORTED_BACKENDS

        assert SUPPORTED_BACKENDS is not None

    def test_get_backend_name_importable(self) -> None:
        """get_backend_name should be importable from encoding._backend."""
        from neurospatial.encoding._backend import get_backend_name

        assert callable(get_backend_name)


# ==============================================================================
# Test edge cases
# ==============================================================================


class TestBackendEdgeCases:
    """Test edge cases and robustness."""

    def test_get_backend_case_sensitivity(self) -> None:
        """Backend names should be case-sensitive (lowercase required)."""
        from neurospatial.encoding._backend import get_backend

        with pytest.raises(ValueError):
            get_backend("NumPy")  # Should be "numpy"

        with pytest.raises(ValueError):
            get_backend("NUMPY")

        with pytest.raises(ValueError):
            get_backend("Jax")  # Should be "jax"

        with pytest.raises(ValueError):
            get_backend("AUTO")  # Should be "auto"

    def test_multiple_calls_return_same_module(self) -> None:
        """Multiple get_backend calls should return the same module object."""
        from neurospatial.encoding._backend import get_backend

        backend1 = get_backend("numpy")
        backend2 = get_backend("numpy")
        assert backend1 is backend2

    @pytest.mark.skipif(
        not _has_jax(), reason="JAX not installed or not available on this platform"
    )
    def test_jax_backend_multiple_calls_consistent(self) -> None:
        """Multiple get_backend('jax') calls should return the same module."""
        from neurospatial.encoding._backend import get_backend

        backend1 = get_backend("jax")
        backend2 = get_backend("jax")
        assert backend1 is backend2
