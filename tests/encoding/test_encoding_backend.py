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

import numpy as np
import pytest

# ==============================================================================
# Helper function (must be defined before used in decorators)
# ==============================================================================


def _has_jax() -> bool:
    """Check if JAX is available on this platform."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


@pytest.fixture(autouse=True)
def _clear_backend_availability_cache():
    """Restore ``_backend`` to a clean state after each test.

    Several tests in this file mock ``sys.platform`` and then call
    ``importlib.reload(backend_module)`` to re-evaluate the module
    under the mocked platform. The ``@patch`` decorator restores
    ``sys.platform`` itself when the test exits, but the module is
    left in whatever state the perturbed reload produced. Without a
    teardown, subsequent tests run against that perturbed module —
    and which test runs next is order-dependent under xdist.

    The teardown reloads the module under the *real* (post-patch)
    ``sys.platform``, so every test starts from a known clean state.
    The ``is_jax_available`` LRU cache is also cleared explicitly,
    in case the reload didn't reset it on this Python version.
    """
    yield

    import importlib

    import neurospatial.encoding._backend as backend_module

    backend_module.is_jax_available.cache_clear()
    importlib.reload(backend_module)
    backend_module.is_jax_available.cache_clear()


# ==============================================================================
# Test SUPPORTED_BACKENDS constant
# ==============================================================================


class TestSupportedBackends:
    """Tests for SUPPORTED_BACKENDS constant."""


class TestIsJaxAvailable:
    """Tests for is_jax_available function."""

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


@pytest.mark.skipif(not _has_jax(), reason="JAX not installed (optional 'jax' extra)")
class TestMinOccupancyPrecisionAcrossBackends:
    """Regression: JAX firing-rate kernels must compare min_occupancy at
    occupancy's dtype.

    A previous refactor passed `jnp.float32(min_occupancy)` to the jit-compiled
    kernel, which truncated the threshold and let JAX keep bins that the NumPy
    kernel masked. A second related bug: JAX's default x64=False mode silently
    truncates `jnp.asarray(float64_arr, dtype=jnp.float64)` to float32, so the
    same divergence reappeared even after switching to dtype-preserving casts.
    The `_core_jax` module enables x64 globally at import time to fix that.

    Skipped if the optional ``jax`` extra is not installed: the test bodies
    ``import jax.numpy as jnp`` and would raise ``ModuleNotFoundError`` on a
    JAX-less dev environment without this guard.
    """

    def test_jax_matches_numpy_at_subfloat32_threshold_singular(self) -> None:
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import (
            compute_firing_rate_single as jax_single,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rate_single as np_single,
        )

        # Threshold and occupancy differ only in the last few float64 bits;
        # under float32 they collapse to equal and the strict ">=" passes.
        threshold = 0.100000005
        occupancy = np.array([0.100000001, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        spike_counts = np.ones(5, dtype=np.float64)

        np_rate = np.asarray(
            np_single(spike_counts, occupancy, min_occupancy=threshold)
        )
        jax_rate = np.asarray(
            jax_single(
                jnp.asarray(spike_counts),
                jnp.asarray(occupancy),
                min_occupancy=threshold,
            )
        )

        # NumPy must mask bin 0 (occupancy below threshold).
        assert np.isnan(np_rate[0]), "test setup not exercising the bug"
        # JAX must agree.
        np.testing.assert_array_equal(np.isnan(np_rate), np.isnan(jax_rate))

    def test_jax_matches_numpy_at_subfloat32_threshold_batch(self) -> None:
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import (
            compute_firing_rates_batch as jax_batch,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rates_batch as np_batch,
        )

        threshold = 0.100000005
        occupancy = np.array([0.100000001, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        spike_counts = np.ones((3, 5), dtype=np.float64)

        np_rates = np.asarray(
            np_batch(spike_counts, occupancy, min_occupancy=threshold)
        )
        jax_rates = np.asarray(
            jax_batch(
                jnp.asarray(spike_counts),
                jnp.asarray(occupancy),
                min_occupancy=threshold,
            )
        )

        assert np.all(np.isnan(np_rates[:, 0])), "test setup not exercising the bug"
        np.testing.assert_array_equal(np.isnan(np_rates), np.isnan(jax_rates))


@pytest.mark.skipif(not _has_jax(), reason="JAX not installed (optional 'jax' extra)")
def test_core_jax_import_enables_x64() -> None:
    """Importing the JAX encoding kernel turns on jax_enable_x64.

    Regression for a precision bug where production code left
    `jax_enable_x64=False` (JAX's default), so `jnp.asarray(float64_arr,
    dtype=jnp.float64)` silently truncated to float32. The encoding
    pipeline computes everything in float64 and is precision-sensitive
    around `min_occupancy` comparisons; we enable x64 at the JAX-backend
    module's import time so the production path (`compute_*_rate(
    backend="jax")`) always runs at float64.

    The conftest fixture forces x64 on for all encoding tests, so a
    naive ``reload + assert True`` here would pass even if the
    import-time toggle in `_core_jax.py` were deleted. Defeat the
    fixture by setting the flag to False *inside* the test, then
    reload `_core_jax`, and assert the import flipped it back on.
    The conftest's teardown still restores the user's original setting.
    """
    import importlib

    import jax

    import neurospatial.encoding._core_jax as core_jax_module

    # Defeat the conftest fixture: simulate a production process whose JAX
    # default (x64=False) is in effect at the moment _core_jax is imported.
    jax.config.update("jax_enable_x64", False)
    assert jax.config.read("jax_enable_x64") is False

    importlib.reload(core_jax_module)

    assert jax.config.read("jax_enable_x64") is True
