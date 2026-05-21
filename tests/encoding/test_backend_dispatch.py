"""Tests for the ``backend`` parameter on the encoding compute functions.

The eight ``compute_*_rate(s)`` functions all accept a ``backend`` kwarg
selecting between ``"numpy"``, ``"jax"``, and ``"auto"``. ``"jax"``
returns JAX arrays when the JAX backend is available, and ``"auto"``
uses JAX when available while falling back to NumPy otherwise. The
previous version of this file exercised three properties per function
across eight functions (24 tests):

- ``test_accepts_numpy_backend`` — call with ``backend="numpy"``,
  assert ``result is not None``.
- ``test_accepts_auto_backend`` — same, with ``backend="auto"``.
- ``test_rejects_invalid_backend`` — call with ``backend="invalid"``,
  assert ``ValueError("Unknown backend")``.

The two "accepts" tests per function only ever asserted ``result is not
None`` — they verify the call doesn't raise, not that the backend
actually applied; that's covered by every functional test in the suite
that exercises the same compute functions with their default backend.
The validation tests are the real contract worth pinning, so we keep
one parametrized version that loops over every entry point.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.encoding._backend import is_jax_available


@pytest.fixture
def simple_env() -> Environment:
    rng = np.random.default_rng(42)
    positions = rng.uniform(-50, 50, (1000, 2))
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def trajectory_times() -> NDArray[np.float64]:
    return np.linspace(0, 10, 1000)


@pytest.fixture
def trajectory_positions() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.uniform(-50, 50, (1000, 2))


@pytest.fixture
def trajectory_headings() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return np.cumsum(rng.uniform(-0.1, 0.1, 1000)) % (2 * np.pi)


@pytest.fixture
def spike_times() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return np.sort(rng.uniform(0, 10, 50))


@pytest.fixture
def multi_spike_times() -> list[NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    return [np.sort(rng.uniform(0, 10, 50)) for _ in range(3)]


@pytest.fixture
def object_positions() -> NDArray[np.float64]:
    return np.array([[10.0, 10.0], [-20.0, 15.0]])


def _make_call(
    func_name: str,
    simple_env,
    spike_times,
    multi_spike_times,
    trajectory_times,
    trajectory_positions,
    trajectory_headings,
    object_positions,
):
    """Build a callable for each compute_* entry point that takes ``backend``."""
    from neurospatial.encoding.directional import (
        compute_directional_rate,
        compute_directional_rates,
    )
    from neurospatial.encoding.egocentric import (
        compute_egocentric_rate,
        compute_egocentric_rates,
    )
    from neurospatial.encoding.spatial import (
        compute_spatial_rate,
        compute_spatial_rates,
    )
    from neurospatial.encoding.view import compute_view_rate, compute_view_rates

    table = {
        "compute_spatial_rate": lambda backend: compute_spatial_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            backend=backend,
        ),
        "compute_spatial_rates": lambda backend: compute_spatial_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            backend=backend,
        ),
        "compute_directional_rate": lambda backend: compute_directional_rate(
            spike_times,
            trajectory_times,
            trajectory_headings,
            backend=backend,
        ),
        "compute_directional_rates": lambda backend: compute_directional_rates(
            multi_spike_times,
            trajectory_times,
            trajectory_headings,
            backend=backend,
        ),
        "compute_view_rate": lambda backend: compute_view_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend=backend,
        ),
        "compute_view_rates": lambda backend: compute_view_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend=backend,
        ),
        "compute_egocentric_rate": lambda backend: compute_egocentric_rate(
            None,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend=backend,
        ),
        "compute_egocentric_rates": lambda backend: compute_egocentric_rates(
            None,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend=backend,
        ),
    }
    return table[func_name]


COMPUTE_FUNCTION_NAMES = [
    "compute_spatial_rate",
    "compute_spatial_rates",
    "compute_directional_rate",
    "compute_directional_rates",
    "compute_view_rate",
    "compute_view_rates",
    "compute_egocentric_rate",
    "compute_egocentric_rates",
]


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
def test_invalid_backend_raises(
    func_name,
    simple_env,
    spike_times,
    multi_spike_times,
    trajectory_times,
    trajectory_positions,
    trajectory_headings,
    object_positions,
):
    """``backend="invalid"`` raises ``ValueError("Unknown backend")``."""
    call = _make_call(
        func_name,
        simple_env,
        spike_times,
        multi_spike_times,
        trajectory_times,
        trajectory_positions,
        trajectory_headings,
        object_positions,
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        call("invalid")


@pytest.fixture(autouse=True)
def _clear_backend_availability_cache():
    """Restore ``is_jax_available``'s LRU cache between tests.

    The fallback test below monkeypatches ``sys.platform`` and clears
    the LRU; if anything in the test body or in ``_make_call`` raises,
    the cache is left in the ``False`` state and the next test sees a
    fake "JAX unavailable" environment. Clearing on both pre- and
    post-yield guarantees every test starts and ends from a clean
    state, regardless of failure mode or xdist scheduling. Mirrors the
    fixture pattern in tests/encoding/test_encoding_backend.py.
    """
    import importlib

    import neurospatial.encoding._backend as backend_module

    backend_module.is_jax_available.cache_clear()
    yield
    backend_module.is_jax_available.cache_clear()
    importlib.reload(backend_module)
    backend_module.is_jax_available.cache_clear()


def _is_jax_array(arr) -> bool:
    """True if ``arr`` is a ``jax.numpy`` array."""
    if not is_jax_available():
        return False
    import jax

    return isinstance(arr, jax.Array)


def _raw_firing_rate_array(result):
    """Pull the unconverted firing-rate array off a single or batch result."""
    return result.firing_rate if hasattr(result, "firing_rate") else result.firing_rates


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
@pytest.mark.parametrize("backend", ["numpy", "jax", "auto"])
def test_backend_smoke_computes_finite_result(
    func_name,
    backend,
    simple_env,
    spike_times,
    multi_spike_times,
    trajectory_times,
    trajectory_positions,
    trajectory_headings,
    object_positions,
):
    """Every entry point computes a finite result on every backend.

    Tripwire if dispatch wiring routes a backend to a no-op or stub.
    The audit found that:

    - The previous spatial-only smoke would have missed a regression in
      ``compute_directional_rate(s)`` ignoring ``backend="jax"``.
    - The previous "finite values" assertion would have *also* missed a
      regression where the JAX path silently returned NumPy arrays —
      the values would still be finite. The contract for
      ``backend="jax"`` (see ``directional.py:1511`` and ``:1786``) is
      that the output is a ``jax.Array``, so we assert that explicitly.
    - ``backend="auto"`` had no smoke coverage outside spatial.

    Explicit JAX backend cases skip when JAX is not available. ``auto``
    always runs so the NumPy fallback path stays covered.
    """
    jax_available = is_jax_available()
    if backend == "jax" and not jax_available:
        pytest.skip("JAX not installed (optional 'jax' extra)")

    call = _make_call(
        func_name,
        simple_env,
        spike_times,
        multi_spike_times,
        trajectory_times,
        trajectory_positions,
        trajectory_headings,
        object_positions,
    )
    result = call(backend)
    # Pull the raw firing-rate array off the result so we can check its
    # origin module before ``np.asarray`` erases JAX-ness.
    raw = _raw_firing_rate_array(result)

    if backend == "jax":
        # ``backend="jax"`` documented contract: output is a JAX array.
        # If the implementation silently dropped through to NumPy this
        # would be the test that catches it.
        assert _is_jax_array(raw), (
            f"{func_name} with backend='jax' returned "
            f"type={type(raw).__name__}; expected a jax.Array."
        )
    elif backend == "auto" and jax_available:
        # ``backend="auto"`` resolves to JAX when available, else NumPy.
        assert _is_jax_array(raw), (
            f"{func_name} with backend='auto' returned "
            f"type={type(raw).__name__}; expected a jax.Array since "
            "the JAX backend is available."
        )
    elif backend == "auto":
        # ``backend="auto"`` must remain usable without JAX.
        assert isinstance(raw, np.ndarray), (
            f"{func_name} with backend='auto' returned "
            f"type={type(raw).__name__}; expected np.ndarray because "
            "the JAX backend is unavailable."
        )
    else:  # backend == "numpy"
        assert isinstance(raw, np.ndarray), (
            f"{func_name} with backend='numpy' returned "
            f"type={type(raw).__name__}; expected np.ndarray."
        )

    firing_rate = np.asarray(raw)
    # Output must be non-empty and contain at least one finite value
    # (bins with zero occupancy are NaN by design — that's documented
    # behavior, not a failure). The contract is that the call actually
    # produced *some* meaningful firing-rate estimate, not just a stub.
    assert firing_rate.size > 0
    finite = np.isfinite(firing_rate)
    assert finite.any(), (
        f"{func_name} with backend={backend!r} returned all-NaN firing "
        "rate — call did not produce a real result."
    )
    # Finite values must be non-negative (firing rate cannot be < 0).
    assert (firing_rate[finite] >= 0).all()


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
def test_auto_backend_falls_back_to_numpy_when_jax_backend_unavailable(
    func_name,
    monkeypatch,
    simple_env,
    spike_times,
    multi_spike_times,
    trajectory_times,
    trajectory_positions,
    trajectory_headings,
    object_positions,
):
    """``backend="auto"`` remains usable when the JAX backend is unavailable.

    Cache cleanup is handled by the autouse
    ``_clear_backend_availability_cache`` fixture above; this test only
    needs to apply the platform patch and then clear so the next
    ``is_jax_available()`` call re-evaluates under the patched platform.
    """
    import sys

    import neurospatial.encoding._backend as backend_module

    # Patch first, *then* clear so the next ``is_jax_available()`` call
    # re-evaluates under the patched platform. Doing it the other way
    # leaves a stale ``True`` cached if anything reads the predicate
    # between the clear and the patch.
    monkeypatch.setattr(sys, "platform", "win32")
    backend_module.is_jax_available.cache_clear()

    call = _make_call(
        func_name,
        simple_env,
        spike_times,
        multi_spike_times,
        trajectory_times,
        trajectory_positions,
        trajectory_headings,
        object_positions,
    )
    result = call("auto")
    raw = _raw_firing_rate_array(result)

    assert isinstance(raw, np.ndarray), (
        f"{func_name} with backend='auto' returned "
        f"type={type(raw).__name__}; expected np.ndarray when the "
        "JAX backend is unavailable."
    )


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
def test_jax_backend_propagates_import_error_when_unavailable(
    func_name,
    monkeypatch,
    simple_env,
    spike_times,
    multi_spike_times,
    trajectory_times,
    trajectory_positions,
    trajectory_headings,
    object_positions,
):
    """``backend="jax"`` must raise ``ImportError`` on JAX-unavailable platforms.

    Closes the end-to-end gap left by
    ``test_encoding_backend.py::test_jax_backend_raises_when_unavailable``,
    which covers only ``get_backend("jax")`` at the unit layer. A future
    refactor that catches-and-falls-back inside any compute function
    would silently violate the documented "raises ImportError" contract;
    this parametrized test fires across every entry point so the
    contract holds end-to-end.
    """
    import sys

    import neurospatial.encoding._backend as backend_module

    monkeypatch.setattr(sys, "platform", "win32")
    backend_module.is_jax_available.cache_clear()

    call = _make_call(
        func_name,
        simple_env,
        spike_times,
        multi_spike_times,
        trajectory_times,
        trajectory_positions,
        trajectory_headings,
        object_positions,
    )
    with pytest.raises(ImportError, match="JAX"):
        call("jax")
