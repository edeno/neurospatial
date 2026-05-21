"""Tests for the ``backend`` parameter on the encoding compute functions.

The eight ``compute_*_rate(s)`` functions accept ``backend`` in
``{"numpy", "jax", "auto"}``. ``"jax"`` returns JAX arrays when the
JAX backend is available, ``"auto"`` falls back to NumPy otherwise.
This file pins three contracts per entry point:

1. ``backend="invalid"`` raises ``ValueError("Unknown backend")``.
2. The configured backend's array type appears on the result's
   ``firing_rate`` / ``firing_rates`` field.
3. ``backend="jax"`` raises ``ImportError`` when JAX is unavailable;
   ``backend="auto"`` falls back to NumPy.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.encoding._backend import is_jax_available
from neurospatial.encoding._base import _is_jax_array


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


@pytest.fixture
def compute_inputs(
    simple_env: Environment,
    spike_times: NDArray[np.float64],
    multi_spike_times: list[NDArray[np.float64]],
    trajectory_times: NDArray[np.float64],
    trajectory_positions: NDArray[np.float64],
    trajectory_headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
) -> tuple:
    """Bundle the seven raw input fixtures every compute function needs.

    Each parametrized test below would otherwise list all seven in its
    signature and forward them positionally to ``_make_call``.
    """
    return (
        simple_env,
        spike_times,
        multi_spike_times,
        trajectory_times,
        trajectory_positions,
        trajectory_headings,
        object_positions,
    )


def _simulate_jax_unavailable(monkeypatch) -> None:
    """Force ``is_jax_available()`` to return ``False`` for this test.

    Patches ``sys.platform`` to ``"win32"`` (the documented JAX-
    unavailable trigger at ``_backend.py:81``) and clears the LRU cache
    so the next ``is_jax_available()`` call re-evaluates under the
    patched platform. The conftest-level
    ``restore_backend_availability_cache`` fixture restores state on
    teardown.
    """
    import neurospatial.encoding._backend as backend_module

    monkeypatch.setattr(sys, "platform", "win32")
    backend_module.is_jax_available.cache_clear()


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
def test_invalid_backend_raises(func_name, compute_inputs):
    """``backend="invalid"`` raises ``ValueError("Unknown backend")``."""
    call = _make_call(func_name, *compute_inputs)
    with pytest.raises(ValueError, match="Unknown backend"):
        call("invalid")


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
@pytest.mark.parametrize("backend", ["numpy", "jax", "auto"])
def test_backend_smoke_computes_finite_result(func_name, backend, compute_inputs):
    """The configured backend dictates the output array type and a real result is produced."""
    jax_available = is_jax_available()
    if backend == "jax" and not jax_available:
        pytest.skip("JAX not installed (optional 'jax' extra)")

    expect_jax = backend == "jax" or (backend == "auto" and jax_available)
    expected_type = "jax.Array" if expect_jax else "np.ndarray"

    call = _make_call(func_name, *compute_inputs)
    result = call(backend)
    raw = result.firing_rate if hasattr(result, "firing_rate") else result.firing_rates

    if expect_jax:
        assert _is_jax_array(raw), (
            f"{func_name} with backend={backend!r} returned "
            f"type={type(raw).__name__}; expected {expected_type}."
        )
    else:
        assert isinstance(raw, np.ndarray), (
            f"{func_name} with backend={backend!r} returned "
            f"type={type(raw).__name__}; expected {expected_type}."
        )

    firing_rate = np.asarray(raw)
    # Bins with zero occupancy are NaN by design; the contract is that
    # *some* finite bin exists (the call wasn't a stub) and finite
    # values are non-negative.
    assert firing_rate.size > 0
    finite = np.isfinite(firing_rate)
    assert finite.any(), (
        f"{func_name} with backend={backend!r} returned all-NaN firing rate."
    )
    assert (firing_rate[finite] >= 0).all()


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
def test_auto_backend_falls_back_to_numpy_when_jax_backend_unavailable(
    func_name, compute_inputs, monkeypatch
):
    """``backend="auto"`` returns NumPy when the JAX backend is unavailable."""
    _simulate_jax_unavailable(monkeypatch)

    call = _make_call(func_name, *compute_inputs)
    result = call("auto")
    raw = result.firing_rate if hasattr(result, "firing_rate") else result.firing_rates

    assert isinstance(raw, np.ndarray), (
        f"{func_name} with backend='auto' returned "
        f"type={type(raw).__name__}; expected np.ndarray when the "
        "JAX backend is unavailable."
    )


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
def test_jax_backend_propagates_import_error_when_unavailable(
    func_name, compute_inputs, monkeypatch
):
    """``backend="jax"`` propagates ``ImportError`` end-to-end when JAX is unavailable.

    ``test_encoding_backend.py::test_jax_backend_raises_when_unavailable``
    covers only ``get_backend("jax")`` at the unit layer; this test
    fires the same condition through every compute entry point so a
    future catches-and-falls-back regression inside any of them
    surfaces immediately.
    """
    _simulate_jax_unavailable(monkeypatch)
    call = _make_call(func_name, *compute_inputs)
    with pytest.raises(ImportError, match="JAX"):
        call("jax")
