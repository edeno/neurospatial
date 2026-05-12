"""Tests for the ``backend`` parameter on the encoding compute functions.

The eight ``compute_*_rate(s)`` functions all accept a ``backend`` kwarg
selecting between ``"numpy"`` (the active implementation) and
``"jax"`` / ``"auto"`` (currently aliased to numpy). The previous
version of this file exercised three properties per function across
eight functions (24 tests):

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


def _has_jax() -> bool:
    """Check if JAX is available on this platform."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


def _firing_rate_array(result):
    """Pull the firing-rate array out of a single- or batch-rate result.

    Single-rate results expose ``.firing_rate``; batch-rate results
    expose ``.firing_rates`` (n_neurons, n_bins). Both are valid output
    shapes for this smoke test.
    """
    return np.asarray(
        getattr(result, "firing_rate", None)
        if hasattr(result, "firing_rate")
        else result.firing_rates
    )


@pytest.mark.parametrize("func_name", COMPUTE_FUNCTION_NAMES)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
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
    """Every entry point computes a finite result on both backends.

    Tripwire if dispatch wiring routes a backend to a no-op or stub.
    The audit found that the previous spatial-only smoke would have
    missed a regression in ``compute_directional_rate(s)`` ignoring
    ``backend="jax"`` entirely, since the only positive smoke covered
    spatial. This parametrized version walks every compute function on
    both NumPy and JAX.

    Skipped per (function, backend) pair if JAX isn't installed.
    """
    if backend == "jax" and not _has_jax():
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
    firing_rate = _firing_rate_array(result)
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
