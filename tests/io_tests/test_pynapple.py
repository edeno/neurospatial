"""Tests for the pynapple ingress/egress shim (``neurospatial.io.pynapple``).

Two tiers:

- **Import-safety** (always run): the functions import even with pynapple
  absent, the module never imports pynapple at load time, and calling
  ``from_pynapple`` without pynapple raises a clear ``ImportError``.
- **Real pynapple flows** (``@pytest.mark.pynapple`` + ``skipif`` when pynapple
  is absent): a real ``TsGroup`` / ``Tsd`` round-trips through the adapters. In
  the default dev env these skip cleanly; the dedicated CI job installs the
  extra and runs them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.io import from_pynapple, to_pynapple

HAS_PYNAPPLE = importlib.util.find_spec("pynapple") is not None


# ---------------------------------------------------------------------------
# Import safety (pynapple absent-friendly)
# ---------------------------------------------------------------------------


def test_from_to_pynapple_importable() -> None:
    # Simply importing the names must work regardless of pynapple availability.
    assert callable(from_pynapple)
    assert callable(to_pynapple)


def test_pynapple_module_import_does_not_load_pynapple() -> None:
    """Importing the shim must not import pynapple at module load time."""
    code = (
        "import sys; import neurospatial.io.pynapple; "
        "assert 'pynapple' not in sys.modules, 'pynapple imported at module load'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    # If pynapple happens to be installed AND eagerly imported elsewhere this
    # would be a false failure, but the subprocess imports only the shim.
    if HAS_PYNAPPLE:
        pytest.skip("pynapple installed; lazy-import assertion is env-specific")
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(HAS_PYNAPPLE, reason="pynapple installed; ImportError not raised")
def test_from_pynapple_raises_clear_importerror_when_absent() -> None:
    with pytest.raises(ImportError, match=r"neurospatial\[pynapple\]"):
        from_pynapple(np.array([0.0, 1.0, 2.0]))


@pytest.mark.skipif(HAS_PYNAPPLE, reason="pynapple installed; ImportError not raised")
def test_to_pynapple_raises_clear_importerror_when_absent() -> None:
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ImportError, match=r"neurospatial\[pynapple\]"):
        to_pynapple(times, values)


# ---------------------------------------------------------------------------
# Real pynapple flows (extra-gated)
# ---------------------------------------------------------------------------


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra")
def test_from_pynapple_tsgroup_to_spatial_rates() -> None:
    import pynapple as nap

    from neurospatial.encoding import compute_spatial_rates

    rng = np.random.default_rng(3)
    positions = np.column_stack([np.linspace(0.0, 100.0, 400), rng.uniform(0, 40, 400)])
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0.0, 40.0, 400)
    spikes = [
        np.sort(rng.uniform(0.0, 40.0, 50)),
        np.sort(rng.uniform(0.0, 40.0, 40)),
    ]

    tsgroup = nap.TsGroup({7: nap.Ts(t=spikes[0]), 9: nap.Ts(t=spikes[1])})
    trains, unit_ids = from_pynapple(tsgroup)
    np.testing.assert_array_equal(np.asarray(unit_ids), np.array([7, 9]))

    # Plain-array reference (ids default to arange).
    from_arrays = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0)
    # A raw TsGroup flows in directly: `as_spike_trains_with_ids` extracts trains
    # by unit-id index (a TsGroup is a UserDict; iterating it would yield KEYS),
    # so the rates match the plain-array path AND the [7, 9] ids flow through.
    from_group = compute_spatial_rates(env, tsgroup, times, positions, bandwidth=5.0)

    np.testing.assert_array_equal(from_group.firing_rates, from_arrays.firing_rates)
    np.testing.assert_array_equal(np.asarray(from_group.unit_ids), np.array([7, 9]))


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra")
def test_raw_tsgroup_into_decode_session_matches_arrays() -> None:
    import pynapple as nap

    from neurospatial.decoding import decode_session

    rng = np.random.default_rng(4)
    positions = np.column_stack([np.linspace(0.0, 100.0, 400), rng.uniform(0, 40, 400)])
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0.0, 40.0, 400)
    spikes = [
        np.sort(rng.uniform(0.0, 40.0, 60)),
        np.sort(rng.uniform(0.0, 40.0, 45)),
        np.sort(rng.uniform(0.0, 40.0, 50)),
    ]

    tsgroup = nap.TsGroup(
        {11: nap.Ts(t=spikes[0]), 22: nap.Ts(t=spikes[1]), 33: nap.Ts(t=spikes[2])}
    )
    trains, _ = from_pynapple(tsgroup)

    # Decode results carry no unit axis, but the raw-TsGroup path must produce a
    # posterior byte-for-byte identical to the plain-array path (C1: trains are
    # extracted by unit-id index, not by iterating the UserDict's keys).
    from_arrays = decode_session(env, trains, times, positions, dt=0.5)
    from_group = decode_session(env, tsgroup, times, positions, dt=0.5)

    np.testing.assert_array_equal(from_group.posterior, from_arrays.posterior)


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra")
def test_from_pynapple_tsdframe_to_positions() -> None:
    import pynapple as nap

    times = np.linspace(0.0, 10.0, 100)
    positions = np.column_stack([times, times * 2.0])
    tsdframe = nap.TsdFrame(t=times, d=positions)

    out_times, out_positions = from_pynapple(tsdframe)
    np.testing.assert_allclose(out_times, times)
    np.testing.assert_allclose(out_positions, positions)


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra")
def test_from_pynapple_intervalset_to_start_end() -> None:
    import pynapple as nap

    start = np.array([0.0, 5.0, 10.0])
    end = np.array([2.0, 7.0, 12.0])
    epochs = nap.IntervalSet(start=start, end=end)

    out_start, out_end = from_pynapple(epochs)
    np.testing.assert_allclose(out_start, start)
    np.testing.assert_allclose(out_end, end)


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra")
def test_to_pynapple_roundtrips_a_track() -> None:
    times = np.linspace(0.0, 10.0, 100)
    positions = np.column_stack([np.sin(times), np.cos(times)])

    tsdframe = to_pynapple(times, positions)
    out_times, out_positions = from_pynapple(tsdframe)

    np.testing.assert_allclose(out_times, times)
    np.testing.assert_allclose(out_positions, positions)
