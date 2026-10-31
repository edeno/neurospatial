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


def test_to_pynapple_rejects_unsorted_times() -> None:
    # pynapple sorts timestamps itself; validation must run (and fail) before
    # pynapple is ever imported, so this needs no pynapple.
    with pytest.raises(ValueError, match=r"indices 1 and 2") as excinfo:
        to_pynapple(np.array([0.0, 2.0, 1.0]), np.array([10.0, 20.0, 30.0]))
    assert "\nFix:" in str(excinfo.value)


def test_to_pynapple_rejects_duplicate_times() -> None:
    with pytest.raises(ValueError, match=r"indices 0 and 1"):
        to_pynapple(np.array([1.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_to_pynapple_rejects_non_finite_times(bad: float) -> None:
    with pytest.raises(ValueError, match="finite") as excinfo:
        to_pynapple(np.array([0.0, bad, 2.0]), np.array([10.0, 20.0, 30.0]))
    assert "\nFix:" in str(excinfo.value)


def test_to_pynapple_rejects_wrong_column_count() -> None:
    times = np.array([0.0, 1.0, 2.0])
    values = np.zeros((3, 2))
    with pytest.raises(ValueError, match=r"3 labels.*2 value columns") as excinfo:
        to_pynapple(times, values, columns=["x", "y", "z"])
    assert "\nFix:" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Real pynapple flows (extra-gated)
# ---------------------------------------------------------------------------


needs_pynapple = [
    pytest.mark.pynapple,
    pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the pynapple extra"),
]

# Two recording epochs with a gap between them, shared by the ingress tests.
GAPPED = np.array([[0.0, 10.0], [10.5, 20.0]])


def _half_open(closed_rows) -> np.ndarray:
    """The half-open ``[start, stop)`` rows holding the same floats as ``[start, end]``."""
    rows = np.array(closed_rows, dtype=np.float64)
    rows[:, 1] = np.nextafter(rows[:, 1], np.inf)
    return rows


HALF_OPEN_GAPPED = _half_open(GAPPED)


def _gapped_support():
    import pynapple as nap

    return nap.IntervalSet(start=GAPPED[:, 0], end=GAPPED[:, 1])


def _pynapple_test(func):
    """Mark a test as needing the pynapple extra and skip it when absent."""
    for mark in needs_pynapple:
        func = mark(func)
    return func


# ---------------------------------------------------------------------------
# Ingress: pynapple -> plain arrays
# ---------------------------------------------------------------------------


@_pynapple_test
def test_from_pynapple_ts_returns_spike_times_and_epochs() -> None:
    import pynapple as nap

    spikes = np.array([0.1, 0.5, 9.9, 12.0, 19.5])
    ts = nap.Ts(t=spikes, time_support=_gapped_support())

    spike_times, epochs = from_pynapple(ts)

    np.testing.assert_array_equal(spike_times, spikes)
    np.testing.assert_array_equal(epochs, HALF_OPEN_GAPPED)


@_pynapple_test
def test_from_pynapple_tsd_returns_times_values_and_gapped_epochs() -> None:
    import pynapple as nap

    times = np.round(np.arange(201) * 0.1, 10)
    speed = np.arange(201.0)
    tsd = nap.Tsd(t=times, d=speed, time_support=_gapped_support())

    out_times, out_values, epochs = from_pynapple(tsd)

    # pynapple drops the samples inside the gap; the gap itself survives only
    # as the epochs, which must come back exactly.
    kept = ~((times > 10.0) & (times < 10.5))
    np.testing.assert_array_equal(out_times, times[kept])
    np.testing.assert_array_equal(out_values, speed[kept])
    np.testing.assert_array_equal(epochs, HALF_OPEN_GAPPED)


@_pynapple_test
def test_from_pynapple_gapped_tsdframe_epochs_exclude_the_gap_from_occupancy() -> None:
    """The returned epochs keep the gap out of occupancy."""
    import pynapple as nap

    times = np.round(np.arange(201) * 0.1, 10)
    positions = np.column_stack(
        [np.linspace(0.0, 100.0, 201), np.linspace(0.0, 40.0, 201)]
    )
    tsdframe = nap.TsdFrame(
        t=times, d=positions, columns=["x", "y"], time_support=_gapped_support()
    )
    env = Environment.from_samples(positions, bin_size=10.0)

    out_times, out_positions, epochs = from_pynapple(tsdframe)

    # Without the epochs the 0.5 s gap (shorter than max_gap) is counted as
    # time spent at the last sample before it.
    assert env.occupancy(out_times, out_positions).sum() == pytest.approx(20.0)
    # The epochs cover 10.0 s + 9.5 s of recording.
    assert env.occupancy(
        out_times, out_positions, epochs=epochs
    ).sum() == pytest.approx(19.5)


@_pynapple_test
def test_from_pynapple_default_tsgroup_support_keeps_the_last_spike() -> None:
    """A spike exactly at a pynapple epoch end is inside the returned window."""
    import pynapple as nap

    from neurospatial.behavior import restrict_spike_trains

    spikes = {1: np.array([1.0, 4.0, 7.5]), 2: np.array([2.0, 9.0])}
    # The default support runs from the first spike to the last, so 9.0 sits
    # exactly on its end.
    tsgroup = nap.TsGroup({uid: nap.Ts(t=t) for uid, t in spikes.items()})
    assert tsgroup.time_support.values.tolist() == [[1.0, 9.0]]

    trains, _, recorded = from_pynapple(tsgroup)

    # Plain half-open rows with the same numbers drop the spike at 9.0 ...
    plain = restrict_spike_trains(trains, np.array([[1.0, 9.0]]))
    assert [len(t) for t in plain] == [3, 1]
    # ... the converted epochs and a directly passed IntervalSet keep it,
    # as pynapple does.
    for window in (recorded, tsgroup.time_support):
        kept = restrict_spike_trains(trains, window)
        for train, uid in zip(kept, [1, 2], strict=True):
            np.testing.assert_array_equal(train, spikes[uid])
    assert len(tsgroup.restrict(tsgroup.time_support)[2]) == 2


@_pynapple_test
def test_pynapple_round_trip_preserves_epoch_membership() -> None:
    import pynapple as nap

    # Samples sit exactly on both ends of each pynapple interval.
    times = np.array([0.0, 1.0, 2.0, 10.5, 12.5])
    tsd = nap.Tsd(
        t=times,
        d=np.arange(5.0),
        time_support=nap.IntervalSet(start=[0.0, 10.5], end=[2.0, 12.5]),
    )

    out_times, out_values, epochs = from_pynapple(tsd)
    back = to_pynapple(out_times, out_values, epochs=epochs)

    np.testing.assert_array_equal(back.t, times)
    np.testing.assert_array_equal(back.values, np.arange(5.0))
    np.testing.assert_array_equal(back.time_support.values, tsd.time_support.values)
    np.testing.assert_array_equal(from_pynapple(back)[2], epochs)


@_pynapple_test
def test_from_pynapple_tsdframe_values_keep_column_order() -> None:
    import pynapple as nap

    times = np.linspace(0.0, 10.0, 11)
    data = np.column_stack([times, 2.0 * times, 3.0 * times])
    tsdframe = nap.TsdFrame(t=times, d=data, columns=["x", "y", "speed"])

    out_times, out_values, epochs = from_pynapple(tsdframe)

    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_values, data)
    np.testing.assert_array_equal(epochs, _half_open([[0.0, 10.0]]))


@_pynapple_test
@pytest.mark.parametrize("columns", [["y", "x"], ["Y", "X"], ["z", "x", "y"]])
def test_from_pynapple_refuses_permuted_spatial_columns(columns) -> None:
    import pynapple as nap

    times = np.linspace(0.0, 1.0, 5)
    tsdframe = nap.TsdFrame(t=times, d=np.zeros((5, len(columns))), columns=columns)

    with pytest.raises(ValueError, match="not in x, y, z order") as excinfo:
        from_pynapple(tsdframe)
    assert "tsdframe.loc[" in str(excinfo.value)


@_pynapple_test
def test_from_pynapple_reordered_frame_returns_x_then_y() -> None:
    import pynapple as nap

    times = np.linspace(0.0, 1.0, 5)
    x = np.arange(5.0)
    y = 10.0 + np.arange(5.0)
    tsdframe = nap.TsdFrame(t=times, d=np.column_stack([y, x]), columns=["y", "x"])

    _, positions, _ = from_pynapple(tsdframe.loc[["x", "y"]])

    np.testing.assert_array_equal(positions, np.column_stack([x, y]))


@_pynapple_test
def test_from_pynapple_intervalset_returns_epoch_rows() -> None:
    epochs = from_pynapple(_gapped_support())

    assert epochs.shape == (2, 2)
    np.testing.assert_array_equal(epochs, HALF_OPEN_GAPPED)


@_pynapple_test
def test_from_pynapple_tsgroup_returns_trains_ids_and_epochs() -> None:
    import pynapple as nap

    spikes = {7: np.array([0.2, 3.0, 11.0]), 9: np.array([1.0, 19.0])}
    tsgroup = nap.TsGroup(
        {uid: nap.Ts(t=t) for uid, t in spikes.items()},
        time_support=_gapped_support(),
    )

    trains, unit_ids, epochs = from_pynapple(tsgroup)

    np.testing.assert_array_equal(np.asarray(unit_ids), [7, 9])
    for train, uid in zip(trains, [7, 9], strict=True):
        np.testing.assert_array_equal(train, spikes[uid])
    np.testing.assert_array_equal(epochs, HALF_OPEN_GAPPED)


@_pynapple_test
def test_from_pynapple_returns_copies_not_pynapple_buffers() -> None:
    import pynapple as nap

    times = np.linspace(0.0, 1.0, 5)
    tsd = nap.Tsd(t=times, d=np.arange(5.0))
    out_times, out_values, epochs = from_pynapple(tsd)
    assert not np.shares_memory(out_times, tsd.t)
    assert not np.shares_memory(out_values, tsd.values)
    assert not np.shares_memory(epochs, tsd.time_support.values)

    tsgroup = nap.TsGroup({1: nap.Ts(t=times)})
    trains, _, _ = from_pynapple(tsgroup)
    assert not np.shares_memory(trains[0], tsgroup[1].t)


@_pynapple_test
@pytest.mark.parametrize(
    "make",
    [
        lambda nap: nap.TsdTensor(t=np.arange(3.0), d=np.zeros((3, 2, 2))),
        lambda nap: np.arange(3.0),
        lambda nap: {"t": np.arange(3.0)},
    ],
    ids=["TsdTensor", "ndarray", "dict"],
)
def test_from_pynapple_unsupported_type_raises_typeerror(make) -> None:
    import pynapple as nap

    with pytest.raises(TypeError, match="Ts, Tsd, TsdFrame, TsGroup or IntervalSet"):
        from_pynapple(make(nap))


# ---------------------------------------------------------------------------
# Raw pynapple objects flowing straight into analyses
# ---------------------------------------------------------------------------


@_pynapple_test
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
    trains, unit_ids, _ = from_pynapple(tsgroup)
    np.testing.assert_array_equal(np.asarray(unit_ids), np.array([7, 9]))

    # Plain-array reference (ids default to arange).
    from_arrays = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0)
    # A raw TsGroup flows in directly: trains are read by unit id (iterating
    # a TsGroup yields its keys), so the rates match the plain-array path
    # and the [7, 9] ids flow through.
    from_group = compute_spatial_rates(env, tsgroup, times, positions, bandwidth=5.0)

    np.testing.assert_array_equal(from_group.firing_rates, from_arrays.firing_rates)
    np.testing.assert_array_equal(np.asarray(from_group.unit_ids), np.array([7, 9]))


@_pynapple_test
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
    trains, _, _ = from_pynapple(tsgroup)

    # Decode results carry no unit axis, but the raw-TsGroup path must produce
    # a posterior identical to the plain-array path.
    from_arrays = decode_session(env, trains, times, positions, dt=0.5)
    from_group = decode_session(env, tsgroup, times, positions, dt=0.5)

    np.testing.assert_array_equal(from_group.posterior, from_arrays.posterior)


# ---------------------------------------------------------------------------
# Egress: arrays / decode results -> pynapple
# ---------------------------------------------------------------------------


@_pynapple_test
def test_to_pynapple_roundtrips_a_track() -> None:
    times = np.linspace(0.0, 10.0, 100)
    positions = np.column_stack([np.sin(times), np.cos(times)])

    tsdframe = to_pynapple(times, positions)
    out_times, out_positions, epochs = from_pynapple(tsdframe)

    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_positions, positions)
    np.testing.assert_array_equal(epochs, _half_open([[0.0, 10.0]]))


@_pynapple_test
def test_to_pynapple_epochs_become_the_time_support() -> None:
    times = np.array([0.5, 1.5, 11.0, 12.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    tsd = to_pynapple(times, values, epochs=[[0.0, 2.0], [10.5, 12.5]])

    # pynapple's closed intervals end one float below the half-open stops.
    np.testing.assert_array_equal(
        _half_open(tsd.time_support.values), [[0.0, 2.0], [10.5, 12.5]]
    )
    out_times, out_values, epochs = from_pynapple(tsd)
    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_values, values)
    np.testing.assert_array_equal(epochs, [[0.0, 2.0], [10.5, 12.5]])


def test_to_pynapple_refuses_a_time_at_a_half_open_stop() -> None:
    with pytest.raises(ValueError, match=r"1 of 2 timestamp\(s\) lie outside"):
        to_pynapple(np.array([0.5, 2.0]), np.array([1.0, 2.0]), epochs=[[0.0, 2.0]])


def test_to_pynapple_refuses_times_outside_epochs() -> None:
    # pynapple would silently drop the sample at 5.0; validation runs before
    # pynapple is imported.
    with pytest.raises(
        ValueError, match=r"1 of 3 timestamp\(s\) lie outside"
    ) as excinfo:
        to_pynapple(
            np.array([0.5, 5.0, 11.0]),
            np.array([1.0, 2.0, 3.0]),
            epochs=[[0.0, 2.0], [10.0, 12.0]],
        )
    assert "\nFix:" in str(excinfo.value)


def test_to_pynapple_decode_result_without_times_raises() -> None:
    from neurospatial.decoding import DecodingResult

    env = Environment.from_samples(np.array([[0.0, 0.0], [10.0, 10.0]]), bin_size=5.0)
    result = DecodingResult(
        posterior=np.full((3, env.n_bins), 1.0 / env.n_bins), env=env
    )

    with pytest.raises(ValueError, match="decode result has no `times`"):
        to_pynapple(result)


@_pynapple_test
def test_to_pynapple_decode_result_keeps_multi_window_spike_window() -> None:
    from neurospatial.decoding import decode_session

    rng = np.random.default_rng(4)
    times = np.linspace(0.0, 40.0, 401)
    positions = np.column_stack(
        [np.linspace(0.0, 100.0, 401), rng.uniform(0.0, 40.0, 401)]
    )
    env = Environment.from_samples(positions, bin_size=10.0)
    spikes = [np.sort(rng.uniform(0.0, 40.0, 80)) for _ in range(3)]
    windows = np.array([[0.0, 10.0], [25.0, 35.0]])
    result = decode_session(env, spikes, times, positions, dt=0.5, spike_window=windows)

    tsdframe = to_pynapple(result)

    # One interval per spike window: the 10-25 s gap is not decoded time.
    np.testing.assert_array_equal(_half_open(tsdframe.time_support.values), windows)
    np.testing.assert_array_equal(tsdframe.t, result.times)
    np.testing.assert_array_equal(tsdframe.values, result.map_position)


@_pynapple_test
def test_to_pynapple_non_result_without_values_raises_typeerror() -> None:
    # values=None + a `times` that is not a decode result (no `.times` /
    # `.map_position`) must raise a clear TypeError naming the expected inputs,
    # not a bare AttributeError.
    with pytest.raises(TypeError, match="decode result"):
        to_pynapple(object(), values=None)


@_pynapple_test
def test_to_pynapple_bad_shapes_raise_valueerror() -> None:
    # Invalid shapes/lengths must raise a neurospatial ValueError at the adapter
    # boundary, not a raw pynapple AssertionError from inside Tsd/TsdFrame.
    times = np.linspace(0.0, 1.0, 5)

    # 2-D times.
    with pytest.raises(ValueError, match=r"`times` must be 1-D"):
        to_pynapple(times.reshape(5, 1), np.zeros(5))

    # 3-D values.
    with pytest.raises(ValueError, match=r"`values` must be 1-D or 2-D"):
        to_pynapple(times, np.zeros((5, 2, 2)))

    # Length mismatch.
    with pytest.raises(ValueError, match=r"same length"):
        to_pynapple(times, np.zeros(4))
