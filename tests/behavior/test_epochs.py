"""Tests for array-native epoch selection (``neurospatial.behavior.epochs``).

Covers the three public entry points -- :func:`in_epochs`, :func:`restrict`,
and :func:`restrict_spike_trains` -- plus the private ``_as_intervals``
normalizer. The headline invariant (the DoD) is **array-vs-IntervalSet
parity**: restricting by ``(start, end)`` arrays equals restricting by an
``IntervalSet``-like object. That parity is tested two ways:

- with a **duck-typed fake** exposing ``.start`` / ``.end`` (runs in the
  default env, no pynapple), and
- with a **real pynapple ``IntervalSet``** (``@pytest.mark.pynapple`` +
  ``skipif`` -- skips cleanly when pynapple is absent).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior import in_epochs, restrict, restrict_spike_trains
from neurospatial.behavior.epochs import _as_intervals
from neurospatial.encoding import SpikeTrains, compute_spatial_rates

HAS_PYNAPPLE = importlib.util.find_spec("pynapple") is not None


@dataclass
class FakeIntervalSet:
    """Minimal duck-typed ``IntervalSet`` (``.start`` / ``.end`` arrays only)."""

    start: np.ndarray
    end: np.ndarray


# ---------------------------------------------------------------------------
# _as_intervals: accepts every documented form
# ---------------------------------------------------------------------------


def test_as_intervals_scalar_tuple() -> None:
    """A 2-tuple of scalars -> one interval."""
    out = _as_intervals((1.0, 5.0))
    np.testing.assert_array_equal(out, np.array([[1.0, 5.0]]))
    assert out.shape == (1, 2)


def test_as_intervals_two_arrays() -> None:
    """Two 1-D (starts, ends) arrays of length != 2 -> parallel n intervals.

    Length-2 arrays are the *ambiguous* case (see
    ``test_as_intervals_ambiguous_nested_pair_raises``); three intervals is
    unambiguous, so this exercises the parallel-arrays path.
    """
    out = _as_intervals((np.array([0.0, 10.0, 20.0]), np.array([5.0, 15.0, 25.0])))
    np.testing.assert_array_equal(
        out, np.array([[0.0, 5.0], [10.0, 15.0], [20.0, 25.0]])
    )


def test_as_intervals_n_by_2_array() -> None:
    """A single (n, 2) array passes through unchanged."""
    arr = np.array([[0.0, 5.0], [10.0, 15.0], [20.0, 25.0]])
    out = _as_intervals(arr)
    np.testing.assert_array_equal(out, arr)


def test_as_intervals_n_by_2_array_two_rows() -> None:
    """A 2-row (n, 2) NumPy array reads as two interval rows, not parallel arrays.

    ``np.array([[0, 5], [10, 15]])`` is unambiguous (it is an ndarray, not a
    bare nested list) and must read as the rows ``(0, 5)`` and ``(10, 15)``.
    """
    out = _as_intervals(np.array([[0.0, 5.0], [10.0, 15.0]]))
    np.testing.assert_array_equal(out, np.array([[0.0, 5.0], [10.0, 15.0]]))


@pytest.mark.parametrize(
    "ambiguous",
    [
        [[0.0, 5.0], [10.0, 15.0]],
        ([0.0, 5.0], [10.0, 15.0]),
        (np.array([0.0, 10.0]), np.array([5.0, 15.0])),
    ],
)
def test_as_intervals_ambiguous_nested_pair_raises(ambiguous) -> None:
    """A length-2 pair of length-2 sequences is irreducibly ambiguous -> raise.

    Could mean two ``(start, end)`` rows or two parallel ``(starts, ends)``
    arrays; force the user to disambiguate with an ``(n, 2)`` array.
    """
    with pytest.raises(ValueError, match="Ambiguous"):
        _as_intervals(ambiguous)


def test_as_intervals_intervalset_like() -> None:
    """A duck-typed IntervalSet (.start/.end) -> column-stacked intervals."""
    iset = FakeIntervalSet(start=np.array([0.0, 10.0]), end=np.array([5.0, 15.0]))
    out = _as_intervals(iset)
    np.testing.assert_array_equal(out, np.array([[0.0, 5.0], [10.0, 15.0]]))


@pytest.mark.parametrize(
    "empty",
    [
        np.empty((0, 2)),
        ([], []),
        [],
    ],
)
def test_as_intervals_empty_forms(empty) -> None:
    """Empty epochs (0 intervals) normalize to shape (0, 2)."""
    out = _as_intervals(empty)
    assert out.shape == (0, 2)


# ---------------------------------------------------------------------------
# in_epochs: inclusive-endpoint mask
# ---------------------------------------------------------------------------


def test_in_epochs_single_interval_inclusive() -> None:
    """Endpoints included under the default closed='both'."""
    t = np.array([-1.0, 0.0, 2.5, 5.0, 6.0])
    mask = in_epochs(t, (0.0, 5.0))
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_in_epochs_multiple_intervals_union() -> None:
    """Union across multiple intervals."""
    t = np.array([0.5, 3.0, 7.0, 12.0, 20.0])
    mask = in_epochs(t, np.array([[0.0, 5.0], [10.0, 15.0]]))
    np.testing.assert_array_equal(mask, [True, True, False, True, False])


def test_in_epochs_overlapping_intervals_union() -> None:
    """Overlapping intervals behave as their union."""
    t = np.array([0.0, 3.0, 6.0, 9.0, 11.0])
    intervals = np.array([[0.0, 6.0], [4.0, 10.0]])
    mask = in_epochs(t, intervals)
    np.testing.assert_array_equal(mask, [True, True, True, True, False])


def test_in_epochs_empty_all_false() -> None:
    """Empty epochs select nothing."""
    t = np.array([0.0, 1.0, 2.0])
    mask = in_epochs(t, np.empty((0, 2)))
    np.testing.assert_array_equal(mask, [False, False, False])
    assert mask.dtype == np.bool_


def test_in_epochs_closed_left() -> None:
    """closed='left' includes start, excludes end."""
    t = np.array([0.0, 2.5, 5.0])
    mask = in_epochs(t, (0.0, 5.0), closed="left")
    np.testing.assert_array_equal(mask, [True, True, False])


def test_in_epochs_closed_right() -> None:
    """closed='right' excludes start, includes end."""
    t = np.array([0.0, 2.5, 5.0])
    mask = in_epochs(t, (0.0, 5.0), closed="right")
    np.testing.assert_array_equal(mask, [False, True, True])


def test_in_epochs_closed_neither() -> None:
    """closed='neither' excludes both endpoints."""
    t = np.array([0.0, 2.5, 5.0])
    mask = in_epochs(t, (0.0, 5.0), closed="neither")
    np.testing.assert_array_equal(mask, [False, True, False])


def test_in_epochs_bad_closed_raises() -> None:
    """An unknown closed value raises ValueError."""
    with pytest.raises(ValueError, match="closed"):
        in_epochs(np.array([1.0]), (0.0, 5.0), closed="inner")


def test_in_epochs_bad_closed_raises_even_on_empty_epochs() -> None:
    """A bad ``closed`` raises even when epochs are empty (validated up front)."""
    with pytest.raises(ValueError, match="closed"):
        in_epochs(np.array([1.0]), np.empty((0, 2)), closed="garbage")


# ---------------------------------------------------------------------------
# restrict: aligned multi-array slicing
# ---------------------------------------------------------------------------


def test_restrict_times_and_positions_aligned() -> None:
    """times and positions are sliced by the SAME mask, order preserved."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    positions = np.column_stack([times * 10.0, times * -1.0])
    t_kept, pos_kept = restrict(times, positions, epochs=(1.0, 3.0))

    expected_mask = (times >= 1.0) & (times <= 3.0)
    np.testing.assert_array_equal(t_kept, times[expected_mask])
    np.testing.assert_array_equal(pos_kept, positions[expected_mask])
    # Order preserved (strictly increasing subset).
    assert np.all(np.diff(t_kept) > 0)


def test_restrict_single_array_returns_in_epoch_spikes() -> None:
    """Single-array usage restricts an event-time array by its own times."""
    spike_train = np.array([0.1, 1.5, 2.9, 4.2, 5.8])
    kept = restrict(spike_train, epochs=(1.0, 3.0))
    np.testing.assert_array_equal(kept, np.array([1.5, 2.9]))
    # No extra arrays -> a bare array (not a 1-tuple).
    assert isinstance(kept, np.ndarray)


def test_restrict_multiple_arrays_returns_tuple() -> None:
    """With extra arrays, restrict returns a tuple (times', *arrays')."""
    times = np.arange(5.0)
    a = times * 2.0
    b = times * 3.0
    out = restrict(times, a, b, epochs=(1.0, 3.0))
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_restrict_length_mismatch_raises() -> None:
    """An aligned-length mismatch raises a clear error naming the array index."""
    times = np.arange(6.0)
    positions = np.zeros((5, 2))  # wrong length
    with pytest.raises(ValueError, match="len"):
        restrict(times, positions, epochs=(1.0, 3.0))


# ---------------------------------------------------------------------------
# Array vs IntervalSet parity (the Definition of Done)
# ---------------------------------------------------------------------------


def test_restrict_array_vs_fake_intervalset_parity() -> None:
    """restrict by (starts, ends) equals restrict by a duck-typed IntervalSet."""
    times = np.linspace(0.0, 20.0, 100)
    positions = np.column_stack([np.sin(times), np.cos(times)])
    starts = np.array([2.0, 12.0])
    ends = np.array([6.0, 16.0])

    # Two length-2 arrays as (starts, ends) is now ambiguous, so pass the
    # unambiguous (n, 2) interval-rows form for the array side.
    t_arr, pos_arr = restrict(times, positions, epochs=np.column_stack([starts, ends]))
    iset = FakeIntervalSet(start=starts, end=ends)
    t_iset, pos_iset = restrict(times, positions, epochs=iset)

    np.testing.assert_array_equal(t_arr, t_iset)
    np.testing.assert_array_equal(pos_arr, pos_iset)


@pytest.mark.pynapple
@pytest.mark.skipif(not HAS_PYNAPPLE, reason="requires the optional pynapple extra")
def test_restrict_array_vs_real_intervalset_parity() -> None:
    """restrict by arrays equals restrict by a real pynapple IntervalSet."""
    import pynapple as nap

    times = np.linspace(0.0, 20.0, 100)
    positions = np.column_stack([np.sin(times), np.cos(times)])
    starts = np.array([2.0, 12.0])
    ends = np.array([6.0, 16.0])

    # Two length-2 arrays as (starts, ends) is now ambiguous, so pass the
    # unambiguous (n, 2) interval-rows form for the array side.
    t_arr, pos_arr = restrict(times, positions, epochs=np.column_stack([starts, ends]))
    iset = nap.IntervalSet(start=starts, end=ends)
    t_iset, pos_iset = restrict(times, positions, epochs=iset)

    np.testing.assert_array_equal(t_arr, t_iset)
    np.testing.assert_array_equal(pos_arr, pos_iset)


# ---------------------------------------------------------------------------
# restrict_spike_trains: ragged, per-train masking
# ---------------------------------------------------------------------------


def test_restrict_spike_trains_list() -> None:
    """Each train is masked by its own timestamps; order/count preserved."""
    trains = [
        np.array([0.1, 1.5, 2.9, 4.2]),
        np.array([0.5, 3.0, 3.5, 6.0]),
    ]
    epochs = (1.0, 3.5)
    out = restrict_spike_trains(trains, epochs)
    assert isinstance(out, list)
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], np.array([1.5, 2.9]))
    np.testing.assert_array_equal(out[1], np.array([3.0, 3.5]))


def test_restrict_spike_trains_accepts_spiketrains_container() -> None:
    """A SpikeTrains container iterates to its per-unit train arrays."""
    st = SpikeTrains(
        [np.array([0.1, 1.5, 2.9]), np.array([0.5, 3.0, 6.0])],
        unit_ids=np.array([7, 9]),
    )
    out = restrict_spike_trains(st, (1.0, 3.5))
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], np.array([1.5, 2.9]))
    np.testing.assert_array_equal(out[1], np.array([3.0]))


def test_restrict_spike_trains_empty_epochs_all_empty() -> None:
    """Empty epochs -> every returned train is empty (but present)."""
    trains = [np.array([0.1, 1.5]), np.array([0.5, 3.0])]
    out = restrict_spike_trains(trains, np.empty((0, 2)))
    assert len(out) == 2
    assert all(t.size == 0 for t in out)


# ---------------------------------------------------------------------------
# Downstream honors restriction
# ---------------------------------------------------------------------------


def test_compute_spatial_rates_honors_restriction() -> None:
    """Restricted encoding differs from unrestricted and equals a hand mask."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 100.0, 2000)
    # Animal sweeps left half early, right half late, so an epoch that keeps
    # only the late window changes which bins are occupied -> rates differ.
    x = np.where(times < 50.0, 20.0, 80.0) + rng.normal(0.0, 2.0, times.shape)
    y = 50.0 + rng.normal(0.0, 2.0, times.shape)
    positions = np.column_stack([x, y])
    env = Environment.from_samples(positions, bin_size=5.0)

    trains = [
        np.sort(rng.uniform(0.0, 100.0, 300)),
        np.sort(rng.uniform(0.0, 100.0, 200)),
    ]
    epochs = (55.0, 100.0)

    # Unrestricted vs restricted.
    unrestricted = compute_spatial_rates(env, trains, times, positions)
    restricted = compute_spatial_rates(
        env,
        restrict_spike_trains(trains, epochs),
        *restrict(times, positions, epochs=epochs),
    )

    # Restriction changes the result.
    assert not np.allclose(
        np.nan_to_num(unrestricted.firing_rates),
        np.nan_to_num(restricted.firing_rates),
    )

    # Restricted equals a hand-masked reference.
    mask = (times >= 55.0) & (times <= 100.0)
    ref_trains = [tr[(tr >= 55.0) & (tr <= 100.0)] for tr in trains]
    reference = compute_spatial_rates(env, ref_trains, times[mask], positions[mask])
    np.testing.assert_array_equal(
        np.nan_to_num(restricted.firing_rates),
        np.nan_to_num(reference.firing_rates),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_as_intervals_start_after_end_raises() -> None:
    """start > end raises, naming the offending interval."""
    with pytest.raises(ValueError, match="start <= end"):
        _as_intervals((5.0, 1.0))


def test_as_intervals_nonfinite_raises() -> None:
    """A non-finite endpoint raises."""
    with pytest.raises(ValueError, match="finite"):
        _as_intervals((0.0, np.inf))


def test_as_intervals_bad_shape_raises() -> None:
    """An (n, 3) array is not a valid interval array."""
    with pytest.raises(ValueError, match=r"shape"):
        _as_intervals(np.zeros((4, 3)))


def test_restrict_top_level_export() -> None:
    """restrict is reachable from the top-level neurospatial namespace."""
    import neurospatial

    assert neurospatial.restrict is restrict
