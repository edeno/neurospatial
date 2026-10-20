"""Environment trajectory methods treat a declared break like a long gap."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

BREAK = 200  # first sample of the second recording
OFFSET = 1024.0  # seconds; dyadic, so shifted times and differences are exact


@pytest.fixture(scope="module")
def arena() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 41.0), np.arange(0.0, 41.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)


@pytest.fixture(scope="module")
def joined():
    """Two 16 Hz recordings on one clock, joined with no time between them.

    The second recording starts in the bin where the first ended, so nothing
    but a declared break separates the two.
    """
    rng = np.random.default_rng(3)
    n = 2 * BREAK
    steps = rng.normal(0.0, 0.8, (n, 2))
    positions = np.clip(np.cumsum(steps, axis=0) + 8.0, 1.0, 39.0)
    positions[BREAK:] += positions[BREAK - 1] - positions[BREAK]
    positions = np.clip(positions, 1.0, 39.0)
    positions[[50, 300]] = np.nan  # missing samples
    positions[120] = [80.0, 80.0]  # outside the arena
    times = np.arange(n) / 16
    shifted = times + np.where(np.arange(n) >= BREAK, OFFSET, 0.0)
    return times, shifted, positions, np.arange(n) == BREAK


def _speed(times, positions):
    return np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1)] * 16


CALLS = {
    "occupancy[start]": lambda env, t, x, kw: env.occupancy(t, x, **kw),
    "occupancy[linear]": lambda env, t, x, kw: env.occupancy(
        t, x, time_allocation="linear", **kw
    ),
    "occupancy[min_speed]": lambda env, t, x, kw: env.occupancy(
        t, x, speed=_speed(t, x), min_speed=2.0, **kw
    ),
    "bin_sequence[dedup]": lambda env, t, x, kw: env.bin_sequence(t, x, **kw),
    "bin_sequence[drop outside]": lambda env, t, x, kw: env.bin_sequence(
        t, x, outside_value=None, **kw
    ),
    "bin_sequence_with_runs": lambda env, t, x, kw: env.bin_sequence_with_runs(
        t, x, **kw
    ),
    "bin_sequence_with_runs[drop outside]": lambda env, t, x, kw: (
        env.bin_sequence_with_runs(t, x, outside_value=None, **kw)
    ),
    "transitions[lag 1]": lambda env, t, x, kw: env.transitions(
        times=t, positions=x, normalize=False, allow_teleports=True, **kw
    ),
    "transitions[lag 3]": lambda env, t, x, kw: env.transitions(
        times=t, positions=x, lag=3, normalize=False, allow_teleports=True, **kw
    ),
}


def _plain(result):
    if hasattr(result, "toarray"):
        return result.toarray()
    if hasattr(result, "run_starts"):
        return np.r_[result.bins, result.run_starts, result.run_lengths]
    return result


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_declared_break_equals_long_gap(arena, joined, call):
    times, shifted, positions, break_before = joined
    declared = _plain(call(arena, times, positions, {"break_before": break_before}))
    np.testing.assert_array_equal(declared, _plain(call(arena, shifted, positions, {})))
    undeclared = _plain(call(arena, times, positions, {}))
    assert declared.shape != undeclared.shape or not np.array_equal(
        declared, undeclared
    )


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_no_breaks_matches_omitting_break_before(arena, joined, call):
    times, _, positions, _ = joined
    expected = _plain(call(arena, times, positions, {}))
    for flags in (np.zeros(times.size, dtype=bool), np.arange(times.size) == 0):
        np.testing.assert_array_equal(
            _plain(call(arena, times, positions, {"break_before": flags})), expected
        )


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
@pytest.mark.parametrize("bad", ["short", "integer"])
def test_invalid_break_before_names_method(arena, joined, call, bad):
    times, _, positions, _ = joined
    flags = (
        np.zeros(times.size - 1, dtype=bool)
        if bad == "short"
        else np.zeros(times.size, dtype=int)
    )
    with pytest.raises(ValueError, match=r"Environment\.\w+\(\) needs break_before"):
        call(arena, times, positions, {"break_before": flags})


def test_occupancy_loses_exactly_the_break_interval(arena, joined):
    times, _, positions, break_before = joined
    assert np.isfinite(positions[[BREAK - 1, BREAK]]).all()
    full = arena.occupancy(times, positions)
    split = arena.occupancy(times, positions, break_before=break_before)
    lost = full - split
    start_bin = arena.bin_at(positions[BREAK - 1 : BREAK])[0]
    assert lost[start_bin] == pytest.approx(times[BREAK] - times[BREAK - 1])
    assert np.count_nonzero(lost) == 1


def test_transitions_skip_the_pair_across_the_break(arena, joined):
    times, _, positions, break_before = joined
    kwargs = {"times": times, "positions": positions, "normalize": False}
    kwargs["allow_teleports"] = True
    full = arena.transitions(**kwargs).toarray()
    split = arena.transitions(break_before=break_before, **kwargs).toarray()
    source, target = arena.bin_at(positions[BREAK - 1 : BREAK + 1])
    expected = np.zeros_like(full)
    expected[source, target] = 1
    np.testing.assert_array_equal(full - split, expected)


def test_transitions_from_bins_apply_breaks(arena):
    bins = np.array([0, 1, 2, 3, 4])
    kwargs = {"normalize": False, "allow_teleports": True}
    split = arena.transitions(
        bins, break_before=np.array([False, False, True, False, False]), **kwargs
    ).toarray()
    expected = arena.transitions(bins, **kwargs).toarray()
    expected[1, 2] = 0
    np.testing.assert_array_equal(split, expected)


def test_model_transitions_refuse_breaks(arena):
    with pytest.raises(ValueError, match="'break_before' is only valid in empirical"):
        arena.transitions(method="random_walk", break_before=np.zeros(3, dtype=bool))


def test_per_sample_bins_drop_only_samples_a_break_isolates(arena):
    times = np.arange(5) / 16
    positions = np.c_[np.arange(5.0) * 4 + 2, np.full(5, 2.0)]
    bins = arena.bin_sequence(times, positions, dedup=False)
    # Samples 0 and 1 touch no interval that a break leaves usable.
    breaks = np.array([False, True, True, False, False])
    np.testing.assert_array_equal(
        arena.bin_sequence(times, positions, dedup=False, break_before=breaks),
        bins[2:],
    )
