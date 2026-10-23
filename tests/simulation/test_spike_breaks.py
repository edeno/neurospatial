"""Head-direction cells derive headings that never cross a declared break."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.simulation import HeadDirectionCellModel, generate_population_spikes
from neurospatial.simulation.spikes import _held_movement_heading

BREAK = 64
OFFSET = 1024.0


@pytest.fixture
def joined():
    """East for 2 s, then a second recording heading North, at 32 Hz."""
    times = np.arange(2 * BREAK) / 32
    east = np.c_[np.arange(BREAK) * 0.5, np.zeros(BREAK)]
    north = np.c_[np.full(BREAK, 40.0), np.arange(BREAK) * 0.5]
    shifted = times + np.where(np.arange(times.size) >= BREAK, OFFSET, 0.0)
    return times, shifted, np.r_[east, north], np.arange(times.size) == BREAK


def _spikes(times, positions, **kwargs):
    model = HeadDirectionCellModel(preferred_direction=np.pi / 4)
    return generate_population_spikes(
        [model], times, positions, seed=7, show_progress=False, **kwargs
    )[0]


def test_derived_heading_with_break_equals_long_gap(joined):
    times, shifted, positions, break_before = joined
    declared = _held_movement_heading(times, positions, break_before=break_before)
    np.testing.assert_array_equal(declared, _held_movement_heading(shifted, positions))
    np.testing.assert_allclose(declared[:BREAK], 0.0, atol=1e-12)
    np.testing.assert_allclose(declared[BREAK:], np.pi / 2, atol=1e-12)
    # Undeclared, the jump between recordings bends the smoothed velocity.
    undeclared = _held_movement_heading(times, positions)
    assert not np.allclose(
        undeclared[BREAK - 3 : BREAK + 3], declared[BREAK - 3 : BREAK + 3]
    )


def test_spikes_use_the_heading_derived_within_each_recording(joined):
    times, shifted, positions, break_before = joined
    expected = _spikes(
        times, positions, headings=_held_movement_heading(shifted, positions)
    )
    np.testing.assert_array_equal(
        _spikes(times, positions, break_before=break_before), expected
    )


def test_no_breaks_matches_omitting_break_before(joined):
    times, _, positions, _ = joined
    np.testing.assert_array_equal(
        _spikes(times, positions, break_before=np.zeros(times.size, dtype=bool)),
        _spikes(times, positions),
    )


def test_breaks_with_headings_raise(joined):
    times, _, positions, break_before = joined
    with pytest.raises(ValueError, match="both headings and break_before"):
        _spikes(
            times,
            positions,
            headings=np.zeros(times.size),
            break_before=break_before,
        )


@pytest.mark.parametrize("flags", [np.zeros(3, dtype=bool), np.zeros(128, dtype=int)])
def test_invalid_break_before_names_function(joined, flags):
    times, _, positions, _ = joined
    with pytest.raises(
        ValueError, match=r"generate_population_spikes\(\) needs break_before"
    ):
        _spikes(times, positions, break_before=flags)


@pytest.fixture
def still_then_north():
    """East for 2 s; then a recording still for 1 s at (40, 0) and North for 1 s."""
    n = 32
    times = np.arange(BREAK + 2 * n) / 32
    east = np.c_[np.arange(BREAK) * 0.5, np.zeros(BREAK)]
    still = np.tile([40.0, 0.0], (n, 1))
    north = np.c_[np.full(n, 40.0), np.arange(1, n + 1) * 0.5]
    return times, np.r_[east, still, north], np.arange(times.size) == BREAK


def test_held_heading_does_not_carry_over_a_break(still_then_north):
    times, positions, break_before = still_then_north
    held = _held_movement_heading(times, positions, break_before=break_before)
    np.testing.assert_array_equal(held[:BREAK], 0.0)
    # The still start of recording 2 takes its own first direction (North).
    np.testing.assert_array_equal(held[BREAK : BREAK + 10], np.pi / 2)
    assert np.isfinite(held).all()


def test_held_heading_equals_separate_recordings(still_then_north):
    times, positions, break_before = still_then_north
    held = _held_movement_heading(times, positions, break_before=break_before)
    first, second = slice(0, BREAK), slice(BREAK, None)
    np.testing.assert_array_equal(
        held,
        np.r_[
            _held_movement_heading(times[first], positions[first]),
            _held_movement_heading(times[second], positions[second]),
        ],
    )


def test_recording_without_movement_raises_naming_it(still_then_north):
    times, positions, break_before = still_then_north
    positions = positions.copy()
    positions[BREAK:] = [40.0, 0.0]
    with pytest.raises(ValueError, match=r"recording.*samples 64.*127"):
        _held_movement_heading(times, positions, break_before=break_before)
    with pytest.raises(ValueError, match="headings="):
        _spikes(times, positions, break_before=break_before)
