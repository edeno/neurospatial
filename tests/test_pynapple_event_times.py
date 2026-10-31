"""pynapple ``Ts``/``Tsd`` event times are read by their timestamps.

A pynapple ``Tsd`` holds its timestamps in ``.t`` and its data in ``.d``;
``np.asarray(tsd)`` returns the data. Every argument that takes event, reward
or peak times must read ``.t``, so passing ``Ts`` or ``Tsd`` gives exactly the
plain-array result even when the ``Tsd`` values (reward sizes, peak
amplitudes) differ from its times.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

nap = pytest.importorskip("pynapple")

pytestmark = pytest.mark.pynapple

EVENTS = np.array([10.0, 30.0, 55.0, 80.0])
TIMES = np.arange(0.0, 100.0, 0.05)
WINDOW = (-1.0, 1.5)


@pytest.fixture(scope="module")
def spikes() -> NDArray[np.float64]:
    return np.sort(np.random.default_rng(3).uniform(0.0, 100.0, 400))


def _as_ts(times: NDArray[np.float64]) -> Any:
    return nap.Ts(t=times)


def _as_tsd(times: NDArray[np.float64]) -> Any:
    # Values that are not the times, such as a reward size per event.
    return nap.Tsd(t=times, d=1000.0 - times)


def _as_tsdframe(times: NDArray[np.float64]) -> Any:
    return nap.TsdFrame(t=times, d=np.c_[-times, 2 * times], columns=["a", "b"])


WRAPPERS = {"Ts": _as_ts, "Tsd": _as_tsd, "TsdFrame": _as_tsdframe}


def _values(out: Any) -> Any:
    """Return comparable arrays from a function's output."""
    if isinstance(out, list):
        return [np.asarray(x) for x in out]
    return np.asarray(out)


def _events_cases(spikes: NDArray[np.float64]) -> dict[str, Callable[[Any], Any]]:
    from neurospatial.behavior import distance_to_reward
    from neurospatial.environment import Environment
    from neurospatial.events import (
        align_continuous_to_events,
        align_spikes_to_events,
        event_count_in_window,
        event_indicator,
        peri_event_histogram,
        population_peri_event_histogram,
        time_to_nearest_event,
    )
    from neurospatial.reactivation import assembly_event_response

    signal = np.sin(TIMES)
    positions = np.c_[np.linspace(0.0, 50.0, TIMES.size), np.zeros(TIMES.size)]
    env = Environment.from_samples(positions, bin_size=5.0)
    reward_positions = positions[np.searchsorted(TIMES, EVENTS)]
    return {
        "time_to_nearest_event": lambda ev: time_to_nearest_event(TIMES, ev),
        "event_count_in_window": lambda ev: event_count_in_window(TIMES, ev, WINDOW),
        "event_indicator": lambda ev: event_indicator(TIMES, ev, WINDOW),
        "peri_event_histogram": lambda ev: peri_event_histogram(
            spikes, ev, WINDOW, bin_size=0.1
        ).histogram,
        "population_peri_event_histogram": lambda ev: population_peri_event_histogram(
            [spikes, spikes[::2]], ev, WINDOW, bin_size=0.1
        ).histograms,
        "align_spikes_to_events": lambda ev: align_spikes_to_events(spikes, ev, WINDOW),
        "align_continuous_to_events": lambda ev: align_continuous_to_events(
            TIMES, signal, ev, WINDOW, dt=0.05
        ).values,
        "distance_to_reward": lambda ev: distance_to_reward(
            env,
            TIMES,
            positions,
            ev,
            reward_positions=reward_positions,
            metric="euclidean",
        ),
        "assembly_event_response": lambda ev: assembly_event_response(
            TIMES, signal, ev, WINDOW
        ).event_responses,
    }


CASES = [
    "time_to_nearest_event",
    "event_count_in_window",
    "event_indicator",
    "peri_event_histogram",
    "population_peri_event_histogram",
    "align_spikes_to_events",
    "align_continuous_to_events",
    "distance_to_reward",
    "assembly_event_response",
]


@pytest.mark.parametrize("wrapper", ["Ts", "Tsd"])
@pytest.mark.parametrize("case", CASES)
def test_pynapple_event_times_match_plain_array(
    case: str, wrapper: str, spikes: NDArray[np.float64]
) -> None:
    call = _events_cases(spikes)[case]
    expected = _values(call(EVENTS))
    actual = _values(call(WRAPPERS[wrapper](EVENTS)))
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_tsd_event_times_use_timestamps_not_values() -> None:
    """The reported reproduction: offsets are from the event times, not values."""
    from neurospatial.events import time_to_nearest_event

    events = nap.Tsd(t=[100.0, 200.0, 300.0, 400.0], d=[1.0, 2.0, 3.0, 4.0])

    offsets = time_to_nearest_event(np.array([100.5, 250.0]), events)

    np.testing.assert_allclose(offsets, [0.5, 50.0])


@pytest.mark.parametrize("wrapper", ["Ts", "Tsd", "TsdFrame"])
def test_pynapple_times_and_spikes_in_peri_event_histogram(
    wrapper: str, spikes: NDArray[np.float64]
) -> None:
    """A one-unit train given as Ts, Tsd or TsdFrame is read by its timestamps."""
    from neurospatial.events import peri_event_histogram

    expected = peri_event_histogram(spikes, EVENTS, WINDOW, bin_size=0.1).histogram
    actual = peri_event_histogram(
        WRAPPERS[wrapper](spikes), EVENTS, WINDOW, bin_size=0.1
    ).histogram

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("wrapper", ["Ts", "Tsd"])
def test_plot_peri_event_response_reads_event_timestamps(
    wrapper: str, spikes: NDArray[np.float64]
) -> None:
    import matplotlib.pyplot as plt

    from neurospatial.events import peri_event_histogram, plot_peri_event_response

    result = peri_event_histogram(spikes, EVENTS, WINDOW, bin_size=0.1)

    def raster_offsets(events: Any) -> list[NDArray[np.float64]]:
        axes = plot_peri_event_response(spikes, events, result)
        raster = axes[0]
        offsets = [np.asarray(c.get_segments()) for c in raster.collections]
        plt.close("all")
        return offsets

    expected = raster_offsets(EVENTS)
    actual = raster_offsets(WRAPPERS[wrapper](EVENTS))

    assert len(actual) == len(expected) > 0
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("wrapper", ["Ts", "Tsd"])
def test_theta_peak_times_accept_pynapple(
    wrapper: str, spikes: NDArray[np.float64]
) -> None:
    from neurospatial.replay import surrogate_spike_trains

    peaks = np.arange(0.0, 100.0, 0.125)

    def shuffled(theta_peak_times: Any) -> list[NDArray[np.float64]]:
        return surrogate_spike_trains(
            [spikes, spikes[::3]],
            kind="theta_phase",
            theta_peak_times=theta_peak_times,
            rng=7,
        )

    expected = shuffled(peaks)
    actual = shuffled(WRAPPERS[wrapper](peaks))

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("wrapper", ["Ts", "Tsd"])
def test_event_overlay_reads_event_timestamps(wrapper: str) -> None:
    from neurospatial.animation import EventOverlay

    overlay = EventOverlay(
        event_times=WRAPPERS[wrapper](EVENTS), event_positions=np.zeros((1, 2))
    )

    assert isinstance(overlay.event_times, dict)
    np.testing.assert_array_equal(overlay.event_times["event"], EVENTS)


def test_tsgroup_as_event_times_is_refused() -> None:
    from neurospatial.events import time_to_nearest_event

    group = nap.TsGroup({1: nap.Ts(t=EVENTS), 2: nap.Ts(t=EVENTS + 1.0)})

    with pytest.raises(ValueError, match="event_times is a spike group"):
        time_to_nearest_event(TIMES, group)
