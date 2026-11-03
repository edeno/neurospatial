"""The spike-drop warning counts only spikes inside the analysed windows.

Head direction is often tracked only during wake while the spikes cover
the whole recording, sleep included. With ``epochs=wake`` the sleep spikes
are excluded by the caller's own choice, not lost, so they must not count
as spikes outside the tracked time. Spike times in milliseconds still warn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import compute_spatial_rates
from neurospatial.decoding import decode_session
from neurospatial.encoding import compute_directional_rate, compute_directional_rates
from neurospatial.simulation import open_field_session

WAKE = (0.0, 100.0)


@pytest.fixture(scope="module")
def wake_tracking():
    """Headings tracked during 100 s of wake; spikes over 1000 s of recording.

    About 90% of the spikes fall in the untracked sleep after wake.
    """
    heading_rng = np.random.default_rng(0)
    times = np.arange(*WAKE, 0.02)
    headings = np.angle(np.exp(1j * np.cumsum(heading_rng.normal(0, 0.1, times.size))))
    spike_rng = np.random.default_rng(1)
    spikes = np.sort(spike_rng.uniform(0.0, 1000.0, 5000))
    return times, headings, spikes


@pytest.fixture(scope="module")
def session_100s():
    """A seeded 100 s open-field session with two place cells."""
    return open_field_session(duration=100.0, n_place_cells=2, rng=0)


@pytest.fixture(scope="module")
def session_60s():
    return open_field_session(duration=60.0, n_place_cells=4, rng=0)


@pytest.fixture(scope="module")
def session_600s():
    return open_field_session(duration=600.0, n_place_cells=1, rng=0)


def _drop_warnings(caught):
    return [w for w in caught if "fell outside the" in str(w.message)]


def test_spike_drop_ignores_spikes_outside_epochs(wake_tracking):
    times, headings, spikes = wake_tracking
    assert np.mean(spikes >= times[-1]) > 0.85  # the sleep spikes
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_directional_rates([spikes], times, headings, epochs=WAKE)
    assert _drop_warnings(caught) == []
    # Without epochs the same spikes are lost to the analysis, so it warns.
    with pytest.warns(UserWarning, match=r"\(90%\) fell outside the position time"):
        compute_directional_rates([spikes], times, headings)


def test_spike_drop_in_milliseconds_still_warns_with_epochs(wake_tracking):
    times, headings, spikes = wake_tracking
    with pytest.warns(
        UserWarning, match=r"spike_times \(100%\) fell outside the position"
    ):
        compute_directional_rates([spikes * 1000], times, headings, epochs=WAKE)


def _tracking(duration):
    rng = np.random.default_rng(2)
    times = np.arange(0.0, duration, 0.02)
    headings = np.angle(np.exp(1j * np.cumsum(rng.normal(0, 0.1, times.size))))
    return times, headings


def test_spike_drop_in_milliseconds_warns_when_tracking_outlasts_spikes():
    """Spikes recorded for the first 270 s of 1500 s of tracking, epochs on them.

    In milliseconds the spikes of the first 0.27 s land inside the epochs
    and the median spike stays below 100 times the tracked time, yet nearly
    every spike lies beyond the tracking. In seconds nothing is lost.
    """
    times, headings = _tracking(1500.0)
    spikes = np.sort(np.random.default_rng(3).uniform(0.0, 270.0, 3000))
    epochs = (0.0, 270.0)
    assert np.any(spikes * 1000 < epochs[1])
    assert np.median(spikes * 1000) < 100 * times[-1]
    with pytest.warns(UserWarning, match=r"\(99%\) fell outside the position"):
        compute_directional_rates([spikes * 1000], times, headings, epochs=epochs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_directional_rates([spikes], times, headings, epochs=epochs)
    assert _drop_warnings(caught) == []


@pytest.mark.parametrize(
    ("recording", "warns"),
    [
        # The widest seconds recording in the real-data survey: 17 times.
        (17 * 600.0, False),
        # 6 h of spikes around 10 min of tracking at its start: 36 times,
        # which magnitudes cannot tell from milliseconds, so it warns.
        (6 * 3600.0, True),
    ],
)
def test_spike_recording_far_longer_than_tracking(recording, warns):
    times, headings = _tracking(600.0)
    spikes = np.sort(np.random.default_rng(4).uniform(0.0, recording, 20000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_directional_rates([spikes], times, headings, epochs=(0.0, 600.0))
    assert bool(_drop_warnings(caught)) is warns


def test_spike_drop_counts_spikes_inside_epochs_outside_tracking(wake_tracking):
    """Epochs that reach past the tracked time still lose those spikes."""
    times, headings, spikes = wake_tracking
    with pytest.warns(UserWarning, match=r"inside epochs fell outside the position"):
        compute_directional_rates([spikes], times, headings, epochs=(0.0, 500.0))


def test_warn_on_drop_false_silences_directional_spike_drop(wake_tracking):
    times, headings, spikes = wake_tracking
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_directional_rate(spikes * 1000, times, headings, warn_on_drop=False)
        compute_directional_rates([spikes * 1000], times, headings, warn_on_drop=False)
    assert _drop_warnings(caught) == []


def test_spatial_and_decoding_spike_drop_ignore_spikes_outside_epochs(session_100s):
    session = session_100s
    sleep_rng = np.random.default_rng(1)
    with_sleep = [
        np.sort(np.r_[train, sleep_rng.uniform(100.0, 1000.0, 900)])
        for train in session.spike_times
    ]
    args = (session.env, with_sleep, session.times, session.positions)
    for call in (compute_spatial_rates, decode_session):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(*args, epochs=WAKE)
        assert _drop_warnings(caught) == [], call.__name__
        with pytest.warns(UserWarning, match="fell outside the position time"):
            call(*args)


@pytest.mark.parametrize("window", ["epochs", "spike_window"])
@pytest.mark.parametrize("family", ["spatial", "directional"])
def test_spike_clock_offset_still_warns_with_windows(
    wake_tracking, session_100s, window, family
):
    """Spikes 1000 s later than the tracking (a misaligned file) are all lost.

    No spike falls inside the analysed window, so the window cannot explain
    the drop.
    """
    times, headings, spikes = wake_tracking
    if family == "directional":

        def call():
            in_wake = spikes[spikes < times[-1]] + 1000.0
            compute_directional_rates([in_wake], times, headings, **{window: [WAKE]})

    else:
        session = session_100s

        def call():
            compute_spatial_rates(
                session.env,
                [session.spike_times[0] + 1000.0],
                session.times,
                session.positions,
                **{window: [WAKE]},
            )

    with pytest.warns(UserWarning, match=r"\(100%\).*fell outside the position"):
        call()


def test_decoder_fit_and_predict_warn_once_each_and_not_for_epochs(session_60s):
    """Each call warns at most once; spikes outside its epochs are not lost."""
    from neurospatial.decoding import BayesianDecoder

    session = session_60s
    sleep_rng = np.random.default_rng(1)
    with_sleep = [
        np.sort(np.r_[train, sleep_rng.uniform(60.0, 600.0, 5 * len(train))])
        for train in session.spike_times
    ]
    train, test = (0.0, 40.0), (40.0, 59.0)
    args = (session.times, session.positions)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = BayesianDecoder(session.env, dt=0.25).fit(
            with_sleep, *args, epochs=train
        )
        decoder.predict(with_sleep, session.times, epochs=test)
    assert _drop_warnings(caught) == []

    in_ms = [t * 1000 for t in session.spike_times]
    for call in (
        lambda: decode_session(session.env, in_ms, *args),
        lambda: decoder.predict(in_ms, session.times, epochs=test),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        assert len(_drop_warnings(caught)) == 1


def test_partial_offset_inside_epochs_is_not_detected(session_600s):
    """A known limit: spikes outside ``epochs`` are not checked.

    Spikes shifted 300 s later still leave half of them inside the tracked
    epoch, and the half outside it looks like untracked (sleep) spikes, so no
    warning can tell the offset apart. The docs say to compare the ranges.
    """
    session = session_600s
    shifted = session.spike_times[0] + 300.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(
            session.env,
            [shifted],
            session.times,
            session.positions,
            epochs=[(0.0, 600.0)],
        )
    assert _drop_warnings(caught) == []
    # Without epochs the same spikes are lost to the analysis, so it warns.
    with pytest.warns(UserWarning, match="fell outside the position"):
        compute_spatial_rates(session.env, [shifted], session.times, session.positions)
