"""Frame-family rates warn about wrong-clock windows and spikes like spatial rates.

Head-direction, view and object-vector rates share the spatial family's
time-window gates, so a window or spike train on the wrong clock (for example
milliseconds) must not silently produce an empty or truncated rate map.
"""

import warnings

import numpy as np
import pytest

from neurospatial.encoding import compute_object_vector_rate


def _drop_warnings(record):
    return [
        w
        for w in record
        if "excluded ALL" in str(w.message) or "fell outside" in str(w.message)
    ]


@pytest.mark.parametrize("plural", [False, True])
def test_epochs_on_wrong_clock_warn(frame_family, continuous_recording, plural):
    f, r = frame_family, continuous_recording
    call = f.plural if plural else f.single
    spikes = [r.spike_times] if plural else r.spike_times
    with pytest.warns(UserWarning, match=r"excluded ALL.*epochs") as record:
        call(*f.args(r, spikes), **f.defaults, epochs=[(1e4, 6e4)])
    caught = [w for w in record if "excluded ALL" in str(w.message)]
    assert len(caught) == 1
    assert caught[0].filename == __file__


@pytest.mark.parametrize("plural", [False, True])
def test_spike_times_on_wrong_clock_warn(frame_family, continuous_recording, plural):
    f, r = frame_family, continuous_recording
    call = f.plural if plural else f.single
    spikes_ms = r.spike_times * 1000.0
    spikes = [spikes_ms] if plural else spikes_ms
    with pytest.warns(
        UserWarning, match="fell outside the position time window"
    ) as record:
        call(*f.args(r, spikes), **f.defaults)
    caught = [w for w in record if "fell outside" in str(w.message)]
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_well_formed_call_is_silent(frame_family, continuous_recording):
    f, r = frame_family, continuous_recording
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        f.single(*f.args(r, r.spike_times), **f.defaults)
    assert _drop_warnings(record) == []


def test_allocentric_object_vector_epochs_on_wrong_clock_warn(continuous_recording):
    r = continuous_recording
    with pytest.warns(UserWarning, match=r"excluded ALL.*epochs"):
        compute_object_vector_rate(
            r.env,
            r.spike_times,
            r.times,
            r.positions,
            np.array([[50.0, 50.0]]),
            distance_range=(0, 100),
            epochs=[(1e4, 6e4)],
        )
