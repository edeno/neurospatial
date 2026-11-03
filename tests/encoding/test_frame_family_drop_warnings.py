"""Frame-family rates refuse wrong-clock windows and warn about spikes like spatial rates.

Head-direction, view and object-vector rates share the spatial family's
time-window gates, so a window or spike train on the wrong clock (for example
milliseconds) must not silently produce an empty or truncated rate map:
windows that exclude every interval raise, and lost spikes warn.
"""

import warnings

import numpy as np
import pytest

from neurospatial import InputError
from neurospatial.encoding import compute_object_vector_rate


def _drop_warnings(record):
    return [
        w
        for w in record
        if "excluded ALL" in str(w.message) or "fell outside" in str(w.message)
    ]


@pytest.mark.parametrize("plural", [False, True])
def test_epochs_on_wrong_clock_raise(frame_family, continuous_recording, plural):
    f, r = frame_family, continuous_recording
    call = f.plural if plural else f.single
    spikes = [r.spike_times] if plural else r.spike_times
    with pytest.raises(InputError, match=r"excluded ALL.*epochs"):
        call(*f.args(r, spikes), **f.defaults, epochs=[(1e4, 6e4)])


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


def test_allocentric_object_vector_epochs_on_wrong_clock_raise(continuous_recording):
    r = continuous_recording
    with pytest.raises(InputError, match=r"excluded ALL.*epochs"):
        compute_object_vector_rate(
            r.env,
            r.spike_times,
            r.times,
            r.positions,
            np.array([[50.0, 50.0]]),
            distance_range=(0, 100),
            distance_bin_size=5.0,
            epochs=[(1e4, 6e4)],
        )


@pytest.mark.parametrize("plural", [False, True])
def test_warn_on_drop_false_silences_spike_drop(
    frame_family, continuous_recording, plural
):
    f, r = frame_family, continuous_recording
    call = f.plural if plural else f.single
    spikes_ms = r.spike_times * 1000.0
    spikes = [spikes_ms] if plural else spikes_ms
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call(*f.args(r, spikes), **f.defaults, warn_on_drop=False)
    assert _drop_warnings(record) == []


@pytest.mark.parametrize("plural", [False, True])
def test_tuning_curve_warn_on_drop_false_silences_spike_drop(
    continuous_recording, plural
):
    from neurospatial.encoding import compute_tuning_curve, compute_tuning_curves

    r = continuous_recording
    values = r.positions[:, 0]
    spikes_ms = r.spike_times * 1000.0
    call = compute_tuning_curves if plural else compute_tuning_curve
    spikes = [spikes_ms] if plural else spikes_ms
    with pytest.warns(UserWarning, match="fell outside the position time window"):
        call(spikes, r.times, values, bin_size=10.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call(spikes, r.times, values, bin_size=10.0, warn_on_drop=False)
    assert _drop_warnings(record) == []
