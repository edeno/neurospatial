"""Decoding warns about inputs that silently bias the posterior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial.decoding import (
    bin_spikes_in_time,
    decode_position,
    decode_position_summary,
)
from neurospatial.encoding import compute_spatial_rates


@pytest.fixture(scope="module")
def trained(open_field_120s):
    """Rate maps of the open-field session and its spike counts in 25 ms bins."""
    session = open_field_120s
    rates = compute_spatial_rates(
        session.env, session.spike_times, session.times, session.positions
    )
    counts, centers = bin_spikes_in_time(
        session.spike_times, dt=0.025, t_start=0.0, t_stop=float(session.times[-1])
    )
    return session.env, rates, counts, centers


def test_zero_filled_rates_warn(trained):
    env, rates, counts, centers = trained
    zero_filled = np.nan_to_num(rates.firing_rates)
    # Bins without a rate estimate are the bins outside the training support.
    n_missing = int(np.count_nonzero(~rates.spatial_support))
    assert n_missing > 0.05 * env.n_bins
    with pytest.warns(UserWarning, match="rate 0 for every neuron") as record:
        decode_position(env, counts, zero_filled, dt=0.025, times=centers)
    assert f"{n_missing} of {env.n_bins} bins" in str(record[0].message)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded = decode_position(env, counts, rates, dt=0.025, times=centers)
        # A prior that excludes the missing bins is the documented array path.
        prior = rates.spatial_support * env.bin_sizes
        with_prior = decode_position(
            env, counts, zero_filled, dt=0.025, times=centers, prior=prior
        )
    np.testing.assert_allclose(with_prior.posterior, decoded.posterior, atol=1e-12)


def test_bin_spikes_dt_in_ms_warns(open_field_120s):
    spikes_in_s = open_field_120s.spike_times
    # Spike times on an absolute clock (about 5e4 s) are still seconds.
    on_clock = [5e4 + train for train in spikes_in_s]
    for trains in (spikes_in_s, on_clock):
        with pytest.warns(UserWarning, match="milliseconds") as record:
            counts, _ = bin_spikes_in_time(trains, dt=25)
        assert "dt=0.025" in str(record[0].message)
        # The counts are still those of 25 s bins; the warning changes nothing.
        assert counts.sum() == sum(train.size for train in trains)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bin_spikes_in_time(on_clock, dt=0.025)


def test_decode_dt_above_one_second_warns(trained):
    env, rates, counts, _ = trained
    with pytest.warns(UserWarning, match="dt=25 s is longer than 1 second"):
        decode_position(env, counts, rates, dt=25)


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
def test_decode_dt_mismatch_raises(trained, decode):
    env, rates, counts, centers = trained
    kwargs = {"time_chunk": 500} if decode is decode_position_summary else {}
    with pytest.raises(ValueError, match="milliseconds") as error:
        decode(env, counts, rates, dt=25, times=centers, **kwargs)
    assert "dt=25 s" in str(error.value)
    assert "0.025 s spacing" in str(error.value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decode(env, counts, rates, dt=0.025, times=centers, **kwargs)


def test_overlapping_windows_decode_silently(open_field_120s, trained):
    """The replay recipe: 20 ms windows stepped by 10 ms, dt twice the spacing."""
    env, rates, _, _ = trained
    counts, grid = bin_spikes_in_time(
        open_field_120s.spike_times, dt=0.01, t_start=10.0, t_stop=12.0
    )
    windows = counts[:-1] + counts[1:]
    centers = (grid[:-1] + grid[1:]) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded = decode_position(env, windows, rates, dt=0.02, times=centers)
    assert decoded.posterior.shape == (centers.size, env.n_bins)


def test_zero_filled_rates_warn_names_the_decoder_fix(open_field_120s, trained):
    from neurospatial.decoding import BayesianDecoder

    env, rates, _, _ = trained
    decoder = BayesianDecoder.from_models(env, np.nan_to_num(rates.firing_rates))
    session = open_field_120s
    with pytest.warns(UserWarning, match="rate 0 for every neuron") as record:
        decoder.predict(session.spike_times, session.times)
    message = str(record[0].message)
    assert "BayesianDecoder.from_rates(rates)" in message
    assert "decode_position" not in message


@pytest.mark.parametrize("decode", [decode_position, decode_position_summary])
def test_long_overlapping_windows_are_not_called_milliseconds(trained, decode):
    """600 ms windows stepped by 5 ms: dt is 120 times the spacing, in seconds."""
    env, rates, counts, _ = trained
    times = np.arange(200) * 0.005
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decode(env, counts[:200], rates, dt=0.6, times=times)


def _decode_or_refuse(trained, dt, step, refused):
    """Decode 200 bins centered ``step`` s apart; check refusal or acceptance."""
    env, rates, counts, _ = trained
    times = np.arange(200) * step
    if refused:
        with pytest.raises(ValueError, match="milliseconds") as error:
            decode_position(env, counts[:200], rates, dt=dt, times=times)
        assert f"{step:g} s spacing" in str(error.value)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A dt above 1 s still gets the general warning, which has no times.
        warnings.filterwarnings("ignore", "(?s).*longer than 1 second")
        decoded = decode_position(env, counts[:200], rates, dt=dt, times=times)
    assert decoded.posterior.shape == (200, env.n_bins)


@pytest.mark.parametrize(
    ("dt", "step", "refused"),
    [
        # 1 ms bins with dt=1 meant as 1 ms: read as ms, a one-step window.
        (1.0, 0.001, True),
        # 25 ms bins with dt=25 meant as 25 ms.
        (25.0, 0.025, True),
        # 600 ms windows stepped by 5 ms: as ms, 0.12 of a step, no window.
        (0.6, 0.005, False),
        # 2 s windows stepped by 10 ms: as ms, 0.2 of a step.
        (2.0, 0.01, False),
    ],
    ids=["1ms-bins-dt-1", "25ms-bins-dt-25", "600ms-by-5ms", "2s-by-10ms"],
)
def test_dt_refused_when_read_as_milliseconds_it_fits_the_step(
    trained, dt, step, refused
):
    _decode_or_refuse(trained, dt, step, refused)


# A step of 2**-7 s makes each edge exact in floating point.
_STEP = 2.0**-7


@pytest.mark.parametrize(
    ("dt", "refused"),
    [
        # Lower edge: dt / 1000 is half the step.
        (500 * _STEP, True),
        (float(np.nextafter(500 * _STEP, 0.0)), False),
        # Upper edge: dt / 1000 is 20 steps.
        (20000 * _STEP, True),
        (float(np.nextafter(20000 * _STEP, np.inf)), False),
    ],
    ids=["half-step", "below-half-step", "twenty-steps", "above-twenty-steps"],
)
def test_milliseconds_band_edges_are_inclusive(trained, dt, refused):
    _decode_or_refuse(trained, dt, _STEP, refused)


@pytest.mark.parametrize(
    ("dt", "warns"), [(1.0, False), (float(np.nextafter(1.0, 2.0)), True)]
)
def test_bin_spikes_dt_warning_starts_above_one_second(open_field_120s, dt, warns):
    trains = open_field_120s.spike_times
    if warns:
        with pytest.warns(UserWarning, match="longer than 1 second"):
            bin_spikes_in_time(trains, dt=dt)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bin_spikes_in_time(trains, dt=dt)


@pytest.mark.parametrize(("n_zero", "warns"), [(5, False), (6, True)])
def test_zero_filled_warning_starts_above_five_percent(n_zero, warns):
    """On 100 bins, 5 all-zero bins is the 5% cutoff; 6 is above it."""
    from neurospatial import Environment

    edges = (np.arange(11.0), np.arange(11.0))
    env = Environment.from_grid_mask(np.ones((10, 10), dtype=bool), edges)
    assert env.n_bins == 100
    models = np.full((2, env.n_bins), 2.0)
    models[:, :n_zero] = 0.0
    counts = np.array([[1, 0], [0, 2], [1, 1]])
    if warns:
        with pytest.warns(UserWarning, match=f"{n_zero} of 100 bins have rate 0"):
            decode_position(env, counts, models, dt=0.1)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decode_position(env, counts, models, dt=0.1)
