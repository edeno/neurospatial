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


@pytest.mark.parametrize(
    ("dt", "refused"), [(1.0, False), (float(np.nextafter(1.0, 2.0)), True)]
)
def test_milliseconds_refusal_needs_dt_above_one_second(trained, dt, refused):
    """At the 1 s cutoff, with dt 200 times the 5 ms spacing either side."""
    env, rates, counts, _ = trained
    times = np.arange(200) * 0.005
    if refused:
        with pytest.raises(ValueError, match="milliseconds"):
            decode_position(env, counts[:200], rates, dt=dt, times=times)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decode_position(env, counts[:200], rates, dt=dt, times=times)


def test_milliseconds_refusal_needs_dt_100_times_the_spacing(trained):
    """dt=25 (meant as ms) on 0.25 s centers is 100 times the spacing."""
    env, rates, counts, _ = trained
    times = np.arange(200) * 0.25
    with pytest.raises(ValueError, match=r"100 times the 0\.25 s spacing"):
        decode_position(env, counts[:200], rates, dt=25.0, times=times)
    with pytest.warns(UserWarning, match="longer than 1 second"):
        decode_position(env, counts[:200], rates, dt=24.9, times=times)


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
