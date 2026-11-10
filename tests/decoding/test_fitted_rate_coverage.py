"""Credible-region coverage when the decoder uses rates fitted on training data.

``test_posterior_calibration.py`` decodes with the generating rates, where
coverage equals region mass exactly. Here a simulated session is split in
time: :class:`BayesianDecoder` fits the rate maps on the first part and
decodes the rest. The decoder's model then holds except for

- error in the fitted rates (noise and smoothing bias),
- motion within a time bin (about one bin per ``dt`` here),
- an occupancy that is not exactly uniform over space.

Coverage is compared with the mean posterior mass of the regions, as in
``test_posterior_calibration.py``. Decoded time bins are autocorrelated, so
the standard error comes from the means of 10-s blocks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial.decoding import BayesianDecoder, credible_region
from neurospatial.simulation import linear_track_session, open_field_session

LEVELS = (0.5, 0.8, 0.95)
BLOCK = 10.0


def _session(kind, duration, seed=0):
    if kind == "track":
        return linear_track_session(
            duration=duration,
            track_length=200.0,
            bin_size=2.0,
            n_place_cells=40,
            n_laps=int(duration / 11),
            rng=seed,
        )
    return open_field_session(
        duration=duration, arena_size=100.0, bin_size=4.0, n_place_cells=60, rng=seed
    )


def _coverage(session, decoder, start, stop):
    """Per level: coverage, mean region mass, block SE of (hit - mass)."""
    result = decoder.predict(session.spike_times, session.times, epochs=(start, stop))
    posterior = np.asarray(result.posterior)
    true_positions = np.column_stack(
        [
            np.interp(result.times, session.times, session.positions[:, d])
            for d in range(session.positions.shape[1])
        ]
    )
    true_bin = session.env.bin_at(true_positions)
    rows = np.flatnonzero(true_bin >= 0)
    blocks = ((result.times[rows] - start) // BLOCK).astype(int)
    out = {}
    for level in LEVELS:
        regions = credible_region(session.env, posterior, level=level)
        hit = np.array([true_bin[i] in regions[i] for i in rows], dtype=float)
        mass = np.array([posterior[i, regions[i]].sum() for i in rows])
        gap = hit - mass
        block_means = np.array([gap[blocks == b].mean() for b in np.unique(blocks)])
        se = block_means.std(ddof=1) / np.sqrt(len(block_means))
        out[level] = (hit.mean(), mass.mean(), se)
    return out


def _fitted(session, train, method="diffusion_kde", dt=0.1):
    with warnings.catch_warnings():
        # ``diffusion_kde`` reports its 5 cm default bandwidth.
        warnings.simplefilter("ignore", UserWarning)
        return BayesianDecoder(session.env, dt=dt, method=method).fit(
            session.spike_times, session.times, session.positions, epochs=(0.0, train)
        )


def _oracle(session, dt=0.1):
    env = session.env
    rates = np.stack(
        [m.firing_rate(np.zeros(env.n_bins), env.bin_centers) for m in session.models]
    )
    return BayesianDecoder.from_models(env, rates, dt=dt)


@pytest.fixture(scope="module")
def track():
    """20 min on a 200 cm track: fit on the first 10, decode the last 10."""
    return _session("track", 1200.0)


def test_generating_rates_cover_at_their_mass(track):
    """Without fitting error the only departures are motion and occupancy."""
    for level, (coverage, mass, se) in _coverage(
        track, _oracle(track), 600.0, 1200.0
    ).items():
        assert mass >= level
        assert abs(coverage - mass) <= 3 * se, (level, coverage, mass, se)


def test_default_fit_on_a_track_is_not_overconfident(track):
    """10 min of training on 100 track bins: coverage at or above mass.

    The 5 cm smoothing of the default estimator widens the posterior a
    little, so coverage may exceed the mass, by 1-2.5 points here.
    """
    for level, (coverage, mass, se) in _coverage(
        track, _fitted(track, 600.0), 600.0, 1200.0
    ).items():
        assert coverage - mass >= -3 * se, (level, coverage, mass, se)
        assert coverage - mass <= 0.03, (level, coverage, mass)


def test_unsmoothed_binned_fit_is_overconfident(track):
    """Unsmoothed 2 cm rate maps from 10 min: the 95% region covers about 90%."""
    coverage, mass, _ = _coverage(
        track, _fitted(track, 600.0, method="binned"), 600.0, 1200.0
    )[0.95]
    assert coverage < mass - 0.04, (coverage, mass)


@pytest.mark.slow
def test_open_field_coverage_needs_long_training():
    """625 open-field bins: 10 min of training cover 84%, 50 min cover 95%."""
    session = _session("open", 3600.0)
    short = _coverage(session, _fitted(session, 600.0), 3000.0, 3600.0)
    coverage, mass, _ = short[0.95]
    assert coverage < mass - 0.08, (coverage, mass)
    long = _coverage(session, _fitted(session, 3000.0), 3000.0, 3600.0)
    for level, (coverage, mass, se) in long.items():
        assert abs(coverage - mass) <= 3 * se, (level, coverage, mass, se)
