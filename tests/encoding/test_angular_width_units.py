"""Angular bin sizes and bandwidths are read in ``angle_unit``.

A bin size or bandwidth written in degrees but passed with the default
``angle_unit="rad"`` gives one bin (a perfect-looking cell) or a kernel that
erases the tuning, so these are refused or warned about.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial.encoding import (
    compute_directional_rate,
    head_direction_cell_significance,
)
from neurospatial.simulation import generate_poisson_spikes


@pytest.fixture(scope="module")
def hd_cell():
    """Five minutes of random-walk headings and a cell tuned to 90°."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 300.0, 0.02)
    headings = np.angle(np.exp(1j * np.cumsum(rng.normal(0.0, 0.1, times.size))))
    rate = 1.0 + 20.0 * np.exp(2.0 * (np.cos(headings - np.pi / 2) - 1.0))
    spikes = generate_poisson_spikes(rate, times, seed=1)
    return spikes, times, headings


def test_angular_bin_size_above_pi_raises(hd_cell):
    spikes, times, headings = hd_cell
    with pytest.raises(ValueError, match="pass angle_unit='deg' if bin_size=6"):
        compute_directional_rate(spikes, times, headings, bin_size=6)
    with pytest.raises(ValueError, match="angle_unit='deg'"):
        head_direction_cell_significance(
            [spikes], times, headings, bin_size=6, n_shuffles=2, rng=0
        )

    result = compute_directional_rate(
        spikes, times, np.degrees(headings), bin_size=6, angle_unit="deg"
    )
    assert result.bin_centers.size == 60
    np.testing.assert_allclose(result.bin_size, np.radians(6.0))
    # The same analysis in radians gives the same tuning curve.
    in_radians = compute_directional_rate(
        spikes, times, headings, bin_size=np.radians(6.0)
    )
    np.testing.assert_allclose(result.firing_rate, in_radians.firing_rate)


def test_wide_angular_bandwidth_warns(hd_cell):
    spikes, times, headings = hd_cell
    with pytest.warns(UserWarning, match="wider than 1 radian") as record:
        wide = compute_directional_rate(spikes, times, headings, bandwidth=10)
    assert "angle_unit='deg'" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        narrow = compute_directional_rate(spikes, times, headings, bandwidth=np.pi / 6)
    # What the warning claims: a 10 rad kernel flattens the tuning curve.
    finite = np.isfinite(wide.firing_rate)
    spread = np.ptp(wide.firing_rate[finite]) / np.nanmean(wide.firing_rate)
    assert spread < 0.05
    assert np.ptp(narrow.firing_rate) / np.mean(narrow.firing_rate) > 1.0


def test_coarse_angular_bins_warn(hd_cell):
    spikes, times, headings = hd_cell
    with pytest.warns(UserWarning, match="leaves 2 heading bins") as record:
        coarse = compute_directional_rate(spikes, times, headings, bin_size=3)
    assert "angle_unit='deg'" in str(record[0].message)
    # What the warning claims: two bins cannot place a 90° preference.
    assert coarse.bin_centers.size == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_directional_rate(spikes, times, headings, bin_size=np.deg2rad(6))
    # Both problems in one call make one warning that names both.
    with pytest.warns(UserWarning, match="heading bins") as record:
        compute_directional_rate(spikes, times, headings, bin_size=3, bandwidth=10)
    assert len(record) == 1
    assert "heading bins" in str(record[0].message)
    assert "wider than 1 radian" in str(record[0].message)
    with pytest.raises(ValueError, match="not a finite angular bin width"):
        compute_directional_rate(spikes, times, headings, bin_size=np.inf)
