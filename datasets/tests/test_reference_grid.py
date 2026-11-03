from __future__ import annotations

import numpy as np
from reference_checks import nan_aware_autocorrelation


def _direct(rate_map, min_overlap):
    """Pearson correlation of the map with itself shifted, one lag at a time."""
    n_x, n_y = rate_map.shape
    out = np.full((2 * n_x - 1, 2 * n_y - 1), np.nan)
    for dx in range(-n_x + 1, n_x):
        for dy in range(-n_y + 1, n_y):
            a = rate_map[max(dx, 0) : n_x + min(dx, 0), max(dy, 0) : n_y + min(dy, 0)]
            b = rate_map[
                max(-dx, 0) : n_x + min(-dx, 0), max(-dy, 0) : n_y + min(-dy, 0)
            ]
            both = np.isfinite(a) & np.isfinite(b)
            if (
                both.sum() >= min_overlap
                and np.std(a[both]) > 0
                and np.std(b[both]) > 0
            ):
                out[dx + n_x - 1, dy + n_y - 1] = np.corrcoef(a[both], b[both])[0, 1]
    return out


def test_nan_aware_autocorrelation_matches_full_overlap():
    rate_map = np.random.default_rng(0).gamma(2.0, 1.0, (9, 7))
    expected = _direct(rate_map, 3)
    got = nan_aware_autocorrelation(rate_map, 3)
    defined = np.isfinite(expected)
    np.testing.assert_allclose(got[defined], expected[defined], atol=1e-10)
    assert np.isnan(got[~defined]).all()


def test_nan_aware_autocorrelation_skips_unvisited_bins():
    rng = np.random.default_rng(1)
    rate_map = rng.gamma(2.0, 1.0, (10, 10))
    rate_map[rng.random((10, 10)) < 0.3] = np.nan
    expected = _direct(rate_map, 5)
    got = nan_aware_autocorrelation(rate_map, 5)
    defined = np.isfinite(expected)
    np.testing.assert_allclose(got[defined], expected[defined], atol=1e-9)
    # Zero-filling would change the correlation wherever unvisited bins overlap.
    zero_filled = _direct(np.nan_to_num(rate_map), 5)
    assert np.nanmax(np.abs(zero_filled - expected)) > 0.05
