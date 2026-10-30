"""fit_isotonic_trajectory on a track whose bins differ in length."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import fit_isotonic_trajectory


def test_weights_are_the_mass_of_the_highest_density_bin():
    """The MAP bin is the bin of highest density, mass per cm.

    Bins are 10, 10, 10, 10 and 5 cm. In the last time bin the 10 cm bin 3
    holds 0.4 (0.04 per cm) and the 5 cm bin 4 holds 0.3 (0.06 per cm), so
    the MAP is bin 4 and its weight is 0.3, not the largest mass 0.4.
    """
    env = Environment.linear_track(
        node_positions=[[0.0, 0.0], [40.0, 0.0], [45.0, 0.0]],
        bin_size=10.0,
        units="cm",
    )
    np.testing.assert_allclose(env.bin_sizes, [10.0, 10.0, 10.0, 10.0, 5.0])
    posterior = np.array(
        [
            [0.7, 0.075, 0.075, 0.075, 0.075],
            [0.075, 0.7, 0.075, 0.075, 0.075],
            [0.075, 0.075, 0.7, 0.075, 0.075],
            [0.1, 0.1, 0.1, 0.4, 0.3],
        ]
    )
    result = fit_isotonic_trajectory(env, posterior, np.arange(4) * 0.01)
    np.testing.assert_allclose(result.fitted_positions, [5.0, 15.0, 25.0, 42.5])
    assert result.r_squared == pytest.approx(1.0)
    assert result.mean_map_probability == pytest.approx((3 * 0.7 + 0.3) / 4)
    assert result.score == pytest.approx(0.6)
