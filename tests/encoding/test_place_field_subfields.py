"""After the subfield rule splits a field, the neighbouring field is still found.

The subfield rule replaces a field by its core when a second field shares the
region. The next candidate peak can then sit on the core's flank, with a
region that runs into the second field. That candidate is a flank and is
discarded, but the second field must still be found from its own peak
(``detect_place_fields`` Notes, steps 4 and 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import detect_place_fields


@pytest.fixture(scope="module")
def box():
    """A 100 x 100 cm open field of 2.5 cm bins."""
    edges = np.arange(0.0, 100.1, 2.5)
    return Environment.from_grid_mask(
        np.ones((40, 40), dtype=bool), grid_edges=(edges, edges), units="cm"
    )


def _two_fields(env, second_peak, separation):
    centers = env.bin_centers
    first = np.array([35.0, 50.0])
    second = first + np.array([separation, 0.0])
    width = 8.0
    rate = 10.0 * np.exp(-np.sum((centers - first) ** 2, axis=1) / (2 * width**2))
    rate += second_peak * np.exp(
        -np.sum((centers - second) ** 2, axis=1) / (2 * width**2)
    )
    return rate, first, second


# The second peak (3 Hz) is below the first field's flank just outside its
# core (0.36 x 10 = 3.6 Hz), so the next candidate after the split is on that
# flank, and its region runs into the second field.
@pytest.mark.parametrize(("second_peak", "separation"), [(3.3, 24.0), (3.0, 26.0)])
def test_neighbouring_field_is_found_after_a_subfield_split(
    box, second_peak, separation
):
    rate, first, second = _two_fields(box, second_peak, separation)
    # The saddle lies above the 0.2 floor, so the two fields form one region
    # at the first peak's threshold, and the subfield rule must split it.
    saddle = rate[box.bin_at(((first + second) / 2)[None])[0]] / rate.max()
    assert saddle > 0.2

    result = detect_place_fields(box, rate, bandwidth=0.0)

    peaks = [box.bin_centers[f[np.argmax(rate[f])]] for f in result.fields]
    assert len(result.fields) == 2
    np.testing.assert_allclose(peaks[0], first, atol=2.5)
    # The sum of two Gaussians pulls the second maximum toward the first
    # field by about one bin.
    np.testing.assert_allclose(peaks[1], second, atol=5.0)
    # The two fields share no bin.
    assert not set(result.fields[0].tolist()) & set(result.fields[1].tolist())
