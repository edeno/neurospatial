"""Sample-count thresholds have the same inclusive meaning across layouts."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0, [True, True, True]),
        (1, [True, True, True]),
        (2, [False, True, True]),
        (3, [False, False, True]),
    ],
)
def test_threshold_retains_bins_with_exactly_the_minimum_samples(
    layout, threshold, expected
):
    """Equality passes, while empty bins stay inactive even at threshold zero."""
    locations = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
    positions = np.repeat(locations, [1, 2, 3], axis=0)

    env = Environment.from_samples(
        positions, bin_size=1.0, layout=layout, bin_count_threshold=threshold
    )

    np.testing.assert_array_equal(env.bin_at(locations) >= 0, expected)
    assert env.n_bins == sum(expected)
