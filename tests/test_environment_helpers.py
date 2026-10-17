"""The shared test helpers report misuse clearly."""

import numpy as np
import pytest
from environment_helpers import layout_of

from neurospatial import Environment


def test_layout_of_says_how_many_layouts_a_build_published():
    """The test helper names the count instead of failing to unpack."""
    positions = np.array([[0.0, 0.0], [4.0, 4.0]])

    def nothing():
        return None

    def two_layouts():
        Environment.from_samples(positions, bin_size=2.0)
        Environment.from_samples(positions, bin_size=1.0)

    for build, count in ((nothing, 0), (two_layouts, 2)):
        with pytest.raises(AssertionError, match=f"but it published {count}"):
            layout_of(build)
    # The suggested call publishes exactly one.
    assert (
        layout_of(lambda: Environment.from_samples(positions, bin_size=2.0)) is not None
    )
