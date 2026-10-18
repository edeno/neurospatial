"""Overlays drawn in arena coordinates refuse a polar environment.

A polar environment's axes are ``(distance, angle)`` around the animal, so an
allocentric heading arrow, a pose, an object-vector line or a camera frame
drawn on them would look plausible and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    VideoOverlay,
    _convert_overlays_to_data,
)

FRAME_TIMES = np.array([0.0, 1.0, 2.0])
POLAR_ROWS = np.array([[5.0, 0.0], [15.0, 0.5], [25.0, -0.5]])  # (cm, rad)


@pytest.fixture(scope="module")
def polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
    )


ARENA_OVERLAYS = {
    "HeadDirectionOverlay": lambda: HeadDirectionOverlay(
        headings=np.zeros(3), times=FRAME_TIMES
    ),
    "BodypartOverlay": lambda: BodypartOverlay(
        data={"nose": POLAR_ROWS}, times=FRAME_TIMES
    ),
    "ObjectVectorOverlay": lambda: ObjectVectorOverlay(
        object_positions=np.array([[10.0, 0.0]]),
        animal_positions=POLAR_ROWS,
        times=FRAME_TIMES,
    ),
    "VideoOverlay": lambda: VideoOverlay(source=np.zeros((3, 8, 8, 3), np.uint8)),
}


@pytest.mark.parametrize("name", ARENA_OVERLAYS)
def test_arena_overlays_refuse_a_polar_environment(polar, name):
    with pytest.raises(CartesianOnlyError, match=f"{name} does not work"):
        _convert_overlays_to_data(
            [ARENA_OVERLAYS[name]()], FRAME_TIMES, len(FRAME_TIMES), polar
        )


def test_position_overlay_draws_polar_rows(polar):
    """Positions in the environment's own (distance, angle) rows still draw."""
    data = _convert_overlays_to_data(
        [PositionOverlay(positions=POLAR_ROWS, times=FRAME_TIMES)],
        FRAME_TIMES,
        len(FRAME_TIMES),
        polar,
    )

    np.testing.assert_allclose(data.positions[0].data, POLAR_ROWS)
