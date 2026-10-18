"""A transform that changes lengths says which unit its coordinates are in."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.transforms import Affine2D, scale_2d, translate


@pytest.fixture(scope="module")
def camera() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 40.0), np.arange(0.0, 40.0))
    return Environment.from_samples(
        np.c_[x.ravel(), y.ravel()], bin_size=4.0, units="px"
    )


def test_a_calibration_relabels_the_units(camera):
    room = camera.apply_transform(scale_2d(0.5), units="cm")

    assert room.units == "cm"
    np.testing.assert_allclose(room.bin_centers, camera.bin_centers * 0.5)


@pytest.mark.parametrize(
    ("transform", "change"),
    [
        (scale_2d(0.5), "scales lengths by 0.5"),
        (
            Affine2D(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
            "stretches or shears lengths",
        ),
    ],
)
def test_changing_lengths_without_units_is_refused(camera, transform, change):
    """Kept 'px' labels would be wrong after a pixel-to-cm calibration."""
    with pytest.raises(ValueError, match=change) as caught:
        camera.apply_transform(transform)

    assert "units='px' to keep it" in str(caught.value)


def test_a_stretch_in_the_same_units_keeps_them_when_asked(camera):
    stretched = camera.apply_transform(scale_2d(1.5), units=camera.units)

    assert stretched.units == "px"


def test_rigid_transforms_keep_the_units(camera):
    angle = 0.3
    rotation = Affine2D(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )

    for transform in (translate(10.0, -5.0), rotation):
        assert camera.apply_transform(transform).units == "px"


def test_an_environment_without_units_can_be_scaled(camera):
    unlabeled = camera.with_metadata(units=None)

    assert unlabeled.apply_transform(scale_2d(2.0)).units is None
