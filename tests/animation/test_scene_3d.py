"""Three-dimensional scenes preserve geometry, z and missing observations."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EpochOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.animation.scene import Animation, AnimationField
from neurospatial.animation.skeleton import Skeleton


@pytest.fixture
def env3d():
    return Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        (
            np.array([0.0, 2.0, 4.0]),
            np.array([10.0, 13.0, 16.0]),
            np.array([20.0, 25.0, 30.0]),
        ),
        units="cm",
    )


def test_3d_frame_preserves_z_for_positions_skeletons_and_vectors(env3d):
    times = np.array([10.0, 10.2, 10.4])
    positions = np.array([[1.0, 11.0, 21.0], [2.0, 12.0, 24.0], [3.0, 14.0, 28.0]])
    scene = Animation(
        env3d,
        AnimationField(np.ones((3, env3d.n_bins)), times),
        overlays=[
            PositionOverlay(positions, times, trail_length=3),
            BodypartOverlay(
                {"nose": positions, "tail": positions + np.array([0.1, 0.1, 0.1])},
                times,
                skeleton=Skeleton(
                    name="mouse", nodes=("nose", "tail"), edges=(("nose", "tail"),)
                ),
            ),
            HeadDirectionOverlay(np.array([[0.0, 0.0, 1.0]] * 3), times, length=2),
            ObjectVectorOverlay(np.array([[3.0, 14.0, 29.0]]), positions, times),
            TimeSeriesOverlay(np.array([1.0, np.nan, 2.0]), times, label="Speed"),
            SpikeRasterOverlay([np.array([10.1])], unit_ids=[72]),
            EpochOverlay([(10.0, 10.3)], label="Run"),
        ],
    )
    image = scene.render_frame(10.2, dpi=35)
    assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
    ax = scene._renderer.figure.axes[0]
    assert ax.name == "3d"
    assert ax.get_zlabel() == "z (cm)"
    # Position and the bodyparts keep their true height, not a z=0 plane.
    z_values = [
        np.asarray(item._offsets3d[2])
        for item in ax.collections
        if hasattr(item, "_offsets3d")
    ]
    assert any(np.allclose(z, [24.0]) for z in z_values)
    lines = [line for line in ax.lines if line.get_visible()]
    assert any(np.allclose(line.get_data_3d()[2], [24.0, 24.1]) for line in lines)
    assert any(np.allclose(line.get_data_3d()[2], [24.0, 29.0]) for line in lines)
    scene.close()


def test_3d_grid_draws_only_active_cells_and_distinguishes_missing(env3d):
    mask = np.ones((2, 2, 2), dtype=bool)
    mask[0, 1, 0] = False
    env = Environment.from_grid_mask(mask, env3d.grid_edges)
    values = np.arange(env.n_bins, dtype=float)
    values[1] = np.nan
    scene = Animation(env, AnimationField(values[None, :], [5.0], vmin=0, vmax=6))
    scene.render_frame(5.0, dpi=30)
    cells = scene._renderer.figure.axes[0].collections[0]
    colors = cells.get_facecolor()
    assert len(colors) == env.n_bins * 6
    # Missing active bins have six gray faces; an observed zero remains colored.
    assert np.sum(np.all(np.isclose(colors[:, :3], 0.75), axis=1)) == 6
    assert not np.allclose(colors[:, :3], 0.75)
    scene.close()


@pytest.mark.parametrize(
    "overlay,match",
    [
        (
            HeadDirectionOverlay(np.array([0.0, 1.0]), np.array([0.0, 0.2])),
            "unit vectors",
        ),
        (
            VideoOverlay(
                np.zeros((2, 2, 2, 3), dtype=np.uint8), times=np.array([0.0, 0.2])
            ),
            "2D environment",
        ),
    ],
)
def test_3d_ambiguous_planar_overlays_fail_before_rendering(env3d, overlay, match):
    with pytest.raises(ValueError, match=match):
        Animation(
            env3d,
            np.ones((2, env3d.n_bins)),
            frame_times=[0.0, 0.2],
            overlays=[overlay],
        )


def test_3d_point_regions_have_explicit_height(env3d):
    from neurospatial.regions import Regions

    regions = Regions()
    regions.add("Reward", point=(1.0, 12.0, 27.0))
    env = Environment.from_bins(env3d.binning, regions=regions, units="cm")
    scene = Animation(
        env, np.ones((1, env.n_bins)), frame_times=[10.0], show_regions=True
    )
    scene.render_frame(10.0, dpi=30)
    ax = scene._renderer.figure.axes[0]
    assert any(text.get_text() == "Reward" for text in ax.texts)
    assert any(
        np.allclose(collection._offsets3d[2], [27.0])
        for collection in ax.collections
        if hasattr(collection, "_offsets3d")
    )
    scene.close()
