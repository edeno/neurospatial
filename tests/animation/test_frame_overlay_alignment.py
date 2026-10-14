"""Visible overlay parity across HTML storage modes and worker boundaries."""

import base64
import io
import json
import re
from dataclasses import replace

import matplotlib.image as mpimg
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._parallel import parallel_render_frames
from neurospatial.animation.backends.html_backend import render_html
from neurospatial.animation.overlays import (
    BodypartData,
    EventData,
    HeadDirectionData,
    OverlayData,
    PositionData,
    TimeSeriesData,
    VideoData,
)
from neurospatial.animation.skeleton import Skeleton


@pytest.fixture
def overlay_scene():
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(0, 110, 10.0), np.arange(0, 110, 10.0)),
    )
    positions = np.array([[25, 25], [40, 30], [55, 60], [70, 70]], dtype=float)
    pose = BodypartData(
        bodyparts={
            "nose": positions + np.array([-5, 0]),
            "tail": positions + np.array([0, -5]),
        },
        skeleton=Skeleton(
            name="test",
            nodes=("nose", "tail"),
            edges=(("nose", "tail"),),
            edge_color="magenta",
            edge_width=3.0,
        ),
        colors={"nose": "magenta", "tail": "magenta"},
    )
    overlays = OverlayData(
        positions=[PositionData(positions, color="red", size=8.0, trail_length=3)],
        bodypart_sets=[pose],
        head_directions=[
            HeadDirectionData(np.array([0.0, 0.5, 1.0, 2.0]), "yellow", length=15.0)
        ],
        events=[
            EventData(
                event_positions={"reward": np.array([[35.0, 45.0]])},
                event_frame_indices={"reward": np.array([1])},
                colors={"reward": "cyan"},
                markers={"reward": "s"},
                size=10.0,
                decay_frames=4,
                border_color="white",
                border_width=0.5,
                opacity=0.9,
            )
        ],
    )
    return env, [np.zeros(env.n_bins) for _ in range(4)], overlays


def _html_frames(path, embed):
    if not embed:
        return [
            mpimg.imread(frame) for frame in sorted(path.with_suffix("").glob("*.png"))
        ]
    match = re.search(r"const frames = (\[.*?\]);", path.read_text(), re.DOTALL)
    assert match is not None
    return [
        mpimg.imread(io.BytesIO(base64.b64decode(encoded)), format="png")
        for encoded in json.loads(match.group(1))
    ]


@pytest.mark.parametrize("embed", [True, False])
@pytest.mark.parametrize(
    "kind", ["position", "bodyparts", "heading", "decay", "regions"]
)
def test_html_exports_visible_overlays_in_both_modes(
    tmp_path, overlay_scene, embed, kind
):
    """Check the images themselves, rather than HTML-file creation alone."""
    env, fields, all_overlays = overlay_scene
    baseline = None
    show_regions = False
    if kind == "position":
        overlays = OverlayData(positions=all_overlays.positions)
    elif kind == "bodyparts":
        overlays = OverlayData(bodypart_sets=all_overlays.bodypart_sets)
    elif kind == "heading":
        # Keep the origin marker in the baseline so the changed pixels prove
        # the heading arrow, rather than its paired position, was rendered.
        baseline = OverlayData(positions=all_overlays.positions)
        overlays = replace(baseline, head_directions=all_overlays.head_directions)
    elif kind == "decay":
        overlays = OverlayData(events=all_overlays.events)
    else:
        env.regions.add("goal", point=np.array([75.0, 25.0]))
        overlays = None  # Regions must work even without another overlay.
        show_regions = True

    kwargs = {
        "embed": embed,
        "dpi": 50,
        "n_workers": 1,
        "cmap": "gray",
        "vmin": 0,
        "vmax": 1,
    }
    base_path = render_html(
        env, fields, tmp_path / "baseline.html", overlay_data=baseline, **kwargs
    )
    overlay_path = render_html(
        env,
        fields,
        tmp_path / "overlays.html",
        overlay_data=overlays,
        show_regions=show_regions,
        **kwargs,
    )
    base_images = _html_frames(base_path, embed)
    overlay_images = _html_frames(overlay_path, embed)
    assert len(base_images) == len(overlay_images) == len(fields)
    # Frame 2 is after the event at frame 1: it also verifies decay support.
    changed = np.any(base_images[2] != overlay_images[2], axis=-1)
    assert np.count_nonzero(changed) > 10
    assert "const overlayData = null;" in overlay_path.read_text()


@pytest.mark.parametrize("embed", [True, False])
def test_html_warned_video_is_skipped_without_reading_or_mutating_it(
    tmp_path, overlay_scene, embed, monkeypatch
):
    env, fields, overlays = overlay_scene
    video = VideoData(
        frame_indices=np.arange(4),
        reader=np.zeros((4, 2, 2, 3), dtype=np.uint8),
        transform_to_env=None,
        env_bounds=(0.0, 100.0, 0.0, 100.0),
        alpha=0.5,
        z_order="above",
    )
    overlays = replace(overlays, videos=[video])

    def must_not_read(*args, **kwargs):
        raise AssertionError("A skipped video must not be read")

    monkeypatch.setattr(VideoData, "get_frame", must_not_read)
    with pytest.warns(UserWarning, match="does not support video overlays"):
        path = render_html(
            env,
            fields,
            tmp_path / "skipped_video.html",
            embed=embed,
            n_workers=1,
            dpi=40,
            overlay_data=overlays,
        )
    assert path.exists()
    assert overlays.videos == [video]
    assert len(_html_frames(path, embed)) == len(fields)


@pytest.mark.parametrize("reuse_artists", [True, False])
def test_parallel_chunks_keep_global_overlay_history_and_time(
    tmp_path, overlay_scene, reuse_artists
):
    """Chunked images must match serial images, including the first chunk frame."""
    env, fields, overlays = overlay_scene
    # Use a nonzero, nonuniform clock to expose accidentally reset cursors.
    frame_times = np.array([101.0, 102.5, 104.0, 106.0])
    sample_times = np.arange(99.0, 109.0, 0.25)
    ts = TimeSeriesData(
        data=np.sin(sample_times),
        times=sample_times,
        start_indices=np.searchsorted(sample_times, frame_times - 1.0),
        end_indices=np.searchsorted(sample_times, frame_times + 1.0),
        label="signal",
        color="blue",
        window_seconds=2.0,
        linewidth=1.0,
        alpha=1.0,
        group=None,
        normalize=False,
        show_cursor=True,
        cursor_color="red",
        global_vmin=-1.0,
        global_vmax=1.0,
    )
    overlays = replace(overlays, timeseries=[ts], frame_times=frame_times)
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    for output_dir, workers in ((serial_dir, 1), (parallel_dir, 2)):
        output_dir.mkdir()
        parallel_render_frames(
            env,
            fields,
            str(output_dir),
            "gray",
            0.0,
            1.0,
            [f"t={time:g} s" for time in frame_times],
            50,
            workers,
            reuse_artists=reuse_artists,
            overlay_data=overlays,
            frame_times=frame_times,
        )

    for frame in range(4):
        name = f"frame_{frame:05d}.png"
        np.testing.assert_array_equal(
            mpimg.imread(serial_dir / name), mpimg.imread(parallel_dir / name)
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_default_worker_path_renders_and_updates_video(
    tmp_path, overlay_scene, workers
):
    """Video frames stay visible and follow global frame indices across chunks."""
    env, fields, _ = overlay_scene
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]], dtype=np.uint8
    )
    frames = np.broadcast_to(colors[:, None, None, :], (4, 2, 2, 3)).copy()
    video = VideoData(
        frame_indices=np.arange(4),
        reader=frames,
        transform_to_env=None,
        env_bounds=(0.0, 100.0, 0.0, 100.0),
        alpha=1.0,
        z_order="above",
    )
    parallel_render_frames(
        env,
        fields,
        str(tmp_path),
        "gray",
        0.0,
        1.0,
        None,
        40,
        workers,
        overlay_data=OverlayData(videos=[video]),
    )
    for frame, color in enumerate(colors):
        image = mpimg.imread(tmp_path / f"frame_{frame:05d}.png")
        matching = np.all(np.isclose(image[..., :3], color / 255.0), axis=-1)
        assert np.count_nonzero(matching) > 100
