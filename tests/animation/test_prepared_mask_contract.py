"""Adapter support has one meaning for sampling, raw geometry and realignment."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._sampling import sample_observation
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    ObjectVectorData,
    PositionData,
    _convert_overlays_to_data,
)

CLOCK = np.arange(4) * 0.1
POINTS = np.array([[0.2, 0.2], [1.0, 1.0], [1.8, 0.2], [1.8, 1.8]])


@dataclass
class Adapter:
    data: object

    def convert_to_data(self, frame_times, n_frames, env):
        return self.data


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


def position_masks(case):
    """Given masks and the effective intervals: a rejected row is deleted."""
    if case == "sample":
        return np.array([True, False, True, True]), None, [True, True, True]
    if case == "interval":
        return None, np.array([False, True, True]), [False, True, True]
    return (
        np.array([True, False, True, True]),
        np.array([True, True, False]),
        [True, True, False],
    )


def joined_rows(samples, expected_intervals):
    """Consecutive valid rows joined across a crossable gap."""
    rows = np.flatnonzero(np.ones(4, dtype=bool) if samples is None else samples)
    return [[a, b] for a, b in pairwise(rows) if expected_intervals[b - 1]]


@pytest.mark.parametrize("case", ["sample", "interval", "both"])
def test_position_preparation_owns_effective_support(env, case):
    sample_mask, interval_mask, expected_intervals = position_masks(case)
    points = POINTS.copy()
    source = PositionData(
        points,
        "red",
        5,
        4,
        times=CLOCK,
        sample_valid=sample_mask,
        interval_valid=interval_mask,
    )
    prepared = _convert_overlays_to_data([Adapter(source)], CLOCK, len(CLOCK), env)[
        0
    ].positions[0]
    expected_valid = np.ones(4, dtype=bool) if sample_mask is None else sample_mask
    np.testing.assert_array_equal(prepared.sample_valid, expected_valid)
    np.testing.assert_array_equal(prepared.interval_valid, expected_intervals)
    np.testing.assert_array_equal(prepared.data[expected_valid], POINTS[expected_valid])
    assert np.isnan(prepared.data[~expected_valid]).all()
    # The caller's arrays are neither shared nor changed.
    assert not np.shares_memory(prepared.data, points)
    assert not np.shares_memory(prepared.data, source.data)
    np.testing.assert_array_equal(points, POINTS)
    if sample_mask is not None:
        assert not np.shares_memory(prepared.sample_valid, sample_mask)
    if interval_mask is not None:
        assert not np.shares_memory(prepared.interval_valid, interval_mask)
    for index, time in enumerate(CLOCK):
        sampled = sample_observation(prepared, prepared.data, time)
        if expected_valid[index]:
            np.testing.assert_array_equal(sampled, POINTS[index])
        else:
            # At a rejected row's time, its neighbors are sampled as if the
            # row were deleted; it lies halfway between them.
            np.testing.assert_allclose(
                sampled, (POINTS[index - 1] + POINTS[index + 1]) / 2
            )
    for index, time in enumerate((CLOCK[:-1] + CLOCK[1:]) / 2):
        sampled = sample_observation(prepared, prepared.data, time)
        assert np.isfinite(sampled).all() == expected_intervals[index]


@pytest.mark.parametrize("kind", ["bodypart", "heading", "object_vector"])
def test_spatial_adapter_rows_respect_rejected_samples(env, kind):
    valid = np.array([True, False, True, True])
    points = POINTS.copy()
    if kind == "bodypart":
        source = BodypartData(
            {"nose": points},
            None,
            None,
            times=CLOCK,
            sample_valid={"nose": valid},
            interval_valid={"nose": np.ones(3, dtype=bool)},
        )
        attribute = "bodypart_sets"
    elif kind == "heading":
        points = np.arange(4.0)
        source = HeadDirectionData(
            points,
            "red",
            1,
            times=CLOCK,
            sample_valid=valid,
            interval_valid=np.ones(3, dtype=bool),
        )
        attribute = "head_directions"
    else:
        source = ObjectVectorData(
            np.array([[1.0, 1.0]]),
            points,
            None,
            "red",
            1,
            True,
            "o",
            5,
            times=CLOCK,
            sample_valid=valid,
            interval_valid=np.ones(3, dtype=bool),
        )
        attribute = "object_vectors"
    data, _ = _convert_overlays_to_data([Adapter(source)], CLOCK, len(CLOCK), env)
    prepared = getattr(data, attribute)[0]
    if kind == "bodypart":
        values = prepared.bodyparts["nose"]
        intervals = prepared.interval_valid["nose"]
    else:
        key = "data" if kind == "heading" else "animal_positions"
        values = getattr(prepared, key)
        intervals = prepared.interval_valid
    assert np.isnan(values[1]).all()
    assert np.isfinite(points).all()
    # The rejected row is crossed like a deleted one.
    np.testing.assert_array_equal(intervals, [True, True, True])
    assert not np.shares_memory(values, points)


@pytest.mark.parametrize("case", ["sample", "interval", "both"])
def test_scene_trails_and_markers_follow_effective_support(env, case):
    samples, intervals, expected_intervals = position_masks(case)
    source = PositionData(
        POINTS.copy(),
        "red",
        5,
        4,
        times=CLOCK,
        sample_valid=samples,
        interval_valid=intervals,
    )
    expected_pairs = [POINTS[pair] for pair in joined_rows(samples, expected_intervals)]
    with Animation(
        env, np.ones((4, env.n_bins)), frame_times=CLOCK, overlays=[Adapter(source)]
    ) as scene:
        scene.render_frame(CLOCK[-1], dpi=25)
        from matplotlib.collections import LineCollection

        actual = [
            segment
            for lines in scene._renderer.figure.axes[0].collections
            if isinstance(lines, LineCollection)
            for segment in lines.get_segments()
        ]
        assert len(actual) == len(expected_pairs)
        for pair, expected in zip(actual, expected_pairs, strict=True):
            np.testing.assert_array_equal(pair, expected)
        # Exact finite observations survive beside intervals that reject interpolation.
        aligned = scene.overlay_data(CLOCK).positions[0]
        np.testing.assert_array_equal(aligned.interval_valid, expected_intervals)
        expected = scene._overlay_data.positions[0].data.copy()
        if samples is not None:
            expected[1] = (POINTS[0] + POINTS[2]) / 2
        np.testing.assert_allclose(aligned.data, expected)
        # A rejected sample is not drawn: the marker sits between its neighbors.
        image = scene.render_frame(CLOCK[1], dpi=50)
        col, row = scene._renderer.figure.axes[0].transData.transform(POINTS[1])
        pixel = image[int(image.shape[0] - row), int(col)]
        drawn = np.array_equal(pixel, [255, 0, 0])
        assert drawn == (samples is None)


@pytest.mark.parametrize("case", ["sample", "interval", "both"])
def test_track_paths_do_not_draw_rejected_samples(case):
    env = Environment.from_grid_mask(np.ones(2, dtype=bool), (np.arange(3.0),))
    samples, intervals, expected_intervals = position_masks(case)
    source = PositionData(
        POINTS[:, :1].copy(),
        "red",
        5,
        4,
        times=CLOCK,
        sample_valid=samples,
        interval_valid=intervals,
    )
    with Animation(
        env, np.ones((4, env.n_bins)), frame_times=CLOCK, overlays=[Adapter(source)]
    ) as scene:
        window = PreparedScene(scene).track_window(scene.fields["field"], CLOCK[-1])
        expected_times = [
            CLOCK[pair] for pair in joined_rows(samples, expected_intervals)
        ]
        assert len(window.paths) == len(expected_times)
        assert window.path_colors == ["red"] * len(expected_times)
        for segment, expected in zip(window.paths, expected_times, strict=True):
            np.testing.assert_array_equal(segment[:, 0], expected)


@pytest.mark.parametrize("mask_kind", ["sample_valid", "interval_valid", "both"])
def test_partial_bodypart_masks_infer_missing_entries_and_realign(env, mask_kind):
    masks = {}
    if mask_kind in ("sample_valid", "both"):
        masks["sample_valid"] = {"nose": np.array([True, False, True, True])}
    if mask_kind in ("interval_valid", "both"):
        masks["interval_valid"] = {"nose": np.array([False, False, True])}
    tail = POINTS.copy()
    tail[2] = np.nan
    source = BodypartData(
        {"nose": POINTS.copy(), "tail": tail}, None, None, times=CLOCK, **masks
    )
    with Animation(
        env, np.ones((4, env.n_bins)), frame_times=CLOCK, overlays=[Adapter(source)]
    ) as scene:
        prepared = scene._overlay_data.bodypart_sets[0]
        assert set(prepared.sample_valid) == {"nose", "tail"}
        assert set(prepared.interval_valid) == {"nose", "tail"}
        np.testing.assert_array_equal(
            prepared.sample_valid["tail"], [True, True, False, True]
        )
        # Without a mask, the missing tail row is crossed like a deleted one.
        np.testing.assert_array_equal(
            prepared.interval_valid["tail"], [True, True, True]
        )
        # Resampling exercises each part's independent support and missing mask entries.
        aligned = scene.overlay_data([0.05, 0.15, 0.25]).bodypart_sets[0]
        np.testing.assert_allclose(
            aligned.bodyparts["tail"],
            [
                POINTS[:2].mean(axis=0),
                0.75 * POINTS[1] + 0.25 * POINTS[3],
                0.25 * POINTS[1] + 0.75 * POINTS[3],
            ],
        )
        if mask_kind == "sample_valid":
            np.testing.assert_allclose(
                aligned.bodyparts["nose"][:2],
                [
                    0.75 * POINTS[0] + 0.25 * POINTS[2],
                    0.25 * POINTS[0] + 0.75 * POINTS[2],
                ],
            )
        else:
            assert np.isnan(aligned.bodyparts["nose"][:2]).all()
        np.testing.assert_allclose(
            aligned.bodyparts["nose"][2], POINTS[2:].mean(axis=0)
        )
    for mask in masks.values():
        assert set(mask) == {"nose"}


@pytest.mark.parametrize("mask_kind", ["sample_valid", "interval_valid"])
@pytest.mark.parametrize("malformed", ["unknown", "shape", "dtype", "mapping"])
def test_bodypart_mask_errors_are_actionable(env, mask_kind, malformed):
    count = 4 if mask_kind == "sample_valid" else 3
    mask = {"nose": np.ones(count, dtype=bool)}
    if malformed == "unknown":
        mask = {"ghost": np.ones(count, dtype=bool)}
        message = "unknown bodyparts.*ghost"
    elif malformed == "shape":
        mask = {"nose": np.ones(count + 1, dtype=bool)}
        message = "boolean array with shape"
    elif malformed == "dtype":
        mask = {"nose": np.ones(count)}
        message = "boolean array with shape"
    else:
        mask = np.ones(count, dtype=bool)
        message = "map bodypart names"
    with pytest.raises(ValueError, match=message):
        BodypartData(
            {"nose": POINTS.copy()}, None, None, times=CLOCK, **{mask_kind: mask}
        )
