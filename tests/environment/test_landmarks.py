"""Landmarks: named cues, monitors and objects, stored with the environment."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from neurospatial import Environment
from neurospatial.encoding import (
    compute_egocentric_rate,
    compute_object_vector_rate,
    compute_object_vector_rates,
    object_vector_cell_significance,
)
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.ops.transforms import translate

MONITOR = LineString([(10.0, 100.0), (90.0, 100.0)])  # along the north wall


def _arena(**options):
    edges = np.arange(0.0, 101.0, 5.0)
    return Environment.from_grid_mask(
        np.ones((20, 20), dtype=bool), grid_edges=(edges, edges), units="cm", **options
    )


@pytest.fixture(scope="module")
def landmarks():
    return Landmarks(
        [
            Landmark("cue_card", "point", [50.0, 105.0], {"role": "cue"}),
            Landmark("monitor", "line", MONITOR),
            Landmark("box", "polygon", box(20, 20, 30, 30)),
        ]
    )


@pytest.fixture(scope="module")
def env(landmarks):
    return _arena().with_landmarks(landmarks)


@pytest.fixture(scope="module")
def recording():
    # A random walk over the arena, and a cell firing near the monitor.
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 900.0, 1 / 30)
    walk = np.cumsum(rng.normal(0, 1.5, (times.size, 2)), axis=0) % 200
    positions = np.where(walk > 100, 200 - walk, walk)
    distance = np.array([MONITOR.distance(Point(p)) for p in positions])
    rate = np.where(distance < 10, 20.0, 1.0)
    spike_rng = np.random.default_rng(1)
    spike_times = times[spike_rng.random(times.size) < rate / 30]
    return times, positions, spike_times


# --- the records -------------------------------------------------------------


def test_landmark_kinds_and_their_nearest_points(landmarks):
    positions = np.array([[50.0, 80.0], [0.0, 120.0], [25.0, 25.0], [np.nan, 1.0]])
    nearest = landmarks.nearest_points(positions)
    assert nearest.shape == (4, 3, 2)
    np.testing.assert_allclose(nearest[:3, 0], [[50.0, 105.0]] * 3)  # a fixed point
    np.testing.assert_allclose(nearest[:3, 1], [[50, 100], [10, 100], [25, 100]])
    np.testing.assert_allclose(nearest[2, 2], [25.0, 25.0])  # inside the box
    assert np.isnan(nearest[3]).all()  # a missing sample stays missing
    np.testing.assert_allclose(
        landmarks.positions(), [[50.0, 105.0], [50.0, 100.0], [25.0, 25.0]]
    )


def test_near_reports_gaze_on_a_landmark(landmarks):
    gaze = np.array([[50.0, 99.0], [50.0, 95.0], [np.nan, 0.0]])
    np.testing.assert_array_equal(
        landmarks.near(gaze, "monitor", tolerance=2.0), [True, False, False]
    )


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (("", "point", [0.0, 0.0]), "nonempty strings"),
        (("cue", "circle", [0.0, 0.0]), "unknown kind"),
        (("cue", "point", [np.nan, 0.0]), "one finite point"),
        (("monitor", "line", [(0.0, 0.0), (0.0, 0.0)]), "valid, nonempty 2-D line"),
        (("box", "polygon", [(0, 0), (1, 1), (2, 2)]), "valid, nonempty 2-D polygon"),
    ],
)
def test_invalid_landmarks_are_refused(arguments, match):
    with pytest.raises(ValueError, match=match):
        Landmark(*arguments)


def test_collections_refuse_repeated_names_and_name_missing_ones(landmarks):
    with pytest.raises(ValueError, match="Two landmarks are named 'cue'"):
        Landmarks([Landmark("cue", "point", [0, 0]), Landmark("cue", "point", [1, 1])])
    with pytest.raises(ValueError, match="different coordinate dimensions"):
        Landmarks([Landmark("a", "point", [0, 0]), Landmark("b", "point", [1, 1, 1])])
    with pytest.raises(KeyError, match="No landmark is named 'door'"):
        landmarks["door"]


def test_point_landmarks_cannot_be_changed_through_their_array(env):
    derived = env.subset(mask=np.arange(env.n_bins) < 100)
    source = np.array([50.0, 105.0])
    landmark = Landmark("cue", "point", source)
    source[0] = -1.0  # the caller's array is copied
    assert landmark.data.tolist() == [50.0, 105.0]

    data = env.landmarks["cue_card"].data
    with pytest.raises(ValueError, match="WRITEABLE"):
        data.setflags(write=True)
    data.shape = (2, 1)  # each access gets its own header
    for holder in (env, derived):
        assert holder.landmarks["cue_card"].data.shape == (2,)
        assert holder.landmarks["cue_card"].data.tolist() == [50.0, 105.0]


# --- on the environment ------------------------------------------------------


def test_landmarks_follow_derivations(env):
    assert list(env.subset(mask=np.arange(env.n_bins) < 100).landmarks) == list(
        env.landmarks
    )
    moved = env.apply_transform(translate(10.0, 0.0)).landmarks
    np.testing.assert_allclose(moved["cue_card"].data, [60.0, 105.0])
    assert moved["monitor"].data.equals(LineString([(20.0, 100.0), (100.0, 100.0)]))
    assert dict(moved["cue_card"].metadata) == {"role": "cue"}
    combined = Environment.combine(
        {"a": env, "b": env.apply_transform(translate(200.0, 0.0))}, auto_bridge=False
    )
    assert list(combined.landmarks)[:3] == ["a/cue_card", "a/monitor", "a/box"]
    assert "Landmarks: cue_card, monitor, box" in env.info()


def test_landmarks_are_saved_with_the_environment(env, tmp_path):
    env.to_file(tmp_path / "arena")
    for restored in (
        Environment.from_dict(env.to_dict()),
        Environment.from_file(tmp_path / "arena"),
        pickle.loads(pickle.dumps(env)),
    ):
        assert list(restored.landmarks) == list(env.landmarks)
        assert restored.landmarks["monitor"].data.equals(MONITOR)
        assert dict(restored.landmarks["cue_card"].metadata) == {"role": "cue"}


def test_landmarks_are_saved_to_nwb(env):
    from datetime import datetime, timezone

    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="landmarks",
        identifier="landmarks",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    restored = read_environment(nwbfile)
    assert list(restored.landmarks) == list(env.landmarks)
    assert restored.landmarks["box"].data.equals(env.landmarks["box"].data)


def test_landmarks_must_match_the_environment(landmarks):
    with pytest.raises(AttributeError, match="with_landmarks"):
        _arena().landmarks = landmarks  # type: ignore[misc]
    polar = Environment.from_polar_egocentric(
        distance_range=(0, 40),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10,
        angle_bin_size=np.pi / 4,
    )
    with pytest.raises(ValueError, match="polar environment cannot hold landmarks"):
        polar.with_landmarks(landmarks)
    with pytest.raises(ValueError, match="has 3 coordinates; the environment has 2"):
        _arena().with_landmarks(Landmarks([Landmark("high", "point", [1.0, 2.0, 3.0])]))


# --- in encoding -------------------------------------------------------------

_BINS = {"distance_range": (0, 50), "n_distance_bins": 5, "n_direction_bins": 8}


def _rings(result):
    return np.nanmean(np.asarray(result.firing_rate).reshape(5, 8), axis=1)


def test_a_monitor_is_measured_from_its_nearest_edge(env, recording):
    times, positions, spike_times = recording
    by_name = compute_object_vector_rate(
        env, spike_times, times, positions, "monitor", **_BINS
    )
    by_center = compute_object_vector_rate(
        env, spike_times, times, positions, [[50.0, 100.0]], **_BINS
    )
    near, far = _rings(by_name)[0], _rings(by_name)[1:]
    assert near > 15 and np.all(far < 2)  # the cell's 10 cm tuning
    assert _rings(by_center)[1] > 5  # the center point smears it outward
    assert by_name.analysis_settings["objects"] == ["monitor"]
    assert by_center.analysis_settings["objects"] is None


def test_point_landmark_names_match_their_coordinates(env, recording):
    times, positions, spike_times = recording
    by_name = compute_egocentric_rate(
        env, spike_times, times, positions, np.zeros(times.size), "cue_card", **_BINS
    )
    by_value = compute_egocentric_rate(
        env, spike_times, times, positions, np.zeros(times.size), [50.0, 105.0], **_BINS
    )
    np.testing.assert_array_equal(by_name.firing_rate, by_value.firing_rate)


def test_positions_per_sample_match_the_landmark(env, recording):
    times, positions, spike_times = recording
    per_sample = env.landmarks.nearest_points(positions, ["monitor"])
    np.testing.assert_array_equal(
        compute_object_vector_rate(
            env, spike_times, times, positions, per_sample, **_BINS
        ).firing_rate,
        compute_object_vector_rate(
            env, spike_times, times, positions, "monitor", **_BINS
        ).firing_rate,
    )


def test_batch_and_significance_accept_landmark_names(env, recording):
    times, positions, spike_times = recording
    rates = compute_object_vector_rates(
        env,
        [spike_times, spike_times[::2]],
        times,
        positions,
        ("monitor", "box"),
        **_BINS,
    )
    assert rates.analysis_settings["objects"] == ["monitor", "box"]
    shuffles = object_vector_cell_significance(
        env, [spike_times], times, positions, "monitor", n_shuffles=3, rng=0, **_BINS
    )
    assert shuffles is not None


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"object_positions": "door"}, "No landmark is named 'door'"),
        (
            {"object_positions": "monitor", "metric": "geodesic"},
            "nearest points move with the animal",
        ),
    ],
)
def test_landmark_names_are_checked(env, recording, options, match):
    times, positions, spike_times = recording
    with pytest.raises((KeyError, ValueError), match=match):
        compute_object_vector_rate(
            env,
            spike_times,
            times,
            positions,
            options.pop("object_positions"),
            **options,
            **_BINS,
        )


def test_names_need_the_environment_that_holds_them(recording):
    times, positions, spike_times = recording
    with pytest.raises(ValueError, match="no environment was given"):
        compute_object_vector_rate(
            None, spike_times, times, positions, "monitor", **_BINS
        )
