"""``with_track_width`` gives a track without a width one, keeping its bins.

A track built without ``track_width``, such as one saved by an earlier
version, refuses 2-D position lookup. With a width it must behave exactly like
the same track built with that width: the same bins, the same lookup and the
same rate maps.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from neurospatial import Environment, Regions
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.transforms import AffineND

_NODES = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)]  # cm, an L-shaped track
_WIDTH = 10.0  # cm


def _track(track_width=None):
    regions = Regions()
    regions.add("reward", point=(100.0, 50.0))
    return Environment.linear_track(
        node_positions=_NODES,
        bin_size=2.0,
        edge_spacing=5.0,
        track_width=track_width,
        name="L track",
        units="cm",
        frame="room",
        regions=regions,
    )


@pytest.fixture(scope="module")
def session():
    """Positions along the track, some 2-4 cm off its centerline, and spikes."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 120.0, 1 / 30)  # s
    along = np.abs((times % 40.0) - 20.0) / 20.0 * 150.0  # 0-150 cm and back
    positions = np.where(
        (along < 100.0)[:, None],
        np.c_[along, np.zeros_like(along)],
        np.c_[np.full_like(along, 100.0), along - 100.0],
    )
    positions = positions + rng.uniform(-4.0, 4.0, positions.shape)
    positions[::50] += [-40.0, 40.0]  # tracking jumps, 40 cm off the track
    spike_times = np.sort(np.random.default_rng(1).uniform(0.0, 120.0, 400))
    return times, positions, spike_times


def _read_back(env, how, tmp_path):
    if how == "memory":
        return env
    if how == "file":
        env.to_file(tmp_path / "track")
        return Environment.from_file(tmp_path / "track")
    if how == "dict":
        return Environment.from_dict(env.to_dict())
    pynwb = pytest.importorskip("pynwb")
    nwbfile = pynwb.NWBFile(
        session_description="track",
        identifier="track",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    env.to_nwb(nwbfile, name="track")
    return Environment.from_nwb(nwbfile, name="track")


@pytest.mark.parametrize("how", ["memory", "file", "dict", "nwb"])
def test_widthless_track_with_a_width_matches_one_built_with_it(how, session, tmp_path):
    times, positions, spike_times = session
    widthless = _read_back(_track(), how, tmp_path)
    with pytest.raises(NotImplementedError, match="with_track_width"):
        widthless.bin_at(positions)

    env = widthless.with_track_width(_WIDTH)
    built = _track(track_width=_WIDTH)

    np.testing.assert_array_equal(env.bin_centers, widthless.bin_centers)
    np.testing.assert_array_equal(env.edge_pairs, widthless.edge_pairs)
    np.testing.assert_array_equal(env.bin_centers, built.bin_centers)
    assert env.is_linearized_track
    assert (env.name, env.units, env.frame) == ("L track", "cm", "room")
    assert list(env.regions) == ["reward"]
    assert env.arena.width == _WIDTH
    assert env.derivations == ("with_track_width",)

    bins = env.bin_at(positions)
    np.testing.assert_array_equal(bins, built.bin_at(positions))
    assert (bins[::50] == -1).all()  # the jumps are 40 cm off the track
    assert (np.delete(bins, np.s_[::50]) >= 0).all()  # within 5 cm of it
    np.testing.assert_array_equal(
        env.to_linear(positions[bins >= 0]), built.to_linear(positions[bins >= 0])
    )
    np.testing.assert_array_equal(
        env.occupancy(times, positions), built.occupancy(times, positions)
    )
    rate = compute_spatial_rate(env, spike_times, times, positions)
    np.testing.assert_array_equal(
        rate.firing_rate,
        compute_spatial_rate(built, spike_times, times, positions).firing_rate,
    )
    assert np.isfinite(rate.firing_rate).any()


def test_the_width_is_saved(tmp_path):
    env = _track().with_track_width(_WIDTH)
    env.to_file(tmp_path / "track")
    loaded = Environment.from_file(tmp_path / "track")
    assert loaded.arena.width == _WIDTH
    np.testing.assert_array_equal(
        loaded.bin_at(np.array([[50.0, 4.0], [50.0, 6.0]])), [24, -1]
    )


def test_a_subset_keeps_its_bins():
    widthless = _track()
    subset = widthless.subset(mask=widthless.bin_centers[:, 0] < 30.0)
    env = subset.with_track_width(_WIDTH)
    np.testing.assert_array_equal(env.bin_centers, subset.bin_centers)
    assert env.derivations == ("subset", "with_track_width")
    np.testing.assert_array_equal(
        env.bin_at(np.array([[11.0, 3.0], [11.0, 30.0], [60.0, 0.0]])),
        [5, -1, -1],
    )


def test_a_transformed_track_takes_the_width_in_its_own_units():
    # Rotate by 90 degrees and double the size: a 10 cm track is now 20 wide.
    matrix = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -3.0], [0.0, 0.0, 1.0]])
    track = _track()
    transformed = track.apply_transform(AffineND(matrix), units=track.units)
    env = transformed.with_track_width(2 * _WIDTH)
    built = _track(track_width=_WIDTH)
    on_and_off = np.array([[50.0, 4.0], [50.0, 6.0], [100.0, 30.0]])
    mapped = on_and_off @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_array_equal(env.bin_at(mapped), built.bin_at(on_and_off))


def test_a_new_width_replaces_the_old_one():
    narrow = _track(track_width=_WIDTH).with_track_width(2.0)
    np.testing.assert_array_equal(
        narrow.bin_at(np.array([[50.0, 0.5], [50.0, 4.0]])), [24, -1]
    )


def test_an_open_field_is_refused():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=5.0
    )
    with pytest.raises(TypeError) as caught:
        env.with_track_width(_WIDTH)
    message = str(caught.value)
    assert "not a linearized track" in message
    assert "Why: " in message
    assert "Fix: " in message and "track_width=" in message


@pytest.mark.parametrize("width", [0.0, -5.0, np.nan, np.inf, True, None, "10"])
def test_a_width_that_is_not_a_positive_number_is_refused(width):
    with pytest.raises(ValueError, match="finite positive width") as caught:
        _track().with_track_width(width)
    assert "Fix: " in str(caught.value)
