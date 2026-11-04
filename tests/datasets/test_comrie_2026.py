"""Comrie et al. (2026), Zenodo 23020757: a synthetic file in its layout, and the real one."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._catalog import sessions
from neurospatial.datasets._checks import run_checks

DEFAULT = "j1620210710_02_r1_figure04_inputs"
#: Planted offset of head_orientation from the direction of movement.
OFFSET = np.radians(13.0)
#: Arm length of the synthetic three-arm star: its arms' ends span 183.6 cm,
#: the layout the preprint's hallways give.
ARM = 106.0


def _entry():
    return next(entry for entry in sessions("comrie_2026") if entry.session == DEFAULT)


def _star():
    """Three arms from a centre node 3, at 90, 210 and 330 degrees."""
    angles = np.radians([90.0, 210.0, 330.0])
    ends = ARM * np.column_stack([np.cos(angles), np.sin(angles)])
    nodes = np.vstack([ends, [0.0, 0.0]]) + 120.0
    edges = np.array([[3, 0], [3, 1], [3, 2]])
    return nodes, edges


def _write(path, *, edge_order=None, shuffle_speed=False):
    """A file in the Zenodo record's layout: 60 s at 500 Hz along the arms.

    The head runs out and back along each arm at about 40 cm/s, 1.5 cm to
    the left of it, and stops at each end for 2 s.
    """
    rng = np.random.default_rng(0)
    nodes, edges = _star()
    rate = 500.0
    legs = []
    for arm in range(3):
        start, end = nodes[3], nodes[arm]
        along = end - start
        normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
        out = np.linspace(0, 1, int(rate * ARM / 40.0))[:, None]
        hold = np.ones((int(rate * 2.0), 1))
        share = np.vstack([out, hold, out[::-1]])
        legs.append(start + share * along + 1.5 * normal)
    positions = np.vstack(legs)
    times = 1.6259e9 + np.arange(len(positions)) / rate
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    speed = np.r_[step[0], step]
    if shuffle_speed:
        speed = rng.permutation(speed)
    direction = np.arctan2(*np.diff(positions, axis=0)[:, ::-1].T)
    orientation = np.angle(np.exp(1j * (np.r_[direction, direction[-1]] + OFFSET)))
    trains = [np.sort(rng.uniform(times[0], times[-1], n)) for n in (300, 0, 150)]
    order = edges if edge_order is None else edge_order
    np.savez(
        path,
        position_time=times,
        **{
            "position/head_position_x": positions[:, 0],
            "position/head_position_y": positions[:, 1],
            "position/head_speed": speed,
            "position/head_orientation": orientation,
            "position/linear_position": np.arange(len(times), dtype=float),
        },
        spike_times=np.concatenate(trains),
        spike_offsets=np.cumsum([0, *map(len, trains)]),
        track_nodes=np.arange(4),
        track_node_positions=nodes,
        track_edges=edges,
        track_edge_distance=np.full(3, ARM),
        linear_edge_order=np.asarray(order),
        linear_edge_spacing=np.float64(15.0),
    )
    return times, positions, orientation, trains


def test_loads_a_file_in_the_records_layout(tmp_path):
    from neurospatial.datasets import comrie_2026

    path = tmp_path / "j16.npz"
    times, positions, orientation, trains = _write(path)
    entry = _entry()
    rec = run_checks(comrie_2026.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_array_equal(rec.positions, positions)
    assert rec.position_units == "cm"
    for got, planted in zip(rec.spike_times, trains, strict=True):
        np.testing.assert_array_equal(got, planted)
    assert list(rec.units["n_spikes"]) == [300, 0, 150]
    np.testing.assert_array_equal(rec.linearized, np.arange(len(times)))
    # head_orientation is kept for comparison, never used as headings.
    assert rec.headings is None
    np.testing.assert_array_equal(rec.params["head_orientation"], orientation)
    (heading,) = [text for text in rec.watch_outs if text.startswith("No headings")]
    assert "circular mean of +13.0 degrees" in heading
    # Every edge is kept; the env is the declared graph.
    assert rec.env_kind == "declared"
    assert rec.env.is_linearized_track
    np.testing.assert_allclose(rec.env.bin_sizes.sum(), 3 * ARM)
    bases = {c.text.split(":")[0]: c.basis for c in rec.conversions}
    assert bases["positions"] == "inferred"


def test_beyond_hallway_share_is_split_by_speed(tmp_path):
    """The planted head runs 1.5 cm off the centreline: inside half of 6.5 cm."""
    from neurospatial.behavior import compute_speed
    from neurospatial.datasets import comrie_2026

    path = tmp_path / "j16.npz"
    times, positions, *_ = _write(path)
    shifted = positions.copy()
    shifted[::4] += 3.0  # a quarter of the samples pushed 3 cm further out
    with np.load(path, allow_pickle=False) as stored:
        arrays = dict(stored)
    arrays["position/head_position_x"] = shifted[:, 0]
    arrays["position/head_position_y"] = shifted[:, 1]
    step = np.linalg.norm(np.diff(shifted, axis=0), axis=1) / np.diff(times)
    arrays["position/head_speed"] = np.r_[step[0], step]
    np.savez(path, **arrays)
    rec = comrie_2026.load_session(_entry(), path)
    nodes, edges = _star()
    # Independent distance to the three arms.
    distance = np.full(len(shifted), np.inf)
    for a, b in edges:
        along = nodes[b] - nodes[a]
        t = np.clip((shifted - nodes[a]) @ along / (along @ along), 0, 1)
        distance = np.minimum(
            distance, np.linalg.norm(shifted - (nodes[a] + t[:, None] * along), axis=1)
        )
    beyond = distance > 3.25
    stopped = compute_speed(times, shifted) < 4.0
    (text,) = [w for w in rec.watch_outs if "hallway width" in w]
    assert text.startswith(
        f"{beyond.mean():.1%} of tracked head samples lie farther than half the "
        f"6.5 cm hallway width from the track graph: {beyond[stopped].mean():.1%} "
        f"while stopped (below 4 cm/s) and {beyond[~stopped].mean():.1%} while moving"
    )
    assert "tracking spread, not misfit" in text


def test_an_edge_left_out_of_the_order_is_refused(tmp_path):
    from neurospatial.datasets import comrie_2026

    path = tmp_path / "j16.npz"
    _write(path, edge_order=np.array([[3, 0], [3, 1]]))
    with pytest.raises(ValueError, match="lists 2 edges, not the 3 edges"):
        comrie_2026.load_session(_entry(), path)


def test_head_speed_out_of_step_with_the_positions_is_refused(tmp_path):
    from neurospatial.datasets import comrie_2026

    path = tmp_path / "j16.npz"
    _write(path, shuffle_speed=True)
    with pytest.raises(ValueError, match="head_speed no longer matches"):
        comrie_2026.load_session(_entry(), path)


@pytest.mark.realdata
def test_loads_default_session():
    """j16 02_r1 as the survey expected it: 203 units, 870,018 spikes, 500 Hz."""
    rec = load("comrie_2026")
    assert rec.session == DEFAULT
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == _entry().n_units == 203
    assert int(rec.units["n_spikes"].sum()) == 870_018
    assert rec.times.size == 709_321
    assert rec.n_dims == 2
    assert 499.0 <= rec.sampling_rate <= 501.0
    graph = rec.env.connectivity
    assert rec.env.n_bins == graph.number_of_nodes()
    assert rec.env.is_linearized_track
    # The authors' graph, every edge kept: env.to_linear is their linearization.
    placed = np.asarray(rec.env.bin_at(rec.positions)) >= 0
    np.testing.assert_allclose(
        rec.env.to_linear(rec.positions[placed]), rec.linearized[placed], atol=1e-9
    )
    orientation = rec.params["head_orientation"]
    assert orientation.min() < -3.0
    assert orientation.max() > 3.0
    assert rec.headings is None
    assert any("+13.1 degrees" in text for text in rec.watch_outs)
    assert rec.provenance.sha256 == _entry().sha256
