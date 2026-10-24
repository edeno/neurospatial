"""A track's topology supports projection; every position lookup needs its width."""

import pickle
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment, Regions
from neurospatial.ops.transforms import AffineND


def _track(width):
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    return Environment.from_graph(
        graph, [(0, 1)], edge_spacing=0.0, bin_size=2.0, track_width=width
    )


def _widthless(factory):
    if factory == "from_graph":
        return _track(None)
    if factory == "linear_track":
        return Environment.linear_track(endpoints=[(0, 0), (10, 0)], bin_size=2.0)
    return Environment.maze(
        kind="t",
        node_positions={
            "start": (0, 0),
            "junction": (0, 10),
            "left": (-10, 10),
            "right": (10, 10),
        },
        bin_size=2.0,
    )


_POSITION_LOOKUPS = {
    "bin_at": lambda env, probes: env.bin_at(probes),
    "contains": lambda env, probes: env.contains(probes),
    "occupancy": lambda env, probes: env.occupancy(np.array([0.0, 0.1]), probes),
    "bin_sequence": lambda env, probes: env.bin_sequence(np.array([0.0, 0.1]), probes),
    "transitions": lambda env, probes: env.transitions(
        times=np.array([0.0, 0.1]), positions=probes
    ),
    "points=": lambda env, probes: env.distance_to(points=probes[:1]),
}


@pytest.mark.parametrize("lookup", sorted(_POSITION_LOOKUPS))
@pytest.mark.parametrize("factory", ["from_graph", "linear_track", "maze"])
def test_a_track_without_width_refuses_position_lookups(factory, lookup):
    """A centerline cannot say how far off it a position is still on the track."""
    env = _widthless(factory)
    probes = np.array([[1.0, 0.0], [1000.0, 1000.0]])
    with pytest.raises(NotImplementedError, match="without track_width"):
        _POSITION_LOOKUPS[lookup](env, probes)
    # Projection onto the centerline does not need the width.
    assert np.isfinite(env.to_linear(probes[:1])).all()


def test_a_one_dimensional_track_contains_only_its_centerline():
    """Positions beyond a 1-D track's ends are off the track; its ends are on it."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0,))
    graph.add_node(1, pos=(100.0,))
    graph.add_edge(0, 1)
    env = Environment.from_graph(graph, [(0, 1)], bin_size=5.0)
    probes = np.array([[-500.0], [-1e-9], [0.0], [50.0], [100.0], [100.001], [5000.0]])
    bins = env.bin_at(probes)
    np.testing.assert_array_equal(
        bins >= 0, [False, False, True, True, True, False, False]
    )
    np.testing.assert_array_equal(bins[[2, 4]], [0, env.n_bins - 1])
    np.testing.assert_array_equal(env.contains(probes), bins >= 0)
    restored = Environment.from_dict(env.to_dict())
    np.testing.assert_array_equal(restored.bin_at(probes), bins)


def test_a_point_region_on_a_track_without_width_is_refused():
    regions = Regions()
    regions.add("reward", point=(1000.0, 1000.0))
    env = _track(None).with_regions(regions)
    with pytest.raises(NotImplementedError, match="without track_width"):
        env.bins_in_region("reward")


def test_a_track_with_width_gives_far_positions_no_bin():
    env = _track(2.0)
    probes = np.array([[1.0, 0.0], [1.0, 0.99], [1.0, 1.01], [500.0, 500.0]])
    np.testing.assert_array_equal(env.bin_at(probes) >= 0, [True, True, False, False])
    with pytest.raises(ValueError, match="outside the environment"):
        env.distance_to(points=[[500.0, 500.0]])


@pytest.mark.parametrize("width", [None, 2.0])
@pytest.mark.parametrize("derivation", ["native", "subset", "transform"])
@pytest.mark.parametrize("storage", ["derived", "pickle", "dict", "file", "nwb"])
def test_track_containment_contract_survives_derivation_and_storage(
    tmp_path, width, derivation, storage
):
    env = _track(width)
    probes = np.array([[1.0, 0.75], [9.0, 0.0], [5.0, 1000.0]])
    expected = np.array([True, True, False])
    if derivation == "subset":
        env = env.subset(mask=env.bin_centers[:, 0] < 5)
        expected[1] = False
    elif derivation == "transform":
        matrix = np.array([[0.0, -2.0, 20.0], [2.0, 0.0, -10.0], [0.0, 0.0, 1.0]])
        env = env.apply_transform(AffineND(matrix))
        probes = probes @ matrix[:2, :2].T + matrix[:2, 2]

    assigned = None if width is None else env.bin_at(probes)
    if storage == "derived":
        restored = env.with_metadata()
    elif storage == "pickle":
        restored = pickle.loads(pickle.dumps(env))
    elif storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "track")
        restored = Environment.from_file(tmp_path / "track")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            "track footprint",
            "track-footprint",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        write_environment(nwbfile, env)
        with pynwb.NWBHDF5IO(str(tmp_path / "track.nwb"), "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(str(tmp_path / "track.nwb"), "r") as io:
            restored = read_environment(io.read())

    for candidate in (env, restored):
        if width is None:
            for lookup in (candidate.bin_at, candidate.contains):
                with pytest.raises(NotImplementedError, match="without track_width"):
                    lookup(probes)
        else:
            np.testing.assert_array_equal(candidate.bin_at(probes), assigned)
            np.testing.assert_array_equal(candidate.contains(probes), expected)


def test_a_mapped_one_dimensional_track_keeps_its_ends():
    """Rounding in an affine map must not push the track's ends outside."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.1,))
    graph.add_node(1, pos=(10.1,))
    graph.add_edge(0, 1, distance=10.0)
    env = Environment.from_graph(graph, [(0, 1)], edge_spacing=0.0, bin_size=1.0)
    ends = np.array([[0.1], [10.1]])
    np.testing.assert_array_equal(env.bin_at(ends), [0, env.n_bins - 1])

    matrix = np.array([[0.3, 13.7], [0.0, 1.0]])
    mapped = env.apply_transform(AffineND(matrix))
    mapped_ends = ends * 0.3 + 13.7
    np.testing.assert_array_equal(mapped.bin_at(mapped_ends), [0, env.n_bins - 1])
    # One bin width past either end is still outside.
    np.testing.assert_array_equal(
        mapped.bin_at(mapped_ends + np.array([[-0.3], [0.3]])), [-1, -1]
    )


@pytest.mark.parametrize("factory", ["linear_track", "maze"])
def test_track_presets_accept_a_declared_width(factory):
    if factory == "linear_track":
        env = Environment.linear_track(
            endpoints=[(0, 0), (10, 0)], bin_size=2.0, track_width=2.0
        )
    else:
        env = Environment.maze(
            kind="t",
            node_positions={
                "start": (0, 0),
                "junction": (0, 10),
                "left": (-10, 10),
                "right": (10, 10),
            },
            bin_size=2.0,
            track_width=2.0,
        )
    np.testing.assert_array_equal(
        env.contains(np.array([[0.5, 0.5], [1000.0, 1000.0]])), [True, False]
    )
