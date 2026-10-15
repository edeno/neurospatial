"""A track's topology supports projection; containment also needs its width."""

import pickle
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.transforms import AffineND


def _track(width):
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    return Environment.from_graph(graph, [(0, 1)], 0.0, 2.0, track_width=width)


@pytest.mark.parametrize("factory", ["from_graph", "linear_track", "maze"])
def test_unknown_track_footprint_refuses_containment_but_retains_projection(factory):
    if factory == "from_graph":
        env = _track(None)
    elif factory == "linear_track":
        env = Environment.linear_track(endpoints=[(0, 0), (10, 0)], bin_size=2.0)
    else:
        env = Environment.maze(
            "t",
            node_positions={
                "start": (0, 0),
                "junction": (0, 10),
                "left": (-10, 10),
                "right": (10, 10),
            },
            bin_size=2.0,
        )
    probes = np.array([[1.0, 0.0], [1000.0, 1000.0]])
    assert (env.bin_at(probes) >= 0).all()
    with pytest.raises(NotImplementedError, match=r"track_width.*bin_at"):
        env.contains(probes)


@pytest.mark.parametrize("width", [None, 2.0])
@pytest.mark.parametrize("derivation", ["native", "subset", "transform"])
@pytest.mark.parametrize("storage", ["copy", "pickle", "dict", "file", "nwb"])
def test_track_containment_contract_survives_derivation_and_storage(
    tmp_path, width, derivation, storage
):
    env = _track(width)
    probes = np.array([[1.0, 0.75], [9.0, 0.0], [5.0, 1000.0]])
    expected = np.array([True, True, False])
    if derivation == "subset":
        env = env.subset(bins=env.bin_centers[:, 0] < 5)
        expected[1] = False
    elif derivation == "transform":
        matrix = np.array([[0.0, -2.0, 20.0], [2.0, 0.0, -10.0], [0.0, 0.0, 1.0]])
        env = env.apply_transform(AffineND(matrix))
        probes = probes @ matrix[:2, :2].T + matrix[:2, 2]

    assigned = env.bin_at(probes)
    if storage == "copy":
        restored = env.copy()
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
        np.testing.assert_array_equal(candidate.bin_at(probes), assigned)
        if width is None:
            with pytest.raises(NotImplementedError, match=r"track_width.*bin_at"):
                candidate.contains(probes)
        else:
            np.testing.assert_array_equal(candidate.contains(probes), expected)


@pytest.mark.parametrize("factory", ["linear_track", "maze"])
def test_track_presets_accept_a_declared_width(factory):
    if factory == "linear_track":
        env = Environment.linear_track(
            endpoints=[(0, 0), (10, 0)], bin_size=2.0, track_width=2.0
        )
    else:
        env = Environment.maze(
            "t",
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
