from __future__ import annotations

import dataclasses
import io
import os
import pickle
import sys
import types

import networkx as nx
import numpy as np
import pytest
from load_rosenberg_labyrinth import _Traj, _TrajUnpickler, maze_graph
from load_yartsev_bat_demo import _drop_close_spikes, _pool


def _load(data: bytes) -> object:
    return _TrajUnpickler(io.BytesIO(data)).load()


def test_trajectory_unpickler_reads_float_and_int_arrays(monkeypatch):
    # Pickle a Traj the way the repository does: a dataclass in MM_Traj_Utils.
    module = types.ModuleType("MM_Traj_Utils")
    traj_class = dataclasses.make_dataclass("Traj", ["fr", "ce", "ke", "no", "re"])
    traj_class.__module__ = "MM_Traj_Utils"
    module.Traj = traj_class
    monkeypatch.setitem(sys.modules, "MM_Traj_Utils", module)
    traj = traj_class(
        fr=np.array([[0, 2]]),
        ce=[np.array([3, 4])],
        ke=[np.array([[0.1, 0.2], [np.nan, 0.3]])],
        no=None,
        re=None,
    )
    loaded = _load(pickle.dumps(traj))
    assert isinstance(loaded, _Traj)
    np.testing.assert_array_equal(loaded.fr, traj.fr)
    np.testing.assert_array_equal(loaded.ke[0], traj.ke[0])


def test_trajectory_unpickler_refuses_other_globals():
    with pytest.raises(pickle.UnpicklingError, match=r"refusing to load posix\.getcwd"):
        _load(pickle.dumps(os.getcwd))


def test_trajectory_unpickler_refuses_object_dtypes():
    # A structured dtype with an object field is how a stream could read raw
    # bytes as object pointers; only built-in i8/f8 are resolved.
    data = pickle.dumps(np.zeros(1, dtype=[("a", "O")]))
    with pytest.raises(pickle.UnpicklingError, match="refusing dtype"):
        _load(data)


def test_maze_graph_matches_the_binary_tree():
    graph = maze_graph()
    assert nx.is_tree(graph)
    degrees = [degree for _, degree in graph.degree()]
    assert graph.number_of_nodes() == 128  # 127 run ends + the entrance
    assert degrees.count(3) == 63
    assert degrees.count(1) == 65  # 64 dead ends + the entrance
    assert graph.nodes["entrance"]["pos"] == (0.0, 7.0)
    assert graph.nodes[0]["pos"] == (7.0, 7.0)  # first junction at the centre
    total = sum(d for _, _, d in graph.edges(data="distance"))
    assert total == 7 + 2 * 4 + 4 * 4 + 8 * 2 + 16 * 2 + 32 * 1 + 64 * 1


def test_close_spikes_dropped_against_their_predecessor():
    # Like the record's script: a spike within 1 ms of the one before it in
    # the train is dropped, even if that one was itself dropped.
    spikes = np.array([0.0, 0.0006, 0.0012, 0.0030])
    np.testing.assert_array_equal(_drop_close_spikes(spikes), [0.0, 0.0030])


def test_pooled_units_keep_the_first_id_and_are_sorted():
    trains = {5: np.array([0.0, 2.0]), 9: np.array([1.0, 2.0005]), 11: np.array([3.0])}
    pooled = _pool(trains, [[5, 9]])
    assert sorted(pooled) == [5, 11]
    np.testing.assert_array_equal(pooled[5], [0.0, 1.0, 2.0])
