"""Track graph attributes that cannot be saved are refused where they come in."""

from __future__ import annotations

import pickle
import threading

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


def _track(where: str, key: str, value: object) -> Environment:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    if where == "node":
        graph.nodes[0][key] = value
    elif where == "edge":
        graph.edges[0, 1][key] = value
    else:
        graph.graph[key] = value
    return Environment.from_graph(
        graph, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
    )


@pytest.mark.parametrize(
    ("where", "key", "value", "element"),
    [
        ("node", "callback", lambda x: x, "node 0"),
        ("edge", "lock", threading.Lock(), "edge (0, 1)"),
    ],
    ids=["lambda", "lock"],
)
def test_unpicklable_attribute_is_refused_at_construction(where, key, value, element):
    with pytest.raises(TypeError) as error:
        _track(where, key, value)

    message = str(error.value)
    assert f"Track graph {element} attribute {key!r}" in message
    assert "pickle" in message and "to_file" in message
    assert message.splitlines()[-1].startswith(f"Fix: remove {key!r}")


JSON_ONLY = pytest.mark.parametrize(
    ("where", "key", "value", "element", "reason"),
    [
        ("edge", "recorded", np.datetime64("2020-01-01"), "edge (0, 1)", "date value"),
        ("node", "labels", {"start", "reward"}, "node 0", "set value"),
        ("metadata", "baseline", float("nan"), "metadata", "non-finite number"),
    ],
    ids=["datetime", "set", "nan"],
)


@JSON_ONLY
def test_json_only_refusal_names_the_attribute_and_keeps_pickle(
    where, key, value, element, reason, tmp_path
):
    env = _track(where, key, value)

    restored = pickle.loads(pickle.dumps(env))
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    for write in (env.to_dict, lambda: env.to_file(tmp_path / "track")):
        with pytest.raises(ValueError) as error:
            write()
        message = str(error.value)
        assert f"track graph {element} attribute {key!r}" in message
        assert f"value ({type(value).__name__}) is or contains a {reason}" in message
        assert "JSON" in message


class _RefusesPickle:
    def __reduce__(self):
        raise ValueError("this object refuses to be pickled")


def test_any_pickling_failure_is_refused_with_the_attribute_named():
    with pytest.raises(
        TypeError, match="Track graph node 0 attribute 'probe'"
    ) as error:
        _track("node", "probe", _RefusesPickle())

    assert isinstance(error.value.__cause__, ValueError)
