"""Private globals in older environment pickles still restore owned values."""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment import _publication


def test_older_publication_pickle_paths_preserve_owned_state(monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4.0),) * 2,
        name="before-publication-extraction",
        units="cm",
    )
    payload = env.to_dict()
    payload["graph"]["graph"]["recording"] = {"tags": ["original"]}
    env = Environment.from_dict(payload)
    env.compute_kernel(0.5)

    # Produce genuine pickle GLOBAL references to the former class owners,
    # without keeping a binary fixture coupled to NumPy's pickle format.
    with monkeypatch.context() as legacy:
        for name in (
            "_PublishedGeometry",
            "_PublishedGraph",
            "_ReadOnlyGraphAttributes",
        ):
            legacy.setattr(
                getattr(_publication, name),
                "__module__",
                "neurospatial.environment.core",
            )
        for name in ("_Metadata", "_Unset"):
            legacy.setattr(
                getattr(_publication, name),
                "__module__",
                "neurospatial.environment._metadata",
            )
        old_pickle = pickle.dumps((env, _publication.UNSET))

    restored, sentinel = pickle.loads(old_pickle)

    assert sentinel is _publication.UNSET
    assert (restored.name, restored.units) == (env.name, env.units)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.compute_kernel(0.5), env.compute_kernel(0.5))
    assert restored._kernel_cache is not env._kernel_cache
    for array in (restored.bin_centers, restored.grid_edges[0], restored.active_mask):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(nx.NetworkXError):
        restored.connectivity.add_node(99)
    with pytest.raises(TypeError, match="read-only"):
        restored.connectivity.nodes[0]["pos"] = (99.0, 99.0)
    exported = restored.connectivity.graph
    exported["recording"]["tags"].append("caller")
    assert restored.connectivity.graph["recording"]["tags"] == ["original"]
