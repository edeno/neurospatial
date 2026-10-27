"""Composition records preserve child lookup and validate stored identities."""

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment._fields import face_geometry
from neurospatial.geometry._binning import KnownFaces
from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._composition import CombinedLocator


def snapshot():
    children = tuple(
        Environment.from_grid_mask(
            np.ones((1, 1), dtype=bool),
            grid_edges=(np.array([shift, shift + 1.0]), np.array([0.0, 1.0])),
        )
        for shift in (0.0, 3.0)
    )
    return CombinedLocator(
        ("left", "right"),
        tuple(
            replace(
                env._binning, faces=KnownFaces(face_geometry(env._cache, env._binning))
            )
            for env in children
        ),
        tuple(env.arena for env in children),
        np.array([[0, 1]], dtype=np.intp),
        np.array([3.0]),
        np.array([0.0]),
    )


@pytest.mark.parametrize(
    "restore",
    [
        lambda x: x,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: decode(*encode(x, "root"), "root"),
    ],
)
def test_lookup_keeps_child_support_and_nonfinite_sentinels(restore):
    value = restore(snapshot())
    np.testing.assert_array_equal(
        value.locate(
            np.array([[0.5, 0.5], [3.5, 0.5], [2.0, 0.5], [np.nan, 0.0], [np.inf, 0.0]])
        ),
        [0, 1, -1, -1, -1],
    )
    with pytest.raises(ValueError, match="shape"):
        value.locate(np.zeros((1, 3)))
    assert not value.bridge_pairs.flags.writeable
    assert not value.bridge_conductances.flags.writeable
    np.testing.assert_array_equal(value.bin_connections().pairs, [[0, 1]])


@pytest.mark.parametrize(
    "change",
    [
        {"keys": ("same", "same")},
        {"arenas": ()},
        {"bridge_pairs": np.array([[0, 0]])},
        {"bridge_conductances": np.array([-1.0])},
        {"bridge_distances": np.array([np.inf])},
    ],
)
def test_malformed_composition_refuses(change):
    with pytest.raises(ValueError):
        replace(snapshot(), **change)


def test_combined_snapshot_publishes_and_round_trips(tmp_path):
    from neurospatial.binning import Binning, MissingFaces
    from neurospatial.geometry._arena import _CombinedArena
    from neurospatial.geometry._polygon import cells_from_locator
    from neurospatial.geometry._polygon_cells import PolygonCells
    from neurospatial.testing import assert_environment_contract

    source = snapshot()
    env = Environment.from_bins(
        Binning(
            "combined",
            source.spatial_bins(),
            source.bin_connections(),
            source,
            MissingFaces("Bridges have no physical faces."),
            shapes=PolygonCells(cells_from_locator(source)),
        ),
        arena=_CombinedArena(source.arenas),
        units="cm",
        frame="lab",
    )
    assert_environment_contract(env, [[0.5, 0.5], [3.5, 0.5], [2.0, 0.5]], [0, 1, -1])
    path = tmp_path / "combined.json"
    env.to_file(path)
    restored = Environment.from_file(path)
    assert restored.arena.domain_known
    assert restored.arena.outline is None
    source.children[0].bins.centers.setflags(write=True)
    source.children[0].bins.centers[:] = 100
    np.testing.assert_array_equal(env.bin_at([[0.5, 0.5]]), [0])
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


def test_publication_rejects_stale_parent_geometry():
    from neurospatial.binning import BinConnections, Binning, MissingFaces

    source = snapshot()
    links = source.bin_connections()
    wrong = BinConnections.from_arrays(links.all_pairs, [99.0])
    with pytest.raises(ValueError, match="combined path distances"):
        Environment.from_bins(
            Binning(
                "combined",
                source.spatial_bins(),
                wrong,
                source,
                MissingFaces("unavailable"),
            )
        )


def test_nested_child_faces_cannot_disagree_with_child_measures():
    child = snapshot().children[0]
    bad_faces = child.build_faces()._replace(volumes=np.array([2.0]))
    # The child binning itself refuses faces whose volumes are not its sizes.
    with pytest.raises(ValueError, match=r"faces\.volumes"):
        replace(child, faces=KnownFaces(bad_faces))


def test_nested_arena_depth_is_bounded():
    from neurospatial.geometry._codecs import decode_arena

    meta = {
        "codec": "UnknownArena",
        "codec_version": 1,
        "n_dims": 2,
        "coordinate_kind": "cartesian",
    }
    for _ in range(600):
        meta = {"codec": "_CombinedArena", "codec_version": 1, "children": [meta]}
    with pytest.raises(ValueError, match="nesting"):
        decode_arena(meta, {})
