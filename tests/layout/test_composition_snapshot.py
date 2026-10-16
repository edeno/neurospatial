"""Composition records preserve child lookup and validate stored identities."""

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout._binning import KnownFaces
from neurospatial.layout._codecs import decode, encode
from neurospatial.layout._composition import CombinedLocator


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
            replace(env._binning, faces=KnownFaces(env._face_geometry))
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
    np.testing.assert_array_equal(value.spatial_bins().pairs, [[0, 1]])


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
    from neurospatial.arena import _CombinedArena
    from neurospatial.binning import Binning, MissingFaces
    from neurospatial.layout._polygon import PolygonCells, cells_from_locator
    from neurospatial.testing import assert_environment_contract

    source = snapshot()
    env = Environment.from_bins(
        Binning(
            "combined",
            source.spatial_bins(),
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
    assert not restored.arena.physical_footprint_known
    source.children[0].bins.centers.setflags(write=True)
    source.children[0].bins.centers[:] = 100
    np.testing.assert_array_equal(env.bin_at([[0.5, 0.5]]), [0])
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


def test_publication_rejects_stale_parent_geometry():
    from neurospatial.binning import Binning, MissingFaces

    source = snapshot()
    wrong = replace(source.spatial_bins(), pair_distances=np.array([99.0]))
    with pytest.raises(ValueError, match="combined path distances"):
        Environment.from_bins(
            Binning("combined", wrong, source, MissingFaces("unavailable"))
        )


def test_nested_child_faces_cannot_disagree_with_child_measures():
    from neurospatial.binning import Binning, MissingFaces

    source = snapshot()
    child = source.children[0]
    bad_faces = child.faces.build()._replace(volumes=np.array([2.0]))
    bad = replace(
        source,
        children=(replace(child, faces=KnownFaces(bad_faces)), source.children[1]),
    )
    with pytest.raises(ValueError, match=r"faces\.volumes"):
        Environment.from_bins(
            Binning("combined", bad.spatial_bins(), bad, MissingFaces("bridge"))
        )


def test_nested_arena_depth_is_bounded():
    from neurospatial.arena import decode_arena

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


def test_mutated_nested_face_array_is_revalidated():
    from neurospatial.binning import Binning, MissingFaces

    env = Environment.from_grid_mask(
        np.ones((2, 1, 1), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(2.0), np.arange(2.0)),
    )
    child = replace(env._binning, faces=KnownFaces(env._face_geometry))
    child.faces.faces.A.setflags(write=True)
    child.faces.faces.A[:] = -1
    combined = CombinedLocator(
        ("child",),
        (child,),
        (env.arena,),
        np.empty((0, 2), dtype=np.intp),
        np.array([]),
        np.array([]),
    )
    with pytest.raises(ValueError, match="face areas"):
        Environment.from_bins(
            Binning(
                "combined", combined.spatial_bins(), combined, MissingFaces("missing")
            )
        )
