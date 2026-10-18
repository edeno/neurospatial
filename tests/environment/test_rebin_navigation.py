"""Factor rebinning must not silently replace declared navigation geometry."""

from copy import deepcopy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena, UnknownArena
from neurospatial.binning import BinConnections, FaceGeometry, FaceNormals, KnownFaces
from neurospatial.environment._fields import face_geometry


def _grid(*, diagonal=False, scale=1.0, arena=None):
    return Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0) * scale,) * 2,
        connect_diagonal_neighbors=diagonal,
        arena=arena,
    )


def _explicit_grid(source, *, spec, navigation=None):
    """Retain exact cells and faces while changing only declared navigation."""
    binning = source._binning
    bins = binning.bins
    pairs = binning.connections.pairs
    faces = face_geometry(source._cache, source._binning)
    keep = np.ones(len(pairs), dtype=bool)
    distances = binning.connections.pair_distances.copy()
    if navigation == "long_path":
        distances[0] *= 7.0
    elif navigation == "missing_link":
        keep[0] = False
    return Environment.from_bins(
        replace(
            binning,
            kind="RecordedGrid",
            connections=BinConnections.from_arrays(pairs[keep], distances[keep]),
            faces=KnownFaces(
                FaceGeometry(
                    pairs[keep],
                    faces.A[keep],
                    faces.distance[keep],
                    bins.sizes,
                )
            ),
            face_normals=FaceNormals(binning.face_normals.values[keep]),
            grid_spec=binning.grid_spec if spec else None,
        ),
        arena=source.arena,
    )


@pytest.mark.parametrize("spec", [False, True], ids=["legacy", "owned-spec"])
@pytest.mark.parametrize("navigation", ["long_path", "missing_link"])
@pytest.mark.parametrize("factor", [1, 2])
def test_factor_rebin_refuses_non_native_navigation(spec, navigation, factor):
    source = _explicit_grid(_grid(), spec=spec, navigation=navigation)
    with pytest.raises(NotImplementedError, match=r"navigation.*subset"):
        source.rebin(factor)


@pytest.mark.parametrize("spec", [False, True], ids=["legacy", "owned-spec"])
@pytest.mark.parametrize("diagonal", [False, True])
def test_native_explicit_grid_navigation_survives_copy_and_load(spec, diagonal):
    native = _grid(diagonal=diagonal, scale=0.05)
    source = _explicit_grid(native, spec=spec)
    expected = native.rebin(2)
    for parent in (source, deepcopy(source), Environment.from_dict(source.to_dict())):
        actual = parent.rebin(2)
        np.testing.assert_array_equal(actual.bin_centers, expected.bin_centers)
        np.testing.assert_array_equal(actual.edge_pairs, expected.edge_pairs)
        np.testing.assert_allclose(
            actual._binning.connections.pair_distances,
            expected._binning.connections.pair_distances,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("diagonal", [False, True])
def test_factor_rebin_accepts_subset_and_wall_filtered_native_graph(diagonal):
    arena = OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),))
    source = _grid(diagonal=diagonal, arena=arena)
    for parent in (source, source.subset(mask=source.bin_centers[:, 1] < 2)):
        child = parent.rebin(1)
        # Rebinning keeps the selected bins and the wall that splits them.
        np.testing.assert_array_equal(child.bin_centers, parent.bin_centers)
        assert set(child.connectivity.edges()) == set(parent.connectivity.edges())
        assert nx.number_connected_components(child.connectivity) == 2


def test_owned_diagonal_policy_applies_when_source_has_no_diagonal_edges():
    active = np.zeros((4, 4), dtype=bool)
    active[0] = True
    active[3, 3] = True  # far from row 0, so no diagonal neighbor in the source
    source = Environment.from_grid_mask(
        active,
        grid_edges=(np.arange(5.0),) * 2,
        connect_diagonal_neighbors=True,
        arena=UnknownArena(2),
    )
    assert all(
        np.count_nonzero(source.bin_centers[u] != source.bin_centers[v]) == 1
        for u, v in source.edge_pairs
    )
    # The stored construction choice remains authoritative for new diagonal
    # links: the coarse blocks holding bins are (0, 0), (0, 1) and (1, 1).
    child = source.rebin(2)
    assert child.n_bins == 3
    assert child.connectivity.number_of_edges() == 3
