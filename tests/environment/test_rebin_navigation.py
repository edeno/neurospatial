"""Factor rebinning must not silently replace declared navigation geometry."""

from copy import deepcopy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena, UnknownArena
from neurospatial.binning import FaceGeometry, FaceNormals, KnownFaces, SpatialBins


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
    faces = source._face_geometry
    keep = np.ones(len(bins.pairs), dtype=bool)
    distances = bins.pair_distances.copy()
    if navigation == "long_path":
        distances[0] *= 7.0
    elif navigation == "missing_link":
        keep[0] = False
    return Environment.from_bins(
        replace(
            binning,
            kind="RecordedGrid",
            bins=SpatialBins(
                bins.centers,
                bins.sizes,
                bins.measure_dim,
                bins.pairs[keep],
                distances[keep],
            ),
            faces=KnownFaces(
                FaceGeometry(
                    bins.pairs[keep],
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
            actual._binning.bins.pair_distances,
            expected._binning.bins.pair_distances,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("diagonal", [False, True])
def test_factor_rebin_accepts_subset_and_wall_filtered_native_graph(diagonal):
    arena = OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),))
    source = _grid(diagonal=diagonal, arena=arena)
    for parent in (source, source.subset(bins=source.bin_centers[:, 1] < 2)):
        child = parent.rebin(1)
        # The original arena still determines the target support, including when
        # the input is a selected analysis region within that physical arena.
        np.testing.assert_array_equal(child.bin_centers, source.bin_centers)
        assert set(child.connectivity.edges()) == set(source.connectivity.edges())
        assert nx.number_connected_components(child.connectivity) == 2


def test_owned_diagonal_policy_applies_when_source_has_no_diagonal_edges():
    active = np.zeros((4, 4), dtype=bool)
    active[0] = True
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
    # An unknown domain retains the existing full-coarse-grid behavior. The
    # stored construction choice remains authoritative for its new diagonal links.
    child = source.rebin(2)
    assert child.n_bins == 4
    assert child.connectivity.number_of_edges() == 6
