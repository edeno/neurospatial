"""Derived environments keep their layout label; provenance records the history.

``env.layout_type`` names the layout the bins were built with. ``subset``,
``clip``, ``rebin`` and ``apply_transform`` keep the parent's label and add
their name to ``env.derivations``; ``combine`` reports ``"Combined"`` and
starts a new history. ``layout_parameters`` keeps the parent's build choices
unless a derivation builds new bins.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment, Regions
from neurospatial.io import from_dict, to_dict
from neurospatial.layout import LayoutType, create_layout
from neurospatial.ops.transforms import translate


def _track():
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=1.0
    )


PARENTS = {
    "RegularGrid": lambda: Environment.from_samples(
        np.random.default_rng(0).uniform(0, 20, (800, 2)), bin_size=2.0
    ),
    "MaskedGrid": lambda: Environment.from_grid_mask(
        np.ones((8, 8), dtype=bool), grid_edges=(np.arange(9.0) * 2.5,) * 2
    ),
    "ImageMask": lambda: Environment.from_pixel_mask(
        np.ones((8, 8), dtype=bool), pixel_size=2.5
    ),
    "ShapelyPolygon": lambda: Environment.from_polygon(
        Point(10, 10).buffer(10), bin_size=2.0
    ),
    "Hexagonal": lambda: Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=3.0, dimension_ranges=[(0, 20), (0, 20)]
        )
    ),
    "TriangularMesh": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
            point_spacing=3.0,
        )
    ),
    "Graph": _track,
}
GRIDS = ("RegularGrid", "MaskedGrid", "ImageMask", "ShapelyPolygon")


def _first_half(env):
    return np.arange(env.n_bins) < env.n_bins // 2


@pytest.mark.parametrize("label", sorted(PARENTS))
def test_subset_and_transform_keep_the_label_and_build_choices(label):
    parent = PARENTS[label]()
    assert parent.layout_type == label
    assert parent.derivations == ()
    child = parent.subset(mask=_first_half(parent))
    shift = (1.0,) * parent.n_dims
    moved = child.apply_transform(translate(*shift)) if parent.n_dims == 2 else child
    for env in (child, moved):
        assert env.layout_type == label == env.binning.kind
        assert env.layout_parameters == parent.layout_parameters
    assert child.derivations == ("subset",)
    if parent.n_dims == 2:
        assert moved.derivations == ("subset", "apply_transform")


@pytest.mark.parametrize("label", [*GRIDS, "Hexagonal"])
def test_clip_keeps_the_label_and_build_choices(label):
    parent = PARENTS[label]()
    clipped = parent.clip(box(4.0, 4.0, 15.0, 15.0))
    assert clipped.layout_type == label
    assert clipped.layout_parameters == parent.layout_parameters
    assert clipped.derivations == ("clip",)


@pytest.mark.parametrize("label", GRIDS)
@pytest.mark.parametrize("how", ["factor", "bin_size"])
def test_rebin_keeps_the_label_and_records_the_new_bins(label, how):
    parent = PARENTS[label]()
    if how == "factor":
        coarse = parent.rebin(2)
    else:
        if parent.binning.grid_spec is None:
            pytest.skip("this grid stores no binning specification")
        coarse = parent.rebin(bin_size=5.0)
    assert coarse.layout_type == label
    assert coarse.derivations == ("rebin",)
    widths = tuple(float(np.diff(edges)[0]) for edges in coarse.grid_edges)
    np.testing.assert_allclose(coarse.layout_parameters["bin_size"], widths)
    assert set(coarse.layout_parameters) == {"bin_size", "connect_diagonal_neighbors"}


def test_history_accumulates_in_order_and_relabelling_is_not_a_derivation():
    parent = PARENTS["RegularGrid"]()
    env = (
        parent.subset(mask=parent.bin_centers[:, 0] < 15.0)
        .rebin(2)
        .apply_transform(translate(5.0, 0.0))
        .clip(box(6.0, 0.0, 18.0, 20.0))
        .with_metadata(name="cropped")
        .with_regions(Regions())
    )
    assert env.layout_type == "RegularGrid"
    assert env.derivations == ("subset", "rebin", "apply_transform", "clip")


def test_combine_reports_the_combined_label_and_starts_a_new_history():
    left = Environment.from_polygon(box(0, 0, 10, 10), bin_size=2.0)
    right = Environment.from_polygon(box(12, 0, 22, 10), bin_size=2.0).subset(
        mask=np.ones(25, dtype=bool)
    )
    combined = Environment.combine({"a": left, "b": right})
    assert combined.layout_type == LayoutType.COMBINED == "Combined"
    assert combined.binning.kind == "Combined"
    assert combined.derivations == ("combine",)
    assert combined.layout_parameters == {}
    assert combined.subset(mask=_first_half(combined)).derivations == (
        "combine",
        "subset",
    )


@pytest.mark.parametrize("label", sorted(PARENTS))
def test_derived_labels_and_history_round_trip(label):
    parent = PARENTS[label]()
    child = parent.subset(mask=_first_half(parent))
    restored = from_dict(json.loads(json.dumps(to_dict(child))))
    assert restored.layout_type == child.layout_type == label
    assert restored.derivations == child.derivations == ("subset",)
    assert restored.layout_parameters == child.layout_parameters
