"""Geometry shared by segmentation previews and accepted annotations."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import MultiPolygon, Point, Polygon

from neurospatial.annotation._polygon_outline import (
    PolygonOutline,
    polygon_outline,
    simplify_polygon_outline,
)
from neurospatial.annotation.segmentation import SegmentationProposal


def _proposal():
    mask = np.zeros((32, 40), dtype=bool)
    for row in range(2, 24):
        mask[row, 2 : 15 + row // 3] = True
    mask[8:12, 7:11] = False
    mask[26:30, 30:36] = True
    return SegmentationProposal(mask, score=0.8, metadata={"provider": "test"})


def test_exact_outline_preserves_mask_support_components_and_holes():
    proposal = _proposal()
    outline = polygon_outline(proposal)
    geometry = MultiPolygon(outline.polygons)
    assert outline.proposal is proposal
    assert outline.simplify_tolerance_px == 0.0
    assert len(outline.polygons) == 2
    assert sum(len(p.interiors) for p in outline.polygons) == 1
    assert geometry.area == proposal.mask.sum()
    for row, col in np.ndindex(proposal.mask.shape):
        assert geometry.covers(Point(col, row)) == bool(proposal.mask[row, col])


def test_simplification_keeps_components_holes_and_original_provenance():
    exact = polygon_outline(_proposal())
    simplified = simplify_polygon_outline(exact, 1.0)
    assert simplified.proposal is exact.proposal
    assert simplified.simplify_tolerance_px == 1.0
    assert simplified.vertex_count < exact.vertex_count
    assert len(simplified.polygons) == 2
    assert sum(len(p.interiors) for p in simplified.polygons) == 1
    assert MultiPolygon(simplified.polygons).is_valid
    assert not MultiPolygon(simplified.polygons).covers(Point(8, 9))
    assert simplified.proposal.metadata == {"provider": "test"}
    assert simplified.proposal.score == 0.8
    assert simplify_polygon_outline(exact, 0.0) is exact


@pytest.mark.parametrize("tolerance", [0.1, 1.0, 100.0])
@pytest.mark.parametrize("with_hole", [False, True])
def test_single_component_simplification_accepts_unwrapped_polygon(
    tolerance, with_hole
):
    mask = np.ones((7, 9), dtype=bool)
    if with_hole:
        mask[2:5, 3:6] = False
    exact = polygon_outline(SegmentationProposal(mask))
    simplified = simplify_polygon_outline(exact, tolerance)
    assert simplified.simplify_tolerance_px == tolerance
    assert simplified.proposal is exact.proposal
    assert len(simplified.polygons) == 1
    assert simplified.polygons[0].is_valid
    assert len(simplified.polygons[0].interiors) == int(with_hole)
    assert simplified.vertex_count <= exact.vertex_count


def test_simplification_uses_cached_exact_polygons(monkeypatch):
    exact = polygon_outline(_proposal())

    def fail_vectorize(*args, **kwargs):
        pytest.fail("Changing simplification must not vectorize the mask again")

    monkeypatch.setattr(
        "neurospatial.annotation._polygon_outline.mask_to_polygons", fail_vectorize
    )
    rough = simplify_polygon_outline(exact, 2.0)
    precise = simplify_polygon_outline(exact, 0.25)
    assert rough.vertex_count <= precise.vertex_count
    assert simplify_polygon_outline(exact, 0.0) is exact
    with pytest.raises(ValueError, match="cached exact"):
        simplify_polygon_outline(rough, 1.0)


def test_global_simplification_keeps_nearby_components_valid():
    mask = np.zeros((24, 24), dtype=bool)
    for row in range(2, 22):
        split = 8 + row // 3
        mask[row, 2:split] = True
        mask[row, split + 1 : 22] = True
    exact = polygon_outline(SegmentationProposal(mask))
    for tolerance in (0.5, 1.0, 3.0, 100.0):
        preview = simplify_polygon_outline(exact, tolerance)
        assert len(preview.polygons) == 2
        assert MultiPolygon(preview.polygons).is_valid
        assert preview.polygons[0].intersection(preview.polygons[1]).area == 0


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, -np.inf, None, "bad"])
def test_malformed_tolerance_rejected_without_changing_exact_geometry(tolerance):
    exact = polygon_outline(_proposal())
    original = tuple(p.wkb for p in exact.polygons)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        simplify_polygon_outline(exact, tolerance)
    assert tuple(p.wkb for p in exact.polygons) == original


def test_outline_and_rendered_rings_are_immutable_independent_snapshots():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    outline = polygon_outline(SegmentationProposal(mask))
    mask[:] = False
    assert outline.proposal.mask.sum() == 24
    with pytest.raises(FrozenInstanceError):
        outline.simplify_tolerance_px = 1.0
    rings = outline.rings
    assert len(rings) == 2
    assert outline.vertex_count == sum(len(ring) - 1 for ring in rings)
    for ring in rings:
        assert_array_equal(ring[0], ring[-1])
        with pytest.raises(ValueError, match="read-only"):
            ring[0, 0] = 100
    rings[0].setflags(write=True)
    rings[0][:] = 100
    assert not np.array_equal(outline.rings[0], rings[0])


def test_outline_rejects_invalid_or_overlapping_components():
    proposal = _proposal()
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with pytest.raises(ValueError, match="nonempty, valid"):
        PolygonOutline(proposal, ())
    with pytest.raises(ValueError, match="valid MultiPolygon"):
        PolygonOutline(proposal, (square, square))
