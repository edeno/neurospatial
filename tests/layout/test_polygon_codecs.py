"""Polygon snapshots keep present and historical locator frames distinct."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial.layout._binning import Binning, MissingFaces, SpatialBins
from neurospatial.layout._codecs import decode, encode
from neurospatial.layout._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClipMapping, ClippedLocator, PolygonCells


def clipped_binning():
    cells = PolygonCells((box(0, 0, 1, 4), box(3, 0, 4, 4)))
    mapping = ClipMapping(np.array([0, 0]), 1)
    base = RectilinearLocator((np.array([0.0, 4.0]),) * 2, (1, 1), None, None)
    locator = ClippedLocator(base, cells, mapping)
    bins = SpatialBins.from_arrays(
        [[0.5, 2.0], [3.5, 2.0]], [4.0, 4.0], 2, np.empty((0, 2), int), []
    )
    return Binning(
        "clipped",
        bins,
        locator,
        MissingFaces("fixture"),
        shapes=cells,
        clip_mapping=mapping,
    )


@pytest.mark.parametrize("part", ["shapes", "clip_mapping", "locator"])
def test_standalone_polygon_codec_roundtrip(part):
    original = getattr(clipped_binning(), part)
    meta, arrays = encode(original, "part")
    json.dumps(meta, allow_nan=False)
    restored = decode(meta, arrays, "part")
    assert type(restored) is type(original)
    if isinstance(restored, PolygonCells):
        assert [p.wkb for p in restored.polygons] == [p.wkb for p in original.polygons]
    elif isinstance(restored, ClipMapping):
        np.testing.assert_array_equal(restored.parent_bin_indices, [0, 0])
        assert not restored.parent_bin_indices.flags.writeable
    else:
        np.testing.assert_array_equal(
            restored.locate([[0.5, 1.0], [2.0, 1.0], [3.5, 1.0], [4.0, 1.0]]),
            [0, -1, 1, -1],
        )
        assert restored._tree is not None
        assert "tree" not in json.dumps(meta)


def test_binning_tables_store_each_snapshot_once_and_restore_shared_identity():
    original = clipped_binning()
    meta, arrays = encode(original, "binning")
    assert len(meta["snapshots"]["polygon_cells"]) == 1
    assert len(meta["snapshots"]["clip_mappings"]) == 1
    assert sum(key.endswith("/wkb") for key in arrays) == 1
    restored = decode(meta, arrays, "binning")
    assert restored.shapes is restored.locator.cells
    assert restored.clip_mapping is restored.locator.mapping
    np.testing.assert_array_equal(
        restored.locator.locate(original.bins.centers), [0, 1]
    )
    for array in arrays.values():
        assert array.flags.writeable
        array[...] = 0
    np.testing.assert_array_equal(restored.clip_mapping.parent_bin_indices, [0, 0])
    np.testing.assert_array_equal(
        restored.locator.locate(original.bins.centers), [0, 1]
    )


@pytest.mark.parametrize("operation", ["subset", "transform"])
def test_present_polygon_snapshot_is_distinct_from_historical_locator_frame(operation):
    original = clipped_binning()
    if operation == "subset":
        cells = PolygonCells((original.shapes.polygons[1],))
        mapping = ClipMapping(np.array([0]), 1)
        locator = SubsetLocator(original.locator, np.array([-1, 0]))
        centers, sizes = original.bins.centers[1:], original.bins.sizes[1:]
    else:
        cells = PolygonCells((box(10, 0, 11, 4), box(13, 0, 14, 4)))
        mapping = ClipMapping(np.array([0, 0]), 1)
        inverse = np.eye(3)
        inverse[0, 2] = -10
        locator = TransformedLocator(original.locator, inverse)
        centers, sizes = original.bins.centers + np.array([10, 0]), original.bins.sizes
    bins = SpatialBins.from_arrays(centers, sizes, 2, np.empty((0, 2), int), [])
    derived = Binning(
        "derived", bins, locator, original.faces, shapes=cells, clip_mapping=mapping
    )
    meta, arrays = encode(derived, "binning")
    assert len(meta["snapshots"]["polygon_cells"]) == 2
    restored = decode(meta, arrays, "binning")
    assert restored.shapes is not restored.locator.base.cells
    assert restored.clip_mapping is not restored.locator.base.mapping
    np.testing.assert_array_equal(
        restored.locator.locate(centers), np.arange(len(centers))
    )
    assert restored.locator.base.cells.polygons[0].bounds == (0, 0, 1, 4)


@pytest.mark.parametrize("field", ["shapes", "clip_mapping"])
@pytest.mark.parametrize("bad_id", [-1, 999, True, 0.5])
def test_snapshot_reference_corruption_is_rejected(field, bad_id):
    meta, arrays = encode(clipped_binning(), "binning")
    key = "polygon_cells_ref" if field == "shapes" else "clip_mapping_ref"
    meta[field][key] = bad_id
    with pytest.raises(ValueError, match=r"snapshot|reference"):
        decode(meta, arrays, "binning")


@pytest.mark.parametrize(
    "corruption", ["missing", "fractional", "descending", "last_offset", "wkb", "dtype"]
)
def test_polygon_packed_arrays_reject_corruption(corruption):
    meta, arrays = encode(clipped_binning().shapes, "cells")
    if corruption == "missing":
        del arrays["cells/wkb"]
    elif corruption == "fractional":
        arrays["cells/offsets"] = arrays["cells/offsets"].astype(float) + 0.5
    elif corruption == "descending":
        arrays["cells/offsets"][1] = -1
    elif corruption == "last_offset":
        arrays["cells/offsets"][-1] -= 1
    elif corruption == "wkb":
        arrays["cells/wkb"][:] = 0
    else:
        arrays["cells/wkb"] = arrays["cells/wkb"].astype(float)
    with pytest.raises(ValueError):
        decode(meta, arrays, "cells")


def test_binning_without_new_mapping_field_is_backward_compatible():
    original = clipped_binning()
    plain = Binning(
        "plain",
        original.bins,
        RectilinearLocator((np.arange(3.0),) * 2, (2, 2), None, None),
        original.faces,
    )
    meta, arrays = encode(plain, "binning")
    meta.pop("clip_mapping", None)
    restored = decode(meta, arrays, "binning")
    assert restored.clip_mapping is None


def test_locator_reference_and_mapping_payload_corruption_is_rejected():
    meta, arrays = encode(clipped_binning(), "binning")
    bad = copy.deepcopy(meta)
    bad["locator"]["cells_ref"] = 99
    with pytest.raises(ValueError, match=r"snapshot|reference"):
        decode(bad, arrays, "binning")
    bad = copy.deepcopy(meta)
    bad["snapshots"]["clip_mappings"][0]["n_parent_bins"] = 0.5
    with pytest.raises(ValueError):
        decode(bad, arrays, "binning")


def test_repeated_clip_reuses_current_snapshot_and_keeps_original_ownership_data():
    first = clipped_binning()
    cells = PolygonCells((box(0, 0, 1, 2), box(3, 2, 4, 4)))
    mapping = ClipMapping(np.array([0, 1]), 2)
    locator = ClippedLocator(first.locator, cells, mapping)
    bins = SpatialBins.from_arrays(
        [[0.5, 1.0], [3.5, 3.0]], [2.0, 2.0], 2, np.empty((0, 2), int), []
    )
    second = Binning(
        "clipped_twice", bins, locator, first.faces, shapes=cells, clip_mapping=mapping
    )
    meta, arrays = encode(second, "binning")
    restored = decode(meta, arrays, "binning")
    assert len(meta["snapshots"]["polygon_cells"]) == 2
    assert restored.shapes is restored.locator.cells
    assert restored.clip_mapping is restored.locator.mapping
    assert restored.locator.base.cells is not restored.shapes
    np.testing.assert_array_equal(
        restored.locator.locate([[0.5, 1.0], [0.5, 3.0], [3.5, 1.0], [3.5, 3.0]]),
        [0, -1, -1, 1],
    )


def test_polygon_holes_roundtrip_and_inline_duplicate_binning_authority_rejects():
    from shapely.geometry import Polygon

    polygon = Polygon(
        box(0, 0, 4, 4).exterior.coords, holes=[box(1, 1, 3, 3).exterior.coords]
    )
    cells = PolygonCells((polygon,))
    restored = decode(*encode(cells, "cells"), "cells")
    assert restored.polygons[0].wkb == polygon.wkb
    original = clipped_binning()
    meta, arrays = encode(original, "binning")
    inline, nested = encode(original.shapes, "binning/shapes")
    arrays.update(nested)
    meta["shapes"] = inline
    with pytest.raises(ValueError, match="snapshot"):
        decode(meta, arrays, "binning")
