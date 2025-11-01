from collections.abc import Mapping

import numpy as np
import pytest

from neurospatial.regions.core import Region, Regions

try:
    import shapely.geometry as shp

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


@pytest.mark.parametrize(
    "coords",
    [
        [1.0, 2.0],
        np.array([3.5, 4.5]),
    ],
)
def test_region_point_creation(coords):
    r = Region(name="A", kind="point", data=coords)
    assert r.name == "A"
    assert r.kind == "point"
    assert np.allclose(r.data, np.asarray(coords))
    assert r.n_dims == len(coords)
    assert isinstance(r.metadata, Mapping)


def test_region_point_invalid_shape():
    with pytest.raises(ValueError):
        Region(name="bad", kind="point", data=[[1, 2], [3, 4]])


def test_region_str_repr():
    r = Region(name="foo", kind="point", data=[0, 1])
    assert str(r) == "foo"


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_polygon_creation():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly)
    assert r.kind == "polygon"
    assert r.n_dims == 2
    assert r.data.equals(poly)


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_polygon_invalid_type():
    with pytest.raises(TypeError):
        Region(name="badpoly", kind="polygon", data=[(0, 0), (1, 1)])


def test_region_unknown_kind():
    with pytest.raises(ValueError):
        Region(name="bad", kind="unknown", data=[1, 2])


def test_region_to_dict_and_from_dict_point():
    r = Region(name="pt", kind="point", data=[1, 2], metadata={"color": "red"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert np.allclose(r2.data, r.data)
    assert r2.metadata["color"] == "red"


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_to_dict_and_from_dict_polygon():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly, metadata={"label": "square"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert r2.data.equals(poly)
    assert r2.metadata["label"] == "square"


def test_regions_add_point_and_remove():
    regs = Regions()
    r = regs.add("pt", point=[1, 2, 3])
    assert regs["pt"] == r
    assert regs.list_names() == ["pt"]
    regs.remove("pt")
    assert "pt" not in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_add_polygon_and_area():
    regs = Regions()
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = regs.add("poly", polygon=poly)
    assert regs["poly"] == r
    area = regs.area("poly")
    assert np.isclose(area, 1.0)
    assert regs.area("poly") == poly.area


def test_regions_add_duplicate_name():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    with pytest.raises(KeyError):
        regs.add("pt", point=[3, 4])


def test_regions_add_both_point_and_polygon():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.add(name="bad", point=(1.0, 2.0), polygon=[(0, 0), (1, 0), (0, 1)])
    if HAS_SHAPELY:
        poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        with pytest.raises(ValueError):
            regs.add("bar", point=[1, 2], polygon=poly)


def test_regions_setitem_key_mismatch():
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    with pytest.raises(ValueError):
        regs["bar"] = r


def test_regions_setitem_duplicate():
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    regs["foo"] = r
    with pytest.raises(KeyError):
        regs["foo"] = r


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_point_and_polygon(tmp_path):
    regs = Regions()
    pt = [0.0, 0.0]
    regs.add("pt", point=pt)
    regs.buffer("pt", distance=1.0, new_name="buf")
    assert "buf" in regs
    assert regs["buf"].kind == "polygon"
    # Buffering a polygon
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    regs.add("poly", polygon=poly)
    regs.buffer("poly", distance=0.5, new_name="buf2")
    assert "buf2" in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_raw_point():
    regs = Regions()
    regs.buffer(np.array([0.0, 0.0]), distance=1.0, new_name="buf")
    assert "buf" in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_invalid_shape():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.buffer(np.array([1.0, 2.0, 3.0]), distance=1.0, new_name="badbuf")


def test_regions_area_point():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    assert regs.area("pt") == 0.0


def test_regions_remove_absent():
    regs = Regions()
    regs.remove("nope")  # Should not raise


def test_regions_repr():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    s = repr(regs)
    assert "Regions" in s and "pt(point)" in s


def test_regions_to_json_and_from_json(tmp_path):
    regs = Regions()
    regs.add("pt", point=[1, 2], metadata={"foo": "bar"})
    path = tmp_path / "regions.json"
    regs.to_json(path)
    loaded = Regions.from_json(path)
    assert "pt" in loaded
    assert loaded["pt"].metadata["foo"] == "bar"


# --- Tests for Regions.update_region() method ---


def test_regions_update_region_point():
    """Test updating an existing point region."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0])
    assert np.allclose(regs["pt"].data, [1.0, 2.0])

    # Update the point to new coordinates
    updated = regs.update_region("pt", point=[3.0, 4.0])
    assert np.allclose(regs["pt"].data, [3.0, 4.0])
    assert updated.name == "pt"
    assert updated.kind == "point"
    assert updated is regs["pt"]  # Returned region is the stored one


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_update_region_polygon():
    """Test updating an existing polygon region."""
    regs = Regions()
    poly1 = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    regs.add("poly", polygon=poly1)
    assert regs["poly"].data.equals(poly1)

    # Update to a different polygon
    poly2 = shp.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    updated = regs.update_region("poly", polygon=poly2)
    assert regs["poly"].data.equals(poly2)
    assert updated.name == "poly"
    assert updated.kind == "polygon"


def test_regions_update_region_with_metadata():
    """Test updating a region while changing metadata."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0], metadata={"color": "red"})
    assert regs["pt"].metadata["color"] == "red"

    # Update with new metadata
    updated = regs.update_region(
        "pt", point=[3.0, 4.0], metadata={"color": "blue", "size": 5}
    )
    assert regs["pt"].metadata["color"] == "blue"
    assert regs["pt"].metadata["size"] == 5
    assert updated is regs["pt"]


def test_regions_update_region_preserves_metadata():
    """Test that metadata is preserved when not explicitly updated."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0], metadata={"color": "red", "size": 10})
    assert regs["pt"].metadata["color"] == "red"
    assert regs["pt"].metadata["size"] == 10

    # Update only the point coordinates, not metadata
    updated = regs.update_region("pt", point=[3.0, 4.0])

    # Metadata should be preserved
    assert regs["pt"].metadata["color"] == "red"
    assert regs["pt"].metadata["size"] == 10
    assert updated.metadata["color"] == "red"
    assert updated.metadata["size"] == 10


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_update_region_change_kind():
    """Test updating a region to a different kind (point -> polygon)."""
    regs = Regions()
    regs.add("region1", point=[1.0, 2.0])
    assert regs["region1"].kind == "point"

    # Change from point to polygon
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    updated = regs.update_region("region1", polygon=poly)
    assert regs["region1"].kind == "polygon"
    assert updated.kind == "polygon"


def test_regions_update_region_nonexistent():
    """Test that updating a nonexistent region raises KeyError."""
    regs = Regions()
    with pytest.raises(KeyError, match="Region 'nonexistent' does not exist"):
        regs.update_region("nonexistent", point=[1.0, 2.0])


def test_regions_update_region_neither_point_nor_polygon():
    """Test that update requires exactly one of point or polygon."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0])

    # Neither point nor polygon specified
    with pytest.raises(
        ValueError, match="Specify \\*\\*one\\*\\* of 'point' or 'polygon'"
    ):
        regs.update_region("pt")


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_update_region_both_point_and_polygon():
    """Test that update rejects both point and polygon."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0])

    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(
        ValueError, match="Specify \\*\\*one\\*\\* of 'point' or 'polygon'"
    ):
        regs.update_region("pt", point=[3.0, 4.0], polygon=poly)
