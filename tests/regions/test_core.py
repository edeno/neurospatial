from collections.abc import Mapping

import numpy as np
import pytest
import shapely
import shapely.geometry as shp

from neurospatial.regions.core import Region, Regions


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


BOWTIE = shp.Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])


@pytest.mark.parametrize(
    ("name", "kind", "data", "error", "match"),
    [
        ("", "point", [1.0, 2.0], ValueError, "names must be nonempty strings"),
        (3, "point", [1.0, 2.0], ValueError, "names must be nonempty strings"),
        ("bad", "point", [[1, 2], [3, 4]], ValueError, r"shape \(n_dims,\)"),
        ("bad", "point", [], ValueError, r"got shape \(0,\)"),
        ("bad", "point", shp.Point(), ValueError, r"got shape \(0,\)"),
        ("bad", "point", [np.nan, 2.0], ValueError, "NaN or infinite"),
        ("bad", "point", [np.inf, 2.0], ValueError, "NaN or infinite"),
        ("bad", "point", ["a", "b"], TypeError, "must be numbers"),
        ("bad", "point", BOWTIE, TypeError, 'use kind="polygon"'),
        ("bad", "polygon", [(0, 0), (1, 1)], TypeError, "must be a shapely Polygon"),
        (
            "bad",
            "polygon",
            shp.MultiPolygon([shp.box(0, 0, 1, 1), shp.box(2, 2, 3, 3)]),
            TypeError,
            "one region per part",
        ),
        ("bad", "polygon", shp.Polygon(), ValueError, "empty polygon"),
        ("bad", "polygon", BOWTIE, ValueError, "(?s)Self-intersection.*make_valid"),
        (
            "bad",
            "polygon",
            shp.Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1)]),
            ValueError,
            "z coordinates",
        ),
        (
            "bad",
            "polygon",
            shp.Polygon([(0, 0), (np.nan, 0), (1, 1)]),
            ValueError,
            "NaN or infinite vertex",
        ),
        ("bad", "unknown", [1, 2], ValueError, "unknown kind 'unknown'"),
    ],
)
def test_invalid_region_geometry_is_refused_naming_the_region(
    name, kind, data, error, match
):
    with pytest.raises(error, match=match) as raised:
        Region(name=name, kind=kind, data=data)
    if isinstance(name, str) and name:
        assert "Region 'bad'" in str(raised.value)


def test_self_intersecting_outline_cannot_claim_bins():
    """A bowtie has zero shapely area yet covered 55 bins of a 10 cm grid."""
    assert BOWTIE.area == 0
    with pytest.raises(ValueError, match="redraw it so its edges do not cross"):
        Region("arena", "polygon", BOWTIE)
    repaired = shapely.make_valid(BOWTIE)
    assert isinstance(repaired, shp.MultiPolygon)  # two triangles; pick one
    triangle = Region("arena", "polygon", repaired.geoms[0])
    assert triangle.data.area == pytest.approx(25.0)


def test_point_given_as_shapely_point_is_its_coordinates():
    region = Region("goal", "point", shp.Point(3.0, 4.0))
    np.testing.assert_array_equal(region.data, [3.0, 4.0])
    assert region.n_dims == 2


def test_region_polygon_creation():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly)
    assert r.kind == "polygon"
    assert r.n_dims == 2
    assert r.data.equals(poly)


def test_region_to_dict_and_from_dict_point():
    r = Region(name="pt", kind="point", data=[1, 2], metadata={"color": "red"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert np.allclose(r2.data, r.data)
    assert r2.metadata["color"] == "red"


def test_region_to_dict_and_from_dict_polygon():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly, metadata={"label": "square"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert r2.data.equals(poly)
    assert r2.metadata["label"] == "square"


def test_region_from_dict_metadata_is_optional():
    region = Region.from_dict({"name": "pt", "kind": "point", "geom": [1.0, 2.0]})
    assert region.metadata == {}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {"name": "goal", "kind": "polygon"},
            r"Region 'goal' \(kind 'polygon'\) is missing \['geom'\]",
        ),
        (
            {"kind": "point", "geom": [1, 2]},
            r"A region record \(kind 'point'\) is missing \['name'\]",
        ),
        (
            {"name": "goal", "kind": "point", "geom": [1, 2], "color": "red"},
            r"Region 'goal' \(kind 'point'\) has unknown keys \['color'\]",
        ),
        (
            {"name": "goal", "kind": "polygon", "geom": [[0, 0], [1, 0], [1, 1]]},
            r"Region 'goal' \(kind 'polygon'\) has geometry shapely cannot read",
        ),
        (
            {"name": "goal", "kind": "polygon", "geom": {"type": "Circle"}},
            r"Region 'goal' \(kind 'polygon'\) has geometry shapely cannot read",
        ),
        (
            {"name": "goal", "kind": "circle", "geom": [1, 2]},
            "Region 'goal' has unknown kind",
        ),
    ],
)
def test_region_from_dict_errors_name_the_record(payload, match):
    with pytest.raises(ValueError, match=match):
        Region.from_dict(payload)


def test_regions_add_point_and_remove():
    regs = Regions()
    r = regs.add("pt", point=[1, 2, 3])
    assert regs["pt"] == r
    assert list(regs) == ["pt"]
    regs.remove("pt")
    assert "pt" not in regs


def test_region_point_value_equality():
    """Equality compares coordinate values without coercing arrays to bool."""
    region = Region("pt", "point", [1, 2, 3], metadata={"labels": ["goal"]})
    assert region == region
    assert region == Region.from_dict(region.to_dict())
    assert region != Region("pt", "point", [1, 2, 4], metadata=region.metadata)
    assert region != Region("pt", "point", [1, 2], metadata=region.metadata)
    assert region != Region("other", "point", [1, 2, 3], metadata=region.metadata)
    assert region != Region("pt", "point", [1, 2, 3])
    assert region != object()


@pytest.mark.parametrize(
    "region",
    [
        Region("pt", "point", [1.0, 2.0]),
        Region("box", "polygon", shp.box(0, 0, 1, 1), metadata={"role": "reward"}),
    ],
    ids=["point", "polygon"],
)
def test_regions_compare_by_value_but_are_unhashable(region):
    assert region == Region.from_dict(region.to_dict())
    with pytest.raises(TypeError, match="unhashable"):
        hash(region)
    assert {region.name: region}[region.name] is region


def test_region_area_is_polygon_area_and_zero_for_points():
    donut = shp.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (8, 2), (8, 8), (2, 8)]]
    )
    assert Region("donut", "polygon", donut).area == pytest.approx(64.0)
    assert Region("pt", "point", [1.0, 2.0]).area == 0.0


def test_regions_add_duplicate_name():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    with pytest.raises(KeyError):
        regs.add("pt", point=[3, 4])


def test_regions_add_both_point_and_polygon():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.add(name="bad", point=(1.0, 2.0), polygon=[(0, 0), (1, 0), (0, 1)])
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        regs.add("bar", point=[1, 2], polygon=poly)


def test_regions_setitem_key_mismatch():
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    with pytest.raises(ValueError):
        regs["bar"] = r


def test_regions_setitem_duplicate_raises():
    """``regions[key] = ...`` refuses to silently overwrite an existing region."""
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    regs["foo"] = r

    # Overwriting via __setitem__ now raises; users must explicitly opt in
    # via update_region() or set().
    with pytest.raises(KeyError, match="already exists"):
        regs["foo"] = Region(name="foo", kind="point", data=[3, 4])

    # The original region is preserved.
    assert np.allclose(regs["foo"].data, [1, 2])


def test_regions_set_replaces_idempotently():
    """``Regions.set`` is the documented idempotent replace path."""
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    regs["foo"] = r

    # First call replaces the existing region; second call is a no-op replace.
    replacement = Region(name="foo", kind="point", data=[3, 4])
    out = regs.set("foo", replacement)
    assert out is replacement
    assert np.allclose(regs["foo"].data, [3, 4])

    regs.set("foo", replacement)  # idempotent; should not raise
    assert np.allclose(regs["foo"].data, [3, 4])


def test_regions_set_rejects_name_mismatch():
    regs = Regions()
    with pytest.raises(ValueError, match=r"must match Region\.name"):
        regs.set("foo", Region(name="bar", kind="point", data=[1, 2]))


@pytest.mark.parametrize("source", ["named", "raw"])
def test_buffered_point_is_a_disk_centered_on_the_point(source):
    regs = Regions()
    regs.add("well", point=[10.0, 20.0], metadata={"role": "reward"})
    zone = regs.buffer(
        "well" if source == "named" else np.array([10.0, 20.0]), 5.0, "zone"
    )
    # Shapely approximates the circle by a polygon inscribed in it.
    assert zone.area == pytest.approx(np.pi * 25.0, rel=0.01)
    assert zone.area < np.pi * 25.0
    np.testing.assert_allclose(zone.center, [10.0, 20.0], atol=1e-9)
    assert dict(zone.metadata) == {}  # **meta replaces; nothing is inherited


def test_buffered_polygon_covers_the_original_and_shrinking_stays_inside():
    regs = Regions()
    square = regs.add("arena", polygon=shp.box(0, 0, 10, 10), metadata={"a": 1})
    grown = regs.buffer("arena", 2.0, "grown", role="margin")
    shrunk = regs.buffer("arena", -2.0, "core")
    assert grown.data.covers(square.data)
    assert square.data.covers(shrunk.data)
    assert shrunk.area == pytest.approx(36.0)
    assert dict(grown.metadata) == {"role": "margin"}


@pytest.mark.parametrize("distance", [np.nan, np.inf, -np.inf])
def test_buffer_distance_must_be_finite(distance):
    regs = Regions()
    regs.add("well", point=[0.0, 0.0])
    with pytest.raises(ValueError, match="distance must be finite"):
        regs.buffer("well", distance, "zone")
    assert "zone" not in regs


@pytest.mark.parametrize(
    ("source", "distance"),
    [("well", 0.0), ("well", -1.0), ("arena", -6.0)],
    ids=["point-zero", "point-negative", "polygon-eroded"],
)
def test_buffer_that_leaves_nothing_is_refused(source, distance):
    regs = Regions()
    regs.add("well", point=[0.0, 0.0])
    regs.add("arena", polygon=shp.box(0, 0, 10, 10))
    with pytest.raises(ValueError, match="leaves nothing"):
        regs.buffer(source, distance, "zone")
    assert "zone" not in regs


def test_buffer_that_splits_a_region_is_refused():
    # Two 10 x 10 rooms joined by a 2-wide corridor.
    dumbbell = shapely.union_all(
        [shp.box(0, 0, 10, 10), shp.box(10, 4, 20, 6), shp.box(20, 0, 30, 10)]
    )
    regs = Regions()
    regs.add("rooms", polygon=dumbbell)
    with pytest.raises(ValueError, match="splits it into 2 parts"):
        regs.buffer("rooms", -2.0, "cores")
    assert "cores" not in regs


def test_regions_buffer_raw_point():
    regs = Regions()
    regs.buffer(np.array([0.0, 0.0]), distance=1.0, new_name="buf")
    assert "buf" in regs


def test_regions_buffer_invalid_shape():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.buffer(np.array([1.0, 2.0, 3.0]), distance=1.0, new_name="badbuf")


def test_region_center_of_a_point_is_a_copy_of_its_coordinates():
    region = Region("pt", kind="point", data=np.array([3.0, 7.0]))
    center = region.center
    np.testing.assert_array_equal(center, [3.0, 7.0])
    center[0] = -1.0  # an independent array, not the read-only data
    np.testing.assert_array_equal(region.data, [3.0, 7.0])


def test_region_center_of_a_polygon_is_its_centroid():
    square = Region("box", kind="polygon", data=shp.box(0, 0, 1, 1))
    np.testing.assert_allclose(square.center, [0.5, 0.5])


def test_u_shaped_region_center_lies_outside_the_region():
    """The documented caveat: a centroid need not be inside a non-convex arm."""
    u_shape = shp.Polygon(
        [(0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (10, 10), (10, 30), (0, 30)]
    )
    region = Region("u_arm", kind="polygon", data=u_shape)
    assert not u_shape.contains(shp.Point(region.center))


def test_regions_remove_absent_raises():
    """Regions.remove now raises on missing name.

    The previous behavior silently absorbed missing names, masking
    typos and double-removes. The full contract is that every
    Region mutation API raises on the disagree-with-reality case.
    """
    regs = Regions()
    with pytest.raises(KeyError):
        regs.remove("nope")


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


class TestRegionMetadataImmutability:
    """Test that Region metadata is truly immutable and isolated from external changes.

    These tests verify that Region metadata is deep-copied on creation, preventing
    accidental mutation through external references to the original metadata dict.
    """

    def test_metadata_isolated_from_external_modification(self):
        """Test that modifying original metadata dict doesn't affect Region.

        Bug: Region.__post_init__ uses dict() which creates a shallow copy.
        This means if you modify the original dict after creating the Region,
        the Region's metadata also changes (for nested dicts/lists).
        """
        # Create metadata dict
        original_metadata = {"color": "red", "type": "goal"}

        # Create region with metadata
        region = Region(
            name="test", kind="point", data=[10.0, 10.0], metadata=original_metadata
        )

        # Verify initial state
        assert region.metadata["color"] == "red"
        assert region.metadata["type"] == "goal"

        # Modify original dict - this should NOT affect the Region
        original_metadata["color"] = "blue"
        original_metadata["type"] = "start"

        # Region metadata should be unchanged (isolated from external changes)
        assert region.metadata["color"] == "red", (
            "Region metadata was mutated by external change to original dict! "
            "Metadata should be deep-copied on creation."
        )
        assert region.metadata["type"] == "goal"

    def test_nested_metadata_isolated_from_external_modification(self):
        """Test that modifying nested structures in original metadata doesn't affect Region.

        This is the critical test for deep copy vs shallow copy. With shallow copy,
        nested dicts/lists are shared by reference.
        """
        # Create metadata with nested dict
        original_metadata = {
            "color": "red",
            "properties": {"size": 10, "visible": True},
            "tags": ["important", "goal"],
        }

        # Create region
        region = Region(
            name="test", kind="point", data=[10.0, 10.0], metadata=original_metadata
        )

        # Verify initial state. Lists freeze to tuples in stored metadata so
        # nested sequences cannot be mutated either.
        assert region.metadata["properties"]["size"] == 10
        assert region.metadata["properties"]["visible"] is True
        assert region.metadata["tags"] == ("important", "goal")

        # Modify nested structures in original metadata
        original_metadata["properties"]["size"] = 20
        original_metadata["properties"]["visible"] = False
        original_metadata["tags"].append("new_tag")

        # Region metadata should be unchanged (deep copy protection)
        assert region.metadata["properties"]["size"] == 10, (
            "Nested metadata dict was mutated! Shallow copy detected. "
            "Need copy.deepcopy() not dict()."
        )
        assert region.metadata["properties"]["visible"] is True
        assert region.metadata["tags"] == ("important", "goal")
        # Serialization thaws tuples back to lists for JSON compatibility.
        assert region.to_dict()["metadata"]["tags"] == ["important", "goal"]

    def test_metadata_empty_dict_default(self):
        """Test that omitting metadata creates an empty dict (not shared reference)."""
        # Create two regions without metadata
        region1 = Region(name="r1", kind="point", data=[1.0, 2.0])
        region2 = Region(name="r2", kind="point", data=[3.0, 4.0])

        # Both should have empty metadata
        assert region1.metadata == {}
        assert region2.metadata == {}

        # They should NOT share the same dict object
        assert region1.metadata is not region2.metadata, (
            "Regions are sharing the same default metadata dict! "
            "Each Region should have its own metadata instance."
        )

    def test_top_level_metadata_assignment_raises(self):
        """Stored metadata must reject top-level mutation after construction."""
        region = Region(
            name="test", kind="point", data=[1.0, 2.0], metadata={"color": "red"}
        )
        with pytest.raises(TypeError):
            region.metadata["color"] = "blue"  # type: ignore[index]

    def test_nested_metadata_mutation_raises(self):
        """Nested containers in stored metadata are read-only at every depth.

        Recursive freezing means a nested mapping fetched from metadata is
        itself a read-only proxy, so attempting to rebind a nested key raises
        rather than silently corrupting the Region.
        """
        region = Region(
            name="test",
            kind="point",
            data=[1.0, 2.0],
            metadata={"properties": {"size": 10}},
        )
        nested = region.metadata["properties"]
        with pytest.raises(TypeError):
            nested["size"] = 999  # type: ignore[index]
        # The canonical value is untouched.
        assert region.metadata["properties"]["size"] == 10


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
    with pytest.raises(KeyError, match="Region 'nonexistent' not found"):
        regs.update_region("nonexistent", point=[1.0, 2.0])


def test_regions_update_region_needs_something_to_change():
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0])
    with pytest.raises(ValueError, match="pass point=, polygon= or metadata="):
        regs.update_region("pt")


@pytest.mark.parametrize(
    "change",
    [{"point": [9.0, 9.0]}, {"polygon": shp.box(0, 0, 2, 2)}, {"metadata": {"a": 1}}],
    ids=["point", "polygon", "metadata"],
)
def test_update_region_keeps_its_position_and_membership_column(change):
    from neurospatial import Environment

    regs = Regions()
    regs.add("start", point=[1.0, 1.0])
    regs.add("goal", point=[5.0, 5.0], metadata={"color": "red"})
    regs.add("home", point=[9.0, 1.0])
    regs.update_region("goal", **change)
    assert list(regs) == ["start", "goal", "home"]

    edges = np.arange(0.0, 11.0, 2.0)
    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(edges, edges)
    ).with_regions(regs)
    membership = env.region_membership()
    np.testing.assert_array_equal(
        membership[:, 0], env.region_mask("start")
    )  # column 0 is still "start", not shifted by the update


def test_update_region_metadata_only_keeps_the_geometry():
    regs = Regions()
    original = regs.add("zone", polygon=shp.box(0, 0, 2, 2), metadata={"color": "red"})
    updated = regs.update_region("zone", metadata={"color": "blue"})
    assert updated.kind == "polygon"
    assert updated.data.equals(original.data)
    assert dict(updated.metadata) == {"color": "blue"}
    assert regs["zone"] is updated


def test_regions_update_region_both_point_and_polygon():
    """Test that update rejects both point and polygon."""
    regs = Regions()
    regs.add("pt", point=[1.0, 2.0])

    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(
        ValueError, match="Specify \\*\\*one\\*\\* of 'point' or 'polygon'"
    ):
        regs.update_region("pt", point=[3.0, 4.0], polygon=poly)


@pytest.mark.parametrize(
    "insert",
    [
        lambda regs, value: Regions([value]),
        lambda regs, value: regs.__setitem__("cue", value),
        lambda regs, value: regs.set("cue", value),
        lambda regs, value: regs.update({"cue": value}),
        lambda regs, value: regs.setdefault("cue", value),
    ],
    ids=["init", "setitem", "set", "update", "setdefault"],
)
def test_regions_hold_only_region_values(insert):
    """A line Landmark in Regions would be published as env.regions['cue']."""
    from neurospatial.landmarks import Landmark

    regs = Regions()
    cue = Landmark("cue", "line", [(0.0, 100.0), (40.0, 100.0)])
    with pytest.raises(TypeError, match="Regions holds Region values, got Landmark"):
        insert(regs, cue)
    assert "cue" not in regs


def test_regions_from_a_mapping_says_to_pass_its_values():
    mapping = {"goal": Region("goal", "point", [1.0, 2.0])}
    with pytest.raises(TypeError, match=r"Regions\(mapping.values\(\)\)"):
        Regions(mapping)
    assert list(Regions(mapping.values())) == ["goal"]


def test_setdefault_without_a_region_is_refused_with_the_fix():
    regs = Regions()
    with pytest.raises(TypeError, match=r"setdefault\('goal'\) has no region"):
        regs.setdefault("goal")
    assert "goal" not in regs
    existing = regs.add("home", point=[0.0, 0.0])
    assert regs.setdefault("home") is existing


@pytest.mark.parametrize(
    "insert",
    [
        lambda regs: Regions([*regs.values(), Region("high", "point", [1, 2, 3])]),
        lambda regs: regs.add("high", point=[1.0, 2.0, 3.0]),
        lambda regs: regs.set("high", Region("high", "point", [1, 2, 3])),
        lambda regs: regs.update_region("goal", point=[1.0, 2.0, 3.0]),
    ],
    ids=["init", "add", "set", "update_region"],
)
def test_regions_refuse_mixed_dimensions(insert):
    regs = Regions()
    regs.add("goal", point=[1.0, 2.0])
    regs.add("zone", polygon=shp.box(0, 0, 1, 1))
    with pytest.raises(ValueError, match=r"'high'|'goal'") as raised:
        insert(regs)
    assert "has 3 coordinates, but the other regions here have 2" in str(raised.value)
    assert list(regs) == ["goal", "zone"]


def test_the_only_region_can_change_dimension():
    regs = Regions()
    regs.add("goal", point=[1.0, 2.0])
    regs.set("goal", Region("goal", "point", [1.0, 2.0, 3.0]))
    assert regs["goal"].n_dims == 3
