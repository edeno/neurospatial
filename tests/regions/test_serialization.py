"""Tests for regions I/O module."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import Point, Polygon

from neurospatial.regions import Region, Regions
from neurospatial.regions.io import (
    _parse_cvat_points,
    _rle_to_mask,
    load_cvat_xml,
    load_labelme_json,
    mask_to_region,
    regions_from_json,
    regions_to_dataframe,
    regions_to_json,
)


class TestRegionsToJson:
    """Tests for regions_to_json and regions_from_json round-trip."""

    def test_round_trip_polygon(self, tmp_path):
        """Test JSON round-trip with polygon region."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        region = Region(name="box", data=poly, kind="polygon")
        regions = Regions([region])

        json_path = tmp_path / "test.json"
        regions_to_json(regions, json_path)

        # Verify file exists
        assert json_path.exists()

        # Load back
        loaded_regions = regions_from_json(json_path)

        assert len(loaded_regions) == 1
        assert "box" in loaded_regions
        assert loaded_regions["box"].kind == "polygon"
        assert loaded_regions["box"].data.equals(poly)

    def test_round_trip_point(self, tmp_path):
        """Test JSON round-trip with point region."""
        region = Region(name="center", data=(5.0, 10.0), kind="point")
        regions = Regions([region])

        json_path = tmp_path / "points.json"
        regions_to_json(regions, json_path)

        loaded_regions = regions_from_json(json_path)

        assert len(loaded_regions) == 1
        assert "center" in loaded_regions
        assert loaded_regions["center"].kind == "point"
        coords = loaded_regions["center"].data
        assert_allclose(coords, (5.0, 10.0))

    def test_round_trip_multiple_regions(self, tmp_path):
        """Test JSON round-trip with multiple regions."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
                Region(name="pt1", data=(0.5, 0.5), kind="point"),
            ]
        )

        json_path = tmp_path / "multi.json"
        regions_to_json(regions, json_path)

        loaded_regions = regions_from_json(json_path)

        assert len(loaded_regions) == 3
        assert "box1" in loaded_regions
        assert "box2" in loaded_regions
        assert "pt1" in loaded_regions

    def test_empty_regions(self, tmp_path):
        """Test writing and reading empty regions collection."""
        regions = Regions([])

        json_path = tmp_path / "empty.json"
        regions_to_json(regions, json_path)

        loaded_regions = regions_from_json(json_path)

        assert len(loaded_regions) == 0

    def test_create_parent_directory(self, tmp_path):
        """Test that parent directories are created if needed."""
        json_path = tmp_path / "nested" / "dir" / "test.json"

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        region = Region(name="box", data=poly, kind="polygon")
        regions = Regions([region])

        regions_to_json(regions, json_path)

        assert json_path.exists()

    def test_custom_indent(self, tmp_path):
        """Test custom indentation in JSON output."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        region = Region(name="box", data=poly, kind="polygon")
        regions = Regions([region])

        json_path = tmp_path / "test.json"
        regions_to_json(regions, json_path, indent=4)

        # Verify file has proper indentation
        content = json_path.read_text(encoding="utf-8")
        assert "    " in content  # 4-space indent

    def test_method_and_function_paths_agree(self, tmp_path):
        """Regions.to_json and regions_to_json must produce identical bytes.

        Regions.to_json/from_json delegate to the io.py free functions; this
        guards against the two paths drifting back apart (e.g. dropping the
        metadata field or diverging on the format tag).
        """
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        regions = Regions(
            [
                Region(
                    name="box",
                    data=poly,
                    kind="polygon",
                    metadata={"color": "red", "nested": {"k": [1, 2]}},
                ),
                Region(name="pt", data=(5.0, 10.0), kind="point", metadata={"id": 7}),
            ]
        )

        func_path = tmp_path / "func.json"
        method_path = tmp_path / "method.json"
        regions_to_json(regions, func_path)
        regions.to_json(method_path)

        assert func_path.read_text(encoding="utf-8") == method_path.read_text(
            encoding="utf-8"
        )

        # Both loaders reconstruct equivalent collections, metadata included.
        # Stored metadata freezes lists to tuples; to_dict thaws them back.
        loaded_func = regions_from_json(func_path)
        loaded_method = Regions.from_json(method_path)
        assert loaded_func.list_names() == loaded_method.list_names()
        assert loaded_func["box"].to_dict()["metadata"]["nested"]["k"] == [1, 2]
        assert loaded_method["box"].to_dict()["metadata"]["nested"]["k"] == [1, 2]
        assert loaded_func["pt"].metadata["id"] == 7

    def test_overwrites_existing_file(self, tmp_path):
        json_path = tmp_path / "regions.json"
        regions_to_json(Regions([Region("a", "point", (1.0, 2.0))]), json_path)
        regions_to_json(Regions([Region("b", "point", (3.0, 4.0))]), json_path)

        assert regions_from_json(json_path).list_names() == ["b"]

    def test_non_finite_value_raises_naming_region(self, tmp_path):
        try:
            bad = Region("bad", "point", (np.nan, 1.0))
        except ValueError:
            pytest.skip("Region already refuses non-finite coordinates")
        regions = Regions([Region("ok", "point", (0.0, 0.0)), bad])
        json_path = tmp_path / "nan.json"

        with pytest.raises(ValueError, match="region 'bad' holds a NaN"):
            regions_to_json(regions, json_path)
        assert not json_path.exists()

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (
                {"format": "CustomFormat-v1", "regions": []},
                "format tag 'CustomFormat-v1'",
            ),
            ({"regions": []}, "no format tag"),
            ([{"name": "a"}], "JSON list, not a regions file"),
            ({"format": "Regions-v1"}, 'no "regions" list'),
            ({"format": "Regions-v1", "regions": [{"name": "a"}]}, "entry 0"),
        ],
    )
    def test_refuses_other_files(self, tmp_path, payload, match):
        json_path = tmp_path / "other.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            regions_from_json(json_path)


def _write_labelme(tmp_path, shapes, **extra):
    json_path = tmp_path / "frame.json"
    json_path.write_text(json.dumps({"shapes": shapes, **extra}), encoding="utf-8")
    return json_path


def _shape(label, points, shape_type="polygon"):
    return {"label": label, "points": points, "shape_type": shape_type}


class TestLoadLabelmeJson:
    """load_labelme_json reads each LabelMe shape type as its geometry."""

    def test_polygon_geometry_and_metadata(self, tmp_path):
        json_path = _write_labelme(
            tmp_path,
            [_shape("reward_zone", [[10, 10], [20, 10], [20, 20], [10, 20]])],
            imagePath="frame_0001.png",
        )

        region = load_labelme_json(json_path)["reward_zone"]

        assert region.kind == "polygon"
        assert region.data.bounds == (10.0, 10.0, 20.0, 20.0)
        assert region.data.area == 100.0
        assert dict(region.metadata) == {
            "source_json": "frame.json",
            "source_image": "frame_0001.png",
            "coord_system": "pixels",
            "label": "reward_zone",
            "shape_type": "polygon",
        }

    def test_missing_shape_type_is_a_polygon(self, tmp_path):
        json_path = _write_labelme(
            tmp_path, [{"label": "zone", "points": [[0, 0], [4, 0], [4, 3]]}]
        )

        assert load_labelme_json(json_path)["zone"].data.area == 6.0

    def test_rectangle_from_any_two_opposite_corners(self, tmp_path):
        json_path = _write_labelme(
            tmp_path, [_shape("box", [[30, 5], [10, 25]], "rectangle")]
        )

        box = load_labelme_json(json_path)["box"].data

        assert box.bounds == (10.0, 5.0, 30.0, 25.0)
        assert box.area == 400.0

    def test_circle_from_centre_and_rim_point(self, tmp_path):
        json_path = _write_labelme(
            tmp_path, [_shape("well", [[50, 40], [53, 44]], "circle")]
        )

        circle = load_labelme_json(json_path)["well"].data

        assert_allclose(circle.centroid.coords[0], (50.0, 40.0), atol=1e-9)
        assert_allclose(circle.bounds, (45.0, 35.0, 55.0, 45.0))
        assert circle.area == pytest.approx(np.pi * 25.0, rel=0.01)

    def test_oriented_rectangle(self, tmp_path):
        corners = [[0, 1], [1, 0], [3, 2], [2, 3]]
        json_path = _write_labelme(
            tmp_path, [_shape("tilted", corners, "oriented_rectangle")]
        )

        assert load_labelme_json(json_path)["tilted"].data.area == pytest.approx(4.0)

    def test_point_and_points(self, tmp_path):
        json_path = _write_labelme(
            tmp_path,
            [
                _shape("start", [[50, 60]], "point"),
                _shape("cue", [[1, 2], [3, 4]], "points"),
            ],
        )

        regions = load_labelme_json(json_path)

        assert regions.list_names() == ["start", "cue_0", "cue_1"]
        assert_allclose(regions["start"].data, (50.0, 60.0))
        assert_allclose(regions["cue_1"].data, (3.0, 4.0))
        assert regions["cue_1"].metadata["label"] == "cue"

    def test_pixel_to_world_applied_after_building_shape(self, tmp_path):
        json_path = _write_labelme(
            tmp_path,
            [
                _shape("well", [[10, 10], [12, 10]], "circle"),
                _shape("start", [[5, 6]], "point"),
            ],
        )

        regions = load_labelme_json(json_path, pixel_to_world=_scale(0.5))

        assert regions["well"].data.area == pytest.approx(np.pi, rel=0.01)
        assert_allclose(regions["start"].data, (2.5, 3.0))
        assert "coord_system" not in regions["well"].metadata

    def test_pixel_to_world_wrong_shape_raises(self, tmp_path):
        json_path = _write_labelme(tmp_path, [_shape("zone", [[0, 0], [4, 0], [4, 3]])])

        with pytest.raises(ValueError, match=r"frame\.json, shape 0"):
            load_labelme_json(json_path, pixel_to_world=lambda p: p[:, :1])

    def test_custom_keys(self, tmp_path):
        json_path = _write_labelme(
            tmp_path, [{"name": "region1", "coords": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        )

        regions = load_labelme_json(json_path, label_key="name", points_key="coords")

        assert regions["region1"].data.area == 1.0

    def test_top_level_list_of_shapes(self, tmp_path):
        json_path = tmp_path / "bare.json"
        json_path.write_text(json.dumps([_shape("a", [[0, 0], [2, 0], [0, 2]])]))

        regions = load_labelme_json(json_path)

        assert regions["a"].data.area == 2.0
        assert "source_image" not in regions["a"].metadata

    def test_repeated_labels_never_collide_with_literal_labels(self, tmp_path):
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        json_path = _write_labelme(
            tmp_path,
            [
                _shape("zone", square),
                _shape("zone_0", square),
                _shape("zone", square),
                _shape("arena", square),
            ],
        )

        regions = load_labelme_json(json_path)

        assert regions.list_names() == ["zone_1", "zone_0", "zone_2", "arena"]
        assert regions["zone_0"].metadata["label"] == "zone_0"
        assert regions["zone_2"].metadata["label"] == "zone"

    @pytest.mark.parametrize("shape_type", ["line", "linestrip"])
    def test_open_line_raises_pointing_to_landmark(self, tmp_path, shape_type):
        json_path = _write_labelme(
            tmp_path, [_shape("wall", [[0, 0], [5, 0]], shape_type)]
        )

        with pytest.raises(ValueError, match=r"(?s)shape 0 \(label 'wall'\).*Landmark"):
            load_labelme_json(json_path)

    @pytest.mark.parametrize("shape_type", ["mask", "ellipse"])
    def test_unsupported_shape_type_raises(self, tmp_path, shape_type):
        json_path = _write_labelme(
            tmp_path, [_shape("blob", [[0, 0], [5, 5]], shape_type)]
        )

        with pytest.raises(ValueError, match=f"shape_type '{shape_type}'"):
            load_labelme_json(json_path)

    @pytest.mark.parametrize(
        ("shapes", "match"),
        [
            ([], "has no shapes"),
            (["not a dict"], "shape 0 is a str"),
            ([{"points": [[0, 0], [1, 0], [1, 1]]}], "shape 0 has label None"),
            ([{"label": "", "points": [[0, 0], [1, 0], [1, 1]]}], "label ''"),
            ([{"label": "zone"}], "no \\(x, y\\) point list"),
            ([_shape("zone", [[10], [20]])], "no \\(x, y\\) point list"),
            ([_shape("zone", [[0, 0], [1], [2, 2]])], "no \\(x, y\\) point list"),
            ([_shape("zone", [[0, 0], [1, 1]])], "2 points; it needs at least 3"),
            ([_shape("box", [[0, 0]], "rectangle")], "it needs exactly 2"),
            ([_shape("box", [[0, 0], [0, 5]], "rectangle")], "not a valid area"),
            (
                [_shape("bowtie", [[0, 0], [2, 2], [2, 0], [0, 2]])],
                "not a valid area: Self-intersection",
            ),
        ],
    )
    def test_malformed_shapes_raise(self, tmp_path, shapes, match):
        json_path = _write_labelme(tmp_path, shapes)

        with pytest.raises(ValueError, match=match):
            load_labelme_json(json_path)

    def test_valid_shape_before_malformed_one_is_not_returned(self, tmp_path):
        json_path = _write_labelme(
            tmp_path,
            [_shape("ok", [[0, 0], [1, 0], [1, 1]]), _shape("bad", [[0, 0]])],
        )

        with pytest.raises(ValueError, match="shape 1 \\(label 'bad'\\)"):
            load_labelme_json(json_path)

    def test_missing_shapes_key_raises(self, tmp_path):
        json_path = tmp_path / "other.json"
        json_path.write_text(json.dumps({"annotations": []}))

        with pytest.raises(ValueError, match="has no shapes"):
            load_labelme_json(json_path)


class TestParseCvatPoints:
    """Tests for _parse_cvat_points helper function."""

    def test_basic_parsing(self):
        """Test basic CVAT points string parsing."""
        points_str = "10.0,20.0;30.0,40.0;50.0,60.0"

        result = _parse_cvat_points(points_str)

        expected = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        assert_allclose(result, expected)

    def test_single_point(self):
        """Test parsing single point."""
        points_str = "5.5,10.5"

        result = _parse_cvat_points(points_str)

        expected = np.array([[5.5, 10.5]])
        assert_allclose(result, expected)

    def test_empty_string(self):
        """Test parsing empty string."""
        result = _parse_cvat_points("")

        assert result.shape == (0, 2)

    def test_empty_pair(self):
        """Test handling empty pair in points string."""
        points_str = "10.0,20.0;;30.0,40.0"

        with pytest.warns(UserWarning, match="Empty point pair"):
            result = _parse_cvat_points(points_str)

        # Should skip empty pair
        expected = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert_allclose(result, expected)

    def test_malformed_pair(self):
        """Test error on malformed point pair."""
        points_str = "10.0,20.0;invalid;30.0,40.0"

        with pytest.raises(ValueError, match="Malformed point string"):
            _parse_cvat_points(points_str)

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        points_str = "  10.0 , 20.0 ; 30.0 , 40.0  "

        result = _parse_cvat_points(points_str)

        expected = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert_allclose(result, expected)


class TestRleToMask:
    """Tests for _rle_to_mask helper function."""

    def test_basic_rle(self):
        """Test basic RLE decoding."""
        rle = "0,5,10,3"  # Start at 0, length 5; start at 10, length 3
        height, width = 5, 5

        mask = _rle_to_mask(rle, height, width)

        assert mask.shape == (5, 5)
        # First 5 pixels (row 0) should be 1
        assert np.all(mask.flat[0:5] == 1)
        # Pixels 10-12 should be 1
        assert np.all(mask.flat[10:13] == 1)

    def test_full_mask(self):
        """Test RLE for full mask."""
        rle = "0,25"  # All pixels
        height, width = 5, 5

        mask = _rle_to_mask(rle, height, width)

        assert np.all(mask == 1)

    def test_empty_mask(self):
        """Test RLE for empty mask."""
        rle = ""
        height, width = 5, 5

        with pytest.raises(ValueError, match="non-integer values"):
            _rle_to_mask(rle, height, width)

    def test_invalid_rle_non_integer(self):
        """Test error on non-integer RLE values."""
        rle = "0,5,abc,3"

        with pytest.raises(ValueError, match="non-integer values"):
            _rle_to_mask(rle, 5, 5)

    def test_invalid_rle_odd_length(self):
        """Test error on odd-length RLE."""
        rle = "0,5,10"  # Missing length for second run

        with pytest.raises(ValueError, match="odd number of values"):
            _rle_to_mask(rle, 5, 5)

    def test_rle_negative_start_raises(self):
        """Negative start index is out of bounds, not a wrap-around."""
        with pytest.raises(ValueError, match="out of bounds"):
            _rle_to_mask("-1,3", 5, 5)

    def test_rle_start_past_end_raises(self):
        """Start at or beyond the last pixel is out of bounds."""
        # 5x5 image has 25 pixels (indices 0..24); start=30 is past the end.
        with pytest.raises(ValueError, match="out of bounds"):
            _rle_to_mask("30,2", 5, 5)

    def test_rle_run_overruns_end_raises(self):
        """A run whose end exceeds the image is out of bounds."""
        # start=23, length=5 -> end=28 > 25 pixels.
        with pytest.raises(ValueError, match="out of bounds"):
            _rle_to_mask("23,5", 5, 5)

    def test_rle_negative_length_raises(self):
        """A negative run length is malformed."""
        with pytest.raises(ValueError, match="negative length"):
            _rle_to_mask("0,-3", 5, 5)

    def test_rle_in_bounds_still_works(self):
        """In-bounds runs decode unchanged (no regression)."""
        mask = _rle_to_mask("0,5,10,3", 5, 5)

        assert mask.shape == (5, 5)
        assert np.all(mask.flat[0:5] == 1)
        assert np.all(mask.flat[10:13] == 1)


class TestCvatPolygonProcessor:
    """Tests for the narrowed exception handling in _process_cvat_polygon."""

    def test_cvat_polygon_processor_reraises_unexpected(self):
        """An unexpected (non-geometry) error propagates instead of being swallowed."""
        import xml.etree.ElementTree as ET
        from unittest import mock

        from neurospatial.regions.io import _process_cvat_polygon

        elem = ET.fromstring(
            '<polygon label="arena" points="10.0,10.0;20.0,10.0;20.0,20.0;10.0,20.0" />'
        )

        # shapely.Polygon raising KeyError simulates a real programming bug,
        # which must surface rather than become a warning + dropped shape.
        with (
            mock.patch(
                "neurospatial.regions.io.shp.Polygon", side_effect=KeyError("boom")
            ),
            pytest.raises(KeyError),
        ):
            _process_cvat_polygon(elem, 0, "img0", None, {})

    def test_cvat_polygon_processor_skips_bad_geometry(self):
        """A malformed-geometry error still warns and returns None (intended skip)."""
        import xml.etree.ElementTree as ET

        from neurospatial.regions.io import _process_cvat_polygon

        # Only two distinct points -> shapely cannot build a valid polygon ring.
        elem = ET.fromstring('<polygon label="arena" points="bad,coords" />')

        with pytest.warns(UserWarning, match="error processing"):
            result = _process_cvat_polygon(elem, 0, "img0", None, {})

        assert result is None


class TestCvatBoxProcessor:
    """Tests for the narrowed exception handling in _process_cvat_box."""

    def test_cvat_box_processor_reraises_unexpected(self):
        """An unexpected (non-geometry) error propagates instead of being swallowed."""
        import xml.etree.ElementTree as ET
        from unittest import mock

        from neurospatial.regions.io import _process_cvat_box

        elem = ET.fromstring(
            '<box label="arena" xtl="10.0" ytl="10.0" xbr="20.0" ybr="20.0" />'
        )

        # shapely.Polygon raising KeyError simulates a real programming bug,
        # which must surface rather than become a warning + dropped shape.
        with (
            mock.patch(
                "neurospatial.regions.io.shp.Polygon", side_effect=KeyError("boom")
            ),
            pytest.raises(KeyError),
        ):
            _process_cvat_box(elem, 0, "img0", None, {})

    def test_cvat_box_processor_skips_bad_geometry(self):
        """A malformed-coordinate error still warns and returns None (intended skip)."""
        import xml.etree.ElementTree as ET

        from neurospatial.regions.io import _process_cvat_box

        # Non-numeric coordinate -> float() raises ValueError (a real skip).
        elem = ET.fromstring(
            '<box label="arena" xtl="bad" ytl="10.0" xbr="20.0" ybr="20.0" />'
        )

        with pytest.warns(UserWarning, match="error processing"):
            result = _process_cvat_box(elem, 0, "img0", None, {})

        assert result is None


class TestLoadCvatXml:
    """Tests for load_cvat_xml function."""

    def test_basic_polygon(self, tmp_path):
        """Test loading basic polygon from CVAT XML."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <polygon label="arena" points="100.0,100.0;200.0,100.0;200.0,200.0;100.0,200.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert isinstance(regions, Regions)
        assert len(regions) == 1
        assert "arena" in regions
        assert regions["arena"].kind == "polygon"

    def test_basic_point(self, tmp_path):
        """Test loading point from CVAT XML."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <points label="feeder" points="150.0,150.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "points.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert len(regions) == 1
        assert "feeder" in regions
        assert regions["feeder"].kind == "point"

    def test_box_element(self, tmp_path):
        """Test loading box element (converted to polygon)."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <box label="bbox" xtl="10" ytl="20" xbr="50" ybr="60" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "box.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert len(regions) == 1
        assert "bbox" in regions
        assert regions["bbox"].kind == "polygon"

    def test_nonexistent_file(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_cvat_xml("nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        """Test error on malformed XML."""
        import xml.etree.ElementTree as ET

        xml_path = tmp_path / "bad.xml"
        xml_path.write_text("<unclosed>")

        with pytest.raises(ET.ParseError):
            load_cvat_xml(xml_path)

    def test_unlabeled_shapes(self, tmp_path):
        """Test handling of unlabeled shapes."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <polygon points="10.0,10.0;20.0,10.0;20.0,20.0;10.0,20.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "unlabeled.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert len(regions) == 1
        assert "unlabeled" in regions

    def test_duplicate_labels(self, tmp_path):
        """Test naming with duplicate labels."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <polygon label="zone" points="10.0,10.0;20.0,10.0;20.0,20.0;10.0,20.0" />
    <polygon label="zone" points="30.0,30.0;40.0,30.0;40.0,40.0;30.0,40.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "duplicates.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert len(regions) == 2
        assert "zone_0" in regions
        assert "zone_1" in regions

    def test_closed_polyline(self, tmp_path):
        """Test closed polyline converted to polygon."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <polyline label="closed" points="10.0,10.0;20.0,10.0;20.0,20.0;10.0,10.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "polyline.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert len(regions) == 1
        assert "closed" in regions
        assert regions["closed"].kind == "polygon"

    def test_open_polyline_skipped(self, tmp_path):
        """Test open polyline is skipped."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="test.png" width="640" height="480">
    <polyline label="open" points="10.0,10.0;20.0,10.0;30.0,20.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "open_polyline.xml"
        xml_path.write_text(xml_content)

        with pytest.warns(UserWarning, match="open or too few points"):
            regions = load_cvat_xml(xml_path)

        assert len(regions) == 0

    def test_label_colors(self, tmp_path):
        """Test that label colors are extracted."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <labels>
      <label>
        <name>arena</name>
        <color>#ff0000</color>
      </label>
    </labels>
  </meta>
  <image id="0" name="test.png" width="640" height="480">
    <polygon label="arena" points="10.0,10.0;20.0,10.0;20.0,20.0;10.0,20.0" />
  </image>
</annotations>
"""
        xml_path = tmp_path / "colors.xml"
        xml_path.write_text(xml_content)

        regions = load_cvat_xml(xml_path)

        assert "arena" in regions
        assert regions["arena"].metadata.get("color") == "#ff0000"


def _scale(factor: float):
    """A pixel_to_world transform that scales both axes by ``factor``."""
    return lambda points: np.asarray(points, dtype=float) * factor


class TestMaskToRegion:
    """mask_to_region traces exact pixel cells into one polygon."""

    def test_outline_follows_pixel_cells_and_keeps_holes(self):
        mask = np.zeros((7, 9), dtype=bool)
        mask[1:6, 2:8] = True
        mask[3, 4] = False  # one-pixel hole

        region = mask_to_region(mask, region_name="arena")

        polygon = region.data
        assert region.kind == "polygon"
        assert polygon.bounds == (1.5, 0.5, 7.5, 5.5)
        assert polygon.area == mask.sum()
        assert len(polygon.interiors) == 1
        assert Polygon(polygon.interiors[0]).bounds == (3.5, 2.5, 4.5, 3.5)
        for y, x in np.ndindex(mask.shape):
            assert polygon.contains(Point(x, y)) == mask[y, x]

    @pytest.mark.parametrize(
        ("mask", "bounds"),
        [
            (np.array([[False, True, False]]), (0.5, -0.5, 1.5, 0.5)),
            (np.ones((1, 4), dtype=bool), (-0.5, -0.5, 3.5, 0.5)),
            (np.ones((3, 1), dtype=bool), (-0.5, -0.5, 0.5, 2.5)),
        ],
    )
    def test_single_pixel_row_and_column(self, mask, bounds):
        region = mask_to_region(mask, region_name="thin")

        assert region.data.bounds == bounds
        assert region.data.area == mask.sum()

    def test_zero_one_integer_mask(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1

        region = mask_to_region(mask, region_name="zone")

        assert region.data.bounds == (0.5, 0.5, 2.5, 2.5)

    def test_pixel_to_world_maps_exterior_and_holes(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False

        region = mask_to_region(mask, region_name="ring", pixel_to_world=_scale(2.0))

        assert region.data.bounds == (-1.0, -1.0, 9.0, 9.0)
        assert Polygon(region.data.interiors[0]).bounds == (3.0, 3.0, 5.0, 5.0)
        assert region.data.area == pytest.approx(4 * mask.sum())
        assert "coord_system" not in region.metadata

    def test_pixel_metadata(self):
        region = mask_to_region(np.ones((2, 2), dtype=bool), region_name="r")

        assert region.metadata["source"] == "mask"
        assert region.metadata["coord_system"] == "pixels"

    def test_simplify_keeps_hole_and_rejects_negative_tolerance(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:28, 2:28] = True
        mask[10:20, 10:20] = False

        region = mask_to_region(mask, region_name="r", simplify_tolerance=1.0)

        assert len(region.data.interiors) == 1
        with pytest.raises(ValueError, match="finite and nonnegative"):
            mask_to_region(mask, region_name="r", simplify_tolerance=-1.0)

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="no foreground"):
            mask_to_region(np.zeros((4, 4), dtype=bool), region_name="empty")

    @pytest.mark.parametrize(
        "mask",
        [
            np.eye(3, dtype=bool),  # corner-touching pixels are separate
            np.array([[1, 0, 1]], dtype=bool),
        ],
    )
    def test_several_components_raise_with_count(self, mask):
        n_areas = int(mask.sum())
        with pytest.raises(ValueError, match=f"{n_areas} separate areas"):
            mask_to_region(mask, region_name="split")

    @pytest.mark.parametrize(
        ("mask", "match"),
        [
            (np.ones(4, dtype=bool), "2-D image"),
            (np.ones((2, 2, 3), dtype=bool), "2-D image"),
            (np.zeros((0, 3), dtype=bool), "2-D image"),
            (np.full((2, 2), 255, dtype=np.uint8), "only 0 and 1"),
            (np.array([[1.0, np.nan]]), "only 0 and 1"),
            (np.array([["a", "b"]]), "only 0 and 1"),
        ],
    )
    def test_rejects_non_mask_input(self, mask, match):
        with pytest.raises(ValueError, match=match):
            mask_to_region(mask, region_name="bad")


class TestPixelToWorldValidation:
    """Every loader checks pixel_to_world output through one helper."""

    def test_wrong_shape_raises(self):
        with pytest.raises(
            ValueError, match=r"returned shape \(\d+,\) .*expected \(\d+, 2\)"
        ):
            mask_to_region(
                np.ones((2, 2), dtype=bool),
                region_name="r",
                pixel_to_world=lambda points: points[:, 0],
            )

    def test_non_finite_raises(self):
        def to_nan(points):
            out = np.asarray(points, dtype=float).copy()
            out[0] = np.nan
            return out

        with pytest.raises(ValueError, match="non-finite coordinates for 1 of"):
            mask_to_region(
                np.ones((2, 2), dtype=bool), region_name="r", pixel_to_world=to_nan
            )


class TestRegionsToDataframe:
    """Tests for regions_to_dataframe function."""

    def test_basic_conversion(self):
        """Test converting regions to DataFrame."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        regions = Regions(
            [
                Region(name="box", data=poly, kind="polygon"),
                Region(name="pt", data=(5.0, 5.0), kind="point"),
            ]
        )

        df = regions_to_dataframe(regions)

        assert len(df) == 2
        assert "name" in df.columns
        assert "kind" in df.columns
        assert "area" in df.columns

    def test_polygon_area(self):
        """Test that polygon area is calculated."""
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # Area = 4
        regions = Regions([Region(name="square", data=poly, kind="polygon")])

        df = regions_to_dataframe(regions)

        assert df.loc[0, "area"] == 4.0

    def test_point_area_zero(self):
        """Test that point regions have zero area."""
        regions = Regions([Region(name="pt", data=(5.0, 5.0), kind="point")])

        df = regions_to_dataframe(regions)

        assert df.loc[0, "area"] == 0.0

    def test_empty_regions(self):
        """Test converting empty regions collection."""
        regions = Regions([])

        df = regions_to_dataframe(regions)

        assert len(df) == 0
        assert isinstance(df.columns, object)  # Has columns even if empty
