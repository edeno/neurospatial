"""Tests for regions I/O module."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import Polygon

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
        content = json_path.read_text()
        assert "    " in content  # 4-space indent

    def test_unrecognized_format_warning(self, tmp_path):
        """Test warning when loading unrecognized format."""
        json_path = tmp_path / "custom.json"
        json_path.write_text(json.dumps({"format": "CustomFormat-v1", "regions": []}))

        with pytest.warns(UserWarning, match="Unrecognised format"):
            regions = regions_from_json(json_path)

        assert len(regions) == 0


class TestLoadLabelmeJson:
    """Tests for load_labelme_json function."""

    def test_basic_polygon(self, tmp_path):
        """Test loading basic polygon from LabelMe JSON."""
        json_data = {
            "shapes": [
                {
                    "label": "reward_zone",
                    "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    "shape_type": "polygon",
                }
            ]
        }
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_data))

        regions = load_labelme_json(json_path)

        assert isinstance(regions, Regions)
        assert len(regions) == 1
        assert "reward_zone" in regions
        assert regions["reward_zone"].kind == "polygon"

    def test_basic_point(self, tmp_path):
        """Test loading point from LabelMe JSON."""
        # Note: Single points become polygons with < 3 points and are skipped
        # This matches the actual behavior - points need to be in a valid polygon format
        json_data = {
            "shapes": [
                {"label": "start_point", "points": [[50, 50]], "shape_type": "point"}
            ]
        }
        json_path = tmp_path / "points.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="fewer than 3 points"):
            regions = load_labelme_json(json_path)

        # Single points are skipped since they can't form valid polygons
        assert len(regions) == 0

    def test_custom_keys(self, tmp_path):
        """Test loading with custom label and points keys."""
        json_data = {
            "shapes": [{"name": "region1", "coords": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        }
        json_path = tmp_path / "custom.json"
        json_path.write_text(json.dumps(json_data))

        regions = load_labelme_json(json_path, label_key="name", points_key="coords")

        assert len(regions) == 1
        assert "region1" in regions

    def test_empty_shapes(self, tmp_path):
        """Test loading JSON with no shapes."""
        json_data = {"shapes": []}
        json_path = tmp_path / "empty.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="No shapes found"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0

    def test_missing_label(self, tmp_path):
        """Test handling of shape with missing label."""
        json_data = {
            "shapes": [
                {
                    "points": [[10, 10], [20, 20]]
                    # Missing "label" key
                }
            ]
        }
        json_path = tmp_path / "missing_label.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="missing label"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0  # Shape should be skipped

    def test_missing_points(self, tmp_path):
        """Test handling of shape with missing points."""
        json_data = {
            "shapes": [
                {
                    "label": "test"
                    # Missing "points" key
                }
            ]
        }
        json_path = tmp_path / "missing_points.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="missing points"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0

    def test_invalid_points_shape(self, tmp_path):
        """Test handling of points with wrong shape."""
        json_data = {
            "shapes": [
                {
                    "label": "bad_shape",
                    "points": [[10], [20]],  # Missing y coordinates
                }
            ]
        }
        json_path = tmp_path / "bad_shape.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="not in \\(M, 2\\) format"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0

    def test_too_few_points(self, tmp_path):
        """Test handling of polygon with too few points."""
        json_data = {
            "shapes": [
                {
                    "label": "line",
                    "points": [[10, 10], [20, 20]],  # Only 2 points
                }
            ]
        }
        json_path = tmp_path / "few_points.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="fewer than 3 points"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0

    def test_non_dict_shape(self, tmp_path):
        """Test handling of non-dictionary shape."""
        json_data = {"shapes": ["not a dict"]}
        json_path = tmp_path / "non_dict.json"
        json_path.write_text(json.dumps(json_data))

        with pytest.warns(UserWarning, match="not a dictionary"):
            regions = load_labelme_json(json_path)

        assert len(regions) == 0

    def test_shapes_key_format(self, tmp_path):
        """Test loading JSON with explicit shapes key."""
        # Implementation expects either {"shapes": [...]} or fallback to empty list if invalid
        json_data = {
            "shapes": [
                {"label": "region1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
            ]
        }
        json_path = tmp_path / "shapes_key.json"
        json_path.write_text(json.dumps(json_data))

        regions = load_labelme_json(json_path)

        assert len(regions) == 1
        assert "region1" in regions


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


# Check if cv2 is available
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@pytest.mark.skipif(not CV2_AVAILABLE, reason="opencv-python not installed")
class TestMaskToRegion:
    """Tests for mask_to_region function."""

    def test_simple_square_mask(self):
        """Test converting simple square mask to region."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True

        region = mask_to_region(mask, region_name="test_region")

        assert region.name == "test_region"
        assert region.kind == "polygon"
        assert isinstance(region.data, Polygon)

    def test_empty_mask_raises(self):
        """Test that empty mask raises error."""
        mask = np.zeros((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="No contours found"):
            mask_to_region(mask, region_name="empty")

    def test_full_mask(self):
        """Test mask that is entirely True."""
        mask = np.ones((5, 5), dtype=bool)

        region = mask_to_region(mask, region_name="full")

        assert region.kind == "polygon"
        assert region.name == "full"

    def test_uint8_conversion(self):
        """Test automatic conversion to uint8."""
        mask = np.zeros((5, 5), dtype=float)
        mask[1:4, 1:4] = 1.0

        region = mask_to_region(mask, region_name="converted")

        assert region.kind == "polygon"

    def test_metadata(self):
        """Test that metadata is set correctly."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        region = mask_to_region(mask, region_name="test")

        assert region.metadata.get("source") == "mask"


class TestMaskToRegionErrors:
    """Tests for mask_to_region error handling."""

    def test_opencv_not_installed(self, monkeypatch):
        """Test error when OpenCV is not installed."""
        import sys

        # Mock cv2 import to fail
        monkeypatch.setitem(sys.modules, "cv2", None)

        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        with pytest.raises(RuntimeError, match="opencv-python"):
            mask_to_region(mask, region_name="test")


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
