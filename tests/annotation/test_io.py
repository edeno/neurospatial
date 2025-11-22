"""Tests for annotation IO functions."""

import json

import numpy as np
import pytest

from neurospatial.annotation.io import regions_from_cvat, regions_from_labelme
from neurospatial.transforms import Affine2D, VideoCalibration


class TestRegionsFromLabelme:
    """Tests for regions_from_labelme function."""

    def test_basic_import(self, tmp_path):
        """Import LabelMe JSON without calibration."""
        json_data = {
            "shapes": [
                {
                    "label": "arena",
                    "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
                {
                    "label": "reward",
                    "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                },
            ]
        }
        json_path = tmp_path / "annotations.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(json_path)

        assert len(regions) == 2
        assert "arena" in regions
        assert "reward" in regions

    def test_with_calibration(self, tmp_path):
        """Apply calibration during import."""
        # 2x scale calibration
        scale_matrix = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        json_data = {
            "shapes": [
                {
                    "label": "test",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
            ]
        }
        json_path = tmp_path / "annotations.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(json_path, calibration=calibration)

        poly = regions["test"].data
        bounds = poly.bounds
        assert bounds[2] == pytest.approx(20.0)  # maxx scaled

    def test_custom_keys(self, tmp_path):
        """Use custom label and points keys."""
        json_data = {
            "shapes": [
                {
                    "name": "custom_name",
                    "coords": [[0, 0], [50, 0], [50, 50], [0, 50]],
                },
            ]
        }
        json_path = tmp_path / "custom.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(
            json_path,
            label_key="name",
            points_key="coords",
        )

        assert "custom_name" in regions


class TestRegionsFromCvat:
    """Tests for regions_from_cvat function."""

    def test_basic_import(self, tmp_path):
        """Import CVAT XML without calibration."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
            <image id="0" name="frame_0.png" width="640" height="480">
                <polygon label="arena" points="0,0;100,0;100,100;0,100"/>
                <polygon label="reward" points="10,10;20,10;20,20;10,20"/>
            </image>
        </annotations>
        """
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_text(xml_content)

        regions = regions_from_cvat(xml_path)

        assert len(regions) == 2
        assert "arena" in regions
        assert "reward" in regions

    def test_with_calibration(self, tmp_path):
        """Apply calibration during CVAT import."""
        scale_matrix = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
            <image id="0" name="frame.png" width="640" height="480">
                <polygon label="scaled" points="0,0;100,0;100,100;0,100"/>
            </image>
        </annotations>
        """
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_text(xml_content)

        regions = regions_from_cvat(xml_path, calibration=calibration)

        poly = regions["scaled"].data
        bounds = poly.bounds
        assert bounds[2] == pytest.approx(50.0)  # maxx at 0.5x scale
