"""Import annotations from external tools (LabelMe, CVAT)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurospatial.ops.transforms import VideoCalibration
    from neurospatial.regions import Regions


def regions_from_labelme(
    json_path: str | Path,
    calibration: VideoCalibration | None = None,
    *,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """Load regions from LabelMe JSON with optional calibration.

    Parameters
    ----------
    json_path : str or Path
        Path to LabelMe JSON file.
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).
    label_key : str, default="label"
        Key in JSON for region name.
    points_key : str, default="points"
        Key in JSON for polygon vertices.

    Returns
    -------
    Regions
        Loaded regions with coordinates in cm (if calibrated) or pixels.

    See Also
    --------
    neurospatial.regions.io.load_labelme_json : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_labelme
    >>> from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
    >>> # Without calibration (pixel coordinates)
    >>> regions = regions_from_labelme("annotations.json")
    >>> # With calibration (cm coordinates)
    >>> transform = calibrate_from_scale_bar((0, 0), (100, 0), 50.0, (640, 480))
    >>> calib = VideoCalibration(transform, (640, 480))
    >>> regions = regions_from_labelme("annotations.json", calibration=calib)

    """
    from neurospatial.regions.io import load_labelme_json

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    return load_labelme_json(
        json_path,
        pixel_to_world=pixel_to_world,
        label_key=label_key,
        points_key=points_key,
    )


def regions_from_cvat(
    xml_path: str | Path,
    calibration: VideoCalibration | None = None,
) -> Regions:
    """Load regions from CVAT XML with optional calibration.

    Parameters
    ----------
    xml_path : str or Path
        Path to CVAT XML export file.
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).

    Returns
    -------
    Regions
        Loaded regions with coordinates in cm (if calibrated) or pixels.

    See Also
    --------
    neurospatial.regions.io.load_cvat_xml : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_cvat
    >>> regions = regions_from_cvat("cvat_export.xml")

    """
    from neurospatial.regions.io import load_cvat_xml

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    return load_cvat_xml(xml_path, pixel_to_world=pixel_to_world)
