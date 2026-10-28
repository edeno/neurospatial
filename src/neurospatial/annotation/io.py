"""Import annotations from external tools (LabelMe, CVAT)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurospatial.ops.transforms import VideoCalibration
    from neurospatial.regions import Regions


def regions_from_labelme(
    json_path: str | Path,
    *,
    calibration: VideoCalibration | None = None,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """Load regions from LabelMe JSON with optional calibration.

    Shape types, region naming and errors follow
    :func:`neurospatial.regions.load_labelme_json`.

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
        Loaded regions with coordinates in cm (if calibrated) or in LabelMe's
        image pixels (x = column, y = row pointing down).
        ``metadata["coord_system"]`` is ``"cm"`` or ``"pixels"``.

    See Also
    --------
    neurospatial.regions.load_labelme_json : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_labelme  # doctest: +SKIP
    >>> from neurospatial.ops.transforms import (  # doctest: +SKIP
    ...     VideoCalibration,
    ...     calibrate_from_scale_bar,
    ... )
    >>> # Without calibration (pixel coordinates)
    >>> regions = regions_from_labelme("annotations.json")  # doctest: +SKIP
    >>> # With calibration (cm coordinates)
    >>> transform = calibrate_from_scale_bar(
    ...     (0, 0), (100, 0), 50.0, (640, 480)
    ... )  # doctest: +SKIP
    >>> calib = VideoCalibration(transform, (640, 480))  # doctest: +SKIP
    >>> regions = regions_from_labelme(
    ...     "annotations.json", calibration=calib
    ... )  # doctest: +SKIP

    """
    from neurospatial.regions.io import load_labelme_json

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    regions = load_labelme_json(
        json_path,
        pixel_to_world=pixel_to_world,
        label_key=label_key,
        points_key=points_key,
    )
    return _with_coordinate_units(regions, calibration)


def regions_from_cvat(
    xml_path: str | Path,
    *,
    calibration: VideoCalibration | None = None,
    image: str | int | None = None,
) -> Regions:
    """Load regions from a CVAT for images 1.1 XML export with optional calibration.

    Supported shapes, region naming, image selection and errors follow
    :func:`neurospatial.regions.load_cvat_xml`.

    Parameters
    ----------
    xml_path : str or Path
        Path to CVAT XML export file.
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).
    image : str or int, optional
        The image whose shapes to read: its ``name`` (str) or ``id`` (int).
        If None, the file must have shapes on exactly one image.

    Returns
    -------
    Regions
        Loaded regions with coordinates in cm (if calibrated) or in CVAT's
        image pixels (x = column, y = row pointing down).
        ``metadata["coord_system"]`` is ``"cm"`` or ``"pixels"``.

    See Also
    --------
    neurospatial.regions.load_cvat_xml : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_cvat  # doctest: +SKIP
    >>> regions = regions_from_cvat("cvat_export.xml")  # doctest: +SKIP

    """
    from neurospatial.regions.io import load_cvat_xml

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    regions = load_cvat_xml(xml_path, pixel_to_world=pixel_to_world, image=image)
    return _with_coordinate_units(regions, calibration)


def _with_coordinate_units(
    regions: Regions, calibration: VideoCalibration | None
) -> Regions:
    """Stamp known annotation units while preserving region data and metadata."""
    from neurospatial.regions import Region, Regions

    units = "cm" if calibration is not None else "pixels"
    return Regions(
        [
            Region(
                name=region.name,
                kind=region.kind,
                data=region.data,
                metadata={**region.metadata, "coord_system": units},
            )
            for region in regions.values()
        ]
    )
