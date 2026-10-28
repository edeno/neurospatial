"""regions/io.py
=============

Helper functions for importing *continuous* ROIs (``Region`` / ``Regions``)
from common labelling formats and exporting them to a shareable JSON schema.

"""

from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
import shapely.geometry as shp
from numpy.typing import NDArray
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon

from neurospatial._exceptions import _format_error
from neurospatial._validation import validate_finite_scalar

from .core import Region, Regions

if TYPE_CHECKING:
    import pandas as pd

    from ..ops.transforms import SpatialTransform

# --------------------------------------------------------------------------
# 1.  Generic JSON <--> Regions round-trip
# --------------------------------------------------------------------------
_SCHEMA_TAG = "Regions-v1"


def regions_to_json(regions: Regions, path: str | Path, *, indent: int = 2) -> None:
    """Write regions to a standard JSON file that any language can read.

    The file holds one object, ``{"format": "Regions-v1", "regions": [...]}``,
    with one :meth:`Region.to_dict` entry (name, kind, GeoJSON-style geometry
    and metadata) per region. Missing parent directories are created and an
    existing file is overwritten.

    Parameters
    ----------
    regions : Regions
        Collection of regions to write.
    path : str or Path
        Destination file path.
    indent : int, default=2
        Indentation of the JSON text, in spaces.

    Raises
    ------
    ValueError
        If a region holds a NaN or infinite coordinate or metadata value,
        which standard JSON cannot store.

    See Also
    --------
    regions_from_json : Read the file back.
    """
    payload = {
        "format": _SCHEMA_TAG,
        "regions": [reg.to_dict() for reg in regions.values()],
    }
    try:
        text = json.dumps(payload, indent=indent, allow_nan=False)
    except ValueError as exc:
        bad = next(
            (
                region.name
                for region in regions.values()
                if not _is_standard_json(region.to_dict())
            ),
            None,
        )
        raise ValueError(
            _format_error(
                f"Cannot write {path}: region {bad!r} holds a NaN or infinite "
                "coordinate or metadata value.",
                why="standard JSON has no NaN or Infinity, and regions must "
                "have finite coordinates",
                fix="find and fix the non-finite value in that region's data "
                "or metadata",
            )
        ) from exc
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _is_standard_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except ValueError:
        return False
    return True


def regions_from_json(path: str | Path) -> Regions:
    """Read a regions file written by :func:`regions_to_json`.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    Regions
        The regions, with their metadata, in file order.

    Raises
    ------
    ValueError
        If the file is not a ``"Regions-v1"`` regions file: the top level is
        not an object, the ``format`` tag is missing or different, the
        ``regions`` list is missing, or an entry is malformed.
    json.JSONDecodeError
        If the file is not valid JSON.

    See Also
    --------
    regions_to_json : Write the file.
    load_labelme_json, load_cvat_xml : Read annotation-tool exports.
    """
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    rewrite = (
        "read annotation-tool exports with load_labelme_json or load_cvat_xml, "
        "or rewrite the regions with regions_to_json"
    )
    if not isinstance(blob, dict):
        raise ValueError(
            _format_error(
                f"{path} holds a JSON {type(blob).__name__}, not a regions file.",
                why=f'a regions file is an object with "format": "{_SCHEMA_TAG}" '
                'and a "regions" list',
                fix=rewrite,
            )
        )
    found = blob.get("format")
    if found != _SCHEMA_TAG:
        what = (
            f"{path} has no format tag."
            if found is None
            else f"{path} has format tag {found!r}."
        )
        raise ValueError(
            _format_error(
                what,
                why=f"only {_SCHEMA_TAG!r} files, written by regions_to_json, are read",
                fix=rewrite,
            )
        )
    entries = blob.get("regions")
    if not isinstance(entries, list):
        raise ValueError(
            _format_error(
                f'{path} has no "regions" list.',
                fix="rewrite the regions with regions_to_json",
            )
        )
    regions = []
    for index, entry in enumerate(entries):
        try:
            regions.append(Region.from_dict(entry))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                _format_error(
                    f"Region entry {index} in {path} is malformed: {exc!r}.",
                    why="each entry needs the name, kind, geom and metadata "
                    "keys written by Region.to_dict",
                    fix="rewrite the regions with regions_to_json",
                )
            ) from exc
    return Regions(regions)


# --------------------------------------------------------------------------
# 2.  LabelMe / CVAT / VIA-style polygon JSON/XML  → Regions
# --------------------------------------------------------------------------
def load_labelme_json(
    json_path: str | Path,
    *,
    pixel_to_world: SpatialTransform | None = None,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """Parse a *.json* file produced by many point-&-click ROI tools.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file.
    pixel_to_world : SpatialTransform, optional
        Callable mapping `(N, 2)` **pixel** coordinates → `(N, 2)` **world**
        coordinates (e.g., centimeters). If `None`, the coordinates are
        assumed to be in pixels.
    label_key : str, optional
        Key in the JSON object that contains the region label.
        Default is "label".
    points_key : str, optional
        Key in the JSON object that contains the list of points.
        Default is "points".

    Returns
    -------
    Regions
        Every polygon becomes a :class:`Region` with ``kind="polygon"``.

    Raises
    ------
    ValueError
        If no shapes are found in the JSON file or if a shape is missing
        required keys.

    """
    data = json.loads(Path(json_path).read_text())
    regions_list: list[Region] = []  # Renamed to avoid conflict with Regions class

    # LabelMe puts shapes in a top-level list; CVAT stores under "shapes"
    shapes_data = data.get("shapes", data if isinstance(data, list) else [])
    if not isinstance(shapes_data, list):  # Ensure shapes_data is a list
        raise ValueError(
            f"Expected 'shapes' to be a list, but got {type(shapes_data).__name__} in {json_path}",
        )
    if not shapes_data:
        warnings.warn(
            f"No shapes found in {json_path}",
            category=UserWarning,
            stacklevel=2,
        )  # Changed to warning, or could be error
        return Regions([])  # Return empty Regions if no shapes

    for i, obj in enumerate(shapes_data):
        if not isinstance(obj, dict):
            warnings.warn(
                f"Shape at index {i} is not a dictionary, skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        name = obj.get(label_key)
        points_data = obj.get(points_key)

        if name is None:
            warnings.warn(
                f"Shape at index {i} is missing label (key: '{label_key}'), skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue
        if points_data is None:
            warnings.warn(
                f"Shape '{name}' (index {i}) is missing points (key: '{points_key}'), skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        try:
            pts_px: NDArray[np.float64] = np.asarray(points_data, dtype=float)  # (M, 2)
            if pts_px.ndim != 2 or pts_px.shape[1] != 2:
                warnings.warn(
                    f"Points for shape '{name}' (index {i}) are not in (M, 2) format, skipping.",
                    category=UserWarning,
                    stacklevel=2,
                )
                continue
        except ValueError as e:
            warnings.warn(
                f"Could not parse points for shape '{name}' (index {i}): {e}, skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        # Ensure polygon has at least 3 points for a valid polygon
        if len(pts_transformed) < 3:
            warnings.warn(
                f"Shape '{name}' (index {i}) has fewer than 3 points after processing, skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue
        try:
            poly = shp.Polygon(pts_transformed)
        except (TypeError, ValueError, ShapelyError) as e:  # Polygon construction
            warnings.warn(
                f"Could not create polygon for shape '{name}' (index {i}): {e}, skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        regions_list.append(
            Region(
                name=str(name),  # Ensure name is a string
                kind="polygon",
                data=poly,
                metadata={"source_json": Path(json_path).name},
            ),
        )
    return Regions(regions_list)


def _rle_to_mask(rle: str, height: int, width: int) -> NDArray[np.uint8]:
    """Convert a Run-Length Encoded string to a binary mask.

    Parameters
    ----------
    rle : str
        The Run-Length Encoding string, e.g., "start1,length1,start2,length2,...".
        Values are 0-indexed and refer to the flattened image.
    height : int
        The height of the image (in pixels).
    width : int
        The width of the image (in pixels).

    Returns
    -------
    mask : NDArray[np.uint8], shape (height, width)
        A binary mask where 1 represents the mask area and 0 is background.

    Raises
    ------
    ValueError
        If RLE string is malformed or values are out of bounds.

    """
    try:
        rle_values = list(map(int, rle.split(",")))
    except ValueError:
        raise ValueError(f"RLE string contains non-integer values: {rle}") from None

    if len(rle_values) % 2 != 0:
        raise ValueError(f"RLE string has an odd number of values: {rle}")

    mask = np.zeros(height * width, dtype=np.uint8)
    n_pixels = height * width

    # Unpack RLE values into the mask. Each (start, length) run must lie fully
    # within the flattened image; out-of-bounds or negative runs are a malformed
    # encoding (a clamped NumPy slice would silently produce the wrong mask).
    for start, length in zip(rle_values[::2], rle_values[1::2], strict=True):
        if length < 0:
            raise ValueError(
                f"RLE run has negative length {length} (start={start}): {rle}"
            )
        if start < 0 or start + length > n_pixels:
            raise ValueError(
                f"RLE run [{start}, {start + length}) is out of bounds for an "
                f"image of {n_pixels} pixels ({height}x{width}): {rle}"
            )
        mask[start : start + length] = 1  # Set the corresponding region to 1

    return mask.reshape((height, width))  # Reshape to image dimensions


def _mask_to_polygon(mask: NDArray[np.uint8]) -> Polygon:
    """Converts a binary mask to a Shapely Polygon.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        A binary mask (2D array).

    Returns
    -------
    shapely.geometry.Polygon
        A Shapely Polygon representing the mask.

    """
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        # Take the largest contour if there are multiple
        contour = max(contours, key=cv2.contourArea)
        return Polygon(contour.reshape(-1, 2))  # Convert to Polygon
    return Polygon()  # Return an empty polygon if no contour found


def _parse_cvat_points(
    points_str: str,
    shape_label: str | None = None,
    shape_index: int | None = None,
) -> NDArray[np.float64]:
    """Parse a CVAT-style points string into a NumPy array.

    CVAT points are formatted as "x1,y1;x2,y2;...;xn,yn".

    Parameters
    ----------
    points_str : str
        String containing points in the format "x1,y1;x2,y2;...;xn,yn".
    shape_label : str, optional
        Label of the shape these points belong to (for warning messages).
    shape_index : int, optional
        Index of the shape these points belong to (for warning messages).


    Returns
    -------
    parsed_points : NDArray[np.float64], shape (N, 2)
        Parsed points as floats.

    Raises
    ------
    ValueError
        If the points string is malformed.

    """
    parsed_points: list[list[float]] = []
    if not points_str.strip():  # Handle empty string case
        return np.empty((0, 2), dtype=float)

    point_pairs = points_str.strip().split(";")
    for i, pt_pair_str in enumerate(point_pairs):
        if not pt_pair_str.strip():  # Handle empty pair (e.g., "x,y;;x,y")
            warning_msg = "Empty point pair found in points string"
            if shape_label and shape_index is not None:
                warning_msg += f" for shape '{shape_label}' (index {shape_index})"
            warning_msg += f" at pair index {i}."
            warnings.warn(
                warning_msg + " Skipping this pair.", category=UserWarning, stacklevel=2
            )
            continue
        try:
            x_str, y_str = pt_pair_str.split(",")
            parsed_points.append([float(x_str), float(y_str)])
        except ValueError:
            # More informative error message
            context = ""
            if shape_label and shape_index is not None:
                context = f" for shape '{shape_label}' (index {shape_index})"
            raise ValueError(
                f"Malformed point string{context}: "
                f"Could not parse '{pt_pair_str}' into two float coordinates. Full string: '{points_str}'",
            ) from None
    return np.array(parsed_points, dtype=float)


def _create_cvat_region(
    name: str,
    shape_data: Polygon | NDArray[np.float64],  # Polygon or Point data
    kind: str,
    xml_path: Path,
    color: str | None,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Region:
    """Helper function to create a Region object for CVAT shapes."""
    from neurospatial.regions.core import Kind

    if kind not in ("point", "polygon"):
        raise ValueError(f"Invalid kind: {kind!r}")
    # After validation, we know kind is either "point" or "polygon"
    kind_typed = cast("Kind", kind)

    metadata = {
        "source_xml": xml_path.name,
        "color": color,
        "plot_kwargs": {"color": color} if color else {},
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    return Region(name=name, kind=kind_typed, data=shape_data, metadata=metadata)


def _get_label_colors(root: ET.Element) -> dict[str, str]:
    """Extract label colors from CVAT XML root element.

    Parameters
    ----------
    root : ET.Element
        Root element of the CVAT XML tree.

    Returns
    -------
    dict[str, str]
        Mapping of label names to their color values.

    """
    label_colors: dict[str, str] = {}
    labels_elem = root.find(".//labels")
    if labels_elem is not None:
        for label_elem in labels_elem.findall("label"):
            name_elem = label_elem.find("name")
            color_elem = label_elem.find("color")
            if (
                name_elem is not None
                and name_elem.text
                and color_elem is not None
                and color_elem.text
            ):
                label_colors[name_elem.text.strip()] = color_elem.text.strip()
    return label_colors


def _process_cvat_polygon(
    polygon_elem: ET.Element,
    elem_idx: int,
    image_id_str: str,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> tuple[str, shp.Polygon, str | None] | None:
    """Process a CVAT polygon element into geometry.

    Returns None if the polygon should be skipped.
    """
    raw_label = polygon_elem.get("label")
    points_str = polygon_elem.get("points")
    processed_label, color = _get_processed_label_and_color(raw_label, label_colors)

    if not points_str:
        warnings.warn(
            f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): missing 'points'. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None

    try:
        pts_px = _parse_cvat_points(points_str, raw_label, elem_idx)
        if pts_px.shape[0] < 3:
            warnings.warn(
                f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
                f"xml_idx: {elem_idx}): < 3 points. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            return None
        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        geom = shp.Polygon(pts_transformed)
        return (processed_label, geom, color)
    except (ValueError, ShapelyError) as e:
        warnings.warn(
            f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): error processing: {e}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None


def _process_cvat_polyline(
    polyline_elem: ET.Element,
    elem_idx: int,
    image_id_str: str,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> tuple[str, shp.Polygon, str | None] | None:
    """Process a CVAT polyline element, converting closed polylines to polygons.

    Returns None if the polyline should be skipped.
    """
    raw_label = polyline_elem.get("label")
    points_str = polyline_elem.get("points")
    processed_label, color = _get_processed_label_and_color(raw_label, label_colors)

    if not points_str:
        warnings.warn(
            f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): missing 'points'. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None

    try:
        pts_px = _parse_cvat_points(points_str, raw_label, elem_idx)
        if pts_px.shape[0] < 2:
            warnings.warn(
                f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
                f"xml_idx: {elem_idx}): < 2 points. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            return None

        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        if pts_transformed.shape[0] >= 3 and np.allclose(
            pts_transformed[0],
            pts_transformed[-1],
        ):
            geom = shp.Polygon(pts_transformed)
            return (processed_label, geom, color)
        warnings.warn(
            f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): open or too few points for polygon. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None
    except (ValueError, ShapelyError) as e:
        warnings.warn(
            f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): error processing: {e}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None


def _process_cvat_points(
    points_elem: ET.Element,
    elem_idx: int,
    image_id_str: str,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> list[tuple[str, shp.Point, str | None]]:
    """Process a CVAT points element into a list of Point geometries.

    Returns empty list if the points should be skipped.
    """
    raw_label_group = points_elem.get("label")
    points_str = points_elem.get("points")
    processed_label_group, color_group = _get_processed_label_and_color(
        raw_label_group,
        label_colors,
    )

    if not points_str:
        warnings.warn(
            f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
            f"xml_idx: {elem_idx}): missing 'points'. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return []

    try:
        pts_px = _parse_cvat_points(points_str, raw_label_group, elem_idx)
        if pts_px.shape[0] == 0:
            warnings.warn(
                f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
                f"xml_idx: {elem_idx}): no valid points. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            return []

        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        results = []
        for pt_data in pts_transformed:
            geom = shp.Point(pt_data)
            results.append((processed_label_group, geom, color_group))
        return results
    except (ValueError, ShapelyError) as e:
        warnings.warn(
            f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
            f"xml_idx: {elem_idx}): error processing: {e}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return []


def _process_cvat_box(
    box_elem: ET.Element,
    elem_idx: int,
    image_id_str: str,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> tuple[str, shp.Polygon, str | None] | None:
    """Process a CVAT box element into a polygon geometry.

    Returns None if the box should be skipped.
    """
    raw_label = box_elem.get("label")
    processed_label, color = _get_processed_label_and_color(raw_label, label_colors)

    try:
        xtl_str = box_elem.get("xtl")
        ytl_str = box_elem.get("ytl")
        xbr_str = box_elem.get("xbr")
        ybr_str = box_elem.get("ybr")

        if xtl_str is None or ytl_str is None or xbr_str is None or ybr_str is None:
            raise ValueError("Missing box coordinates")

        xtl = float(xtl_str)
        ytl = float(ytl_str)
        xbr = float(xbr_str)
        ybr = float(ybr_str)
        pts_px = np.array([[xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]])
        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        geom = shp.Polygon(pts_transformed)
        return (processed_label, geom, color)
    except (TypeError, ValueError, ShapelyError) as e:
        warnings.warn(
            f"Image '{image_id_str}', Box (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): error processing: {e}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None


def _process_cvat_mask(
    mask_elem: ET.Element,
    elem_idx: int,
    image_id_str: str,
    img_h: int,
    img_w: int,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> tuple[str, shp.Polygon, str | None] | None:
    """Process a CVAT mask element into a polygon geometry.

    Returns None if the mask should be skipped.
    Raises ImportError if cv2 is not available.
    """
    raw_label = mask_elem.get("label")
    rle_str = mask_elem.get("rle")
    processed_label, color = _get_processed_label_and_color(raw_label, label_colors)

    if not rle_str:
        warnings.warn(
            f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): missing 'rle'. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None

    try:
        mask_array = _rle_to_mask(rle_str, img_h, img_w)
        mask_poly_px = _mask_to_polygon(mask_array)
        if mask_poly_px.is_empty:
            warnings.warn(
                f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
                f"xml_idx: {elem_idx}): empty polygon from mask. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            return None

        final_geom: shp.Polygon
        if pixel_to_world:
            coords_px = np.array(mask_poly_px.exterior.coords)
            coords_transformed = pixel_to_world(coords_px)
            interiors_transformed = [
                pixel_to_world(np.array(i.coords)) for i in mask_poly_px.interiors
            ]
            final_geom = shp.Polygon(coords_transformed, holes=interiors_transformed)
        else:
            final_geom = mask_poly_px

        return (processed_label, final_geom, color)
    except ImportError:
        # Let this bubble up so caller can handle it
        raise
    except (ValueError, ShapelyError) as e:
        warnings.warn(
            f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
            f"xml_idx: {elem_idx}): error processing: {e}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        return None


def _get_processed_label_and_color(
    raw_label_xml: str | None,
    label_colors: dict[str, str],
) -> tuple[str, str | None]:
    """Process raw label from XML and get its color.

    Parameters
    ----------
    raw_label_xml : str or None
        Raw label string from XML attribute.
    label_colors : dict[str, str]
        Mapping of label names to color values.

    Returns
    -------
    tuple[str, str | None]
        Processed label (or "unlabeled") and associated color.

    """
    processed_label = (
        raw_label_xml.strip()
        if raw_label_xml and raw_label_xml.strip()
        else "unlabeled"
    )
    color = None
    if raw_label_xml and raw_label_xml.strip():
        color = label_colors.get(raw_label_xml.strip())
    return processed_label, color


def _collect_shapes_from_image(
    image: ET.Element,
    image_id_str: str,
    pixel_to_world: SpatialTransform | None,
    label_colors: dict[str, str],
) -> list[tuple[str, Any, str, str | None]]:
    """Collect all shapes from a CVAT image element.

    Returns a list of (label, geometry, kind, color) tuples.
    """
    collected_shapes_data: list[tuple[str, Any, str, str | None]] = []

    # Polygons
    for elem_idx, polygon_elem in enumerate(image.findall("polygon")):
        result = _process_cvat_polygon(
            polygon_elem,
            elem_idx,
            image_id_str,
            pixel_to_world,
            label_colors,
        )
        if result:
            label, geom, color = result
            collected_shapes_data.append((label, geom, "polygon", color))

    # Polylines
    for elem_idx, polyline_elem in enumerate(image.findall("polyline")):
        result = _process_cvat_polyline(
            polyline_elem,
            elem_idx,
            image_id_str,
            pixel_to_world,
            label_colors,
        )
        if result:
            label, geom, color = result
            collected_shapes_data.append((label, geom, "polygon", color))

    # Points
    for elem_idx, points_elem in enumerate(image.findall("points")):
        results = _process_cvat_points(
            points_elem,
            elem_idx,
            image_id_str,
            pixel_to_world,
            label_colors,
        )
        for label, geom, color in results:
            collected_shapes_data.append((label, geom, "point", color))

    # Boxes
    for elem_idx, box_elem in enumerate(image.findall("box")):
        result = _process_cvat_box(
            box_elem,
            elem_idx,
            image_id_str,
            pixel_to_world,
            label_colors,
        )
        if result:
            label, geom, color = result
            collected_shapes_data.append((label, geom, "polygon", color))

    # Masks
    image_width_str = image.get("width")
    image_height_str = image.get("height")

    if image_width_str and image_height_str:
        try:
            img_w, img_h = int(image_width_str), int(image_height_str)
            if img_w > 0 and img_h > 0:
                for elem_idx, mask_elem in enumerate(image.findall("mask")):
                    try:
                        result = _process_cvat_mask(
                            mask_elem,
                            elem_idx,
                            image_id_str,
                            img_h,
                            img_w,
                            pixel_to_world,
                            label_colors,
                        )
                        if result:
                            label, geom, color = result
                            collected_shapes_data.append(
                                (label, geom, "polygon", color)
                            )
                    except ImportError:
                        warnings.warn(
                            f"Image '{image_id_str}', Mask (xml_idx: {elem_idx}): "
                            f"OpenCV (cv2) missing. Skipping remaining masks.",
                            category=UserWarning,
                            stacklevel=2,
                        )
                        break
            else:
                warnings.warn(
                    f"Image '{image_id_str}': non-positive width/height. "
                    f"Cannot process RLE.",
                    category=UserWarning,
                    stacklevel=2,
                )
        except ValueError:
            warnings.warn(
                f"Image '{image_id_str}': invalid width/height. Cannot process RLE.",
                category=UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            f"Image '{image_id_str}': missing width/height. Cannot process RLE.",
            category=UserWarning,
            stacklevel=2,
        )

    return collected_shapes_data


def _generate_region_names(
    collected_shapes_data: list[tuple[str, Any, str, str | None]],
    path_obj: Path,
) -> list[Region]:
    """Generate unique region names and create Region objects.

    Handles the two-pass naming strategy: count label occurrences,
    then generate names with indices for duplicates.
    """
    # Count label occurrences
    label_total_counts: dict[str, int] = {}
    for label, _, _, _ in collected_shapes_data:
        label_total_counts[label] = label_total_counts.get(label, 0) + 1

    # Generate unique names and create regions
    running_indices_for_multi_labels: dict[str, int] = {}
    regions: list[Region] = []

    for label, geom, kind, color in collected_shapes_data:
        total_occurrences = label_total_counts[label]
        if total_occurrences == 1:
            region_name = label
        else:
            current_idx = running_indices_for_multi_labels.get(label, 0)
            region_name = f"{label}_{current_idx}"
            running_indices_for_multi_labels[label] = current_idx + 1

        regions.append(_create_cvat_region(region_name, geom, kind, path_obj, color))

    return regions


def load_cvat_xml(
    xml_path: str | Path,
    *,
    pixel_to_world: SpatialTransform | None = None,
) -> Regions:
    """Parse a *.xml* file produced by CVAT.

    Region names are generated based on their label. If a label appears
    multiple times within an image, a running index is appended (e.g., tumor_0,
    tumor_1). If a label is unique within an image, no index is added (e.g., tumor).
    Unlabeled shapes follow the same logic (e.g., unlabeled or unlabeled_0).
    Open polylines are ignored. Other shapes are converted to Regions.

    Parameters
    ----------
    xml_path : str or Path
        Path to the CVAT XML file.
    pixel_to_world : SpatialTransform, optional
        Callable mapping `(N,2)` pixel coords → `(N,2)` world coords.

    Returns
    -------
    Regions
        A collection of :class:`Region` objects.

    Raises
    ------
    FileNotFoundError
        If `xml_path` does not exist.
    ET.ParseError
        If `xml_path` is not a valid XML file.

    """
    path_obj = Path(xml_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(path_obj)
    except ET.ParseError as e:
        raise ET.ParseError(f"Error parsing XML file {xml_path}: {e}") from e

    root = tree.getroot()
    label_colors = _get_label_colors(root)

    all_regions_in_file: list[Region] = []
    for image_idx, image in enumerate(root.findall("image")):
        image_id_str = image.get("id", f"image{image_idx}")

        # Collect all shapes from this image
        collected_shapes_data = _collect_shapes_from_image(
            image,
            image_id_str,
            pixel_to_world,
            label_colors,
        )

        # Generate unique names and create Region objects
        regions = _generate_region_names(collected_shapes_data, path_obj)
        all_regions_in_file.extend(regions)

    return Regions(all_regions_in_file)


# --------------------------------------------------------------------------
# 3.  Binary mask  → single Region
# --------------------------------------------------------------------------
def _pixels_to_world(
    points_px: NDArray[np.float64],
    pixel_to_world: SpatialTransform | None,
    *,
    where: str,
) -> NDArray[np.float64]:
    """Apply ``pixel_to_world`` to ``(n, 2)`` pixel points and check its output.

    Returns ``points_px`` unchanged when ``pixel_to_world`` is None. ``where``
    names the shape being converted for error messages.
    """
    if pixel_to_world is None:
        return points_px
    world = np.asarray(pixel_to_world(points_px), dtype=np.float64)
    if world.shape != points_px.shape:
        raise ValueError(
            _format_error(
                f"pixel_to_world returned shape {world.shape} for {where}; "
                f"expected {points_px.shape}, one (x, y) row per pixel point.",
                why="regions are 2-D, so each pixel (x, y) must map to one "
                "world (x, y)",
                fix="pass a 2-D transform such as "
                "VideoCalibration.transform_px_to_cm or a 2-D AffineND",
            )
        )
    if not np.isfinite(world).all():
        n_bad = int((~np.isfinite(world).all(axis=1)).sum())
        raise ValueError(
            _format_error(
                f"pixel_to_world returned non-finite coordinates for {n_bad} "
                f"of {len(world)} points of {where}.",
                why="a region needs finite world coordinates",
                fix="check that the transform is defined over the whole image, "
                "for example that a homography does not send these pixels to "
                "infinity",
            )
        )
    return world


def _polygon_to_world(
    polygon: Polygon, pixel_to_world: SpatialTransform | None, *, where: str
) -> Polygon:
    """Map every ring of a pixel polygon, exterior and holes, to world units."""
    if pixel_to_world is None:
        return polygon
    return Polygon(
        _pixels_to_world(
            np.asarray(polygon.exterior.coords), pixel_to_world, where=where
        ),
        [
            _pixels_to_world(np.asarray(ring.coords), pixel_to_world, where=where)
            for ring in polygon.interiors
        ],
    )


def _simplify_together(
    polygons: Sequence[Polygon], tolerance: float
) -> tuple[Polygon, ...]:
    """Simplify components as one MultiPolygon, keeping every component and hole.

    Simplifying together keeps components from overlapping one another. No
    dilation or hole filling occurs.
    """
    geometry = MultiPolygon(polygons).simplify(tolerance, preserve_topology=True)
    if not isinstance(geometry, Polygon | MultiPolygon) or not geometry.is_valid:
        raise ValueError("Simplification did not preserve valid outline geometry.")
    # GEOS may unwrap a one-component MultiPolygon into a Polygon.
    simplified = (geometry,) if isinstance(geometry, Polygon) else tuple(geometry.geoms)
    if len(simplified) != len(polygons) or sorted(
        len(polygon.interiors) for polygon in simplified
    ) != sorted(len(polygon.interiors) for polygon in polygons):
        raise ValueError("Simplification did not preserve every component and hole.")
    return simplified


def _mask_to_pixel_polygons(
    mask: NDArray[np.bool_], *, simplify_tolerance: float = 0.0
) -> list[Polygon]:
    """Trace the foreground pixel cells of a 2-D boolean mask exactly.

    Pixel ``(x, y)`` = ``mask[y, x]`` covers ``[x-.5, x+.5] × [y-.5, y+.5]``,
    so the polygons have the mask's exact area. Pixels that meet only at a
    corner are separate components (four-neighbor connectivity); every
    component and hole is kept. Positive ``simplify_tolerance`` (pixels)
    simplifies all components together with topology preservation.

    Returns the polygons sorted by bounds and area, or an empty list for an
    empty foreground. ``mask`` must already be a 2-D boolean array.
    """
    validate_finite_scalar(
        simplify_tolerance, name="simplify_tolerance (pixels)", positive=False
    )
    # One rectangle per horizontal run of foreground pixels, built at once.
    edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, stops = np.nonzero(edges == -1)
    if not rows.size:
        return []
    merged = shapely.union_all(
        shapely.box(starts - 0.5, rows - 0.5, stops - 0.5, rows + 0.5)
    )
    polygons = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
    if simplify_tolerance:
        polygons = list(_simplify_together(polygons, simplify_tolerance))
    return sorted(
        polygons, key=lambda polygon: (*polygon.bounds, polygon.area, polygon.wkb)
    )


def mask_to_region(
    mask: NDArray[np.bool_],
    *,
    region_name: str,
    pixel_to_world: SpatialTransform | None = None,
    simplify_tolerance: float = 0.0,
) -> Region:
    """Trace one connected area of a 2-D image mask into a polygon Region.

    The outline follows the pixel cells exactly, so the polygon has the
    mask's area and keeps its holes.

    Parameters
    ----------
    mask : ndarray of bool or 0/1 values, shape (height, width)
        Image mask indexed ``mask[row, column]``, where True (or 1) marks the
        area. The foreground must be one connected component; pixels that
        meet only at a corner are not connected.
    region_name : str
        Name of the resulting region.
    pixel_to_world : SpatialTransform, optional
        Callable mapping ``(n, 2)`` pixel ``(x, y)`` coordinates to
        ``(n, 2)`` world coordinates, such as
        ``VideoCalibration.transform_px_to_cm``. It is applied to the
        exterior and to every hole. If None, the polygon stays in image
        pixels: ``x`` is the column, ``y`` the row (pointing down), and pixel
        ``(x, y)`` covers ``[x-.5, x+.5] × [y-.5, y+.5]``.
    simplify_tolerance : float, default=0.0
        Finite nonnegative simplification distance in pixels. Zero keeps the
        exact pixel outline and area; positive values simplify the outline
        with topology preservation, keeping every hole.

    Returns
    -------
    Region
        A ``kind="polygon"`` region. ``metadata["source"]`` is ``"mask"``, and
        ``metadata["coord_system"]`` is ``"pixels"`` when ``pixel_to_world``
        is None.

    Raises
    ------
    ValueError
        If the mask is not a nonempty 2-D boolean or 0/1 array, has no
        foreground, or has more than one connected component.

    See Also
    --------
    neurospatial.annotation.mask_to_polygons : Every component of a mask, in
        pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.regions import mask_to_region
    >>> mask = np.zeros((5, 6), dtype=bool)
    >>> mask[1:4, 1:5] = True
    >>> mask[2, 2] = False  # a one-pixel hole
    >>> region = mask_to_region(mask, region_name="arena")
    >>> region.data.area, len(region.data.interiors)
    (11.0, 1)
    >>> region.data.bounds
    (0.5, 0.5, 4.5, 3.5)
    """
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            _format_error(
                f"mask must be a nonempty 2-D image (height, width), got shape "
                f"{array.shape}.",
                fix="pass one image-sized boolean array, indexed mask[row, column]",
            )
        )
    if array.dtype != np.bool_:
        if not (np.issubdtype(array.dtype, np.number) and np.isin(array, (0, 1)).all()):
            raise ValueError(
                _format_error(
                    f"mask must hold booleans or only 0 and 1, got dtype "
                    f"{array.dtype} with other values.",
                    why="a value such as 255, a probability or NaN does not say "
                    "which pixels belong to the region",
                    fix="threshold it first, for example mask_to_region(image > 0, ...)",
                )
            )
        array = array.astype(np.bool_)
    polygons = _mask_to_pixel_polygons(array, simplify_tolerance=simplify_tolerance)
    if not polygons:
        raise ValueError(
            _format_error(
                f"mask for region {region_name!r} has no foreground pixels.",
                fix="mark the region's pixels True",
            )
        )
    if len(polygons) > 1:
        sizes = sorted((round(p.area) for p in polygons), reverse=True)[:5]
        raise ValueError(
            _format_error(
                f"mask for region {region_name!r} has {len(polygons)} separate "
                f"areas (largest pixel counts {sizes}).",
                why="a Region holds one polygon, and pixels that touch only at "
                "a corner are not connected",
                fix="select one component (for example with "
                "scipy.ndimage.label), or make one region per area from "
                "neurospatial.annotation.mask_to_polygons",
            )
        )
    polygon = _polygon_to_world(
        polygons[0], pixel_to_world, where=f"region {region_name!r}"
    )
    metadata: dict[str, Any] = {"source": "mask"}
    if pixel_to_world is None:
        metadata["coord_system"] = "pixels"
    return Region(name=region_name, kind="polygon", data=polygon, metadata=metadata)


# --------------------------------------------------------------------------
# 4.  Utility: Regions → pandas DataFrame
# --------------------------------------------------------------------------
def regions_to_dataframe(regions: Regions) -> pd.DataFrame:
    """Return a *pandas* DataFrame summary.

    Parameters
    ----------
    regions : Regions
        Collection of regions to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for region name, kind, area, and metadata.

    """
    import pandas as pd

    records: list[Mapping[str, Any]] = []
    for reg in regions.values():
        rec = reg.to_dict()
        if reg.kind == "polygon":
            assert isinstance(reg.data, Polygon)
            rec["area"] = float(reg.data.area)
        else:
            rec["area"] = 0.0
        records.append(rec)

    return pd.DataFrame.from_records(records)
