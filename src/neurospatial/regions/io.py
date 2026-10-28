"""Read and write regions: JSON files, annotation-tool exports and image masks.

* :func:`regions_to_json` / :func:`regions_from_json` store regions with their
  metadata in a standard ``"Regions-v1"`` JSON file.
* :func:`load_labelme_json` and :func:`load_cvat_xml` read LabelMe and
  CVAT for images 1.1 annotations, in image pixels unless a ``pixel_to_world``
  transform is given.
* :func:`mask_to_region` traces one connected area of an image mask.

Image pixel coordinates are ``(x, y)`` with ``x`` the column and ``y`` the row,
pointing down.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
import shapely.geometry as shp
from numpy.typing import NDArray
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

from neurospatial._exceptions import _format_error
from neurospatial._validation import validate_finite_scalar

from .core import Kind, Region, Regions
from .ops import _transform_points, _transform_polygon

if TYPE_CHECKING:
    from ..ops.transforms import SpatialTransform

# --------------------------------------------------------------------------
# 1.  Generic JSON <--> Regions round-trip
# --------------------------------------------------------------------------
_SCHEMA_TAG = "Regions-v1"


def regions_to_json(
    regions: Mapping[str, Region], path: str | Path, *, indent: int = 2
) -> None:
    """Write regions to a standard JSON file that any language can read.

    The file holds one object, ``{"format": "Regions-v1", "regions": [...]}``,
    with one :meth:`Region.to_dict` entry (name, kind, GeoJSON-style geometry
    and metadata) per region. Missing parent directories are created and an
    existing file is overwritten.

    Parameters
    ----------
    regions : Mapping[str, Region]
        Collection of regions to write.
    path : str or Path
        Destination file path.
    indent : int, default=2
        Indentation of the JSON text, in spaces.

    See Also
    --------
    regions_from_json : Read the file back.
    """
    payload = {
        "format": _SCHEMA_TAG,
        "regions": [reg.to_dict() for reg in regions.values()],
    }
    # Region refuses non-finite coordinates and metadata, so this cannot fail.
    text = json.dumps(payload, indent=indent, allow_nan=False)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


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
# 2.  Shared helpers: pixel coordinates, mask tracing, names
# --------------------------------------------------------------------------
def _canvas_to_world(
    geometry: Polygon | NDArray[np.float64],
    pixel_to_world: SpatialTransform | None,
    *,
    where: str,
) -> Polygon | NDArray[np.float64]:
    """Map an annotation tool's polygon or ``(n, 2)`` points to world units.

    LabelMe and CVAT put pixel ``(column, row)`` on ``[column, column + 1]``;
    shifting half a pixel up and left centres it on ``(column, row)``, the
    package's pixel frame, before ``pixel_to_world`` is applied.
    """
    if isinstance(geometry, Polygon):
        return _transform_polygon(
            affinity.translate(geometry, -0.5, -0.5),
            pixel_to_world,
            name="pixel_to_world",
            where=where,
        )
    return _transform_points(
        geometry - 0.5, pixel_to_world, name="pixel_to_world", where=where
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
    # The union keeps a vertex at every row along a straight edge; a zero
    # tolerance drops only those collinear vertices, so the outline is exact.
    merged = shapely.simplify(
        shapely.union_all(
            shapely.box(starts - 0.5, rows - 0.5, stops - 0.5, rows + 0.5)
        ),
        0.0,
    )
    polygons = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
    if simplify_tolerance:
        polygons = list(_simplify_together(polygons, simplify_tolerance))
    return sorted(
        polygons, key=lambda polygon: (*polygon.bounds, polygon.area, polygon.wkb)
    )


def _unique_region_names(labels: Sequence[str]) -> list[str]:
    """Name regions from their labels, one rule for every loader.

    A label used once keeps its name. A repeated label becomes ``label_0``,
    ``label_1``, … in file order, skipping any suffix that is already a label
    or a generated name, so generated names never collide.
    """
    counts = Counter(labels)
    taken = set(labels)
    next_index: dict[str, int] = {}
    names = []
    for label in labels:
        if counts[label] == 1:
            names.append(label)
            continue
        index = next_index.get(label, 0)
        while f"{label}_{index}" in taken:
            index += 1
        name = f"{label}_{index}"
        taken.add(name)
        next_index[label] = index + 1
        names.append(name)
    return names


def _labelled_regions(
    loaded: Sequence[tuple[str, Polygon | NDArray[np.float64], str]],
    metadata: Mapping[str, Any],
    *,
    pixel_to_world: SpatialTransform | None,
    label_metadata: Callable[[str], Mapping[str, Any]] = lambda label: {},
) -> Regions:
    """Build an annotation file's regions, named by label, in file order.

    ``loaded`` holds ``(label, geometry, shape_type)`` in world units: a
    polygon, or ``(n, 2)`` points that each become a point region. Each
    region's metadata is ``metadata``, ``"coord_system": "pixels"`` when
    ``pixel_to_world`` is None, its ``label`` and ``shape_type``, then
    ``label_metadata(label)``.
    """
    entries: list[tuple[str, Kind, Polygon | NDArray[np.float64], str]] = []
    for label, geometry, shape_type in loaded:
        if isinstance(geometry, Polygon):
            entries.append((label, "polygon", geometry, shape_type))
        else:
            entries.extend((label, "point", point, shape_type) for point in geometry)
    shared = dict(metadata)
    if pixel_to_world is None:
        shared["coord_system"] = "pixels"
    names = _unique_region_names([label for label, *_ in entries])
    return Regions(
        Region(
            name=name,
            kind=kind,
            data=geometry,
            metadata={
                **shared,
                "label": label,
                "shape_type": shape_type,
                **label_metadata(label),
            },
        )
        for name, (label, kind, geometry, shape_type) in zip(
            names, entries, strict=True
        )
    )


# --------------------------------------------------------------------------
# 3.  LabelMe JSON and CVAT XML  → Regions
# --------------------------------------------------------------------------
_LABELME_AREA_TYPES = ("polygon", "rectangle", "oriented_rectangle", "circle")
_LABELME_POINT_TYPES = ("point", "points")
_LINE_FIX = (
    "draw the area as a polygon, or keep the line as a landmark with "
    'neurospatial.Landmark(name, "line", points) in world coordinates'
)


def _labelme_shape(
    shape: Any, index: int, json_path: Path, *, label_key: str, points_key: str
) -> tuple[str, str, NDArray[np.float64], str]:
    """Return ``(label, shape_type, points, where)`` of one LabelMe shape, or raise.

    ``where`` names the shape, its file and label, for later errors.
    """
    where = f"{json_path.name}, shape {index}"
    if not isinstance(shape, dict):
        raise ValueError(
            _format_error(
                f"{where} is a {type(shape).__name__}, not a shape object.",
                fix="check that the file is a LabelMe annotation",
            )
        )
    label = shape.get(label_key)
    if not isinstance(label, str) or not label:
        raise ValueError(
            _format_error(
                f"{where} has label {label!r} under key {label_key!r}.",
                why="each region is named by its shape's label",
                fix="give the shape a nonempty label, or pass label_key= "
                "naming the key that holds it",
            )
        )
    where = f"{where} (label {label!r})"
    shape_type = shape.get("shape_type") or "polygon"
    if shape_type in ("line", "linestrip"):
        raise ValueError(
            _format_error(
                f"{where} is a {shape_type!r}: an open line is not a region.",
                why="a region is an area or a point",
                fix=_LINE_FIX,
            )
        )
    if shape_type not in (*_LABELME_AREA_TYPES, *_LABELME_POINT_TYPES):
        raise ValueError(
            _format_error(
                f"{where} has shape_type {shape_type!r}, which is not read.",
                why="supported shape types are "
                + ", ".join(
                    repr(t) for t in (*_LABELME_AREA_TYPES, *_LABELME_POINT_TYPES)
                ),
                fix="redraw the area as a polygon; for a mask, decode it and "
                "pass the image mask to mask_to_region",
            )
        )
    raw = shape.get(points_key)
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        points = np.empty(0)
    if raw is None or points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError(
            _format_error(
                f"{where} has no (x, y) point list under key {points_key!r}.",
                fix="store the points as [[x, y], ...], or pass points_key= "
                "naming the key that holds them",
            )
        )
    if not np.isfinite(points).all():
        raise ValueError(
            _format_error(
                f"{where} has non-finite point coordinates.",
                fix="fix or delete the shape in LabelMe",
            )
        )
    expected = {"rectangle": 2, "circle": 2, "point": 1, "oriented_rectangle": 4}
    n_points = expected.get(shape_type)
    if (n_points is not None and len(points) != n_points) or (
        shape_type == "polygon" and len(points) < 3
    ):
        need = "at least 3" if n_points is None else f"exactly {n_points}"
        raise ValueError(
            _format_error(
                f"{where} is a {shape_type!r} with {len(points)} points; "
                f"it needs {need}.",
                fix="fix or delete the shape in LabelMe",
            )
        )
    return label, shape_type, points, where


def _labelme_polygon(
    shape_type: str, points: NDArray[np.float64], where: str
) -> Polygon:
    """Build the pixel polygon of a LabelMe area shape and check it."""
    if shape_type == "rectangle":
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        polygon = shapely.box(x0, y0, x1, y1)
    elif shape_type == "circle":
        # LabelMe stores the centre and one point on the rim.
        radius = float(np.hypot(*(points[1] - points[0])))
        polygon = shp.Point(points[0]).buffer(radius)
    else:  # polygon, oriented_rectangle
        polygon = Polygon(points)
    return _require_area(polygon, where, tool="LabelMe")


def _require_area(polygon: Polygon, where: str, *, tool: str) -> Polygon:
    """Return ``polygon`` if it is a valid area with positive size, else raise."""
    if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
        reason = "zero area" if polygon.is_valid else explain_validity(polygon)
        raise ValueError(
            _format_error(
                f"{where} is not a valid area: {reason}.",
                why="a region polygon needs a positive area and edges that do "
                "not cross",
                fix=f"fix the shape's vertices in {tool}",
            )
        )
    return polygon


def load_labelme_json(
    json_path: str | Path,
    *,
    pixel_to_world: SpatialTransform | None = None,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """Read the shapes of a LabelMe annotation file as regions.

    Shapes are read by their ``shape_type`` (a missing type is a polygon, as
    in LabelMe):

    * ``"polygon"`` (3 or more vertices) and ``"oriented_rectangle"``
      (4 corners) become polygons.
    * ``"rectangle"`` (2 opposite corners) becomes an axis-aligned box.
    * ``"circle"`` (centre and one rim point) becomes a 64-vertex polygon.
    * ``"point"`` becomes a point region, and each point of ``"points"``
      becomes its own point region.

    Open ``"line"`` and ``"linestrip"`` shapes, masks and any other type raise.

    Region names come from labels. A label used once keeps its name; a
    repeated label becomes ``label_0``, ``label_1``, … in file order,
    skipping any name that is already a label.

    Parameters
    ----------
    json_path : str or Path
        Path to the LabelMe ``.json`` file: an object with a ``"shapes"``
        list, or a bare list of shape objects.
    pixel_to_world : SpatialTransform, optional
        Callable mapping ``(n, 2)`` pixel ``(x, y)`` coordinates to
        ``(n, 2)`` world coordinates, such as
        ``VideoCalibration.transform_px_to_cm``. Shapes are built in pixels
        and their vertices are then transformed. If None, coordinates stay in
        image pixels: ``x`` is the column, ``y`` the row (pointing down).
        LabelMe puts pixel ``(column, row)`` on ``[column, column + 1]``;
        shapes are moved half a pixel up and left so that pixel ``(x, y)`` is
        centred on ``(x, y)``, as in video frames, :func:`mask_to_region` and
        :class:`~neurospatial.ops.transforms.VideoCalibration`.
    label_key : str, default="label"
        Key of each shape's label.
    points_key : str, default="points"
        Key of each shape's ``[[x, y], ...]`` points.

    Returns
    -------
    Regions
        One region per area or point, in file order. Each region's metadata
        holds ``"label"`` (the label before any suffix), ``"shape_type"``,
        ``"source_json"`` (the file name), ``"source_image"`` (LabelMe's
        ``imagePath``, when present) and, without ``pixel_to_world``,
        ``"coord_system": "pixels"``.

    Raises
    ------
    ValueError
        If the file has no shapes, a shape is not an object, has no label or
        no ``(n, 2)`` points, has the wrong number of points for its type, is
        an open line or an unsupported type, or is not a valid area; or if
        ``pixel_to_world`` does not return finite ``(n, 2)`` coordinates. The
        message names the file, the shape index and the label.

    See Also
    --------
    neurospatial.annotation.regions_from_labelme : The same import with a
        video calibration, recording the coordinate units.
    load_cvat_xml : Read a CVAT export.
    """
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        shapes = data.get("shapes")
        image_path = data.get("imagePath")
    else:
        shapes, image_path = data, None
    if not isinstance(shapes, list) or not shapes:
        raise ValueError(
            _format_error(
                f"{path.name} has no shapes.",
                why='a LabelMe file lists its annotations under "shapes"',
                fix="annotate at least one region in LabelMe and save, or check "
                "that this is a LabelMe file",
            )
        )
    parsed = [
        _labelme_shape(shape, index, path, label_key=label_key, points_key=points_key)
        for index, shape in enumerate(shapes)
    ]
    # LabelMe's canvas paints the image from (0, 0) and clips points to
    # [0, width], so pixel (c, r) covers [c, c + 1]: the canvas frame.
    loaded = [
        (
            label,
            _canvas_to_world(
                points
                if shape_type in _LABELME_POINT_TYPES
                else _labelme_polygon(shape_type, points, where),
                pixel_to_world,
                where=where,
            ),
            shape_type,
        )
        for label, shape_type, points, where in parsed
    ]
    metadata: dict[str, Any] = {"source_json": path.name}
    if isinstance(image_path, str):
        metadata["source_image"] = image_path
    return _labelled_regions(loaded, metadata, pixel_to_world=pixel_to_world)


# CVAT for images 1.1 children of <image> that become regions, and the
# image-level tags that carry no geometry and are skipped.
_CVAT_SHAPE_TAGS = ("polygon", "box", "ellipse", "mask", "polyline", "points")
_CVAT_SKIPPED_TAGS = ("tag",)
_CVAT_REEXPORT = 're-export the task from CVAT with the "CVAT for images 1.1" format'


def _cvat_number(
    elem: ET.Element, key: str, where: str, *, default: float | None = None
) -> float:
    """Read one finite numeric attribute of a CVAT shape, or raise."""
    raw = elem.get(key)
    if raw is None and default is not None:
        return default
    try:
        value = float(raw) if raw is not None else np.nan
    except ValueError:
        value = np.nan
    if not np.isfinite(value):
        raise ValueError(
            _format_error(
                f"{where} has {key}={raw!r}; expected a finite number.",
                fix=_CVAT_REEXPORT,
            )
        )
    return value


def _parse_cvat_points(points_str: str | None, *, where: str) -> NDArray[np.float64]:
    """Parse CVAT's ``"x1,y1;x2,y2;..."`` points into an ``(n, 2)`` array."""
    try:
        points = np.array(
            [
                [float(v) for v in pair.split(",")]
                for pair in (points_str or "").split(";")
            ],
            dtype=np.float64,
        )
    except ValueError:
        points = np.empty(0)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(
            _format_error(
                f"{where} has points={points_str!r}; expected finite "
                '"x1,y1;x2,y2;..." pairs.',
                fix=_CVAT_REEXPORT,
            )
        )
    return points


def _rle_to_mask(rle: str, width: int, height: int, *, where: str) -> NDArray[np.bool_]:
    """Decode a CVAT mask's ``rle`` over its own ``width × height`` box.

    CVAT writes ``rle`` as alternating run lengths in row-major order over the
    mask's tight box, starting with a run of background (0) pixels, which may
    be 0 long (``MaskConverter.rle`` and ``dm_mask_to_cvat_rle`` in
    cvat/apps/dataset_manager/formats/transformations.py; ``cropMask`` in
    cvat-core/src/object-utils.ts walks the runs the same way). The box sits
    at the ``left``/``top`` attributes. Returns ``mask[row, column]``.
    """
    try:
        counts = np.array([int(v) for v in rle.split(",")], dtype=np.int64)
    except ValueError:
        counts = np.array([-1])
    if (counts < 0).any() or counts.sum() != width * height:
        raise ValueError(
            _format_error(
                f"{where} has rle={rle[:60]!r}, which does not cover its "
                f"{width} x {height} pixel box.",
                why="CVAT stores a mask as nonnegative run lengths that add up "
                "to width x height",
                fix=_CVAT_REEXPORT,
            )
        )
    # Runs alternate background, foreground, ... starting with background.
    values = np.arange(counts.size) % 2 == 1
    return np.repeat(values, counts).reshape(height, width)


def _cvat_mask_polygon(
    elem: ET.Element, where: str, image_size: tuple[int, int] | None
) -> Polygon:
    """Trace a CVAT ``<mask>`` into one canvas-pixel polygon, keeping its holes."""
    left, top, width, height = (
        _cvat_number(elem, key, where) for key in ("left", "top", "width", "height")
    )
    if (
        not all(v == int(v) for v in (left, top, width, height))
        or min(width, height) < 1
    ):
        raise ValueError(
            _format_error(
                f"{where} has left={left}, top={top}, width={width}, "
                f"height={height}; expected whole pixels and a nonempty box.",
                fix=_CVAT_REEXPORT,
            )
        )
    if image_size is not None and (
        left < 0
        or top < 0
        or left + width > image_size[0]
        or top + height > image_size[1]
    ):
        raise ValueError(
            _format_error(
                f"{where} covers columns {int(left)}-{int(left + width) - 1} and "
                f"rows {int(top)}-{int(top + height) - 1}, outside the "
                f"{image_size[0]} x {image_size[1]} image.",
                fix=_CVAT_REEXPORT,
            )
        )
    mask = _rle_to_mask(elem.get("rle") or "", int(width), int(height), where=where)
    polygons = _mask_to_pixel_polygons(mask)
    if not polygons:
        raise ValueError(
            _format_error(
                f"{where} is a mask with no foreground pixels.",
                fix="delete the empty mask in CVAT",
            )
        )
    if len(polygons) > 1:
        raise ValueError(
            _format_error(
                f"{where} is a mask with {len(polygons)} separate areas.",
                why="a Region holds one polygon, and pixels that touch only at a "
                "corner are not connected",
                fix="in CVAT, keep one connected area per mask shape (split or "
                "erase the others)",
            )
        )
    # The tracer centres pixel (x, y) on (x, y); CVAT's other shapes are in
    # its canvas frame, where pixel (x, y) covers [x, x + 1], so match them.
    return affinity.translate(polygons[0], left + 0.5, top + 0.5)


def _cvat_box_polygon(elem: ET.Element, where: str) -> Polygon:
    """A CVAT ``<box>``, rotated about its centre by ``rotation`` degrees."""
    xtl, ytl, xbr, ybr = (
        _cvat_number(elem, key, where) for key in ("xtl", "ytl", "xbr", "ybr")
    )
    if xbr <= xtl or ybr <= ytl:
        raise ValueError(
            _format_error(
                f"{where} has xtl={xtl}, ytl={ytl}, xbr={xbr}, ybr={ybr}; the "
                "bottom-right corner must be right of and below the top-left one.",
                fix="fix or delete the box in CVAT",
            )
        )
    return _cvat_rotated(
        shapely.box(xtl, ytl, xbr, ybr),
        elem,
        where,
        centre=((xtl + xbr) / 2, (ytl + ybr) / 2),
    )


def _cvat_ellipse_polygon(elem: ET.Element, where: str) -> Polygon:
    """A CVAT ``<ellipse>`` as a 64-vertex polygon, rotated like a box."""
    cx, cy, rx, ry = (
        _cvat_number(elem, key, where) for key in ("cx", "cy", "rx", "ry")
    )
    if rx <= 0 or ry <= 0:
        raise ValueError(
            _format_error(
                f"{where} has radii rx={rx}, ry={ry}; both must be positive.",
                fix="fix or delete the ellipse in CVAT",
            )
        )
    ellipse = affinity.scale(shp.Point(cx, cy).buffer(1.0), rx, ry, origin=(cx, cy))
    return _cvat_rotated(ellipse, elem, where, centre=(cx, cy))


def _cvat_rotated(
    polygon: Polygon, elem: ET.Element, where: str, *, centre: tuple[float, float]
) -> Polygon:
    """Apply CVAT's ``rotation`` attribute, degrees about the shape's centre.

    CVAT rotates corners with ``x' = cx + cos(a)(x - cx) - sin(a)(y - cy)``,
    ``y' = cy + sin(a)(x - cx) + cos(a)(y - cy)`` in image coordinates, so a
    positive angle turns clockwise on screen (y points down)
    (``RotatedBoxesToPolygons._rotate_point`` in
    cvat/apps/dataset_manager/formats/transformations.py). Shapely's
    ``rotate`` applies the same matrix.
    """
    angle = _cvat_number(elem, "rotation", where, default=0.0)
    return affinity.rotate(polygon, angle, origin=centre) if angle else polygon


def _cvat_shape_geometry(
    elem: ET.Element, where: str, image_size: tuple[int, int] | None
) -> Polygon | NDArray[np.float64]:
    """One CVAT shape in canvas pixels: a polygon, or ``(n, 2)`` points."""
    tag = elem.tag
    if tag in ("points", "polygon", "polyline"):
        points = _parse_cvat_points(elem.get("points"), where=where)
    if tag == "points":
        return points
    if tag == "polygon":
        if len(points) < 3:
            raise ValueError(
                _format_error(
                    f"{where} has {len(points)} points; a polygon needs at least 3.",
                    fix="fix or delete the polygon in CVAT",
                )
            )
        polygon = Polygon(points)
    elif tag == "polyline":
        if len(points) < 4 or not np.allclose(points[0], points[-1]):
            raise ValueError(
                _format_error(
                    f"{where} is an open polyline: an open line is not a region.",
                    why="a region is an area or a point; only a polyline whose "
                    "last point repeats its first encloses an area",
                    fix=_LINE_FIX,
                )
            )
        polygon = Polygon(points)
    elif tag == "box":
        polygon = _cvat_box_polygon(elem, where)
    elif tag == "ellipse":
        polygon = _cvat_ellipse_polygon(elem, where)
    elif tag == "mask":
        polygon = _cvat_mask_polygon(elem, where, image_size)
    else:
        raise ValueError(
            _format_error(
                f"{where} is a <{tag}> shape, which is not read.",
                why="supported shapes are <"
                + ">, <".join(_CVAT_SHAPE_TAGS)
                + ">; cuboids and skeletons do not describe one 2-D area",
                fix="delete the shape in CVAT, or draw the area as a polygon",
            )
        )
    return _require_area(polygon, where, tool="CVAT")


def _get_label_colors(root: ET.Element) -> dict[str, str]:
    """Map each CVAT label name to its ``<color>`` from the export's meta."""
    label_colors: dict[str, str] = {}
    labels = root.find(".//labels")
    for label_elem in labels.findall("label") if labels is not None else ():
        name, color = label_elem.findtext("name"), label_elem.findtext("color")
        if name and color:
            label_colors[name.strip()] = color.strip()
    return label_colors


def _describe_cvat_image(image: ET.Element) -> str:
    return f"{image.get('name')!r} (id {image.get('id')})"


def _list_cvat_images(images: Sequence[ET.Element], limit: int = 10) -> str:
    listed = ", ".join(_describe_cvat_image(image) for image in images[:limit])
    return listed + (f", ... ({len(images)} in all)" if len(images) > limit else "")


def _select_cvat_image(
    images: list[ET.Element], image: str | int | None, path: Path
) -> ET.Element:
    """Return the one ``<image>`` whose shapes become regions, or raise."""
    annotated = [
        img
        for img in images
        if any(child.tag not in _CVAT_SKIPPED_TAGS for child in img)
    ]
    if image is None:
        if len(annotated) == 1:
            return annotated[0]
        if not annotated:
            raise ValueError(
                _format_error(
                    f"{path.name} has no shapes in any of its {len(images)} images.",
                    fix="draw the regions in CVAT before exporting",
                )
            )
        raise ValueError(
            _format_error(
                f"{path.name} has shapes on {len(annotated)} images: "
                f"{_list_cvat_images(annotated)}.",
                why="one image's shapes make one set of regions; shapes from "
                "different frames may not share a coordinate frame",
                fix=f"pass image= with the image name or id, for example "
                f"image={annotated[0].get('name')!r}",
            )
        )
    key, wanted = ("name", image) if isinstance(image, str) else ("id", str(image))
    matches = [img for img in images if img.get(key) == wanted]
    if len(matches) != 1:
        found = "no" if not matches else f"{len(matches)}"
        raise ValueError(
            _format_error(
                f"{path.name} has {found} images with {key} {wanted!r}.",
                fix=f"pass one of: {_list_cvat_images(images)}",
            )
        )
    if matches[0] not in annotated:
        raise ValueError(
            _format_error(
                f"Image {_describe_cvat_image(matches[0])} in {path.name} has "
                "no shapes.",
                fix=f"pass an annotated image: {_list_cvat_images(annotated)}"
                if annotated
                else "draw the regions in CVAT before exporting",
            )
        )
    return matches[0]


def _image_size(image: ET.Element) -> tuple[int, int] | None:
    try:
        size = (int(image.get("width", "")), int(image.get("height", "")))
    except ValueError:
        return None
    return size if min(size) > 0 else None


def load_cvat_xml(
    xml_path: str | Path,
    *,
    pixel_to_world: SpatialTransform | None = None,
    image: str | int | None = None,
) -> Regions:
    """Read the shapes of one image in a CVAT for images 1.1 export as regions.

    CVAT draws on a canvas where pixel ``(column, row)`` covers
    ``[column, column + 1] × [row, row + 1]``. Shapes are moved half a pixel
    up and left into neurospatial's pixel frame, where pixel ``(x, y)`` is
    centred on ``(x, y)`` as in video frames, masks from
    :func:`mask_to_region` and :class:`~neurospatial.ops.transforms.VideoCalibration`.

    Supported shapes:

    * ``<polygon>`` becomes a polygon.
    * ``<box>`` becomes a rectangle, turned by its ``rotation`` attribute
      (degrees about the box centre; positive is clockwise on screen).
    * ``<ellipse>`` becomes a 64-vertex polygon, rotated like a box.
    * ``<mask>`` is decoded from its run lengths and traced exactly along
      pixel edges, keeping holes. A mask must be one connected area.
    * ``<polyline>`` whose last point repeats its first becomes a polygon.
    * Each point of ``<points>`` becomes a point region.

    Open polylines, cuboids, skeletons and malformed shapes raise; image-level
    ``<tag>`` labels have no geometry and are skipped. Video exports
    (``<track>``) are refused.

    Region names come from labels. A label used once keeps its name; a
    repeated label becomes ``label_0``, ``label_1``, … in file order,
    skipping any name that is already a label.

    Parameters
    ----------
    xml_path : str or Path
        Path to the CVAT XML export.
    pixel_to_world : SpatialTransform, optional
        Callable mapping ``(n, 2)`` pixel ``(x, y)`` coordinates to
        ``(n, 2)`` world coordinates, such as
        ``VideoCalibration.transform_px_to_cm``. Shapes are built in pixels
        and every vertex, holes included, is then transformed. If None,
        coordinates stay in image pixels: ``x`` is the column, ``y`` the
        row (pointing down), with the first pixel centred on ``(0, 0)``.
    image : str or int, optional
        The image whose shapes to read: its ``name`` (str) or ``id`` (int).
        If None, the file must have shapes on exactly one image.

    Returns
    -------
    Regions
        One region per area or point, in file order. Each region's metadata
        holds ``"label"`` (the label before any suffix), ``"shape_type"``
        (the XML tag), ``"source_xml"`` (the file name), ``"source_image"``
        and ``"image_id"`` (the image's name and id), ``"color"`` and
        ``"plot_kwargs"`` (the label color, when the export lists one) and,
        without ``pixel_to_world``, ``"coord_system": "pixels"``.

    Raises
    ------
    FileNotFoundError
        If ``xml_path`` does not exist.
    xml.etree.ElementTree.ParseError
        If ``xml_path`` is not valid XML.
    ValueError
        If the file is not a CVAT for images export, holds tracks, has no
        shapes, has shapes on several images and ``image`` is None, ``image``
        matches no annotated image, a shape is malformed or unsupported, or
        ``pixel_to_world`` does not return finite ``(n, 2)`` coordinates.
        Shape errors name the file, the image, the shape index and the label.

    See Also
    --------
    neurospatial.annotation.regions_from_cvat : The same import with a video
        calibration, recording the coordinate units.
    load_labelme_json : Read a LabelMe file.
    """
    path = Path(xml_path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ET.ParseError(f"Error parsing XML file {xml_path}: {e}") from e
    if root.tag != "annotations":
        raise ValueError(
            _format_error(
                f"{path.name} has root element <{root.tag}>, not <annotations>.",
                why="CVAT XML exports start with <annotations>",
                fix=f"check that this is a CVAT export, or {_CVAT_REEXPORT}",
            )
        )
    n_tracks = len(root.findall("track"))
    if n_tracks:
        raise ValueError(
            _format_error(
                f"{path.name} holds {n_tracks} <track> elements, which are not read.",
                why='this is a "CVAT for video 1.1" export, where shapes follow '
                "objects across frames",
                fix=_CVAT_REEXPORT + ", which stores the shapes of each frame "
                "under <image>",
            )
        )
    images = root.findall("image")
    if not images:
        raise ValueError(
            _format_error(
                f"{path.name} has no <image> elements.",
                fix=_CVAT_REEXPORT,
            )
        )
    selected = _select_cvat_image(images, image, path)
    image_name = selected.get("name")
    image_id_raw = selected.get("id")
    image_id = (
        int(image_id_raw) if image_id_raw and image_id_raw.isdigit() else image_id_raw
    )
    image_size = _image_size(selected)
    label_colors = _get_label_colors(root)

    loaded: list[tuple[str, Polygon | NDArray[np.float64], str]] = []
    for index, elem in enumerate(selected):
        if elem.tag in _CVAT_SKIPPED_TAGS:
            continue
        label = (elem.get("label") or "").strip()
        where = (
            f"{path.name}, image {_describe_cvat_image(selected)}, shape {index} "
            f"(<{elem.tag}>, label {label!r})"
        )
        if not label:
            raise ValueError(
                _format_error(
                    f"{where} has no label.",
                    why="each region is named by its shape's label",
                    fix="give the shape a label in CVAT and re-export",
                )
            )
        geometry = _cvat_shape_geometry(elem, where, image_size)
        loaded.append(
            (label, _canvas_to_world(geometry, pixel_to_world, where=where), elem.tag)
        )

    def label_metadata(label: str) -> dict[str, Any]:
        color = label_colors.get(label)
        return {"color": color, "plot_kwargs": {"color": color} if color else {}}

    return _labelled_regions(
        loaded,
        {"source_xml": path.name, "source_image": image_name, "image_id": image_id},
        pixel_to_world=pixel_to_world,
        label_metadata=label_metadata,
    )


# --------------------------------------------------------------------------
# 4.  Binary image mask  → single Region
# --------------------------------------------------------------------------
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
    polygon = _transform_polygon(
        polygons[0],
        pixel_to_world,
        name="pixel_to_world",
        where=f"region {region_name!r}",
    )
    metadata: dict[str, Any] = {"source": "mask"}
    if pixel_to_world is None:
        metadata["coord_system"] = "pixels"
    return Region(name=region_name, kind="polygon", data=polygon, metadata=metadata)
