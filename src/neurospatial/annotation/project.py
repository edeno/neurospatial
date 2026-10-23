"""Portable annotation drafts with an embedded reference frame.

Projects preserve editable pixel coordinates, including unfinished geometry.
Geometry validity is checked when exporting an annotation, rather than when
saving work in progress.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from neurospatial._results import read_only_copy
from neurospatial.annotation._types import MultipleBoundaryStrategy, RegionType
from neurospatial.ops.transforms import Affine2D, VideoCalibration
from neurospatial.regions.core import (
    _freeze_metadata,
    _require_json_metadata,
    _thaw_metadata,
)

__all__ = ["AnnotationProject", "AnnotationShape"]

_SHAPE_TYPES = frozenset({"polygon", "path", "line", "rectangle", "ellipse"})
_MODES = frozenset({"environment", "regions", "both"})
_VERSION = 1
_FORMAT = "neurospatial.annotation"


@dataclass(frozen=True, eq=False)
class AnnotationShape:
    """One editable shape in pixel ``(x, y)`` coordinates.

    Parameters
    ----------
    vertices : array-like, shape (n_vertices, 2)
        Finite pixel coordinates, owned as a read-only copy. Unfinished and
        self-intersecting shapes are allowed in a draft.
    shape_type : {"polygon", "path", "line", "rectangle", "ellipse"}
        Drawing primitive, retained on resume.
    name : str
        Editable label; empty labels are allowed in drafts.
    role : {"environment", "hole", "region", "wall", "landmark"}
        Scientific meaning of the shape.
    identifier : str
        Stable nonempty ring identity; generated when omitted.
    parent_id : str
        Identity of the exterior ring for an interior ring, or an empty string.
    metadata : mapping
        JSON attributes with string keys, recursively copied and made
        read-only. Non-finite floats such as ``NaN`` are kept.
    """

    vertices: NDArray[np.float64]
    _: KW_ONLY
    shape_type: str = "polygon"
    name: str = ""
    role: RegionType = "region"
    identifier: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own coordinates and metadata without requiring valid geometry."""
        vertices = read_only_copy(self.vertices, np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("Shape vertices must have shape (n_vertices, 2).")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Shape vertices must contain finite pixel coordinates.")
        if self.shape_type not in _SHAPE_TYPES:
            raise ValueError(f"Unknown shape_type {self.shape_type!r}.")
        if self.role not in get_args(RegionType):
            raise ValueError(f"Unknown annotation role {self.role!r}.")
        for name in ("name", "identifier", "parent_id"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"Shape {name} must be a string.")
        if not self.identifier:
            raise ValueError("Shape identifier must be nonempty.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Shape metadata must be a mapping of JSON values.")
        metadata = _thaw_metadata(self.metadata)
        # Non-finite floats are kept, as Region metadata allows them.
        _require_json_metadata(metadata, allow_nan=True)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "metadata", _freeze_metadata(metadata))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Reconstruct immutable metadata and arrays after pickle or deepcopy."""
        return (
            _restore_shape,
            (
                self.vertices,
                {
                    "shape_type": self.shape_type,
                    "name": self.name,
                    "role": self.role,
                    "identifier": self.identifier,
                    "parent_id": self.parent_id,
                    "metadata": _thaw_metadata(self.metadata),
                },
            ),
        )


def _restore_shape(
    vertices: NDArray[np.float64], settings: dict[str, Any]
) -> AnnotationShape:
    return AnnotationShape(vertices, **settings)


def _copy_calibration(
    calibration: VideoCalibration, frame_size: tuple[int, int]
) -> VideoCalibration:
    if not isinstance(calibration, VideoCalibration):
        raise ValueError("calibration must be a VideoCalibration or None.")
    dimensions = calibration.frame_size_px
    if (
        len(dimensions) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer))
            for n in dimensions
        )
        or tuple(dimensions) != frame_size
    ):
        raise ValueError(
            "Calibration frame_size_px must match the reference frame "
            f"(width, height) {frame_size}."
        )
    matrix = np.array(calibration.transform_px_to_cm.A, dtype=np.float64, copy=True)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix[2], [0.0, 0.0, 1.0])
    ):
        raise ValueError("Calibration transform must be a finite 3 by 3 affine matrix.")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Calibration transform must be invertible.") from exc
    if not np.all(np.isfinite(inverse)):
        raise ValueError("Calibration inverse transform must be finite.")
    matrix.setflags(write=False)
    return VideoCalibration(Affine2D(matrix), frame_size)


@dataclass(frozen=True, eq=False, init=False)
class AnnotationProject:
    """A resumable annotation draft, independent of the original video.

    Parameters
    ----------
    frame : ndarray of uint8, shape (height, width, 3)
        RGB reference image, owned as a read-only copy.
    shapes : sequence of AnnotationShape
        Editable shapes in pixel coordinates, including incomplete drafts.
    video_path : str or Path or None
        Optional provenance; the video is not opened when loading a project.
    frame_index : int
        Nonnegative index of the reference frame in the source video.
    calibration : VideoCalibration or None
        Pixel-to-physical transform matching the reference frame dimensions.
        The project owns a copy and the property returns independent copies.
    mode : {"environment", "regions", "both"}
        Annotation workflow to resume.
    bin_size : float or None
        Positive bin width for environment export, in calibrated units or pixels.
    simplify_tolerance : float or None
        Nonnegative simplification tolerance used during export.
    multiple_boundaries : {"last", "first", "error"}
        Export policy when multiple environment boundaries are drawn.

    Notes
    -----
    ``save`` writes one atomic NPZ file containing arrays and versioned JSON.
    Loading never enables pickle and does not require the original video.
    """

    frame: NDArray[np.uint8]
    shapes: tuple[AnnotationShape, ...]
    video_path: str | None
    frame_index: int
    mode: Literal["environment", "regions", "both"]
    bin_size: float | None
    simplify_tolerance: float | None
    multiple_boundaries: MultipleBoundaryStrategy
    _calibration: VideoCalibration | None = field(repr=False)

    def __init__(
        self,
        frame: NDArray[np.uint8],
        *,
        shapes: Sequence[AnnotationShape] = (),
        video_path: str | Path | None = None,
        frame_index: int = 0,
        calibration: VideoCalibration | None = None,
        mode: Literal["environment", "regions", "both"] = "both",
        bin_size: float | None = None,
        simplify_tolerance: float | None = None,
        multiple_boundaries: MultipleBoundaryStrategy = "last",
    ) -> None:
        """Validate a structurally complete snapshot of an editable draft."""
        frame = read_only_copy(frame)
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) == 0
        ):
            raise ValueError(
                "Project frame must be a nonempty RGB uint8 image (height, width, 3)."
            )
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, (int, np.integer))
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer.")
        if video_path is not None and not isinstance(video_path, (str, Path)):
            raise ValueError("video_path must be a string, Path, or None.")
        if mode not in _MODES:
            raise ValueError(f"Unknown annotation mode {mode!r}.")
        if multiple_boundaries not in get_args(MultipleBoundaryStrategy):
            raise ValueError(
                f"Unknown multiple_boundaries policy {multiple_boundaries!r}."
            )
        for name, value, positive in (
            ("bin_size", bin_size, True),
            ("simplify_tolerance", simplify_tolerance, False),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or (value <= 0 if positive else value < 0)
            ):
                raise ValueError(
                    f"{name} must be {'positive' if positive else 'nonnegative'} and finite, or None."
                )
        shapes_tuple = tuple(shapes)
        if any(not isinstance(shape, AnnotationShape) for shape in shapes_tuple):
            raise ValueError("shapes must contain AnnotationShape objects.")
        identities = {shape.identifier: shape for shape in shapes_tuple}
        if len(identities) != len(shapes_tuple):
            raise ValueError("Shape identifiers must be unique within a project.")
        for shape in shapes_tuple:
            if shape.parent_id:
                parent = identities.get(shape.parent_id)
                if parent is None:
                    raise ValueError(
                        f"Shape {shape.identifier!r} references missing parent {shape.parent_id!r}."
                    )
                if parent is shape or parent.parent_id:
                    raise ValueError(
                        "Interior rings must reference a top-level exterior ring."
                    )
                if (
                    shape.shape_type != "polygon"
                    or parent.shape_type != "polygon"
                    or shape.role != parent.role
                ):
                    raise ValueError(
                        "Interior rings must be polygons with the same role as their parent."
                    )
        owned_calibration = (
            None
            if calibration is None
            else _copy_calibration(calibration, (frame.shape[1], frame.shape[0]))
        )
        for name, stored_value in (
            ("frame", frame),
            ("shapes", shapes_tuple),
            ("video_path", None if video_path is None else str(video_path)),
            ("frame_index", int(frame_index)),
            ("mode", mode),
            ("bin_size", None if bin_size is None else float(bin_size)),
            (
                "simplify_tolerance",
                None if simplify_tolerance is None else float(simplify_tolerance),
            ),
            ("multiple_boundaries", multiple_boundaries),
            ("_calibration", owned_calibration),
        ):
            object.__setattr__(self, name, stored_value)

    @property
    def calibration(self) -> VideoCalibration | None:
        """Return an independent calibration, preserving the saved transform."""
        if self._calibration is None:
            return None
        return _copy_calibration(
            self._calibration, (self.frame.shape[1], self.frame.shape[0])
        )

    def save(self, path: str | Path) -> None:
        """Atomically save to exactly ``path``, preserving any filename suffix.

        An existing file is replaced only after the complete archive is written.
        The destination directory must already exist.
        """
        path = Path(path)
        document = {
            "format": _FORMAT,
            "version": _VERSION,
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "calibration": None
            if self._calibration is None
            else self._calibration.to_dict(),
            "mode": self.mode,
            "bin_size": self.bin_size,
            "simplify_tolerance": self.simplify_tolerance,
            "multiple_boundaries": self.multiple_boundaries,
            "shapes": [
                {
                    "shape_type": shape.shape_type,
                    "name": shape.name,
                    "role": shape.role,
                    "identifier": shape.identifier,
                    "parent_id": shape.parent_id,
                    "metadata": _thaw_metadata(shape.metadata),
                }
                for shape in self.shapes
            ],
        }
        metadata = np.frombuffer(json.dumps(document).encode("utf-8"), dtype=np.uint8)
        arrays: dict[str, Any] = {
            f"shape_{i}": shape.vertices for i, shape in enumerate(self.shapes)
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(
                    stream, frame=self.frame, metadata=metadata, **arrays
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> AnnotationProject:
        """Load a project without the source video or pickle deserialization.

        Raises
        ------
        ValueError
            If the project version is unsupported or the archive is corrupt.
        OSError
            If the file cannot be opened.
        """
        # Open outside the corruption handler to retain useful filesystem errors.
        with Path(path).open("rb") as stream:
            try:
                with np.load(stream, allow_pickle=False) as archive:
                    raw = archive["metadata"]
                    if raw.dtype != np.uint8 or raw.ndim != 1:
                        raise ValueError("metadata must be a UTF-8 byte array")
                    document = json.loads(raw.tobytes().decode("utf-8"))
                    if (
                        not isinstance(document, dict)
                        or document.get("format") != _FORMAT
                    ):
                        raise ValueError("unrecognized project format")
                    version = document.get("version")
                    if type(version) is not int or version != _VERSION:
                        raise _UnsupportedVersionError(
                            f"Unsupported annotation project version {version!r}; supported version is {_VERSION}."
                        )
                    _require_keys(
                        document,
                        {
                            "format",
                            "version",
                            "video_path",
                            "frame_index",
                            "calibration",
                            "mode",
                            "bin_size",
                            "simplify_tolerance",
                            "multiple_boundaries",
                            "shapes",
                        },
                    )
                    rows = document["shapes"]
                    if not isinstance(rows, list):
                        raise ValueError("shapes must be a list")
                    expected = {
                        "frame",
                        "metadata",
                        *[f"shape_{i}" for i in range(len(rows))],
                    }
                    if set(archive.files) != expected or len(archive.files) != len(
                        expected
                    ):
                        raise ValueError(
                            "archive arrays do not match the shape records"
                        )
                    shapes = []
                    for i, row in enumerate(rows):
                        _require_keys(
                            row,
                            {
                                "shape_type",
                                "name",
                                "role",
                                "identifier",
                                "parent_id",
                                "metadata",
                            },
                        )
                        shapes.append(AnnotationShape(archive[f"shape_{i}"], **row))
                    calibration = document["calibration"]
                    if calibration is not None:
                        _require_keys(
                            calibration, {"transform_px_to_cm", "frame_size_px"}
                        )
                        # Do not use from_dict: it silently truncates fractional sizes.
                        calibration = VideoCalibration(
                            Affine2D(
                                np.asarray(
                                    calibration["transform_px_to_cm"], dtype=float
                                )
                            ),
                            tuple(calibration["frame_size_px"]),
                        )
                    return cls(
                        archive["frame"],
                        shapes=shapes,
                        video_path=document["video_path"],
                        frame_index=document["frame_index"],
                        calibration=calibration,
                        mode=document["mode"],
                        bin_size=document["bin_size"],
                        simplify_tolerance=document["simplify_tolerance"],
                        multiple_boundaries=cast(
                            "MultipleBoundaryStrategy", document["multiple_boundaries"]
                        ),
                    )
            except _UnsupportedVersionError:
                raise
            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                IndexError,
                UnicodeError,
                BadZipFile,
                EOFError,
            ) as exc:
                raise ValueError(
                    f"Corrupt annotation project {str(path)!r}: {exc}"
                ) from exc

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild owned arrays and nested metadata after pickle or deepcopy."""
        return (
            _restore_project,
            (
                self.frame,
                {
                    "shapes": self.shapes,
                    "video_path": self.video_path,
                    "frame_index": self.frame_index,
                    "calibration": self.calibration,
                    "mode": self.mode,
                    "bin_size": self.bin_size,
                    "simplify_tolerance": self.simplify_tolerance,
                    "multiple_boundaries": self.multiple_boundaries,
                },
            ),
        )


def _restore_project(
    frame: NDArray[np.uint8], settings: dict[str, Any]
) -> AnnotationProject:
    return AnnotationProject(frame, **settings)


class _UnsupportedVersionError(ValueError):
    """Recognized archive whose schema requires a different reader."""


def _require_keys(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"expected fields {sorted(keys)}")
