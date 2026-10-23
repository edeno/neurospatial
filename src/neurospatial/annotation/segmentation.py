"""Viewer-independent, prompt-based proposals for arena outlines.

All coordinates are pixel ``(x, y)``; array indexing is ``[y, x]``. These
providers produce proposals for review, without changing an annotation.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon

from neurospatial._results import ReadOnlyArraysMixin, read_only_copy
from neurospatial._validation import validate_finite_scalar
from neurospatial.regions.core import _require_json_metadata

__all__ = [
    "RegionGrowProvider",
    "SegmentAnythingProvider",
    "SegmentationPrompt",
    "SegmentationProposal",
    "SegmentationProvider",
    "mask_to_polygons",
]


@dataclass(frozen=True)
class SegmentationPrompt(ReadOnlyArraysMixin):
    """Foreground/background clicks and an optional bounding box.

    Parameters
    ----------
    points : array-like, shape (n_points, 2)
        Finite pixel ``(x, y)`` coordinates. Integers denote pixel centers.
    labels : array-like, shape (n_points,)
        One for foreground and zero for background.
    box : array-like, shape (4,), optional
        Pixel bounds ``(x_min, y_min, x_max, y_max)`` with increasing limits.

    Notes
    -----
    Arrays are owned, read-only copies. Image bounds are checked by providers.
    An empty ``(0, 2)`` points array allows box-only prompts for providers that
    support them; :class:`RegionGrowProvider` needs a foreground click.
    """

    points: NDArray[np.float64]
    labels: NDArray[np.int64]
    _: KW_ONLY
    box: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        labels = np.asarray(self.labels)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(
                "Prompt points must be finite pixel (x, y) pairs, shape (N, 2)."
            )
        if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all():
            raise ValueError(
                "Prompt labels must have shape (N,) with 1=foreground and 0=background."
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels.astype(np.int64))
        if self.box is not None:
            bounds = np.asarray(self.box, dtype=np.float64)
            if (
                bounds.shape != (4,)
                or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[2]
                or bounds[1] >= bounds[3]
            ):
                raise ValueError(
                    "Prompt box must be finite (x_min, y_min, x_max, y_max) with increasing bounds."
                )
            object.__setattr__(self, "box", bounds)
        self._own_arrays()


@dataclass(frozen=True, init=False)
class SegmentationProposal(ReadOnlyArraysMixin):
    """A nonempty mask and optional confidence and JSON provenance.

    Parameters
    ----------
    mask : ndarray of bool, shape (height, width)
        Proposed foreground. Owns a read-only copy, including holes and every
        connected component; providers must not silently return an empty mask.
    score : float, optional
        Finite provider confidence. Scores need not be comparable across providers.
    metadata : mapping, optional
        JSON-compatible provenance with string keys. Copied on input and on
        every read so caller edits cannot change the recorded provenance.
    """

    mask: NDArray[np.bool_]
    score: float | None
    _metadata: dict[str, Any]

    def __init__(
        self,
        mask: NDArray[np.bool_],
        *,
        score: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        mask = _validate_mask(mask, nonempty=True)
        if score is not None and not np.isfinite(score):
            raise ValueError("Proposal score must be finite, or None if unavailable.")
        owned_metadata = {} if metadata is None else deepcopy(dict(metadata))
        try:
            encoded = json.dumps(owned_metadata, allow_nan=False)
            _require_json_metadata(owned_metadata, allow_nan=False)
            # Round-tripping also owns nested values and normalizes JSON tuples.
            owned_metadata = json.loads(encoded)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Proposal metadata must contain finite, JSON-compatible provenance."
            ) from exc
        object.__setattr__(self, "mask", read_only_copy(mask))
        object.__setattr__(self, "score", None if score is None else float(score))
        object.__setattr__(self, "_metadata", owned_metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return an independent copy of the JSON provenance."""
        return deepcopy(self._metadata)


class SegmentationProvider(Protocol):
    """An image-to-mask adapter requiring no viewer or GUI imports."""

    def predict(
        self, image: NDArray[np.uint8], prompt: SegmentationPrompt
    ) -> SegmentationProposal:
        """Propose foreground for an RGB uint8 image and pixel (x, y) prompt."""
        ...


def _validate_mask(mask: Any, *, nonempty: bool = False) -> NDArray[np.bool_]:
    array: NDArray[np.bool_] = np.asarray(mask)
    if array.dtype != np.bool_ or array.ndim != 2 or min(array.shape, default=0) == 0:
        raise ValueError(
            "Segmentation mask must be a nonzero-size 2D boolean array (height, width)."
        )
    if nonempty and not array.any():
        raise ValueError(
            "Segmentation produced no foreground; add a foreground click or adjust the prompt/settings."
        )
    return array


def _inside_pixels(
    points: NDArray[np.float64], bounds: tuple[float, float, float, float]
) -> bool:
    """Return whether every ``(x, y)`` point lies on pixels ``[x0, x1) × [y0, y1)``.

    Integer coordinates are pixel centers, so pixel ``i`` covers
    ``[i - 0.5, i + 0.5)``. Non-finite points are outside.
    """
    x0, y0, x1, y1 = bounds
    return bool(
        ((points >= [x0 - 0.5, y0 - 0.5]) & (points < [x1 - 0.5, y1 - 0.5])).all()
    )


def _pixel_indices(points: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return the ``(x, y)`` index of the pixel under each point."""
    return np.floor(points + 0.5).astype(np.int64)


def _clicked_components(
    mask: NDArray[np.bool_], points: NDArray[np.float64]
) -> tuple[NDArray[np.int32], int, NDArray[np.int32]]:
    """Label four-connected components and the label under each point.

    Returns the label image, the number of components and one label per
    point, zero where a point misses the foreground.
    """
    components, count = ndimage.label(mask)
    pixels = _pixel_indices(points)
    return components, count, components[pixels[:, 1], pixels[:, 0]]


def _validate_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.asarray(image)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) == 0
    ):
        raise ValueError(
            "Segmentation image must be a nonzero-size RGB uint8 array (height, width, 3)."
        )
    return image


def _validate_inputs(
    image: NDArray[np.uint8], prompt: SegmentationPrompt
) -> NDArray[np.uint8]:
    image = _validate_image(image)
    height, width = image.shape[:2]
    if not _inside_pixels(prompt.points, (0, 0, width, height)):
        raise ValueError(
            f"Prompt (x, y) points fall outside the {width} by {height} image; use pixel coordinates."
        )
    if prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        if x1 < -0.5 or y1 < -0.5 or x0 >= width - 0.5 or y0 >= height - 0.5:
            raise ValueError(
                "Prompt box lies outside the image; choose bounds intersecting the image."
            )
    return image


@dataclass(frozen=True, kw_only=True)
class RegionGrowProvider:
    """CPU color segmentation connected to foreground clicks.

    Parameters
    ----------
    tolerance : float, default=0.1
        Maximum RGB Euclidean distance from any foreground seed, normalized
        by ``255 * sqrt(3)`` to range from zero to one. Background seed colors
        exclude pixels at least as close to background as to foreground.

    Notes
    -----
    Uses four-neighbor connectivity and clips to the optional box. Clicks
    select the nearest pixel center. Similar colors in disconnected areas
    are excluded unless those areas also have a foreground click. This simple
    color assumption works best with an arena distinct from its surroundings.
    """

    tolerance: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or not 0 <= self.tolerance <= 1:
            raise ValueError(
                "Region-grow tolerance must be finite and between 0 and 1."
            )

    def predict(
        self, image: NDArray[np.uint8], prompt: SegmentationPrompt
    ) -> SegmentationProposal:
        """Return all color-matched components containing a foreground click."""
        image = _validate_inputs(image, prompt)
        if not (prompt.labels == 1).any():
            raise ValueError(
                "RegionGrowProvider needs at least one foreground click (label=1)."
            )
        height, width = image.shape[:2]
        pixels = _pixel_indices(prompt.points)
        colors = image.astype(np.float64) / (255 * np.sqrt(3))
        seeds = colors[pixels[:, 1], pixels[:, 0]]

        def nearest_squared(
            values: NDArray[np.float64], seed_colors: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            # Squared distances order pixels as distances do, without a root.
            # One seed at a time avoids an image-size by number-of-seeds array.
            nearest = np.full(values.shape[:-1], np.inf)
            for seed in seed_colors:
                difference = values - seed
                nearest = np.minimum(
                    nearest, np.einsum("...k,...k->...", difference, difference)
                )
            return nearest

        foreground = nearest_squared(colors, seeds[prompt.labels == 1])
        eligible = foreground <= self.tolerance**2
        if (prompt.labels == 0).any():
            # Only pixels close enough to the foreground can be excluded.
            eligible[eligible] = foreground[eligible] < nearest_squared(
                colors[eligible], seeds[prompt.labels == 0]
            )
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
            rows, columns = np.ogrid[:height, :width]
            eligible &= (columns >= x0) & (columns <= x1) & (rows >= y0) & (rows <= y1)
        components, _, clicked = _clicked_components(
            eligible, prompt.points[prompt.labels == 1]
        )
        return SegmentationProposal(
            np.isin(components, clicked[clicked != 0]),
            metadata={
                "provider": "region_grow",
                "tolerance": self.tolerance,
                "connectivity": 4,
            },
        )


@dataclass(frozen=True)
class _SAMRefinement:
    """One successful prediction, explicitly carried between UI requests.

    The image and logits are owned, read-only snapshots. This state is never
    attached to a saved annotation or updated by an in-flight worker.
    """

    owner: object
    image: NDArray[np.uint8]
    prompt: SegmentationPrompt
    proposal: SegmentationProposal
    logits: NDArray[Any]
    steps: int

    def accepts(self, prompt: SegmentationPrompt) -> bool:
        """Allow unchanged prompts or appended clicks of either polarity."""
        if self.prompt.box is None or prompt.box is None:
            if self.prompt.box is not prompt.box:
                return False
        elif not np.array_equal(self.prompt.box, prompt.box):
            return False
        # The widget groups inside points before outside points. Appending an
        # inside point therefore shifts outside points in the combined array.
        for label in (0, 1):
            old = self.prompt.points[self.prompt.labels == label]
            new = prompt.points[prompt.labels == label]
            if not np.array_equal(old, new[: len(old)]):
                return False
        return True


class SegmentAnythingProvider:
    """Adapt a SAM2-style image predictor without importing model libraries.

    Parameters
    ----------
    predictor : object
        An exclusively owned predictor exposing ``set_image(image)`` and
        ``predict(point_coords=..., point_labels=..., box=...,
        multimask_output=True)`` returning masks, scores and low-resolution
        logits. Do not reuse or mutate this predictor outside the adapter:
        its embeddings are cached for repeated prompts on the same image.
        In the annotation editor it must also support ``mask_input`` for
        previous-logit refinement with ``multimask_output=False``.

    Notes
    -----
    Uses the highest-scoring mask and preserves all its components and holes.
    Following Meta's guidance, a single click requests SAM's three candidate
    masks for the ambiguous prompt; several clicks or a box request one mask.
    An owned image snapshot detects changes in content, including in-place
    edits, without relying on object identity. Instances are not thread-safe.
    Direct ``predict`` calls are independent of earlier prompts. The annotation
    editor separately manages previous-logit refinement for consecutive clicks.
    The API follows Meta's ``sam2.sam2_image_predictor.SAM2ImagePredictor``:
    https://github.com/facebookresearch/sam2 .
    """

    def __init__(self, predictor: Any) -> None:
        if not callable(getattr(predictor, "set_image", None)) or not callable(
            getattr(predictor, "predict", None)
        ):
            raise TypeError(
                "SAM predictor must provide callable set_image(image) and predict(...)."
            )
        self._predictor = predictor
        self._cached_image: NDArray[np.uint8] | None = None
        self._refinement_owner = object()

    @classmethod
    def from_sam2(
        cls,
        config_file: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ) -> SegmentAnythingProvider:
        """Load installed SAM 2 using an existing local checkpoint.

        Install PyTorch and Meta's SAM 2 following
        https://github.com/facebookresearch/sam2#installation . Supply the
        installed model's config name (for example
        ``configs/sam2.1/sam2.1_hiera_l.yaml``) and a matching checkpoint file.
        No weights are downloaded and no global torch settings are changed.

        ``device=None`` uses a CUDA GPU when available, then Apple silicon
        (``"mps"``), then the CPU. On Apple silicon, SAM 2.1 Large computes a
        frame's image features about ten times faster than on the CPU. Meta
        describes MPS support as preliminary; pass ``device="cpu"`` if its
        outlines look wrong.
        """
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM 2 checkpoint does not exist: {checkpoint}. Supply a local checkpoint matching config_file; weights are not downloaded automatically."
            )
        try:
            torch = importlib.import_module("torch")
            build_sam2 = importlib.import_module("sam2.build_sam").build_sam2
            predictor_class = importlib.import_module(
                "sam2.sam2_image_predictor"
            ).SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SegmentAnythingProvider.from_sam2 requires PyTorch and Meta SAM 2. Install them following https://github.com/facebookresearch/sam2#installation, or use RegionGrowProvider without model dependencies."
            ) from exc
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        try:
            model = build_sam2(str(config_file), str(checkpoint), device=device)
            predictor = predictor_class(model)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load SAM 2 config {str(config_file)!r} and checkpoint {str(checkpoint)!r} on device {device!r}. Check the matching model configuration, checkpoint, and installed SAM 2/PyTorch versions: {exc}"
            ) from exc
        return cls(predictor)

    def predict(
        self, image: NDArray[np.uint8], prompt: SegmentationPrompt
    ) -> SegmentationProposal:
        """Reuse image embeddings and select the highest-scoring mask."""
        return self._predict_with_refinement(image, prompt).proposal

    def _prepare(self, image: NDArray[np.uint8]) -> None:
        """Compute image features ahead of the first prompt."""
        image = _validate_image(image)
        try:
            self._set_image(image)
        except Exception as exc:
            self._cached_image = None
            raise RuntimeError(
                f"SAM image preparation failed; check the installed predictor: {exc}"
            ) from exc

    def _set_image(self, image: NDArray[np.uint8]) -> None:
        """Compute features unless this exact image is already cached."""
        if self._cached_image is None or not np.array_equal(image, self._cached_image):
            self._cached_image = None
            owned_image = read_only_copy(image)
            # Keep our snapshot separate from any predictor-owned buffer.
            self._predictor.set_image(np.array(owned_image, copy=True))
            self._cached_image = owned_image

    def _predict_with_refinement(
        self,
        image: NDArray[np.uint8],
        prompt: SegmentationPrompt,
        previous: _SAMRefinement | None = None,
    ) -> _SAMRefinement:
        """Return raw logits without committing them to any interaction session."""
        image = _validate_inputs(image, prompt)
        if not len(prompt.points) and prompt.box is None:
            raise ValueError(
                "SAM segmentation needs foreground/background clicks or a bounding box."
            )
        if previous is not None:
            if (
                previous.owner is not self._refinement_owner
                or not np.array_equal(image, previous.image)
                or not previous.accepts(prompt)
            ):
                previous = None
            elif len(prompt.points) == len(previous.prompt.points):
                # Repeated Update requests must not drift by feeding a mask
                # back into itself without any new information.
                return previous
        try:
            self._set_image(image)
            refinement_args = (
                {} if previous is None else {"mask_input": previous.logits.copy()}
            )
            output = self._predictor.predict(
                point_coords=np.array(prompt.points, copy=True)
                if len(prompt.points)
                else None,
                point_labels=np.array(prompt.labels, copy=True)
                if len(prompt.points)
                else None,
                box=None if prompt.box is None else np.array(prompt.box, copy=True),
                # One click is ambiguous: ask for SAM's candidate masks. More
                # clicks, a box or a refinement give one better mask.
                multimask_output=previous is None
                and prompt.box is None
                and len(prompt.points) == 1,
                **refinement_args,
            )
        except Exception as exc:
            # A failed predictor may have changed its image state.
            self._cached_image = None
            raise RuntimeError(
                f"SAM segmentation failed; check the installed predictor and prompt: {exc}"
            ) from exc
        if not isinstance(output, tuple | list) or len(output) != 3:
            raise ValueError("SAM predictor must return (masks, scores, logits).")
        masks, scores, logits = (np.asarray(value) for value in output)
        if (
            masks.ndim != 3
            or masks.shape[1:] != image.shape[:2]
            or not masks.shape[0]
            or scores.shape != (masks.shape[0],)
            or logits.ndim != 3
            or logits.shape[0] != masks.shape[0]
            or min(logits.shape, default=0) == 0
            or masks.dtype.kind not in "bifu"
            or scores.dtype.kind not in "ifu"
            or logits.dtype.kind not in "bifu"
            or not np.isfinite(scores).all()
            or not np.isfinite(logits).all()
            # Binary values are finite; boolean masks need no scan.
            or (masks.dtype != np.bool_ and not ((masks == 0) | (masks == 1)).all())
        ):
            raise ValueError(
                "SAM predictor returned invalid masks/scores/logits; expected finite binary masks (N, height, width), scores (N,), and finite logits."
            )
        selected = int(np.argmax(scores))
        steps = 0 if previous is None else previous.steps + 1
        proposal = SegmentationProposal(
            masks[selected].astype(np.bool_),
            score=float(scores[selected]),
            metadata={
                "provider": "segment_anything",
                "selected_mask": selected,
                "refinement_steps": steps,
            },
        )
        assert self._cached_image is not None
        return _SAMRefinement(
            owner=self._refinement_owner,
            image=self._cached_image,
            prompt=prompt,
            proposal=proposal,
            logits=read_only_copy(logits[selected : selected + 1]),
            steps=steps,
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


def mask_to_polygons(
    mask: NDArray[np.bool_], *, simplify_tolerance: float = 0.0
) -> list[Polygon]:
    """Convert every foreground pixel cell to polygons in pixel (x, y).

    Parameters
    ----------
    mask : ndarray of bool, shape (height, width)
        Foreground support; an empty foreground returns an empty list.
    simplify_tolerance : float, default=0.0
        Finite nonnegative simplification distance in pixels. Zero preserves
        exact support and area. Positive values simplify all components
        together with topology preservation, so components cannot overlap and
        every component and hole is kept.

    Returns
    -------
    list of shapely.geometry.Polygon
        Deterministically sorted by bounds and area. Pixel ``(x, y)`` denotes
        its center, so one pixel covers ``[x-.5, x+.5] × [y-.5, y+.5]``.
        All connected components and interior holes are preserved.

    Raises
    ------
    ValueError
        If simplification cannot keep every component and hole valid; use a
        smaller ``simplify_tolerance``.

    Examples
    --------
    >>> mask_to_polygons(np.array([[True, True]]))[0].area
    2.0
    """
    mask = _validate_mask(mask)
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
