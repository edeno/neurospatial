"""Coordinate transforms and video calibration.

`AffineND` is one composable, homogeneous affine transform for points of any
dimension. Build one with a factory (`translate`, `scale_2d`, `flip_y`,
`identity`, `translate_3d`, `scale_3d`, `from_rotation_matrix`,
`identity_nd`), estimate one from landmark pairs with `estimate_transform`, or
calibrate video pixels to centimeters with `calibrate_from_scale_bar`,
`calibrate_from_landmarks` and `VideoCalibration`. Compose transforms with
``@`` and apply one by calling it on an ``(..., n_dims)`` array.

`flip_y_data`, `convert_to_cm` and `convert_to_pixels` apply a pixel/cm
conversion in one call and return the converted array.
"""

# ruff: noqa: N806  - uppercase matrix names (A, R, X) follow mathematical convention
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._validation import is_positive_integer
from neurospatial.geometry import ReadOnlyState, _check_homogeneous

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Core classes
    "AffineND",
    "SpatialTransform",
    "VideoCalibration",
    # 2D factory functions
    "flip_y",
    "identity",
    "scale_2d",
    "translate",
    # 3D factory functions
    "from_rotation_matrix",
    "identity_nd",
    "scale_3d",
    "translate_3d",
    # Calibration functions
    "calibrate_from_landmarks",
    "calibrate_from_scale_bar",
    "simple_scale",
    # Helper functions
    "convert_to_cm",
    "convert_to_pixels",
    "flip_y_data",
    # Transform estimation
    "estimate_transform",
]


# ---------------------------------------------------------------------
# 1.  Composable transform objects
# ---------------------------------------------------------------------
@runtime_checkable
class SpatialTransform(Protocol):
    """Callable that maps an (N, n_dims) array of points → (N, n_dims) array."""

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]: ...


_AFFINE_FIX = (
    "put the linear part in A[:n_dims, :n_dims], the translation in "
    "A[:n_dims, n_dims] and [0, ..., 0, 1] in the last row, or build the "
    "transform with translate(), scale_2d(), from_rotation_matrix() or "
    "estimate_transform()"
)


def _check_affine_matrix(matrix: NDArray[np.float64]) -> None:
    """Raise a ValueError naming what makes ``matrix`` not an invertible affine map."""
    try:
        _check_homogeneous(matrix)
    except ValueError as exc:
        fix = _AFFINE_FIX
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 2
            or matrix.shape[0] != matrix.shape[1]
        ):
            what = (
                "AffineND needs a square (n_dims + 1, n_dims + 1) homogeneous "
                f"matrix, got shape {matrix.shape}."
            )
            why = "the extra row and column hold the translation"
        elif not np.all(np.isfinite(matrix)):
            what = "AffineND got a matrix with NaN or infinite entries."
            why = "a non-finite entry sends every point to NaN or infinity"
        elif not np.array_equal(matrix[-1], np.r_[np.zeros(len(matrix) - 1), 1.0]):
            what = (
                f"AffineND needs the last row [0, ..., 0, 1], got "
                f"{matrix[-1].tolist()}."
            )
            why = (
                "any other last row is a projective map, which bends straight "
                "walls and changes areas unevenly, not an affine one"
            )
        else:
            what = (
                "AffineND got a singular matrix: its linear part "
                f"{matrix[:-1, :-1].tolist()} has zero determinant."
            )
            why = (
                "a singular map collapses space onto a line or plane, so it "
                "has no inverse to carry points, bins or video pixels back"
            )
            fix = "use an invertible matrix with nonzero scale factors; " + fix
        raise ValueError(_format_error(what, why=why, fix=fix)) from exc


@dataclass(frozen=True, eq=False)
class AffineND(ReadOnlyState, SpatialTransform):
    """Invertible affine transform as an (n_dims+1) × (n_dims+1) homogeneous matrix.

    Works for 2D, 3D, or higher dimensions. For N-dimensional points, uses
    (N+1) × (N+1) homogeneous matrix A such that:

        [x'₁, x'₂, ..., x'ₙ, 1]^T  =  A @ [x₁, x₂, ..., xₙ, 1]^T

    Parameters
    ----------
    A : array-like, shape (n_dims+1, n_dims+1)
        Homogeneous transformation matrix. It encodes rotation, scaling,
        translation and shear; it must be finite, its bottom row must be
        ``[0, ..., 0, 1]`` and its linear part ``A[:n_dims, :n_dims]`` must
        be invertible. The transform keeps a read-only copy.

    Attributes
    ----------
    A : NDArray[np.float64], shape (n_dims+1, n_dims+1)
        Read-only homogeneous transformation matrix.
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D, etc.).

    Raises
    ------
    ValueError
        If ``A`` is not square, has NaN or infinite entries, has a bottom row
        other than ``[0, ..., 0, 1]``, or is singular.

    Notes
    -----
    Two transforms are equal (``==``) when their matrices are equal.

    Examples
    --------
    2D scaling followed by translation:

    >>> import numpy as np
    >>> from neurospatial.ops.transforms import translate, scale_2d
    >>> transform = translate(10, 20) @ scale_2d(2.0)
    >>> transform(np.array([[0, 0], [1, 1]]))
    array([[10., 20.],
           [12., 22.]])

    3D translation:

    >>> from neurospatial.ops.transforms import translate_3d
    >>> transform = translate_3d(10, 20, 30)
    >>> points = np.array([[0, 0, 0], [1, 1, 1]])
    >>> transform(points)
    array([[10., 20., 30.],
           [11., 21., 31.]])

    3D rotation using scipy:

    >>> from scipy.spatial.transform import Rotation
    >>> from neurospatial.ops.transforms import from_rotation_matrix
    >>> R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    >>> transform = from_rotation_matrix(R)
    >>> np.allclose(transform(np.array([[1, 0, 0]])), [[0, 1, 0]])
    True

    See Also
    --------
    from_rotation_matrix : Create transform from rotation matrix
    estimate_transform : Fit a transform to landmark pairs
    """

    A: NDArray[np.float64]  # shape (n_dims+1, n_dims+1)

    def __post_init__(self) -> None:
        """Validate the matrix and keep a read-only copy."""
        A = np.array(self.A, dtype=np.float64)
        _check_affine_matrix(A)
        A.setflags(write=False)
        object.__setattr__(self, "A", A)

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)."""
        return int(self.A.shape[0] - 1)

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is an AffineND with an equal matrix."""
        if not isinstance(other, AffineND):
            return NotImplemented
        return bool(np.array_equal(self.A, other.A))

    __hash__ = None  # type: ignore[assignment]

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transformation to N-dimensional points.

        Parameters
        ----------
        pts : NDArray[np.float64], shape (..., n_dims)
            Points to transform (2D, 3D, or higher dimensions).

        Returns
        -------
        NDArray[np.float64], shape (..., n_dims)
            Transformed points.

        Raises
        ------
        ValueError
            If the last axis of ``pts`` does not have length ``n_dims``.

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate_3d
        >>> transform = translate_3d(10, 20, 30)
        >>> points = np.array([[0, 0, 0], [1, 1, 1]])
        >>> transform(points)
        array([[10., 20., 30.],
               [11., 21., 31.]])
        """
        pts = np.asanyarray(pts, dtype=float)
        original_shape = pts.shape
        n_dims = self.n_dims

        # Validate dimensions
        if pts.ndim == 0 or pts.shape[-1] != n_dims:
            given = pts.shape[-1] if pts.ndim else 0
            raise ValueError(
                f"Transform is {n_dims}D but points have {given} dimensions. "
                f"Expected shape (..., {n_dims}), got {pts.shape}"
            )

        # Flatten to (n_points, n_dims)
        pts_flat = pts.reshape(-1, n_dims)
        n_points = pts_flat.shape[0]

        # Add homogeneous coordinate
        pts_h = np.c_[pts_flat, np.ones((n_points, 1))]  # (n_points, n_dims+1)

        # Apply transformation
        out_h = pts_h @ self.A.T  # (n_points, n_dims+1)

        # Normalize by homogeneous coordinate and extract spatial dims
        out = out_h[:, :n_dims] / out_h[:, n_dims : n_dims + 1]

        # Reshape to original shape
        return np.asarray(out.reshape(original_shape), dtype=np.float64)

    def inverse(self) -> AffineND:
        """Compute the inverse transformation.

        Returns
        -------
        AffineND
            New AffineND representing the inverse transformation.

        Raises
        ------
        ValueError
            If the inverse has entries beyond floating-point range (a nearly
            singular transform).

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate_3d
        >>> transform = translate_3d(10, 20, 30)
        >>> inv = transform.inverse()
        >>> points = np.array([[10, 20, 30]])
        >>> inv(points)
        array([[0., 0., 0.]])
        """
        return AffineND(np.asarray(np.linalg.inv(self.A), dtype=np.float64))

    def compose(self, other: AffineND) -> AffineND:
        """Compose this transformation with another.

        Parameters
        ----------
        other : AffineND
            Transformation to compose with (applied first).

        Returns
        -------
        AffineND
            New transformation representing ``self ∘ other``.

        Raises
        ------
        ValueError
            If transforms have different dimensions.

        Notes
        -----
        The result applies `other` first, then `self`. Composition order
        matters: ``a.compose(b)`` ≠ ``b.compose(a)`` in general.

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate_3d, scale_3d
        >>> t1 = translate_3d(10, 0, 0)
        >>> t2 = scale_3d(2.0)
        >>> combined = t1.compose(t2)
        >>> points = np.array([[1, 1, 1]])
        >>> combined(points)
        array([[12.,  2.,  2.]])
        """
        if self.n_dims != other.n_dims:
            raise ValueError(
                f"Cannot compose {self.n_dims}D transform with {other.n_dims}D transform"
            )
        return AffineND(self.A @ other.A)

    def __matmul__(self, other: AffineND) -> AffineND:
        """Compose transformations using @ operator (``other`` applied first)."""
        return self.compose(other)


def identity() -> AffineND:
    """Return the 2D identity transform.

    Returns
    -------
    AffineND
        Identity transformation (no change to input points).

    Examples
    --------
    >>> transform = identity()
    >>> points = np.array([[1, 2], [3, 4]])
    >>> transform(points)
    array([[1., 2.],
           [3., 4.]])

    """
    return AffineND(np.eye(3))


# Factory helpers for the most common ops ---------------------------------
def scale_2d(sx: float = 1.0, sy: float | None = None) -> AffineND:
    """Create uniform or anisotropic scaling transformation.

    Parameters
    ----------
    sx : float, default=1.0
        Scale factor for x-axis.
    sy : float or None, default=None
        Scale factor for y-axis. If None, uses `sx` for uniform scaling.

    Returns
    -------
    AffineND
        Scaling transformation.

    Examples
    --------
    Uniform scaling:

    >>> transform = scale_2d(2.0)
    >>> points = np.array([[1, 2]])
    >>> transform(points)
    array([[2., 4.]])

    Anisotropic scaling:

    >>> transform = scale_2d(sx=2.0, sy=0.5)
    >>> points = np.array([[1, 2]])
    >>> transform(points)
    array([[2., 1.]])

    """
    sy = sx if sy is None else sy
    return AffineND(np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]))


def translate(tx: float = 0.0, ty: float = 0.0) -> AffineND:
    """Create translation transformation.

    Parameters
    ----------
    tx : float, default=0.0
        Translation in x direction.
    ty : float, default=0.0
        Translation in y direction.

    Returns
    -------
    AffineND
        Translation transformation.

    Examples
    --------
    >>> transform = translate(10, 20)
    >>> points = np.array([[0, 0], [1, 1]])
    >>> transform(points)
    array([[10., 20.],
           [11., 21.]])

    """
    return AffineND(np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]))


def flip_y(frame_height_px: float) -> AffineND:
    """Flip the *y*-axis of pixel coordinates so that origin moves
    from top-left to bottom-left.

    Parameters
    ----------
    frame_height_px : float
        Height of the video frame in pixels.

    Returns
    -------
    AffineND
        Transformation that flips y-axis around frame center.

    """
    return AffineND(
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, frame_height_px], [0.0, 0.0, 1.0]]),
    )


# --- Video Calibration Helpers ---------------------------------------
def calibrate_from_scale_bar(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    known_length_cm: float,
    frame_size_px: tuple[int, int],
) -> AffineND:
    """Build pixel→cm transform from a scale bar of known length.

    Creates an affine transformation that converts video pixel coordinates
    to environment centimeter coordinates. The transform includes Y-axis flip
    to convert from video origin (top-left) to environment origin (bottom-left).

    Parameters
    ----------
    p1_px : tuple[float, float]
        First endpoint of scale bar in pixel coordinates (x, y).
    p2_px : tuple[float, float]
        Second endpoint of scale bar in pixel coordinates (x, y).
    known_length_cm : float
        Real-world length of the scale bar in centimeters. Must be positive.
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels.

    Returns
    -------
    AffineND
        Transform that converts pixel coords to cm coords with Y-flip.

    Raises
    ------
    ValueError
        If known_length_cm is not positive, or scale bar has zero length.

    Notes
    -----
    Pixel coordinates use (x_px, y_px) = (column, row) ordering.
    The transform assumes uniform scaling (same cm_per_px for x and y).

    The returned transform composes:
    1. Y-flip via ``flip_y(frame_height)`` - converts top-left to bottom-left origin
    2. Uniform scaling via ``scale_2d(cm_per_px)`` - converts pixels to centimeters

    **Coordinate Flow** (see ``animation/COORDINATES.md`` for details):

    .. code-block:: text

        video_px (y-down, origin top-left)
            │
            │  flip_y(frame_height)
            ▼
        flipped_px (y-up, origin bottom-left)
            │
            │  scale_2d(cm_per_px)
            ▼
        env_cm (y-up, in centimeters)

    **Y-FLIP POLICY**: The Y-flip happens ONCE here in calibration.
    All downstream code (matplotlib imshow, napari) should use ``origin="lower"``
    or equivalent to preserve this orientation. Do NOT add another Y-flip elsewhere.

    Examples
    --------
    >>> from neurospatial.ops.transforms import calibrate_from_scale_bar
    >>> # Scale bar from (100, 200) to (300, 200) represents 50 cm
    >>> transform = calibrate_from_scale_bar(
    ...     p1_px=(100.0, 200.0),
    ...     p2_px=(300.0, 200.0),
    ...     known_length_cm=50.0,
    ...     frame_size_px=(640, 480),
    ... )
    >>> import numpy as np
    >>> point_px = np.array([[200.0, 240.0]])
    >>> point_cm = transform(point_px)
    """
    # Validate inputs
    if known_length_cm <= 0:
        raise ValueError(
            f"WHAT: known_length_cm must be positive (got {known_length_cm}).\n"
            f"WHY: A scale bar must have positive real-world length.\n"
            f"Fix: Provide a positive value for known_length_cm."
        )

    # Compute pixel distance between endpoints
    dx = p2_px[0] - p1_px[0]
    dy = p2_px[1] - p1_px[1]
    px_distance = np.sqrt(dx * dx + dy * dy)

    if px_distance == 0:
        raise ValueError(
            f"WHAT: Scale bar has zero pixel length (p1={p1_px}, p2={p2_px}).\n"
            f"WHY: Cannot compute scale from coincident endpoints.\n"
            f"Fix: Provide two distinct points for the scale bar."
        )

    # Compute cm per pixel (uniform scaling assumed)
    cm_per_px = known_length_cm / px_distance

    # Compose: Y-flip then scale
    # Order: scale_2d @ flip_y means "apply flip_y first, then scale_2d"
    # flip_y converts y_px to (frame_height - y_px)
    # scale_2d converts pixels to cm
    _, frame_height = frame_size_px
    return scale_2d(cm_per_px, cm_per_px) @ flip_y(frame_height)


def calibrate_from_landmarks(
    landmarks_px: NDArray[np.float64],
    landmarks_cm: NDArray[np.float64],
    frame_size_px: tuple[int, int],
    *,
    kind: str = "similarity",
) -> AffineND:
    """Build pixel→cm transform from corresponding landmark pairs.

    Creates an affine transformation from at least 3 corresponding point pairs
    using the specified transform type (rigid, similarity, or affine).

    Parameters
    ----------
    landmarks_px : ndarray of shape (n_points, 2)
        Landmark coordinates in video pixels as (x_px, y_px) = (column, row).
        Must have at least 3 points.
    landmarks_cm : ndarray of shape (n_points, 2)
        Corresponding coordinates in environment space as (x_cm, y_cm).
        Must have the same number of points as landmarks_px.
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels.
    kind : {"rigid", "similarity", "affine"}, default="similarity"
        Type of transform to estimate:
        - "rigid": rotation + translation (3 DOF)
        - "similarity": uniform scale + rotation + translation (4 DOF)
        - "affine": full affine (6 DOF)

    Returns
    -------
    AffineND
        Transform that converts pixel coords to cm coords.

    Raises
    ------
    ValueError
        If fewer than 3 landmarks provided, landmark arrays have different
        lengths, a landmark has a NaN or infinite coordinate, the landmarks
        are collinear, or transform estimation fails.

    Notes
    -----
    Pixel coordinates use (x_px, y_px) = (column, row) ordering in image space.
    The frame_size_px is used to apply Y-flip before transform estimation,
    converting from video coordinates (origin top-left) to standard Cartesian
    coordinates (origin bottom-left). The returned transform includes this
    Y-flip, so pass raw video pixel coordinates to it without flipping them first.

    For best results, use landmarks that:
    - Span the full video frame
    - Are well-distributed (not collinear)
    - Have accurate pixel and world coordinates

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.transforms import calibrate_from_landmarks
    >>> # Arena corners in video pixels and environment cm
    >>> corners_px = np.array([[50, 50], [590, 50], [590, 430], [50, 430]], dtype=float)
    >>> corners_cm = np.array([[0, 80], [100, 80], [100, 0], [0, 0]], dtype=float)
    >>> transform = calibrate_from_landmarks(
    ...     landmarks_px=corners_px,
    ...     landmarks_cm=corners_cm,
    ...     frame_size_px=(640, 480),
    ...     kind="similarity",
    ... )
    """
    # Validate landmark arrays
    landmarks_px = np.asarray(landmarks_px, dtype=np.float64)
    landmarks_cm = np.asarray(landmarks_cm, dtype=np.float64)

    if landmarks_px.ndim != 2 or landmarks_px.shape[1] != 2:
        raise ValueError(
            f"WHAT: landmarks_px must have shape (n_points, 2), got {landmarks_px.shape}.\n"
            f"WHY: Each landmark needs (x, y) coordinates.\n"
            f"Fix: Provide landmarks as shape (n_points, 2) array."
        )

    if landmarks_cm.ndim != 2 or landmarks_cm.shape[1] != 2:
        raise ValueError(
            f"WHAT: landmarks_cm must have shape (n_points, 2), got {landmarks_cm.shape}.\n"
            f"WHY: Each landmark needs (x, y) coordinates.\n"
            f"Fix: Provide landmarks as shape (n_points, 2) array."
        )

    n_px = len(landmarks_px)
    n_cm = len(landmarks_cm)

    if n_px != n_cm:
        raise ValueError(
            f"WHAT: landmarks_px ({n_px} points) and landmarks_cm ({n_cm} points) "
            f"must have the same number of points.\n"
            f"WHY: Each pixel landmark must have a corresponding cm landmark.\n"
            f"Fix: Ensure both arrays have the same length."
        )

    if n_px < 3:
        raise ValueError(
            f"WHAT: Need at least 3 landmarks for transform estimation (got {n_px}).\n"
            f"WHY: Each landmark gives 2 equations; rigid (3 degrees of freedom) "
            f"and similarity (4) fits need 2 landmarks and affine (6) fits need 3, "
            f"and a third landmark lets the fit residual reveal a misplaced one.\n"
            f"Fix: Provide at least 3 non-collinear landmark pairs."
        )

    _require_finite_landmarks(landmarks_px, "landmarks_px")
    _require_finite_landmarks(landmarks_cm, "landmarks_cm")

    # Apply Y-flip to pixel coordinates before estimation
    # This converts from video coords (origin top-left) to standard coords
    _, frame_height = frame_size_px
    landmarks_px_flipped = landmarks_px.copy()
    landmarks_px_flipped[:, 1] = frame_height - landmarks_px[:, 1]

    # Check for collinear/degenerate landmarks before estimation
    # Use singular values of centered points to detect if they span 2D
    # If the ratio S[0]/S[1] is huge, points are nearly collinear
    centered_px = landmarks_px_flipped - landmarks_px_flipped.mean(axis=0)
    centered_cm = landmarks_cm - landmarks_cm.mean(axis=0)

    _, s_px, _ = np.linalg.svd(centered_px, full_matrices=False)
    _, s_cm, _ = np.linalg.svd(centered_cm, full_matrices=False)

    # Check both source and destination for collinearity
    # Condition threshold: smallest singular value should be > 1e-6 * largest
    for name, s in [("pixel", s_px), ("environment", s_cm)]:
        if len(s) >= 2 and s[1] < 1e-6 * s[0]:
            ratio = s[0] / (s[1] + 1e-15)
            raise ValueError(
                f"WHAT: Landmark calibration has ill-conditioned {name} points "
                f"(spread ratio={ratio:.1e}).\n"
                f"WHY: Landmarks are collinear or too close together. "
                f"2D transforms require points that span both x and y directions.\n"
                f"Fix: Use landmarks that span the full video frame with good spread. "
                f"Ensure at least 3 non-collinear points."
            )

    result = estimate_transform(landmarks_px_flipped, landmarks_cm, kind=kind)

    # The fit accepts flipped pixels; compose the flip so callers can supply
    # raw image pixels, as they do for calibrate_from_scale_bar.
    return result @ flip_y(frame_height)


@dataclass(frozen=True, eq=False)
class VideoCalibration:
    """Video-to-environment coordinate calibration.

    Stores the pixel-to-cm transform, its inverse and the frame size it was
    measured on. It is immutable, so the inverse always matches the forward
    transform.

    Parameters
    ----------
    transform_px_to_cm : AffineND
        2D transform that converts pixel coordinates to cm coordinates.
        Should include Y-axis flip (video origin top-left to env bottom-left).
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels; two positive integers.

    Attributes
    ----------
    transform_px_to_cm : AffineND
        Forward transform from pixels to cm.
    frame_size_px : tuple[int, int]
        Video frame dimensions.
    transform_cm_to_px : AffineND
        Inverse transform from cm to pixels, computed at construction.
    cm_per_px : float
        Approximate scale factor (computed from transform matrix).

    Raises
    ------
    TypeError
        If ``transform_px_to_cm`` is not an ``AffineND``.
    ValueError
        If the transform is not 2D, or ``frame_size_px`` is not two positive
        integers.

    Notes
    -----
    The calibration can be serialized to/from dict for JSON storage. Two
    calibrations are equal (``==``) when their transforms and frame sizes are.

    Examples
    --------
    >>> from neurospatial.ops.transforms import (
    ...     calibrate_from_scale_bar,
    ...     VideoCalibration,
    ... )
    >>> transform = calibrate_from_scale_bar(
    ...     p1_px=(0.0, 0.0),
    ...     p2_px=(100.0, 0.0),
    ...     known_length_cm=50.0,
    ...     frame_size_px=(640, 480),
    ... )
    >>> calib = VideoCalibration(transform, frame_size_px=(640, 480))
    >>> calib.cm_per_px  # Approximate scale
    0.5
    """

    transform_px_to_cm: AffineND
    frame_size_px: tuple[int, int]
    transform_cm_to_px: AffineND = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the transform and frame size and compute the inverse."""
        transform = self.transform_px_to_cm
        if not isinstance(transform, AffineND):
            raise TypeError(
                _format_error(
                    "VideoCalibration needs an AffineND transform_px_to_cm, got "
                    f"{type(transform).__name__}.",
                    fix="wrap the 3 x 3 homogeneous matrix: AffineND(matrix), or "
                    "build it with calibrate_from_scale_bar or "
                    "calibrate_from_landmarks",
                )
            )
        if transform.n_dims != 2:
            raise ValueError(
                _format_error(
                    "VideoCalibration needs a 2D transform from (x, y) pixels, "
                    f"got a {transform.n_dims}D one.",
                    fix="pass a transform built from a 3 x 3 homogeneous matrix",
                )
            )
        size: tuple[object, ...] = tuple(self.frame_size_px)
        if len(size) != 2 or not all(is_positive_integer(n) for n in size):
            raise ValueError(
                _format_error(
                    "frame_size_px must be two positive integers (width, height), "
                    f"got {self.frame_size_px!r}.",
                    fix="pass the video frame size in pixels, for example "
                    "(frame.shape[1], frame.shape[0])",
                )
            )
        object.__setattr__(
            self,
            "frame_size_px",
            (int(size[0]), int(size[1])),  # type: ignore[call-overload]
        )
        object.__setattr__(self, "transform_cm_to_px", transform.inverse())

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` has an equal transform and frame size."""
        if not isinstance(other, VideoCalibration):
            return NotImplemented
        return (
            self.transform_px_to_cm == other.transform_px_to_cm
            and self.frame_size_px == other.frame_size_px
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def cm_per_px(self) -> float:
        """Get approximate scale factor (cm per pixel).

        Returns the mean length of the transformed x and y unit vectors.
        This is invariant under rotations and reflections; for anisotropic
        scaling or shear it summarizes the two axis scales with their mean.

        Returns
        -------
        float
            Approximate cm per pixel scale factor.
        """
        axis_scales = np.linalg.norm(self.transform_px_to_cm.A[:2, :2], axis=0)
        return float(np.mean(axis_scales))

    def to_dict(self) -> dict:
        """Serialize calibration to dict for JSON storage.

        Returns
        -------
        dict
            Dictionary with transform matrix and frame size.
        """
        return {
            "transform_px_to_cm": self.transform_px_to_cm.A.tolist(),
            "frame_size_px": list(self.frame_size_px),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VideoCalibration:
        """Restore calibration from dict.

        Parameters
        ----------
        d : dict
            Dictionary from to_dict().

        Returns
        -------
        VideoCalibration
            Restored calibration object.
        """
        transform = AffineND(np.array(d["transform_px_to_cm"], dtype=np.float64))
        return cls(
            transform_px_to_cm=transform, frame_size_px=tuple(d["frame_size_px"])
        )


# --- 3D Transform Factories ------------------------------------------
def translate_3d(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> AffineND:
    """Create 3D translation transformation.

    Parameters
    ----------
    tx : float, default=0.0
        Translation in x direction.
    ty : float, default=0.0
        Translation in y direction.
    tz : float, default=0.0
        Translation in z direction.

    Returns
    -------
    AffineND
        3D translation transformation.

    Examples
    --------
    >>> from neurospatial.ops.transforms import translate_3d
    >>> transform = translate_3d(10, 20, 30)
    >>> points = np.array([[0, 0, 0], [1, 1, 1]])
    >>> transform(points)
    array([[10., 20., 30.],
           [11., 21., 31.]])
    """
    A = np.eye(4)
    A[:3, 3] = [tx, ty, tz]
    return AffineND(A)


def scale_3d(
    sx: float = 1.0, sy: float | None = None, sz: float | None = None
) -> AffineND:
    """Create 3D scaling transformation.

    Parameters
    ----------
    sx : float, default=1.0
        Scale factor for x-axis.
    sy : float or None, default=None
        Scale factor for y-axis. If None, uses `sx` for uniform scaling.
    sz : float or None, default=None
        Scale factor for z-axis. If None, uses `sx` for uniform scaling.

    Returns
    -------
    AffineND
        3D scaling transformation.

    Examples
    --------
    Uniform scaling:

    >>> from neurospatial.ops.transforms import scale_3d
    >>> transform = scale_3d(2.0)
    >>> points = np.array([[1, 2, 3]])
    >>> transform(points)
    array([[2., 4., 6.]])

    Anisotropic scaling:

    >>> transform = scale_3d(sx=2.0, sy=0.5, sz=3.0)
    >>> points = np.array([[1, 2, 3]])
    >>> transform(points)
    array([[2., 1., 9.]])
    """
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    A = np.diag([sx, sy, sz, 1.0])
    return AffineND(A)


def from_rotation_matrix(
    rotation_matrix: NDArray[np.float64],
    translation: NDArray[np.float64] | None = None,
) -> AffineND:
    """Create affine transform from rotation matrix (2D or 3D).

    Parameters
    ----------
    rotation_matrix : NDArray[np.float64], shape (n_dims, n_dims)
        Rotation matrix: orthogonal (``R @ R.T == I`` within 1e-6) with
        determinant +1. For 3D, use scipy.spatial.transform.Rotation to
        generate.
    translation : NDArray[np.float64], shape (n_dims,), optional
        Translation vector. If None, no translation is applied.

    Returns
    -------
    AffineND
        Affine transformation combining rotation and optional translation.

    Raises
    ------
    ValueError
        If ``rotation_matrix`` is not square, not orthogonal, or a reflection
        (determinant -1), or ``translation`` has the wrong shape.

    Examples
    --------
    3D rotation from scipy:

    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> from neurospatial.ops.transforms import from_rotation_matrix
    >>> # 90-degree rotation around z-axis
    >>> rot_mat = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    >>> transform = from_rotation_matrix(rot_mat)
    >>> points = np.array([[1, 0, 0], [0, 1, 0]])
    >>> transformed = transform(points)
    >>> np.allclose(transformed, [[0, 1, 0], [-1, 0, 0]])
    True

    With translation:

    >>> transform = from_rotation_matrix(rot_mat, translation=np.array([10, 20, 30]))
    >>> points = np.array([[1, 0, 0]])
    >>> np.allclose(transform(points), [[10, 21, 30]])
    True
    """
    rot = np.asarray(rotation_matrix, dtype=float)
    if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
        raise ValueError(f"Rotation matrix must be square, got shape {rot.shape}")

    n_dims = rot.shape[0]
    if not np.all(np.isfinite(rot)) or not np.allclose(
        rot @ rot.T, np.eye(n_dims), rtol=0, atol=1e-6
    ):
        raise ValueError(
            _format_error(
                "rotation_matrix is not orthogonal: R @ R.T differs from the "
                "identity by up to "
                f"{np.max(np.abs(rot @ rot.T - np.eye(n_dims))):.3g}.",
                why="a rotation keeps lengths and angles; this matrix would also "
                "scale or shear the points",
                fix="build the rotation with scipy.spatial.transform.Rotation "
                "(.as_matrix()), or wrap a general linear map in AffineND",
            )
        )
    if np.linalg.det(rot) < 0:
        raise ValueError(
            _format_error(
                "rotation_matrix has determinant -1: it is a reflection, not a "
                "rotation.",
                why="a reflection mirrors the arena, which a rotation never does",
                fix="check the axis order or sign of the matrix; to reflect on "
                "purpose, build the transform with AffineND(matrix)",
            )
        )
    A = np.eye(n_dims + 1)
    A[:n_dims, :n_dims] = rot

    if translation is not None:
        t = np.asarray(translation, dtype=float)
        if t.shape != (n_dims,):
            raise ValueError(
                f"Translation vector must have shape ({n_dims},), got {t.shape}"
            )
        A[:n_dims, n_dims] = t

    return AffineND(A)


def identity_nd(*, n_dims: int = 2) -> AffineND:
    """Return the N-dimensional identity transform.

    Parameters
    ----------
    n_dims : int, default=2
        Number of dimensions (2 for 2D, 3 for 3D, etc.).

    Returns
    -------
    AffineND
        Identity transformation (no change to input points).

    Examples
    --------
    >>> from neurospatial.ops.transforms import identity_nd
    >>> transform = identity_nd(n_dims=3)
    >>> points = np.array([[1, 2, 3]])
    >>> transform(points)
    array([[1., 2., 3.]])
    """
    return AffineND(np.eye(n_dims + 1))


# ---------------------------------------------------------------------
# Quick NumPy helpers that build and apply a 2D AffineND
# ---------------------------------------------------------------------
def flip_y_data(
    data: NDArray[np.float64] | tuple | list,
    frame_size_px: tuple[float, float],
) -> NDArray[np.float64]:
    """Flip y-axis of coordinates so that the origin moves from
    image-space top-left to Cartesian bottom-left.

    Parameters
    ----------
    data : NDArray[np.float64] or tuple or list
        Input coordinates in pixel space, shape (..., 2).
    frame_size_px : tuple[float, float]
        Size of the video frame in pixels (width, height).

    Returns
    -------
    NDArray[np.float64]
        Flipped coordinates, shape (..., 2).

    Notes
    -----
    Equivalent to::

        AffineND([[1, 0, 0], [0, -1, H], [0, 0, 1]])(data)

    but without the user having to build the transform.

    """
    transform = flip_y(frame_height_px=frame_size_px[1])
    return transform(np.asanyarray(data, dtype=float))


def _check_cm_per_px(cm_per_px: float) -> None:
    if not (np.isfinite(cm_per_px) and cm_per_px > 0):
        raise ValueError(
            _format_error(
                f"cm_per_px must be a positive, finite number, got {cm_per_px!r}.",
                why="it is the length in centimeters of one video pixel",
                fix="measure it from a scale bar or a known arena dimension "
                "(calibrate_from_scale_bar does this from two pixel points)",
            )
        )


def convert_to_cm(
    data_px: NDArray[np.float64] | tuple | list,
    frame_size_px: tuple[float, float],
    *,
    cm_per_px: float,
) -> NDArray[np.float64]:
    """Convert pixel coordinates to centimeter coordinates.

    Pixel  →  centimeter coordinates *and* y-flip in one shot.

    Internally constructs ``scale_2d(cm_per_px) @ flip_y(H)`` and applies it.

    Parameters
    ----------
    data_px : array-like
        Input coordinates in pixel space, shape (..., 2).
    frame_size_px : tuple[float, float]
        Size of the video frame in pixels (width, height).
    cm_per_px : float
        Length of one pixel in centimeters. Required: there is no default
        that would not silently relabel pixels as centimeters.

    Returns
    -------
    NDArray[np.float64]
        Converted coordinates in centimeters, shape (..., 2).

    Raises
    ------
    ValueError
        If ``cm_per_px`` is not a positive, finite number.

    Examples
    --------
    >>> convert_to_cm(np.array([[100.0, 80.0]]), (640, 480), cm_per_px=0.5)
    array([[ 50., 200.]])

    """
    _check_cm_per_px(cm_per_px)
    transform = scale_2d(cm_per_px) @ flip_y(frame_height_px=frame_size_px[1])
    return transform(np.asanyarray(data_px, dtype=float))


def convert_to_pixels(
    data_cm: NDArray[np.float64] | tuple | list,
    frame_size_px: tuple[float, float],
    *,
    cm_per_px: float,
) -> NDArray[np.float64]:
    """Convert centimeter coordinates to pixel coordinates with y-flip.

    Parameters
    ----------
    data_cm : NDArray[np.float64] or tuple or list
        Input coordinates in centimeter space, shape (..., 2).
    frame_size_px : tuple[float, float]
        Size of the video frame in pixels (width, height).
    cm_per_px : float
        Length of one pixel in centimeters. Required, as for `convert_to_cm`.

    Returns
    -------
    NDArray[np.float64]
        Converted coordinates in pixel space, shape (..., 2).

    Raises
    ------
    ValueError
        If ``cm_per_px`` is not a positive, finite number.

    Notes
    -----
    Inverse of `convert_to_cm`. Internally constructs ``flip_y(H) @ scale_2d(1/cm_per_px)``.

    """
    _check_cm_per_px(cm_per_px)
    transform = flip_y(frame_height_px=frame_size_px[1]) @ scale_2d(1.0 / cm_per_px)
    return transform(np.asanyarray(data_cm, dtype=float))


def _require_finite_landmarks(points: NDArray[np.float64], name: str) -> None:
    """Refuse landmark rows with NaN or infinite coordinates, naming them."""
    bad = np.flatnonzero(~np.all(np.isfinite(points), axis=1))
    if bad.size:
        raise ValueError(
            _format_error(
                f"{name} has NaN or infinite coordinates in row(s) {bad.tolist()}.",
                why="every landmark enters the fit, so a missing coordinate "
                "leaves the transform undefined",
                fix=f"drop row(s) {bad.tolist()} from both landmark arrays, or "
                "annotate those landmarks",
            )
        )


# ---------------------------------------------------------------------
# 3.  Transform estimation from point correspondences
# ---------------------------------------------------------------------
def estimate_transform(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    *,
    kind: str = "rigid",
) -> AffineND:
    """Estimate transformation from point correspondences (2D or 3D).

    Given pairs of corresponding points in source and destination coordinate
    systems, compute the best-fit transformation (rigid, similarity, or affine).
    Automatically detects dimensionality from input points.

    Parameters
    ----------
    src : NDArray[np.float64], shape (N, n_dims)
        Source points (N >= 2 for rigid/similarity, N >= 3 for affine).
        n_dims can be 2 (2D), 3 (3D), or higher.
    dst : NDArray[np.float64], shape (N, n_dims)
        Destination points corresponding to src (same dimensionality).
    kind : {"rigid", "similarity", "affine"}, default="rigid"
        Type of transformation to estimate:

        - "rigid": Rotation + translation (preserves distances and angles)
        - "similarity": Rotation + uniform scaling + translation
          (preserves angles, scales distances uniformly)
        - "affine": Full affine (rotation, scaling, shear, translation).
          Fitted to noisy landmarks, it absorbs the noise as a small stretch
          or shear, so smoothing on an environment transformed by it refuses.
          Use "similarity" or "rigid" to align an arena whose shape is known.

    Returns
    -------
    AffineND
        Estimated transformation that maps src → dst.
        For 2D points, returns 3×3 matrix. For 3D points, returns 4×4 matrix.

    Raises
    ------
    ValueError
        If there are too few point pairs for the requested kind; if a
        landmark has a NaN or infinite coordinate (the message names its
        row); if the landmarks do not determine the transform (coincident
        landmarks; for rigid and similarity fits, 3-D landmarks on one line;
        for affine fits, landmarks that do not span every axis); or if src
        and dst differ in shape.

    Examples
    --------
    2D transformation:

    >>> import numpy as np
    >>> from neurospatial.ops.transforms import estimate_transform
    >>> # Define corresponding 2D points
    >>> src_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    >>> # Rotated 45 degrees and translated
    >>> angle = np.pi / 4
    >>> dst_pts = src_pts @ [
    ...     [np.cos(angle), -np.sin(angle)],
    ...     [np.sin(angle), np.cos(angle)],
    ... ] + [5, 5]
    >>> transform = estimate_transform(src_pts, dst_pts, kind="rigid")
    >>> transformed = transform(src_pts)
    >>> np.allclose(transformed, dst_pts)
    True

    3D transformation:

    >>> from scipy.spatial.transform import Rotation
    >>> # Create 3D source points
    >>> src_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> # Apply known transformation
    >>> R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    >>> dst_3d = src_3d @ R.T + [10, 20, 30]
    >>> # Estimate transformation
    >>> transform = estimate_transform(src_3d, dst_3d, kind="rigid")
    >>> transformed = transform(src_3d)
    >>> np.allclose(transformed, dst_3d)
    True

    Notes
    -----
    All three kinds minimize the summed squared distance between the
    transformed src landmarks and the dst landmarks, in any dimensionality.
    Rigid and similarity fits find the rotation with the Kabsch SVD solution
    and its determinant sign-correction, so the rotation is always proper
    (det = +1), even when the landmarks are coplanar or reflected. The
    similarity scale is the least-squares scale for that rotation
    (Umeyama 1991). Affine fits solve the linear least-squares problem
    directly.

    Umeyama, S. (1991). Least-squares estimation of transformation parameters
    between two point patterns. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 13(4), 376-380.

    For cross-session alignment, collect 3-4 landmark points (e.g., corners
    of arena) in both sessions and use this function to compute the alignment.

    See Also
    --------
    AffineND : N-D affine transformation class
    Environment.apply_transform : Apply a transform to an environment.

    """
    src = np.asanyarray(src, dtype=float)
    dst = np.asanyarray(dst, dtype=float)

    if src.shape != dst.shape:
        raise ValueError(
            f"src and dst must have same shape, got {src.shape} and {dst.shape}"
        )

    if src.ndim != 2:
        raise ValueError(
            f"src and dst must be 2D arrays (N points x n_dims), got shape {src.shape}"
        )

    for name, points in (("src", src), ("dst", dst)):
        _require_finite_landmarks(points, name)

    n_points = src.shape[0]
    n_dims = src.shape[1]
    landmark_fix = (
        "use landmarks spread across the arena, such as its corners, so they "
        "span every axis"
    )

    if kind in ("rigid", "similarity"):
        if n_points < 2:
            raise ValueError(
                f"{kind} transform requires at least 2 point pairs, got {n_points}"
            )

        # Center the points
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        if not np.any(src_centered):
            raise ValueError(
                _format_error(
                    f"All {n_points} src landmarks are at the same point.",
                    why=f"coincident landmarks fix no rotation{' or scale' if kind == 'similarity' else ''}",
                    fix=landmark_fix,
                )
            )

        # Kabsch: the rotation R minimizing sum ||R @ src_i - dst_i||^2 over
        # the centered points comes from the SVD of the cross-covariance
        # H = src_centered.T @ dst_centered = U @ diag(S) @ Vt, as
        # R = V @ D @ U.T with D = diag(1, ..., 1, d) and d = sign(det(V @ U.T)).
        # Folding d into the factors gives the closest proper rotation
        # (det = +1) when the optimum would otherwise be a reflection.
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        # R is unique only when H has rank n_dims - 1 or more: landmarks on a
        # line in 3-D leave the turn about that line free, and dst landmarks
        # that coincide leave every rotation equally good.
        tolerance = S[0] * max(n_points, n_dims) * np.finfo(np.float64).eps
        if S[0] == 0 or (n_dims > 1 and S[n_dims - 2] <= tolerance):
            raise ValueError(
                _format_error(
                    f"The landmarks do not determine a {n_dims}-D rotation: "
                    "the src or dst landmarks coincide or lie on one line.",
                    why="such landmarks leave the rotation free, so any answer "
                    "would be arbitrary",
                    fix=landmark_fix,
                )
            )
        V = Vt.T
        d = np.sign(np.linalg.det(V @ U.T))
        # With d = -1 the best proper rotation turns against the smallest
        # singular direction; if that singular value is repeated, every
        # direction in its plane serves equally well and R is free.
        if d < 0 and n_dims > 1 and S[n_dims - 2] - S[n_dims - 1] <= tolerance:
            raise ValueError(
                _format_error(
                    "The dst landmarks are a mirror image of the src landmarks, "
                    f"so no {n_dims}-D rotation fits them better than any other.",
                    why="a rigid or similarity transform cannot reflect, and for "
                    "these landmarks every rotation leaves the same error",
                    fix="list the landmarks in the same order in src and dst and "
                    "flip image y to point up (calibrate_from_landmarks does "
                    'this); to fit a reflection on purpose, use kind="affine"',
                )
            )
        D = np.eye(n_dims)
        D[-1, -1] = d
        R = V @ D @ U.T

        if kind == "rigid":
            scale = 1.0
        else:
            # Least-squares scale for that rotation (Umeyama 1991):
            # trace(diag(S) @ D) / sum ||src_centered||^2.
            scale = float(S @ np.diag(D)) / float(np.sum(src_centered**2))

        # T(x) = scale * R @ x + t, with t mapping the src centroid onto
        # the dst centroid.
        A = np.eye(n_dims + 1)
        A[:n_dims, :n_dims] = scale * R
        A[:n_dims, n_dims] = dst_mean - scale * (R @ src_mean)
        return AffineND(A)

    elif kind == "affine":
        if n_points < n_dims + 1:
            raise ValueError(
                f"affine transform requires at least {n_dims + 1} point pairs "
                f"for {n_dims}D, got {n_points}"
            )

        # Least squares for every output coordinate at once: rows of the
        # design matrix are [x1, ..., xn, 1], so X @ result = dst with result
        # of shape (n_dims+1, n_dims) holding A[:n_dims, :].T.
        X = np.c_[src, np.ones(n_points)]  # (N, n_dims+1)
        result, _, rank, _ = np.linalg.lstsq(X, dst, rcond=None)
        if rank < n_dims + 1:
            span = {0: "one point", 1: "one line", 2: "one plane"}.get(
                rank - 1, f"a {rank - 1}-D subspace"
            )
            raise ValueError(
                _format_error(
                    f"The src landmarks lie on {span}, so they do not determine "
                    f"a {n_dims}-D affine transform.",
                    why="an affine fit needs landmarks that span every axis; "
                    "away from them the fitted map is arbitrary",
                    fix=landmark_fix
                    + (
                        ', or fit kind="similarity", which such landmarks determine'
                        if rank == n_dims
                        else ""
                    ),
                )
            )
        A = np.eye(n_dims + 1)
        A[:n_dims, :] = result.T
        return AffineND(A)

    else:
        raise ValueError(
            f"Invalid kind: {kind!r}. Must be 'rigid', 'similarity', or 'affine'."
        )


# ---------------------------------------------------------------------
# 4.  Simple calibration helper
# ---------------------------------------------------------------------
def simple_scale(
    px_per_cm: float,
    offset_px: tuple[float, float] = (0.0, 0.0),
) -> AffineND:
    """Create a simple transform that converts pixel units to centimeters.

    This returns an AffineND whose matrix, when applied to [x_px, y_px, 1]^T,
    yields coordinates in centimeters.

    Parameters
    ----------
    px_per_cm : float
        Number of pixels per centimeter. Must be nonzero.
    offset_px : tuple of two floats, optional (default: (0.0, 0.0))
        A pixel offset (x_offset, y_offset). The returned transform first
        subtracts this offset (in pixels) before scaling.

    Returns
    -------
    AffineND
        An affine transformation that converts pixel coordinates to centimeters.

    Raises
    ------
    ValueError
        If `px_per_cm` is zero.

    Notes
    -----
    The returned transformation represents the following matrix operation::

        [x_cm]   [1/px_per_cm      0       -offset_px[0]/px_per_cm]   [x_px]
        [y_cm] = [     0       1/px_per_cm  -offset_px[1]/px_per_cm] * [y_px]
        [ 1  ]   [     0            0                  1            ]   [ 1  ]

    Examples
    --------
    >>> from neurospatial.ops.transforms import simple_scale
    >>> transform = simple_scale(px_per_cm=10.0)
    >>> import numpy as np
    >>> points = np.array([[10.0, 20.0]])
    >>> transform(points)
    array([[1., 2.]])

    With offset:

    >>> transform = simple_scale(px_per_cm=10.0, offset_px=(5.0, 10.0))
    >>> points = np.array([[5.0, 10.0]])
    >>> transform(points)
    array([[0., 0.]])

    """
    if px_per_cm == 0:
        raise ValueError("px_per_cm must be nonzero to avoid division by zero.")

    # Compute scale factors
    sx = sy = 1.0 / px_per_cm

    # Ensure offset_px has exactly two values
    try:
        ox, oy = float(offset_px[0]), float(offset_px[1])
    except (TypeError, IndexError, ValueError) as e:
        raise ValueError(
            f"offset_px must be a tuple of two numeric values (x, y), got {type(offset_px).__name__} with value {offset_px}.",
        ) from e

    # Build a 3×3 affine matrix: scale then translate
    tx = -ox * sx
    ty = -oy * sy
    A = np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return AffineND(A)
