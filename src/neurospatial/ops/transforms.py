"""Coordinate transforms and calibration (ops/transforms.py).
=============================================================

N-dimensional affine transformations with a unified API.

New Import Path (v0.4+)
-----------------------
This module has moved from ``neurospatial.transforms`` to ``neurospatial.ops.transforms``.

.. code-block:: python

    # New import path (recommended)
    from neurospatial.ops.transforms import Affine2D, translate, scale_2d

    # Old path still works via re-export from top-level __init__.py
    from neurospatial.transforms import Affine2D  # deprecated

Two complementary APIs
----------------------
1.  *Composable objects* (`AffineND`, `Affine2D`, `Affine3D`)
    Build a transform once, reuse everywhere, keep provenance.
2.  *Quick helpers* (`flip_y_data`, `convert_to_cm`, `convert_to_pixels`)
    One-liners for scripts that just need a NumPy array back.

For 2D (backward compatible):
    Use `Affine2D` or factory functions like `translate()`, `scale_2d()`.

For 3D (new in v0.3):
    Use `Affine3D` or factory functions like `translate_3d()`, `scale_3d()`,
    or `from_rotation_matrix()` with scipy.spatial.transform.Rotation.

Notes
-----
**Version History**:
- v0.2.x and earlier: 2D-only (`Affine2D`)
- v0.3+: Added 3D support (`AffineND`, `Affine3D`)
- v0.4+: Moved to ``neurospatial.ops.transforms``; merged ``calibration.py``
"""

# ruff: noqa: N806  - uppercase matrix names (A, R, X) follow mathematical convention
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.layout.base import LayoutEngine

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Core classes
    "Affine2D",
    "Affine3D",
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
    "apply_transform_to_environment",
    "estimate_transform",
]


# ---------------------------------------------------------------------
# 1.  Composable transform objects
# ---------------------------------------------------------------------
@runtime_checkable
class SpatialTransform(Protocol):
    """Callable that maps an (N, n_dims) array of points → (N, n_dims) array."""

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class AffineND(SpatialTransform):
    """N-D affine transform expressed as an (n_dims+1) × (n_dims+1) homogeneous matrix.

    Works for 2D, 3D, or higher dimensions. For N-dimensional points, uses
    (N+1) × (N+1) homogeneous matrix A such that:

        [x'₁, x'₂, ..., x'ₙ, 1]^T  =  A @ [x₁, x₂, ..., xₙ, 1]^T

    Attributes
    ----------
    A : NDArray[np.float64], shape (n_dims+1, n_dims+1)
        Homogeneous transformation matrix. Encodes rotation, scaling,
        translation, and shear. The bottom row is always [0, 0, ..., 0, 1].
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D, etc.).

    Examples
    --------
    3D translation:

    >>> import numpy as np
    >>> from neurospatial.ops.transforms import translate_3d
    >>> transform = translate_3d(10, 20, 30)
    >>> points = np.array([[0, 0, 0], [1, 1, 1]])
    >>> transformed = transform(points)
    >>> transformed
    array([[10., 20., 30.],
           [11., 21., 31.]])

    3D rotation using scipy:

    >>> from scipy.spatial.transform import Rotation
    >>> from neurospatial.ops.transforms import from_rotation_matrix
    >>> R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    >>> transform = from_rotation_matrix(R)
    >>> points = np.array([[1, 0, 0]])
    >>> transformed = transform(points)
    >>> np.allclose(transformed, [[0, 1, 0]])
    True

    See Also
    --------
    Affine2D : Alias for 2D transforms (backward compatible)
    Affine3D : Alias for 3D transforms
    from_rotation_matrix : Create transform from rotation matrix
    """

    A: NDArray[np.float64]  # shape (n_dims+1, n_dims+1)

    def __post_init__(self) -> None:
        """Validate transformation matrix shape."""
        A = np.asarray(self.A, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Affine matrix must be square, got shape {A.shape}")
        # Store validated array
        object.__setattr__(self, "A", A)

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)."""
        return int(self.A.shape[0] - 1)

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
        if pts.shape[-1] != n_dims:
            raise ValueError(
                f"Transform is {n_dims}D but points have {pts.shape[-1]} dimensions. "
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
        np.linalg.LinAlgError
            If transformation matrix is singular (non-invertible).

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
        """Compose transformations using @ operator."""
        return self.compose(other)


@dataclass(frozen=True, slots=True)
class Affine2D(SpatialTransform):
    """2-D affine transform expressed as a 3 × 3 homogeneous matrix *A* such that

        [x', y', 1]^T  =  A @ [x, y, 1]^T

    Attributes
    ----------
    A : NDArray[np.float64], shape (3, 3)
        Homogeneous transformation matrix representing the affine transformation.
        The matrix encodes rotation, scaling, translation, and shear operations.
        The bottom row is always [0, 0, 1].

    Examples
    --------
    Create a transform that scales then translates:

    >>> import numpy as np
    >>> from neurospatial.ops.transforms import translate, scale_2d
    >>> transform = translate(10, 20) @ scale_2d(2.0)
    >>> points = np.array([[0, 0], [1, 1]])
    >>> transformed = transform(points)
    >>> transformed
    array([[10., 20.],
           [12., 22.]])

    """

    A: NDArray[np.float64]  # shape (3, 3)

    # ---- core --------------------------------------------------------
    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transformation to points.

        Parameters
        ----------
        pts : NDArray[np.float64], shape (..., 2)
            2D points to transform.

        Returns
        -------
        NDArray[np.float64], shape (..., 2)
            Transformed points.

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate
        >>> transform = translate(10, 20)
        >>> points = np.array([[0, 0], [1, 1]])
        >>> transform(points)
        array([[10., 20.],
               [11., 21.]])

        """
        pts = np.asanyarray(pts, dtype=float)
        pts_h = np.c_[pts.reshape(-1, 2), np.ones((pts.size // 2, 1))]
        out = pts_h @ self.A.T
        out = out[:, :2] / out[:, 2:3]
        return np.asarray(out.reshape(pts.shape), dtype=np.float64)

    # ---- helpers -----------------------------------------------------
    def inverse(self) -> Affine2D:
        """Compute the inverse transformation.

        Returns
        -------
        Affine2D
            New Affine2D representing the inverse transformation.

        Raises
        ------
        np.linalg.LinAlgError
            If transformation matrix is singular (non-invertible).

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate
        >>> transform = translate(10, 20)
        >>> inv = transform.inverse()
        >>> points = np.array([[10, 20]])
        >>> inv(points)
        array([[0., 0.]])

        """
        return Affine2D(np.asarray(np.linalg.inv(self.A), dtype=np.float64))

    def compose(self, other: Affine2D) -> Affine2D:
        """Compose this transformation with another.

        Parameters
        ----------
        other : Affine2D
            Transformation to compose with (applied first).

        Returns
        -------
        Affine2D
            New transformation representing ``self ∘ other``.

        Notes
        -----
        The resulting transformation applies `other` first, then `self`.
        Composition order matters: ``a.compose(b)`` ≠ ``b.compose(a)`` in general.

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate
        >>> t1 = translate(10, 0)
        >>> t2 = translate(0, 20)
        >>> combined = t1.compose(t2)
        >>> points = np.array([[0, 0]])
        >>> combined(points)
        array([[10., 20.]])

        """
        return Affine2D(self.A @ other.A)

    # Pythonic shorthand:  t3 = t1 @ t2
    def __matmul__(self, other: Affine2D) -> Affine2D:
        """Compose transformations using @ operator.

        Parameters
        ----------
        other : Affine2D
            Transformation to compose with.

        Returns
        -------
        Affine2D
            Composed transformation (``self @ other``).

        See Also
        --------
        compose : Functional composition interface.

        Examples
        --------
        >>> from neurospatial.ops.transforms import translate, scale_2d
        >>> t1 = translate(10, 0)
        >>> t2 = scale_2d(2.0)
        >>> combined = t1 @ t2
        >>> points = np.array([[0, 0], [1, 1]])
        >>> combined(points)
        array([[10.,  0.],
               [12.,  2.]])

        """
        return self.compose(other)


def identity() -> Affine2D:
    """Return the identity transform.

    Returns
    -------
    Affine2D
        Identity transformation (no change to input points).

    Examples
    --------
    >>> transform = identity()
    >>> points = np.array([[1, 2], [3, 4]])
    >>> transform(points)
    array([[1., 2.],
           [3., 4.]])

    """
    return Affine2D(np.eye(3))


# Factory helpers for the most common ops ---------------------------------
def scale_2d(sx: float = 1.0, sy: float | None = None) -> Affine2D:
    """Create uniform or anisotropic scaling transformation.

    Parameters
    ----------
    sx : float, default=1.0
        Scale factor for x-axis.
    sy : float or None, default=None
        Scale factor for y-axis. If None, uses `sx` for uniform scaling.

    Returns
    -------
    Affine2D
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
    return Affine2D(np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]))


def translate(tx: float = 0.0, ty: float = 0.0) -> Affine2D:
    """Create translation transformation.

    Parameters
    ----------
    tx : float, default=0.0
        Translation in x direction.
    ty : float, default=0.0
        Translation in y direction.

    Returns
    -------
    Affine2D
        Translation transformation.

    Examples
    --------
    >>> transform = translate(10, 20)
    >>> points = np.array([[0, 0], [1, 1]])
    >>> transform(points)
    array([[10., 20.],
           [11., 21.]])

    """
    return Affine2D(np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]))


def flip_y(frame_height_px: float) -> Affine2D:
    """Flip the *y*-axis of pixel coordinates so that origin moves
    from top-left to bottom-left.

    Parameters
    ----------
    frame_height_px : float
        Height of the video frame in pixels.

    Returns
    -------
    Affine2D
        Transformation that flips y-axis around frame center.

    """
    return Affine2D(
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, frame_height_px], [0.0, 0.0, 1.0]]),
    )


# --- Video Calibration Helpers ---------------------------------------
def calibrate_from_scale_bar(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    known_length_cm: float,
    frame_size_px: tuple[int, int],
) -> Affine2D:
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
    Affine2D
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
            f"HOW: Provide a positive value for known_length_cm."
        )

    # Compute pixel distance between endpoints
    dx = p2_px[0] - p1_px[0]
    dy = p2_px[1] - p1_px[1]
    px_distance = np.sqrt(dx * dx + dy * dy)

    if px_distance == 0:
        raise ValueError(
            f"WHAT: Scale bar has zero pixel length (p1={p1_px}, p2={p2_px}).\n"
            f"WHY: Cannot compute scale from coincident endpoints.\n"
            f"HOW: Provide two distinct points for the scale bar."
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
    kind: str = "similarity",
) -> Affine2D:
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
        - "rigid": rotation + translation (4 DOF)
        - "similarity": uniform scale + rotation + translation (5 DOF)
        - "affine": full affine (6 DOF)

    Returns
    -------
    Affine2D
        Transform that converts pixel coords to cm coords.

    Raises
    ------
    ValueError
        If fewer than 3 landmarks provided, or landmark arrays have different
        lengths, or transform estimation fails.

    Notes
    -----
    Pixel coordinates use (x_px, y_px) = (column, row) ordering in image space.
    The frame_size_px is used to apply Y-flip before transform estimation,
    converting from video coordinates (origin top-left) to standard Cartesian
    coordinates (origin bottom-left).

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
            f"HOW: Provide landmarks as shape (n_points, 2) array."
        )

    if landmarks_cm.ndim != 2 or landmarks_cm.shape[1] != 2:
        raise ValueError(
            f"WHAT: landmarks_cm must have shape (n_points, 2), got {landmarks_cm.shape}.\n"
            f"WHY: Each landmark needs (x, y) coordinates.\n"
            f"HOW: Provide landmarks as shape (n_points, 2) array."
        )

    n_px = len(landmarks_px)
    n_cm = len(landmarks_cm)

    if n_px != n_cm:
        raise ValueError(
            f"WHAT: landmarks_px ({n_px} points) and landmarks_cm ({n_cm} points) "
            f"must have the same number of points.\n"
            f"WHY: Each pixel landmark must have a corresponding cm landmark.\n"
            f"HOW: Ensure both arrays have the same length."
        )

    if n_px < 3:
        raise ValueError(
            f"WHAT: Need at least 3 landmarks for transform estimation (got {n_px}).\n"
            f"WHY: Affine transforms have at least 4 degrees of freedom.\n"
            f"HOW: Provide at least 3 non-collinear landmark pairs."
        )

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
                f"HOW: Use landmarks that span the full video frame with good spread. "
                f"Ensure at least 3 non-collinear points."
            )

    # Use estimate_transform from this module (returns AffineND, wrap in Affine2D)
    result = estimate_transform(landmarks_px_flipped, landmarks_cm, kind=kind)

    return Affine2D(result.A)


@dataclass
class VideoCalibration:
    """Container for video-to-environment coordinate calibration.

    Stores the pixel-to-cm transform along with video metadata for
    convenient access to both forward and inverse transforms.

    Parameters
    ----------
    transform_px_to_cm : Affine2D
        Transform that converts pixel coordinates to cm coordinates.
        Should include Y-axis flip (video origin top-left to env bottom-left).
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels.

    Attributes
    ----------
    transform_px_to_cm : Affine2D
        Forward transform from pixels to cm.
    frame_size_px : tuple[int, int]
        Video frame dimensions.
    transform_cm_to_px : Affine2D
        Inverse transform from cm to pixels (cached property).
    cm_per_px : float
        Approximate scale factor (computed from transform matrix).

    Notes
    -----
    The calibration can be serialized to/from dict for JSON storage.

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

    transform_px_to_cm: Affine2D
    frame_size_px: tuple[int, int]

    # Cached inverse transform
    _transform_cm_to_px: Affine2D | None = None

    @property
    def transform_cm_to_px(self) -> Affine2D:
        """Get the inverse transform (cm to pixels).

        Returns
        -------
        Affine2D
            Transform that converts cm coordinates to pixel coordinates.
        """
        if self._transform_cm_to_px is None:
            # Use object.__setattr__ to bypass frozen=True if using frozen dataclass
            object.__setattr__(
                self, "_transform_cm_to_px", self.transform_px_to_cm.inverse()
            )
        return self._transform_cm_to_px  # type: ignore[return-value]

    @property
    def cm_per_px(self) -> float:
        """Get approximate scale factor (cm per pixel).

        Returns the average of x and y scale factors from the transform matrix.

        Returns
        -------
        float
            Approximate cm per pixel scale factor.
        """
        # Extract scale from transform matrix diagonal
        # For uniform scaling, A[0,0] = A[1,1] = scale
        # For non-uniform, average the absolute values
        sx = float(abs(self.transform_px_to_cm.A[0, 0]))
        sy = float(abs(self.transform_px_to_cm.A[1, 1]))
        return (sx + sy) / 2.0

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
        transform = Affine2D(np.array(d["transform_px_to_cm"]))
        frame_size_list = d["frame_size_px"]
        frame_size = (int(frame_size_list[0]), int(frame_size_list[1]))
        return cls(transform_px_to_cm=transform, frame_size_px=frame_size)


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
        Rotation matrix (orthogonal matrix with det(rotation_matrix) = 1).
        For 3D, use scipy.spatial.transform.Rotation to generate.
    translation : NDArray[np.float64], shape (n_dims,), optional
        Translation vector. If None, no translation is applied.

    Returns
    -------
    AffineND
        Affine transformation combining rotation and optional translation.

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
    >>> transform(points)  # doctest: +SKIP
    array([[10., 21., 30.]])
    """
    rot = np.asarray(rotation_matrix, dtype=float)
    if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
        raise ValueError(f"Rotation matrix must be square, got shape {rot.shape}")

    n_dims = rot.shape[0]
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


def identity_nd(n_dims: int = 2) -> AffineND:
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


# Convenience aliases
Affine3D = AffineND  # Type alias for 3D transforms


# ---------------------------------------------------------------------
# Quick NumPy helpers that *internally* build and apply Affine2D
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

        Affine2D([[1, 0, 0], [0, -1, H], [0, 0, 1]])(data)

    but without the user having to build the transform.

    """
    transform = flip_y(frame_height_px=frame_size_px[1])
    return transform(np.asanyarray(data, dtype=float))


def convert_to_cm(
    data_px: NDArray[np.float64] | tuple | list,
    frame_size_px: tuple[float, float],
    cm_per_px: float = 1.0,
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
    cm_per_px : float, optional
        Conversion factor from pixels to centimeters (default is 1.0).

    Returns
    -------
    NDArray[np.float64]
        Converted coordinates in centimeters, shape (..., 2).

    """
    transform = scale_2d(cm_per_px) @ flip_y(frame_height_px=frame_size_px[1])
    return transform(np.asanyarray(data_px, dtype=float))


def convert_to_pixels(
    data_cm: NDArray[np.float64] | tuple | list,
    frame_size_px: tuple[float, float],
    cm_per_px: float = 1.0,
) -> NDArray[np.float64]:
    """Convert centimeter coordinates to pixel coordinates with y-flip.

    Parameters
    ----------
    data_cm : NDArray[np.float64] or tuple or list
        Input coordinates in centimeter space, shape (..., 2).
    frame_size_px : tuple[float, float]
        Size of the video frame in pixels (width, height).
    cm_per_px : float, default=1.0
        Conversion factor from pixels to centimeters.

    Returns
    -------
    NDArray[np.float64]
        Converted coordinates in pixel space, shape (..., 2).

    Notes
    -----
    Inverse of `convert_to_cm`. Internally constructs ``flip_y(H) @ scale_2d(1/cm_per_px)``.

    """
    transform = flip_y(frame_height_px=frame_size_px[1]) @ scale_2d(1.0 / cm_per_px)
    return transform(np.asanyarray(data_cm, dtype=float))


# ---------------------------------------------------------------------
# 3.  Transform estimation from point correspondences
# ---------------------------------------------------------------------
def estimate_transform(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
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
        - "affine": Full affine (rotation, scaling, shear, translation)

    Returns
    -------
    AffineND
        Estimated transformation that maps src → dst.
        For 2D points, returns 3×3 matrix. For 3D points, returns 4×4 matrix.

    Raises
    ------
    ValueError
        If insufficient points for the requested transformation type,
        or if points are degenerate (collinear, etc.),
        or if dimensionality mismatch between src and dst.

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
    Uses Procrustes analysis for rigid and similarity transforms, and
    least-squares for affine transforms. The Procrustes method works for
    any dimensionality.

    For cross-session alignment, collect 3-4 landmark points (e.g., corners
    of arena) in both sessions and use this function to compute the alignment.

    See Also
    --------
    AffineND : N-D affine transformation class
    Affine2D : 2D affine transformation (same as AffineND with n_dims=2)
    Affine3D : 3D affine transformation (same as AffineND with n_dims=3)
    apply_transform_to_environment : Apply transform to Environment

    """
    from scipy.linalg import orthogonal_procrustes

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

    n_points = src.shape[0]
    n_dims = src.shape[1]

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

        # Estimate rotation using Procrustes
        # Note: orthogonal_procrustes finds R such that ||src @ R - dst|| is minimized
        # But we want transformation T(x) = x @ R_transform^T
        # So R_transform = R^T
        R_proc, _ = orthogonal_procrustes(src_centered, dst_centered)
        R = R_proc.T

        # Ensure R is a proper rotation (det(R) = +1, not -1)
        # If det(R) < 0, we have a reflection; flip last axis to get rotation
        if np.linalg.det(R) < 0:
            # Flip the last column to convert reflection to rotation
            R[:, -1] = -R[:, -1]

        if kind == "rigid":
            # Rigid: rotation + translation
            # T(x) = R @ x + t
            # Solve for t: dst_mean = R @ src_mean + t
            t = dst_mean - R @ src_mean

            # Build homogeneous matrix
            A = np.eye(n_dims + 1)
            A[:n_dims, :n_dims] = R
            A[:n_dims, n_dims] = t

            return AffineND(A)

        else:  # similarity
            # Similarity: rotation + uniform scale + translation
            # Estimate scale: ratio of RMS distances from centroid
            src_rms = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
            dst_rms = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

            if src_rms < 1e-10:
                raise ValueError("Source points are degenerate (all at same location)")

            scale = dst_rms / src_rms

            # Build transformation: T(x) = scale * R @ x + t
            # where t = dst_mean - scale * R @ src_mean
            t = dst_mean - scale * (R @ src_mean)

            A = np.eye(n_dims + 1)
            A[:n_dims, :n_dims] = scale * R
            A[:n_dims, n_dims] = t

            return AffineND(A)

    elif kind == "affine":
        if n_points < n_dims + 1:
            raise ValueError(
                f"affine transform requires at least {n_dims + 1} point pairs "
                f"for {n_dims}D, got {n_points}"
            )

        # Solve affine transform using least squares
        # For each dimension i: T_i(x) = [a_i1, a_i2, ..., a_in, t_i] @ [x1, x2, ..., xn, 1]^T

        # Build design matrix: [x1, x2, ..., xn, 1] for each point
        X = np.c_[src, np.ones(n_points)]  # (N, n_dims+1)

        # Solve for all output dimensions at once using vectorized lstsq
        # X @ A_partial.T = dst, where A_partial is (n_dims, n_dims+1)
        # lstsq with multiple RHS: X @ result = dst, result shape (n_dims+1, n_dims)
        A = np.eye(n_dims + 1)
        result, _, _, _ = np.linalg.lstsq(X, dst, rcond=None)
        # result has shape (n_dims+1, n_dims), we need to transpose for A[dim, :]
        A[:n_dims, :] = result.T

        return AffineND(A)

    else:
        raise ValueError(
            f"Invalid kind: {kind!r}. Must be 'rigid', 'similarity', or 'affine'."
        )


def apply_transform_to_environment(
    env: Environment,
    transform: AffineND | Affine2D,
    *,
    name: str | None = None,
) -> Environment:
    """Apply N-D affine transformation to an Environment, returning a new instance.

    This function creates a new Environment with transformed bin_centers and
    updated connectivity graph. All other properties (regions, metadata) are
    copied from the source environment. Supports 2D, 3D, or higher-dimensional
    transformations.

    Parameters
    ----------
    env : Environment
        Source environment to transform (any dimensionality).
    transform : AffineND or Affine2D
        Transformation to apply. Must match environment dimensionality:
        - For 2D environments: use Affine2D or AffineND with n_dims=2
        - For 3D environments: use AffineND with n_dims=3
    name : str, optional
        Name for the new environment. If None, appends "_transformed" to original name.

    Returns
    -------
    Environment
        New Environment instance with transformed coordinates.

    Raises
    ------
    ValueError
        If transform dimensionality doesn't match environment dimensionality.
    RuntimeError
        If environment is not fitted.

    Examples
    --------
    2D transformation (backward compatible):

    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.ops.transforms import (  # doctest: +SKIP
    ...     estimate_transform,
    ...     apply_transform_to_environment,
    ... )
    >>> # Create environment from session 1
    >>> env1 = Environment.from_samples(data1, bin_size=2.0)  # doctest: +SKIP
    >>> # Estimate transform from landmarks
    >>> transform = estimate_transform(  # doctest: +SKIP
    ...     landmarks_session1, landmarks_session2, kind="rigid"
    ... )
    >>> # Transform environment to session 2 coordinates
    >>> env1_aligned = apply_transform_to_environment(  # doctest: +SKIP
    ...     env1, transform, name="session1_aligned"
    ... )

    3D transformation:

    >>> import numpy as np  # doctest: +SKIP
    >>> from scipy.spatial.transform import Rotation  # doctest: +SKIP
    >>> from neurospatial.ops.transforms import from_rotation_matrix  # doctest: +SKIP
    >>> # Create 3D environment
    >>> positions_3d = np.random.randn(1000, 3) * 20  # doctest: +SKIP
    >>> env_3d = Environment.from_samples(positions_3d, bin_size=5.0)  # doctest: +SKIP
    >>> # Apply 45-degree rotation around z-axis
    >>> R = Rotation.from_euler("z", 45, degrees=True).as_matrix()  # doctest: +SKIP
    >>> transform_3d = from_rotation_matrix(
    ...     R, translation=[10, 20, 30]
    ... )  # doctest: +SKIP
    >>> env_3d_rotated = apply_transform_to_environment(
    ...     env_3d, transform_3d
    ... )  # doctest: +SKIP

    See Also
    --------
    estimate_transform : Estimate transformation from point pairs
    Affine2D : 2D affine transformation class
    AffineND : N-dimensional affine transformation class
    from_rotation_matrix : Create transform from rotation matrix (scipy integration)

    Notes
    -----
    This function is pure: it does not modify the source environment.

    The transformation is applied to:
    - bin_centers
    - graph node 'pos' attributes
    - regions (points and polygons)

    Edge distances and vectors are recomputed after transformation.

    For 2D environments, edge 'angle_2d' attributes are also recomputed.

    """
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

    # Validate
    if not getattr(env, "_is_fitted", False):
        raise RuntimeError(
            "Environment must be fitted before applying transforms. "
            "Use a factory method like Environment.from_samples()."
        )

    # Validate dimensionality match
    transform_dims = transform.n_dims if isinstance(transform, AffineND) else 2
    if env.n_dims != transform_dims:
        raise ValueError(
            f"Transform dimensionality ({transform_dims}D) does not match "
            f"environment dimensionality ({env.n_dims}D)."
        )

    # Transform bin centers
    transformed_centers = transform(env.bin_centers)

    # Create new connectivity graph with updated node positions

    new_graph = env.connectivity.copy()
    for node_id in new_graph.nodes:
        old_pos = new_graph.nodes[node_id]["pos"]
        new_pos = transform(np.array([old_pos]))[0]
        new_graph.nodes[node_id]["pos"] = tuple(new_pos)

    # Recompute edge attributes (distance, vector, and angle_2d for 2D)
    for u, v in new_graph.edges:
        pos_u = np.array(new_graph.nodes[u]["pos"])
        pos_v = np.array(new_graph.nodes[v]["pos"])
        vec = pos_v - pos_u
        dist = float(np.linalg.norm(vec))

        new_graph.edges[u, v]["vector"] = tuple(vec)
        new_graph.edges[u, v]["distance"] = dist

        # Recompute angle_2d for 2D environments only
        if env.n_dims == 2 and "angle_2d" in new_graph.edges[u, v]:
            angle = float(np.arctan2(vec[1], vec[0]))
            new_graph.edges[u, v]["angle_2d"] = angle

    # Transform dimension_ranges (N-dimensional)
    transformed_dim_ranges = None
    if env.dimension_ranges is not None:
        # Generate all corner points of the N-dimensional bounding box
        # For N dims, we have 2^N corners (all combinations of low/high per dimension)
        n_dims = env.n_dims
        ranges = env.dimension_ranges

        # Generate corner points using itertools.product
        from itertools import product

        corner_indices = list(product([0, 1], repeat=n_dims))  # All combinations of 0/1
        corners = np.array(
            [
                [ranges[dim][idx] for dim, idx in enumerate(corner)]
                for corner in corner_indices
            ]
        )

        # Transform all corners
        transformed_corners = transform(corners)

        # New bounding box: min/max along each dimension
        transformed_dim_ranges = [
            (transformed_corners[:, dim].min(), transformed_corners[:, dim].max())
            for dim in range(n_dims)
        ]

    # Create new Environment using from_layout pattern
    # We'll create a minimal layout wrapper

    class TransformedLayout:
        """Minimal layout wrapper for transformed environment."""

        def __init__(
            self,
            centers: NDArray[np.float64],
            graph: nx.Graph,
            dim_ranges: list[tuple[float, float]] | None,
            original_layout: Any,
        ) -> None:
            self.bin_centers = centers
            self.connectivity = graph
            self.dimension_ranges = dim_ranges
            self.is_1d = original_layout.is_1d
            self._layout_type_tag = f"{original_layout._layout_type_tag}_transformed"
            self._build_params_used = {
                **getattr(original_layout, "_build_params_used", {}),
                "transformed": True,
            }

            # Copy grid attributes if present
            for attr in ("grid_edges", "grid_shape", "active_mask"):
                if hasattr(original_layout, attr):
                    setattr(self, attr, getattr(original_layout, attr))

        def build(self) -> None:
            pass  # Already built

        def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int64]:
            # Use KD-tree on transformed centers
            from scipy.spatial import cKDTree

            kdtree = cKDTree(self.bin_centers)
            _, indices = kdtree.query(points)
            return np.asarray(indices, dtype=np.int64)

        def bin_sizes(self) -> NDArray[np.float64]:
            # Approximate from nearest neighbors
            from scipy.spatial import cKDTree

            if len(self.bin_centers) < 2:
                return np.array([1.0] * len(self.bin_centers), dtype=np.float64)
            kdtree = cKDTree(self.bin_centers)
            dists, _ = kdtree.query(self.bin_centers, k=2)
            return np.asarray(dists[:, 1] ** 2, dtype=np.float64)  # Approximate area

        def plot(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "Plotting not implemented for transformed layouts"
            )

    transformed_layout = TransformedLayout(
        transformed_centers, new_graph, transformed_dim_ranges, env.layout
    )

    # Create new environment
    new_name = name if name is not None else f"{env.name}_transformed"
    # Cast to LayoutEngine since TransformedLayout satisfies the protocol structurally
    new_env = Environment(
        name=new_name, layout=cast("LayoutEngine", transformed_layout)
    )
    new_env._setup_from_layout()

    # Transform and copy regions
    if env.regions and len(env.regions) > 0:
        transformed_regions = []
        for region in env.regions.values():
            if region.kind == "point":
                # Transform point
                old_point = np.array(region.data)
                new_point = transform(old_point.reshape(1, -1))[0]
                new_region = Region(
                    name=region.name,
                    kind="point",
                    data=new_point,
                    metadata={**region.metadata, "transformed": True},
                )
            elif region.kind == "polygon":
                # Transform polygon
                import shapely.geometry as shp
                from shapely.geometry import Polygon

                # Type narrowing: region.data is a Polygon when kind == "polygon"
                if not isinstance(region.data, Polygon):
                    raise TypeError(
                        f"Region '{region.name}' has kind='polygon' but data is not a Polygon"
                    )
                old_coords = np.array(region.data.exterior.coords)
                new_coords = transform(old_coords)
                new_poly = shp.Polygon(new_coords)
                new_region = Region(
                    name=region.name,
                    kind="polygon",
                    data=new_poly,
                    metadata={**region.metadata, "transformed": True},
                )

            transformed_regions.append(new_region)

        new_env.regions = Regions(transformed_regions)

    # Copy units and frame
    if hasattr(env, "units"):
        new_env.units = env.units
    if hasattr(env, "frame"):
        new_env.frame = f"{env.frame}_transformed" if env.frame else "transformed"

    return new_env


# ---------------------------------------------------------------------
# 4.  Simple calibration helper (merged from calibration.py)
# ---------------------------------------------------------------------
def simple_scale(
    px_per_cm: float,
    offset_px: tuple[float, float] = (0.0, 0.0),
) -> Affine2D:
    """Create a simple Affine2D transform that converts pixel units to centimeters.

    This returns an Affine2D matrix which, when applied to [x_px, y_px, 1]^T,
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
    Affine2D
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
    return Affine2D(A)
