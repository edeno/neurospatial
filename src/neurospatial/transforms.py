"""transforms.py - minimal 2-D coordinate transforms
=================================================

**Important:** This module is specifically for 2D transformations. For 3D environments,
use ``scipy.spatial.transform.Rotation`` or implement custom 3D transformation matrices.
See docs/dimensionality_support.md for details on 3D support status.

Two complementary APIs
----------------------
1.  *Composable objects* (`Affine2D`, `SpatialTransform`)
    Build a transform once, reuse everywhere, keep provenance.
2.  *Quick helpers* (`flip_y_data`, `convert_to_cm`, `convert_to_pixels`)
    One-liners for scripts that just need a NumPy array back.

All functions assume coordinates are shaped ``(..., 2)`` and are no-ops on
the x-axis unless you chain additional transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.layout.base import LayoutEngine


# ---------------------------------------------------------------------
# 1.  Composable transform objects
# ---------------------------------------------------------------------
@runtime_checkable
class SpatialTransform(Protocol):
    """Callable that maps an (N, 2) array of points → (N, 2) array."""

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]: ...


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
    >>> from neurospatial.transforms import translate, scale_2d
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
        >>> from neurospatial.transforms import translate
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
        >>> from neurospatial.transforms import translate
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
        >>> from neurospatial.transforms import translate
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
        >>> from neurospatial.transforms import translate, scale_2d
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
    T = scale_2d(cm_per_px) @ flip_y(frame_height_px=frame_size_px[1])
    return T(np.asanyarray(data_px, dtype=float))


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
    T = flip_y(frame_height_px=frame_size_px[1]) @ scale_2d(1.0 / cm_per_px)
    return T(np.asanyarray(data_cm, dtype=float))


# ---------------------------------------------------------------------
# 3.  Transform estimation from point correspondences
# ---------------------------------------------------------------------
def estimate_transform(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    kind: str = "rigid",
) -> Affine2D:
    """Estimate 2D transformation from point correspondences.

    Given pairs of corresponding points in source and destination coordinate
    systems, compute the best-fit transformation (rigid, similarity, or affine).

    Parameters
    ----------
    src : NDArray[np.float64], shape (N, 2)
        Source points (N >= 2 for rigid/similarity, N >= 3 for affine).
    dst : NDArray[np.float64], shape (N, 2)
        Destination points corresponding to src.
    kind : {"rigid", "similarity", "affine"}, default="rigid"
        Type of transformation to estimate:

        - "rigid": Rotation + translation (preserves distances and angles)
        - "similarity": Rotation + uniform scaling + translation
          (preserves angles, scales distances uniformly)
        - "affine": Full affine (rotation, scaling, shear, translation)

    Returns
    -------
    Affine2D
        Estimated transformation that maps src → dst.

    Raises
    ------
    ValueError
        If insufficient points for the requested transformation type,
        or if points are degenerate (collinear, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.transforms import estimate_transform
    >>> # Define corresponding points (e.g., landmarks in two sessions)
    >>> src_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    >>> # Rotated 45 degrees and translated
    >>> angle = np.pi / 4
    >>> dst_pts = src_pts @ [
    ...     [np.cos(angle), -np.sin(angle)],
    ...     [np.sin(angle), np.cos(angle)],
    ... ] + [5, 5]
    >>> T = estimate_transform(src_pts, dst_pts, kind="rigid")
    >>> transformed = T(src_pts)
    >>> np.allclose(transformed, dst_pts)
    True

    Notes
    -----
    Uses Procrustes analysis for rigid and similarity transforms, and
    least-squares for affine transforms.

    For cross-session alignment, collect 3-4 landmark points (e.g., corners
    of arena) in both sessions and use this function to compute the alignment.

    See Also
    --------
    Affine2D : 2D affine transformation class
    apply_transform_to_environment : Apply transform to Environment

    """
    from scipy.linalg import orthogonal_procrustes

    src = np.asanyarray(src, dtype=float)
    dst = np.asanyarray(dst, dtype=float)

    if src.shape != dst.shape:
        raise ValueError(
            f"src and dst must have same shape, got {src.shape} and {dst.shape}"
        )

    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            f"src and dst must be (N, 2) arrays for 2D transforms, got shape {src.shape}"
        )

    n_points = src.shape[0]

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
        # If det(R) < 0, we have a reflection; flip one axis to get rotation
        if np.linalg.det(R) < 0:
            # Flip the second column to convert reflection to rotation
            R[:, 1] = -R[:, 1]

        if kind == "rigid":
            # Rigid: rotation + translation
            # T(x) = R @ x + t
            # Solve for t: dst_mean = R @ src_mean + t
            t = dst_mean - R @ src_mean

            # Build homogeneous matrix
            A = np.eye(3)
            A[:2, :2] = R
            A[:2, 2] = t

            return Affine2D(A)

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

            A = np.eye(3)
            A[:2, :2] = scale * R
            A[:2, 2] = t

            return Affine2D(A)

    elif kind == "affine":
        if n_points < 3:
            raise ValueError(
                f"affine transform requires at least 3 point pairs, got {n_points}"
            )

        # Solve affine transform using least squares
        # T(x, y) = [a, b, tx] @ [x, y, 1]^T  for x-coordinate
        #           [c, d, ty] @ [x, y, 1]^T  for y-coordinate

        # Build design matrix: [x, y, 1] for each point
        X = np.c_[src, np.ones(n_points)]  # (N, 3)

        # Solve for each row of transformation matrix independently
        # For x: [a, b, tx] = argmin ||X @ [a, b, tx]^T - dst_x||^2
        # For y: [c, d, ty] = argmin ||X @ [c, d, ty]^T - dst_y||^2
        params_x = np.linalg.lstsq(X, dst[:, 0], rcond=None)[0]  # [a, b, tx]
        params_y = np.linalg.lstsq(X, dst[:, 1], rcond=None)[0]  # [c, d, ty]

        # Build homogeneous matrix
        A = np.eye(3)
        A[0, :] = params_x  # [a, b, tx]
        A[1, :] = params_y  # [c, d, ty]

        return Affine2D(A)

    else:
        raise ValueError(
            f"Invalid kind: {kind!r}. Must be 'rigid', 'similarity', or 'affine'."
        )


def apply_transform_to_environment(
    env: Environment,
    transform: Affine2D,
    *,
    name: str | None = None,
) -> Environment:
    """Apply 2D affine transformation to an Environment, returning a new instance.

    This function creates a new Environment with transformed bin_centers and
    updated connectivity graph. All other properties (regions, metadata) are
    copied from the source environment.

    Parameters
    ----------
    env : Environment
        Source environment to transform (must be 2D).
    transform : Affine2D
        Transformation to apply.
    name : str, optional
        Name for the new environment. If None, appends "_transformed" to original name.

    Returns
    -------
    Environment
        New Environment instance with transformed coordinates.

    Raises
    ------
    ValueError
        If environment is not 2D (transforms only support 2D currently).
    RuntimeError
        If environment is not fitted.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.transforms import (
    ...     estimate_transform,
    ...     apply_transform_to_environment,
    ... )
    >>> # Create environment from session 1
    >>> env1 = Environment.from_samples(data1, bin_size=2.0)
    >>> # Estimate transform from landmarks
    >>> T = estimate_transform(landmarks_session1, landmarks_session2, kind="rigid")
    >>> # Transform environment to session 2 coordinates
    >>> env1_aligned = apply_transform_to_environment(env1, T, name="session1_aligned")

    See Also
    --------
    estimate_transform : Estimate transformation from point pairs
    Affine2D : 2D affine transformation class

    Notes
    -----
    This function is pure: it does not modify the source environment.

    The transformation is applied to:
    - bin_centers
    - graph node 'pos' attributes
    - regions (points and polygons)

    Edge distances and vectors are recomputed after transformation.

    """
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

    # Validate
    if not getattr(env, "_is_fitted", False):
        raise RuntimeError(
            "Environment must be fitted before applying transforms. "
            "Use a factory method like Environment.from_samples()."
        )

    if env.n_dims != 2:
        raise ValueError(
            f"apply_transform_to_environment only supports 2D environments, "
            f"got {env.n_dims}D. For 3D, use scipy.spatial.transform.Rotation."
        )

    # Transform bin centers
    transformed_centers = transform(env.bin_centers)

    # Create new connectivity graph with updated node positions

    new_graph = env.connectivity.copy()
    for node_id in new_graph.nodes:
        old_pos = new_graph.nodes[node_id]["pos"]
        new_pos = transform(np.array([old_pos]))[0]
        new_graph.nodes[node_id]["pos"] = tuple(new_pos)

    # Recompute edge attributes (distance, vector, angle_2d)
    for u, v in new_graph.edges:
        pos_u = np.array(new_graph.nodes[u]["pos"])
        pos_v = np.array(new_graph.nodes[v]["pos"])
        vec = pos_v - pos_u
        dist = float(np.linalg.norm(vec))

        new_graph.edges[u, v]["vector"] = tuple(vec)
        new_graph.edges[u, v]["distance"] = dist

        # Recompute angle_2d if present
        if "angle_2d" in new_graph.edges[u, v]:
            angle = float(np.arctan2(vec[1], vec[0]))
            new_graph.edges[u, v]["angle_2d"] = angle

    # Transform dimension_ranges
    transformed_dim_ranges = None
    if env.dimension_ranges is not None:
        # Transform corner points
        lo_x, hi_x = env.dimension_ranges[0]
        lo_y, hi_y = env.dimension_ranges[1]
        corners = np.array([[lo_x, lo_y], [hi_x, lo_y], [hi_x, hi_y], [lo_x, hi_y]])
        transformed_corners = transform(corners)

        # New bounding box
        new_lo_x, new_hi_x = (
            transformed_corners[:, 0].min(),
            transformed_corners[:, 0].max(),
        )
        new_lo_y, new_hi_y = (
            transformed_corners[:, 1].min(),
            transformed_corners[:, 1].max(),
        )
        transformed_dim_ranges = [(new_lo_x, new_hi_x), (new_lo_y, new_hi_y)]

    # Create new Environment using from_layout pattern
    # We'll create a minimal layout wrapper

    class TransformedLayout:
        """Minimal layout wrapper for transformed environment."""

        def __init__(self, centers, graph, dim_ranges, original_layout):
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

        def build(self):
            pass  # Already built

        def point_to_bin_index(self, points):
            # Use KD-tree on transformed centers
            from scipy.spatial import cKDTree

            kdtree = cKDTree(self.bin_centers)
            _, indices = kdtree.query(points)
            return indices

        def bin_sizes(self):
            # Approximate from nearest neighbors
            from scipy.spatial import cKDTree

            if len(self.bin_centers) < 2:
                return np.array([1.0] * len(self.bin_centers))
            kdtree = cKDTree(self.bin_centers)
            dists, _ = kdtree.query(self.bin_centers, k=2)
            return dists[:, 1] ** 2  # Approximate area

        def plot(self, *args, **kwargs):
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
