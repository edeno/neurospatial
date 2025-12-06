"""Backward-compatibility shim for neurospatial.transforms.

.. deprecated:: 0.4.0
    Use ``from neurospatial.ops.transforms import ...`` instead.
    This module re-exports all public symbols from :mod:`neurospatial.ops.transforms`
    for backward compatibility.

The canonical location is now ``neurospatial.ops.transforms``.
"""

# Re-export everything from the new location

from neurospatial.ops.transforms import (
    Affine2D,
    Affine3D,
    AffineND,
    SpatialTransform,
    VideoCalibration,
    apply_transform_to_environment,
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
    convert_to_cm,
    convert_to_pixels,
    estimate_transform,
    flip_y,
    flip_y_data,
    from_rotation_matrix,
    identity,
    identity_nd,
    scale_2d,
    scale_3d,
    simple_scale,
    translate,
    translate_3d,
)

__all__ = [
    "Affine2D",
    "Affine3D",
    "AffineND",
    "SpatialTransform",
    "VideoCalibration",
    "apply_transform_to_environment",
    "calibrate_from_landmarks",
    "calibrate_from_scale_bar",
    "convert_to_cm",
    "convert_to_pixels",
    "estimate_transform",
    "flip_y",
    "flip_y_data",
    "from_rotation_matrix",
    "identity",
    "identity_nd",
    "scale_2d",
    "scale_3d",
    "simple_scale",
    "translate",
    "translate_3d",
]
