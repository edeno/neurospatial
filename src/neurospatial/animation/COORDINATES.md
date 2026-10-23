# Video overlay coordinate contracts

Internal developer reference for calibration, preparation and rendering.
All matrices below use homogeneous column vectors; the rightmost transform
acts first.

## Pixel centers and physical units

A video frame is an array of shape `(height, width, 3)`. Array indices are
`(row, column)`, while calibration coordinates are `(column, row)`. Pixel
centers run from `(0, 0)` to `(width - 1, height - 1)`; the image's outer edges
are at `-0.5`, `width - 0.5` and `height - 0.5`. `frame_size_px` is
`(width, height)`.

`VideoCalibration.transform_px_to_cm`, defined in `neurospatial.ops.transforms`,
maps original video pixel centers into centimeters. It includes the fitted
translation, rotation, scaling and any reflection. Scale-bar calibration
composes `scale_2d(cm_per_px) @ flip_y_transform(frame_height)`: the pixel-space
reflection acts before scaling its translation into centimeters. Landmark
calibration fits the supplied correspondence; do not assume every calibration
has a particular reflection or an origin at an environment corner.

Environment coordinates use the environment's declared units and coordinate
frame. They are not necessarily centimeters or anchored at the bottom-left
corner. `VideoOverlay.convert_to_data` produces a complete
`VideoData.transform_to_env` in those units. Metric declarations such as `m`,
`cm` and `mm` are converted; absent units retain the historical centimeter
convention. Calibrated video rejects unknown units and pixel units.
`calibrate_video(..., landmarks_env=...)` accepts environment units and converts
them before fitting the canonical centimeter calibration. Scale-bar lengths
and `cm_per_px` remain in centimeters.

## Preparation owns crop, resize and unit conversion

The reader applies crop and downsampling before a renderer sees the image.
For crop origin `(crop_x, crop_y)`, cropped dimensions `(crop_w, crop_h)` and
processed dimensions `(width, height)`, define:

```text
sx = crop_w / width
sy = crop_h / height

P = [[sx,  0, crop_x + (sx - 1) / 2],
     [ 0, sy, crop_y + (sy - 1) / 2],
     [ 0,  0,                    1]]

T = U @ C @ P
```

`P` maps processed pixel centers to original pixel centers, matching the
resize rule `original = (processed + 0.5) * scale - 0.5` plus crop translation.
Use actual dimensions, since integer rounding can make `sx` and `sy` differ
from the requested downsample factor. Without a crop, use zero offsets and
original dimensions. `C` is `transform_px_to_cm`; `U` scales centimeters
to environment units, for example `0.01` for meters or `10` for millimeters.
The resulting `T` is `VideoData.transform_to_env.A`.

Backends consume this prepared transform. They must not reapply crop,
downsampling, centimeter conversion or a presumed camera reflection.

## Backend display coordinates

The display adapter accounts for its own array ordering and field geometry.
There is no universal napari pixel space or universal `imshow` origin.

### Matplotlib scenes (notebook widget and file exports)

Calibrated images use `origin="upper"` with the raw-pixel extent
`(-0.5, width - 0.5, height - 0.5, -0.5)` and the artist transform
`Affine2D(T) + ax.transData`. Together, this places array pixel `[row, col]`
at raw coordinate `(col, row)` before applying calibration. `origin="upper"`
does not introduce a second calibration reflection in this arrangement.
Axes limits and other overlays stay in environment coordinates.

The implementation is `_scene_rendering.SceneRenderer`.

### Fastplotlib

`overlays.VideoData.pixel_to_env` returns `T` directly for calibrated video;
napari uses the same matrix. The renderer embeds this 2-D affine in its native 4-D transform; depth
controls whether video lies above or below fields. Pixel centers and spatial
overlays retain environment coordinates.

### Napari scenes

`backends._napari_scene` uses physical environment coordinates with reversed
axis order: `(y, x)` in 2-D and `(z, y, x)` in 3-D. Regular field images are
transposed into napari's array order, and their affine includes grid cell
centers, cell sizes and any environment frame transform.

For 2-D video, let `S` swap the first two coordinates of a homogeneous vector.
The video layer's affine is `S @ T @ S`: the input swap converts napari
`(row, column)` to calibration `(column, row)`, and the output swap converts
environment `(x, y)` to native `(y, x)`. This path does not apply a fixed
vertical reflection.

## Uncalibrated fallback

Without calibration, renderers fit video to `env_bounds` and preparation warns
that alignment may be inaccurate. Scene Matplotlib, scene napari and
fastplotlib place the first row at the top; fastplotlib and scene napari use
half-pixel offsets so image edges coincide with the bounds. These fallback
conventions are not a calibrated coordinate contract. Supply a
calibration when physical registration or consistent orientation across all
entry points matters.

## Verification

Check asymmetric landmarks and known pixel centers, including a nonzero crop,
rounded resize dimensions, non-centimeter environment units and nonsquare grid
cells. Distinguish the last pixel center `(width - 1, height - 1)` from the
outer edge `(width - 0.5, height - 0.5)`. Verify rendered pixels as well as
matrices; an array transpose or image extent can defeat a correct affine.

Relevant implementation boundaries:

- `neurospatial.ops.transforms`: canonical calibration and affine composition.
- `neurospatial._video_io`: decoded frames, crop and resize geometry.
- `animation.calibration` and `animation.overlays`: environment-unit conversion
  and the prepared `VideoData` contract.
- `animation._scene_rendering`, `animation.backends._napari_scene` and
  `animation.backends.fastplotlib_backend`: display-specific array ordering
  and placement.
