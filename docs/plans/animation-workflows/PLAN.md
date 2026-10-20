# Animation workflows

Implement the animation review in an isolated worktree, retaining
`Environment.animate_fields` as the compatibility entry point.

## Data and ownership

- Observed overlays use the same timestamp, missing-data, `max_gap=0.5` and
  `break_before` conventions as behavior and decoding. Exact observations remain
  visible; interpolation cannot manufacture observations across a recording gap.
- A reusable `Animation` owns prepared metadata and readers it opens. Large field
  sources remain lazy and caller-owned. Named `AnimationField` streams carry
  their own clocks, support and color scales.
- Result adapters retain environment, physical units and unit identity.
  `OverlayProtocol` adapts inputs to the supported internal data types; it is
  not a renderer extension interface.
- Backend capabilities are explicit. Unsupported overlays warn in the legacy
  dispatcher and fail preflight in the scene workflow.

## Viewing and export

- `AnimationView` exposes recording-time `seek`, half-open `select_time`,
  `export`, `close`, and the native viewer.
- Notebook scene viewing and export share the same renderer, scales and panels.
  GPU views retain their native controls; exports preserve scene data and panels,
  rather than arbitrary native camera changes or added graphics.
- `EventInspector` retains separate identified event rows, even when intervals
  overlap. Metadata and annotations survive filtering, selection and table
  export. Selected event intervals drive the same scene export.
- Scene lifetime is explicit (`close` and context manager). Closing one view
  releases that viewer; the scene remains reusable until explicitly closed.

## Repairs and validation

Correct calibrated video coordinates after crop/downsampling and in napari's
physical scale, retain missing field colors, combine grouped trace limits,
preserve duplicate trace labels, flush the last widget selection, honor JPEG
encoding, and resolve external HTML frame paths relative to the output.

Test scientific invariants, lazy sources, identified-event round trips, and a
complete selected-event export. Run affected animation tests, API contracts,
lint and typing against this worktree's source. Record GUI limitations separately
from headless rendering results.

## Implemented

- Shared observed-interval sampling and gap-aware trails across renderers.
- Unit/environment adapters and calibrated crop/resize transformations.
- Reusable named scenes, native-clock sources, notebook/GPU view handles,
  complete MP4/GIF export and projected track panels.
- Identified event filtering, selection, annotations and typed table export.
- Backend repairs and explicit capability diagnostics.
- Worked examples, changelog, public API snapshot and regression tests.

The scene interface covers 1-D/2-D scenes and notebook/fastplotlib views.
Napari's existing direct interface remains available; its scene adapter is
explicitly refused until independent stream clocks can be preserved.

Validation includes pixel-identical notebook/export frames, a real MP4 checked
with ffprobe (10 fps, three frames, 0.3 seconds), Ruff and mypy across all 297
source files. GPU drawing is unavailable in this environment. Qt tests that
construct a real viewer require a functioning display; the unmarked playback
controller integration module and time-series dock class are excluded from the
headless run. Napari's affine, layer data and support calculations are tested
without constructing a viewer. Test imports were verified against this worktree.

Final animation, environment integration and public API run: **1,338 passed,
7 skipped, 174 deselected**. Executable animation documentation and the 33-test
NWB overlay suite also pass.

## Remaining architectural recommendations

- Extend the common scene/view interface to napari and 3-D scenes.
- Support independent field color scales in the fastplotlib scene adapter.
- Carry native viewer edits into export if exact native-view capture is needed;
  current exports reproduce the scene, not native camera or custom graphics.
- Move the video reader used by annotation into a shared video-I/O module, and
  move pure styling helpers out of the Matplotlib parallel-rendering module.
- Complete live GPU/Qt validation on a machine with the required display/GPU.

Legacy backend overlay coverage still differs. The capability checks make
omissions explicit; complete scene rendering is available through the new
notebook/export workflow. The direct HTML object-vector warning was corrected
in a follow-up audit to recommend supported viewing/export paths.
