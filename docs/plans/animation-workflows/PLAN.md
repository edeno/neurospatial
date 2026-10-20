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
- Notebook, fastplotlib and napari views preserve independent stream clocks and
  scales. View exports capture the current scientific canvas and panels,
  including native camera edits and custom graphics. `render="scene"` selects
  the standard scene layout. Both paths use the same physical export clock and
  atomic writer, restoring view time and playback even after encoding failures.
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
- Reusable named scenes, native-clock sources, notebook/GPU/napari view handles,
  complete MP4/GIF export and projected track panels.
- Three-dimensional scene rendering with physical cells or explicit bin-center
  fallback, native napari volumes, and preserved spatial overlay coordinates.
- Independent per-field scales and labels in spatial and projected-track panels.
- Neutral video I/O for annotation and animation, with legacy import/pickle
  compatibility; shared styling and track geometry without renderer dependencies.
- Identified event filtering, selection, annotations and typed table export.
- Backend repairs and explicit capability diagnostics.
- Worked examples, changelog, public API snapshot and regression tests.

The scene interface covers 1-D/2-D/3-D scenes and notebook/fastplotlib views,
plus 2-D/3-D napari views. Native capture includes custom graphics and camera
edits, with scientific dock panels included when visible.

Initial validation included pixel-identical notebook/export frames, a real MP4
checked with ffprobe (10 fps, three frames, 0.3 seconds), Ruff and mypy across
all 297 source files. Test imports were verified against this worktree.

Initial animation, environment integration and public API run: **1,338 passed,
7 skipped, 174 deselected**. Executable animation documentation and the 33-test
NWB overlay suite also pass.

## Architectural follow-up

The five remaining recommendations are implemented: napari/3-D scene adapters,
independent fastplotlib scales, native-view capture, shared dependency
extraction, and live GPU/Qt validation. Native tests use the Apple Metal adapter
and Cocoa Qt/OpenGL outside the sandbox; offscreen Qt does not provide an
OpenGL context on this machine.

Follow-up validation:

- Broad animation, environment integration and API suite: **1,340 passed,
  6 skipped, 176 deselected**. Final scene/view/API checks after integration:
  **57 passed, 3 deselected**.
- Actual Metal rendering/capture and worker lifecycle suite: **67 passed**.
  A 3-D view exports a real GIF with independent stream clocks, native camera
  edits and custom graphics; decoded pixels match the captured canvas.
- Actual Cocoa napari suite: **136 passed**; playback/dock integration and
  window-close cleanup: **19 passed**.
- Shared video/annotation, styling and projection checks pass, including
  fresh-process import boundaries and cropped/prefetched reader pickles.
- Executable animation documentation, Ruff/formatting, and mypy across all
  **302 source files** pass.

Export frames that are within floating-point roundoff of an observation are
snapped to that stream's exact clock. This prevents `100.1 + 0.1` from holding
the sample before `100.2`; genuinely intermediate times and half-open stops
retain their meaning.

Explicit limits remain where the data does not define a spatial embedding:
3-D scenes reject planar video, polygon regions, scale bars and angle-only
headings; point regions retain all three coordinates.
Napari scenes require 2-D/3-D coordinates, reject polygon regions with holes,
and refuse grid layout with overlays because napari separates individual
layers. These restrictions are documented with supported alternatives.

Legacy backend overlay coverage still differs. The capability checks make
omissions explicit; complete scene rendering is available through the new
notebook/export workflow. The direct HTML object-vector warning was corrected
in a follow-up audit to recommend supported viewing/export paths.

## Data-contract review

The six findings from the architecture and data-contract review are addressed:

- Fastplotlib track panels consume each field's native timestamps and valid
  intervals. Independent bounded caches retain overlapping rows without
  painting across breaks or discarding samples on a coarser scene clock.
- Trajectory-localized events use shared observed support, including missing
  samples, `max_gap` and `break_before`. Events without supported tracking keep
  their timestamps and missing locations.
- Track projection requires Cartesian coordinates and matching declared units
  and frames. Both unspecified declarations retain the caller's responsibility
  for correspondence; one-sided or contradictory metadata fails explicitly.
- Overlay preparation acquires video readers once and cleans them up
  transactionally on failure. Requested display clocks reuse prepared data and
  borrow the scene's readers without transferring ownership.
- `AnimationField` metadata is frozen and independently snapshotted by a scene.
  Copies and serialization retain read-only clocks and support; field sources
  remain lazy and caller-owned.
- Views consume a typed backend interface for recording time, seeking, capture
  and cleanup. Scene preflight checks dimensions, overlay coverage, layout and
  options before native viewer allocation. Unsupported 1-D regions and
  fastplotlib scale bars fail explicitly; supplied napari viewers honor
  requested title, dimensions and visibility.

Regression coverage includes native GPU track support and lazy window reads,
event locations beside gaps and exact observations, conversion failures,
borrowed reader lifetime, frozen metadata and early capability failures.

Final review validation:

- Broad animation, environment integration and API suite: **1,416 passed,
  6 skipped, 181 deselected**.
- Metal rendering, capture, native field support and missing-event tests:
  **82 passed**.
- Cocoa napari tests: **140 passed**, including missing-event locations and
  supplied-viewer options; playback/dock integration: **18 passed**.
- NWB overlay integration and executable animation documentation: **34 passed**.
- Ruff and formatting pass; mypy passes across all **303 source files**.

## Prepared-data boundary follow-up

The next review identified six additional contract gaps. The follow-up closes
them at preparation and sampling boundaries:

- Preserve mixed raster labels through the canonical identity resolver, so
  integer and string unit IDs cannot select one another's rate maps.
- Retain decoding `spike_window` as immutable half-open field epochs. User
  epochs intersect known recording support; widened requests cannot fill gaps.
- Convert canonical centimeter video calibration into environment metric
  units, including calibration fitting and coverage diagnostics.
- Preserve custom video source-index mappings and convert each adapter once.
  Renderers use the prepared mapping through `VideoData.get_frame`.
- Draw unambiguous 1-D spatial overlays on the zero-amplitude baseline and
  refuse unsupported headings and video before allocation.
- Accept frozen custom adapters without rewriting their attributes. Snapshot
  prepared data metadata, with explicit ownership of video resources.

Shared sampling, field statistics and scientific-panel preparation live outside
the Matplotlib scene renderer. Notebook/export and napari consume this prepared
contract; napari no longer allocates an Agg scene renderer to interpret data.

Native track caches and projected scales also honor exact sample support,
including recordings whose supported samples fall between coarse scene clock
entries. Missing custom video mappings hide the corresponding layer instead of
leaving stale pixels visible.

Follow-up validation:

- Broad animation, environment integration and API suite: **1,468 passed,
  6 skipped, 181 deselected**. Final focused support/metadata checks:
  **19 passed**; prepared-data, model-layer and sampling checks: **26 passed**.
- Actual Metal rendering, capture, field support and missing-event suite:
  **83 passed**.
- Actual Cocoa napari suite: **140 passed**; playback/dock integration:
  **18 passed**.
- NWB overlay integration and all executable documentation: **64 passed**.
- Ruff and formatting pass; mypy passes across all **305 source files**.
- Fresh-process preparation, including track projection, imports neither
  `matplotlib.pyplot` nor an Agg canvas.

## Sampling and lifetime review

The remaining five findings are addressed at shared boundaries:

- Prepared spatial overlays resolve inherited `frame_times` explicitly.
  Fastplotlib and scene consumers share one sampler for interpolation, exact
  sample masks and interval support, including custom adapters.
- Scenes retain their views and close playback, workers and event-inspection
  controls before releasing readers. Closing one view unregisters it and keeps
  the scene reusable; caller-supplied napari viewers retain unrelated content.
- A separate reader owner handles legacy dispatch failures, completed exports
  and interactive closure. Pure validation precedes acquisition; consumers
  borrow readers, and cleanup attempts every resource while preserving primary
  errors. Native construction failures also drain acquired consumers.
- Constant signal normalization preserves missing NaNs. Prepared signal values
  are final and are not normalized again by individual renderers.
- Object vectors prepare one finite recording-wide rate peak and use a shared
  bounded opacity calculation. Resampling and selected windows retain that
  scale without rescanning the recording on each frame.

The coordinate guide now documents original and processed pixel centers,
canonical centimeter calibration, environment units and backend-specific
display transforms, including the separate legacy napari raster convention.

Validation:

- Broad animation, environment integration, public API and argument contracts:
  **1,556 passed, 6 skipped, 186 deselected**.
- Actual Metal rendering, capture, observation/field support and lifecycle:
  **99 passed**, including native canvas closure and failed construction after
  worker allocation.
- Actual Cocoa napari suite: **145 passed**; playback/dock integration:
  **18 passed**. A live supplied viewer retains its caller layers and controls
  while the closed scene, view and controller can be garbage-collected.
- NWB overlays and all executable documentation: **64 passed**.
- Ruff and formatting pass; mypy passes across all **306 source files**.
