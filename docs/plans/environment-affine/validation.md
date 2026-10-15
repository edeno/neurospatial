# Stage E validation record

## E0 — Contracts and baseline

Recorded on 2026-10-15 before feature implementation. The parent documentation
example repair changes no library calculations.

- `scripts/environment_baseline.py record`: **4,072 probes**, including 584
  expected refusals, at `/private/tmp/environment-stage-e-base.npz`.
- `scripts/environment_affine_baseline.py`: deterministic native/rotation/
  reflection/shear IDs, measures, interpolation, smoothing and renderers at
  `/private/tmp/environment-affine-e0.json`.
- Native affine-linear interpolation returns **7.75**. Rotation and reflection
  retain identical smoothing arrays but refuse linear interpolation and use a
  scatter renderer. Shear retains lookup and measures but refuses faces.
- Probe elapsed time: **0.245267 seconds**; process peak RSS:
  **195117056 bytes** on this macOS runner. These observations include
  plotting and are not a performance claim.
- Independent E0 review completed. Contracts are frozen in `PLAN.md`.

The parent CI documentation example used unseeded position samples, so a point
at (5, 5) could fall outside the inferred active cells. Replacing its input with
an explicit full grid passes all **3 region doctests** locally. This fix is
propagated through the existing PR stack. Remote reruns are pending; no remote
matrix pass is claimed.

## E1 — Face-normal data

Completed and independently reviewed on 2026-10-15. Public `FaceNormals`
records own their arrays and preserve pair orientation through native capture,
subsetting, clipping, physical-wall edge removal and affine transforms. Polar
records refuse Cartesian normal components. Native face construction and mesh
lookup memos remain lazy. Codec version 2 stores the optional component; legacy
version 1 loads without it. No new numerical transform capabilities are enabled.

The independent reviewer found wall-edge row filtering and large-offset hexagon
rounding regressions during implementation. Both were reproduced and repaired;
reviewer reruns passed at offsets through 1e9, including mesh rotation/subsetting
and storage. Final independent signoff passed all 58 then-existing normal cases
with RuntimeWarnings treated as errors. A final Cartesian-coordinate rejection
case was added afterward and passes in the 126-test local strict gate. The
reviewer also independently checked that final case and all codec contracts
(**85 passed**), with no remaining confirmed blockers.

| Check | Result |
| --- | --- |
| Relevant layout/environment/I/O/NWB/transform/wall suite | 4,471 passed, 7 skipped |
| Full CPU suite | 13,612 passed, 31 skipped |
| Final normals and Stage D wall/arena regressions, RuntimeWarnings as errors | 126 passed |
| Normal and codec contracts | 143 passed |
| Source doctests | 596 passed, 126 skipped |
| Source plus five typing contracts | 246 files passed |
| Public API snapshot and argument conventions | 4 passed |
| Repository Ruff and formatting | Passed |
| Empty face-array NWB probes | Two isolated cells in 2-D and 3-D passed |
| Baseline numerical/behavior comparison | 4,070 probes exactly unchanged; only two parent-example docstrings differ |
| Affine baseline cases | IDs, measures, smoothing, interpolation and renderer types exactly unchanged |

The full CPU run collected 56 normal cases; the three final malformed-input,
zero-area/refusal and coordinate-kind cases were checked afterward in the final
strict gate. The wider suite uses the existing CPU runner exclusions (51 GUI
nodes plus slow, napari, gui and wallclock markers). No GUI/GPU run is claimed.
A broad exploratory run with every RuntimeWarning promoted failed the existing
all-NaN plot cases; the final strict gate targets changed geometry and wall code.
No cached real recording exists under `data/`; E1 uses deterministic simulated
geometry. The real tracking/image-mask workflow gate remains part of E5.

`MissingFaces` retains no area authority. Zero normal rows may conservatively
suppress geometric contacts as navigation-only edges; validation can check
nonzero directions but cannot reconstruct discarded activity. Such normals
never grant numerical capability. E3 must continue to honor that restriction.

Logs and captures:

- `/private/tmp/stage-e-e1-cpu.log`
- `/private/tmp/stage-e-e1-strict-gate-final.log`
- `/private/tmp/stage-e-e1-doctests.log`
- `/private/tmp/stage-e-e1-mypy-final-commit.log`
- `/private/tmp/stage-e-e1-compare.log`
- `/private/tmp/environment-stage-e-e1-final.npz`

The parent documentation example also needed formatter corrections after its
merge through the immutable-region API. Those corrections are published across
all existing PRs and formatted locally. Remote reruns are pending; neither that
matrix nor an unpublished Stage E PR is described as passing.

## E2 — Retained current grids and exact cells

Completed with independent review. FramedGrid owns source axes/mask and a
composed world frame; Binning codec 3 accepts legacy versions 1 and 2. Loading
checks source order, masks, axes, frame/inverse and locator consistency; source
axes never appear as world grid_edges. Partial clipping drops the tensor grid.
Reflected/deformed hexagons become exact PolygonCells in the same bin order.
Explicit-bin bounds include transformed corners; inferred mask domains assert
no physical walls.

- 928 geometry/derivation/frame/normal/codec/file/NWB tests passed under numerical
  RuntimeWarning errors (`/private/tmp/stage-e-e2-final.log`).
- Source typing: 242 files passed.
- Independent review: 119 focused tests plus seeded nested transform/subset/
  restoration probes at offsets through 1e9 passed. Corrupted coefficients were
  rejected. No confirmed blockers.
- Grid interpolation, allocation and rendering remain for E4; no new numerical
  face capability is claimed by E2.

## E3 — Certified transformed faces

Area-vector transforms use scaled normal directions and logarithmic area
stretch. New numerical support requires source and transformed orthogonality
within 1e-12 and a source flux distance matching the geometric center link.
Retained source coordinates avoid large-origin cancellation, including combined
components. Similarity/native diagonal formulas remain unchanged. Missing faces
never regain authority through a later inverse transform.

- The strict geometry/derivation suite passed 537 tests before the final
  bridge-normal regression was added. The final affine suite passed 22 tests.
- Source typing passed for 243 files; scoped Ruff checks passed.
- Independent review confirmed the fixes for inconsistent combined normal
  duplicates, large-origin component vectors and explicit bridge-zero normals.
  Identity, similarity, subset and dict restoration probes passed.
- Seeded independent 2-D tangent-length and 3-D cross-product oracles cover
  100 maps per dimension; conservation, constants, zero net divergence and all
  four skew-cell numerical refusals are tested.

Logs: `/private/tmp/stage-e-e3-final.log` (contains the corrected test-oracle
failure for accessing unavailable bridge faces),
`/private/tmp/stage-e-e3-bridge-final.log`, `/private/tmp/stage-e-e3-mypy.log`.

## E4 — Source-grid consumers

Linear interpolation, movement allocation and planar QuadMesh rendering now
retain source axes. Nearest interpolation still uses world Euclidean distances.
Native paths retain their arithmetic. Framed consumers share bounded recovery
of source center/edge planes: a forward-residual bound with inverse sensitivity,
capped at a quarter of the smallest source cell width per axis. Unresolved
precision refuses; meaningful missing-cell weights remain missing.

Independent review reproduced two roundoff defects during implementation:
interpolation on source center planes returned NaN, and boundary-following
trajectories could fill an inactive neighbor after a world lookup round trip.
Both have regression tests at arbitrary rotation angles and offsets through
1e9. Canonical source ownership now governs linear dwell and spike counts;
native N-D start gates and count normalization remain compatible.

- 116 strict consumer/frame/face/physical-wall tests passed.
- Broader environment/layout/NWB/animation tests: 3,981 passed, with six old
  tests failing because they used rotated grids to exercise scatter rendering.
  Those fixtures now use explicit point-only bins; all six pass. Exact
  transformed-grid rendering has its own corner/value/mask tests.
- Source and five typing contracts: 248 files passed. Ruff checks passed.
- Reviewer reran both original reproductions, 70 strict consumer/wall cases and
  seeded nested 3-D transform/subset comparisons. Errors were about 2e-15 at
  zero origin and at most 6.3e-7 at offset 1e9, consistent with stored coordinate
  rounding; no filled support remained.
- Matplotlib/video/HTML and widget QuadMesh updates use source shape/mask.
  Napari raster helpers explicitly refuse framed grids. FFT/grid raster metrics
  retain their existing restrictions; 2-D circular grid shuffling now explains
  its world-aligned-grid requirement instead of indexing a flattened shape.

Logs: `/private/tmp/stage-e-e4-roundoff-final.log`,
`/private/tmp/stage-e-e4-source-owner.log`, `/private/tmp/stage-e-e4-wide-final.log`,
`/private/tmp/stage-e-e4-scatter.log`, `/private/tmp/stage-e-final-typing.log`.
