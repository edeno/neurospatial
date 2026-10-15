# Stage F validation record

## Parent repairs and F0

See [f0-contracts.md](f0-contracts.md) for reproductions, fixed parent commits,
scientific baseline, native face inputs and migration decisions. The full
matrix reruns remain pending; no platform-wide pass is implied by local results.

## F1 — Native face data and coordinate publication

- Fresh behavior capture: 4,072 probes; comparison to F0 is identical.
- Environment/layout/NWB environment tests: 3,477 passed.
- New owned coordinate/native recipe tests: 29 passed (all built-in families;
  polar 1/2/8 angular bins, open/circular, deepcopy/pickle/dictionary).
- Final extracted polar/shared formula gate: 1,150 passed.
- Source mypy: 246 files pass. Ruff/format and whitespace checked.
- Native faces retain closed numerical recipes, not builders or opaque callable
  providers. Formulas are shared with the native layout construction tools.
  Construction stays lazy; skew warnings remain attached to first numerical use.
- Polar circularity normalizes on publication; current shared graph/coordinate
  operations retain their previous values. The stored layout facade and public
  sibling classes are transitional until their later migration phases.
- Independent review: ownership/coordinate checks passed; final review recorded
  at the phase commit. No full scientific CPU/docs suite is claimed at this gate.

## F2 — Direct geometry consumers

- Migrated encoding/grid metrics, animation raster/index reads, simulation and
  shuffling to published indexing/components. Native hex logical indexing and
  framed-grid raster refusals remain distinct.
- Plot dispatch is now a component function; affine environment integration is
  separated from primitive transforms, retaining the public wrapper/signature.
- 4,072 baseline probes compare identically. Source/static-contract typing passes.
- CPU suite: 13,757 passed, 31 skipped; targeted consumer/rendering tests: 265
  passed. A subsequent policy-focused run passed 353 tests.
- Independent review exposed a segmentation migration regression. Five focused
  tests now preserve native hex axis-only labeling, explicit bins, legacy saved
  settings, old-state pickle normalization and from_layout resolved defaults.
  The resolved builder option is normalized once into owned state rather than
  inferred from coordinate differences or read from mutable provenance.
- After the final normalization repair all five regressions pass; the full CPU
  suite will run again at the final stage gate. GUI/GPU remain unrun.

## Additional parent CI repair

The completed #74 Windows/Python 3.10 job crashed its worker while rejecting
circular region metadata. The previous rejection relied on interpreter recursion
limits. Explicit ancestor-path detection now rejects cycles promptly while
permitting shared acyclic containers. Two direct thaw regressions failed first;
315 Python 3.10 region tests pass (five optional skips), and independent review
passed 199 Python 3.13 region/core tests. Commit `9d5ce567` was propagated and
pushed through all affected parents; the Windows rerun is pending.

## F3 — One public Environment

- Both factories return the identical runtime class. `coordinate_kind` exposes
  the owned Cartesian/polar meaning. The old polar class/export/create entry
  point is removed; no forwarding alias remains.
- Bound and unbound Cartesian operations and property access refuse polar data;
  factory-inferred generic types also reject these calls statically. Source and
  static contracts pass for Python 3.10 semantics (250 checked files).
- 522 focused geometry/storage/refusal checks pass. The broader CPU run passed
  13,778 tests and found four outdated expectations/documentation checks; these
  were migrated and their complete target modules pass (156 tests). The final
  CPU gate will repeat after bridge removal.
- The baseline retains every non-API probe: 68 removed old-polar-class API rows,
  one new coordinate_kind row, 368 refusal-text class-name changes, and two
  intended API entries (copy's Self return annotation and the polar factory's
  shared-class return/documentation). All numeric arrays, refusal types and
  remaining text match. Later documentation edits are reviewed separately.
- Public export snapshot now covers neurospatial.environment explicitly; four
  existing common exports are added to its coverage. No extra polar export.
- Independent review found no source correctness issues. The structural-input
  test and stale sibling-class documentation were migrated following review.
