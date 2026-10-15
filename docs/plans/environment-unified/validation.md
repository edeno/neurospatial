# Stage F validation record

F0–F5 are complete and independently reviewed. The final local gates, exact
scientific comparisons, real-recording checks and measured costs are below.

## Parent repairs and F0

See [f0-contracts.md](f0-contracts.md) for reproductions, fixed parent commits,
scientific baseline, native face inputs and migration decisions. The full
matrix reruns were pending at this checkpoint; see the follow-up review below
for the latest parent CI results. No platform-wide pass is implied by local
results.

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

## F4a — Remove mixin hosts

- The common dataclass now owns its stable forwarding methods; domain algorithms
  live in private module functions. No mixin classes, _BaseEnvironment or broad
  EnvironmentProtocol host remains. Domain consumers expose their actual reads.
- Source plus static consumer contracts: 250 files pass.
- Environment/layout/NWB checks: 3,593 passed; one obsolete-host collection import
  was then migrated. The complete repair target modules pass 75 tests; five
  narrow-signature/warning checks also pass.
- Numeric/refusal probes match the F3 capture. Six incidental docstring import
  edits were reverted; all 85 public method/property signatures and docstrings
  now compare identically against that capture.
- Independent review checked extraction/cache parity and found one scale-bar
  warning-location regression. Two tests failed first, and the warning helper
  now computes stack depth at the actual warning site. Re-review passed all ten
  focused repair checks with no remaining findings.
- The layout bridge is the remaining F4 work; this checkpoint does not claim
  final Stage F completion or a final full-suite gate.

## F4b — Direct owned publication

- Removed the public layout property, layout-bearing constructor, BinningLayout,
  _OwnedLayout, LayoutLocator and LayoutFaces. Builders are consumed once by a
  construction adapter; derived values and readers publish owned binning data.
  Shared geometric consistency checks remain in layout/_validation.py.
- Environment/layout/file/NWB gate: 3,943 passed, seven skipped, four regressions
  identified and repaired. Independent review reran all nine focused checks:
  coincident native track centers, lazy restored mesh lookup, normalized default
  segmentation with a supplied arena, and actionable saved-graph errors pass.
  Stored/explicit bin-center validation remains strict; it probes a detached
  locator so published query structures stay lazy.
- Source and all static consumer contracts pass (254 files). Public export
  snapshot passes. All 4,004 probes match the first F4b capture; versus F4a the
  only additional API removal is Environment.layout. Scientific probes remain.
- Tests now exercise owned publication and pure validation instead of the
  deleted wrapper. Historical layout-bearing pickles refuse with migration to
  supported dict/file/NWB storage; current copy/deepcopy/pickle remain covered.
- Full CPU validation is running for the final F5 gate. Final documentation,
  dependency contracts and performance comparison remain before completion.

## F5 — Final architecture and workflow gates

The final common class has direct geometry publication, thin domain entry points
and no stored layout interface. `.importlinter` enforces two dependency contracts:
geometry data cannot depend on the public environment, native builders, encoding,
UI or optional storage libraries; numerical operations cannot depend on concrete
environment APIs or builders. Type-only imports are excluded. The only ignored
edge is `ops.transforms -> environment._affine`, the existing public convenience
function's lazy construction adapter. No entire numerical package is exempted.
`import-linter` is dev-only. The Python 3.13 CI leg runs the architecture gate and
all static contract files. No runtime dependency or scientific wire schema changed.

The migration guide is [environment-unified.md](../../migration/environment-unified.md).
Its examples execute in the documentation suite. Current architecture and NWB
contributor instructions describe direct publication and owned plugin snapshots.
The final source AST audit finds zero `.layout` or `._layout` attributes. Whole
state ownership tests cover native families after copy, deepcopy, pickle and
supported dictionary restoration; no builder or delegated callback remains.

### Final local results

Same isolated Python 3.13.5 `.venv`, dev/NWB/xarray/animation dependencies and
`uv run --no-sync` as earlier phases. Commands use `MPLBACKEND=Agg`,
`MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache` and
`UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache`. CPU workers and standalone
scientific examples limit BLAS/OpenMP/NumExpr threads to one.

| Gate | Result |
| --- | --- |
| Full CPU suite | 13,837 passed, 31 skipped |
| Geometry, frames, walls, explicit/derived/composed bins and file/NWB, with RuntimeWarnings as errors | 1,086 passed |
| Slow-inclusive plotting and saved geometry | 540 passed |
| Source doctests, serial | 585 passed, 124 skipped |
| Executable Markdown, API snapshot, argument conventions and import checks | 68 passed, three skipped |
| Python 3.10 coordinate/ownership/warning/region/polygon checks | 157 passed |
| Source and every static consumer contract | 254 files passed |
| Import-linter | Two contracts kept, none broken |
| Repository Ruff, formatting and diff whitespace | Passed |

The CPU command remains `/private/tmp/run_immutable_cpu.py`: four workers,
`not slow and not napari and not gui and not wallclock`, napari plugin disabled,
and the existing 51 GUI-node exclusions. GUI/GPU execution is not claimed.
The doctest command uses `--doctest-modules src/neurospatial -n 0`, excluding
annotation's two Qt widgets and animation/config.py, as in CI. Final raw logs
are `/private/tmp/stage-f-f5-cpu.log`, `f5-strict.log`,
`f5-slow-plot-storage.log`, `f5-doctests-final.log`, `f5-doc-api-final.log`,
`f5-py310.log`, `f5-mypy.log` and `f5-imports-final.log`.

The earlier full run passed 13,784 tests and found seven remaining tests using
the removed constructor; those now test initialization guards and public arena
adapters through the current paths. A source doctest also caught two stale
layout-property examples. These were repaired before the passing final gates.
The first documentation/API command named a nonexistent test file and ran zero
tests; only its corrected 68-test rerun is counted above.

### Scientific and API comparison

The final capture `/private/tmp/environment-stage-f-final.npz` has 4,004 probes
versus 4,072 at F0. All scientific arrays, refusal types and remaining behavior
match exactly. The entire diff is:

- 68 old-polar-class API rows and Environment.layout removed; coordinate_kind
  added. No polar numerical cases were removed.
- 368 refusal strings replace the removed polar class name with Environment.
- Seven common API rows document reviewed changes: copy returns Self;
  from_polar_egocentric returns the common coordinate-typed class; bin_sizes,
  boundary_bins, rebin, with_metadata and with_regions describe owned geometry
  or the common class and use current examples.

The audit `/private/tmp/f5-baseline-audit.json` reports no numeric differences
and no other non-API text differences. Direct layout-bearing construction and
old polar imports are removed intentionally. Historical layout/class-bearing
pickle paths are outside the F0 migration contract; supported dict/file/NWB
readers and current copies/pickles remain covered.

### Real recording

The pinned J16 version-1.0 recording was fetched with the repository loader into
`/private/tmp/neurospatial-stage-f-recording`, then verified against SHA-256
`60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3`.
This removes the earlier cached-recording validation limit. No data was added
to the repository and no existing user recording was replaced.

`/private/tmp/stage_f_real_recording.py` follows Tutorial 04: build the archived
track with its edge order/spacing and 2 cm bins; compute rates for the first
eight units with bandwidth 8 cm, minimum occupancy 0.5 s, max_gap 0.5 s and the
explicit recording window; also construct a 5 cm environment from observed
positions. Compare against repaired parent `c074957e` using its source through
PYTHONPATH and the identical interpreter/data/parameters.

The first 1,800 samples (3.598 s) passed, then the first 60,000 samples
(119.998 s, 7,677 input spikes) passed. Each comparison checks 14 arrays exactly:
bins, pairs, measures, lookup, track coordinates, occupancy, population firing
rates/support, restoration results, and position-inferred geometry/lookup/time.
The expanded case has 248 track bins and 256 inferred bins. This establishes
parity on this recording and these slices; it is not a new biological validation
of the estimators. Raw arrays/logs use `stage-f-real-{small-,}{parent,final}.npz`
and `f5-real-{expanded-,}{parent,final}.log` under `/private/tmp`.

### Performance and cost

Repeated the F0 script `/private/tmp/stage_f_benchmark.py`, same interpreter,
fixed seed 540, three repeats, 20x20 grid, 525-bin hex grid, 400-bin polar grid,
one million identical query points and thread limits of one. Raw results:
`stage-f-performance-base.json` and `stage-f-performance-final.json` under
`/private/tmp`. Values below are medians, with tracemalloc enabled in both runs.

| Family | Construction, base → final (s) | First faces (s) | First smoothing (s) | 1M lookup (s) | Peak, base → final (MB) |
| --- | --- | --- | --- | --- | --- |
| Grid | 0.2696 → 0.3284 (+21.8%) | 0.0832 → 0.0834 | 0.1620 → 0.1680 | 0.0922 → 0.0902 | 103.23 → 101.96 |
| Hex | 0.4364 → 0.4978 (+14.1%) | 0.0653 → 0.0728 | 0.1242 → 0.1307 | 0.0784 → 0.0743 | 139.98 → 139.15 |
| Polar | 0.2946 → 0.3497 (+18.7%) | 0.0931 → 0.0936 | 0.1704 → 0.1784 | 0.0745 → 0.0752 | 76.36 → 75.03 |

Construction is slower; this is not a performance optimization. A 20-build grid
profile (`/private/tmp/stage-f-construction-profile.txt`) spends 0.249 s in the
additional graph/binning consistency check and 1.130 s cumulatively in deep
copying, within 2.328 s total. Validation and ownership remain explicit rather
than weakened to recover time. Lookup and peak memory stay close to baseline;
hex first-face time increased about 11.4% in this three-repeat measurement.
These small deterministic cases do not establish large-recording performance.

### Final independent review and parent CI

Independent final review approved the source and documentation after checking
74 ownership/coordinate tests, both dependency contracts, the passing full-gate
logs, the baseline audit and both real-recording comparisons. Earlier review
findings (coordinate wording and a zero-test invocation) were corrected before
the passing final documentation gate. No remaining findings were reported.

Parent repairs were reproduced, regression-tested and pushed to their owning
branches, then merged forward without rewriting the stack. Latest reviewed
heads: #70 `9d5ce567`, #71 `87b1cd6c`, #72 `79de81ba`, #73 `f41cf100`,
#74 `1fe8cb1b`, #76 `c074957e`. The final read-only review found no failures:
#67, #68, #70, #73 and #74 each have 24 successful checks and one skipped check.
#71, #72 and #76 had running/queued matrix jobs at that checkpoint; the follow-up
review below records their completed results. The repaired Windows
3.10 cyclic-metadata check passed on its owning #70 branch. Cancelled duplicate
runs have superseding checks and are not counted as failures. Stage F's new
CI configuration has passed locally; Stage F has not been published as a PR.
Those pending remote jobs and unrun GUI/GPU checks were not reported as passes.

## Post-implementation architecture review fixes

The follow-up review identified four concrete responsibility and typing gaps.
All four are addressed without changing public signatures, method docstrings or
scientific results:

- The runtime coordinate-capability attribute hook is hidden from static
  analysis. Type checkers again reject misspelled attributes and unsatisfied
  consumer protocols. Four negative-contract checks failed before the repair
  with unused-ignore errors and pass afterward.
- Readers validate storage records, then publish through the shared geometry
  validator once. Dictionary, file and NWB regressions reproduced two graph
  and component validations before the repair; each now runs once. Malformed
  geometry remains rejected by publication.
- Subset derivation, component/source lookup and bridge-table assembly live in
  domain functions. Public methods retain signatures, guards and docstrings.
- Explicit bins and clipping share `layout/_graph.py` for standard graph
  projection. Pair-row IDs, graph vector orientation and source-grid IDs are
  preserved. Subsets keep their dedicated metadata-preserving graph path.

The starting commit is `3c722f11`. The fresh before/after captures
`/private/tmp/environment-responsibilities-{base,final}.npz` contain 4,004
identical probes, including API signatures and docstrings. The same pinned J16
60,000-sample recording slice produces 14 arrays identical to the completed
Stage F capture, using the parameters recorded above.

The isolated environment and CPU exclusions are unchanged. Follow-up checks:

| Gate | Result |
| --- | --- |
| Full CPU suite | 13,837 passed, 31 skipped |
| Geometry, layouts, file/NWB and adapters, RuntimeWarnings as errors | 4,034 passed, seven skipped |
| All-NaN plotting target modules, normal warning handling | 20 passed |
| Source doctests | 585 passed, 124 skipped |
| Executable Markdown, API, argument and import checks | 68 passed, three skipped |
| Python 3.10 coordinate, ownership, subset and combine tests | 133 passed, one skipped |
| Source and all static contracts | 256 files passed |
| Import-linter | Two contracts kept |
| Repository Ruff, formatting and whitespace | Passed |

The strict geometry run excludes exactly two all-NaN plotting nodes. Both fail
with the same NumPy warning on the archived starting source under
RuntimeWarnings-as-errors; both pass with normal warning handling. No plotting
implementation changed. The initial strict run's two failures are retained in
`environment-responsibilities-tests.log`; the baseline reproduction is
`environment-responsibilities-baseline-warnings.log`. Passing follow-up logs
use `/private/tmp/environment-responsibilities-` with suffixes `strict.log`,
`plotting.log`, `doctests.log`, `doc-api.log`, `py310.log`, `mypy-final.log`
and `imports-final.log`; the full CPU log is `cpu.log`. GUI/GPU execution remains
unrun.

Independent review found no remaining source or architecture issues and passed
147 focused tests plus both Python 3.10 static contracts. The final read-only
parent CI refresh confirms that #67, #68, #70, #71, #72, #73, #74 and #76 each
have 24 successful checks and one skipped check. Superseded cancelled duplicates
are excluded. Stage F remains local; no new PR is implied by these parent results.


## Supplied review repairs, 2026-10-15

Review issues and their owning PRs are recorded in
[the tracker](../environment-review/TRACKER.md). Parent fixes were merged
forward without rewriting published history. The additional structural work
uses a separate branch above Stage F.

An isolated CPU environment was created from `uv.lock` with `dev`, `nwb` and
`xarray` extras, then the widget dependencies installed by CI. It deliberately
has no napari plugin, so GUI fixtures cannot affect unrelated CPU tests.
The exact full-suite command at `8c35617e` was:

```bash
PYTHONPATH=src MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests -n4 -q -m 'not slow and not napari and not wallclock'
```

Result: **13,846 passed, 335 skipped** in 287 seconds. Warnings remain enabled,
including established numerical edge-case warnings. Three affine regressions
found in the earlier clean run were fixed on the owning Stage E PR and rerun
before this gate. Optional widget/docstring checks also pass with CI's widget
installation. A subsequent metadata-cycle preflight has separate complete IO
and Python 3.10 ingress verification. This is CPU validation; GUI/GPU checks
are not claimed.


After the cycle-preflight merge, the complete IO gate passed **590 tests** with
seven optional skips and RuntimeWarnings treated as errors. Source and all
static contracts passed for 257 files. Repository Ruff, formatting and whitespace
checks passed. Python 3.10 ingress tests passed 15 tests with two optional skips;
independent cycle probes included shared references and 50,000-level acyclic
nesting in the iterative helper.
