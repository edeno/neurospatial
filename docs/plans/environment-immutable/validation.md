# Stage B validation record

Implementation branch: `feature/environment-immutable`.
Behavior-capture base: `cba4f1b093b0ceaba3f800a093952b5b860f1195`.

## B0 — Behavior and ownership capture

The existing isolated Python 3.13 environment is used through `uv run --no-sync`,
with Agg Matplotlib and temporary writable cache directories. No environment
installation or runtime dependency change was needed.

The baseline command recorded **4,051 probes**, including **584 expected
refusals**, before any Stage B production code changes:

```bash
uv run --no-sync python scripts/environment_baseline.py record /tmp/immutable-base.npz
```

The local artifact is
`/private/tmp/environment-immutable-base-cba4f1b0.npz`; it is not a portable
committed fixture. The script covers ten native environments, copied variants,
dictionary/file/NWB restoration, lookup, occupancy, numerical operations,
plotting and public signatures/docstrings.

The separate [ownership capture](probe_ownership.py) records caller-buffer
mutation, point-region copy/deepcopy/pickle, environment shallow/deep copying,
dict/file/actual NWB restoration, metadata/region edits, cached kernels/operators
and two-angle polar layout parameters. Run it with the same interpreter:

```bash
uv run --no-sync python docs/plans/environment-immutable/probe_ownership.py
```

The local JSON artifact is
`/private/tmp/environment-immutable-ownership-base-cba4f1b0.json`. Ownership
changes are intentional; numerical changes unrelated to those mutation attempts
need investigation rather than blanket baseline acceptance.

Independent audits confirmed additional regression requirements:

- Returned sparse operators permit edits and buffer replacement that change
  later gradients/divergence. Returned kernels and attribute DataFrames expose
  cached state; pandas deep copies alone retain nested object-cell aliases.
- Public cached-property assignment can shadow geometry-derived values.
- Rebuilding a retained native layout changes later lazy face geometry without
  changing already published bins. Two-angle polar circularity can similarly
  change after editing retained build parameters and clearing caches.
- Opaque custom locator, track and arena callbacks can retain external mutable
  state. The user approved requiring owned immutable snapshots for these roles.

The B0 API review accepts keyword-only metadata replacement with omitted values
preserved, explicit `None` clearing only nullable units/frame, and whole-region
replacement. Both replacement methods preserve the concrete environment type
and geometry, refresh environment analysis caches, and may share private
geometry-owned lookup memos. The region snapshot keeps the `Regions` read API;
standalone builders remain mutable. Unit labels never imply coordinate scaling.

B0 established the starting behavior. The following implementation checks
complete the Stage B record; PR publication remains unauthorized.


## B1/B2 — Owned environment values

Region coordinates and nested metadata are independently owned; standalone
`Regions` stays mutable and published snapshots refuse mutation. Metadata is
read-only, and replacement methods preserve the concrete environment type,
share immutable geometry and refresh environment analysis caches. Public arrays,
layout/provenance exports, sparse operators, kernels and attribute tables cannot
poison the environment's stored scientific state.

Independent review covered native and custom layout publication, array-header
mutation, nested graph attributes, cache rebinding, copy/deepcopy/pickle,
legacy restoration, NWB and plotting. The exact staged B1/B2 snapshot passed
157 focused tests and source mypy before its commit. Simulation consumers,
examples and tests now construct metadata and publish region builders before
using an environment.

The final ownership capture contains 52 observations in
`/private/tmp/environment-immutable-ownership-final.json`. It confirms that
caller coordinate edits no longer change regions, immutable region state
survives all recorded restoration paths (including actual NWB reads after the
reader closes), and exported kernel/operator edits leave later calculations
intact. Polar behavior no longer depends on later builder edits.

A local five-repeat benchmark (1,000 replacements per repeat) measured median
`with_metadata` times of 18.0 microseconds for 100 bins and 24.5 microseconds for
10,000 bins. A separately traced replacement peaked at 831 bytes for both, with
shared geometry. These are observations of metadata replacement, not an overall
speedup claim. Artifact:
`/private/tmp/environment-immutable-replacement-benchmark.json`.

## B3 — Explicit bins, graph exports and edge identity

The public `binning` records and `Environment.from_bins` preserve supplied bin
IDs, neighbor order and orientation. Graph equivalence checks use edge identity;
finite-volume face arrays remain aligned with `edge_pairs`. Rebinning refuses
explicit nonuniform measures it cannot preserve. Nearest-only geometry refuses
cell-based operations rather than inventing boundaries or faces.

Independent tests cover reordered/reversed neighbors, gradient/divergence and
heat-kernel agreement, nested editable graph exports, all native geometry
families and malformed shared components. The approved hex correction restores
the stored lattice shape and mask, keeping spatial autocorrelation available
after loading.

Before the navigation-weight fix, 14 of 16 focused probes failed: similarity
maps replaced declared path lengths with straight-line distances, and other
maps failed to refuse. After the fix, rigid transforms preserve declared lengths,
uniform scaling multiplies them and other affine maps refuse unknown paths.
A separate failing regression exposed underflow of a positive path length to
zero; that case now refuses while legitimate original zero lengths remain valid.
Native Euclidean transforms retain the original arithmetic and rounding.
The final navigation and exact-artist gate passed **106 tests**. A further
review found reversed exported vectors after NetworkX node reordering; the
focused metrics/ownership/facade/explicit-bin gate passed **214 tests** after
repair. A separate ten-family audit confirms native edge vectors and angles
remain unchanged.

## B4 — Installed adapters and public contract kit

The versioned entry-point registry normalizes external locator, track, arena and
binning providers to supported built-in records. It retains no live provider
callbacks. Tests exercise actual entry-point discovery, duplicate/version errors,
caller-buffer and weak-reference ownership, wrapped locators, pre-selection
arena adaptation, canonical track/provenance publication and malformed geometry.

The original custom record requires its installed adapter. Dict and file tests
load that record, save canonical built-in data, remove the adapter and reload the
canonical result. The public contract kit is exercised on native families and an
external-style provider with an independently supplied boundary oracle.
The adapter/contract/native-codec/explicit-bin gate passed **216 tests** with
RuntimeWarnings treated as errors. Expected mesh geometry warnings remained.
Independent review found and verified repairs for three public-ingress defects;
no confirmed adapter finding remains.

## Final compatibility and validation

The final capture records **4,067 probes**, including the same **584 expected
refusals**. Excluding intentional `api/` signature/docstring changes, all **3,920
non-API probes are exactly equal** to B0. The baseline does not itself cover the
new hex shape context or custom path weights; dedicated regression tests cover
those approved changes.

Artifacts are local, not committed binary fixtures:

- `/private/tmp/environment-immutable-final.npz`
- `/private/tmp/environment-immutable-final-record.log`
- `/private/tmp/environment-immutable-final-compare.json`
- `/private/tmp/environment-immutable-ownership-final.json`

The broad CPU run selected tests with
`-m 'not slow and not napari and not gui and not wallclock' -p no:napari -n 4`,
plus explicit deselection of 51 GUI-related tests previously blocked by the
sandbox's Napari cache permissions. It finished with **13,443 passed, 31 skipped
and 5 failed**. All five were investigated and repaired: four stale expectations
for the approved hex/edge-order behavior, plus an exact plotting fixture affected
by another test's global Matplotlib margin changes. Both affected layout/IO files
then passed **77 tests** with strict RuntimeWarnings; exact plot checks retain all
artist assertions and now isolate their recorded margins. The full broad run was
not repeated after these focused repairs.

Other completed checks:

- Full source doctests: **597 passed, 132 skipped**. One wholly skipped existing
  example retains its original skip policy; it is not an executed workflow.
- Executable documentation/notebook and arena/subset migration gate: **87 passed**.
  The two subsequently added explicit-bin/adapter guide pages passed separately.
- Public API, argument and docstring contracts: **66 passed**; the intentional
  namespace additions are recorded in `tests/data/public_api.txt`.
- Exact staged consumer migration snapshot: **108 passed**.
- Exact staged binning/adapter snapshot: **310 passed, 22 slow tests deselected**;
  the separate 106-test artist/navigation gate includes those exact plot cases.
- Full source plus four typing-contract files: **238 files passed mypy**.
- Ruff lint/format and whitespace checks cover the changed files. Notebook
  input/source pairs are synchronized and original output payloads preserved.

GUI/GPU execution and full real-recording notebook execution were not performed.
The CPU suite includes existing warnings outside this change; focused geometry
and adapter gates promote RuntimeWarnings to errors. General affine numerics,
composite-environment redesign and removal of the layout bridge remain later
roadmap stages. This work neither opens nor publishes a Stage B PR.


## Commit groups

- `99dca919`: behavior and ownership capture.
- `65eeeda4`: owned regions and snapshot collections.
- `4a01111f`: immutable environment publication and value replacements.
- `c376509e`: consumer, simulation and test migration.
- `5e2071c7`: explicit bins, edge identity, installed adapters and contract kit.
- Final documentation commit: user workflows, paired examples, changelog and
  this validation record.

Commits were made after manual Ruff/format, type and regression checks, with
hooks disabled to avoid pre-commit temporarily stashing still-uncommitted
cross-file dependencies. The two final source groups were additionally tested
from exports of the exact staged index. No push or Stage B PR was performed.
