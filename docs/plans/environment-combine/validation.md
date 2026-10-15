# Stage C validation record

## C0 baseline and design

Branch: `feature/environment-combine`, based on completed Stage B commit
`bab36f5f277e391e54b8f93f34eb7b44582eb7b5`.

Stage B was pushed and opened as [PR #70](https://github.com/edeno/neurospatial/pull/70)
against `feature/environment-arena`. GitHub's returned base/head and exact head
commit were checked. Stage C has not been pushed and has no PR.

Use the existing isolated environment with `uv run --no-sync`,
`MPLBACKEND=Agg`, `MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache` and
`UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache`. No dependency installation was
needed.

Before production changes:

```bash
uv run --no-sync python scripts/environment_baseline.py record /tmp/combine-base.npz
uv run --no-sync pytest -p no:napari -n 0 -q \
  tests/environment/test_composite.py \
  tests/environment/test_composite_validation.py \
  tests/environment/test_composite_dimension_error.py \
  tests/environment/test_composite_new_methods.py
uv run --no-sync python docs/plans/environment-combine/probe_composition.py
```

Results:

- **4,067 environment probes**, including **584 expected refusals**. Local
  capture: `/private/tmp/environment-combine-base-bab36f5f.npz`. All 4,067
  probes compare exactly with the final Stage B capture.
- **59 composite tests passed**, with 40 existing warnings. Log:
  `/private/tmp/environment-combine-base-tests.log`.
- The deterministic [composition probe](probe_composition.py) records one
  automatically inferred MNN bridge and none with `auto_bridge=False`; gap
  lookup remains outside. The two exact single-cell components have an inferred
  straight-line bridge length of `sqrt(18)`.
- Overlapping planar bins use the first component; overlapping 3D cells are
  accepted. Mismatched `cm`/`m` labels are also accepted by the legacy composite.
- Unbounded nearest locators, and intersecting finite-radius nearest locators,
  both claim the same test point. Crossing measured track footprints both
  contain their intersection. Complementary subsets of one unrestricted nearest
  locator instead partition its unbounded domain; certification must inspect
  wrappers rather than reject every unbounded leaf. These are witnesses, not
  exhaustive geometry certification.
- Legacy `edge_attributes()` raises `KeyError('source_bin')` on the probe, and
  the ordinary `to_file` API is absent. These observations inform the migration;
  C0 does not change legacy production behavior.

Probe output: `/private/tmp/environment-combine-observations-bab36f5f.json`.
The script records old behavior rather than encoding defects as desired tests.
Production regression tests now assert the reviewed new contract; the C0
script remains a historical legacy-behavior probe and runs on base commit
`bab36f5f`, where `CompositeEnvironment` still exists.

At C0, the expanded geometry/coupling document was a proposal. The user had
approved retaining automatic bridges and broadening the design before
implementation; remaining decisions were explicitly listed. This initial
documentation-only step did not establish implementation completion or rerun
the full suite, GUI/GPU or real-data validation.


## Independent C0 review

Independent review approved C0 as a planning proposal, with no remaining blocking
design error found. Approval does not resolve the listed user decisions. The
review prompted three design corrections before implementation:

- Reuse child binnings/offsets and assemble scalar diffusion from child blocks
  plus explicit bridge conductances. No new public or stored face-index map is
  needed for the proposed first contract.
- Refuse whole-environment smoothing/diffusion when a retained child lacks valid
  numerical geometry; do not silently isolate that child numerically.
- Make certification wrapper-aware: complementary nearest-locator subsets can
  partition an unbounded domain, and track overlap must inspect active linear
  bins/masks rather than the full retained coordinate-conversion graph.

The composition script passes Ruff lint/format checks; whitespace checks pass.
C0 review did not itself approve the then-open major decisions. The subsequent
implementation contract and user migration choice are recorded below.

## C1–C4 implementation and review

Completed on `feature/environment-combine`, stacked on `bab36f5f`. The user
requested implementation after reviewing the expanded design and explicitly
chose immediate removal of `CompositeEnvironment` after migration. No Stage C
push or pull request was made.

The implementation adds owned built-in composition locator and arena records,
child-to-combined identities, disjoint-support certificates, deterministic MNN
bridges and scalar diffusion coupling. The public factory returns an ordinary
`Environment`. Consumers, exports, error tests and the API snapshot were migrated
with removal of the legacy class. The user guide has runnable construction and
conductance examples.

### Independent review and regressions

Review was performed independently at the snapshot/validation boundary and again
on integrated scientific behavior. Confirmed findings were reproduced, fixed
and checked again:

- Revalidate nested child face numbers, volumes and pair alignment, as well as
  duplicated aggregate faces and navigation lengths. A saved wrapper cannot
  silently disagree with its child geometry.
- Bound recursive arena decoding before recursive construction; a 600-level
  malformed input now raises `ValueError`, not `RecursionError`.
- Reapply walls and arena consistency in each child's source frame. Rounded
  globally transformed coordinates previously refused valid rotated walls.
- Assemble subset diffusion only from retained child bins; removing an
  uncertified child no longer blocks the remaining certified geometry.
- Certify transformed measured-track supports and refuse ill-conditioned
  nearly parallel segment clearance. The previous segment-distance calculation
  could falsely certify a long crossing pair as disjoint.

The final independent checks reported no remaining confirmed defect:

- **18 round-trip matrix cases:** dimensions 1/2/3, bridges on/off and native,
  subset or rotated/scaled/translated geometry, including offsets of `1e9`.
  Checked deepcopy, pickle, dict and JSON, geometry, edge order, source mapping,
  diffusion and conserved mass.
- **22 additional strict cases:** linear exposure in 1/3/4 dimensions, reversed
  and stationary trajectories, transformed/subset boxes, partial mask domains,
  nested bridges and conductance scaling/refusal at floating-point limits.

Independent logs are local artifacts at
`/private/tmp/stage_c_independent_roundtrips.log` and
`/private/tmp/stage_c_final_review.log`. Permanent regression cases live in
`tests/layout/test_composition_snapshot.py` and
`tests/environment/test_combine.py`.

### Final verification

All Python commands used the existing isolated environment. No package was
installed or updated. The final gates passed:

| Check | Result |
| --- | --- |
| Focused composition, polar capability, explicit-bin, adapter and NWB regressions; runtime warnings treated as errors | **276 passed**, 2 expected mesh-geometry warnings |
| Broad CPU suite | **13,485 passed, 31 skipped**, 257.05 seconds |
| Source doctests | **596 passed, 126 skipped**, 176.19 seconds |
| Source and four typing contracts with mypy | **Success**, 240 files |
| Exact staged foundation tests and mypy | **163 passed**; mypy success, 236 files |
| Exact staged integration: feature, API/import/error/typing and executable-document tests | **213 passed, 3 skipped** |
| Final composition suite, including four added tie/cutoff/boundary acceptance cases | **62 passed**, runtime warnings treated as errors |
| Runnable combine guide | **1 passed**, 21 unrelated documents deselected |
| Ruff lint/format and Git whitespace checks | **Passed** for changed Python files and staged changes |

Broad CPU invocation:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache \
UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache \
uv run --no-sync pytest tests -p no:napari -n 4 -q \
  -m 'not slow and not napari and not gui and not wallclock' --tb=short
```

As in the [Stage B CPU gate](../environment-immutable/validation.md), the actual
invocation additionally deselected 51 GUI cases blocked by the sandbox's Napari
cache permissions. The exact local list is
`/private/tmp/immutable-gui-exclusions.json`, applied by
`/private/tmp/run_immutable_cpu.py`. This result does not claim GUI coverage.
The broad run emitted existing numerical, plotting and dependency warnings;
the focused scientific regression gate promoted runtime warnings to errors.

The four final acceptance cases were added after the broad run; all 62
composition tests were then rerun without production changes. The staged
integration gate log is `/private/tmp/combine-integration-staged.log`.

An earlier broad invocation overlapped live source edits and had stale worker
imports and an outdated capability expectation. The result above is the fresh
rerun after those issues were resolved; it is the acceptance result. Final logs:
`/private/tmp/environment-combine-cpu-final.log`,
`/private/tmp/combine-frozen-focused.log`,
`/private/tmp/combine-doctests.log`, and
`/private/tmp/combine-mypy-release.log`.

### Native behavior comparison

Final baseline capture contains **4,072 probes**, including **584 expected
refusals**. Every one of the **3,920 non-API native probes** matches the Stage B
capture exactly, including lookup, geometry, numeric outputs, plotting,
persistence and expected errors. The five added API probes correspond to
`combine` on the shared factory surface and the three Cartesian composition
inspection members. Removal of the root `CompositeEnvironment` export is the
intentional public API snapshot change.

Captures: `/private/tmp/environment-combine-base-bab36f5f.npz` and
`/private/tmp/environment-combine-final.npz`. The existing baseline script was
used for recording; non-API keys were compared with its exact comparison logic.
No native numerical difference was accepted as refactor fallout.

### Scope and unrun checks

This is simulated/deterministic scientific validation, including real NWB file
round trips with access after reader closure. No new real-recording notebook
workflow, GUI/GPU execution, slow suite or wall-clock performance benchmark was
run. The local `data/` directory contains only the README and download helper;
no cached real recording was available for a slice test. No performance
improvement is claimed.

Conservative geometry checks can refuse disjoint mixed supports or ambiguous
track clearance. Combined environments have no single global track coordinate
or regular grid. Gradient/divergence refuse nonphysical bridges; positive
geometric couplings require supported scaling. Clipping retained bridges needs
a separately designed reconnection policy. These limits are explicit in the
[user guide](../../user-guide/composite-environments.md) and
[geometry contract](geometry-and-coupling.md).

### Commit groups

- `2808cf2f`: C0 design and baseline capture.
- `0ebc21dd`: owned composition records, overlap certificates, recursive codecs
  and validation at publication/loading.
- `74652f0a`: public combine factory, bridges and numerical consumers, derivation,
  legacy removal, regression tests and runnable migration guide.
- Final documentation commit: completed contract, roadmap status and this
  validation record.

The first implementation group was exported from the index and tested using
`PYTHONPATH=<snapshot>/src` with the existing virtual environment. The integrated
snapshot was checked the same way. Ruff, formatting, mypy and the relevant tests
were run explicitly; commits disabled automatic hooks to avoid the hook runner
stashing cross-file dependencies from the remaining logical groups.
