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
Production regression tests will assert the reviewed new contract in each
implementation phase.

The expanded geometry/coupling document is a proposal. The user has approved
retaining automatic bridges and broadening the design before implementation;
remaining decisions are explicitly listed. C0 completion does not establish
Stage C implementation completion. Full-suite, GUI/GPU and real-data validation
have not been rerun for this documentation-only start.


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
Remaining major decisions are listed in the design and are not treated as
approved merely because C0 is recorded.
