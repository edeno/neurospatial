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

Implementation and verification in progress. No new numerical transform
capabilities are claimed by this phase.
