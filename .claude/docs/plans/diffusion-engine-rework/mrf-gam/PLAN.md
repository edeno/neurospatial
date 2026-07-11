# MRF-GAM place-field estimator — Implementation Plan

**Status:** Not started.

Adds a `method="glm"` penalized-Poisson GAM estimator to `compute_spatial_rate` and
`compute_spatial_rates`: occupancy enters as a log-offset (never a denominator), the smoothness
penalty is the diffusion energy (reusing the cached finite-volume eigenbasis), and `λ` is chosen
by REML — producing finite, likelihood-based place fields where the ratio estimators NaN at low
occupancy. Ships alongside a hard rename `smoothing_method → method` across every smoothing
encoder so the estimator axis has one uniform name. Design rationale:
[../design-mrf-gam.md](../design-mrf-gam.md).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained: inputs to read, contracts/designs it depends on, tasks, validation slice, fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a concrete algorithm/signature?** [designs.md](designs.md).
4. **Need broader scope / risks / integration refs?** [overview.md](overview.md).
5. **Need the reference source map?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — integration points, scope, metrics, risks, rollout.
- [shared-contracts.md](shared-contracts.md) — `method` param + validation, constants, `MRFBasis`/`MRFFit`, result GAM fields.
- [designs.md](designs.md) — module layout, resolver, Newton fit, REML, deviance, degenerate dispatch, JAX mirror.
- Phases (each ships as a separable PR):
  - [phase-0-rename.md](phase-0-rename.md) — hard-rename `smoothing_method → method` across all smoothing encoders + results + NWB + decoder. Behavior-preserving; no glm yet.
  - [phase-1-eigenbasis-resolver.md](phase-1-eigenbasis-resolver.md) — live-component rank-based eigenbasis resolver (`MRFBasis`), reusing PR2 geometry. Internal.
  - [phase-2-fit-reml.md](phase-2-fit-reml.md) — batched penalized-Poisson Newton fit + REML λ selection + deviance (`MRFFit`). NumPy/SciPy core. Internal.
  - [phase-3-glm-api.md](phase-3-glm-api.md) — wire `method="glm"` into `compute_spatial_rate(s)`, result GAM fields, validation, degenerate cases. Feature lands for encoding.
  - [phase-4-persistence-decoder.md](phase-4-persistence-decoder.md) — NWB GAM-diagnostic round-trip + `BayesianDecoder`/`decode_session` glm support.
  - [phase-5-jax-accel.md](phase-5-jax-accel.md) — float32 JAX mirror of the fit/REML with verified parity.
  - [phase-6-per-neuron-lambda.md](phase-6-per-neuron-lambda.md) — `pooled=False` per-neuron REML λ.
- [appendix.md](appendix.md) — `non_local_detector` reference file:line map (commit `89c943c`).
