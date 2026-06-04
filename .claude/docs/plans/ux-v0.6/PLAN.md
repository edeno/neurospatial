# neurospatial v0.6 — UX Implementation Plan

**Status:** Draft for approval · **Created:** 2026-06-04 · **Target release:** 0.6.0
**Source of truth:** [`UX_REVIEW_2026-06-04_synthesis.md`](../../../reviews/UX_REVIEW_2026-06-04_synthesis.md) (synthesis + maintainer refinements + Scientific-Python design constraints).

---

## Vision

Make the scientifically-safe path the easy path for a beginner-to-intermediate neuroscientist scaling to 1000s of neurons — **without** building a parallel data universe. Keep the array-first functional core; add a thin friendly layer; preserve unit identity; make large-data safe by default; interoperate (optionally) with pynapple/NWB/xarray.

**What we are NOT doing:** no mutable god-object `Session`; no mandatory xarray/NWB; no dask yet; no new bespoke container classes where a standard type (xarray.Dataset, dataclass-of-arrays) suffices; no return-type-changing mode flags.

## Governing constraints

All phases obey [`api-contract.md`](./api-contract.md), which encodes:
- **Maintainer refinements:** frozen Session OK (no god-object); raw arrays are the universal baseline (interop optional/additive); `to_dataframe` → two named methods; design-for-chunking-now (defer dask impl).
- **Scientific-Python design principles:** standard types over bespoke containers; **return-type stability** (no mode flags that change return shape); duck-typing/Protocols over `isinstance`; layered friendly/strict; immutability; raise-don't-print + specific errors + no permissive defaults; keyword-only args; I/O separate from logic.

## Phases & sequencing

| Phase | Theme | Risk | Breaking? | Ships as |
|---|---|---|---|---|
| **0** | [Safety & docs](./phase-0-safety-and-docs.md) | Low | No (additive) | `0.6.0` dev (could back-port to `0.5.1`) |
| **1** | [Identity & consistency](./phase-1-identity-and-consistency.md) | Medium | Yes (deprecations + 2 behavior changes) | `0.6.0` |
| **2** | [Scale safety](./phase-2-scale-safety.md) | Medium | No (additive + perf) | `0.6.0` |
| **3** | [Interop & ergonomics](./phase-3-interop-and-ergonomics.md) | High / exploratory | No (additive) | `0.6.0` (after a design brainstorm) |

**Order:** 0 → 1 → 2 → 3. Phase 0 lands first and is reviewed before Phase 1 (it's the urgent safety + docs work and validates the workflow). Phases 1 and 2 are largely independent and can overlap. **Phase 3 gets a short design/brainstorm doc before any code** (Session/decoder shape, pynapple object surface) — it's the only phase with real design risk.

## Versioning & deprecation policy

- All renamed/reordered public API keeps the **old form as a deprecated alias** that emits `DeprecationWarning` and still works in `0.6.x`; aliases removed in `0.7.0`. (Pre-1.0, but we still give one release of runway because there are external users.)
- **Two clean-break behavior changes in 0.6** (D1 + DoD #4 → no transition shim; documented loudly in CHANGELOG `### Breaking changes`): `to_xarray()` returns a labeled `xr.Dataset` (was `DataArray` with integer coords); batch `to_dataframe()` becomes dense tidy and per-unit summaries move to the new `summary_table()`. Renamed/reordered *callables* still ship deprecated aliases (removed 0.7); these two *return-shape* changes do not (a shim that silently changes a return shape is worse than a clean, announced break).
- Every task: fail-before/pass-after tests, `ruff` + `mypy` clean, CHANGELOG entry, CI green (incl. the `-n0` doctest job and Windows/locale).

## Execution model

Subagent-driven (same as the review-remediation effort): one integration branch `feat/ux-v0.6`, per-task worktrees, implementer → spec review → quality review, one PR per phase (Phase 1 may split into 1a identity / 1b naming-contract). Phase 0 PR opened and reviewed before starting Phase 1.

## Definition of done (v0.6.0)

1. The critical units/time footgun warns loudly; the documented golden path runs in ≤10 lines for both encode and decode.
2. `unit_ids` survive every population operation and appear as real coords in xarray exports.
3. A **memory-safe summary decode** (`decode_position_summary` → `DecodingSummary`, MAP/mean/entropy, never materializing the full posterior) runs 1 hr / 5000 bins in <500 MB. The full-posterior `decode_position` keeps its `DecodingResult` contract unchanged but gains `dtype=float32` and `time_chunk` to bound the transient peak.
4. Terminal verbs (`to_dataframe`/`summary_table`/`to_xarray`/`summary`/`plot`) mean the same thing on every result class; one written API naming contract lives in CLAUDE.md.
5. Docs teach exactly one beginner path; no broken/stale snippets; CI exercises them.
6. Optional pynapple/NWB ingress + xarray.Dataset/NWB egress work, with raw arrays still the baseline.

## Decisions (resolved 2026-06-04)

- **D1 → clean break.** `to_xarray()` returns a labeled `xr.Dataset` in 0.6; no `to_dataset()` alias, no one-release DataArray shim. Loud CHANGELOG `### Breaking changes`.
- **D2 → pack it all into 0.6.** All of Phase 3 (Protocols + pynapple ingress, lazy NWB + result round-trip, `BayesianDecoder` fit/predict, `Session` bundle, `SpikeTrains`) targets **0.6.0**, gated behind the short Phase-3 RFC for design. **Escape hatch:** if the RFC finds the combined scope (Session + pynapple + decoder) too wide, Phase 3 may split and ship a subset in 0.6 with the rest in 0.7 — this **never blocks Phases 0–2**, which are independent and self-contained.
- **D3 → `Session`.** The frozen bundle is named `Session` (immutable, return-new; never a god-object).
