# UX v0.6 Phase 1 Re-Review - Identity And Consistency

> **Status: superseded / resolved (2026-07-03).** Historical v0.6 review artifact — its findings were remediated and verified against `main`; retained for provenance. Do not treat its API references as current.

Reviewed: `.claude/docs/plans/ux-v0.6/phase-1-identity-and-consistency.md`
Date: 2026-06-04
Reviewer: Codex

## Verdict

Phase 1 now looks release-ready from the UX/API-consistency perspective. The previous review's actionable findings have been fixed: docs/examples teach the new `detect_region_crossings` call shape, `ViewRatesResult.summary_table()` uses the shared `peak_x` / `peak_y` vocabulary, mixed-type duplicate `unit_ids` produce a friendly error, `DecodingResult.to_xarray()` validates bad `times`, and the docs-site changelog now mentions the v0.6 breaking changes.

The work is still appropriately scoped. It is doing enough to create a truly better surface for beginner/intermediate neuroscience users working with large populations, and it is not doing too much. The changes standardize identity, terminal verbs, labels, and copy-paste workflows without trying to solve higher-level workflow design in this phase.

## Findings

No blocking findings.

## Residual Release Note

[pyproject.toml:3](/Users/edeno/Documents/GitHub/neurospatial/pyproject.toml:3) still reports `0.5.0`, while v0.6 deprecation messages and migration docs say "since 0.6". I do not consider this a Phase 1 implementation blocker if the version bump is intentionally deferred until the final v0.6 release step. If this branch is meant to ship immediately as v0.6, update package/docs version strings as part of the release checklist.

## Checks Performed

- Scanned for stale active-surface `peak_view_x` / `peak_view_y` references outside historical plans: none found.
- Scanned docs/examples for invalid `detect_region_crossings(..., env=env)` or deprecated old-order teaching examples: only migration/deprecation tables still show old order, intentionally.
- Confirmed [src/neurospatial/_results.py:364](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/_results.py:364) no longer uses `np.unique` for duplicate `unit_ids`.
- Confirmed [src/neurospatial/behavior/segmentation.py:439](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/behavior/segmentation.py:439) rejects new-order positional `region_name` with a clear keyword-only error.
- Confirmed [src/neurospatial/encoding/view.py:997](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/view.py:997) documents `peak_x` / `peak_y`.
- Confirmed [src/neurospatial/decoding/_result.py:547](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/_result.py:547) validates `times` before xarray construction.

## Verification

- `uv run pytest tests/encoding/test_unit_identity.py tests/encoding/test_terminal_verbs_contract.py tests/encoding/test_spatial_xarray_interop.py tests/encoding/test_naming_contract.py tests/decoding/test_decode_position_result_arg.py tests/decoding/test_xarray_interop.py tests/events/test_unit_identity.py tests/events/test_psth_terminal_verbs.py tests/behavior/test_detect_region_crossings_argorder.py tests/environment/test_factory_presets.py tests/test_doc_snippets_helper.py -q` - 167 passed.
- `uv run python scripts/test_doc_snippets.py` - 22 passed, 2 skipped.
- `uv run ruff check ...` on touched implementation/tests/examples - passed.
- Manual repro: mixed duplicate `unit_ids=[1, "a", 1]` now raises the intended `ValueError`.
- Manual repro: `detect_region_crossings(position_bins, times, env, "goal")` now raises the intended keyword-only `TypeError`.

