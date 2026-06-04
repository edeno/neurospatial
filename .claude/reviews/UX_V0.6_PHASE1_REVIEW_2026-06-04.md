# UX v0.6 Phase 1 Review - Identity And Consistency

Reviewed: `.claude/docs/plans/ux-v0.6/phase-1-identity-and-consistency.md`
Date: 2026-06-04
Reviewer: Codex, with independent agents Russell, Noether, and Epicurus

## Verdict

Phase 1 is directionally right and materially improves the API surface. The core implementation now has the important UX primitives neuroscientists need at 1000+ neuron scale: real `unit_ids`, dense per-bin tables, per-unit `summary_table()`, labeled `xarray.Dataset` exports, consistent batch `classify()`, and experiment-shaped environment factories.

I would not call Phase 1 done for release yet. The code is mostly there, but the public teaching surface is not: several docs/examples still show invalid or deprecated calls, the docs-site changelog is stale, and one result family keeps stale view-specific summary column names. Those are not cosmetic issues. For beginner/intermediate Python users, copy-paste docs and column names are the API.

## Findings

### High - docs/examples still teach invalid or deprecated `detect_region_crossings` calls

- [docs/user-guide/trajectory-and-behavioral-analysis.md:239](/Users/edeno/Documents/GitHub/neurospatial/docs/user-guide/trajectory-and-behavioral-analysis.md:239) shows `detect_region_crossings(position_bins, times, region_name="goal", env=env, ...)`, but the implementation signature at [src/neurospatial/behavior/segmentation.py:322](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/behavior/segmentation.py:322) does not accept `env=`. This copy-paste raises `TypeError: detect_region_crossings() got an unexpected keyword argument 'env'`.
- The same guide still teaches the old positional order at [docs/user-guide/trajectory-and-behavioral-analysis.md:244](/Users/edeno/Documents/GitHub/neurospatial/docs/user-guide/trajectory-and-behavioral-analysis.md:244) and [docs/user-guide/trajectory-and-behavioral-analysis.md:247](/Users/edeno/Documents/GitHub/neurospatial/docs/user-guide/trajectory-and-behavioral-analysis.md:247).
- [examples/14_behavioral_segmentation.py:163](/Users/edeno/Documents/GitHub/neurospatial/examples/14_behavioral_segmentation.py:163) has the same invalid `env=env` form.
- [examples/23_path_progression.py:243](/Users/edeno/Documents/GitHub/neurospatial/examples/23_path_progression.py:243) still uses the deprecated old order.

Fix these to `detect_region_crossings(position_bins, times, env, region_name="goal", ...)`, update the mirrored `docs/examples` copies and notebooks, and add at least one docs snippet/example smoke test for this call. This is the highest-priority UX gap because it directly breaks a beginner copy-paste path.

### Medium - the MkDocs changelog omits Phase 1 breaking changes

`mkdocs.yml` exposes [docs/changelog.md](/Users/edeno/Documents/GitHub/neurospatial/mkdocs.yml:217) in the docs site, but that page only contains old generic entries at [docs/changelog.md:8](/Users/edeno/Documents/GitHub/neurospatial/docs/changelog.md:8). The root [CHANGELOG.md](/Users/edeno/Documents/GitHub/neurospatial/CHANGELOG.md:12) has the correct v0.6 breaking-change and deprecation material.

Users reading the hosted docs will miss the two clean breaks: `to_xarray()` now returns `Dataset`, and batch `to_dataframe()` is dense while summaries moved to `summary_table()`. Either sync `docs/changelog.md` from the root changelog or make the docs site link the root changelog.

### Medium - `ViewRatesResult.summary_table()` keeps stale `peak_view_x` / `peak_view_y` names

Phase 1 correctly deprecates `peak_view_location()` in favor of `peak_location()` / `peak_locations()` at [src/neurospatial/encoding/view.py:263](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/view.py:263) and [src/neurospatial/encoding/view.py:819](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/view.py:819). But the per-unit table still documents and emits `peak_view_x` / `peak_view_y` at [src/neurospatial/encoding/view.py:997](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/view.py:997) and [src/neurospatial/encoding/view.py:1080](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/view.py:1080).

That weakens the "same word, same meaning" contract in the exact table users will sort and filter. Unless there is a strong domain reason to keep the prefix, rename these to `peak_x` / `peak_y` to match spatial summaries and the renamed accessor. Since v0.6 is already a breaking release, this is the right moment to clean it up.

### Medium - mixed int/string `unit_ids` crash `to_xarray()` with a raw NumPy `TypeError`

[src/neurospatial/_results.py:365](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/_results.py:365) uses `np.unique(ids, return_counts=True)` to detect duplicate `unit_ids`. The contract allows integer or string labels, and mixed labels can naturally appear when users combine loaded unit IDs with hand-added labels. NumPy sorts object arrays during `unique`, so a mixed array such as `[1, "a", 1]` raises:

```text
TypeError: '<' not supported between instances of 'str' and 'int'
```

Use an order-preserving hash/count duplicate check instead, and keep the friendly `ValueError` that names duplicate labels. This preserves the xarray label-selection contract without exposing beginners to a NumPy internals error.

### Low - transitional `detect_region_crossings` misclassifies a common mistake

`detect_region_crossings(position_bins, times, env, "goal")` is the new order plus a positional `region_name`. Because the dispatcher branches on `arg4 is not None` at [src/neurospatial/behavior/segmentation.py:421](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/behavior/segmentation.py:421), it treats this as the old order, emits the old-order deprecation warning, then fails later with `AttributeError: 'str' object has no attribute 'regions'`.

Reject this shape immediately with a clear `TypeError` telling users that `region_name` is keyword-only in the new API. Add a test next to [tests/behavior/test_detect_region_crossings_argorder.py:40](/Users/edeno/Documents/GitHub/neurospatial/tests/behavior/test_detect_region_crossings_argorder.py:40).

### Low - direct `DecodingResult.to_xarray()` with bad `times` gets generic xarray errors

[src/neurospatial/decoding/_result.py:542](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/_result.py:542) trusts `self.times` directly. `decode_position()` validates time length, but users can construct `DecodingResult` directly. A wrong-length `times` array currently falls through to xarray's generic "conflicting sizes for dimension 'time'" error at [src/neurospatial/decoding/_result.py:567](/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/_result.py:567).

Mirror the `decode_position()` validation in `DecodingResult.to_xarray()` or `__post_init__`: 1-D only, length equals `posterior.shape[0]`, with a neurospatial-style WHY/HOW message.

## Strategic UX Assessment

This phase is not doing too much. The changes are structural but justified: identity, terminal verbs, and xarray labels are the minimum viable foundation for 1000+ neuron workflows.

It is doing enough in the implementation core, but not enough in the teaching surface. For this audience, "API consistency" has to include docs, examples, notebooks, error messages, and table column names. A clean `summary_table()` is not enough if the first guide still teaches an invalid call.

The factory presets are a good example of the right amount of radical change: `open_field()` gives a beginner-friendly front door, while `linear_track()` and `maze()` deliberately refuse to infer topology from raw position clouds. That protects users from plausible but wrong analysis.

## Open Question

[pyproject.toml:3](/Users/edeno/Documents/GitHub/neurospatial/pyproject.toml:3) still says `0.5.0`, while warnings and docs say "since 0.6". If version bump is intentionally deferred until the end of the v0.6 plan, fine. If Phase 1 is release-candidate work, update the package version before shipping.

## Verification

- `uv run pytest tests/encoding/test_unit_identity.py tests/encoding/test_terminal_verbs_contract.py tests/encoding/test_spatial_xarray_interop.py tests/encoding/test_naming_contract.py tests/decoding/test_decode_position_result_arg.py tests/decoding/test_xarray_interop.py tests/events/test_unit_identity.py tests/events/test_psth_terminal_verbs.py tests/behavior/test_detect_region_crossings_argorder.py tests/environment/test_factory_presets.py tests/test_doc_snippets_helper.py -q` - 163 passed.
- `uv run python scripts/test_doc_snippets.py` - 21 passed, 2 skipped.
- Reproduced `build_population_dataset(..., unit_ids=np.asarray([1, "a", 1], dtype=object))` failing with raw `TypeError`.
- Reproduced `detect_region_crossings(position_bins, times, env, "goal")` failing with misleading deprecation warning plus `AttributeError`.
- Reproduced docs-style `detect_region_crossings(..., region_name="goal", env=env)` failing with unexpected keyword `env`.

