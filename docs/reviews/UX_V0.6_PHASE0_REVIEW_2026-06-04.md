# UX v0.6 Phase 0 Review - 2026-06-04

> **Resolution (2026-06-04, branch `feat/ux-v0.6`): all four findings below were
> verified and fixed in a Phase 0 follow-up.**
>
> - **High — `decode_session(encoding_models=...)` silent spike-drop:** fixed in
>   `09b1dea` + `891a5be`. `decode_session` now surfaces one out-of-window
>   units-mismatch warning in both branches (the encoder owns it in the
>   `encoding_models=None` branch — preserving its inactive-bin warning too —
>   and `decode_session` does its own check in the passthrough branch), with a
>   `warn_on_drop` kwarg and encoding-models/inactive-bin regression tests.
> - **High — golden path not taught:** fixed in `720547f` + `b0b1e55`.
>   `examples/20_bayesian_decoding` now leads with `decode_session` (manual
>   3-call path kept as "Advanced"); `docs/user-guide/workflows.md` gained a
>   one-call decode workflow; a CI snippet covers it; README points to it.
> - **Medium — install page version banner:** fixed in `0abba24`
>   (`docs/getting-started/installation.md` v0.4.0 → v0.5.0).
> - **Low — notebook CI only ran nb 11:** fixed in `0abba24`
>   (`test_notebooks.yml` now also executes `20_bayesian_decoding.ipynb`).
>
> The original findings are preserved verbatim below for the audit trail.

---

Scope: current `.claude/docs/plans/ux-v0.6/phase-0-safety-and-docs.md` plus the Phase 0 implementation already present in `src/`, `tests/`, docs, and examples.

## Findings

### High: `decode_session(..., encoding_models=...)` still has a silent spike-drop path

Phase 0's critical safety goal is to warn on spike/trajectory time-unit mismatch. The encoding path now does this, but `decode_session` skips encoding when `encoding_models` is provided and then calls `bin_spikes_in_time` directly:

- `src/neurospatial/decoding/session.py:203-226`
- `src/neurospatial/decoding/_binning.py:156-158`

`bin_spikes_in_time` bins with `np.histogram`; spikes outside the explicit `[t_start, t_stop]` grid are excluded with no warning. The `decode_session` docstring even says this at `src/neurospatial/decoding/session.py:123-126`. That means a common reuse workflow can still turn millisecond spike times plus second trajectory times into an all-zero count matrix, then produce a plausible posterior from the no-spike likelihood.

Recommendation: add the same drop-fraction warning to the decode-time binning boundary, or have `decode_session` compute one aggregate warning before calling `bin_spikes_in_time`. Include `encoding_models` in the fail-before/pass-after test, since that is the branch that bypasses the encoder warning. If a silence escape hatch exists, use the same `warn_on_drop=False` spelling as encoding.

### High: the new golden-path wrapper is not visible in beginner docs

`decode_session` exists and has tests, but the visible user surface still teaches the manual decode path. `examples/20_bayesian_decoding.py` imports `bin_spikes_in_time` and `decode_position` at lines 48-57, then walks users through binning and decoding manually at lines 286-320. `docs/user-guide/workflows.md` only covers the place-field workflow; `docs/snippets.yml` only adds the place-field workflow snippet. A search of README/docs/examples/quickstart found no `decode_session` mention.

This means Phase 0 adds a good API but does not yet make it the easy path. For UX, that is almost the same as not adding it.

Recommendation: make `decode_session` the first Bayesian decoding path in `examples/20_bayesian_decoding.py`, README or workflows, and the snippet harness. Keep the manual three-call path as an "advanced/custom control" section.

### Medium: the docs version sweep missed a user-facing install page

`docs/getting-started/installation.md:101` still says `neurospatial v0.4.0 has been tested with:`. Phase 0 lists the README and docs index version bumps, but not this page.

Recommendation: add this page to 0.6 or use a targeted docs grep for release-status/version banners, excluding historical plans and reviews.

### Low: Phase 0 touches notebook 20, but notebook CI still only executes notebook 11

`.github/workflows/test_notebooks.yml:37-46` executes `examples/11_place_field_analysis.ipynb` only. That is fine as a representative notebook smoke test, but Phase 0 specifically updates `examples/20_bayesian_decoding`; without any execution coverage, the decode tutorial can drift again.

Recommendation: either add a lightweight execution/smoke path for notebook 20 or add a small CI-tested docs snippet that exercises `decode_session`.

## What Looks Good

The high-risk encoding warning path is much stronger now: tests cover single-neuron, batch, `compute_spatial_rates`, `warn_on_drop=False`, inactive bins, and `n_jobs=2` warning aggregation. The `decode_session` implementation also fixed the earlier empty-`times` boundary with a clear error, and the targeted Phase 0 tests pass.

Validation/error-message tasks are in good shape: array-like trajectory inputs, `Inf` event times, direct `Environment()` guidance, internal-doc reference removal, and `distance_to` region errors all have credible coverage.

## Verification

Ran:

```bash
uv run pytest tests/encoding/test_encoding_binning.py tests/decoding/test_decode_session.py tests/behavior/test_trajectory_metrics.py tests/events/test_alignment.py tests/test_no_internal_doc_refs.py tests/ops/test_distance_utilities.py -q
```

Result: 188 passed.

Ran:

```bash
uv run python scripts/test_doc_snippets.py
```

Result: 19 passed, 0 failed, 2 skipped.
