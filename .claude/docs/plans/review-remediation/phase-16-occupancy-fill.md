# Phase 16 — Reconcile `min_occupancy` NaN with decoding

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Convergence item C2** (flagged by both reviews). The library's *recommended* encoding call (`min_occupancy=0.5`) produces encoding models with NaN in low-occupancy bins, and the *recommended* decode call (`decode_position(validate=True)`) rejects NaN — so the documented golden path crashes, and the examples paper over it with a silent `np.nan_to_num`. This phase makes encode→decode compose without manual NaN scrubbing.

**Depends on:** phase 2 (occupancy index fix) should land first, since occupancy values are an input to the masking decision here.

## Decision (RESOLVED): `fill_value: float | None = None`

Resolved by the maintainer (2026-06-02). See [overview.md Open Questions #1](overview.md#open-questions).

- **Where the fix lives:** add `fill_value: float | None = None` to `compute_spatial_rate`/`compute_spatial_rates`; **and** keep `decode_position`'s defensive zero-rate handling of any residual NaN encoding-model bins (excluded from the Poisson sum) with a one-time `warnings.warn`.
- **The default is `None` (opt-in), NOT `0.0`.** `fill_value=None` preserves the current NaN behavior in masked/low-occupancy bins, so existing callers see no behavior change. Users opt into zero-fill by passing `fill_value=0.0` explicitly.

The Tasks below implement this resolved decision: `fill_value=None` default on the encoder **plus** defensive zero-rate handling in the decoder. The documented golden path passes `fill_value=0.0` explicitly.

**Reconcile with phase 1's `validate=True` NaN-rejection.** Phase 1 adds NaN-rejection to `decode_position(validate=True)`. Phase 16's decoder-side "treat NaN encoding-model bins as zero-rate (excluded)" must be made **compatible** with that guard so the two phases do not both fire on the same NaN bins: the zero-rate exclusion for **encoding-model** NaNs must run **before/instead of** the `validate=True` rejection for those bins specifically (the validate guard should still reject NaN in the *other* decode inputs — e.g. observed spike counts — but must not reject an encoding-model bin that phase 16 has already handled as zero-rate). The executor should reconcile these when both phases have landed (check whichever lands second against the other's behavior), so a `validate=True` decode of a `fill_value=None` model does not raise on the very bins the zero-rate path is designed to absorb.

**Inputs to read first:**

- [encoding/spatial.py:87](../../../../src/neurospatial/encoding/spatial.py#L87) — `PlaceFieldsResult` / `compute_spatial_rate(s)`: where `min_occupancy` masking sets NaN.
- [decoding/posterior.py:265](../../../../src/neurospatial/decoding/posterior.py#L265) and `decoding/likelihood.py` — where the Poisson likelihood consumes encoding-model rates and where `validate=True` rejects NaN.
- `examples/` (the notebook/script the review cited as injecting `np.nan_to_num`, ~example 20) — the silent scrub to remove.

## Tasks

- **Encoder:** add `fill_value: float | None = None` to `compute_spatial_rate` and `compute_spatial_rates`. After computing rates, replace masked/low-occupancy NaN bins with `fill_value` when it is not `None`; the default `None` keeps NaN, so existing callers see no behavior change. Keep `occupancy` and the masked-bin set available on the result for callers who need it. (Read the actual masking code — verify exactly which array carries the NaN.)
- **Decoder:** in `decode_position`/the Poisson likelihood, treat any remaining NaN encoding-model bin as contributing zero to the per-bin log-likelihood (i.e. exclude it), emitting a single `warnings.warn` the first time per call rather than raising. This is defense-in-depth so a `fill_value=None` model still decodes. **Make this compatible with phase 1's `validate=True` NaN-rejection** (see the decision section above): the zero-rate exclusion for encoding-model NaN bins must happen **before/instead of** the `validate=True` rejection *for those encoding-model bins specifically*, so the two phases do not both fire on the same NaN bins (the validate guard still rejects NaN in other decode inputs). Reconcile this when both phases have landed.
- Remove the silent `np.nan_to_num` from the example(s): the golden-path encode call now passes `fill_value=0.0` explicitly (e.g. `compute_spatial_rates(..., min_occupancy=0.5, fill_value=0.0)`), so the example no longer needs `np.nan_to_num`. Change the documented `min_occupancy=0.5` recommendation text in QUICKSTART/CLAUDE so it explains `fill_value` and shows the explicit `fill_value=0.0` golden path.
- Docstrings for both functions; CHANGELOG entry describing the new default behavior.

## Deliberately not in this phase

- The `decode_position` `n_bins` mismatch check — already in phase 1.
- The occupancy active-bin index fix — phase 2.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_spatial_rate_fill_value_zero` | With `min_occupancy>0` creating low-occupancy bins, `fill_value=0.0` yields finite (zero) rates there, no NaN. |
| `test_spatial_rate_fill_value_none_preserves_nan` | Default `fill_value=None` keeps NaN in masked bins (no behavior change for existing callers). |
| `test_encode_decode_golden_path` | The documented `compute_spatial_rates(min_occupancy=0.5, fill_value=0.0)` → `decode_position()` path runs end-to-end with NO manual NaN scrubbing (fails before this phase). |
| `test_decode_warns_on_nan_model` | A model with NaN bins decodes (NaN bins excluded) and warns once. |

## Fixtures

Reuse decoding `conftest` env + trajectory; construct a session with deliberately low occupancy in a subset of bins.

## Review

Dispatch `code-reviewer`. Confirm the decision-gate resolution is recorded; the golden-path test fails before and passes after; no `np.nan_to_num` remains in examples; the new default is documented in docstrings + CHANGELOG; behavior change is intentional and explained.
