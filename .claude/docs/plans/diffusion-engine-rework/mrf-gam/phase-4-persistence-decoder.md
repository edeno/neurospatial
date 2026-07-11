# Phase 4 — Persistence + decoder glm support

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md#result-fields)

Make glm results persist to NWB (with GAM diagnostics) and flow through the decoder — the
functional path (`decode_session` / `decode_session_summary`) and the `BayesianDecoder` class.
The `method` rename already landed in phase-0; this phase adds the glm-specific fields and
round-trip.

**Inputs to read first:**

- `src/neurospatial/io/nwb/_fields.py:434` (`write_spatial_rates`), `:616` (metadata write — now `"method"`), `:697` (`read_place_field`), `:855-862` (`SpatialRatesResult(...)` construction).
- `src/neurospatial/decoding/session.py:92,100` (`decode_session`), `:348-356` (`_build_encoding_model`), `:566-574,:609` (`_encode_and_bin` forward).
- `src/neurospatial/decoding/estimator.py:41` (`BayesianDecoder`), `:151-153` (config fields), `:246,:351-353` (`fit` forward).
- [shared-contracts.md#result-fields](shared-contracts.md#result-fields) — the diagnostics to persist.

**Contracts referenced:**

- [Result GAM fields](shared-contracts.md#result-fields) — the diagnostics to round-trip; ratio results persist them absent/`None`.
- [The `method` parameter + validation](shared-contracts.md#method-param) — the decoder mirrors the same method-specific validation as phase-3.

## Tasks

- **NWB** (`io/nwb/_fields.py`):
  - `write_spatial_rates`: write `bandwidth` as nullable (`None` for glm). **GAM diagnostics round-trip** ([spec §8](../design-mrf-gam.md)):
    - Metadata scalars: `penalty`, `rank`, `n_iter`, `converged`, `reml_objective`, and the `(rank,)` `penalty_weights` vector.
    - Per-unit table columns: `deviance` `(n_units,)`, and `coefficients` as a fixed-length `(rank,)` per-unit vector column.
  - `read_place_field`: reconstruct the GAM fields (absent → `None`); preserve shapes. **Bump the encoding-model schema version again** (glm-diagnostics addition) — or extend phase-0's bump if phases ship together; document which.
- **Decoder — functional path** (`decoding/session.py`): thread `penalty`/`rank` through `decode_session`, `_build_encoding_model`, `_encode_and_bin` into `compute_spatial_rates(method=..., penalty=..., rank=...)`. Update the `Literal` cast at the encode step to include `"glm"`.
- **Decoder — class** (`decoding/estimator.py`): add `penalty: float | None = None`, `rank: int | None = None` fields to `BayesianDecoder`; make `bandwidth`/`min_occupancy` nullable; add the same method-specific validation as phase-3 in `__post_init__`; forward the glm params in `fit` (`:351-353`). Update the class docstring.
- **Docs:** CHANGELOG — glm usable through NWB + both decoder paths; nullable `bandwidth` in `BayesianDecoder`. Update any decoding QUICKSTART/docstring example that fixes `bandwidth`.

## Deliberately not in this phase

- **JAX** — phase-5.
- **Per-neuron λ** — phase-6.
- **Changing the ratio decode path** — only additive glm forwarding; ratio decoding is untouched.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_nwb_glm_roundtrip` | a glm `SpatialRatesResult` (with all GAM diagnostics, `bandwidth=None`) writes and reads back equal — field-by-field, shapes preserved. |
| `test_nwb_ratio_roundtrip_unchanged` | a `diffusion_kde` result round-trips with GAM fields absent/`None` (no regression from phase-0). |
| `test_nwb_reads_legacy_key` (carried from phase-0) | still green — old `"smoothing_method"` files read. |
| `test_decode_session_glm` | `decode_session(..., method="glm", penalty=None)` runs end-to-end and produces a valid `DecodingResult`. |
| `test_bayesian_decoder_glm` | `BayesianDecoder(method="glm", rank=100).fit(...).predict(...)` runs; `bandwidth`/`min_occupancy` nullable; method-specific validation rejects `bandwidth=5.0` + `method="glm"`. |
| `test_decoder_validation_mirrors_encoder` | the decoder's mutual-exclusivity/value-domain errors match `compute_spatial_rate`'s (same messages/branches). |

## Fixtures

- Reuse the phase-3 simulated glm result + the decoding fixtures (`tests/decoding/conftest.py`).
- A checked-in tiny NWB file with the legacy `"smoothing_method"` key (from phase-0) continues to back the legacy-read test. glm round-trip writes to a temp NWB in-test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- glm round-trips through NWB field-for-field; ratio round-trip unchanged; legacy read still works.
- Both decoder entry points (`decode_session`/`decode_session_summary` and `BayesianDecoder`) accept + forward glm params; validation mirrors the encoder.
- "Deliberately not in this phase" honored — no JAX, no `pooled`, ratio path untouched.
- Tests assert field-level round-trip equality and end-to-end decode, not smoke-only; fixtures shared.
- Schema-version bump documented; docstrings/tests carry no plan references; CHANGELOG updated.
