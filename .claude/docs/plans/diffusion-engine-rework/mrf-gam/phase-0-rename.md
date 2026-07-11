# Phase 0 — Hard-rename `smoothing_method → method` (behavior-preserving)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md#method-param)

Rename `smoothing_method → method` on **every** smoothing encoder, result class, and downstream
consumer, so the estimator axis has one uniform name before glm is added. No new estimator, no
numerical change. This is a breaking API rename (no alias, per project policy) plus an NWB
metadata-key change that still reads old files.

**Inputs to read first:**

- [shared-contracts.md#method-param](shared-contracts.md#method-param) — the target `method` param + value set per encoder (glm only on `compute_spatial_rate(s)`; others stay 3-valued).
- [overview.md](overview.md) — "Current codebase integration points" lists every site with line refs.
- `src/neurospatial/encoding/_smoothing.py:544` — `_validate_smoothing_parameters(method, bandwidth)` already uses `method` internally; the encoders just stop translating the name.

**Contracts referenced:**

- [The `method` parameter](shared-contracts.md#method-param) — this phase renames the param + fields but does **not** add glm values or method-specific validation (that's phase-3). `bandwidth` stays `float` (widened to `float | None` in phase-3).

## Tasks

- **Encoders — rename the public param `smoothing_method → method`** (keep each encoder's existing `Literal` value set and default; **egocentric default stays `"binned"`**), and update internal forwards:
  - `src/neurospatial/encoding/spatial.py`: `compute_spatial_rate` (`:1894`, param `:1900`), `compute_spatial_rates` (`:2237`, param `:2243`), `detect_place_fields` (`:3320`), `is_place_cell` (`:3507`), `compute_directional_place_fields` (`:3166`). Where these forward into `compute_spatial_rate`, pass `method=`. Fix the internal `method=smoothing_method` echo at `:2209,:2654` (now just `method=method`).
  - `src/neurospatial/encoding/view.py`: `compute_view_rate` (`:1095`), `compute_view_rates` (`:1372`), the helper at `:1743`, and internal forwards `:1367,:1678,:1723,:1816`.
  - `src/neurospatial/encoding/egocentric.py`: `compute_egocentric_rate` (`:1311`), `compute_egocentric_rates` (`:1599`), helper `:1257`, forward `:1575`.
- **Result classes — rename the field `smoothing_method → method`** on all four, and every reader (`to_dataframe`, `summary`, `summary_table`, `plot`, `__repr__`, docstrings, `to_xarray` attrs):
  - `SpatialRateResult` (`spatial.py:300`), `SpatialRatesResult` (`:955`), `ViewRateResult` (`view.py:216`), `ViewRatesResult` (`:544`), `EgocentricRateResult` (`egocentric.py:108`), `EgocentricRatesResult` (`:508`). Update the constructor call sites that pass `smoothing_method=` (e.g. `spatial.py:1018,2232,2689`).
- **NWB** (`src/neurospatial/io/nwb/_fields.py`):
  - `write_spatial_rates` (`:434`): write metadata key `"method"` instead of `"smoothing_method"` (`:616`); read `result.method`.
  - `read_place_field` (`:697`): read `"method"` with a **defensive fallback** to `"smoothing_method"` (`:782-792`) so old files still load; construct `SpatialRatesResult(method=...)`.
  - **Bump the encoding-model schema version** (find the version constant this module writes; increment it). Document the key rename + fallback in the module docstring.
- **Decoder** (`src/neurospatial/decoding/`):
  - `session.py`: `decode_session` param (`:100`), `_build_encoding_model` param (`:356`), `_encode_and_bin` param (`:574`), and the `compute_spatial_rates(smoothing_method=…)` forwards at `:323,:519,:609`. Update the `cast("Literal[...]", smoothing_method)` at `:507` to the renamed local.
  - `estimator.py`: `BayesianDecoder.smoothing_method` field (`:152`) → `method`; the `compute_spatial_rates(smoothing_method=self.smoothing_method)` forward (`:352`); docstring (`:69`) and the `__post_init__` validation note (`:168`).
- **Incidental:** `src/neurospatial/simulation/validation.py` (param `:27`, forwards `:248,:674`); `src/neurospatial/__init__.py` doctest (`:155`).
- **Tests:** update every `smoothing_method=` / `.smoothing_method` in `tests/` (grep-complete; `test_encoding_spatial.py`, `test_encoding_view.py`, `test_compute_egocentric_rate(s).py`, `test_decode_session.py`, `test_estimator.py`, `test_fields_roundtrip.py`, `test_fields.py`, the jax/backend dispatch tests, etc.) to `method=` / `.method`.
- **Docs:** CHANGELOG entry — the hard rename across all smoothing encoders + result classes (breaking), the NWB key change with read-fallback, and a note that `glm` (spatial-only) lands in a later phase. Update `CLAUDE.md` / QUICKSTART snippets that show `smoothing_method=` (grep `.claude/` and `README`/docs for the old kwarg).

## Deliberately not in this phase

- **Any glm logic** — no `"glm"` in any `Literal`, no `penalty`/`rank` params, no GAM result fields, no method-specific validation. Phases 1–3.
- **Widening `bandwidth` to `float | None`** — stays `float`; phase-3 widens it when glm produces `None`.
- **NWB GAM-diagnostic columns** — phase-4.

## Validation slice

| Test | Asserts |
| --- | --- |
| full existing suite (`uv run pytest -m "not slow and not napari"`) | green after mechanical rename — no numerical behavior change. |
| `test_method_kwarg_each_encoder` | `compute_spatial_rate/rates`, `compute_view_rate/rates`, `compute_egocentric_rate/rates`, `detect_place_fields`, `is_place_cell` accept `method=` and produce identical output to the pre-rename baseline (parametrized). |
| `test_old_kwarg_rejected` | passing `smoothing_method=` to any renamed encoder raises `TypeError` (unexpected keyword) — the alias is gone. |
| `test_result_field_renamed` | each result class exposes `.method`, not `.smoothing_method`; `to_dataframe`/`summary_table` carry the `method` value. |
| `test_nwb_reads_legacy_key` | an NWB file written with the old `"smoothing_method"` metadata key still reads (fallback path); a freshly written file uses `"method"` and round-trips. |
| `test_egocentric_default_preserved` | `compute_egocentric_rate` default `method` is `"binned"` (not `"diffusion_kde"`). |

## Fixtures

- Reuse existing encoding/decoding fixtures (`conftest.py`); only kwarg names change.
- **Baseline capture** for `test_method_kwarg_each_encoder`: since this is a behavior-preserving rename, capture each encoder's output on a small fixture on `main` (pre-rename) and assert equality after — or assert the two methods agree within exact tolerance in-test by constructing the same inputs. A checked-in NWB file with the legacy key (tiny, synthesized once) backs `test_nwb_reads_legacy_key`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Every task in this phase is implemented as specified; the grep for `smoothing_method` across `src/` returns **only** the NWB read-fallback string literal.
- The "Deliberately not in this phase" list is honored — no `"glm"`, no `penalty`/`rank`, no field widening.
- Validation slice tests pass; no numerical output changed vs baseline.
- Tests aren't trivial — the rename tests assert output equality / kwarg rejection, not just attribute presence. Shared setup is in fixtures.
- Docstrings, test names, and module names don't reference this plan or its phases.
- CHANGELOG + doc snippets updated, not deferred.
