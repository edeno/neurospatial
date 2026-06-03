# Phase 20 — xarray output adapters & decode-error alignment

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#result-object-contract)

Adds `to_xarray()` to the array-shaped results and an internalized decode→error alignment (DESIGN-REVIEW Med). xarray is the de-facto container in modern decoding pipelines, and the decode-grid → ground-truth-position alignment is recurring hand-rolled `searchsorted` glue in the journeys.

**Depends on:** phase 17 (`ResultMixin`) — `to_xarray()` attaches to that mixin/its results. `xarray` stays an **optional** dependency (lazy import inside the method).

**Inputs to read first:**

- [shared-contracts.md Result-object contract](shared-contracts.md#result-object-contract) — the mixin `to_xarray()` extends.
- The `DecodingResult` class (decoding module — read its `posterior`/`times`/`env` attributes for the coord construction) and [encoding/spatial.py](../../../../src/neurospatial/encoding/spatial.py) `SpatialRatesResult`.
- `decoding/metrics.py` — the existing `decoding_error` and how callers currently align decoded vs. true positions.

## Tasks

- Add `to_xarray()` to `DecodingResult` returning a `DataArray`/`Dataset` with dims `("time", "bin")` (coords: `time` from `result.times`, `bin` from bin indices or centers). Import `xarray` lazily inside the method; raise an actionable `ImportError` ("install neurospatial[xarray] or xarray") if absent.
- Handle `DecodingResult.times is None` (its default): `to_xarray()` must **not** crash when `times` is unset. Fall back to an integer time index — `time = np.arange(n_time)` — for the `time` coord, rather than passing `None` into the coord (which would raise). Document this fallback in the method docstring (the `time` coord is a positional integer index when `result.times` is `None`).
- Add `to_xarray()` to `SpatialRatesResult` with dims `("neuron", "bin")`.
- Add `DecodingResult.error_against(true_times, true_positions, *, metric="euclidean")` that internally aligns the decode grid to the provided ground truth (via `searchsorted`/interpolation) and returns the per-time error — removing the hand-rolled alignment from user code. Reuse the existing `decoding_error` core.
- Add `xarray` to an optional dependency group in `pyproject.toml` (`[project.optional-dependencies] xarray = ["xarray"]`); do **not** make it a hard dependency.
- Docstrings + runnable Examples (registered in phase 23). CHANGELOG entry.

## Deliberately not in this phase

- Internal use of xarray — this is output-only; no internal arrays become `DataArray`s.
- The `decoding_error` signature/arg-order fix — phase 22.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_decoding_result_to_xarray_dims` | Returns dims `("time","bin")` with coords matching `result.times`; values equal `result.posterior`. |
| `test_to_xarray_times_none` | When `result.times is None`, `to_xarray()` does not crash and the `time` coord is the integer index `np.arange(n_time)` (not `None`); `bin` coord and `posterior` values are still correct. |
| `test_spatial_rates_to_xarray_dims` | Dims `("neuron","bin")`; values equal the firing-rate matrix. |
| `test_to_xarray_without_xarray_raises` | Patching the import to fail raises a clear `ImportError`. |
| `test_error_against_matches_manual` | `error_against(...)` equals a hand-aligned `decoding_error` on a fixture (fails-equivalently before existing). |

## Fixtures

Reuse decoding `conftest`; gate xarray tests with `pytest.importorskip("xarray")` except the missing-import test which patches it out.

## Review

Dispatch `code-reviewer`. Confirm: `xarray` imported lazily and optional (not in core deps); dims/coords correct; `error_against` matches manual alignment; missing-dependency path tested; CHANGELOG + optional-deps documented; no plan/phase references in code/test names.
