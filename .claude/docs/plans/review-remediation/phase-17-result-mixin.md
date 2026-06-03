# Phase 17 — Unified result-object contract

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#result-object-contract)

**Convergence item C11.** Encoding `*RateResult` classes share `SpatialResultMixin`, but several user-facing results are bare dataclasses (`PlaceFieldsResult`, `DirectionalPlaceFields`) with no `.plot()`/`.compare()`/`.to_dataframe()`, so analysis-ending verbs ("compare them", "to a table") dead-end to manual NumPy. This phase introduces one uniform result surface and backfills the bare classes.

## Decision (RESOLVED)

Resolved by the maintainer (2026-06-02). See [overview Open Questions #2](overview.md#open-questions) and [shared-contracts.md Result-object contract](shared-contracts.md#result-object-contract).

1. **Method set** guaranteed by `ResultMixin`: `to_dataframe()`, `summary()`, and `plot()`. **`plot()` is OPTIONAL per result type** — the base may provide a default that raises `NotImplementedError`; not every result must implement `plot()`. `to_dataframe()` and `summary()` are expected on every result.
2. **Relationship to the existing `SpatialResultMixin`** ([encoding/_base.py:158](../../../../src/neurospatial/encoding/_base.py#L158)): `SpatialResultMixin` **re-parents under `ResultMixin`** (additive — all existing accessors preserved).

`ResultMixin` is a new base in `src/neurospatial/_results.py` that `SpatialResultMixin` comes to extend.

**Inputs to read first:**

- [encoding/_base.py:158](../../../../src/neurospatial/encoding/_base.py#L158) — existing `SpatialResultMixin` surface (do not remove its accessors).
- [encoding/spatial.py:87](../../../../src/neurospatial/encoding/spatial.py#L87) — `PlaceFieldsResult` (bare dataclass).
- [encoding/spatial.py:1712](../../../../src/neurospatial/encoding/spatial.py#L1712) — `DirectionalPlaceFields` (bare dataclass; the "compare them" dead-end).

## Tasks

- Add `src/neurospatial/_results.py` defining `ResultMixin` per [shared-contracts.md](shared-contracts.md#result-object-contract): `to_dataframe()` (tidy/long; `pandas` imported lazily), `summary()` (dict of scalar metrics), `plot(ax=None, **kwargs)` (returns the axis). **`plot()` is optional per result type** — provide a base default that `raise NotImplementedError` so a result without a meaningful plot is valid; results that can plot override it (returning the axis). Provide sensible default implementations where a uniform one exists; leave `to_dataframe()` abstract where each result must specialize.
- Re-parent `SpatialResultMixin` onto `ResultMixin` (additive; all current accessors preserved — verify by reading).
- Backfill `DirectionalPlaceFields`: add `.plot()` (per-direction overlay), `.correlation(label_a, label_b)` and/or a directionality index, `.to_dataframe()`.
- Backfill `PlaceFieldsResult` with the mixin surface.
- Add `.summary()` to the encoding/decoding result classes that lack it.
- Docstrings for every new method; CHANGELOG entry. **Additive only** — do not rename or remove existing attributes.

## Deliberately not in this phase

- `to_xarray()` — phase 20 (extends this mixin once it exists).
- Behavior-module result objects' `.plot()`/`.to_dataframe()` — fold in here only if cheap; otherwise note as a follow-up.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_directional_place_fields_plot` | `DirectionalPlaceFields().plot()` returns an Axes with one artist per direction (Agg backend). |
| `test_directional_place_fields_compare` | `.correlation(a, b)` returns the expected value on a fixture with a known relationship. |
| `test_result_to_dataframe_tidy` | `to_dataframe()` on two different result types `pd.concat` cleanly (tidy form). |
| `test_existing_accessors_preserved` | Pre-existing `SpatialRateResult.firing_rate`/`.occupancy`/`.spatial_information()` still work unchanged. |

## Fixtures

Reuse encoding `conftest` fixtures; synthesize a `DirectionalPlaceFields` with two directions whose correlation is known.

## Review

Dispatch `code-reviewer`. Confirm: decision recorded; mixin is additive (the preserved-accessors test passes); bare dataclasses now carry the surface; `pandas`/`matplotlib` imported lazily; no existing attribute renamed/removed; CHANGELOG updated; no plan/phase references in code/test names.
