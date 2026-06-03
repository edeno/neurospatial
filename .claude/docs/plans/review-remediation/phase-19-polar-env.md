# Phase 19 — Promote the polar egocentric environment to a distinct type

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Convergence item C4** — the widest-blast-radius design change. Polar egocentric space is currently wedged into `Environment` via a hidden `coordinate_kind` flag whose `_check_cartesian` guard disables ~5 core methods at runtime, so the central noun silently means two geometries. This is also the root cause of the cm/rad unit-mixing (radians stored as edge "distance") and the silent polar→Cartesian NWB round-trip. This phase gives polar space its own type and fixes the geometry.

The correctness phases deliberately do **not** patch polar code paths (occupancy, factories edge-distance, encoding bandwidth) — those Theme-4 polar items are all consolidated here so the redesign fixes them once, in the right place.

## Decision (RESOLVED)

Resolved by the maintainer (2026-06-02). See [overview Open Questions #3](overview.md#open-questions).

1. **Type shape:** a **separate** `EgocentricPolarEnvironment` class that implements the shared environment protocol — **NOT a subclass of `Environment`**. This avoids the Liskov issues of inheriting (then forbidding) Cartesian-only methods at the cost of a bit more surface.
2. **Migration of `from_polar_egocentric`'s return type** ([factories.py:844](../../../../src/neurospatial/environment/factories.py#L844)) — it stays the public entry-point name and returns the new `EgocentricPolarEnvironment` type. Keep the existing public call site working (it may remain a module-level factory or move onto the new class — executor's choice, as long as `from_polar_egocentric(...)` keeps working).
3. **Geodesic on polar:** fix edge distances to proper polar lengths (arc `r·Δθ`, radial `Δr`, diagonal `sqrt(Δr² + (r·Δθ)²)`).

**Inputs to read first:**

- [environment/core.py:257](../../../../src/neurospatial/environment/core.py#L257) — `coordinate_kind` field and `is_polar` (core.py:934-944); the `_check_cartesian` guard that disables `bin_at`/`distance_between`/euclidean `distance_to`.
- [environment/factories.py:844](../../../../src/neurospatial/environment/factories.py#L844) — `from_polar_egocentric`; the radians-as-edge-distance construction the review flagged (~`factories.py:980-1002`).
- `io/nwb/_environment.py` — `coordinate_kind` persistence added in phase 9; this phase changes what the loaded object's *type* is.
- `encoding/egocentric.py` — the `gaussian_kde` cm/rad single-bandwidth mixing on a polar env (Theme-4 important), fixed here as part of giving polar its own geometry.

## Tasks

- Introduce a separate `EgocentricPolarEnvironment` class implementing the shared protocol (not a subclass of `Environment`). Move the polar-specific behavior (polar bin geometry, proper polar edge lengths, the methods that are meaningful in polar coordinates) onto it; remove the `coordinate_kind` flag + `_check_cartesian` runtime guards from `Environment` (the Cartesian type no longer needs to defend against being polar).
- Fix the edge-distance geometry to proper polar lengths (arc/radial/diagonal) so geodesic distance and density smoothing are physically correct; fix `bin_sizes` accordingly.
- Fix the egocentric `gaussian_kde` bandwidth so cm and radians are not collapsed into one scalar (separate radial/angular bandwidths, or document the required units).
- `from_polar_egocentric` returns the new type; update `io/nwb` so the round-trip restores the new type (not just a flag on `Environment`).
- Update every internal caller and type hint that assumed `Environment` could be polar (grep `coordinate_kind`, `is_polar`, `_check_cartesian`). Update CLAUDE.md's polar guidance and the user-guide docs. CHANGELOG entry describing the type change.

## Deliberately not in this phase

- The NWB `coordinate_kind` *flag* persistence — already added in phase 9 (this phase changes the restored *type*).
- General egocentric *ops* (bearing/distance transforms) that operate on arrays, not envs — unaffected.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_polar_env_is_distinct_type` | `from_polar_egocentric(...)` returns `EgocentricPolarEnvironment`, not `Environment`; Cartesian-only methods are absent/clearly errored, not silently disabled. |
| `test_polar_edge_distances_physical` | Two identical physical moves at different radii have equal arc length (fails before: ~5× difference). |
| `test_polar_geodesic_correct` | A known polar geodesic matches the analytic value within tolerance. |
| `test_polar_nwb_roundtrip_type` | NWB round-trip restores an `EgocentricPolarEnvironment` (integration, `pynwb`-gated). |

## Fixtures

Build a small polar egocentric env via `from_polar_egocentric` with known radii; analytic expected arc/geodesic values computed in-test.

## Review

Dispatch `code-reviewer`. Confirm: decision recorded; `coordinate_kind`/`_check_cartesian` runtime guards removed from `Environment`; polar edge geometry physically correct (the equal-arc test passes); all `coordinate_kind`/`is_polar` callers migrated (grep clean); NWB restores the new type; CLAUDE.md + user-guide updated; CHANGELOG updated; no plan/phase references in code/test names.
