# Phase 2 — Environment occupancy & indexing correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase fixes four silent-correctness / crash defects in `src/neurospatial/environment/`. The headline bug is **occupancy corruption on masked RegularGrids** (the common `Environment.from_samples` case): `occupancy(time_allocation='linear')` writes *full-grid* flat indices into an array indexed by *active-bin* id, so mass lands in the wrong bins (or out of bounds) whenever any bin is inactive. The other three are an interpolate reshape crash on holed grids, a rebin diagonal-connectivity misinference, and a self-contradictory error message for non-finite timestamps.

Scope is strictly `src/neurospatial/environment/` plus its regression tests. No public API signatures change; behavior changes only on inputs that were previously silently wrong or crashing.

**Inputs to read first:**

- [../../../../src/neurospatial/environment/trajectory.py:1116-1208](../../../../src/neurospatial/environment/trajectory.py) — `_allocate_time_linear`: builds `occupancy = np.zeros(self.n_bins)` (active-bin sized) but accumulates into `occupancy[bin_idx]` where `bin_idx` comes from `_position_to_flat_index` / `_compute_ray_grid_intersections` and is a **full-grid** flat index. The `if 0 <= bin_idx < self.n_bins` guards (lines ~1195, ~1202) only bound-check against the active count — they do **not** translate the index — so on a masked grid this writes the right total mass into the wrong active bins.
- [../../../../src/neurospatial/environment/trajectory.py:1312-1359](../../../../src/neurospatial/environment/trajectory.py) — `_position_to_flat_index`: returns the row-major **full-grid** flat index (`flat_idx += nd_index[dim] * stride`, `stride *= grid_shape[dim]`), with `-1` only for out-of-grid-bounds. It has no notion of `active_mask`; an inactive (holed) bin still returns a positive full-grid index. This is the index that must be mapped to active-bin id before accumulation.
- [../../../../src/neurospatial/environment/trajectory.py:1210-1310](../../../../src/neurospatial/environment/trajectory.py) — `_compute_ray_grid_intersections`: produces `list[(bin_idx, time_in_bin)]` where `bin_idx` is the `_position_to_flat_index` full-grid index. Confirms the full-grid indices flow up into `_allocate_time_linear` unchanged.
- [../../../../src/neurospatial/environment/trajectory.py:307-360](../../../../src/neurospatial/environment/trajectory.py) — `occupancy`: the `time_allocation == "start"` branch (lines ~343-354) correctly uses `bin_at(positions)` (active-bin ids) and `np.bincount(..., minlength=self.n_bins)`; only the `"linear"` branch (lines ~356-360) delegates to the broken helper. The two branches must agree bin-for-bin on a masked grid.
- [../../../../src/neurospatial/environment/trajectory.py:224-235](../../../../src/neurospatial/environment/trajectory.py) — `occupancy` timestamp validation: `np.all(np.diff(times) >= 0)`. With a NaN in `times`, every `>=` comparison is `False`, so `np.diff(times) < 0` is also `False`; `decreasing_indices` is empty, yielding the self-contradictory message "Found 0 decreasing interval(s) at indices: []".
- [../../../../src/neurospatial/environment/trajectory.py:540-548](../../../../src/neurospatial/environment/trajectory.py) — `_bin_sequence`: identical monotonicity check with the identical NaN failure mode. Both call sites need the finite guard.
- [../../../../src/neurospatial/environment/fields.py:528-557](../../../../src/neurospatial/environment/fields.py) — `_interpolate_linear`: already raises `NotImplementedError` for non-`RegularGrid` layouts, but then does `field.reshape(grid_shape)` where `field` is **active-bin** sized (length `n_bins`) and `grid_shape` is the **full** grid. On a holed grid `n_bins < prod(grid_shape)`, so `reshape` raises a bare `ValueError: cannot reshape array of size N into shape (...)`.
- [../../../../src/neurospatial/environment/transforms.py:256-290](../../../../src/neurospatial/environment/transforms.py) — `rebin`: infers `connect_diagonal` from `self.connectivity.degree(center_flat_idx)` where `center_flat_idx = center_node * grid_shape[1] + grid_shape[1] // 2` is a **full-grid** index, but `self.connectivity` is keyed by **active-bin** ids. On a masked grid this samples the wrong node (or a missing node → falls through to `connect_diagonal = True`), so the coarsened grid silently flips 4-conn vs 8-conn.
- [../../../../src/neurospatial/layout/helpers/regular_grid.py:692-717](../../../../src/neurospatial/layout/helpers/regular_grid.py) — the canonical full-grid→active-bin inverse-map idiom to reuse: `active_mask_flat = active_mask.ravel(); inverse_map = np.full(active_mask_flat.size, -1, np.intp); inverse_map[np.flatnonzero(active_mask_flat)] = np.arange(n_active)`. Active-bin ids are assigned in `np.flatnonzero(active_mask.ravel())` order — exactly the order `bin_centers = full_grid_bin_centers[active_mask.ravel()]` uses ([../../../../src/neurospatial/layout/engines/regular_grid.py:235](../../../../src/neurospatial/layout/engines/regular_grid.py)).

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — use `validate_finite(times, name="times")` (default `allow_nan=False`, so it raises on both NaN and Inf) for the finite-timestamp guard in both `occupancy` and `_bin_sequence`. Do **not** weaken to NaN-dropping: timestamps with NaN are an upstream error, not a convenience case. If `src/neurospatial/_validation.py` does not yet exist in the tree this phase lands on, add the `validate_finite` function exactly as specified in the contract (it is the same module phases 4–9 share); do not fork a private copy under `environment/`.

## Tasks

### Task 1 — Map full-grid flat indices to active-bin ids in linear time allocation (the headline fix)

The accumulation array in `_allocate_time_linear` is active-bin sized; the `bin_idx` values flowing out of `_compute_ray_grid_intersections` / `_position_to_flat_index` are full-grid flat indices. Translate them through the layout's `active_mask` before accumulating.

Build the inverse map **once** at the top of `_allocate_time_linear`, reusing the idiom from `layout/helpers/regular_grid.py:692-702`. When the grid is fully active (`active_mask` is `None` or all-True), the inverse map is the identity on valid indices, so the fix is a no-op there — preserving current behavior on unmasked grids.

Edit [../../../../src/neurospatial/environment/trajectory.py:1149-1208](../../../../src/neurospatial/environment/trajectory.py) (`_allocate_time_linear` body). Replace the section from the `# Initialize occupancy array` comment through the `return occupancy` with:

```python
        # Initialize occupancy array (indexed by ACTIVE-bin id, length self.n_bins)
        occupancy = np.zeros(self.n_bins, dtype=np.float64)

        # The ray-intersection helpers below (_position_to_flat_index /
        # _compute_ray_grid_intersections) return FULL-GRID row-major flat
        # indices, but `occupancy` is indexed by active-bin id. On a masked
        # grid (the common from_samples case) these differ, so we must
        # translate every full-grid index to its active-bin id before
        # accumulating. Build the inverse map once (same idiom as
        # layout/helpers/regular_grid.py). For inactive bins the map holds -1.
        active_mask = getattr(layout, "active_mask", None)
        if active_mask is None:
            # Fully-active grid: full-grid flat index == active-bin id.
            full_to_active = None
        else:
            active_mask_flat = active_mask.ravel()
            full_to_active = np.full(active_mask_flat.size, -1, dtype=np.intp)
            active_indices = np.flatnonzero(active_mask_flat)
            full_to_active[active_indices] = np.arange(
                active_indices.size, dtype=np.intp
            )

        def _to_active(full_idx: int) -> int:
            """Translate a full-grid flat index to active-bin id (-1 if inactive)."""
            if full_to_active is None:
                return full_idx
            if 0 <= full_idx < full_to_active.size:
                return int(full_to_active[full_idx])
            return -1

        # Process each valid interval
        for i in np.where(valid_mask)[0]:
            start_pos = positions[i]
            end_pos = positions[i + 1]
            interval_time = dt[i]

            # Choose weight based on return_seconds parameter
            weight = interval_time if return_seconds else 1.0

            # Get starting and ending bin indices (already active-bin ids,
            # from bin_at in occupancy()).
            start_bin = bin_indices[i]
            end_bin = bin_indices[i + 1]

            # If both points are in same bin, simple allocation
            if start_bin == end_bin and start_bin >= 0:
                occupancy[start_bin] += weight
                continue

            # Compute ray-grid intersections. These return FULL-GRID indices.
            bin_times = self._compute_ray_grid_intersections(
                start_pos, end_pos, list(grid_edges), grid_shape, interval_time
            )

            # Translate full-grid indices to active-bin ids and accumulate.
            if return_seconds:
                for full_idx, time_in_bin in bin_times:
                    active_idx = _to_active(full_idx)
                    if active_idx >= 0:
                        occupancy[active_idx] += time_in_bin
            else:
                # Convert time proportions to count proportions (sum to 1.0).
                # Mass that falls on inactive bins is dropped, matching the
                # 'start' branch's behaviour of excluding outside/inactive
                # samples, so the two allocation modes agree bin-for-bin.
                total_allocated = sum(time for _, time in bin_times)
                if total_allocated > 0:
                    for full_idx, time_in_bin in bin_times:
                        active_idx = _to_active(full_idx)
                        if active_idx >= 0:
                            occupancy[active_idx] += (
                                time_in_bin / total_allocated
                            ) * weight

        return occupancy
```

Notes for the executor:
- `layout` is already bound at the top of `_allocate_time_linear` via `layout = cast("RegularGridLayout", self.layout)` ([trajectory.py:1153](../../../../src/neurospatial/environment/trajectory.py)); `grid_edges`/`grid_shape` are already extracted just below it. Keep those lines; only the accumulation block changes.
- The old bound-check `if 0 <= bin_idx < self.n_bins` (lines ~1195, ~1202) is the removed code path — it compared a full-grid index against the active count and is fully superseded by `_to_active`. Do not leave it behind.
- Do **not** change `_position_to_flat_index` or `_compute_ray_grid_intersections` — they correctly compute full-grid geometry; the translation belongs at the accumulation seam so the ray walk stays in full-grid coordinates (where the grid edges live).
- Total occupancy on a *fully-active* grid is unchanged. On a masked grid, total occupancy now equals the `'start'`-branch total minus any mass the straight-line path spent crossing inactive (holed) bins — which is correct, since the animal cannot be in a bin that does not exist.

### Task 2 — Fix `_interpolate_linear` reshape on holed grids

`field` is active-bin sized; `grid_shape` is the full grid. On a holed grid `field.reshape(grid_shape)` crashes. Scatter the active values into a full-grid NaN array first, then reshape. The interpolator already uses `fill_value=np.nan` for out-of-bounds queries, so NaN at inactive cells is consistent: query points over a hole interpolate to NaN.

Edit [../../../../src/neurospatial/environment/fields.py:554-557](../../../../src/neurospatial/environment/fields.py). Replace:

```python
        # Reshape field to grid
        # Note: RegularGridLayout stores bin_centers in row-major order
        field_grid = field.reshape(grid_shape)
```

with:

```python
        # Scatter active-bin values into a full-grid array before reshaping.
        # `field` is indexed by active-bin id (length n_bins); `grid_shape`
        # is the FULL grid. On a holed/masked grid n_bins < prod(grid_shape),
        # so a direct reshape would raise. Inactive cells are filled with NaN,
        # which the RegularGridInterpolator (fill_value=np.nan) already treats
        # as "no data" — query points over a hole interpolate to NaN.
        n_full = int(np.prod(grid_shape))
        active_mask = getattr(self.layout, "active_mask", None)
        if active_mask is None or field.shape[0] == n_full:
            # Fully-active grid: field already covers every cell.
            field_grid = field.reshape(grid_shape)
        else:
            full_field = np.full(n_full, np.nan, dtype=np.float64)
            full_field[np.flatnonzero(active_mask.ravel())] = field
            field_grid = full_field.reshape(grid_shape)
```

Notes:
- `np` is already imported in `fields.py`; confirm and add `import numpy as np` to the module top only if absent (it is used throughout, so it should already be present).
- This keeps `mode='linear'` working on holed grids rather than raising. The existing `NotImplementedError` for non-`RegularGrid` layouts ([fields.py:530-536](../../../../src/neurospatial/environment/fields.py)) is unchanged.

### Task 3 — Fix `rebin` diagonal-connectivity inference to use an active-bin node

`rebin` samples a node's degree to decide 4-conn vs 8-conn, but indexes `self.connectivity` with a full-grid flat index. The connectivity graph is keyed by active-bin ids. Sample a node that is guaranteed present in the active graph, and infer from the **maximum** observed degree rather than one arbitrary node (a center node can sit on a hole boundary and under-report its degree).

Edit [../../../../src/neurospatial/environment/transforms.py:256-277](../../../../src/neurospatial/environment/transforms.py). Replace the block from `# --- Build new connectivity graph ---` through the `connect_diagonal = True` fallback:

```python
        # --- Build new connectivity graph ---

        # Infer whether the original grid used diagonal connections by looking
        # at the MAXIMUM node degree in the active connectivity graph. The
        # graph is keyed by ACTIVE-bin ids (0..n_bins-1), so we must not index
        # it with a full-grid flat index. Interior nodes have the full
        # neighbour count; boundary/hole-adjacent nodes have fewer, so the max
        # over all active nodes is the robust signal:
        #   2D: 4-conn -> max degree 4, 8-conn -> max degree 8
        #   3D: 6-conn -> max degree 6, 26-conn -> max degree 26
        if self.connectivity.number_of_nodes() > 0:
            max_degree = max(
                (deg for _, deg in self.connectivity.degree()), default=0
            )
            connect_diagonal = max_degree > 2 * n_dims
        else:
            # Empty graph (degenerate): default to the common case.
            connect_diagonal = True
```

Notes:
- This removes the `center_node` / `center_flat_idx` full-grid index computation ([transforms.py:258-276](../../../../src/neurospatial/environment/transforms.py)) entirely — name it as removed in the PR description.
- `self.connectivity.degree()` returns `(node, degree)` pairs; `max(..., default=0)` guards the empty graph. `n_dims` is already bound earlier in `rebin`.
- Behavior on a fully-active grid is unchanged for any grid with at least one interior node (the previous heuristic also keyed off an interior node when the full-grid index happened to coincide with an active id, which it does on unmasked grids). The masked-grid case is the one that was wrong.

### Task 4 — Explicit non-finite timestamp guard in `occupancy` and `_bin_sequence`

Add a `validate_finite` check on `times` **before** the monotonicity check in both methods, so a NaN/Inf timestamp produces a clear "times contains N non-finite value(s)" error instead of "Found 0 decreasing interval(s) at indices: []".

In `occupancy`, edit [../../../../src/neurospatial/environment/trajectory.py:224-228](../../../../src/neurospatial/environment/trajectory.py). After `times = np.asarray(times, dtype=np.float64)` / `positions = np.asarray(...)` and **before** the `# Validate monotonicity of timestamps` block, insert:

```python
        # Reject non-finite timestamps up front. Without this, a NaN makes
        # every np.diff comparison False, so the monotonicity check below
        # would raise the self-contradictory "0 decreasing interval(s)".
        from neurospatial._validation import validate_finite

        times = validate_finite(times, name="times")
```

In `_bin_sequence`, edit [../../../../src/neurospatial/environment/trajectory.py:515-541](../../../../src/neurospatial/environment/trajectory.py). After `times = np.asarray(times, dtype=np.float64)` and **before** the `# Check for monotonic timestamps` block (it can go right after the `np.asarray` calls near line 516), insert the identical guard:

```python
        # Reject non-finite timestamps up front (see occupancy() for rationale).
        from neurospatial._validation import validate_finite

        times = validate_finite(times, name="times")
```

Notes:
- **Default to a single module-level `from neurospatial._validation import validate_finite` import at the top of `trajectory.py`**, and drop the inline `from ... import` lines from both bodies. `_validation.py` imports only numpy, so there is no circular-import risk — the module-level import is the intended form, not a fallback. (The inline snippets above show the call site for clarity; in the implementation the import is hoisted.) Create `src/neurospatial/_validation.py` per the contract verbatim if it is not yet present in the tree (see the Contracts section above).
- `validate_finite` returns the float64-coerced array; reassigning `times` is harmless and keeps the downstream code identical.
- Do **not** add the guard to `positions` here — that is a broader input-validation change out of scope for this phase (NaN positions already map to `bin_at == -1` and are filtered).

### Task 5 — Docstring updates for the two behavior-clarifying fixes

- `occupancy` docstring ([../../../../src/neurospatial/environment/trajectory.py:104-222](../../../../src/neurospatial/environment/trajectory.py)): in the **Mass conservation** Notes section (lines ~185-195), add a sentence that for `time_allocation='linear'` on a masked grid, time the straight-line path spends crossing **inactive** bins is dropped (not reassigned), so the linear total may be slightly below the start-allocation total by exactly that crossing time. Add a `Raises` entry noting that non-finite `times` raise `ValueError` naming `times`.
- `_interpolate_linear` docstring ([../../../../src/neurospatial/environment/fields.py:508-526](../../../../src/neurospatial/environment/fields.py)): note in the Returns/Notes that on holed grids, query points over inactive cells return NaN (consistent with out-of-bounds `fill_value`).
- No public-API signature changes, README, or QUICKSTART edits are needed (occupancy/interpolate/rebin signatures are unchanged). Phase 23 owns runnable-doc CI.

## Deliberately not in this phase

- **Polar / egocentric environment factories** (`from_polar_egocentric`, `factories.py`/`core.py` polar paths) — phase 19. They live in `environment/` but are a separate defect cluster; touching them here would balloon the diff and the review.
- **`min_occupancy` → NaN reconciliation with `decode_position`** — phase 16 (which explicitly depends on this occupancy fix landing first; see overview "Sequencing"). This phase only fixes *where* occupancy mass lands, not how downstream rate computation fills empty bins.
- **General `positions` finite-validation / a shared trajectory input-validator** — only the `times` finite guard ships here, because it is what produces the self-contradictory error. Broader `ops/`-level input validation is phase 8.
- **Refactoring `_position_to_flat_index` to return active-bin ids directly** — intentionally left full-grid, because the ray walk needs full-grid geometry; the translation belongs at the accumulation seam (Task 1).

## Validation slice

Each fix gets a fail-before / pass-after regression test. Put them in `tests/test_environment.py` (or `tests/environment/test_trajectory.py` if that split exists — match the repo's current layout; do not create a new top-level test module if one already covers these methods).

| Test | Asserts |
| --- | --- |
| `test_occupancy_linear_masked_grid_mass_conservation` | On a holed grid (Fixture A), `occupancy(times, positions, time_allocation='linear')` returns an array of length `env.n_bins`, all mass sits in **active** bins (no out-of-bounds write, no IndexError), and `occ.sum()` equals the total valid interval time minus the time the straight-line path spends over the hole (compute the expected hole-crossing time from the fixture geometry). **Fails before** (mass scattered to wrong active ids / wrong total). |
| `test_occupancy_linear_matches_start_on_fully_active_grid` | On an **all-active** regular grid, `time_allocation='linear'` and `'start'` agree on total mass (`occ_linear.sum() == approx(occ_start.sum())`) and the fix is a no-op (regression guard that Task 1 didn't change unmasked behavior). |
| `test_occupancy_linear_peak_bin_is_where_animal_was` | On Fixture A, the bin with the most linear-allocated occupancy is the active bin containing the densest dwell region — i.e. the index translation is correct, not just the total. **Fails before** (peak lands on a mis-translated bin). |
| `test_occupancy_nonfinite_times_raises_clear_error` | `occupancy(times_with_nan, positions)` raises `ValueError` whose message contains `"times"` and `"non-finite"` (not `"0 decreasing interval(s)"`). Parametrize over NaN and `+inf`. **Fails before** (raises the self-contradictory monotonicity message for NaN). |
| `test_bin_sequence_nonfinite_times_raises_clear_error` | Same assertion for `env.bin_sequence(times_with_nan, positions)`. **Fails before.** |
| `test_interpolate_linear_holed_grid_returns_nan_over_hole` | On Fixture A, `env.interpolate(field, query_points, mode='linear')` returns finite values at points over active bins and `NaN` at a query point centered on the hole — and does **not** raise. **Fails before** (`ValueError: cannot reshape array of size ...`). |
| `test_interpolate_linear_fully_active_unchanged` | On an all-active grid, `mode='linear'` results are identical (`np.allclose`) before/after the scatter change (regression guard). |
| `test_rebin_preserves_diagonal_connectivity_on_masked_grid` | Build a holed 8-connected grid (Fixture A built with `connect_diagonal_neighbors=True`), `rebin` it, and assert the coarsened connectivity is also 8-connected (some interior node has degree 8). Then repeat with a 4-connected source and assert the result is 4-connected. **Fails before** (full-grid-index sampling mis-detects connectivity on the masked grid). |

Mark none of these `slow` — they all run on tiny synthetic grids.

## Fixtures

**Fixture A — masked/holed regular grid (`conftest.py`).** A `pytest` fixture that builds an `Environment.from_samples`-style masked `RegularGrid` with a known interior hole, so `n_bins < prod(grid_shape)` and active-bin ids differ from full-grid flat indices.

Construct it by sampling positions on a grid with a square punched out of the middle, then:

```python
import numpy as np
import pytest
from neurospatial import Environment


@pytest.fixture
def holed_grid_env():
    """A 2D RegularGrid with an interior hole (inactive bins).

    Positions densely cover a 20x20 cm square EXCEPT a 6x6 cm hole in the
    middle, so `from_samples` infers inactive bins there. With bin_size=2.0
    this yields a 10x10 full grid (100 cells) but fewer active bins, so
    active-bin ids != full-grid flat indices — the exact condition that
    exposes the occupancy/interpolate/rebin index bugs.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 20.0, size=(4000, 2))
    # Punch a hole: drop samples in [7, 13] x [7, 13].
    hole = (
        (pts[:, 0] >= 7.0)
        & (pts[:, 0] <= 13.0)
        & (pts[:, 1] >= 7.0)
        & (pts[:, 1] <= 13.0)
    )
    pts = pts[~hole]
    env = Environment.from_samples(pts, bin_size=2.0, bin_count_threshold=1)
    assert env.n_bins < np.prod(env.layout.grid_shape), (
        "fixture must have inactive bins to exercise the index mapping"
    )
    return env
```

For the rebin connectivity tests, build a second variant (or parametrize) that passes `connect_diagonal_neighbors=True` through the factory so the source graph is 8-connected; verify via the factory/layout API how diagonal connectivity is requested and assert the source has a degree-8 node before rebinning. For the timestamp tests, derive `times`/`positions` from the fixture's active bin centers (a short straight trajectory through the densest region) and inject a NaN/Inf into a copy of `times`.

The fully-active comparison tests use a plain `Environment.from_samples` over a gap-free square (no fixture needed, or a sibling `full_grid_env` fixture).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- All five tasks are implemented as specified; the index translation in Task 1 reuses the `active_mask`/`flatnonzero` idiom (not a hand-rolled variant) and is a verified no-op on fully-active grids.
- The "Deliberately not in this phase" list is honored — no drift into polar factories (phase 19), `min_occupancy` fill (phase 16), or broad `positions` validation (phase 8).
- Validation slice passes: `uv run pytest tests/test_environment.py -q` (or the matching trajectory/fields test module) — each regression test fails on `main` and passes on the branch. Spot-check the mass-conservation test actually fails before the fix.
- Tests aren't trivial — the mass-conservation test asserts the *expected hole-crossing-adjusted* total, not just "doesn't raise"; the peak-bin test asserts the translation, not only the sum. Shared grid setup is in `conftest.py` fixtures, not copy-pasted. (`testing-anti-patterns` covers the failure modes.)
- The removed code paths are actually gone: the old `if 0 <= bin_idx < self.n_bins` accumulation guards in `_allocate_time_linear`, and the `center_node`/`center_flat_idx` block in `rebin`. No orphans.
- Docstrings (Task 5) are updated, and no docstring, test name, or module name references this plan, "phase 2", or any milestone tag.
- `uv run ruff check src/neurospatial/environment/ && uv run ruff format --check src/neurospatial/environment/` and `uv run mypy src/neurospatial/environment/` are clean.
