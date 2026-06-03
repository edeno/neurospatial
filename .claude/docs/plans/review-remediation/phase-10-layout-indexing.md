# Phase 10 — Layout: indexing & axis-order correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase fixes five silent-corruption / silent-fallback bugs in the layout
engines and helpers. All are correctness bugs that produce *plausible but wrong*
bin lookups or bin counts — they do not raise, so downstream place fields and
decoders are quietly corrupted. Scope is strictly `src/neurospatial/layout/`
(with one one-line forwarding-key update in `environment/factories.py`, called
out explicitly below).

**Inputs to read first:**

- [../../../../src/neurospatial/layout/engines/image_mask.py:130](../../../../src/neurospatial/layout/engines/image_mask.py) — `build()` constructs `grid_edges`/`bin_centers`/`active_mask` in **(y, x)** order (lines 130–157) while `dimension_ranges` is **(x, y)** (lines 133–136). The inherited `point_to_bin_index` (see mixins below) digitizes `points[:, i]` against `grid_edges[i]`, so an `(x, y)` query point is digitized against `y_edges` — wrong for non-square pixels. The `build` parameter is named `bin_size` (line 62) while the public factory exposes `pixel_size`.
- [../../../../src/neurospatial/layout/mixins.py:239](../../../../src/neurospatial/layout/mixins.py) — `_GridMixin.point_to_bin_index` (lines 239–268) forwards `grid_edges`, `grid_shape`, `active_mask` into `_points_to_regular_grid_bin_ind`. This is the consumer that makes the ImageMask axis order load-bearing. **Do not change this method** — fix the producer (ImageMask `build`) so its arrays match the (x, y) contract the mixin already assumes.
- [../../../../src/neurospatial/layout/helpers/regular_grid.py:490](../../../../src/neurospatial/layout/helpers/regular_grid.py) — canonical (x, y) construction to copy: edges per dimension, `centers_per_dim = [get_centers(e) ...]`, `meshgrid(*centers_per_dim, indexing="ij")`, `stack(..., axis=-1)`; `grid_shape` is the per-dimension center counts in coordinate order (lines 515–521).
- [../../../../src/neurospatial/layout/engines/masked_grid.py:89](../../../../src/neurospatial/layout/engines/masked_grid.py) — `build()` assigns `self.active_mask = active_mask` (line 89) with **no dtype validation**; a float/int mask flows into `full_grid_bin_centers[self.active_mask.ravel()]` (line 119) and the connectivity helper, silently mis-selecting bins (fancy-vs-boolean indexing).
- [../../../../src/neurospatial/layout/helpers/regular_grid.py:638](../../../../src/neurospatial/layout/helpers/regular_grid.py) — `_points_to_regular_grid_bin_ind` (lines 638–648): on a dimensionality mismatch between `points`, `grid_edges`, and `grid_shape` it `warnings.warn(...)` and returns all `-1` instead of raising. A caller passing 3-D points to a 2-D grid gets a silent "no bins found" result.
- [../../../../src/neurospatial/layout/engines/graph.py:365](../../../../src/neurospatial/layout/engines/graph.py) — `GraphLayout.linear_point_to_bin_ind` (lines 365–395) promises indices "relative to the set of *active* 1D bins" but returns whatever `_find_bin_for_linear_position` returns.
- [../../../../src/neurospatial/layout/helpers/graph.py:440](../../../../src/neurospatial/layout/helpers/graph.py) — `_find_bin_for_linear_position` (lines 440–516) returns indices "relative to the **full** set of bins defined by `bin_edges_1d`" (gap-inclusive) — see its Returns docstring at lines 470–474. With gaps present these differ from active-bin indices, so the public method's return values are off by the number of preceding gap bins.
- [../../../../src/neurospatial/layout/helpers/utils.py:121](../../../../src/neurospatial/layout/helpers/utils.py) — `get_n_bins` computes `np.ceil(extent / bin_size_arr).astype(np.int32)` (line 121). For a large extent / small bin_size the int32 cast silently overflows (wraps negative or to a small/garbage count); the result is then re-cast to int64 at line 125, locking in the corrupted value.

**Contracts referenced:** none. (This phase raises plain `ValueError`/`TypeError` from
within `layout/`; it does **not** consume the shared `_validation.py` helpers.)

## Tasks

### Task 1 — ImageMaskLayout: (x, y) axis order + `pixel_size` parameter name

Rewrite the geometry-construction tail of
`ImageMaskLayout.build` (`engines/image_mask.py`, lines 123–166) so that
`grid_edges`, `grid_shape`, `bin_centers`, and `active_mask` are all in
**(x, y)** order — matching every other grid engine and the `(x, y)`
`dimension_ranges` the method already produces. Also rename the build parameter
`bin_size` → `pixel_size` (keep accepting the legacy `bin_size` key as a
deprecated alias so the factory and any external callers do not break).

Replace the signature and body from line 57 onward. New signature:

```python
    @capture_build_params
    def build(
        self,
        *,
        image_mask: NDArray[np.bool_],  # Defines candidate pixels, shape (n_rows, n_cols)
        pixel_size: float | tuple[float, float] | None = None,
        connect_diagonal_neighbors: bool = True,
        bin_size: float | tuple[float, float] | None = None,  # deprecated alias
    ) -> None:
```

Resolve the alias at the top of the body, before any validation:

```python
        # Resolve the pixel-size argument. ``pixel_size`` is the public name;
        # ``bin_size`` is accepted as a deprecated alias for backward
        # compatibility with callers that forwarded the legacy key.
        if pixel_size is not None and bin_size is not None:
            raise ValueError(
                "Pass either 'pixel_size' or the deprecated 'bin_size' alias, "
                "not both."
            )
        if pixel_size is None:
            if bin_size is None:
                pixel_size = 1.0  # one unit per pixel
            else:
                warnings.warn(
                    "'bin_size' is deprecated for ImageMaskLayout; use "
                    "'pixel_size' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                pixel_size = bin_size
```

(Add `import warnings` to the module imports.)

Then validate `image_mask` and `pixel_size` exactly as before but referencing
`pixel_size` (lines 87–121 of the original become identical checks on
`pixel_size`; keep the `TypeError` for non-array `image_mask`, the `ndim != 2`
check, the boolean-dtype check, the positivity checks, the
"at least one True" check). The per-component resolution becomes:

```python
        # Determine pixel sizes for x and y (units per pixel)
        pixel_size_x: float
        pixel_size_y: float
        if isinstance(pixel_size, (float, int, np.number)):
            pixel_size_x = float(pixel_size)
            pixel_size_y = float(pixel_size)
        elif (
            isinstance(pixel_size, (list, tuple, np.ndarray)) and len(pixel_size) == 2
        ):
            pixel_size_x = float(pixel_size[0])  # width of a pixel (x)
            pixel_size_y = float(pixel_size[1])  # height of a pixel (y)
        else:
            raise ValueError(
                "pixel_size for ImageMaskLayout must be a float or a "
                "2-element sequence (width, height).",
            )
        if pixel_size_x <= 0 or pixel_size_y <= 0:
            raise ValueError("pixel_size components must be positive.")
```

Now build the grid in (x, y) order. The input `image_mask` is indexed
`[row, col]` = `[y, x]`; the engine's coordinate convention is (x, y), so the
**x axis is dimension 0**. `grid_shape`, `grid_edges`, and `bin_centers` are all
expressed in (x, y); the mask therefore has to be transposed to `(n_cols, n_rows)`
so its ravel aligns with the (x, y) `meshgrid(..., indexing="ij")` ordering:

```python
        n_rows, n_cols = image_mask.shape  # rows = y, cols = x

        # Coordinate convention: dimension 0 is x (cols), dimension 1 is y (rows).
        # grid_shape, grid_edges, bin_centers, and active_mask are all in (x, y)
        # order so that point_to_bin_index digitizes points[:, 0] against x_edges
        # and points[:, 1] against y_edges (matching dimension_ranges and every
        # other grid engine).
        self.grid_shape = (n_cols, n_rows)

        # Safety check: warn or error if grid is very large
        n_dims = 2  # ImageMask is always 2D
        check_grid_size_safety(self.grid_shape, n_dims)

        x_edges = np.arange(n_cols + 1) * pixel_size_x
        y_edges = np.arange(n_rows + 1) * pixel_size_y
        self.grid_edges = (x_edges, y_edges)
        self.dimension_ranges = (
            (x_edges[0], x_edges[-1]),
            (y_edges[0], y_edges[-1]),
        )

        x_centers = (np.arange(n_cols) + 0.5) * pixel_size_x
        y_centers = (np.arange(n_rows) + 0.5) * pixel_size_y
        # indexing="ij" over (x_centers, y_centers): outer loop x, inner loop y,
        # giving the same x-major ravel order as grid_shape == (n_cols, n_rows).
        xv, yv = np.meshgrid(x_centers, y_centers, indexing="ij")
        full_grid_bin_centers = np.stack((xv.ravel(), yv.ravel()), axis=1)

        # image_mask is (rows, cols) == (y, x); transpose to (x, y) so its ravel
        # aligns row-for-row with full_grid_bin_centers above.
        self.active_mask = np.ascontiguousarray(image_mask.T)
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(self.connectivity, n_dims=2)
```

Update the `build` docstring (Parameters / Raises) to document `pixel_size`
(and the deprecated `bin_size` alias), and note the (x, y) convention.

**Factory forwarding (the one allowed `environment/` edit).** Update
`environment/factories.py` line 689 so `from_pixel_mask` forwards the public key
name instead of the legacy one:

```python
        layout_params = {
            "image_mask": image_mask,
            "pixel_size": pixel_size,
            "connect_diagonal_neighbors": connect_diagonal_neighbors,
        }
```

and delete the now-stale "ImageMaskLayout still uses the legacy 'bin_size' key"
comment (lines 686–688). Verify no other internal caller passes `bin_size=` to
the ImageMask engine: `grep -rn "ImageMask" src/neurospatial | grep -i bin_size`
must come back empty after this change.

### Task 2 — MaskedGridLayout: validate `active_mask` dtype

In `MaskedGridLayout.build` (`engines/masked_grid.py`), before assigning
`self.active_mask` (line 89), reject non-boolean masks so int/float arrays cannot
silently turn boolean indexing into fancy indexing:

```python
        if not isinstance(active_mask, np.ndarray):
            raise TypeError(
                f"active_mask must be a NumPy array, got {type(active_mask).__name__}."
            )
        if active_mask.dtype != np.bool_:
            raise ValueError(
                f"active_mask must have boolean dtype (np.bool_), got "
                f"{active_mask.dtype}. Convert with `mask.astype(bool)` — but be "
                f"sure the values are genuine True/False flags, not bin data."
            )
        self.active_mask = active_mask
```

Update the build docstring's Raises section to list the new `TypeError` and the
`ValueError` for non-boolean dtype.

### Task 3 — `_points_to_regular_grid_bin_ind`: raise on dimensionality mismatch

In `helpers/regular_grid.py`, replace the warn-and-return-`-1` block (lines
638–648) with an explicit `ValueError`:

```python
    n_dims = valid_points.shape[1]
    if n_dims != len(grid_edges) or n_dims != len(grid_shape):
        raise ValueError(
            f"Dimensionality mismatch: points have {n_dims} dimension(s), but "
            f"grid_edges has {len(grid_edges)} and grid_shape has "
            f"{len(grid_shape)}. Points must match the grid's dimensionality."
        )
```

The `warnings` import in this module is still used elsewhere — confirm with
`grep -n "warnings\." src/neurospatial/layout/helpers/regular_grid.py`; only
remove the import if this was its last use (it is not — leave it).

### Task 4 — GraphLayout.linear_point_to_bin_ind: return active-bin indices

`_find_bin_for_linear_position` returns **full-grid** (gap-inclusive) indices.
`GraphLayout.linear_point_to_bin_ind` must convert those to **active-bin**
indices (the contract its docstring already promises) before returning. Build the
full→active remap from `self.active_mask` (the same mapping the rest of the layout
uses: active bins are numbered 0..n_active-1 in mask order; gap/inactive bins map
to -1), and apply it, preserving the existing `-1` sentinel for out-of-range /
in-gap positions.

Replace the body of `linear_point_to_bin_ind` (`engines/graph.py`, lines
385–395) with:

```python
        if self.grid_edges is None or self.active_mask is None:
            raise RuntimeError("Layout not built; grid_edges or active_mask missing.")

        full_grid_ind = _find_bin_for_linear_position(
            data_points,
            bin_edges=self.grid_edges[0],
            active_mask=self.active_mask,
        )
        full_grid_ind = np.atleast_1d(np.asarray(full_grid_ind, dtype=int))

        # _find_bin_for_linear_position returns indices into the FULL (gap-
        # inclusive) bin list. Remap to active-bin indices: active bins are
        # numbered 0..n_active-1 in mask order; inactive/gap bins -> -1.
        n_total_bins = self.active_mask.size
        full_to_active = np.full(n_total_bins, -1, dtype=int)
        full_to_active[self.active_mask] = np.arange(int(self.active_mask.sum()))

        active_ind = np.full(full_grid_ind.shape, -1, dtype=int)
        in_range = full_grid_ind >= 0
        active_ind[in_range] = full_to_active[full_grid_ind[in_range]]
        return active_ind
```

This always returns a 1-D `NDArray[np.int_]` (matching the annotation), so the
prior scalar-to-array special-case is no longer needed. Keep the docstring's
"relative to the set of *active* 1D bins" wording — it is now accurate.

### Task 5 — `get_n_bins`: int64 / Python-int bin count, no int32 overflow

In `helpers/utils.py`, replace line 121 so the bin count is computed without an
intermediate `int32` cast, and guard against a still-pathological result:

```python
    # Calculate number of bins, ensuring at least 1 bin even if extent is 0.
    # Compute in float then cast straight to int64 (never int32) so a large
    # extent / small bin_size cannot silently overflow.
    n_bins_float = np.ceil(np.asarray(extent, dtype=np.float64) / bin_size_arr)
    if np.any(n_bins_float > np.iinfo(np.int64).max):
        raise ValueError(
            f"Requested binning overflows: extent / bin_size implies "
            f"{n_bins_float.max():.3g} bins, exceeding the int64 limit. "
            f"Increase bin_size or reduce the dimension range."
        )
    n_bins = n_bins_float.astype(np.int64)
    n_bins[n_bins == 0] = 1  # Handle zero-extent case
    return n_bins
```

(The final `np.asarray(..., dtype=np.int64)` re-cast at the old line 125 is now
redundant — `n_bins` is already int64 — so drop it.)

## Deliberately not in this phase

- **Backfilling tests for the zero-coverage public validators
  `validate_bin_size` / `validate_dimension_ranges`** (`layout/validation.py`)
  → **phase 25**. This phase only adds the targeted tests in the Validation
  slice below; the broad validator-coverage sweep is a separate deliverable.
- **`TriangularMeshLayout.build` keyword-only-argument convention** alignment
  → **phase 22**. It is in `layout/` but is an API-consistency change, not an
  indexing-correctness bug; folding it in here would mix concerns.
- **Generalizing the (x, y) convention into a shared coordinate-order helper.**
  The fix here is local to ImageMask; do not refactor RegularGrid /
  MaskedGrid edge construction "while in here."
- **Consuming the shared `_validation.py` finite/length helpers.** The five
  guards here are dtype / dimensionality / overflow checks, not the
  finite/length family; keep them inline in `layout/`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_image_mask_nonsquare_pixel_bin_lookup` | (fail-before/pass-after) Build an ImageMask from a deliberately **non-square** mask (e.g. `np.ones((2, 5), bool)`, i.e. 2 rows × 5 cols) with `pixel_size=(3.0, 1.0)` (x-width 3, y-height 1). A query point at the center of the `(row=0, col=4)` pixel — world `(x=13.5, y=0.5)` — maps via `layout.point_to_bin_index([[13.5, 0.5]])` to the bin whose `bin_centers` row equals `[13.5, 0.5]`. Pre-fix this digitizes x=13.5 against y_edges (max 2.0) and returns -1; post-fix it returns the correct active-bin index. Also assert `layout.dimension_ranges == ((0.0, 15.0), (0.0, 2.0))` and that `grid_edges[0]` (x) has 6 entries, `grid_edges[1]` (y) has 3. |
| `test_image_mask_bin_centers_match_dimension_ranges` | Every `bin_centers[:, 0]` lies within `dimension_ranges[0]` (x) and every `bin_centers[:, 1]` within `dimension_ranges[1]` (y). Pre-fix the x/y columns are swapped relative to the ranges. |
| `test_image_mask_pixel_size_param_and_bin_size_alias` | `build(image_mask=..., pixel_size=2.0)` works; `build(image_mask=..., bin_size=2.0)` works **and** emits a `DeprecationWarning`; passing both raises `ValueError`. The two valid calls produce identical `bin_centers`. |
| `test_image_mask_square_pixel_unchanged` | Regression guard: for a square mask with scalar `pixel_size`, `n_bins`, `bin_centers` set, and connectivity edge count are unchanged from the pre-fix behavior (square case was correct by symmetry). |
| `test_masked_grid_rejects_float_mask` | `MaskedGridLayout().build(active_mask=mask.astype(float), grid_edges=...)` raises `ValueError` naming the dtype. An `int` mask likewise raises. A genuine `bool` mask still builds. |
| `test_masked_grid_rejects_non_array_mask` | Passing a Python `list` mask raises `TypeError`. |
| `test_points_to_bin_dimensionality_mismatch_raises` | `_points_to_regular_grid_bin_ind` with 3-D `points` against a 2-D `grid_edges`/`grid_shape` raises `ValueError` (pre-fix: warns and returns all -1). |
| `test_graph_linear_point_to_bin_active_indices_with_gap` | (fail-before/pass-after) On a graph layout that has ≥1 gap bin, query linear positions in the segment **after** the gap. `linear_point_to_bin_ind` returns indices in `[0, n_active_bins)` and equal to the active-bin index (verify against `bin_centers`), not the gap-inclusive full-grid index. Positions in the gap and out of range return -1. |
| `test_get_n_bins_large_extent_no_int32_overflow` | `get_n_bins(positions=None-path, bin_size=1e-6, dimension_range=[(0.0, 5000.0)])` (extent 5000, → 5e9 bins) returns a positive int64 value `5_000_000_000`, never a negative/wrapped int32. Use `dimension_range` so no giant array is allocated. |
| `test_get_n_bins_overflow_guard_raises` | An extent/bin_size combination implying `> 2**63` bins raises `ValueError` with an actionable message (not a silent wrap). |

Mark none of these `slow` — all are pure-Python / tiny-array unit tests.

## Fixtures

All fixtures are synthesized inline (no checked-in data, no real-data slice):

- ImageMask tests build small boolean masks with `np.ones((R, C), bool)` /
  hand-written `np.array([[...]], dtype=bool)`; non-square cases use distinct
  `R != C` and distinct `pixel_size` x/y components so the axis swap is
  observable.
- MaskedGrid tests build a 2-D `grid_edges` tuple (e.g.
  `(np.arange(4.0), np.arange(3.0))`) plus a matching-shape mask cast to
  `float` / `int` / `list` for the negative cases.
- The graph test needs a 1-D `GraphLayout` **with a gap**. Reuse the existing
  graph-with-gap construction already used in the graph-layout tests
  (`tests/layout/` — find the fixture/helper that builds a linearized track with
  `edge_spacing > 0`); if none is shared, add a small `conftest.py` fixture in
  `tests/layout/` that builds a two-edge track separated by a gap, rather than
  copy-pasting the construction into each test.
- `get_n_bins` tests pass `dimension_range` explicitly (and a dummy 1×1
  `positions` array where the signature requires it) so no large grid is ever
  materialized.

Place new tests in the existing per-engine test modules where they fit
(`tests/layout/test_image_mask_layout.py`, `tests/layout/test_masked_grid_layout.py`
if present else `tests/layout/test_layout_engine.py`, and a `helpers` test module
for `get_n_bins` / `_points_to_regular_grid_bin_ind`). Run with `uv run pytest`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:

- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into
  phases 22 / 25, no opportunistic refactor of RegularGrid/MaskedGrid edge
  construction.
- Validation slice tests pass under `uv run pytest`; each fail-before/pass-after
  test was confirmed to actually fail on the pre-fix code (don't just assert it
  passes now).
- Tests aren't trivial — the non-square ImageMask test genuinely distinguishes
  (x, y) from (y, x) (it would pass on the buggy code if pixels were square or
  the mask were square, so the mask must be non-square **and** pixel_size
  anisotropic). No `assert True`; shared graph-with-gap setup is a fixture, not
  copy-paste.
- Docstrings, test names, and module names don't reference this plan or its
  phases/milestones.
- The legacy `bin_size` forwarding in `environment/factories.py` and its stale
  comment are removed; `grep` confirms no internal caller still passes
  `bin_size=` to the ImageMask engine.
- The `warnings` import in `helpers/regular_grid.py` is left intact only if still
  used; no orphaned imports remain after the Task 3 / Task 5 edits.
- `uv run ruff check . && uv run ruff format .` and
  `uv run mypy src/neurospatial/` are clean for the touched files.
