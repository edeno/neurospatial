# Phase 11 — Regions: validation correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase hardens four correctness bugs in `src/neurospatial/regions/` where the
code's *documented* contract and its *actual* behavior diverge silently:

1. `_rle_to_mask` promises a `ValueError` on out-of-bounds RLE indices but never
   raises one — it silently produces a wrong mask (or relies on NumPy's clamping
   of out-of-range slices).
2. The CVAT element processors catch `(ValueError, Exception)` — which is just
   `Exception` — and convert *every* error (including programming errors) into a
   warning + dropped shape, so real failures vanish.
3. `point_tolerance` defaults to `1e-8`, a float-equality epsilon applied to
   physical cm/pixel coordinates, so point-region membership returns all-`False`
   for any real (non-bit-exact) query position.
4. `region_center` advertises an `NDArray | None` return and a "None if empty"
   docstring, but raises `IndexError` on an empty polygon instead of returning
   `None`.

All four are scoped strictly to `src/neurospatial/regions/`. No public API is
renamed; these are behavior fixes to existing functions.

**Inputs to read first:**

- [src/neurospatial/regions/io.py:212-250](../../../../src/neurospatial/regions/io.py) — `_rle_to_mask`. The docstring Raises section (lines 230-233) promises `ValueError` "If RLE string is malformed **or values are out of bounds**." The body validates only non-integer (line 238) and odd-count (line 241) cases. The unpack loop (lines 247-248) uses `zip(..., strict=False)` and writes `mask[start : start + length] = 1` with **no bounds check**: a negative `start` indexes from the end of the flat array (wrong region set), and `start`/`start+length` past `height*width` is silently clamped by NumPy slicing (some run lost). Out-of-bounds therefore produces a wrong mask, never the promised `ValueError`.
- [src/neurospatial/regions/io.py:435,494,548,653](../../../../src/neurospatial/regions/io.py) — **four** `except (ValueError, Exception) as e:` sites, one per CVAT element processor (`_process_cvat_polygon` 435, `_process_cvat_polyline` 494, `_process_cvat_points` 548, `_process_cvat_mask` 653). `(ValueError, Exception)` is exactly `Exception` (ValueError ⊂ Exception), so each swallows `KeyError`, `TypeError`, `AttributeError`, etc. into a `UserWarning` + dropped shape. The mask site (line 653) already re-raises `ImportError` ahead of it (lines 650-651) — the pattern for "let real failures bubble" exists; it just isn't applied to the genuinely unexpected exceptions.
- [src/neurospatial/regions/ops.py:115-189](../../../../src/neurospatial/regions/ops.py) — `_get_points_in_single_region_mask`. The `point` branch (lines 146-170) tests membership with `(np.abs(xs - px) <= point_tolerance) & (np.abs(ys - py) <= point_tolerance)`. With `point_tolerance` defaulting to `1e-8` (a float-equality epsilon), any real position that isn't bit-identical to the stored point coordinate is excluded → all-`False`.
- [src/neurospatial/regions/ops.py:192-359](../../../../src/neurospatial/regions/ops.py) — the two public callers `points_in_any_region` (default at line 197) and `regions_containing_points` (default at line 256), both `point_tolerance: float = POINT_TOLERANCE`. These are the only two public entry points; both must get the corrected default and an honest docstring.
- [src/neurospatial/_constants.py:29](../../../../src/neurospatial/_constants.py) — `POINT_TOLERANCE = 1e-8`. This is the float-equality epsilon being misused as a physical tolerance. (Confirm no *other* module relies on it as an equality epsilon before changing its meaning — `grep -rn POINT_TOLERANCE src/` shows it imported only by `regions/ops.py:61`.)
- [src/neurospatial/regions/core.py:562-591](../../../../src/neurospatial/regions/core.py) — `region_center`. Signature returns `NDArray[np.float64] | None`; docstring (lines 572-574) says "or None if the region is empty or center cannot be determined." The polygon branch (lines 589-591) does `np.array(region.data.centroid.coords[0], dtype=float)`. An **empty** `shapely.Polygon` has `is_empty == True` and `centroid.coords` is empty, so `coords[0]` raises `IndexError` (verified: `Polygon().centroid.coords[0]` → `IndexError: index out of range`). The promised `None` is never produced.
- [tests/regions/test_serialization.py:388-427](../../../../tests/regions/test_serialization.py) — `TestRleToMask` (existing tests: `test_basic_rle`, `test_full_mask`, `test_empty_mask`, `test_invalid_rle_non_integer`). New out-of-bounds tests extend this class; `_rle_to_mask` is imported at line 13.
- [tests/regions/test_ops.py:110-178](../../../../tests/regions/test_ops.py) — `TestGetPointsInSingleRegionMask` and the `points_in_any_region` tests. Note `test_point_region_inside` (line 112) passes points **exactly equal** to the region coordinate, which is why the all-`False` bug never surfaced. `_get_points_in_single_region_mask`, `points_in_any_region`, `regions_containing_points` are imported at lines 11-14.
- [tests/regions/test_core.py:188-218](../../../../tests/regions/test_core.py) — region-construction / `area` / json patterns; new `region_center` tests live alongside these.

## Tasks

### 1. Bounds-validate `_rle_to_mask` and make the unpack strict (`regions/io.py:236-250`)

Replace the unpack loop so each `(start, length)` pair is validated against
`[0, height*width]` and the docstring's promised `ValueError` actually fires.
Also switch `zip(..., strict=False)` → `strict=True` (the odd-count guard at line
241 already guarantees even length, so `strict=True` is a correctness assertion,
not a behavior change).

Replace lines 244-250:

```python
    mask = np.zeros(height * width, dtype=np.uint8)
    n_pixels = height * width

    # Unpack RLE values into the mask. Each (start, length) run must lie fully
    # within the flattened image; out-of-bounds or negative runs are a malformed
    # encoding (a clamped NumPy slice would silently produce the wrong mask).
    for start, length in zip(rle_values[::2], rle_values[1::2], strict=True):
        if length < 0:
            raise ValueError(
                f"RLE run has negative length {length} (start={start}): {rle}"
            )
        if start < 0 or start + length > n_pixels:
            raise ValueError(
                f"RLE run [{start}, {start + length}) is out of bounds for an "
                f"image of {n_pixels} pixels ({height}x{width}): {rle}"
            )
        mask[start : start + length] = 1  # Set the corresponding region to 1

    return mask.reshape((height, width))  # Reshape to image dimensions
```

The docstring Raises section (lines 230-233) already documents "values are out of
bounds"; no docstring change needed — the code now matches it.

### 2. Narrow the CVAT processor `except` clauses (`regions/io.py:435,494,548,653`)

At each of the four sites, replace `except (ValueError, Exception) as e:` with a
narrowed clause that warns-and-skips on the *expected* malformed-geometry errors
and lets everything else propagate. The expected failures from
`_parse_cvat_points` (bad coordinate strings) and `shapely.Polygon(...)` (bad
geometry) are `ValueError` and shapely's `shapely.errors.GEOSException` /
`shapely.errors.ShapelyError`. A bare `except Exception` masks programming bugs;
narrowing makes those surface.

For each site, change:

```python
    except (ValueError, Exception) as e:
```

to:

```python
    except (ValueError, ShapelyError) as e:
```

and add the import near the top of `regions/io.py` (alongside the existing
`import shapely as shp` / shapely imports):

```python
from shapely.errors import ShapelyError
```

`ShapelyError` is the base class for shapely geometry errors (including
`GEOSException`), so this keeps the "skip a malformed annotated shape" behavior
while letting `KeyError`/`TypeError`/`AttributeError` (real bugs) propagate. The
mask site (line 653) keeps its preceding `except ImportError: raise` (lines
650-651) ahead of the narrowed clause — order is unchanged; only the broad tuple
is narrowed.

### 3. Give `point_tolerance` a physically-meaningful default (`regions/ops.py`, `_constants.py`)

The root problem is using a float-equality epsilon (`1e-8`) as a *spatial*
tolerance for "is this position at this point landmark." This phase is scoped to
`regions/`, which has no `Environment`/grid available, so the fix is a
physically-meaningful default tolerance (not bin-mapping, which would require an
`Environment` and is therefore out of scope — see "Deliberately not in this
phase"). Two coordinated edits:

**(a)** In `src/neurospatial/_constants.py`, keep the existing float-equality
epsilon under its current name for any equality use and add a *separate*,
clearly-named spatial default. Do **not** silently repurpose `POINT_TOLERANCE`'s
meaning. Add (next to line 29):

```python
# Float-equality epsilon (geometry coincidence, not a spatial tolerance).
POINT_TOLERANCE = 1e-8

# Default spatial tolerance (in the regions' coordinate units, e.g. cm or px)
# for treating a query position as "at" a point landmark. A point region has no
# area, so exact float equality is never the intent; this is the radius of the
# square neighborhood around the landmark that counts as a hit.
POINT_REGION_TOLERANCE = 1.0
```

**(b)** In `src/neurospatial/regions/ops.py`:

- Update the import at line 61:

  ```python
  from .._constants import POINT_REGION_TOLERANCE
  ```

  (drop the `POINT_TOLERANCE` import — it is not used elsewhere in this module;
  confirm with `grep -n POINT_TOLERANCE src/neurospatial/regions/ops.py`.)

- Change the default on `points_in_any_region` (line 197) and
  `regions_containing_points` (line 256) from
  `point_tolerance: float = POINT_TOLERANCE` to
  `point_tolerance: float = POINT_REGION_TOLERANCE`.

- Update both docstring `point_tolerance` parameter descriptions (lines 213-214
  and 280-281) from the bare "Tolerance for comparing query points to point
  Regions." to:

  ```
  point_tolerance : float, default=POINT_REGION_TOLERANCE
      Spatial half-width (in the regions' coordinate units) of the square
      neighborhood around a point region that counts as a hit. Point regions
      have no area, so membership is a proximity test, not exact equality.
      Ignored for polygon regions. Pass a smaller value for stricter matching.
  ```

The membership math at `_get_points_in_single_region_mask` lines 167-169 is
**unchanged** — `np.abs(xs - px) <= point_tolerance` is already a neighborhood
test; only the default magnitude was wrong. Leave the `point_tolerance` parameter
on `_get_points_in_single_region_mask` (line 118) as a required positional
(callers in this module always pass it); no default needed there.

### 4. Return `None` from `region_center` for empty geometry (`regions/core.py:587-591`)

Make the polygon branch honor the `| None` return for an empty polygon, matching
the docstring. Replace lines 587-591:

```python
        if region.kind == "point":
            return np.asarray(region.data, dtype=float)
        # region.kind == "polygon"
        assert isinstance(region.data, Polygon)
        if region.data.is_empty:
            # Empty polygon has no centroid coordinates; the documented
            # contract is to return None rather than raise.
            return None
        return np.array(region.data.centroid.coords[0], dtype=float)
```

The signature (`-> NDArray[np.float64] | None`, line 562) and docstring (lines
570-574) already advertise this; no signature/docstring change needed — the code
now matches them.

### 5. Documentation

No README/CHANGELOG/QUICKSTART changes are required: all four are bug fixes to
existing functions, and the only signature-surface change (the `point_tolerance`
default value) is documented inline in the affected docstrings (task 3). Do
**not** touch `.claude/` docs, `PATTERNS.md`, or `API_REFERENCE.md`.

## Deliberately not in this phase

- **Mapping point regions to their containing *bin* (à la `bins_in_region` /
  `region_mask`).** Those methods live in `src/neurospatial/environment/`
  (`environment/regions.py:56,331`), require a fitted `Environment`, and are out
  of this phase's `regions/`-only scope. The proximity-tolerance fix (task 3) is
  the in-scope, `Environment`-free resolution. Any future bin-snapping
  convenience belongs with the Environment regions surface, not here.
- **`ops.py` doc/example drift — `transform=` vs `pixel_to_world=`.** The module
  docstring examples (`ops.py:21-47`) and `load_labelme_json` use mismatched
  parameter names; reconciling that naming is **phase 23** (doc/API-drift sweep),
  not this correctness phase.
- **`plot_regions` signature drift** and the **`PATTERNS.md` "overwrite warns vs
  raises"** inconsistency → **phase 23**.
- **`region_center` parameter-naming convention** (`region_name` vs the
  canonical `name` used by sibling methods like `area(self, name)`) → **phase
  22** (naming-consistency pass). This phase changes only the *empty-geometry
  return path*, not the parameter name.
- **Retiring `POINT_TOLERANCE`.** Task 3 *keeps* the float-equality epsilon
  constant (it may be referenced elsewhere later) and adds a separate spatial
  default; it does not delete `POINT_TOLERANCE`.
- **Re-raising vs warning policy for malformed CVAT shapes.** Task 2 preserves
  the existing warn-and-skip behavior for *geometry* errors; it does not change
  the project's annotation-loading UX (no switch to hard-fail on first bad
  shape). Only genuinely unexpected exception types stop being swallowed.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_rle_negative_start_raises` | `_rle_to_mask("-1,3", 5, 5)` raises `ValueError` matching "out of bounds". **Fails before** (negative index wraps; wrong mask, no raise). |
| `test_rle_start_past_end_raises` | `_rle_to_mask("30,2", 5, 5)` (start ≥ 25) raises `ValueError` "out of bounds". **Fails before** (clamped slice, no raise). |
| `test_rle_run_overruns_end_raises` | `_rle_to_mask("23,5", 5, 5)` (start+length=28 > 25) raises `ValueError` "out of bounds". **Fails before** (run silently truncated). |
| `test_rle_negative_length_raises` | `_rle_to_mask("0,-3", 5, 5)` raises `ValueError` "negative length". **Fails before**. |
| `test_rle_in_bounds_still_works` | `_rle_to_mask("0,5,10,3", 5, 5)` returns the same mask as today (no regression; mirrors existing `test_basic_rle`). |
| `test_cvat_polygon_processor_reraises_unexpected` | A `_process_cvat_polygon` call whose geometry step raises a non-geometry error (e.g. patch `shp.Polygon` to raise `KeyError`) propagates the `KeyError` rather than warning-and-returning `None`. **Fails before** (swallowed into `UserWarning`, returns `None`). |
| `test_cvat_polygon_processor_skips_bad_geometry` | A polygon element with malformed points still warns + returns `None` (no regression; the *intended* skip path survives narrowing). |
| `test_point_region_membership_at_real_coordinate` | A point region at `[50.0, 50.0]` and a query position `[50.0, 50.0]` (and one a few mm away within default tolerance) return membership `True` via `points_in_any_region` with the **default** `point_tolerance`. **Fails before** (default `1e-8` → all-`False` unless bit-exact; a `[50.0, 50.0]` literal that *is* bit-exact passes, so use a query like `[50.0001, 49.9998]` to expose the bug). |
| `test_point_region_membership_far_position_false` | A query position `[60.0, 60.0]` (well outside default tolerance) returns `False` — the default isn't so large it swallows everything. |
| `test_point_region_strict_tolerance_still_available` | Passing `point_tolerance=1e-8` explicitly reproduces strict bit-exact behavior (the knob still works). |
| `test_region_center_empty_polygon_returns_none` | `Regions([Region("e", kind="polygon", data=shapely.Polygon())]).region_center("e")` returns `None` (not `IndexError`). **Fails before** (`IndexError: index out of range`). |
| `test_region_center_nonempty_polygon_centroid` | A unit-square polygon region's `region_center` returns `[0.5, 0.5]` (no regression). |
| `test_region_center_point_unchanged` | A point region `[3.0, 7.0]` returns `[3.0, 7.0]` (no regression). |

All tests are pure-Python unit tests (shapely is already a hard dependency of
`regions/`); none need `pytest.mark.slow` or `pytest.mark.integration`.

## Fixtures

No new shared fixtures or checked-in data. Tests construct `Region` / `Regions`
inline (matching the existing `tests/regions/test_core.py` and
`tests/regions/test_ops.py` style) and call `_rle_to_mask` / `_get_…_mask` with
literal arguments. For the CVAT processor re-raise test, build a minimal
`xml.etree.ElementTree.Element` with a valid `points` attribute and use
`unittest.mock.patch` to make `shapely.Polygon` raise `KeyError` (a
non-geometry, non-`ShapelyError` exception) — mirroring how `tests/regions/`
already exercises the CVAT helpers with synthetic XML elements. Add the new
`_rle_to_mask` tests to the existing `TestRleToMask` class
(`tests/regions/test_serialization.py:388`); add the point-membership tests to
the existing `points_in_any_region` test group (`tests/regions/test_ops.py`); add
the `region_center` tests to `tests/regions/test_core.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:

- All four tasks are implemented as specified, scoped strictly to
  `src/neurospatial/regions/` and `src/neurospatial/_constants.py`.
- **All four** `except (ValueError, Exception)` sites (io.py:435, 494, 548, 653)
  are narrowed — not just the one in the finding — and the mask site's preceding
  `except ImportError: raise` is preserved.
- The "Deliberately not in this phase" list is honored: no Environment/bin
  snapping, no `transform=`/`pixel_to_world=` rename, no `region_name`→`name`
  rename, no `plot_regions`/`PATTERNS.md` edits, no deletion of `POINT_TOLERANCE`.
- `POINT_TOLERANCE`'s meaning is **not** silently repurposed — a separate
  `POINT_REGION_TOLERANCE` is introduced and the `ops.py` defaults point at it.
- Validation-slice tests pass and genuinely fail on the pre-fix code (spot-check
  by stashing each fix). In particular the point-membership test uses a
  *non-bit-exact* query coordinate so it actually exercises the all-`False` bug;
  it is not a tautology that passes only because the literal happens to be exact.
- Tests are not trivial; shared construction stays inline-per-test per the
  existing `tests/regions/` style (no copy-pasted setup that belongs in a
  fixture).
- Docstrings, test names, and module names do not reference this plan or its
  phase number.
- `uv run pytest tests/regions -q`, `uv run ruff check . && uv run ruff format
  .`, and `uv run mypy src/neurospatial/regions/` all pass.
