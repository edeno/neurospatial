# Phase 12 — Annotation: geometry & track-state correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase is one PR scoped strictly to `src/neurospatial/annotation/`. It fixes four
correctness defects where the annotation layer either silently produces a degenerate
result or crashes where it documents that it warns:

1. Alpha-shape boundary inference returns a non-`Polygon` / empty geometry and hands
   back a degenerate boundary.
2. `edge_order`/`edge_spacing` manual overrides go stale after `delete_node` /
   `delete_edge`, silently producing a wrong linearization. There is **no** regression
   test for this today; this phase adds one.
3. `validate_region_overlap` crashes on an invalid (self-intersecting) polygon,
   violating the module's documented "warn, never raise" contract.
4. `shapes_to_regions` silently drops `<3`-vertex polygons; `annotate_track_graph`
   silently discards invalid `initial_edges`.

**Inputs to read first:**

- [../../../../src/neurospatial/annotation/_boundary_inference.py:289](../../../../src/neurospatial/annotation/_boundary_inference.py) — `result = alphashape.alphashape(positions, alpha)` (line 289); the `MultiPolygon` branch is handled (295-308) but the final `return result` (310) returns whatever `alphashape` produced, including `Point`/`LineString`/`GeometryCollection`/empty geometry. This is the degenerate-boundary bug.
- [../../../../src/neurospatial/annotation/_track_state.py:205](../../../../src/neurospatial/annotation/_track_state.py) — `delete_node` (205-249) reindexes `self.edges` and `self.start_node` but never touches `self.edge_order_override` / `self.edge_spacing_override`.
- [../../../../src/neurospatial/annotation/_track_state.py:344](../../../../src/neurospatial/annotation/_track_state.py) — `delete_edge` (344-362) pops from `self.edges` but never touches the overrides.
- [../../../../src/neurospatial/annotation/_track_state.py:79](../../../../src/neurospatial/annotation/_track_state.py) — field declarations `edge_order_override: list[tuple[int, int]] | None` (79) and `edge_spacing_override: list[float] | None` (80); setters/clearers at `set_edge_order` (474), `clear_edge_order` (489), `set_edge_spacing` (498), `clear_edge_spacing` (514).
- [../../../../src/neurospatial/annotation/_track_helpers.py:338](../../../../src/neurospatial/annotation/_track_helpers.py) — `build_track_graph_result` consumes `state.edge_order_override` (338-350) and `state.edge_spacing_override` (352-364) verbatim; this is where a stale override becomes a wrong linearization. Read it to confirm the consumption path the regression test exercises.
- [../../../../src/neurospatial/annotation/validation.py:1](../../../../src/neurospatial/annotation/validation.py) — module docstring (11-12): "All validation functions emit warnings rather than raising errors." This is the contract item 3 violates.
- [../../../../src/neurospatial/annotation/validation.py:262](../../../../src/neurospatial/annotation/validation.py) — `validate_region_overlap` calls `poly1.intersects(poly2)` (262), `poly1.intersection(poly2)` (265), `.area` (266, 269-270); any of these raises `shapely.errors.GEOSException` (TopologyException) on an invalid polygon.
- [../../../../src/neurospatial/annotation/converters.py:112](../../../../src/neurospatial/annotation/converters.py) — `if len(pts_world) < 3: continue` (112-114) in `shapes_to_regions` (def at line 21) silently drops degenerate polygons.
- [../../../../src/neurospatial/annotation/track_graph.py:208](../../../../src/neurospatial/annotation/track_graph.py) — `for node1, node2 in initial_edges: state.add_edge(node1, node2)` (208-210); `add_edge` returns `bool` (False on out-of-range index, self-loop, or duplicate) and the return value is discarded, so invalid `initial_edges` vanish silently. `annotate_track_graph` def at line 56; `initial_edges` param at line 63.
- [../../../../src/neurospatial/annotation/_track_state.py:305](../../../../src/neurospatial/annotation/_track_state.py) — `add_edge` (305-342) — read the four `return False` paths so the warning in `track_graph.py` reports the right reason.
- [../../../../tests/annotation/test_edge_order.py:209](../../../../tests/annotation/test_edge_order.py) — `test_result_uses_manual_edge_order` shows the existing override + `build_track_graph_result` test pattern the new regression test mirrors. Confirm no `delete_node`/`delete_edge` staleness test exists here.
- [../../../../tests/annotation/test_validation.py](../../../../tests/annotation/test_validation.py) and [../../../../tests/annotation/test_converters.py](../../../../tests/annotation/test_converters.py) — existing fixture/assertion style for these modules.

**Contracts referenced:** none. (This phase does not touch `_validation.py` finite/length guards or the result-object mixin; it is geometry/state correctness local to `annotation/`.)

## Tasks

### Task 1 — Validate alpha-shape result is a non-empty `Polygon`

In `_boundary_inference.py`, the helper that wraps `alphashape.alphashape` returns `result`
directly after handling the `MultiPolygon` case. When the input is degenerate (collinear,
near-duplicate, or too-sparse points), `alphashape` returns a `Point`, `LineString`,
empty `Polygon`, or `GeometryCollection`. Downstream code treats the return as a
`Polygon` and produces a degenerate boundary. Validate before returning.

Add the guard after the `MultiPolygon` branch (replacing the bare `return result` at the
end of the function, ~line 310). Import `Polygon` alongside the existing `MultiPolygon`
import at line 273.

```python
    from shapely.geometry import MultiPolygon, Polygon

    # ... existing MultiPolygon handling (take largest, warn) ...
    if isinstance(result, MultiPolygon):
        ...
        return largest

    # Validate the single-geometry result before returning. alphashape can
    # return a Point / LineString / empty Polygon / GeometryCollection when the
    # input points are collinear, near-duplicate, or too sparse for the chosen
    # alpha; downstream code assumes a non-empty Polygon.
    if not isinstance(result, Polygon) or result.is_empty:
        raise ValueError(
            f"Alpha shape produced a degenerate boundary "
            f"({type(result).__name__}"
            f"{', empty' if getattr(result, 'is_empty', False) else ''}) "
            f"from {n_points} position(s) at alpha={alpha:.3f}. "
            f"This usually means the positions are nearly collinear, "
            f"contain too few distinct points, or alpha is too small.\n\n"
            f"To fix:\n"
            f"  1. Increase alpha from {alpha:.3f} to ~{alpha * 2:.3f} for a "
            f"looser boundary\n"
            f"  2. Use method='convex_hull' for a guaranteed single polygon\n"
            f"  3. Add more position samples or remove duplicate points",
        )

    return result
```

Update the function's NumPy `Raises` section (currently lists only `ImportError`) to add
the new `ValueError`.

### Task 2 — Reindex/clear `edge_order` and `edge_spacing` overrides on delete

The manual overrides store edges as `(node_idx, node_idx)` tuples and spacings positional
to the edge list. After `delete_node` reindexes nodes and edges, and after `delete_edge`
removes an edge, the overrides are silently stale, and `build_track_graph_result`
(`_track_helpers.py:338-364`) feeds them straight into the linearization.

The simplest correct fix is to **clear** both overrides on any delete: the user explicitly
set a manual order/spacing against a specific topology, and a mutating edit invalidates
that intent. After clearing, `build_track_graph_result` falls back to `infer_edge_layout`,
which is the safe (auto-inferred) path. This avoids partial-reindex subtleties (a deleted
node can drop an edge that appears mid-override) while guaranteeing the result is never
silently wrong.

Add a private helper on `TrackBuilderState` and call it from both deleters. Both deleters
already call `self._save_for_undo()` first, so clearing is captured by undo.

```python
    def _invalidate_edge_layout_overrides(self) -> None:
        """Clear manual edge-order/spacing overrides after a topology edit.

        A manual ``edge_order_override`` / ``edge_spacing_override`` is indexed
        against a specific node/edge topology. Deleting a node or edge changes
        that topology, so a retained override would silently mis-order the
        linearization. We clear the overrides and fall back to inferred layout.
        """
        self.edge_order_override = None
        self.edge_spacing_override = None
```

In `delete_node` (205-249), after the existing reindex/start-node block (i.e. after the
`if self.start_node is not None:` block ending at line 248):

```python
        # A manual edge-order/spacing override is keyed to the old topology.
        self._invalidate_edge_layout_overrides()
```

In `delete_edge` (344-362), after `self.edges.pop(edge_idx)` (line 362):

```python
        # A manual edge-order/spacing override is keyed to the old topology.
        self._invalidate_edge_layout_overrides()
```

Note: `_save_for_undo()` is already called before the mutation in both methods, so an
undo restores the overrides along with the topology. Do not move the `_save_for_undo()`
calls.

### Task 3 — `validate_region_overlap` must warn, not crash, on invalid polygons

`validate_region_overlap` (`validation.py`, `validate_region_overlap` def at line 202)
calls `intersects` / `intersection` / `.area` on user-drawn polygons (262-266). A
self-intersecting polygon raises `shapely.errors.GEOSException`, breaking the documented
"warn, never raise" contract (module docstring lines 11-12).

Guard each pair's geometry work in a `try/except shapely.errors.GEOSException`, append an
issue string, and warn (when `warn_overlap`) instead of propagating. Add the import at the
top of `validation.py` next to the existing shapely imports (lines 20-22):

```python
from shapely.errors import GEOSException
```

Wrap the per-pair overlap computation (lines 262-285, from `if not poly1.intersects(poly2):`
through the `if max_overlap > overlap_threshold:` block) in:

```python
            try:
                if not poly1.intersects(poly2):
                    continue

                intersection = poly1.intersection(poly2)
                intersection_area = intersection.area

                overlap_frac_1 = (
                    intersection_area / poly1.area if poly1.area > 0 else 0
                )
                overlap_frac_2 = (
                    intersection_area / poly2.area if poly2.area > 0 else 0
                )
                max_overlap = max(overlap_frac_1, overlap_frac_2)

                if max_overlap > overlap_threshold:
                    issues.append(
                        f"Regions '{r1.name}' and '{r2.name}' overlap heavily "
                        f"({max_overlap:.1%} of smaller region)",
                    )
                    if warn_overlap:
                        warnings.warn(
                            f"Regions '{r1.name}' and '{r2.name}' overlap "
                            f"heavily ({max_overlap:.1%} of the smaller "
                            f"region's area). This may be intentional, but "
                            f"could indicate a drawing error.",
                            UserWarning,
                            stacklevel=3,
                        )
            except GEOSException as exc:
                issues.append(
                    f"Could not check overlap between '{r1.name}' and "
                    f"'{r2.name}': invalid geometry ({exc})",
                )
                if warn_overlap:
                    warnings.warn(
                        f"Could not check overlap between regions '{r1.name}' "
                        f"and '{r2.name}' because one of them has invalid "
                        f"geometry ({exc}). Fix the polygon (e.g. it may be "
                        f"self-intersecting) and re-validate.",
                        UserWarning,
                        stacklevel=3,
                    )
                continue
```

The `intersects`/`intersection`/`.area`/overlap logic is the same code that exists today
(262-285); only the `try`/`except` wrapper and the indentation change.

### Task 4 — Warn on dropped `<3`-vertex polygons and discarded `initial_edges`

**4a — `converters.py` `shapes_to_regions`** (def at line 21; drop site 112-114). Replace
the silent `continue` with a warning that names the offending shape, then `continue`. The
loop variable `name` is already bound (line 95). `warnings` is imported inside the function
(line 87).

```python
        # Skip degenerate polygons (need ≥3 vertices to form an area), but warn
        # so the user knows a drawn shape was dropped rather than annotated.
        if len(pts_world) < 3:
            warnings.warn(
                f"Skipping shape '{name}': a polygon needs at least 3 vertices "
                f"to define an area, but this shape has {len(pts_world)}.",
                UserWarning,
                stacklevel=2,
            )
            continue
```

**4b — `track_graph.py` `annotate_track_graph`** (initial-edge loop 208-210). `add_edge`
returns `False` when an edge is invalid (out-of-range index, self-loop, or duplicate). Warn
on `False` so a caller-supplied `initial_edges` entry is not silently dropped. `warnings` is
not currently imported at module top in `track_graph.py` — add `import warnings` to the
module imports (top of file) rather than inside the function, to keep it available.

```python
    if initial_edges is not None:
        for node1, node2 in initial_edges:
            if not state.add_edge(node1, node2):
                warnings.warn(
                    f"Ignoring invalid initial edge ({node1}, {node2}): "
                    f"edges must reference existing nodes "
                    f"(0..{len(state.nodes) - 1}), must not be self-loops, "
                    f"and must not duplicate an existing edge.",
                    UserWarning,
                    stacklevel=2,
                )
```

Update the `initial_edges` parameter docstring (line 92-93) to note that invalid edges are
ignored with a warning.

### Task 5 — Docstring touch-ups (user-facing)

- `_boundary_inference.py`: add `ValueError` to the alpha-shape helper's `Raises` section
  (Task 1).
- `track_graph.py`: extend the `initial_edges` parameter description to state that invalid
  edges are ignored with a warning (Task 4b).

No README/CHANGELOG entries are required for these (all four are bugfixes to existing
documented behavior, not new API). Do not add new public symbols.

## Deliberately not in this phase

- **`from neurospatial.annotation import Role` `ImportError` doc bug and the
  `annotate_video` wrong-default docstring → phase 23.** `annotation/__init__.py` exports
  `RegionType` (line 10), not `Role`; any docstring/example importing `Role` is a
  documentation defect, not a geometry/state bug. Likewise the `annotate_video` default
  documented incorrectly is a docstring fix. Both are doc-only and batched into the
  documentation-correctness phase.
- **Keyword-only-argument convention for `calibration` / `method` / `simplify_tolerance`
  → phase 22.** Making these keyword-only across the annotation entry points is an API
  ergonomics change (signature churn), not a correctness fix, and belongs with the
  argument-convention sweep. Do not reorder or `*`-gate signatures here.
- **No partial reindexing of overrides.** Task 2 *clears* overrides on delete rather than
  attempting to surgically reindex them. A reindex-preserving scheme (drop the deleted
  edge from the order, shift remaining tuples) is intentionally out of scope — clearing is
  the correct, low-risk fix; revisit only if a user explicitly needs override survival
  across deletes.
- **No changes outside `src/neurospatial/annotation/`.** If a fix appears to need a change
  in `regions/`, `ops/`, or `layout/`, stop — that signals the boundary was drawn wrong.

## Validation slice

All tests live under `tests/annotation/`. Each must fail on `main` (before the fix) and
pass after.

| Test | Asserts |
| --- | --- |
| `test_boundary_inference.py::test_alpha_shape_collinear_points_raises` | Calling the alpha-shape helper with (near-)collinear points that drive `alphashape` to a non-`Polygon`/empty result raises `ValueError` whose message names "degenerate" and suggests `convex_hull`. **Fails before:** returns a degenerate geometry, no raise. |
| `test_boundary_inference.py::test_alpha_shape_valid_points_returns_polygon` | A well-separated 2D point cloud still returns a non-empty `shapely.Polygon` (guards against the new check over-rejecting valid input). |
| `test_edge_order.py::test_delete_node_clears_edge_order_override` | After `set_edge_order(...)` then `delete_node(...)`, `state.edge_order_override is None`. **Fails before:** override retained (stale). |
| `test_edge_order.py::test_delete_node_clears_edge_spacing_override` | After `set_edge_spacing(...)` then `delete_node(...)`, `state.edge_spacing_override is None`. **Fails before:** retained. |
| `test_edge_order.py::test_delete_edge_clears_overrides` | After setting both overrides then `delete_edge(...)`, both are `None`. **Fails before:** retained. |
| `test_edge_order.py::test_stale_override_does_not_corrupt_linearization` | Build a state, `set_edge_order` against the current topology, `delete_node` on a node that drops an edge, then `build_track_graph_result(state, calibration=None)`; assert the result's `edge_order` matches the *inferred* layout for the post-delete topology (only references surviving edges), not the stale override. This is the regression test that does not exist today. **Fails before:** result reflects the stale override. |
| `test_edge_order.py::test_undo_restores_overrides_after_delete` | After `set_edge_order(...)`, `delete_node(...)` (clears override), then `undo()`, the override is restored — confirms `_save_for_undo` ordering is preserved. |
| `test_validation.py::test_validate_region_overlap_invalid_polygon_warns_not_raises` | Two regions where one is a self-intersecting (bowtie) polygon: `validate_region_overlap` returns a non-empty issues list and emits a `UserWarning` (assert via `pytest.warns`), and does **not** raise `GEOSException`. **Fails before:** raises. |
| `test_validation.py::test_validate_region_overlap_valid_polygons_unchanged` | Two heavily overlapping valid polygons still produce the existing overlap warning/issue (guards against the try/except swallowing the normal path). |
| `test_converters.py::test_shapes_to_regions_warns_on_degenerate_polygon` | A shapes list containing a 2-vertex shape emits a `UserWarning` naming the shape and is excluded from the returned regions. **Fails before:** silently dropped, no warning. |
| `test_track_graph.py::test_annotate_track_graph_warns_on_invalid_initial_edge` | Invalid `initial_edges` is rejected with a `UserWarning`. Drive this at the `TrackBuilderState` + edge-application level (see Fixtures) so no napari viewer is launched; if the only reachable path is through `annotate_track_graph`, mark the test `@pytest.mark.napari` / `pytest.importorskip("napari")`. **Fails before:** silently discarded. |

## Fixtures

- **Synthesized in-test, no checked-in data.** Alpha-shape tests: a collinear/near-collinear
  `np.ndarray` of shape `(n, 2)` for the degenerate case, and a square-ish random cloud for
  the valid case (seed with `np.random.default_rng(0)`).
- **Track-state tests:** build `TrackBuilderState` directly (mirror the existing
  `tests/annotation/test_edge_order.py` pattern, e.g. `test_result_uses_manual_edge_order`
  at line 209) — add 3-4 nodes via `add_node`, connect via `add_edge`, set overrides, then
  delete. No napari, no GUI.
- **Validation tests:** construct `Regions` with `Region(kind="polygon", data=<shapely
  Polygon>)`; build a self-intersecting "bowtie" polygon explicitly,
  e.g. `Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])` (`assert not poly.is_valid`).
- **Converter test:** pass `shapes_data` containing a `(2, 2)` array (2 vertices) alongside
  a valid `(4, 2)` shape; assert the valid one survives and a warning fires for the dropped
  one.
- **`initial_edges` warning:** prefer exercising `TrackBuilderState.add_edge` (returns
  `False`) plus the warning wrapper directly so the test runs in the default (non-napari)
  suite. Only fall back to launching `annotate_track_graph` (and marking `napari`) if the
  warning cannot be reached otherwise.
- The default test command excludes `napari` and `slow`
  (`addopts = ... -m "not slow and not napari"`); keep the geometry/state tests in the
  default suite (they need only `shapely` / `track_linearization`, both core deps) and mark
  only any unavoidable viewer-launching test as `napari`.

Run: `uv run pytest tests/annotation/ -v`. For the napari-marked test (if any):
`uv run pytest tests/annotation/ -m napari -v`.

## Review

Before opening the PR, dispatch an independent reviewer (`code-reviewer` or equivalent)
against the diff. Confirm:

- All five tasks are implemented as specified, scoped strictly to
  `src/neurospatial/annotation/`. No edits to `regions/`, `ops/`, `layout/`, or any other
  module.
- The "Deliberately not in this phase" list is honored — no `Role`/`annotate_video`
  docstring fixes, no keyword-only signature changes, no override reindexing scheme.
- Validation-slice tests pass under `uv run pytest tests/annotation/`. Geometry/state tests
  run in the default (non-napari) suite; any unavoidable viewer-launching test is marked
  `@pytest.mark.napari` (and uses `pytest.importorskip("napari")`), since napari is a heavy
  optional/GUI dependency.
- Tests are non-trivial: the staleness regression test asserts the *linearization output*
  changed (inferred vs stale order), not merely that an attribute is `None`; the validation
  test uses `pytest.warns` and asserts no `GEOSException` escapes. No `assert True`; no
  assertion that only re-checks a value the test just set. Shared setup lives in fixtures,
  not copy-pasted.
- Docstrings, test names, and any new symbol names do not reference this plan, "phase 12",
  or "review-remediation".
- `uv run ruff check src/neurospatial/annotation/ tests/annotation/` and
  `uv run ruff format --check` are clean; `uv run mypy src/neurospatial/annotation/` passes
  (the `shapely.errors.GEOSException` import and the new helper are typed).
- No orphaned code: the old bare `return result`, the silent `continue`, and the
  unchecked `add_edge` call are fully replaced, not left alongside the new paths.
