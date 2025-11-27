# TASKS.md - Track Graph Annotation Implementation

## Overview

Implementation tasks for adding interactive track graph building functionality to the annotation module. Output integrates with `Environment.from_graph()` for 1D linearized track environments.

**Goal**: Users can interactively build track graphs on video frames in napari, with output ready for `Environment.from_graph()`.

---

## Milestone 1: Core Infrastructure

### Task 1.1: Create Type Definitions

**File**: `src/neurospatial/annotation/_track_types.py`

**Actions**:

- [x] Create `_track_types.py` with `TrackGraphMode` type alias
- [x] Define: `TrackGraphMode = Literal["add_node", "add_edge", "delete"]`

**Success Criteria**:

- Type alias imports correctly
- Mypy passes without errors

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Simple (< 20 lines)

---

### Task 1.2: Implement TrackBuilderState

**File**: `src/neurospatial/annotation/_track_state.py`

**Dependencies**: Task 1.1

**Actions**:

- [x] Create `TrackBuilderState` dataclass with fields:
  - `mode: TrackGraphMode` (default: "add_node")
  - `nodes: list[tuple[float, float]]`
  - `edges: list[tuple[int, int]]`
  - `node_labels: list[str]`
  - `start_node: int | None`
  - `edge_start_node: int | None` (transient for two-click edge creation)
  - `undo_stack: list[dict]` and `redo_stack: list[dict]`
  - `_max_undo_depth: int = 50`

- [x] Implement snapshot methods:
  - `_snapshot() -> dict` - Create serializable state copy
  - `_restore_snapshot(snapshot: dict)` - Restore from snapshot
  - `_save_for_undo()` - Save before mutation, clear redo stack

- [x] Implement undo/redo:
  - `undo() -> bool` - Restore previous state
  - `redo() -> bool` - Restore next state

- [x] Implement node operations:
  - `add_node(x, y, label=None) -> int` - Add node, return index
  - `delete_node(idx)` - Delete node + connected edges, reindex remaining
  - `set_start_node(idx)` - Designate start for linearization
  - `find_nearest_node(x, y, threshold) -> int | None`

- [x] Implement edge operations:
  - `add_edge(node1, node2) -> bool` - Add if valid (no self-loops/duplicates)
  - `delete_edge(edge_idx)` - Delete by index

- [x] Implement validation:
  - `to_track_graph() -> nx.Graph` - Build graph from state (pixel coords)
  - `validate() -> dict` - Use `check_track_graph_validity()`
  - `is_valid_for_save() -> tuple[bool, list[str], list[str]]`
  - `get_effective_start_node() -> int | None` - Default to 0 if unset

**Success Criteria**:

- All operations save to undo stack before mutation
- `delete_node` correctly reindexes edges and updates `start_node`
- Self-loops and duplicate edges are rejected
- Undo/redo work correctly for all operations
- Unit tests pass for all methods

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Medium (~200 lines)

---

### Task 1.3: Create Graph Building Helpers

**File**: `src/neurospatial/annotation/_track_helpers.py`

**Dependencies**: Task 1.2

**Actions**:

- [x] Implement `transform_nodes_to_output(nodes_px, calibration) -> list[tuple]`:
  - Return pixel coords if no calibration
  - Use `calibration.transform_px_to_cm()` if provided
  - Match behavior of `annotate_video` exactly

- [x] Implement `build_track_graph_from_positions(node_positions, edges) -> nx.Graph`:
  - Use `track_linearization.make_track_graph()` internally
  - Ensures proper `distance` and `edge_id` edge attributes

- [x] Implement `build_track_graph_result(state, calibration) -> TrackGraphResult`:
  - Store `pixel_positions` (original)
  - Transform `node_positions` using calibration
  - Build track graph from transformed positions
  - Call `infer_edge_layout()` for edge_order and edge_spacing
  - Handle empty/invalid graphs gracefully (return None for track_graph)

**Success Criteria**:

- `transform_nodes_to_output` produces identical results to `annotate_video` coordinate transforms
- `build_track_graph_from_positions` creates valid graph with proper attributes
- Result matches `TrackGraphResult` schema

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Medium (~150 lines)

---

### Task 1.4: Write Unit Tests for State Management

**File**: `tests/annotation/test_track_state.py`

**Dependencies**: Tasks 1.1, 1.2, 1.3

**Actions**:

- [x] Test node operations:
  - `test_add_node_returns_index`
  - `test_add_node_with_label`
  - `test_delete_node_removes_connected_edges`
  - `test_delete_node_reindexes_remaining_edges`
  - `test_delete_node_updates_start_node`
  - `test_find_nearest_node_within_threshold`
  - `test_find_nearest_node_outside_threshold_returns_none`

- [x] Test edge operations:
  - `test_add_edge_success`
  - `test_add_edge_rejects_self_loop`
  - `test_add_edge_rejects_duplicate`
  - `test_delete_edge`

- [x] Test undo/redo:
  - `test_undo_restores_previous_state`
  - `test_redo_restores_next_state`
  - `test_undo_empty_stack_returns_false`
  - `test_new_action_clears_redo_stack`
  - `test_undo_stack_depth_limit`

- [x] Test validation:
  - `test_is_valid_for_save_requires_nodes_and_edges`
  - `test_is_valid_for_save_warns_no_start_node`
  - `test_get_effective_start_node_defaults_to_zero`
  - `test_validate_uses_check_track_graph_validity`

- [x] Test graph building:
  - `test_to_track_graph_has_pos_attributes`
  - `test_transform_nodes_to_output_with_calibration`
  - `test_transform_nodes_to_output_without_calibration`
  - `test_build_track_graph_result_complete`

**Success Criteria**:

- All tests pass
- Coverage > 90% for `_track_state.py` and `_track_helpers.py`
- Edge cases covered (empty state, single node, etc.)

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Medium (~300 lines of tests)

---

## Milestone 2: Widget and Layers

### Task 2.1: Create Layer Setup

**File**: `src/neurospatial/annotation/_track_widget.py` (partial)

**Dependencies**: Milestone 1 complete

**Actions**:

- [x] Define color constants (colorblind-safe):

  ```python
  NODE_COLOR = "#1f77b4"       # Blue
  EDGE_COLOR = "#ff7f0e"       # Orange
  START_NODE_COLOR = "#2ca02c" # Green
  SELECTED_COLOR = "#d62728"   # Red
  PREVIEW_COLOR = "#7f7f7f"    # Gray (dashed)
  ```

- [x] Implement `setup_track_layers(viewer) -> tuple[Shapes, Points]`:
  - Create Shapes layer for edges (middle, `shape_type="path"`)
  - Create Points layer for nodes (top, interactive)
  - Set proper z-ordering (edges below nodes)
  - Return `(edges_layer, nodes_layer)`

**Success Criteria**:

- Layers created in correct z-order
- Nodes are clickable (on top)
- Colors match spec

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Simple (~50 lines)

---

### Task 2.2: Implement Event Handlers

**File**: `src/neurospatial/annotation/_track_widget.py` (partial)

**Dependencies**: Task 2.1

**Actions**:

- [x] Implement `_sync_layers_from_state(state, nodes_layer, edges_layer)`:
  - Update Points layer data from `state.nodes`
  - Update Shapes layer data from `state.edges`
  - Highlight start node (larger size, green color)

- [x] Implement node click handler:
  - In `add_node` mode: Add node at click position
  - In `delete` mode: Delete nearest node within threshold
  - In `add_edge` mode: Select node for edge creation

- [x] Implement edge creation (two-click pattern):
  - First click: Set `state.edge_start_node`, show preview line
  - Second click: Call `state.add_edge()`, clear preview
  - Escape: Cancel edge creation

- [x] Implement edge preview:
  - Show dashed line from start node to cursor
  - Update on mouse move when `edge_start_node` is set
  - Use `PREVIEW_COLOR` with dashed style

- [x] Implement keyboard shortcuts:
  - `A` → add_node mode
  - `E` → add_edge mode
  - `X` → delete mode
  - `Shift+S` → set selected node as start
  - `Delete` → delete selected item
  - `Ctrl+Z` → undo
  - `Ctrl+Shift+Z` → redo
  - `Escape` → cancel edge / close without save
  - `Ctrl+Enter` → save and close

**Success Criteria**:

- All keyboard shortcuts work
- Two-click edge creation pattern works with visual feedback
- Preview line updates smoothly
- State changes reflect immediately in layers

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: High (~300 lines)

---

### Task 2.3: Create Control Widget

**File**: `src/neurospatial/annotation/_track_widget.py` (partial)

**Dependencies**: Tasks 2.1, 2.2

**Actions**:

- [x] Implement `create_track_widget(viewer, edges_layer, nodes_layer, state) -> QWidget`:
  - Use magicgui for widget construction

- [x] Add UI components:
  - Mode selector (RadioButtons: add_node/add_edge/delete)
  - Node list (Select widget with labels)
  - Edge list (Select widget)
  - Node label input (LineEdit for naming selected node)
  - Start node button ("Set as Start")
  - Validation status label
  - Delete buttons (Delete Node / Delete Edge)
  - Save and Close button

- [x] Add help text panel:

  ```
  Track Graph Builder
  -------------------
  1. Press A → Click to add nodes
  2. Press E → Click two nodes to connect
  3. Press X → Click node/edge to delete
  4. Select node → Shift+S to set as start
  5. Ctrl+Enter to save

  Shortcuts: A (add) | E (edge) | X (delete) | Ctrl+Z (undo)
  ```

- [x] Implement visual mode indicator:
  - Update status bar: `"Track Graph Mode: ADD_NODE"`
  - Show current mode in bold in widget header

- [x] Implement save validation dialog:
  - Block if < 2 nodes or < 1 edge
  - Show errors in modal dialog
  - Show warnings but allow save
  - Default start node to 0 with warning if unset

**Success Criteria**:

- Widget docks correctly in napari
- Mode selector syncs with keyboard shortcuts
- Node/edge lists update in real-time
- Save validates before closing
- Help text visible

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: High (~350 lines)

---

### Task 2.4: Write Widget Integration Tests

**File**: `tests/annotation/test_track_widget.py`

**Dependencies**: Tasks 2.1-2.3

**Actions**:

- [x] Test layer setup:
  - `test_setup_track_layers_returns_correct_types`
  - `test_layers_z_order_correct`
  - `test_node_layer_is_interactive`

- [x] Test event handlers (with mock viewer):
  - `test_click_in_add_node_mode_adds_node`
  - `test_click_in_delete_mode_removes_node`
  - `test_two_click_edge_creation`
  - `test_escape_cancels_edge_creation`
  - `test_keyboard_shortcuts_change_mode`

- [x] Test widget components:
  - `test_mode_selector_syncs_with_state`
  - `test_node_list_updates_on_add`
  - `test_edge_list_updates_on_add`
  - `test_start_node_button_updates_state`

- [x] Test validation:
  - `test_save_blocked_with_empty_graph`
  - `test_save_shows_warnings_but_allows`

**Success Criteria**:

- All tests pass with mocked napari viewer
- Event handlers trigger correct state changes
- Widget syncs bidirectionally with state

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Medium (~250 lines of tests)

---

## Milestone 3: Entry Point and Result

### Task 3.1: Implement TrackGraphResult

**File**: `src/neurospatial/annotation/track_graph.py`

**Dependencies**: Milestone 2 complete

**Actions**:

- [x] Create `TrackGraphResult` NamedTuple with fields:
  - `track_graph: nx.Graph | None`
  - `node_positions: list[tuple[float, float]]`
  - `edges: list[tuple[int, int]]`
  - `edge_order: list[tuple[int, int]]`
  - `edge_spacing: NDArray[np.float64]`
  - `node_labels: list[str]`
  - `start_node: int | None`
  - `pixel_positions: list[tuple[float, float]]`

- [x] Implement `to_environment(bin_size, edge_spacing=None, name="") -> Environment`:
  - Raise `ValueError` if `track_graph` is None
  - Use `self.edge_spacing` if `edge_spacing` not provided
  - Call `Environment.from_graph()` with all parameters

**Success Criteria**:

- All fields correctly typed
- `to_environment()` produces valid Environment
- Method handles edge_spacing override correctly

**Status**: COMPLETE (2025-11-27)

**Estimated Complexity**: Simple (~80 lines)

---

### Task 3.2: Implement annotate_track_graph Entry Point

**File**: `src/neurospatial/annotation/track_graph.py`

**Dependencies**: Task 3.1

**Actions**:

- [ ] Implement `annotate_track_graph()` with signature:

  ```python
  def annotate_track_graph(
      video_path: str | Path | None = None,
      *,
      image: NDArray[np.uint8] | None = None,
      frame_index: int = 0,
      calibration: VideoCalibration | None = None,
      initial_nodes: NDArray[np.float64] | None = None,
      initial_edges: list[tuple[int, int]] | None = None,
      initial_node_labels: list[str] | None = None,
  ) -> TrackGraphResult:
  ```

- [ ] Implement input validation:
  - Require either `video_path` or `image`
  - Validate `frame_index` in range for video

- [ ] Load background image:
  - Use `VideoReader` for video files (same as `annotate_video`)
  - Use provided image array directly

- [ ] Create napari viewer:
  - Add video frame as bottom Image layer (RGB)
  - Title: "Track Graph Builder"

- [ ] Initialize state:
  - Create `TrackBuilderState`
  - Populate with `initial_nodes`, `initial_edges`, `initial_node_labels`

- [ ] Set up layers and widget:
  - Call `setup_track_layers(viewer)`
  - Call `create_track_widget(viewer, edges_layer, nodes_layer, state)`
  - Dock widget

- [ ] Run napari and return result:
  - `napari.run()` (blocking)
  - Call `build_track_graph_result(state, calibration)`
  - Return `TrackGraphResult`

**Success Criteria**:

- Function opens napari with video frame
- Initial data populates correctly
- Returns valid `TrackGraphResult` on close
- Handles both video and image inputs

**Estimated Complexity**: Medium (~150 lines)

---

### Task 3.3: Add Module Exports

**File**: `src/neurospatial/annotation/__init__.py`

**Dependencies**: Tasks 3.1, 3.2

**Actions**:

- [ ] Add exports:

  ```python
  from neurospatial.annotation.track_graph import (
      annotate_track_graph,
      TrackGraphResult,
  )
  ```

- [ ] Update `__all__` list to include new exports

**Success Criteria**:

- `from neurospatial.annotation import annotate_track_graph` works
- `from neurospatial.annotation import TrackGraphResult` works

**Estimated Complexity**: Simple (< 10 lines)

---

### Task 3.4: Write End-to-End Tests

**File**: `tests/annotation/test_track_graph.py`

**Dependencies**: Tasks 3.1-3.3

**Actions**:

- [ ] Test input validation:
  - `test_requires_video_or_image`
  - `test_frame_index_out_of_range_raises`

- [ ] Test with mock viewer (skip napari.run):
  - `test_annotate_with_video_path`
  - `test_annotate_with_image_array`
  - `test_initial_data_populates_state`

- [ ] Test result construction:
  - `test_result_has_all_fields`
  - `test_result_to_environment_success`
  - `test_result_to_environment_no_graph_raises`
  - `test_result_edge_spacing_override`

- [ ] Test calibration:
  - `test_coordinates_in_pixels_without_calibration`
  - `test_coordinates_in_cm_with_calibration`
  - `test_pixel_positions_preserved`

- [ ] Test integration with Environment.from_graph:
  - `test_full_workflow_creates_valid_environment`
  - `test_environment_has_correct_bin_count`

**Success Criteria**:

- All tests pass
- Full workflow from annotation to Environment works
- Calibration transforms coordinates correctly

**Estimated Complexity**: Medium (~300 lines of tests)

---

## Milestone 4: Polish

### Task 4.1: Implement Edge Order Editing UI

**File**: `src/neurospatial/annotation/_track_widget.py` (enhancement)

**Dependencies**: Milestone 3 complete

**Actions**:

- [ ] Add edge order list widget:
  - Show edges in `edge_order` sequence
  - Allow drag-and-drop reordering
  - "Reset to Auto" button (re-run `infer_edge_layout()`)

- [ ] Add edge spacing input:
  - Default: use `infer_edge_layout()` values
  - Optional: let user set custom spacing per edge
  - "Use Default Spacing" checkbox

- [ ] Update result building:
  - Use manual edge_order if modified
  - Use custom edge_spacing if provided

**Success Criteria**:

- Edge order can be manually reordered
- Custom spacing can be set
- Auto-inference can be reset

**Estimated Complexity**: Medium (~150 lines)

---

### Task 4.2: Add 1D Preview (Optional Enhancement)

**File**: `src/neurospatial/annotation/_track_widget.py` (enhancement)

**Dependencies**: Task 4.1

**Actions**:

- [ ] Add "Preview Linearization" button
- [ ] Open matplotlib figure showing linearized track layout:
  - Use `track_linearization.plot_track_graph()` or equivalent
  - Show edge order visually
  - Update when edge_order changes

**Success Criteria**:

- Preview shows linearized track structure
- Updates when edge order changes
- Helps user verify correct ordering

**Estimated Complexity**: Medium (~100 lines)

---

### Task 4.3: Update CLAUDE.md Documentation

**File**: `CLAUDE.md`

**Dependencies**: Milestone 3 complete

**Actions**:

- [ ] Add to Quick Reference:

  ```python
  # Annotate track graph interactively (v0.X.0+)
  from neurospatial.annotation import annotate_track_graph

  result = annotate_track_graph("maze.mp4", calibration=calib)
  env = result.to_environment(bin_size=2.0)
  ```

- [ ] Add to Common Patterns section:
  - Track graph annotation workflow
  - Using with Environment.from_graph()
  - Calibration for pixel-to-cm conversion

- [ ] Add keyboard shortcuts table:

  | Key | Action |
  |-----|--------|
  | `A` | Add node mode |
  | `E` | Add edge mode |
  | `X` | Delete mode |
  | ... | ... |

- [ ] Add troubleshooting section:
  - "No start node set" warning
  - Edge order issues
  - Calibration coordinate conventions

**Success Criteria**:

- CLAUDE.md includes complete annotate_track_graph documentation
- Examples are correct and tested
- Keyboard shortcuts documented

**Estimated Complexity**: Medium (~100 lines of docs)

---

### Task 4.4: Add Docstrings and Examples

**Files**: All new files

**Dependencies**: All previous tasks

**Actions**:

- [ ] Add NumPy-style docstrings to all public functions:
  - `annotate_track_graph()`
  - `TrackGraphResult`
  - `TrackBuilderState`

- [ ] Add doctests:

  ```python
  >>> result = annotate_track_graph(image=frame)  # doctest: +SKIP
  >>> env = result.to_environment(bin_size=2.0)  # doctest: +SKIP
  ```

- [ ] Run `uv run pytest --doctest-modules src/neurospatial/annotation/track_graph.py`

**Success Criteria**:

- All public APIs have complete docstrings
- Docstrings follow NumPy format
- Examples are accurate

**Estimated Complexity**: Medium (~200 lines of docs)

---

### Task 4.5: Final Review and Testing

**Dependencies**: All previous tasks

**Actions**:

- [ ] Run full test suite: `uv run pytest tests/annotation/`
- [ ] Run type checking: `uv run mypy src/neurospatial/annotation/`
- [ ] Run linting: `uv run ruff check src/neurospatial/annotation/`
- [ ] Manual testing:
  - Open video file, create track graph
  - Open image, create track graph
  - Use calibration
  - Test all keyboard shortcuts
  - Test undo/redo
  - Test save validation
- [ ] Verify integration: `result.to_environment()` creates valid Environment

**Success Criteria**:

- All tests pass
- Mypy passes without errors
- Ruff passes without errors
- Manual testing confirms UX works as designed
- Integration with Environment.from_graph() verified

---

## Summary

| Milestone | Tasks | Estimated Lines | Key Deliverable |
|-----------|-------|-----------------|-----------------|
| 1. Core Infrastructure | 4 | ~650 | State management, graph building |
| 2. Widget and Layers | 4 | ~950 | napari UI, event handlers |
| 3. Entry Point | 4 | ~540 | `annotate_track_graph()` API |
| 4. Polish | 5 | ~550 | Edge ordering, docs, testing |
| **Total** | **17** | **~2700** | Complete track graph annotation |

## Dependencies Graph

```
Task 1.1 ─┬─> Task 1.2 ─┬─> Task 1.3 ─┬─> Task 1.4
          │            │            │
          └────────────┴────────────┴─> Milestone 2
                                        │
Task 2.1 ─> Task 2.2 ─> Task 2.3 ─> Task 2.4
                                        │
                                        v
Task 3.1 ─> Task 3.2 ─> Task 3.3 ─> Task 3.4
                                        │
                                        v
Task 4.1 ─> Task 4.2 (optional)
    │
Task 4.3 ─> Task 4.4 ─> Task 4.5
```

## Notes

- Tasks should be completed in order within each milestone
- Milestones can be reviewed independently
- Task 4.2 (1D Preview) is optional enhancement
- All code should follow CLAUDE.md patterns and conventions
