# Plan: Track Graph Annotation for neurospatial

## Overview

Add interactive track graph building functionality to the annotation module, allowing users to define track graphs (nodes and edges) on video frames using napari. The output integrates with `Environment.from_graph()` for creating 1D linearized track environments.

## Requirements

### From track_linearization

The `make_track_from_image_interactive` function in `track_linearization` allows users to:

1. Click to add nodes (waypoints)
2. Click two nodes to create edges (track segments) - **two-click pattern**
3. Delete nodes/edges
4. Output a track graph for `get_linearized_position()`

### For neurospatial Integration

The annotation must produce:

1. **track_graph**: NetworkX Graph with nodes having `'pos'` attribute
2. **edge_order**: Ordered list of edges for linearization
3. **edge_spacing**: Spacing between edges (user configurable)
4. **bin_size**: Bin size for discretization

These are the exact parameters needed for `Environment.from_graph()`.

## Architecture Design

### Layer Structure in napari

Like `annotate_video`, we display a video frame as the background image layer. Track graph annotation uses **three layers**:

| Layer | Purpose | napari Layer Type | Z-Order |
|-------|---------|-------------------|---------|
| Video Frame | Background reference image | `Image` (RGB) | Bottom |
| Edges | Track segment connections | `Shapes` (path type) | Middle |
| Nodes | Track waypoints | `Points` | Top |

**Layer order matters**: Nodes on top so they're clickable, edges in middle for visibility, video at bottom as reference.

```python
# Layer creation order (bottom to top)
viewer.add_image(frame, name="video_frame", rgb=True)  # Bottom
edges_layer = viewer.add_shapes(...)                    # Middle
nodes_layer = viewer.add_points(...)                    # Top (interactive)
```

This matches `annotate_video`'s pattern where the video frame provides spatial context for annotation.

### File Structure

```
src/neurospatial/annotation/
    __init__.py          # Add exports: annotate_track_graph, TrackGraphResult
    _track_types.py      # NEW: TrackGraphMode type alias
    _track_state.py      # NEW: TrackBuilderState dataclass
    _track_helpers.py    # NEW: Graph building helpers
    _track_widget.py     # NEW: napari widget for track building
    track_graph.py       # NEW: annotate_track_graph() entry point
    ... (existing files unchanged)
```

### Data Flow

```
User Interaction --> napari Layers --> TrackBuilderState --> TrackGraphResult
                                                                    |
                                                                    v
                                                        Environment.from_graph()
```

## Detailed Component Design

### 1. Types (`_track_types.py`)

```python
from typing import Literal

# Interaction modes for track graph builder
TrackGraphMode = Literal["add_node", "add_edge", "delete"]
```

### 2. State Management (`_track_state.py`)

Pure state object (napari-independent, testable):

```python
@dataclass
class TrackBuilderState:
    """Track graph builder state."""

    mode: TrackGraphMode = "add_node"
    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)  # Optional labels

    # Linearization control
    start_node: int | None = None  # For infer_edge_layout() DFS start

    # Edge creation state (transient)
    edge_start_node: int | None = None  # First node in two-click edge creation

    # Undo/redo stacks (critical UX feature)
    # Snapshot format: cheap primitives only, explicit deep copy
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)
    _max_undo_depth: int = 50  # Limit memory usage

    def _snapshot(self) -> dict:
        """Create serializable snapshot of mutable state (explicit deep copy)."""
        return {
            "nodes": [tuple(n) for n in self.nodes],      # list of tuples (immutable)
            "edges": [tuple(e) for e in self.edges],      # list of tuples (immutable)
            "node_labels": list(self.node_labels),         # shallow copy of strings
            "start_node": self.start_node,                 # int or None (immutable)
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        """Restore state from snapshot."""
        self.nodes = [tuple(n) for n in snapshot["nodes"]]
        self.edges = [tuple(e) for e in snapshot["edges"]]
        self.node_labels = list(snapshot["node_labels"])
        self.start_node = snapshot["start_node"]

    def _save_for_undo(self) -> None:
        """Save current state before mutation. Clears redo stack."""
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()  # New action invalidates redo
        # Limit stack depth
        if len(self.undo_stack) > self._max_undo_depth:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore previous state. Returns True if undo was possible."""
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._snapshot())  # Save current for redo
        self._restore_snapshot(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        """Restore next state. Returns True if redo was possible."""
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._snapshot())  # Save current for undo
        self._restore_snapshot(self.redo_stack.pop())
        return True

    def add_node(self, x: float, y: float, label: str | None = None) -> int:
        """Add node, return its index. Saves snapshot for undo."""

    def add_edge(self, node1: int, node2: int) -> bool:
        """Add edge if valid (no self-loops, no duplicates). Saves snapshot."""

    def delete_node(self, idx: int) -> None:
        """Delete node and all connected edges. Reindex remaining nodes.

        Also updates start_node if affected by reindexing:
        - If deleted node IS start_node: set start_node = None
        - If deleted node index < start_node: decrement start_node by 1
        Saves snapshot before mutation.
        """

    def delete_edge(self, edge_idx: int) -> None:
        """Delete edge by index. Saves snapshot."""

    def set_start_node(self, idx: int) -> None:
        """Designate node as start for linearization."""

    def find_nearest_node(self, x: float, y: float, threshold: float) -> int | None:
        """Find node within threshold distance."""

    def to_track_graph(self) -> nx.Graph:
        """Build NetworkX graph from current state (pixel coordinates).

        NOTE: For internal validation only. The final output graph is built
        by build_track_graph_from_positions() using transformed coordinates.
        """

    def validate(self) -> dict[str, Any]:
        """Validate graph using check_track_graph_validity().

        Builds temporary graph in pixel coordinates for validation.
        """

    def is_valid_for_save(self) -> tuple[bool, list[str], list[str]]:
        """Check if state is valid for saving. Returns (is_valid, errors, warnings)."""
        errors = []
        warnings = []

        if len(self.nodes) < 2:
            errors.append("Need at least 2 nodes")
        if len(self.edges) < 1:
            errors.append("Need at least 1 edge")
        if self.start_node is None and len(self.nodes) > 0:
            # Default to node 0 (matches infer_edge_layout behavior)
            warnings.append("No start node set. Defaulting to Node 0.")

        return len(errors) == 0, errors, warnings

    def get_effective_start_node(self) -> int | None:
        """Get start node, defaulting to 0 if not explicitly set."""
        if self.start_node is not None:
            return self.start_node
        if len(self.nodes) > 0:
            return 0  # Default: first created node
        return None
```

### 3. Result Type (`track_graph.py`)

```python
class TrackGraphResult(NamedTuple):
    """Result from track graph annotation session.

    Attributes
    ----------
    track_graph : nx.Graph or None
        NetworkX graph with node 'pos' and edge 'distance'/'edge_id' attributes.
        Created via track_linearization.make_track_graph().
    node_positions : list[tuple[float, float]]
        Node coordinates in output units (cm if calibrated).
    edges : list[tuple[int, int]]
        Edge connections as (node_i, node_j) tuples.
    edge_order : list[tuple[int, int]]
        Ordered edge list for linearization (from infer_edge_layout).
    edge_spacing : NDArray[np.float64]
        Spacing between consecutive edges (from infer_edge_layout).
    node_labels : list[str]
        Optional labels for each node (e.g., "start", "goal").
    start_node : int or None
        Designated start node for linearization.
    pixel_positions : list[tuple[float, float]]
        Original node coordinates in pixels.

    Methods
    -------
    to_environment(bin_size, **kwargs) -> Environment
        Convenience method to create Environment.from_graph().
    """

    track_graph: nx.Graph | None
    node_positions: list[tuple[float, float]]
    edges: list[tuple[int, int]]
    edge_order: list[tuple[int, int]]
    edge_spacing: NDArray[np.float64]
    node_labels: list[str]
    start_node: int | None
    pixel_positions: list[tuple[float, float]]

    def to_environment(
        self,
        bin_size: float,
        edge_spacing: float | Sequence[float] | None = None,
        name: str = "",
    ) -> Environment:
        """Create Environment from annotated track graph.

        Parameters
        ----------
        bin_size : float
            Bin size for discretization.
        edge_spacing : float or Sequence, optional
            Override inferred edge_spacing. If None, uses self.edge_spacing.
        name : str, optional
            Name for the environment.
        """
        if self.track_graph is None:
            raise ValueError("Cannot create Environment: no track graph")

        spacing = edge_spacing if edge_spacing is not None else self.edge_spacing
        return Environment.from_graph(
            graph=self.track_graph,
            edge_order=self.edge_order,
            edge_spacing=spacing,
            bin_size=bin_size,
            name=name,
        )
```

### 4. Widget (`_track_widget.py`)

Magicgui-based widget with:

**UI Components:**

- Mode selector: RadioButtons for add_node/add_edge/delete
- Node list: Select widget showing nodes with labels
- Edge list: Select widget showing edges
- **Node label input**: LineEdit for naming selected node
- **Start node selector**: Button to designate start node for linearization
- **Validation status**: Label showing graph validity (from `check_track_graph_validity()`)
- Delete buttons: Delete selected node/edge
- Save and Close button

**Keyboard Shortcuts (matching track_linearization):**

| Key | Action |
|-----|--------|
| `A` | Switch to add_node mode (Add) |
| `E` | Switch to add_edge mode (Edge) |
| `X` | Switch to delete mode |
| `Shift+S` | Set selected node as start |
| `Delete` | Delete selected item |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Shift+Z` | Redo |
| `Escape` | Cancel edge creation / close without save |
| `Ctrl+Enter` | Save and close |

**Why A/E/X?** Matches `track_linearization.make_track_from_image_interactive()` for ecosystem consistency.

**Layer Setup:**

```python
def setup_track_layers(viewer: napari.Viewer) -> tuple[Shapes, Points]:
    """Create Shapes layer for edges and Points layer for nodes.

    Returns (edges_layer, nodes_layer) to match unpacking order in annotate_track_graph.
    """
    # Colorblind-safe palette (defined below in UX Requirements)
    NODE_COLOR = "#1f77b4"  # Blue
    EDGE_COLOR = "#ff7f0e"  # Orange

    # Edges layer (middle - below nodes for clickability)
    edges = viewer.add_shapes(
        name="Track Edges",
        shape_type="path",
        edge_color=EDGE_COLOR,
        edge_width=3,
    )

    # Nodes layer (top - interactive, clickable)
    nodes = viewer.add_points(
        name="Track Nodes",
        size=15,
        face_color=NODE_COLOR,
        border_color="white",
        border_width=2,
    )

    return edges, nodes  # Order: edges first (middle layer), nodes second (top layer)
```

**Widget Factory:**

```python
def create_track_widget(
    viewer: napari.Viewer,
    edges_layer: Shapes,
    nodes_layer: Points,
    state: TrackBuilderState,
) -> QWidget:
    """Create the track graph builder widget.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    edges_layer : Shapes
        The shapes layer for edges.
    nodes_layer : Points
        The points layer for nodes.
    state : TrackBuilderState
        The shared state object (modified in place by handlers).

    Returns
    -------
    QWidget
        The magicgui-based control widget.
    """
```

**Event Handlers:**

- `on_points_data_changed`: Sync state when nodes added/removed
- `on_click`: Handle node/edge creation based on mode
- `on_edge_preview`: Show preview line while creating edge

### 5. Entry Point (`track_graph.py`)

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
    """
    Launch interactive napari annotation to build a track graph.

    Opens a napari viewer with a video frame or image as background.
    Users can:
    - Click to add nodes (waypoints on the track)
    - Click two nodes to create edges (two-click pattern)
    - Delete nodes/edges as needed
    - Designate start node for linearization

    Parameters
    ----------
    video_path : str or Path, optional
        Path to video file. Extracts frame at frame_index.
        Either video_path or image must be provided.
    image : NDArray, optional
        Background image array (H, W, 3) RGB uint8.
        Use this for static images or pre-loaded frames.
        Either video_path or image must be provided.
    frame_index : int, default=0
        Which frame to extract from video (ignored if image provided).
    calibration : VideoCalibration, optional
        Pixel-to-cm transform. If provided, output coordinates are in cm.
    initial_nodes : NDArray, optional
        Pre-existing node positions to display for editing.
    initial_edges : list of tuple, optional
        Pre-existing edge connections.
    initial_node_labels : list of str, optional
        Labels for initial nodes (e.g., ["start", "junction", "goal"]).

    Returns
    -------
    TrackGraphResult
        Named tuple with track_graph, node_positions, edges, edge_order.

    Examples
    --------
    >>> # From video file
    >>> from neurospatial.annotation import annotate_track_graph
    >>> result = annotate_track_graph("maze.mp4")
    >>> env = result.to_environment(bin_size=2.0)

    >>> # From static image
    >>> import matplotlib.pyplot as plt
    >>> img = plt.imread("maze_photo.png")
    >>> result = annotate_track_graph(image=img)

    >>> # With calibration (convert pixels to cm)
    >>> from neurospatial.transforms import VideoCalibration
    >>> result = annotate_track_graph("maze.mp4", calibration=calib)
    >>> # node_positions now in cm
    """
```

**Implementation flow:**

```python
def annotate_track_graph(...) -> TrackGraphResult:
    # 1. Validate inputs
    if video_path is None and image is None:
        raise ValueError("Either video_path or image must be provided")

    # 2. Load background image (same pattern as annotate_video)
    if video_path is not None:
        from neurospatial.animation._video_io import VideoReader
        reader = VideoReader(str(video_path))
        if frame_index >= reader.n_frames:
            raise IndexError(
                f"frame_index {frame_index} out of range. "
                f"Video has {reader.n_frames} frames (indices 0-{reader.n_frames - 1})."
            )
        frame = reader[frame_index]
    else:
        frame = image

    # 3. Create napari viewer with video frame as bottom layer
    viewer = napari.Viewer(title="Track Graph Builder")
    viewer.add_image(frame, name="video_frame", rgb=True)

    # 4. Initialize state (with optional initial data)
    state = TrackBuilderState()
    if initial_nodes is not None:
        for i, pos in enumerate(initial_nodes):
            label = initial_node_labels[i] if initial_node_labels else None
            state.add_node(pos[0], pos[1], label=label)
    if initial_edges is not None:
        for node1, node2 in initial_edges:
            state.add_edge(node1, node2)

    # 5. Add track graph layers (edges, then nodes)
    edges_layer, nodes_layer = setup_track_layers(viewer)

    # 6. Add control widget
    widget = create_track_widget(viewer, edges_layer, nodes_layer, state)
    viewer.window.add_dock_widget(widget)

    # 7. Run napari (blocking)
    napari.run()

    # 8. Extract and transform results
    return build_track_graph_result(state, calibration)


def build_track_graph_result(
    state: TrackBuilderState,
    calibration: VideoCalibration | None,
) -> TrackGraphResult:
    """Build TrackGraphResult from annotation state.

    Transforms coordinates using calibration and builds the track graph
    using track_linearization.make_track_graph().

    IMPORTANT: The track_graph node 'pos' attributes use the SAME
    coordinate system as node_positions (cm if calibrated, else pixels).
    This ensures consistency when using Environment.from_graph().

    Parameters
    ----------
    state : TrackBuilderState
        Final state from annotation session.
    calibration : VideoCalibration or None
        Pixel-to-cm transform. If None, coordinates stay in pixels.

    Returns
    -------
    TrackGraphResult
        Complete result with track graph and all metadata.
    """
    # Store original pixel positions
    pixel_positions = list(state.nodes)

    # Transform to output coordinates (cm if calibrated)
    node_positions = transform_nodes_to_output(state.nodes, calibration)

    # Build track graph using TRANSFORMED coordinates
    # The graph's node 'pos' attributes must match node_positions
    if len(state.nodes) >= 2 and len(state.edges) >= 1:
        track_graph = build_track_graph_from_positions(
            node_positions,  # Use transformed positions, not state.nodes
            state.edges,
        )
        edge_order, edge_spacing = infer_edge_layout(
            track_graph,
            start_node=state.get_effective_start_node(),
        )
    else:
        track_graph = None
        edge_order = []
        edge_spacing = np.array([])

    return TrackGraphResult(
        track_graph=track_graph,
        node_positions=node_positions,
        edges=list(state.edges),
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        node_labels=list(state.node_labels),
        start_node=state.get_effective_start_node(),
        pixel_positions=pixel_positions,
    )
```

## Coordinate Handling

**IMPORTANT**: Reuse existing `VideoCalibration` utilities from `neurospatial.transforms` to ensure identical behavior with `annotate_video` and other tools.

| Stage | Coordinates |
|-------|-------------|
| napari display | (row, col), origin top-left |
| Internal storage | (x, y) pixels |
| Output (no calib) | (x, y) pixels |
| Output (with calib) | (x, y) cm, origin bottom-left |

Transform flow:

```
napari click → (row, col) → swap to (x, y) px → calibration → (x, y) cm
```

### Implementation (reuse existing utilities)

```python
from neurospatial.transforms import VideoCalibration
from track_linearization import make_track_graph

def transform_nodes_to_output(
    nodes_px: list[tuple[float, float]],
    calibration: VideoCalibration | None,
) -> list[tuple[float, float]]:
    """Transform node positions from pixels to output coordinates.

    Reuses VideoCalibration.transform_px_to_cm() for consistency
    with annotate_video and other annotation tools.
    """
    if calibration is None:
        return nodes_px  # Return pixel coordinates

    import numpy as np
    nodes_array = np.array(nodes_px)
    # Use the SAME transform method as annotate_video
    nodes_cm = calibration.transform_px_to_cm(nodes_array)
    return [tuple(pos) for pos in nodes_cm]


def build_track_graph_from_positions(
    node_positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
) -> nx.Graph:
    """Build track graph from transformed node positions.

    Uses track_linearization.make_track_graph() to ensure proper
    'distance' and 'edge_id' edge attributes.

    Parameters
    ----------
    node_positions : list of tuple
        Node coordinates in output units (cm or pixels).
    edges : list of tuple
        Edge connections as (node_i, node_j) pairs.

    Returns
    -------
    nx.Graph
        Track graph with node 'pos' and edge 'distance'/'edge_id' attributes.
    """
    return make_track_graph(
        node_positions=node_positions,
        edges=edges,
    )
```

### Shared coordinate utilities

These functions from the existing annotation module should be reused:

| Function | Location | Purpose |
|----------|----------|---------|
| `VideoCalibration.transform_px_to_cm()` | `neurospatial.transforms` | Pixel → cm transform |
| `calibrate_from_landmarks()` | `neurospatial.transforms` | Create calibration from point pairs |
| `calibrate_from_scale_bar()` | `neurospatial.transforms` | Create calibration from scale bar |
| napari (row,col) → (x,y) swap | `annotation.converters` | Coordinate axis swap |

## Integration with track_linearization

### Required Graph Structure

The output track graph must match track_linearization's expected format:

**Node attributes:**

```python
node_attrs = {'pos': (x, y)}  # 2D coordinates
```

**Edge attributes:**

```python
edge_attrs = {
    'distance': float,  # Euclidean length (auto-computed)
    'edge_id': int,     # Unique edge identifier (auto-assigned)
}
```

We'll use `track_linearization.make_track_graph()` internally to ensure proper formatting.

### Key Functions to Leverage

| Function | Purpose | How We Use It |
|----------|---------|---------------|
| `make_track_graph()` | Create graph with proper attributes | Build output from annotated nodes/edges |
| `infer_edge_layout()` | Auto-determine edge order via DFS | Offer as default edge ordering strategy |
| `validate_track_graph()` | Check graph validity | Validate before returning result |
| `check_track_graph_validity()` | Detailed validation report | Show warnings in UI |

### Edge Order Inference

Use `track_linearization.infer_edge_layout()` for automatic edge ordering:

```python
from track_linearization import infer_edge_layout

# Auto-infer edge order from graph structure
edge_order, edge_spacing = infer_edge_layout(
    track_graph,
    start_node=user_selected_start_node,  # Optional: let user designate start
    spacing_between_unconnected_components=15.0,
)
```

**Start node selection**: Allow users to designate a "start node" in the UI. This affects how DFS traverses the graph and produces more intuitive linearization for complex mazes.

### Validation Integration

Before returning results, validate the graph:

```python
from track_linearization import check_track_graph_validity

report = check_track_graph_validity(track_graph)
if not report['valid']:
    # Show errors in UI before allowing save
    for error in report['errors']:
        viewer.status = f"Error: {error}"
```

## Edge Order Determination

Four approaches (prioritized):

1. **`infer_edge_layout()` (recommended)**: Auto-determine via DFS from start node
   - Uses track_linearization's proven algorithm
   - Handles branching tracks and disconnected components
   - User can select start node for different traversal orders

2. **Creation order (fallback)**: Edges ordered by when they were created
   - Simple and predictable
   - Works well for linear tracks built sequentially

3. **Manual ordering**: User reorders in the widget
   - Drag-and-drop interface for edge list
   - Full control for complex layouts

4. **Graph traversal from selected node**: Let user pick start node, DFS from there
   - Combines auto-inference with user guidance

## Implementation Steps

### Phase 1: Core Infrastructure

1. Create `_track_types.py` with `TrackGraphMode`
2. Create `_track_state.py` with `TrackBuilderState`
3. Create `_track_helpers.py` with graph building utilities
4. Write unit tests for state management

### Phase 2: Widget and Layers

5. Create `_track_widget.py` with napari widget
6. Implement layer setup (Points + Shapes)
7. Implement event handlers for node/edge creation
8. Write integration tests

### Phase 3: Entry Point and Result

9. Create `track_graph.py` with `annotate_track_graph()`
10. Implement `TrackGraphResult` with `to_environment()` method
11. Add exports to `__init__.py`
12. Write end-to-end tests

### Phase 4: Polish

13. Add edge order editing UI
14. Handle calibration transforms
15. Update CLAUDE.md documentation
16. Add docstrings and examples

## UX Requirements (from Review)

### Critical UX Features

1. **Visual mode indicator**
   - Status bar: `"Track Graph Mode: ADD_NODE"`
   - Cursor change: standard → crosshair → forbidden for each mode
   - Widget header shows current mode in bold

2. **Two-click edge creation pattern**
   - Click first node → status shows "Edge start: Node 3"
   - Click second node → create edge
   - Press Escape to cancel mid-creation
   - Show dashed preview line while creating

3. **Undo/Redo stack**
   - `Ctrl+Z` undo, `Ctrl+Shift+Z` redo
   - Store state snapshots before each mutation
   - Essential for complex mazes (20+ nodes)

4. **Start node visibility**
   - Larger size (20 vs 15)
   - Different color (green)
   - Text label "START" overlay
   - High-contrast border (yellow)

5. **Save validation**
   - **Default start node**: If no start node set, default to Node 0 with warning
     (matches `infer_edge_layout()` behavior)
   - Block if empty graph: require 2+ nodes, 1+ edge
   - Show validation errors in modal dialog, not just status bar
   - Warnings shown but don't block save

### Colorblind-Safe Palette

```python
# Colorblind-safe colors (blue/orange/green)
NODE_COLOR = "#1f77b4"       # Blue (normal nodes)
EDGE_COLOR = "#ff7f0e"       # Orange (edges)
START_NODE_COLOR = "#2ca02c" # Green (start node)
SELECTED_COLOR = "#d62728"   # Red (selected highlight)
PREVIEW_COLOR = "#7f7f7f"    # Gray (edge preview, dashed)
```

### Help Text (shown in widget)

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

## Design Decisions (Resolved from track_linearization Analysis)

1. **Edge order strategy**: Use `infer_edge_layout()` as default with optional manual override
   - Auto-inference handles most cases well
   - Provide start node selector for user control
   - Manual reordering as fallback for complex layouts

2. **Node labeling**: Yes, support optional labels
   - Useful for: "start", "goal", "junction_A", "reward_left"
   - Maps to region names when creating Environment
   - Store in node features, not graph attributes

3. **Graph building**: Use `track_linearization.make_track_graph()` internally
   - Ensures proper `distance` and `edge_id` attributes
   - Validated by `check_track_graph_validity()` before save

4. **Preview mode**: Yes, show edge preview while creating
   - Dashed line from start node to cursor
   - Matches track_linearization's UX

5. **Start node designation**: Add "Set as Start" button/shortcut
   - Affects `infer_edge_layout()` traversal order
   - Visual indicator (different color/icon) for start node

## Open Questions

1. **Edge spacing UI**: Should users be able to set custom spacing between edges, or use the default from `infer_edge_layout()`?
   - Default: 15.0 for unconnected components, 0.0 for connected
   - Could add a "spacing" input field

2. **1D preview**: Should we show a matplotlib preview of the linearized track layout?
   - Could use `plot_graph_as_1D()` equivalent
   - Helps users verify edge order before saving

## Testing Strategy

1. **Unit tests** (`tests/annotation/test_track_state.py`):
   - State transitions (add/delete nodes/edges)
   - Edge validity checks (no self-loops, no duplicates)
   - Coordinate transformations
   - Graph building from state

2. **Integration tests** (`tests/annotation/test_track_graph.py`):
   - Full annotation workflow (mock napari viewer)
   - Calibration transforms
   - `to_environment()` integration

3. **Visual tests** (manual):
   - Node rendering on video frame
   - Edge rendering between nodes
   - Mode switching and keyboard shortcuts

## Success Criteria

- [ ] Users can interactively build track graphs on video frames
- [ ] Output integrates seamlessly with `Environment.from_graph()`
- [ ] Coordinate calibration works correctly
- [ ] Edge order can be determined automatically or manually adjusted
- [ ] Tests pass and coverage is adequate
- [ ] Documentation updated with examples
