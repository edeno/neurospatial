# Troubleshooting Guide

Common errors and their fixes.

---

## Common Gotchas

### 1. Always use `uv run`

❌ **Wrong:**

```bash
python script.py
pytest
pip install package
```

✅ **Right:**

```bash
uv run python script.py
uv run pytest
uv add package
```

### 2. Check `_is_fitted` state

❌ **Wrong:**

```python
env = Environment()  # Not fitted!
env.bin_at([10.0, 5.0])  # RuntimeError
```

✅ **Right:**

```python
env = Environment.from_samples(positions, bin_size=2.0)
env.bin_at([10.0, 5.0])  # Works
```

### 3. bin_size is required

❌ **Wrong:**

```python
env = Environment.from_samples(data)  # TypeError
```

✅ **Right:**

```python
env = Environment.from_samples(positions, bin_size=2.0)
```

### 4. Regions are immutable

❌ **Wrong:**

```python
env.regions['goal'].point = new_point  # AttributeError
```

⚠️ **Discouraged** (emits warning):

```python
env.regions['goal'] = new_region  # UserWarning
```

✅ **Right:**

```python
env.regions.update_region('goal', point=new_point)  # No warning
env.regions.add('new_goal', point=point)
del env.regions['old_goal']
```

### 5. Check `is_1d` before linearization

❌ **Wrong:**

```python
env = Environment.from_samples(positions, bin_size=2.0)  # 2D grid
linear_pos = env.to_linear(position)  # AttributeError
```

✅ **Right:**

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    bin_idx = env.bin_at(position)
```

### 6. Protocol, not inheritance

❌ **Wrong:**

```python
class MyLayout(LayoutEngine):  # LayoutEngine is Protocol, not class
    pass
```

✅ **Right:**

```python
class MyLayout:
    """Implements LayoutEngine protocol."""
    def build(self, ...): ...
    def point_to_bin_index(self, ...): ...
    # Implement all required methods
```

### 7. NumPy docstrings required

❌ **Wrong** (Google style):

```python
def foo(x, y):
    """Does foo.

    Args:
        x: First parameter
        y: Second parameter
    """
```

✅ **Right** (NumPy style):

```python
def foo(x, y):
    """Does foo.

    Parameters
    ----------
    x : type
        First parameter
    y : type
        Second parameter
    """
```

### 8. Memory safety checks (v0.2.1+)

⚠️ **Creates large grid** (will warn but succeed):

```python
positions = np.random.uniform(0, 100000, (1000, 2))
env = Environment.from_samples(positions, bin_size=1.0)  # ResourceWarning
```

✅ **Better:**

```python
# Option 1: Increase bin_size
env = Environment.from_samples(positions, bin_size=10.0)

# Option 2: Filter active bins
env = Environment.from_samples(positions, bin_size=1.0, infer_active_bins=True)

# Option 3: Disable warning (if intentional)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

### 9. Overlay temporal alignment (v0.4.0+)

❌ **Wrong** (missing frame_times):

```python
# frame_times is REQUIRED
# env.animate_fields(fields_10hz, overlays=[overlay])  # TypeError
```

✅ **Right:**

```python
# Position at 120 Hz, fields at 10 Hz - provide timestamps
position_overlay = PositionOverlay(
    data=trajectory_120hz,
    times=timestamps_120hz  # Overlay timestamps
)
env.animate_fields(
    fields_10hz,
    frame_times=timestamps_10hz,  # Field timestamps - REQUIRED
    overlays=[position_overlay],  # Auto-interpolated
)
```

### 10. HTML backend overlay limitations (v0.4.0+)

⚠️ **Will warn:**

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[bodypart_overlay, head_direction_overlay],
    backend="html"  # UserWarning: HTML doesn't support these overlays
)
```

✅ **Right:**

```python
# Option 1: Use only supported overlays
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[position_overlay],
    show_regions=True,
    backend="html"
)

# Option 2: Use different backend
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[bodypart_overlay],
    backend="napari"
)
```

### 11. Overlay coordinates use environment space

❌ **Wrong:**

```python
# Don't manually swap x,y or invert
positions_wrong = np.column_stack([y_coords, x_coords])
overlay = PositionOverlay(data=positions_wrong)  # Display will be wrong!
```

✅ **Right:**

```python
# Pass (x, y) format - system handles conversion
positions = np.column_stack([x_coords, y_coords])
overlay = PositionOverlay(data=positions)
```

### 12. VideoOverlay requires 2D environments (v0.5.0+)

❌ **Wrong:**

```python
env_1d = Environment.from_graph(track_graph, ...)  # 1D environment
video = VideoOverlay(source="session.mp4", calibration=calib)
# env_1d.animate_fields(fields, frame_times=frame_times, overlays=[video])  # ValueError
```

✅ **Right:**

```python
env_2d = Environment.from_samples(positions, bin_size=2.0)  # 2D grid
video = VideoOverlay(source="session.mp4", calibration=calib)
env_2d.animate_fields(fields, frame_times=frame_times, overlays=[video])
```

### 13. Animation API migration (v0.15.0+)

❌ **Old code** (v0.14.x):

```python
env.animate_fields(fields, fps=30)
```

✅ **New code** (v0.15.0+):

```python
frame_times = np.arange(len(fields)) / 30.0  # Create timestamps
env.animate_fields(fields, frame_times=frame_times)  # speed=1.0 default
env.animate_fields(fields, frame_times=frame_times, speed=0.1)  # Slow motion
```

**Migration:**

1. Add `frame_times` parameter (use timestamps from your data)
2. Replace `fps=X` with `speed=Y` where `speed = X / sample_rate_hz`
3. For slow motion: use `speed < 1.0`
4. For fast forward: use `speed > 1.0`

---

## Error Messages

### `ModuleNotFoundError: No module named 'neurospatial'`

**Cause**: Dependencies not installed or wrong Python environment.

**Solution**:

```bash
uv sync  # From project root
uv run python -c "import neurospatial; print(neurospatial.__file__)"
```

### Tests fail with import errors

**Cause**: Running pytest without `uv run`.

**Solution**:

```bash
uv run pytest  # Not: pytest
```

### `RuntimeError: Environment must be fitted before calling this method`

**Cause**: Calling spatial queries on unfitted Environment.

**Solution**: Use factory methods:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Not: Environment()
```

### UserWarning when overwriting a region

**Cause**: Using assignment to overwrite existing region.

**Solution**:

```python
env.regions.update_region('goal', point=new_point)  # Preferred - no warning
# or
env.regions['goal'] = new_region  # Works but emits UserWarning
```

### `AttributeError: 'Environment' object has no attribute 'to_linear'`

**Cause**: Calling `to_linear()` on N-D environment.

**Solution**:

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    bin_idx = env.bin_at(position)
```

### `ValueError: No active bins found`

**Cause**: bin_size too large, threshold too high, or data too sparse.

**Solution**: Read the detailed error message for diagnostics. Common fixes:

```python
# Reduce bin_size
env = Environment.from_samples(positions, bin_size=1.0)  # Was 10.0

# Lower threshold
env = Environment.from_samples(positions, bin_size=2.0, bin_count_threshold=1)

# Enable morphological operations
env = Environment.from_samples(positions, bin_size=2.0, dilate=True, fill_holes=True)
```

### Pre-commit hooks fail on commit

**Cause**: Linting or formatting issues.

**Solution**:

```bash
git commit -m "message"  # Hooks auto-fix
git add .  # Stage fixes
git commit -m "message"  # Commit again
```

Or manually:

```bash
uv run ruff check . && uv run ruff format .
git add .
git commit -m "message"
```

### `ResourceWarning: Creating large grid` (v0.2.1+)

**Cause**: Grid estimated to use >100MB memory.

**Solution**: This is a warning, not an error. Consider:

```python
# Fix 1: Increase bin_size
env = Environment.from_samples(positions, bin_size=10.0)

# Fix 2: Enable active bin filtering
env = Environment.from_samples(positions, bin_size=1.0, infer_active_bins=True)

# Fix 3: Disable warning (if intentional)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

### `ImportError: pynwb is required for NWB integration` (v0.7.0+)

**Cause**: NWB dependencies are optional.

**Solution**:

```bash
pip install neurospatial[nwb-full]
# or
uv add neurospatial[nwb-full]
```

### TimeSeriesOverlay not showing in HTML backend (v0.14.0+)

**Cause**: HTML backend does not support TimeSeriesOverlay.

**Solution**: Use video or napari backend:

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[timeseries_overlay],
    backend="video",  # or "napari"
    save_path="animation.mp4"
)
```

### `KeyError: No Position found` when reading NWB (v0.7.0+)

**Cause**: NWB file doesn't contain Position data in expected location.

**Solution**: Check where Position data is stored:

```python
from neurospatial.nwb import read_position

# Check what's in the NWB file
print(nwbfile.processing.keys())
print(nwbfile.acquisition.keys())

# If Position is in custom processing module
positions, timestamps = read_position(nwbfile, processing_module="custom_module")

# If there are multiple SpatialSeries, specify which one
positions, timestamps = read_position(nwbfile, position_name="position_xy")
```

### `ValueError: Place field '{name}' already exists` when writing to NWB (v0.7.0+)

**Cause**: Attempting to write a place field with a name that already exists.

**Solution**:

```python
from neurospatial.nwb import write_place_field

# Replace existing place field
write_place_field(nwbfile, env, field, name="cell_001", overwrite=True)

# Or use a different name
write_place_field(nwbfile, env, field, name="cell_001_v2")
```

### Environment round-trip loses some properties (v0.7.0+)

**Cause**: NWB storage uses reconstructed layout, not original layout engine.

**Solution**: This is expected behavior. Round-trip preserves:

✓ **Preserved:**

- `bin_centers` (exact)
- `connectivity` graph structure and edge weights
- `dimension_ranges`
- `units` and `frame` metadata
- Regions (points and polygons)

✗ **Not preserved:**

- Original layout engine type (reconstructed as generic layout)
- Grid-specific metadata (`grid_shape`, `grid_edges`, `active_mask`)
- Layout engine's `is_1d` property (always `False` for reconstructed)

Spatial queries still work identically after round-trip.

### "No start node set" warning in track graph annotation (v0.9.0+)

**Cause**: Saving a track graph without explicitly setting a start node.

**Solution**: This is a warning, not an error. The annotation will proceed with Node 0 as default. To set a specific start node:

1. Select a node in the Node List
2. Press Shift+S to set it as start
3. The start node is marked with ★ in the list

### Track graph edge order seems wrong (v0.9.0+)

**Cause**: Automatic edge ordering via DFS may not match expected linearization.

**Solution**: The edge order is inferred by depth-first search from the start node. You can:

1. **Set a different start node**: DFS traversal depends on which node you start from
2. **Manually reorder edges**: Use Edge Order list with Move Up/Down buttons
3. **Reset to auto**: Click "Reset to Auto" to re-run `infer_edge_layout()`
4. **Preview first**: Click "Preview Linearization" to see 1D layout before saving

```python
# After annotation, override edge order:
env = result.to_environment(bin_size=2.0, edge_spacing=[0.0, 10.0, 0.0])
```

### `ValueError: Cannot create Environment: no track graph` (v0.9.0+)

**Cause**: Calling `result.to_environment()` when track graph is empty or invalid.

**Solution**: Annotation requires at least 2 nodes and 1 edge:

```python
from neurospatial.annotation import annotate_track_graph

result = annotate_track_graph("maze.mp4")

if result.track_graph is not None:
    env = result.to_environment(bin_size=2.0)
else:
    print("No valid track graph created")
    print(f"Nodes: {len(result.node_positions)}")
    print(f"Edges: {len(result.edges)}")
```

### Track graph coordinates don't match video (v0.9.0+)

**Cause**: Calibration not applied or incorrect calibration parameters.

**Solution**: Check calibration settings:

```python
# Without calibration: coordinates in pixels
result = annotate_track_graph("maze.mp4")
print(result.node_positions)  # Pixel values

# With calibration: coordinates in cm
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar

transform = calibrate_from_scale_bar(
    point1=(100, 300),
    point2=(300, 300),
    length_cm=50.0,
    frame_size_px=(640, 480),
)
calib = VideoCalibration(transform, (640, 480))
result = annotate_track_graph("maze.mp4", calibration=calib)
print(result.node_positions)  # Transformed to cm
```

### Time series data and times must be same length

**Cause**: TimeSeriesOverlay data and times arrays have different lengths.

**Solution**:

```python
# Wrong - mismatched lengths
data = np.linspace(0, 100, 1000)  # 1000 points
times = np.linspace(0, 10, 500)   # 500 points
overlay = TimeSeriesOverlay(data=data, times=times)  # ValueError!

# Right - matched lengths
data = np.linspace(0, 100, 1000)
times = np.linspace(0, 10, 1000)  # Same length
overlay = TimeSeriesOverlay(data=data, times=times)  # Works
```

---

## Performance Issues

### Slow test execution

**Cause**: Running tests without parallelization.

**Solution**:

```bash
uv add --dev pytest-xdist
uv run pytest -n auto  # Use all CPU cores
```

### Large session animation is slow

**Cause**: Scanning millions of values for colormap range.

**Solution**: Pre-compute from subset:

```python
from neurospatial.animation import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
)

# Pre-compute colormap range from subset (~10K frames)
vmin, vmax = estimate_colormap_range_from_subset(fields, seed=42)

# Get recommended napari settings
napari_config = large_session_napari_config(n_frames=500_000, sample_rate_hz=250)

# Combine
env.animate_fields(
    fields,
    frame_times=session_times,
    backend="napari",
    vmin=vmin,
    vmax=vmax,
    **napari_config
)
```

### Type errors despite correct code

**Cause**: May be using outdated type stubs or IDE not recognizing runtime checks.

**Note**: This project includes a `py.typed` marker (v0.2.1+) for type checking support. IDE warnings may be false positives that can be ignored if tests pass.
