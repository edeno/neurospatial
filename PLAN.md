# Continuous Variable Animation Overlay - Design Plan

## Overview

Add the ability to visualize continuous time-varying variables (speed, head direction, acceleration, LFP) alongside spatial field animations. Variables are displayed in a **column to the right** of the spatial plot as scrolling time series centered on the current frame.

## Requirements

1. **Position**: Right column adjacent to spatial plot
2. **Display**: Scrolling time series with fixed window centered on current frame
3. **Multiple variables**: Support overlaid on single plot OR stacked as rows
4. **No downsampling**: Preserve full temporal resolution (important for LFP)
5. **Vertical cursor**: Show current time position

## Use Cases

| Variable | Description | Rate | Typical Range |
|----------|-------------|------|---------------|
| Speed | Animal movement speed | Position rate (30-120 Hz) | 0-150 cm/s |
| Head Direction | Heading angle | Position rate or IMU (100+ Hz) | 0-360 degrees |
| Acceleration | Movement acceleration | Position rate | -500 to 500 cm/s^2 |
| LFP | Local field potential | High rate (1-2 kHz) | -1000 to 1000 uV |

## Visual Design

### Layout

```
+---------------------------+------------------+
|                           |    Speed (cm/s)  |
|                           |  ~~~~/\~~|~~~~~  |  <- scrolling window
|                           |         ^        |     with cursor
|    [spatial field]        +------------------+
|                           |    LFP (uV)      |
|                           |  /\/\/\|\/\/\/\  |  <- stacked rows
|                           |        ^         |
+---------------------------+------------------+
```

### Single Plot with Multiple Variables (Overlaid)

```
+---------------------------+----------------------+
|                           |                      |
|                           |  ~~Speed~~           |
|    [spatial field]        |  --Accel--|----      |  <- overlaid, normalized
|                           |           ^          |
|                           |                      |
+---------------------------+----------------------+
```

### Stacked Rows (Separate Y-axes)

```
+---------------------------+------------------+
|                           |  Speed           |
|                           |  ~~~|~~~~        |
|    [spatial field]        +------------------+
|                           |  LFP             |
|                           |  /\|/\/\         |
+---------------------------+------------------+
```

## Proposed API

### TimeSeriesOverlay (Public Dataclass)

```python
from neurospatial import TimeSeriesOverlay

# Single variable
speed_overlay = TimeSeriesOverlay(
    data=speed,                    # Shape: (n_samples,)
    times=timestamps,              # Shape: (n_samples,), required
    label="Speed (cm/s)",          # Y-axis label
    color="cyan",                  # Line color
    window_seconds=2.0,            # Show ±1 second around current frame
)

# Multiple variables - stacked rows (default)
overlays = [
    TimeSeriesOverlay(data=speed, times=t, label="Speed", color="cyan"),
    TimeSeriesOverlay(data=lfp, times=t_lfp, label="LFP", color="yellow"),
]
env.animate_fields(fields, overlays=overlays, frame_times=frame_times)

# Multiple variables - overlaid on single plot
overlays = [
    TimeSeriesOverlay(data=speed, times=t, label="Speed", color="cyan", group="behavior"),
    TimeSeriesOverlay(data=accel, times=t, label="Accel", color="orange", group="behavior"),
    TimeSeriesOverlay(data=lfp, times=t_lfp, label="LFP", color="yellow"),  # separate row
]
# Variables with same `group` are overlaid; different groups get separate rows
```

### Full Parameter List

```python
@dataclass
class TimeSeriesOverlay:
    """Time series visualization in right column during animation.

    Displays continuous variables as scrolling time series plots alongside
    spatial field animations. Multiple overlays can be stacked as rows or
    overlaid on the same plot using the `group` parameter.

    Parameters
    ----------
    data : ndarray of shape (n_samples,), dtype float64
        Time series values. No downsampling is applied - full resolution
        is preserved. NaN values create gaps in the line.
    times : ndarray of shape (n_samples,), dtype float64
        Timestamps for each sample, in seconds. Must be monotonically
        increasing. Required (no default).
    label : str, optional
        Label for Y-axis or legend. Default is "".
    color : str, optional
        Line color (matplotlib color string). Default is "white".
    window_seconds : float, optional
        Total time window to display, centered on current frame.
        E.g., 2.0 shows ±1 second. Default is 2.0.
    linewidth : float, optional
        Line width in points. Default is 1.0.
    alpha : float, optional
        Line opacity (0.0-1.0). Default is 1.0.
    group : str | None, optional
        Group name for overlaying multiple variables on same plot.
        Variables with the same group share axes and are overlaid.
        Variables with different groups (or None) get separate rows.
        Default is None (separate row for each).
    normalize : bool, optional
        If True, normalize to 0-1 range for overlaying variables with
        different scales. Only used when group is set. Default is False.
    show_cursor : bool, optional
        Show vertical line at current time. Default is True.
    cursor_color : str, optional
        Color for cursor line. Default is "red".
    vmin : float | None, optional
        Minimum Y-axis value. If None, auto-computed from data in window.
        Default is None.
    vmax : float | None, optional
        Maximum Y-axis value. If None, auto-computed from data in window.
        Default is None.
    interp : {"linear", "nearest"}, optional
        Interpolation method for computing the value at the current cursor time.
        Used when displaying cursor value (e.g., "Speed: 45.3 cm/s" tooltip).
        - "linear": Linearly interpolate between neighboring samples
        - "nearest": Use nearest sample value
        Default is "linear".

    Examples
    --------
    Single time series (speed):

    >>> speed_overlay = TimeSeriesOverlay(
    ...     data=speed,
    ...     times=timestamps,
    ...     label="Speed (cm/s)",
    ...     color="cyan",
    ...     window_seconds=3.0,  # Show ±1.5 seconds
    ... )
    >>> env.animate_fields(fields, overlays=[speed_overlay], frame_times=frame_times)

    Multiple stacked rows:

    >>> overlays = [
    ...     TimeSeriesOverlay(data=speed, times=t, label="Speed", color="cyan"),
    ...     TimeSeriesOverlay(data=lfp, times=t_lfp, label="LFP", color="yellow"),
    ... ]
    >>> env.animate_fields(fields, overlays=overlays, frame_times=frame_times)

    Overlaid variables (same group):

    >>> overlays = [
    ...     TimeSeriesOverlay(data=speed, times=t, label="Speed", color="cyan",
    ...                       group="kinematics", normalize=True),
    ...     TimeSeriesOverlay(data=accel, times=t, label="Accel", color="orange",
    ...                       group="kinematics", normalize=True),
    ... ]
    """

    data: NDArray[np.float64]
    times: NDArray[np.float64]
    label: str = ""
    color: str = "white"
    window_seconds: float = 2.0
    linewidth: float = 1.0
    alpha: float = 1.0
    group: str | None = None
    normalize: bool = False
    show_cursor: bool = True
    cursor_color: str = "red"
    vmin: float | None = None
    vmax: float | None = None
    interp: Literal["linear", "nearest"] = "linear"
```

### Internal Data Container

```python
@dataclass
class TimeSeriesData:
    """Internal container for time series data aligned to animation frames.

    Created by conversion pipeline, not instantiated by users.

    Key design: Matches existing overlay pattern where convert_to_data() does
    frame alignment upfront. Precomputes window indices per frame so backends
    only do cheap O(1) slicing, not O(log n) searchsorted per frame.

    Architectural note: Unlike other overlays (PositionData, BodypartData) which
    store frame-aligned data arrays of shape (n_frames, ...), TimeSeriesData
    stores full-resolution data plus precomputed indices. This is necessary
    because time series are displayed as scrolling windows, not single points
    per frame. The "frame alignment" here means precomputing which slice of
    the full data to display for each frame, not interpolating to frame times.
    """
    # Full resolution data (no downsampling)
    data: NDArray[np.float64]       # Shape: (n_samples,)
    times: NDArray[np.float64]      # Shape: (n_samples,)

    # Precomputed window indices per frame (frame alignment done in convert_to_data)
    start_indices: NDArray[np.int64]  # Shape: (n_frames,) - start of window for each frame
    end_indices: NDArray[np.int64]    # Shape: (n_frames,) - end of window for each frame

    # Display settings
    label: str
    color: str
    window_seconds: float
    linewidth: float
    alpha: float
    group: str | None
    normalize: bool
    show_cursor: bool
    cursor_color: str

    # Y-axis limits (global by default for stable scales)
    global_vmin: float  # Min across all data
    global_vmax: float  # Max across all data
    use_global_limits: bool = True  # If False, use per-window autoscaling (opt-in)

    # Interpolation for cursor value computation
    interp: Literal["linear", "nearest"] = "linear"

    def get_window_slice(self, frame_idx: int) -> tuple[NDArray, NDArray]:
        """O(1) window extraction using precomputed indices."""
        start = self.start_indices[frame_idx]
        end = self.end_indices[frame_idx]
        return self.data[start:end], self.times[start:end]

    def get_cursor_value(self, current_time: float) -> float | None:
        """Get interpolated value at cursor time for tooltip display.

        Returns None if current_time is outside data range or data is empty.
        """
        if len(self.times) == 0:
            return None
        if current_time < self.times[0] or current_time > self.times[-1]:
            return None

        if self.interp == "nearest":
            idx = np.searchsorted(self.times, current_time)
            # Choose closer of idx-1 or idx
            if idx == 0:
                return float(self.data[0])
            if idx >= len(self.times):
                return float(self.data[-1])
            if current_time - self.times[idx - 1] < self.times[idx] - current_time:
                return float(self.data[idx - 1])
            return float(self.data[idx])
        else:  # linear
            return float(np.interp(current_time, self.times, self.data))
```

### convert_to_data() Implementation

```python
def convert_to_data(
    self,
    frame_times: NDArray[np.float64],
    n_frames: int,
    env: Any,
) -> TimeSeriesData:
    """Convert overlay to internal data with precomputed frame alignment.

    Follows existing overlay pattern: all frame alignment done here,
    backends receive ready-to-use data with O(1) per-frame access.
    """
    # Validate inputs
    # NOTE: We do NOT use _validate_finite_values for data because NaN values
    # are allowed (they create gaps in the line). We only validate times.
    _validate_finite_values(self.times, name="TimeSeriesOverlay.times")  # Times must be finite
    _validate_monotonic_time(self.times, name="TimeSeriesOverlay.times")

    # Validate no Inf values in data (NaN is allowed for gaps, Inf is not)
    if np.any(np.isinf(self.data)):
        n_inf = np.sum(np.isinf(self.data))
        raise ValueError(
            f"TimeSeriesOverlay.data contains {n_inf} Inf values. "
            f"NaN values are allowed (create gaps), but Inf is not supported."
        )

    # Vectorized precomputation of window indices for all frames
    half_window = self.window_seconds / 2
    start_indices = np.searchsorted(self.times, frame_times - half_window)
    end_indices = np.searchsorted(self.times, frame_times + half_window)

    # Compute global limits for stable y-axis (default behavior)
    finite_mask = np.isfinite(self.data)
    if finite_mask.any():
        global_vmin = float(np.min(self.data[finite_mask]))
        global_vmax = float(np.max(self.data[finite_mask]))
    else:
        global_vmin, global_vmax = 0.0, 1.0

    # Override with explicit limits if provided
    if self.vmin is not None:
        global_vmin = self.vmin
    if self.vmax is not None:
        global_vmax = self.vmax

    # Apply normalization if requested
    output_data = self.data
    if self.normalize:
        range_val = global_vmax - global_vmin
        if range_val > 0:
            output_data = (self.data - global_vmin) / range_val
        else:
            output_data = np.zeros_like(self.data)  # Constant data -> 0
        # After normalization, limits become [0, 1]
        global_vmin, global_vmax = 0.0, 1.0

    return TimeSeriesData(
        data=output_data,
        times=self.times,
        start_indices=start_indices,
        end_indices=end_indices,
        label=self.label,
        color=self.color,
        window_seconds=self.window_seconds,
        linewidth=self.linewidth,
        alpha=self.alpha,
        group=self.group,
        normalize=self.normalize,
        show_cursor=self.show_cursor,
        cursor_color=self.cursor_color,
        global_vmin=global_vmin,
        global_vmax=global_vmax,
        use_global_limits=True,  # Stable scales by default
        interp=self.interp,  # For cursor value computation
    )
```

## Edge Case Handling

### Conflicting Parameters in Same Group

When multiple `TimeSeriesOverlay` instances share the same `group` but have different `window_seconds`, `vmin`, or `vmax`:

**Decision: Warning + First Wins**

- Emit `UserWarning` listing the conflicting parameters
- Use values from the **first overlay** in the group (by order in list)
- Rationale: Strict errors would block users unnecessarily; first-wins is predictable

```python
# Example: These share group="kinematics" but have different window_seconds
overlays = [
    TimeSeriesOverlay(data=speed, times=t, group="kinematics", window_seconds=2.0),  # Used
    TimeSeriesOverlay(data=accel, times=t, group="kinematics", window_seconds=5.0),  # Ignored with warning
]
# UserWarning: Group 'kinematics' has conflicting window_seconds (2.0 vs 5.0). Using first value: 2.0
```

**Affected parameters** (per-group, not per-overlay):

- `window_seconds` - shared X-axis range
- `vmin`, `vmax` - shared Y-axis range (unless `normalize=True`)
- `show_cursor`, `cursor_color` - shared cursor

**Per-overlay parameters** (allowed to differ within group):

- `color`, `linewidth`, `alpha` - each line can have different styling
- `label` - used for legend
- `normalize` - can mix normalized and non-normalized (though unusual)

### normalize Interaction with Y-axis Limits

When `normalize=True`, the data is scaled to [0, 1] range using `global_vmin` and `global_vmax`:

```python
# In convert_to_data() when normalize=True:
if self.normalize:
    normalized_data = (self.data - global_vmin) / (global_vmax - global_vmin)
    # Store normalized data, set global_vmin=0, global_vmax=1
```

**Behavior in grouped plots:**

| Scenario | Y-axis Limits | Notes |
|----------|---------------|-------|
| All `normalize=False` | Shared from first overlay's `global_vmin/vmax` | Warning if limits differ |
| All `normalize=True` | Fixed [0, 1] | Each variable scaled independently |
| Mixed (unusual) | First overlay's limits | Normalized variables rescaled to unnormalized range |

**Recommendation:** Within a group, either all overlays should use `normalize=True` or all `normalize=False`. Mixing is allowed but produces confusing y-axis semantics.

```python
# Good: All normalized (compare relative changes)
overlays = [
    TimeSeriesOverlay(data=speed, times=t, group="kinematics", normalize=True),
    TimeSeriesOverlay(data=accel, times=t, group="kinematics", normalize=True),
]

# Good: All unnormalized with compatible units
overlays = [
    TimeSeriesOverlay(data=speed_x, times=t, group="speed_components", normalize=False),
    TimeSeriesOverlay(data=speed_y, times=t, group="speed_components", normalize=False),
]

# Avoid: Mixed normalization in same group
overlays = [
    TimeSeriesOverlay(data=speed, times=t, group="mixed", normalize=True),   # Scaled to [0,1]
    TimeSeriesOverlay(data=accel, times=t, group="mixed", normalize=False),  # Original scale
]  # Confusing - which y-axis do the tick labels represent?
```

### Frame Times Outside Overlay Time Range

When `frame_times[i]` falls outside the overlay's `times` range:

**Decision: Show Available Data (No Clamping)**

- Window extraction returns whatever data falls within `[current_time - window/2, current_time + window/2]`
- If window is partially outside data range → partial data shown (empty on one side)
- If window is entirely outside data range → empty plot for that frame
- Cursor always shows at `current_time` (never clamped)

```
Data times:     [1.0 -------- 10.0]
Frame time:     0.5 (before data)
Window ±1s:     [-0.5 to 1.5]
Result:         Only data from [1.0, 1.5] shown; left side of window empty
                Cursor at 0.5 (visible but no data there)

Frame time:     5.0 (within data)
Window ±1s:     [4.0 to 6.0]
Result:         Full window of data shown
                Cursor at 5.0

Frame time:     12.0 (after data)
Window ±1s:     [11.0 to 13.0]
Result:         Empty plot (no data in window)
                Cursor at 12.0 (visible but no data)
```

**Rationale:**

- No artificial clamping that could mislead users
- Naturally handles sessions where different signals have different time ranges
- Empty regions clearly indicate "no data here"

### NaN Handling in Data

**Decision: NaN creates gaps in line**

- `np.nan` values in `data` create discontinuities in the plotted line
- Matplotlib's default behavior: line breaks at NaN
- No interpolation across NaN gaps

```python
# Data with gap
data = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
# Renders as two separate line segments: [1,2] and [5,6]
```

## Implementation Details

### Rendering Architecture

**Key design principle**: Artist reuse pattern (matches existing `OverlayArtistManager`).

- Create `Figure`, `Axes`, `Line2D` artists **once** during setup
- On each frame: use precomputed indices for O(1) slicing, call `line.set_data()`
- Never recreate axes or lines per frame

### Shared Rendering Logic

Factor common logic into `_timeseries.py` (parallel to `_parallel.py`):

```python
# src/neurospatial/animation/_timeseries.py

@dataclass
class TimeSeriesArtistManager:
    """Manages matplotlib artists for time series rendering.

    Created once, reused across all frames. Matches pattern of
    OverlayArtistManager in video/widget backends.
    """
    axes: list[plt.Axes]           # One per group/row
    lines: dict[str, Line2D]       # label -> Line2D artist
    cursors: list[Line2D]          # One cursor per axes
    value_texts: dict[str, Text]   # label -> Text artist for cursor value display
    frame_times: NDArray[np.float64]
    group_window_seconds: dict[int, float]  # group_idx -> window_seconds (per-group)

    @classmethod
    def create(
        cls,
        fig: Figure,
        timeseries_data: list[TimeSeriesData],
        frame_times: NDArray[np.float64],
        dark_theme: bool = True,
    ) -> "TimeSeriesArtistManager":
        """Create axes and artists once, configure for napari dark theme."""
        groups = _group_timeseries(timeseries_data)
        n_rows = len(groups)

        # Create axes (NOT shared x-axis since window_seconds may differ per group)
        axes = fig.subplots(n_rows, 1, squeeze=False, sharex=False)[:, 0]

        # Build per-group window_seconds (first overlay in each group wins)
        # Also check for conflicting parameters and emit warnings
        group_window_seconds: dict[int, float] = {}
        group_normalize: dict[int, bool] = {}
        for ts_data in timeseries_data:
            group_idx = _get_group_index(ts_data, groups)
            if group_idx not in group_window_seconds:
                group_window_seconds[group_idx] = ts_data.window_seconds
                group_normalize[group_idx] = ts_data.normalize
            else:
                # Check for conflicts and warn
                if ts_data.window_seconds != group_window_seconds[group_idx]:
                    warnings.warn(
                        f"Group has conflicting window_seconds "
                        f"({group_window_seconds[group_idx]} vs {ts_data.window_seconds}). "
                        f"Using first value: {group_window_seconds[group_idx]}",
                        UserWarning,
                    )
                if ts_data.normalize != group_normalize[group_idx]:
                    warnings.warn(
                        f"Group has mixed normalize settings. "
                        f"This produces confusing y-axis semantics. "
                        f"Consider using normalize=True for all or none.",
                        UserWarning,
                    )

        # Style for napari dark theme
        if dark_theme:
            fig.patch.set_facecolor('#262930')  # napari background
            for ax in axes:
                ax.set_facecolor('#262930')
                ax.tick_params(colors='white', labelsize=8)
                ax.spines[:].set_visible(False)  # No spines
                for spine in ax.spines.values():
                    spine.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')

        # Create Line2D artists and value text annotations (one per overlay)
        lines = {}
        value_texts = {}
        for ts_data in timeseries_data:
            group_idx = _get_group_index(ts_data, groups)
            ax = axes[group_idx]
            line, = ax.plot([], [], color=ts_data.color,
                           linewidth=ts_data.linewidth, alpha=ts_data.alpha,
                           label=ts_data.label)
            key = ts_data.label or id(ts_data)
            lines[key] = line

            # Create text annotation for cursor value display
            value_text = ax.text(
                0.98, 0.95, '', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                color=ts_data.color if dark_theme else 'black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#262930' if dark_theme else 'white',
                         alpha=0.7, edgecolor='none')
            )
            value_texts[key] = value_text

            # Set stable y-limits (global by default)
            if ts_data.use_global_limits:
                margin = (ts_data.global_vmax - ts_data.global_vmin) * 0.05
                ax.set_ylim(ts_data.global_vmin - margin, ts_data.global_vmax + margin)

        # Create cursor lines (one per axes)
        cursors = []
        for ax in axes:
            cursor = ax.axvline(x=0, color='red', linewidth=1, alpha=0.8)
            cursors.append(cursor)

        # Minimal labels, shared x-axis labels only on bottom
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel('Time (s)', fontsize=9, color='white' if dark_theme else 'black')

        # Add legends for grouped overlays
        for ax in axes:
            if len(ax.lines) > 1:  # Multiple lines in this axes
                ax.legend(loc='upper right', fontsize=7, framealpha=0.5)

        fig.tight_layout()

        return cls(axes=list(axes), lines=lines, cursors=cursors,
                   value_texts=value_texts, frame_times=frame_times,
                   group_window_seconds=group_window_seconds)

    def update(self, frame_idx: int, timeseries_data: list[TimeSeriesData]) -> None:
        """Update all artists for given frame. O(n_overlays) with O(1) slicing."""
        current_time = self.frame_times[frame_idx]
        groups = _group_timeseries(timeseries_data)

        for ts_data in timeseries_data:
            key = ts_data.label or id(ts_data)

            # O(1) slice using precomputed indices
            y_data, t_data = ts_data.get_window_slice(frame_idx)

            # Update line data
            line = self.lines[key]
            line.set_data(t_data, y_data)

            # Update cursor value text using interpolation method
            cursor_value = ts_data.get_cursor_value(current_time)
            value_text = self.value_texts[key]
            if cursor_value is not None and ts_data.label:
                value_text.set_text(f"{ts_data.label}: {cursor_value:.2f}")
            else:
                value_text.set_text('')

        # Update cursors and x-axis limits (per-group window_seconds)
        for group_idx, ax in enumerate(self.axes):
            half_window = self.group_window_seconds[group_idx] / 2
            ax.set_xlim(current_time - half_window, current_time + half_window)

        for cursor in self.cursors:
            cursor.set_xdata([current_time, current_time])


def _render_timeseries_column(
    manager: TimeSeriesArtistManager,
    frame_idx: int,
    timeseries_data: list[TimeSeriesData],
) -> None:
    """Shared rendering logic for video/widget backends.

    Called per frame. Uses artist manager for efficient updates.
    Parallel to _render_all_overlays() in _parallel.py.
    """
    manager.update(frame_idx, timeseries_data)
```

### Backend Implementations

#### Napari Backend

```python
def _add_timeseries_dock(
    viewer: napari.Viewer,
    timeseries_data: list[TimeSeriesData],
    frame_times: NDArray[np.float64],
) -> None:
    """Add time series dock widget to napari viewer."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QWidget, QVBoxLayout

    groups = _group_timeseries(timeseries_data)
    n_rows = len(groups)

    # Create figure ONCE
    fig = Figure(figsize=(3, 1.5 * n_rows), dpi=100)

    # Create artist manager ONCE (handles all setup)
    manager = TimeSeriesArtistManager.create(
        fig, timeseries_data, frame_times, dark_theme=True
    )

    canvas = FigureCanvasQTAgg(fig)

    # Create dock widget
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas)

    viewer.window.add_dock_widget(widget, name="Time Series", area="right")

    # Throttle updates to max 30-40 Hz even if napari fps is higher
    last_update_time = [0.0]
    min_update_interval = 1.0 / 40.0  # 40 Hz max

    def on_frame_change(event):
        import time
        current_time = time.time()
        if current_time - last_update_time[0] < min_update_interval:
            return  # Skip update, too soon

        frame_idx = viewer.dims.current_step[0]
        manager.update(frame_idx, timeseries_data)
        canvas.draw_idle()
        last_update_time[0] = current_time

    viewer.dims.events.current_step.connect(on_frame_change)

    # Initial render
    manager.update(0, timeseries_data)
    canvas.draw()
```

#### Video Backend

```python
from matplotlib.gridspec import GridSpec

def _setup_video_figure_with_timeseries(
    env: Environment,
    timeseries_data: list[TimeSeriesData],
    frame_times: NDArray[np.float64],
    dpi: int = 100,
) -> tuple[Figure, plt.Axes, TimeSeriesArtistManager]:
    """Create figure with spatial field + time series column."""
    groups = _group_timeseries(timeseries_data)
    n_rows = max(1, len(groups))

    # GridSpec: spatial field spans all rows on left, time series stacked on right
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[3, 1], hspace=0.1)

    # Spatial field axes (spans all rows)
    ax_field = fig.add_subplot(gs[:, 0])

    # Time series axes (right column)
    ax_ts_list = [fig.add_subplot(gs[i, 1]) for i in range(n_rows)]

    # Create artist manager for time series
    # Note: We pass a sub-figure or use the existing axes
    manager = TimeSeriesArtistManager.create_from_axes(
        ax_ts_list, timeseries_data, frame_times, dark_theme=False
    )

    return fig, ax_field, manager


# In render_video():
# - Create figure and manager ONCE per worker
# - Per frame: call manager.update(frame_idx, timeseries_data)
# - Follows same pattern as OverlayArtistManager
```

#### Widget Backend

Same approach as video backend - create figure/manager once, update per frame.

#### HTML Backend (Phase 5)

```python
# In _serialize_overlay_data() for HTML backend:
# Include precomputed indices in JSON so JS does O(1) slicing

class TimeSeriesDataJSON(TypedDict):
    data: list[float]
    times: list[float]
    start_indices: list[int]  # Precomputed per frame
    end_indices: list[int]    # Precomputed per frame
    label: str
    color: str
    window_seconds: float
    global_vmin: float
    global_vmax: float
```

### OverlayData Extension

```python
@dataclass
class OverlayData:
    """Container for all overlay data."""
    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    videos: list[VideoData] = field(default_factory=list)
    events: list[EventData] = field(default_factory=list)
    regions: dict[int, list[str]] | None = None
    timeseries: list[TimeSeriesData] = field(default_factory=list)  # NEW
```

## File Changes

| File | Change |
|------|--------|
| `src/neurospatial/animation/overlays.py` | Add `TimeSeriesOverlay`, `TimeSeriesData` with precomputed indices |
| `src/neurospatial/animation/overlays.py` | Update `OverlayData` with `timeseries: list[TimeSeriesData]` field |
| `src/neurospatial/animation/overlays.py` | Update `_convert_overlays_to_data()` dispatch (see below) |

### _convert_overlays_to_data() Updates

Add to initialization section (~line 2912):

```python
timeseries_data_list: list[TimeSeriesData] = []
```

Add dispatch branch after EventData (~line 2941):

```python
elif isinstance(internal_data, TimeSeriesData):
    timeseries_data_list.append(internal_data)
```

Update error message (~line 2943-2946):

```python
raise TypeError(
    f"convert_to_data() must return PositionData, BodypartData, "
    f"HeadDirectionData, VideoData, EventData, or TimeSeriesData, "
    f"got {type(internal_data).__name__}."
)
```

Add to OverlayData constructor (~line 2961-2968):

```python
overlay_data = OverlayData(
    positions=position_data_list,
    bodypart_sets=bodypart_data_list,
    head_directions=head_direction_data_list,
    videos=video_data_list,
    events=event_data_list,
    regions=normalized_regions,
    timeseries=timeseries_data_list,  # NEW
)
```

| File | Change |
|------|--------|
| `src/neurospatial/animation/_timeseries.py` | **New** - `TimeSeriesArtistManager`, `_render_timeseries_column()` |
| `src/neurospatial/animation/backends/napari_backend.py` | Add `_add_timeseries_dock()`, update `render_napari()` |
| `src/neurospatial/animation/backends/video_backend.py` | Add GridSpec layout, integrate `TimeSeriesArtistManager` |
| `src/neurospatial/animation/backends/widget_backend.py` | Add time series subplot, integrate artist manager |
| `src/neurospatial/animation/backends/html_backend.py` | Add `TimeSeriesDataJSON` to `OverlayDataJSON` (Phase 5) |
| `src/neurospatial/animation/__init__.py` | Export `TimeSeriesOverlay` |
| `CLAUDE.md` | Document new feature |
| `tests/animation/test_timeseries_overlay.py` | **New** - comprehensive tests |

## Implementation Order

### Phase 1: Core Infrastructure

1. Add `TimeSeriesOverlay` dataclass to `overlays.py`
2. Add `TimeSeriesData` internal container
3. Implement `convert_to_data()` method
4. Update `OverlayData` with `timeseries` field
5. Update `_convert_overlays_to_data()` to handle new overlay type

### Phase 2: Napari Backend

1. Implement `_add_timeseries_dock()` with matplotlib figure
2. Implement `_update_timeseries_plot()` for frame updates
3. Handle grouping (overlaid vs stacked)
4. Connect to `viewer.dims.events.current_step`
5. Test with single and multiple time series

### Phase 3: Video Backend

1. Modify `render_video()` to use GridSpec layout
2. Add time series rendering per frame
3. Handle aspect ratio changes from added column
4. Test video export

### Phase 4: Widget Backend

1. Extend widget layout for time series column
2. Update rendering loop
3. Test in Jupyter notebooks

### Phase 5: Polish

1. HTML backend support (if feasible)
2. Performance optimization for high-rate data (LFP)
3. Add tests for edge cases (NaN, single point, etc.)
4. Update documentation

## Visualization Design Principles

Following Heer-style guidance for effective time series visualization:

### Default Configuration (Task-Optimized)

**Primary tasks** the design optimizes for:

- "How does speed/acceleration relate to field activity at this moment?"
- "Are LFP events time-locked to spatial features?"
- "When did behavioral state changes occur?"

**Defaults that support these tasks:**

| Setting | Default | Rationale |
|---------|---------|-----------|
| Layout | Stacked rows | Easier comparison, less overplotting |
| Y-axis | Global limits (stable) | Avoids "breathing" axes, supports pattern detection |
| Window | 2-3 seconds | Shows sufficient context without overwhelming |
| Max overlaid | 2-3 per group | Prevents visual clutter |
| X-axis | Shared across rows | Aligned vertical cursor, linked time scale |
| Labels | Minimal ticks, clear units | Reduces non-data ink |

### Color Palette

High-contrast colors for napari dark theme:

```python
TIMESERIES_PALETTE = [
    "#00FFFF",  # cyan (high visibility)
    "#FF8C00",  # orange
    "#FF00FF",  # magenta
    "#FFFF00",  # yellow
    "#00FF00",  # green
    "#FF6B6B",  # coral
]
```

### Reducing Cognitive Load

1. **One variable ↔ one color**: Consistent encoding throughout session
2. **Clear labels**: Include units (e.g., "Speed (cm/s)", not just "Speed")
3. **Minimal chrome**: No heavy grids, few tick marks, hidden spines
4. **Stable scales**: Global y-limits by default (opt-in autoscaling)

### Interaction Features (v1)

- **Toggle overlays**: Checkbox or legend click to show/hide individual variables
- **Cursor alignment**: Vertical line exactly matches napari time slider
- **Cursor value display**: Shows interpolated value at cursor position (e.g., "Speed: 45.3 cm/s")
  - Uses `interp` parameter: "linear" for smooth interpolation, "nearest" for exact samples
  - Positioned in upper-right of each time series plot

### Future Enhancements (v2)

1. **Focus+Context Overview Strip**:

   ```
   +---------------------------+------------------+
   |                           |  Speed (detail)  |
   |    [spatial field]        |  ~~~|~~~~        |
   |                           +------------------+
   |                           |  [===|===]       |  <- overview (full session)
   +---------------------------+------------------+
   ```

   - Small strip showing entire session timeline
   - Brushed window indicates current animated region
   - Enables "focus+context" navigation

2. **Epoch Snapshots**:
   - Pin/bookmark specific time windows
   - Compare epochs side-by-side (small multiples)
   - Better for precise comparison than scrubbed animation

3. **Linked Highlighting**:
   - Hover on time series highlights corresponding spatial bin
   - Click to jump to that frame

## Performance Considerations

### High Sample Rate Data (LFP at 1-2 kHz)

- **No downsampling**: Full data stored in `TimeSeriesData`
- **Precomputed indices**: O(1) slicing per frame (searchsorted done once in convert_to_data)
- **Artist reuse**: `line.set_data()` instead of redrawing
- **Throttled updates**: Max 40 Hz matplotlib redraws even at higher napari fps

### Memory Estimation

For a 1-hour session at 1 kHz:

- Data: 3.6M samples × 8 bytes = 28.8 MB per variable
- Indices: 2 × n_frames × 8 bytes (negligible for typical frame counts)
- Acceptable for typical use cases

### Rendering Performance

- Window extraction: **O(1)** per frame (precomputed indices)
- Line update: O(window_samples) per frame
- For 2-second window at 1 kHz: ~2000 points per frame (fast)
- Throttling prevents matplotlib from becoming bottleneck

## Testing Strategy

| Test | Description |
|------|-------------|
| `test_timeseries_overlay_creation` | Valid construction, parameter validation |
| `test_timeseries_convert_to_data` | Conversion pipeline |
| `test_timeseries_window_extraction` | Window slicing correctness |
| `test_timeseries_cursor_value_linear` | Linear interpolation for cursor values |
| `test_timeseries_cursor_value_nearest` | Nearest-neighbor for cursor values |
| `test_timeseries_grouping` | Overlaid vs stacked grouping |
| `test_timeseries_normalize` | Normalization scales data to [0, 1] |
| `test_timeseries_group_window_seconds` | Per-group window_seconds respected |
| `test_timeseries_mixed_normalize_warning` | Warning when mixing normalize in group |
| `test_timeseries_napari_dock` | Dock widget creation (integration) |
| `test_timeseries_video_render` | Video export with time series |
| `test_timeseries_high_rate` | Performance with 1 kHz data |

## Success Criteria

- [ ] `TimeSeriesOverlay` works with napari backend
- [ ] Multiple time series can be stacked as rows
- [ ] Multiple time series can be overlaid with `group` parameter
- [ ] Scrolling window centered on current frame
- [ ] No downsampling - full resolution preserved
- [ ] Video export includes time series column
- [ ] Widget backend includes time series
- [ ] Performance acceptable for 1 kHz data over 1-hour sessions
