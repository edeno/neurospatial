# Events Module Implementation Plan

**Last Updated**: 2025-12-04
**Status**: Planning (Reviewed)

---

## Executive Summary

This plan defines the implementation of an events module for neurospatial that:

1. Uses `pd.DataFrame` as the interchange format (aligns with existing `read_events()`)
2. Is fully compatible with NWB's `ndx-events.EventsTable` and core `TimeIntervals`
3. Follows established codebase patterns (dataclasses, NumPy docstrings, protocols)
4. Enables event-triggered neural analysis (PSTHs, rate cubes, GLM regressors)

---

## Design Principles

### 1. DataFrame as Interchange Format (Not a New Dataclass)

**Rationale**: The existing `nwb/_events.py` already uses `pd.DataFrame` for event data. This is the right choice because:

- Direct compatibility with `ndx-events.EventsTable.to_dataframe()`
- Easy manipulation with pandas (filtering, grouping, merging)
- No new abstraction to learn
- Natural serialization to/from NWB

**Standard Columns** (following ndx-events conventions):

- `timestamp` (required): Event time in seconds from session start
- `duration` (optional): Event duration in seconds (NaN if instantaneous)
- `label` (optional): Event category/type as string
- `value` (optional): Continuous value associated with event

**Spatial Columns** (neurospatial extensions):

- `x`, `y`, `z` (optional): Event position in environment coordinates

Additional domain-specific columns are allowed (e.g., `event_type`, `reward_amount`, `trial_id`, `region`).

**Design Note**: We intentionally do not store derived spatial data (`bin_index`, computed `region`) as columns. These depend on an Environment configuration and can become stale. Instead:

- Store coordinates (`x`, `y`) as the source of truth
- Compute `bin_index` via `env.bin_at([x, y])` when needed
- Query region membership via `events_in_region(events, env, "goal")`
- If `region` is an intrinsic event property (e.g., opto stim trigger zone), users add it as a domain-specific column

### 2. Point Events vs Intervals

| Type | Use Case | Columns | NWB Storage |
|------|----------|---------|-------------|
| **Point events** | Reward delivery, lick, TTL pulse | `timestamp`, optional `duration` | `EventsTable` |
| **Intervals** | Trials, epochs, behavioral states | `start_time`, `stop_time` | `TimeIntervals` |
| **Spatial events** | Reward at location, zone entry | `timestamp`, `x`, `y`, `bin_index` | `EventsTable` + spatial cols |

Both are represented as `pd.DataFrame` with appropriate columns. The distinction is semantic, not structural.

### 3. Spatial Events

Many behavioral events have an associated spatial location:

- **Reward deliveries**: Where the animal received reward
- **Licks/pokes**: Sensor location in environment
- **Zone crossings**: Entry/exit positions at region boundaries
- **Spike events**: Position at time of spike (for visualization)
- **Choice points**: Location of behavioral decisions

Spatial columns enable:

- Filtering events by region: `events[events["region"] == "goal"]`
- Spatial event rate maps: count events per spatial bin
- Distance-based analyses: events within X cm of location
- Visualization: overlay events on environment plots

### 4. Intervals vs Point Events

**Point events** (instantaneous) and **intervals** (duration) are both represented as DataFrames:

| Type | Required Columns | Optional | Use Case |
|------|------------------|----------|----------|
| Point event | `timestamp` | `label`, `value`, `x`, `y` | Rewards, licks, spikes |
| Interval | `start_time`, `stop_time` | `label`, `trial_id` | Trials, epochs, states |

**Interval handling strategy**:

1. **Existing support**: NWB `TimeIntervals` already uses `start_time`/`stop_time` convention
2. **Conversion functions**: Convert between formats as needed
3. **Analysis functions**: Accept either format with automatic detection

```python
def intervals_to_events(
    intervals: pd.DataFrame,
    which: Literal["start", "stop", "both"] = "start",
) -> pd.DataFrame:
    """Convert intervals to point events (start and/or stop times)."""

def events_to_intervals(
    start_events: pd.DataFrame,
    stop_events: pd.DataFrame,
    *,
    match_by: str | None = None,
) -> pd.DataFrame:
    """Pair start/stop events into intervals."""

def filter_by_intervals(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    include: bool = True,
) -> pd.DataFrame:
    """Filter events to those within (or outside) intervals."""
```

**Why not a separate Interval class?**

- NWB already defines the format (`TimeIntervals`)
- DataFrame operations handle intervals naturally
- Conversion functions bridge the gap when needed

### 5. Follow Existing Patterns

- **NumPy docstrings** with "Which Function Should I Use?" sections
- **Frozen dataclasses** for result objects (like `HeadDirectionMetrics`, `Trial`)
- **Free functions** for analysis (not methods on a class)
- **Lazy imports** for optional dependencies (NWB)
- **Comprehensive validation** with diagnostic error messages

---

## Module Structure

```
src/neurospatial/
├── events/
│   ├── __init__.py              # Public API with lazy imports
│   ├── _core.py                 # Result dataclasses, validation helpers
│   ├── detection.py             # Event detection from trajectories/signals
│   ├── intervals.py             # Interval conversion and filtering
│   ├── regressors.py            # GLM regressor generation
│   └── alignment.py             # Peri-event alignment and analysis
└── nwb/
    └── _events.py               # (existing) Extend with new helpers
```

---

## Which Function Should I Use?

This section helps you find the right function for your analysis.

### I want to detect events from data

| Task | Function |
|------|----------|
| Detect when animal enters/exits a region | `extract_region_crossing_events()` |
| Detect when a signal crosses a threshold | `extract_threshold_crossing_events()` |
| Detect movement onset from position data | `extract_movement_onset_events()` |

### I want to compute GLM regressors

| Question | Function |
|----------|----------|
| How long since the last reward? (retrospective) | `time_since_event()` |
| How long until the next reward? (prospective) | `time_to_event()` |
| How many rewards in the last 5 seconds? | `event_count_in_window()` |
| Was there a reward in this time bin? | `event_indicator()` |
| Decaying reward signal with 2s time constant? | `exponential_kernel()` |
| How far from where I last got reward? | `distance_to_event_at_time(which="last")` |

### I want to analyze neural responses to events

| Task | Function |
|------|----------|
| Get per-trial spike times for raster plots | `align_spikes_to_events()` |
| Compute PSTH (firing rate around events) | `peri_event_histogram()` |
| Compute population PSTH (multiple units) | `population_peri_event_histogram()` |
| Align licks to rewards (event-to-event) | `align_events()` |

### I want to work with intervals (trials, epochs)

| Task | Function |
|------|----------|
| Convert trial intervals to start/stop events | `intervals_to_events()` |
| Pair start and stop events into intervals | `events_to_intervals()` |
| Filter events to those within trials | `filter_by_intervals()` |

### I want to work with spatial event positions

| Task | Function |
|------|----------|
| Add x, y positions to my events DataFrame | `add_positions()` |
| Filter events to those in a region | `events_in_region()` |
| Compute spatial rate map of events | `spatial_event_rate()` |

---

## Common Parameters

First-time users often ask "what value should I use?" Here are typical ranges:

### PSTH and Peri-Event Analysis

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `window` | `(-0.5, 1.0)` to `(-2.0, 5.0)` | Start negative (before event), end positive (after) |
| `bin_size` | `0.01` to `0.05` | Seconds. 10ms typical for single-unit, 50ms for population |
| `baseline_window` | `(-0.5, -0.1)` | Pre-event period for baseline subtraction |

### GLM Regressors

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `tau` (exponential decay) | `0.5` to `5.0` | Time constant in seconds. Shorter = faster decay |
| `max_time` | `10.0` to `60.0` | Cap time since event at this value |
| `fill_before_first` | `np.nan` or `max_time` | Use `max_time` for finite design matrix |

### Spatial Event Parameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `bandwidth` (smoothing) | `5.0` to `20.0` | In same units as environment (typically cm) |
| `min_dwell_time` | `0.1` to `0.5` | Seconds. Filters brief boundary crossings |

### Example: "Good Defaults" for Reward PSTH

```python
# Typical reward-triggered analysis
psth = peri_event_histogram(
    spike_times, reward_times,
    window=(-1.0, 3.0),    # 1s before to 3s after reward
    bin_size=0.025,         # 25ms bins (good balance)
    baseline_window=(-1.0, -0.2),  # Pre-reward baseline
)
```

---

## Public API

### Core Dataclass

```python
# src/neurospatial/events/_core.py

# --- Validation Helpers ---

def validate_events_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    timestamp_column: str = "timestamp",
    context: str = "",
) -> None:
    """
    Validate events DataFrame structure with diagnostic error messages.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame to validate.
    required_columns : list[str], optional
        Additional required columns beyond timestamp.
    timestamp_column : str, default="timestamp"
        Name of timestamp column.
    context : str, optional
        Additional context for error messages (e.g., function name).

    Raises
    ------
    TypeError
        If df is not a DataFrame.
        WHAT: "Expected pd.DataFrame, got {type}"
        WHY: "Events must be a pandas DataFrame for NWB compatibility"
        HOW: "Convert using pd.DataFrame({'timestamp': times})"

    ValueError
        If required columns are missing.
        WHAT: "Missing required columns: {missing}"
        WHY: "These columns are needed for {context}"
        HOW: "Add columns: df['column'] = values"

    ValueError
        If timestamp column contains non-numeric values.
        WHAT: "Timestamp column contains non-numeric values"
        WHY: "Timestamps must be numeric (seconds)"
        HOW: "Convert timestamps: df['timestamp'] = df['timestamp'].astype(float)"

    Examples
    --------
    >>> validate_events_dataframe(df, required_columns=["x", "y"], context="spatial_event_rate")
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pd.DataFrame, got {type(df).__name__}.\n"
            "  WHY: Events must be a pandas DataFrame for NWB compatibility.\n"
            "  HOW: Convert using pd.DataFrame({'timestamp': times})"
        )

    all_required = [timestamp_column] + (required_columns or [])
    missing = [col for col in all_required if col not in df.columns]
    if missing:
        context_str = f" for {context}" if context else ""
        raise ValueError(
            f"Missing required columns: {missing}.\n"
            f"  WHY: These columns are needed{context_str}.\n"
            f"  HOW: Add missing columns to DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )

    if not np.issubdtype(df[timestamp_column].dtype, np.number):
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains non-numeric values.\n"
            "  WHY: Timestamps must be numeric (seconds from session start).\n"
            f"  HOW: Convert timestamps: df['{timestamp_column}'] = df['{timestamp_column}'].astype(float)"
        )


def validate_spatial_columns(
    df: pd.DataFrame,
    *,
    require_positions: bool = False,
    context: str = "",
) -> bool:
    """
    Check if DataFrame has spatial columns (x, y); optionally require them.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame to check.
    require_positions : bool, default=False
        If True, raise ValueError when spatial columns missing.
    context : str, optional
        Context for error messages.

    Returns
    -------
    bool
        True if spatial columns (x, y) present, False otherwise.

    Raises
    ------
    ValueError
        If require_positions=True and spatial columns missing.
        WHAT: "Events DataFrame missing spatial columns"
        WHY: "{context} requires event positions"
        HOW: "Use add_positions(events, positions, times)"
    """
    has_x = "x" in df.columns
    has_y = "y" in df.columns
    has_positions = has_x and has_y

    if require_positions and not has_positions:
        raise ValueError(
            "Events DataFrame missing spatial columns ('x', 'y').\n"
            f"  WHY: {context} requires event positions.\n"
            "  HOW: Use add_positions(events, positions, times)"
        )

    return has_positions


# --- Result Dataclass ---

@dataclass(frozen=True)
class PeriEventResult:
    """Result from peri-event analysis.

    Attributes
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds). Negative = before, positive = after.
    histogram : NDArray[np.float64], shape (n_bins,)
        Spike count or firing rate per time bin.
    sem : NDArray[np.float64], shape (n_bins,)
        Standard error of the mean across events.
    n_events : int
        Number of events used in analysis.
    window : tuple[float, float]
        Time window (start, end) relative to event.
    bin_size : float
        Width of time bins (seconds).

    Methods
    -------
    firing_rate() : NDArray[np.float64]
        Convert histogram to firing rate (Hz).

    See Also
    --------
    plot_peri_event_histogram : Visualize PSTH results.
    """

    bin_centers: NDArray[np.float64]
    histogram: NDArray[np.float64]
    sem: NDArray[np.float64]
    n_events: int
    window: tuple[float, float]
    bin_size: float

    def firing_rate(self) -> NDArray[np.float64]:
        """Convert spike counts to firing rate (Hz)."""
        return self.histogram / self.bin_size


@dataclass(frozen=True)
class PopulationPeriEventResult:
    """Result from population peri-event analysis.

    Attributes
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds).
    histograms : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit spike count or firing rate.
    sem : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit standard error of the mean across events.
    mean_histogram : NDArray[np.float64], shape (n_bins,)
        Population average histogram.
    n_events : int
        Number of events used in analysis.
    n_units : int
        Number of units in population.
    window : tuple[float, float]
        Time window (start, end) relative to event.
    bin_size : float
        Width of time bins (seconds).
    """

    bin_centers: NDArray[np.float64]
    histograms: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_histogram: NDArray[np.float64]
    n_events: int
    n_units: int
    window: tuple[float, float]
    bin_size: float

    def firing_rates(self) -> NDArray[np.float64]:
        """Convert spike counts to firing rates (Hz) for all units."""
        return self.histograms / self.bin_size


def plot_peri_event_histogram(
    result: PeriEventResult,
    *,
    ax: "Axes | None" = None,
    show_sem: bool = True,
    color: str = "C0",
    as_rate: bool = True,
    title: str | None = None,
    xlabel: str = "Time from event (s)",
    ylabel: str | None = None,
) -> "Axes":
    """
    Plot peri-event time histogram (PSTH).

    Parameters
    ----------
    result : PeriEventResult
        Result from peri_event_histogram().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    show_sem : bool, default=True
        Show shaded SEM region.
    color : str, default="C0"
        Line color.
    as_rate : bool, default=True
        Plot as firing rate (Hz) rather than spike counts.
    title : str, optional
        Plot title. Default: "PSTH (n={n_events} events)".
    xlabel : str, default="Time from event (s)"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Default: "Firing rate (Hz)" or "Spike count".

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    See Also
    --------
    peri_event_histogram : Compute PSTH from spikes and events.

    Examples
    --------
    >>> result = peri_event_histogram(spikes, events, window=(-1, 2))
    >>> plot_peri_event_histogram(result, title="Reward response")
    """
    ...
```

### Detection Functions

```python
# src/neurospatial/events/detection.py

def extract_region_crossing_events(
    env: Environment,
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    region: str,
    *,
    direction: Literal["both", "entry", "exit"] = "both",
    min_dwell_time: float = 0.0,
) -> pd.DataFrame:
    """
    Extract region crossing events as DataFrame for event-based analysis.

    This function wraps the existing `detect_region_crossings()` from
    segmentation and returns a DataFrame compatible with the events module.
    Use this for event-triggered analysis; use `detect_region_crossings()`
    for trial segmentation workflows.

    Parameters
    ----------
    env : Environment
        Environment with region definitions.
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Bin indices from trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to trajectory (seconds).
    region : str
        Name of region in env.regions.
    direction : {"both", "entry", "exit"}, default="both"
        Which crossing types to detect. Uses "entry"/"exit" to match
        existing segmentation module terminology.
    min_dwell_time : float, default=0.0
        Minimum time in region before exit counts (seconds).
        Helps filter brief boundary crossings.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (float64), direction ("entry" or "exit"), region (str),
        x (float64), y (float64), bin_index (int64).
        Compatible with write_region_crossings() and all events functions.

    See Also
    --------
    detect_region_crossings : Original function in segmentation module.
    segment_trials : Segment trajectory into behavioral trials.
    write_region_crossings : Write crossings to NWB.

    Examples
    --------
    >>> crossings = extract_region_crossing_events(env, bins, times, "goal")
    >>> entries = crossings[crossings["direction"] == "entry"]
    >>> print(f"Animal entered goal zone {len(entries)} times")
    """


def extract_threshold_crossing_events(
    signal: NDArray[np.float64],
    times: NDArray[np.float64],
    threshold: float,
    *,
    direction: Literal["rising", "falling", "both"] = "rising",
    min_separation: float = 0.0,
    refractory_period: float = 0.0,
) -> pd.DataFrame:
    """
    Extract threshold crossing events from continuous signal.

    Parameters
    ----------
    signal : NDArray[np.float64], shape (n_samples,)
        Continuous signal (e.g., speed, LFP amplitude).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    threshold : float
        Threshold value to cross.
    direction : {"rising", "falling", "both"}, default="rising"
        Which crossings to detect.
    min_separation : float, default=0.0
        Minimum time between crossings (seconds).
    refractory_period : float, default=0.0
        Time after crossing before another can be detected.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, direction ("rising" or "falling"), value.

    Examples
    --------
    >>> # Detect when animal starts running (speed > 5 cm/s)
    >>> run_starts = extract_threshold_crossing_events(speed, times, 5.0, direction="rising")
    """


def extract_movement_onset_events(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    speed_threshold: float = 5.0,
    min_duration: float = 0.5,
    smoothing_window: float = 0.1,
) -> pd.DataFrame:
    """
    Extract movement onset events from position data.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    speed_threshold : float, default=5.0
        Speed threshold for movement (units/s).
    min_duration : float, default=0.5
        Minimum movement duration (seconds).
    smoothing_window : float, default=0.1
        Speed smoothing window (seconds).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, duration, peak_speed, x, y (position at onset).
    """


def add_positions(
    events: pd.DataFrame,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Add x, y position columns to events by interpolating from trajectory.

    For each event, finds the position at that time by linear interpolation
    of the trajectory.

    Parameters
    ----------
    events : pd.DataFrame
        Events with timestamp column.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for trajectory.
    timestamp_column : str, default="timestamp"
        Name of timestamp column in events.

    Returns
    -------
    pd.DataFrame
        Events with added columns: x, y (and z if 3D trajectory).

    Notes
    -----
    This function only adds coordinates. To query region membership or
    bin indices, use `events_in_region()` or `env.bin_at()` respectively.

    Examples
    --------
    >>> # Add positions to reward events
    >>> rewards_with_pos = add_positions(rewards, positions, times)
    >>> # Filter by region using query function
    >>> goal_rewards = events_in_region(rewards_with_pos, env, "goal")
    """


def events_in_region(
    events: pd.DataFrame,
    env: Environment,
    region: str,
) -> pd.DataFrame:
    """
    Filter events to those occurring within a region.

    Events must have x, y columns. Use `add_positions()` first if needed.

    Parameters
    ----------
    events : pd.DataFrame
        Events with x, y columns.
    env : Environment
        Environment with region definitions.
    region : str
        Region name to filter by.

    Returns
    -------
    pd.DataFrame
        Subset of events within the specified region.

    Raises
    ------
    ValueError
        If events DataFrame lacks x, y columns.

    Examples
    --------
    >>> rewards_with_pos = add_positions(rewards, positions, times)
    >>> goal_rewards = events_in_region(rewards_with_pos, env, "goal")
    """


def spatial_event_rate(
    events: pd.DataFrame,
    env: Environment,
    occupancy: NDArray[np.float64],
    *,
    smooth: bool = True,
    bandwidth: float = 5.0,
) -> NDArray[np.float64]:
    """
    Compute spatial rate map of events (events per second per bin).

    Events must have x, y columns. Use `add_positions()` first if needed.

    Parameters
    ----------
    events : pd.DataFrame
        Events with x, y columns.
    env : Environment
        Spatial environment.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time spent in each bin (seconds). Use `env.compute_occupancy()`.
    smooth : bool, default=True
        Apply spatial smoothing.
    bandwidth : float, default=5.0
        Smoothing bandwidth.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Event rate per spatial bin (events/second).

    Examples
    --------
    >>> rewards_with_pos = add_positions(rewards, positions, times)
    >>> occupancy = env.compute_occupancy(positions, times)
    >>> reward_rate = spatial_event_rate(rewards_with_pos, env, occupancy)
    >>> env.plot_field(reward_rate, title="Reward rate (rewards/s)")
    """
```

### Interval Functions

```python
# src/neurospatial/events/intervals.py

def intervals_to_events(
    intervals: pd.DataFrame,
    which: Literal["start", "stop", "both"] = "start",
    *,
    start_column: str = "start_time",
    stop_column: str = "stop_time",
    preserve_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert intervals to point events (start and/or stop times).

    Parameters
    ----------
    intervals : pd.DataFrame
        Intervals with start_time and stop_time columns.
    which : {"start", "stop", "both"}, default="start"
        Which interval boundaries to extract as events.
    start_column : str, default="start_time"
        Name of start time column.
    stop_column : str, default="stop_time"
        Name of stop time column.
    preserve_columns : list[str], optional
        Additional columns to preserve in output.

    Returns
    -------
    pd.DataFrame
        Events with 'timestamp' column. If which="both", includes
        'boundary' column ("start" or "stop").

    Examples
    --------
    >>> # Get trial start times as events
    >>> trial_starts = intervals_to_events(trials, which="start")
    >>> # Align neural activity to trial starts
    >>> psth = peri_event_histogram(spikes, trial_starts["timestamp"].values)
    """


def events_to_intervals(
    start_events: pd.DataFrame,
    stop_events: pd.DataFrame,
    *,
    match_by: str | None = None,
    max_duration: float | None = None,
) -> pd.DataFrame:
    """
    Pair start and stop events into intervals.

    Parameters
    ----------
    start_events : pd.DataFrame
        Events marking interval starts.
    stop_events : pd.DataFrame
        Events marking interval stops.
    match_by : str, optional
        Column to match start/stop events (e.g., "trial_id").
        If None, pairs sequentially (first start with first stop, etc.).
    max_duration : float, optional
        Maximum interval duration. Longer intervals are excluded.

    Returns
    -------
    pd.DataFrame
        Intervals with 'start_time', 'stop_time', and 'duration' columns.

    Raises
    ------
    ValueError
        If start/stop counts don't match (when match_by is None).

    Examples
    --------
    >>> # Convert zone entry/exit events to dwell intervals
    >>> entries = crossings[crossings["direction"] == "entry"]
    >>> exits = crossings[crossings["direction"] == "exit"]
    >>> dwell_intervals = events_to_intervals(entries, exits)
    """


def filter_by_intervals(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    include: bool = True,
    timestamp_column: str = "timestamp",
    start_column: str = "start_time",
    stop_column: str = "stop_time",
) -> pd.DataFrame:
    """
    Filter events to those within (or outside) intervals.

    Parameters
    ----------
    events : pd.DataFrame
        Events to filter.
    intervals : pd.DataFrame
        Intervals defining inclusion/exclusion periods.
    include : bool, default=True
        If True, keep events within intervals.
        If False, keep events outside intervals.
    timestamp_column : str, default="timestamp"
        Name of timestamp column in events.
    start_column : str, default="start_time"
        Name of start time column in intervals.
    stop_column : str, default="stop_time"
        Name of stop time column in intervals.

    Returns
    -------
    pd.DataFrame
        Filtered events.

    Examples
    --------
    >>> # Keep only rewards during running epochs
    >>> running_rewards = filter_by_intervals(rewards, running_epochs)
    >>> # Exclude events during artifact periods
    >>> clean_events = filter_by_intervals(events, artifacts, include=False)
    """
```

### GLM Regressor Functions

```python
# src/neurospatial/events/regressors.py

def time_since_event(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    max_time: float | None = None,
    fill_before_first: float | None = None,
    nan_policy: Literal["raise", "fill", "propagate"] = "propagate",
) -> NDArray[np.float64]:
    """
    Compute time since most recent event for each sample.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute regressor.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    max_time : float, optional
        Maximum time to return (clips at this value).
        Useful for capping distant events.
    fill_before_first : float, optional
        Value to use before first event. Default: NaN.
    nan_policy : {"raise", "fill", "propagate"}, default="propagate"
        How to handle NaN values in output:
        - "raise": Raise ValueError if any output would be NaN
        - "fill": Fill NaN with `fill_before_first` (required if policy="fill")
        - "propagate": Keep NaN values (default, suitable for GLMs with NaN handling)

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time since most recent event (seconds).
        NaN for samples before first event (unless fill_before_first set).

    Raises
    ------
    ValueError
        If nan_policy="raise" and output would contain NaN values.
        If nan_policy="fill" but fill_before_first is not provided.

    Notes
    -----
    Common use cases:
    - Time since reward: Captures reward expectation decay
    - Time since cue: Captures cue-triggered anticipation
    - Time since zone entry: Captures spatial context

    Examples
    --------
    >>> from neurospatial.events import time_since_event
    >>> sample_times = np.linspace(0, 10, 100)
    >>> reward_times = np.array([2.0, 5.0, 8.0])
    >>> time_since_reward = time_since_event(sample_times, reward_times)
    >>> # Use in GLM design matrix
    >>> X = sm.add_constant(time_since_reward[:, None])
    """


def time_to_event(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    max_time: float | None = None,
    fill_after_last: float | None = None,
    nan_policy: Literal["raise", "fill", "propagate"] = "propagate",
) -> NDArray[np.float64]:
    """
    Compute time until next event for each sample.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute regressor.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    max_time : float, optional
        Maximum time to return (clips at this value).
    fill_after_last : float, optional
        Value to use after last event. Default: NaN.
    nan_policy : {"raise", "fill", "propagate"}, default="propagate"
        How to handle NaN values in output:
        - "raise": Raise ValueError if any output would be NaN
        - "fill": Fill NaN with `fill_after_last` (required if policy="fill")
        - "propagate": Keep NaN values (default, suitable for GLMs with NaN handling)

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time until next event (seconds).
        NaN for samples after last event (unless fill_after_last set).

    Raises
    ------
    ValueError
        If nan_policy="raise" and output would contain NaN values.
        If nan_policy="fill" but fill_after_last is not provided.

    Notes
    -----
    Useful for prospective coding analyses:
    - Time to expected reward
    - Time to anticipated zone entry
    - Predictive neural signals
    """


def event_count_in_window(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
) -> NDArray[np.int64]:
    """
    Count events within time window around each sample.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute count.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, end) relative to sample_time.
        E.g., (-1.0, 0.0) counts events in previous 1 second.

    Returns
    -------
    NDArray[np.int64], shape (n_samples,)
        Number of events within window of each sample.

    Examples
    --------
    >>> # Count rewards in previous 5 seconds
    >>> recent_rewards = event_count_in_window(times, reward_times, (-5.0, 0.0))
    """


def event_indicator(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    window: float = 0.0,
) -> NDArray[np.bool_]:
    """
    Binary indicator: is there an event at/near this sample?

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps.
    window : float, default=0.0
        Half-width of window around event (seconds).
        If 0, only exact matches count.

    Returns
    -------
    NDArray[np.bool_], shape (n_samples,)
        True if event occurs within window of sample.
    """


def exponential_kernel(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    tau: float,
    *,
    direction: Literal["causal", "acausal", "symmetric"] = "causal",
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Convolve events with exponential kernel.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps.
    tau : float
        Time constant (seconds).
    direction : {"causal", "acausal", "symmetric"}, default="causal"
        Kernel direction:
        - "causal": exp(-t/tau) for t > 0 (decaying after event)
        - "acausal": exp(t/tau) for t < 0 (anticipatory before event)
        - "symmetric": both directions
    normalize : bool, default=True
        If True, kernel integrates to 1.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Convolved event signal.

    Notes
    -----
    This is a continuous-time convolution, not discrete.
    Efficient implementation uses sorted merging.

    Examples
    --------
    >>> # Reward expectation signal with 2s decay
    >>> reward_signal = exponential_kernel(times, reward_times, tau=2.0)
    """


def distance_to_event_at_time(
    sample_times: NDArray[np.float64],
    sample_positions: NDArray[np.float64],
    events: pd.DataFrame,
    env: Environment | None = None,
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    which: Literal["last", "next"] = "last",
) -> NDArray[np.float64]:
    """
    Compute distance to the spatial location of the last/next event.

    Finds the temporally relevant event (last before or next after each
    sample) and computes the spatial distance to where that event occurred.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps.
    sample_positions : NDArray[np.float64], shape (n_samples, n_dims)
        Sample positions.
    events : pd.DataFrame
        Events with timestamp, x, y columns.
    env : Environment, optional
        Required for geodesic distance.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric.
    which : {"last", "next"}, default="last"
        Use last event before sample or next event after sample.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance to event location.

    Notes
    -----
    Useful for analyzing prospective/retrospective spatial coding:
    - "last": How far from where I just got reward?
    - "next": How far from where I will get reward?

    Examples
    --------
    >>> # Distance to last reward location (retrospective)
    >>> dist_from_last = distance_to_event_at_time(
    ...     times, positions, rewards, env, which="last"
    ... )
    >>> # Distance to next reward location (prospective)
    >>> dist_to_next = distance_to_event_at_time(
    ...     times, positions, rewards, env, which="next"
    ... )
    """
```

### Peri-Event Analysis

```python
# src/neurospatial/events/alignment.py

def align_spikes_to_events(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float] = (-0.5, 1.0),
) -> list[NDArray[np.float64]]:
    """
    Align spike times to events, returning per-event spike times.

    This is the low-level function that returns spike times relative to
    each event. Use this for custom analyses or raster plots. For
    aggregated histograms, use `peri_event_histogram()` instead.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Spike times (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event times to align to (seconds).
    window : tuple[float, float], default=(-0.5, 1.0)
        Time window around events (start, end) in seconds.
        Negative = before event, positive = after.

    Returns
    -------
    list[NDArray[np.float64]]
        List of length n_events. Each element contains spike times
        relative to that event (negative = before, positive = after).

    See Also
    --------
    peri_event_histogram : Compute aggregated PSTH from aligned spikes.

    Examples
    --------
    >>> aligned = align_spikes_to_events(spikes, rewards, window=(-1, 2))
    >>> # Plot raster
    >>> for i, trial_spikes in enumerate(aligned):
    ...     plt.vlines(trial_spikes, i, i+1)
    >>> # Compute custom statistic per trial
    >>> spike_counts = [len(s) for s in aligned]
    """


def peri_event_histogram(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    window: tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    baseline_window: tuple[float, float] | None = None,
) -> PeriEventResult:
    """
    Compute peri-event time histogram (PSTH).

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Spike times (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event times to align to (seconds).
    window : tuple[float, float], default=(-0.5, 1.0)
        Time window around events (start, end) in seconds.
        Negative = before event, positive = after.
    bin_size : float, default=0.01
        Width of time bins (seconds). 10ms is typical.
    baseline_window : tuple[float, float], optional
        Window for baseline subtraction. If provided, subtract
        mean firing rate in this window from histogram.

    Returns
    -------
    PeriEventResult
        Result object with histogram and SEM.

    Notes
    -----
    **Standard Workflow**:

    1. Align spikes to events
    2. Bin aligned spikes
    3. Compute mean ± SEM across events
    4. Optionally baseline-subtract

    Examples
    --------
    >>> from neurospatial.events import peri_event_histogram, plot_peri_event_histogram
    >>> result = peri_event_histogram(spike_times, reward_times, window=(-1, 2))
    >>> plot_peri_event_histogram(result)  # Standard PSTH figure
    >>> print(f"Peak at {result.bin_centers[np.argmax(result.histogram)]:.2f}s")
    """


def population_peri_event_histogram(
    spike_trains: list[NDArray[np.float64]],
    event_times: NDArray[np.float64],
    *,
    window: tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    baseline_window: tuple[float, float] | None = None,
) -> "PopulationPeriEventResult":
    """
    Compute peri-event time histogram for multiple units.

    Parameters
    ----------
    spike_trains : list[NDArray[np.float64]]
        List of spike time arrays, one per unit. Each array has shape (n_spikes,).
    event_times : NDArray[np.float64], shape (n_events,)
        Event times to align to (seconds).
    window : tuple[float, float], default=(-0.5, 1.0)
        Time window around events (start, end) in seconds.
    bin_size : float, default=0.01
        Width of time bins (seconds).
    baseline_window : tuple[float, float], optional
        Window for baseline subtraction.

    Returns
    -------
    PopulationPeriEventResult
        Result object with per-unit histograms and population statistics.
        - histograms: shape (n_units, n_bins)
        - mean_histogram: shape (n_bins,) population average
        - sem: shape (n_units, n_bins) per-unit SEM across events

    Examples
    --------
    >>> spike_trains = [unit1_spikes, unit2_spikes, unit3_spikes]
    >>> result = population_peri_event_histogram(spike_trains, reward_times)
    >>> # Plot population average
    >>> plt.plot(result.bin_centers, result.mean_histogram)
    """


def align_events(
    events_df: pd.DataFrame,
    reference_events: pd.DataFrame,
    *,
    window: tuple[float, float] = (-1.0, 1.0),
    reference_column: str = "timestamp",
    event_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Align events to reference events (peri-event extraction).

    For each reference event, extract all events within the window
    and compute their time relative to the reference.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events to align. Must have column specified by event_column.
    reference_events : pd.DataFrame
        Reference events. Must have column specified by reference_column.
    window : tuple[float, float], default=(-1.0, 1.0)
        Time window around reference events.
    reference_column : str, default="timestamp"
        Column name for reference timestamps.
    event_column : str, default="timestamp"
        Column name for event timestamps.

    Returns
    -------
    pd.DataFrame
        Aligned events with additional columns:
        - relative_time: time relative to reference event
        - reference_index: index of reference event in reference_events

    Examples
    --------
    >>> # Align licks to reward deliveries
    >>> aligned = align_events(licks_df, rewards_df, window=(-0.5, 2.0))
    >>> # Plot lick times relative to reward
    >>> plt.hist(aligned["relative_time"], bins=50)
    """
```

---

## Integration with Existing Code

### NWB Module Extensions

The existing `nwb/_events.py` already has `read_events()` and `write_region_crossings()`. Add:

```python
# Add to nwb/_events.py

def events_to_dataframe(
    events_table: "EventsTable",
) -> pd.DataFrame:
    """Convert ndx-events EventsTable to standardized DataFrame."""
    # Already exists as events_table_to_dataframe in _adapters.py
    # Just re-export or rename for consistency


def dataframe_to_events_table(
    df: pd.DataFrame,
    name: str,
    description: str = "Events",
) -> "EventsTable":
    """Convert DataFrame to ndx-events EventsTable for writing to NWB."""
    ...


def write_events(
    nwbfile: "NWBFile",
    events: pd.DataFrame,
    name: str,
    *,
    description: str = "Event data",
    processing_module: str = "behavior",
    overwrite: bool = False,
) -> None:
    """
    Write generic events DataFrame to NWB EventsTable.

    This is a more flexible version of write_laps/write_region_crossings.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to write to.
    events : pd.DataFrame
        Events data. Must have 'timestamp' column.
        Other columns become EventsTable columns.
    name : str
        Name for EventsTable in NWB.
    description : str, default="Event data"
        Description for EventsTable.
    processing_module : str, default="behavior"
        Processing module to store events in.
    overwrite : bool, default=False
        If True, replace existing table with same name.
    """
```

### Metrics Module Integration

Add to `metrics/__init__.py`:

```python
# Circular basis for events (already exists for angles)
# No changes needed - circular_basis works for any periodic variable

# For event-related metrics, create new module or add to existing
```

### Top-Level Exports

Add to `__init__.py`:

```python
# Events module
from neurospatial.events import (
    # Detection
    extract_region_crossing_events,
    extract_threshold_crossing_events,
    extract_movement_onset_events,
    # Spatial utilities
    add_positions,
    events_in_region,
    spatial_event_rate,
    # Interval utilities
    intervals_to_events,
    events_to_intervals,
    filter_by_intervals,
    # Temporal regressors
    time_since_event,
    time_to_event,
    event_count_in_window,
    event_indicator,
    exponential_kernel,
    # Spatial regressors
    distance_to_event_at_time,
    # Analysis
    align_spikes_to_events,
    peri_event_histogram,
    population_peri_event_histogram,
    align_events,
    # Visualization
    plot_peri_event_histogram,
    # Result types
    PeriEventResult,
    PopulationPeriEventResult,
    # Validation helpers
    validate_events_dataframe,
    validate_spatial_columns,
)
```

---

## Implementation Order

### Phase 1: Core Infrastructure

1. Create `events/` module structure
2. Implement `PeriEventResult` and `PopulationPeriEventResult` dataclasses
3. Implement validation helpers
4. Add basic tests

### Phase 2: Temporal GLM Regressors

5. Implement `time_since_event()` and `time_to_event()`
6. Implement `event_count_in_window()` and `event_indicator()`
7. Implement `exponential_kernel()`
8. Add comprehensive tests with edge cases

### Phase 3: Detection Functions

9. Implement `extract_region_crossing_events()` (wraps segmentation)
10. Implement `extract_threshold_crossing_events()`
11. Implement `extract_movement_onset_events()`
12. Add tests

### Phase 4: Interval Utilities

13. Implement `intervals_to_events()`
14. Implement `events_to_intervals()`
15. Implement `filter_by_intervals()`
16. Add tests

### Phase 5: Spatial Event Utilities

1. Implement `add_positions()`
2. Implement `events_in_region()`
3. Implement `spatial_event_rate()`
4. Implement `distance_to_event_at_time()`

### Phase 6: Peri-Event Analysis

1. Implement `align_spikes_to_events()`
2. Implement `peri_event_histogram()`
3. Implement `population_peri_event_histogram()`
4. Implement `align_events()`
5. Add tests and doctest examples

### Phase 7: NWB Integration

1. Add `write_events()` to NWB module
2. Ensure round-trip compatibility with spatial columns
3. Add integration tests

### Phase 8: Documentation

1. Update QUICKSTART.md with events examples
2. Update API_REFERENCE.md
3. Add to CLAUDE.md documentation

---

## Edge Case Handling

All functions must handle these edge cases gracefully with clear behavior:

### Empty Events

| Function | Behavior for Empty Events |
|----------|---------------------------|
| `time_since_event(t, [])` | Returns array of NaN (or raises if `nan_policy="raise"`) |
| `time_to_event(t, [])` | Returns array of NaN (or raises if `nan_policy="raise"`) |
| `event_count_in_window(t, [], window)` | Returns array of zeros |
| `event_indicator(t, [])` | Returns array of False |
| `exponential_kernel(t, [], tau)` | Returns array of zeros |
| `peri_event_histogram(spikes, [])` | Raises `ValueError`: "No events provided" |
| `align_events(events, [])` | Returns empty DataFrame |

### Single Event

| Function | Behavior for Single Event |
|----------|---------------------------|
| `time_since_event(t, [e])` | Valid: time since that one event |
| `peri_event_histogram(spikes, [e])` | Warning: "Only 1 event, SEM undefined" (SEM=NaN) |
| `event_triggered_ratemap(...)` | Works but noisy; logs warning |

### Events Outside Sample Time Range

| Condition | Behavior |
|-----------|----------|
| All events before first sample | `time_since_event` returns all NaN before clipping |
| All events after last sample | `time_to_event` returns all NaN before clipping |
| Spikes outside peri-event window | Excluded from histogram (no warning) |

### Identical Timestamps

| Condition | Behavior |
|-----------|----------|
| Multiple events at same time | Treated as single event for distance calculations |
| Duplicate spikes at same time | Both counted in PSTH |

### NaN/Inf in Input

| Input | Behavior |
|-------|----------|
| NaN in `sample_times` | Raises `ValueError`: "sample_times contains NaN" |
| NaN in `event_times` | Raises `ValueError`: "event_times contains NaN" |
| Inf in `positions` | Raises `ValueError`: "positions contains inf" |

### Unsorted Inputs

| Input | Behavior |
|-------|----------|
| Unsorted `event_times` | Automatically sorted internally (no warning) |
| Unsorted `sample_times` | Works correctly (uses searchsorted on sorted events) |
| Unsorted trajectory | Works but may produce unexpected results for interpolation |

---

## Testing Strategy

### Unit Tests

- Each function gets dedicated test file
- Test edge cases: empty arrays, single event, events outside range
- Test numerical precision for time calculations

### Integration Tests

- Round-trip: detect → write to NWB → read back → verify
- GLM workflow: events → regressors → fit → validate
- PSTH workflow: spikes + events → histogram → plot

### Property-Based Tests (hypothesis)

- Time calculations are consistent (time_since + time_to = correct)
- Event counts are always non-negative integers
- Regressors handle unsorted inputs correctly

---

## Example Workflows

### GLM with Event Regressors

```python
from neurospatial import Environment, compute_place_field
from neurospatial.events import time_since_event, exponential_kernel
from neurospatial.basis import geodesic_rbf_basis
import statsmodels.api as sm

# Create environment and spatial basis
env = Environment.from_samples(positions, bin_size=2.0)
spatial_basis = geodesic_rbf_basis(env, n_centers=50, sigma=10.0)

# Get spatial design matrix
bin_indices = env.bin_sequence(trajectory, times)
X_spatial = spatial_basis[:, bin_indices].T

# Add event regressors
X_time_since_reward = time_since_event(times, reward_times, max_time=10.0)
X_reward_decay = exponential_kernel(times, reward_times, tau=2.0)

# Combine design matrix
X = np.column_stack([X_spatial, X_time_since_reward[:, None], X_reward_decay[:, None]])
X = sm.add_constant(X)

# Fit GLM
model = sm.GLM(spike_counts, X, family=sm.families.Poisson())
result = model.fit()

# Extract spatial coefficients and reconstruct place field
beta_spatial = result.params[1:len(spatial_basis)+1]
place_field = beta_spatial @ spatial_basis
env.plot_field(place_field, title="Place Field (controlling for reward)")
```

### Peri-Event Analysis

```python
from neurospatial.events import (
    peri_event_histogram,
    population_peri_event_histogram,
    extract_region_crossing_events,
    plot_peri_event_histogram,
)

# Extract reward zone entries as events DataFrame
reward_entries = extract_region_crossing_events(
    env, trajectory_bins, times, "reward_zone",
    direction="entry"
)

# Compute PSTH around reward entries (single unit)
psth = peri_event_histogram(
    spike_times,
    reward_entries["timestamp"].values,
    window=(-2, 3),
    bin_size=0.05,
)
plot_peri_event_histogram(psth)
print(f"Peak response at {psth.bin_centers[np.argmax(psth.histogram)]:.2f}s")

# Compute population PSTH (multiple units)
spike_trains = [unit1_spikes, unit2_spikes, unit3_spikes]
pop_result = population_peri_event_histogram(
    spike_trains,
    reward_entries["timestamp"].values,
    window=(-2, 3),
    bin_size=0.05,
)

# Plot population average with individual units
fig, ax = plt.subplots()
for i, unit_hist in enumerate(pop_result.histograms):
    ax.plot(pop_result.bin_centers, unit_hist, alpha=0.3, label=f"Unit {i+1}")
ax.plot(pop_result.bin_centers, pop_result.mean_histogram, "k-", lw=2, label="Population")
ax.set_xlabel("Time from reward (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.legend()
```

---

## Design Decisions

1. **`detect_region_crossings` stays in `segmentation` module**
   - The segmentation module has a specific behavioral role (trial segmentation)
   - `extract_region_crossing_events()` in events module wraps it for event-based analysis
   - No aliasing needed; clear separation of concerns

2. **Multi-unit PETHs will be supported**
   - Add `population_peri_event_histogram()` that accepts spike trains from multiple units
   - Returns shape `(n_units, n_bins)` with per-unit and population statistics

3. **Rate cubes deferred to future version**
   - `event_triggered_ratemap()` is NOT included in initial implementation
   - Memory-intensive; needs lazy evaluation design
   - Can be added later once core events functionality is stable

---

## Success Criteria

1. **All existing tests pass** after integration
2. **NWB round-trip works** for all event types
3. **GLM workflow example** runs successfully
4. **PSTH matches** scipy/elephant implementations (validated)
5. **Documentation** is complete and follows codebase patterns
6. **Type checking passes** (mypy)
7. **Linting passes** (ruff)
