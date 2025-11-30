"""Time series rendering for animation backends.

This module provides the TimeSeriesArtistManager class and helper functions
for rendering time series overlays alongside spatial field animations.

The architecture follows the existing overlay pattern:
1. TimeSeriesData (from overlays.py) contains precomputed window indices
2. TimeSeriesArtistManager creates matplotlib artists once and updates them per frame
3. Helper functions organize overlays into groups for stacked vs overlaid display
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.animation.overlays import TimeSeriesData


def _group_timeseries(
    timeseries_data: list[TimeSeriesData],
) -> list[list[TimeSeriesData]]:
    """Group time series overlays by their group parameter.

    Overlays with the same `group` value are placed in the same group
    (overlaid on same axes). Overlays with `group=None` each get their
    own group (stacked as separate rows).

    Parameters
    ----------
    timeseries_data : list[TimeSeriesData]
        List of time series data containers to group.

    Returns
    -------
    list[list[TimeSeriesData]]
        List of groups, where each group is a list of overlays that
        should share the same axes.

    Examples
    --------
    >>> # Two overlays with same group -> one group
    >>> data1 = TimeSeriesData(..., group="kinematics")
    >>> data2 = TimeSeriesData(..., group="kinematics")
    >>> groups = _group_timeseries([data1, data2])
    >>> len(groups)  # 1 group with 2 items
    1

    >>> # Two overlays with no group -> two groups
    >>> data1 = TimeSeriesData(..., group=None)
    >>> data2 = TimeSeriesData(..., group=None)
    >>> groups = _group_timeseries([data1, data2])
    >>> len(groups)  # 2 groups with 1 item each
    2
    """
    if not timeseries_data:
        return []

    # Track which named groups we've seen
    named_groups: dict[str, list[TimeSeriesData]] = {}
    # Ungrouped items get their own list
    ungrouped: list[list[TimeSeriesData]] = []

    for ts_data in timeseries_data:
        if ts_data.group is None:
            # Each ungrouped overlay gets its own group
            ungrouped.append([ts_data])
        else:
            # Named groups are combined
            if ts_data.group not in named_groups:
                named_groups[ts_data.group] = []
            named_groups[ts_data.group].append(ts_data)

    # Build final list: named groups first, then ungrouped
    result: list[list[TimeSeriesData]] = []
    result.extend(named_groups.values())
    result.extend(ungrouped)

    return result


def _get_group_index(
    ts_data: TimeSeriesData,
    groups: list[list[TimeSeriesData]],
) -> int:
    """Get the index of the group containing a time series overlay.

    Parameters
    ----------
    ts_data : TimeSeriesData
        Time series data to find.
    groups : list[list[TimeSeriesData]]
        Grouped time series data from _group_timeseries().

    Returns
    -------
    int
        Index of the group containing ts_data.

    Raises
    ------
    ValueError
        If ts_data is not found in any group.
    """
    for idx, group in enumerate(groups):
        # Use identity check (is) rather than equality (in) because
        # TimeSeriesData contains numpy arrays which have ambiguous truth values
        for item in group:
            if item is ts_data:
                return idx
    raise ValueError(f"TimeSeriesData with label '{ts_data.label}' not found in groups")


@dataclass
class TimeSeriesArtistManager:
    """Manages matplotlib artists for time series rendering.

    Creates matplotlib Figure, Axes, Line2D, and Text artists once during
    setup, then efficiently updates them each frame. This avoids the overhead
    of recreating artists per frame.

    Parameters
    ----------
    axes : list[Axes]
        One Axes per group (row).
    lines : dict[str, Line2D]
        Mapping from label (or id) to Line2D artist.
    cursors : list[Line2D]
        One cursor line per Axes.
    value_texts : dict[str, Text]
        Mapping from label to Text artist for cursor value display.
    frame_times : NDArray[np.float64]
        Animation frame timestamps.
    group_window_seconds : dict[int, float]
        Window width per group index.

    Notes
    -----
    The manager is designed to be created once per worker/figure and updated
    each frame via `update()`. This follows the same pattern as
    OverlayArtistManager in _parallel.py.
    """

    axes: list[Axes]
    lines: dict[str, Line2D]
    cursors: list[Line2D | None]  # None if cursor disabled for that group
    value_texts: dict[str, Text]
    frame_times: NDArray[np.float64]
    group_window_seconds: dict[int, float]
    _ts_data_keys: dict[int, str] = field(default_factory=dict, repr=False)
    _groups: list[list[Any]] = field(default_factory=list, repr=False)  # Cached groups

    @classmethod
    def create(
        cls,
        fig: Figure,
        timeseries_data: list[TimeSeriesData],
        frame_times: NDArray[np.float64],
        dark_theme: bool = True,
    ) -> TimeSeriesArtistManager:
        """Create artist manager with matplotlib figure and artists.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to add axes to.
        timeseries_data : list[TimeSeriesData]
            Time series data containers from conversion pipeline.
        frame_times : NDArray[np.float64]
            Animation frame timestamps.
        dark_theme : bool, default=True
            Apply napari-style dark theme.

        Returns
        -------
        TimeSeriesArtistManager
            Initialized manager ready for per-frame updates.
        """
        if not timeseries_data:
            # Empty manager for no time series
            return cls(
                axes=[],
                lines={},
                cursors=[],
                value_texts={},
                frame_times=frame_times,
                group_window_seconds={},
                _ts_data_keys={},
                _groups=[],
            )

        groups = _group_timeseries(timeseries_data)
        n_rows = len(groups)

        # Create axes as subplots (one per row/group)
        axes_array = fig.subplots(n_rows, 1, squeeze=False)
        axes: list[Axes] = list(axes_array[:, 0])

        # Build per-group settings and check for conflicts
        group_window_seconds: dict[int, float] = {}
        group_normalize: dict[int, bool] = {}

        for ts_data in timeseries_data:
            group_idx = _get_group_index(ts_data, groups)

            if group_idx not in group_window_seconds:
                group_window_seconds[group_idx] = ts_data.window_seconds
                group_normalize[group_idx] = ts_data.normalize
            else:
                # Check for conflicts and warn
                existing_window = group_window_seconds[group_idx]
                if ts_data.window_seconds != existing_window:
                    warnings.warn(
                        f"Group has conflicting window_seconds "
                        f"({existing_window} vs {ts_data.window_seconds}). "
                        f"Using first value: {existing_window}",
                        UserWarning,
                        stacklevel=2,
                    )

                existing_normalize = group_normalize[group_idx]
                if ts_data.normalize != existing_normalize:
                    warnings.warn(
                        f"Group has mixed normalize settings "
                        f"({existing_normalize} vs {ts_data.normalize}). "
                        f"This produces confusing y-axis semantics. "
                        f"Consider using normalize=True for all or none.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Apply dark theme styling
        if dark_theme:
            napari_bg = "#262930"
            fig.patch.set_facecolor(napari_bg)
            for ax in axes:
                ax.set_facecolor(napari_bg)
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.yaxis.label.set_color("white")
                ax.xaxis.label.set_color("white")

        # Create Line2D artists and value text annotations
        lines: dict[str, Line2D] = {}
        value_texts: dict[str, Text] = {}
        ts_data_keys: dict[int, str] = {}

        for ts_data in timeseries_data:
            group_idx = _get_group_index(ts_data, groups)
            ax = axes[group_idx]

            # Create line artist (empty initially)
            (line,) = ax.plot(
                [],
                [],
                color=ts_data.color,
                linewidth=ts_data.linewidth,
                alpha=ts_data.alpha,
                label=ts_data.label if ts_data.label else None,
            )

            # Use label as key, or generate unique key from id
            key = ts_data.label if ts_data.label else str(id(ts_data))
            lines[key] = line
            ts_data_keys[id(ts_data)] = key

            # Create text annotation for cursor value display
            text_color = ts_data.color if dark_theme else "black"
            bbox_props = {
                "boxstyle": "round,pad=0.2",
                "facecolor": "#262930" if dark_theme else "white",
                "alpha": 0.7,
                "edgecolor": "none",
            }
            value_text = ax.text(
                0.98,
                0.95,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=text_color,
                bbox=bbox_props,
            )
            value_texts[key] = value_text

            # Set stable y-limits from global values
            if ts_data.use_global_limits:
                y_range = ts_data.global_vmax - ts_data.global_vmin
                margin = y_range * 0.05 if y_range > 0 else 0.5
                ax.set_ylim(
                    ts_data.global_vmin - margin,
                    ts_data.global_vmax + margin,
                )

            # Add y-label from first overlay in group
            if ts_data.label and not ax.get_ylabel():
                ax.set_ylabel(
                    ts_data.label, fontsize=9, color="white" if dark_theme else "black"
                )

        # Create cursor lines (one per axes, None if disabled)
        cursors: list[Line2D | None] = []
        for group_idx, ax in enumerate(axes):
            # Check if any overlay in this group wants cursor
            show_cursor_in_group = False
            cursor_color = "red"  # Default

            for ts_data in timeseries_data:
                if (
                    _get_group_index(ts_data, groups) == group_idx
                    and ts_data.show_cursor
                ):
                    show_cursor_in_group = True
                    cursor_color = ts_data.cursor_color
                    break

            # Only create cursor if requested by any overlay in group
            if show_cursor_in_group:
                cursor = ax.axvline(x=0, color=cursor_color, linewidth=1, alpha=0.8)
            else:
                cursor = None
            cursors.append(cursor)

        # Configure x-axis labels (only on bottom axes)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        if axes:
            label_color = "white" if dark_theme else "black"
            axes[-1].set_xlabel("Time (s)", fontsize=9, color=label_color)

        # Add legends for axes with multiple lines
        for ax in axes:
            ax_lines = [
                line
                for line in ax.get_lines()
                if (label := line.get_label())
                and isinstance(label, str)
                and not label.startswith("_")
            ]
            if len(ax_lines) > 1:
                ax.legend(loc="upper left", fontsize=7, framealpha=0.5)

        fig.tight_layout()

        return cls(
            axes=axes,
            lines=lines,
            cursors=cursors,
            value_texts=value_texts,
            frame_times=frame_times,
            group_window_seconds=group_window_seconds,
            _ts_data_keys=ts_data_keys,
            _groups=groups,  # Cache groups to avoid recomputing in update()
        )

    def update(
        self,
        frame_idx: int,
        timeseries_data: list[TimeSeriesData],
    ) -> None:
        """Update all artists for given frame.

        Parameters
        ----------
        frame_idx : int
            Current animation frame index.
        timeseries_data : list[TimeSeriesData]
            Same time series data passed to create().
        """
        if not timeseries_data or not self.axes:
            return

        current_time = float(self.frame_times[frame_idx])

        # Update lines and value texts
        for ts_data in timeseries_data:
            # Key was populated in create() - always present for valid inputs
            key = self._ts_data_keys.get(id(ts_data))
            if key is None or key not in self.lines:
                continue

            # O(1) slice using precomputed indices
            y_data, t_data = ts_data.get_window_slice(frame_idx)

            # Update line data
            line = self.lines[key]
            line.set_data(t_data, y_data)

            # Update cursor value text (get_cursor_value returns None if outside range)
            value_text = self.value_texts.get(key)
            if value_text is not None and ts_data.show_cursor:
                cursor_value = ts_data.get_cursor_value(current_time)
                if cursor_value is not None and ts_data.label:
                    value_text.set_text(f"{ts_data.label}: {cursor_value:.2f}")
                else:
                    value_text.set_text("")  # Clear when no value available

        # Update x-axis limits per group
        for group_idx, ax in enumerate(self.axes):
            window_seconds = self.group_window_seconds.get(group_idx, 2.0)
            half_window = window_seconds / 2
            ax.set_xlim(current_time - half_window, current_time + half_window)

        # Update cursor positions (skip if cursor disabled)
        for cursor in self.cursors:
            if cursor is not None:
                cursor.set_xdata([current_time, current_time])
