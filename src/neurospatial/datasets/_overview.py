"""The overview figure and the interactive scene of a loaded session."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import wrap_angle
from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._recording import tracked_mask

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from neurospatial import Environment
    from neurospatial.animation import Animation
    from neurospatial.datasets._recording import Recording

__all__: list[str] = []

#: Seconds of tracking the overview's time panel and the animation open on.
WINDOW_SECONDS = 60.0

#: Frames per second of the scene's clock for faster tracking (viewers play at most 60).
MAX_FRAME_RATE = 30.0

#: The most opaque occupancy is drawn over a session's video, so the camera
#: shows through every bin, the most visited too.
VIDEO_FIELD_OPACITY = 0.5

_EPOCH_COLORS = ("gold", "tab:cyan", "tab:pink", "tab:olive", "tab:purple")


def _note(ax: Axes, title: str, text: str) -> None:
    """Replace a panel that cannot be drawn with a note saying why."""
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.35",
    )


def _missing(ax: Axes, title: str, what: str) -> None:
    """Say the dataset has no data for this panel."""
    _note(ax, title, f"not in this dataset\n({what})")


def _min_speed(rec: Recording) -> float | None:
    value = rec.params.get("min_speed")
    return None if value is None else float(value)


def occupancy_field(
    rec: Recording, env: Environment | None = None
) -> NDArray[np.float64]:
    """Seconds in each bin over the whole session, NaN where never visited.

    Parameters
    ----------
    rec : Recording
        A session with positions and an environment.
    env : Environment, optional
        The bins to count in; default ``rec.env``.

    Returns
    -------
    ndarray, shape (n_bins,)
        ``env.occupancy(rec.times, rec.positions)`` with unvisited bins
        (zero seconds) set to NaN, so they draw as missing rather than as zero.
    """
    env = rec.env if env is None else env
    assert env is not None
    assert rec.positions is not None
    occupancy: NDArray[np.float64] = env.occupancy(rec.times, rec.positions)
    return np.where(occupancy > 0, occupancy, np.nan)


def _panel_environment(rec: Recording, ax: Axes) -> None:
    title = "Environment and path"
    if rec.env is None or rec.positions is None:
        _missing(ax, title, "no positions")
        return
    if rec.env.n_dims == 3:
        _top_view_path(rec, ax)
        return
    rec.env.plot(ax=ax)
    if rec.positions.shape[1] == 2:
        ax.scatter(
            rec.positions[:, 0],
            rec.positions[:, 1],
            s=1,
            c="tab:red",
            linewidths=0,
            alpha=0.5,
            zorder=5,
            rasterized=True,
        )
    else:
        _along_track_strip(rec, ax)
    ax.set_title(title)


def _along_track_strip(rec: Recording, ax: Axes) -> None:
    """The tracked samples' positions along a 1-D track, as a histogram below it.

    One bar per environment bin, on the track's own x axis, so the strip
    shows where along the track the samples lie.
    """
    assert rec.env is not None
    assert rec.positions is not None
    along = rec.positions[tracked_mask(rec.positions), 0]
    centers = rec.env.bin_centers[:, 0]
    order = np.argsort(centers)
    half = rec.env.bin_sizes[order] / 2
    edges = np.unique(np.r_[centers[order] - half, centers[order][-1] + half[-1]])
    strip = ax.inset_axes((0.0, 0.0, 1.0, 0.38), sharex=ax)
    strip.hist(along, bins=edges.tolist(), color="tab:red", alpha=0.7)
    strip.set_ylabel("samples", fontsize=8)
    strip.tick_params(labelsize=7)
    strip.set_xlabel(f"position along the track ({rec.position_units})")
    ax.tick_params(labelbottom=False)


def _panel_occupancy(rec: Recording, ax: Axes) -> None:
    if rec.env is None or rec.positions is None:
        _missing(ax, "Occupancy", "no positions")
        return
    if rec.env.n_dims == 3:
        _top_view_occupancy(rec, ax)
        return
    rec.env.plot_field(
        occupancy_field(rec), ax=ax, colorbar_label="s", nan_color="lightgray"
    )
    ax.set_title("Occupancy (gray: never visited)")


TOP_VIEW = "top view (x\u2013y, summed over z)"


def _top_view_path(rec: Recording, ax: Axes) -> None:
    """A 3-D session's path seen from above, inside the room's x–y bounds."""
    from matplotlib.patches import Rectangle

    assert rec.env is not None
    assert rec.positions is not None
    if rec.env.dimension_ranges is not None:
        (x0, x1), (y0, y1) = rec.env.dimension_ranges[:2]
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, color="0.4", linestyle="--"
            )
        )
    ax.scatter(
        rec.positions[:, 0],
        rec.positions[:, 1],
        s=1,
        c="tab:red",
        linewidths=0,
        alpha=0.5,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="datalim")
    unit = rec.position_units
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"Environment and path\n{TOP_VIEW}")


def top_view_occupancy(rec: Recording) -> tuple[Any, NDArray[np.float64]]:
    """A 3-D grid session's occupancy summed over z, NaN where never visited.

    Parameters
    ----------
    rec : Recording
        A session whose environment is an axis-aligned 3-D grid.

    Returns
    -------
    edges : tuple of ndarray or None
        The grid's x and y cell edges, shapes (n_x + 1,) and (n_y + 1,);
        None when the environment is not an axis-aligned grid.
    seconds : ndarray, shape (n_x, n_y)
        Seconds in each x–y column, all heights together; NaN where no
        height of the column was visited (or the column holds no bin).
        Empty without a grid.
    """
    assert rec.env is not None
    grid = rec.env._binning.rectilinear
    if grid is None or rec.env._binning.grid_frame is not None:
        return None, np.empty((0, 0))
    occupancy = np.nan_to_num(occupancy_field(rec))
    volume = np.zeros(grid.shape)
    volume[grid.active_mask] = occupancy
    seconds = volume.sum(axis=2)
    return grid.edges[:2], np.where(seconds > 0, seconds, np.nan)


def _top_view_occupancy(rec: Recording, ax: Axes) -> None:
    title = f"Occupancy, {TOP_VIEW}\n(gray: never visited)"
    edges, seconds = top_view_occupancy(rec)
    if edges is None:
        _note(
            ax,
            "Occupancy",
            "3-D environment without an axis-aligned\ngrid: open rec.animation()",
        )
        return
    ax.set_facecolor("lightgray")
    # pcolormesh leaves NaN (never visited) cells undrawn: the gray shows.
    mesh = ax.pcolormesh(edges[0], edges[1], seconds.T, cmap="viridis")
    ax.figure.colorbar(mesh, ax=ax, label="s")
    ax.set_aspect("equal", adjustable="datalim")
    unit = rec.position_units
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(title)


def _panel_speed(rec: Recording, ax: Axes) -> None:
    if rec.positions is None:
        _missing(ax, "Speed", "no positions")
        return
    if rec._speed is None:
        _note(ax, "Speed", "no speed: no two tracked\nsamples within 1 s")
        return
    speed = rec._speed[np.isfinite(rec._speed)]
    _median, _p95, p999, fastest = rec._speed_quantiles
    # A few tracking jumps would set the axis; the histogram stops at the
    # 99.9th percentile and the panel says what lies beyond it.
    limited = bool(np.isfinite(p999) and p999 > speed.min() and fastest > p999)
    counts, _, _ = ax.hist(
        speed,
        bins=60,
        range=(float(speed.min()), p999) if limited else None,
        color="0.4",
    )
    # A session spent mostly still (a bat on its perch) puts most samples
    # in the first bar; a log count axis keeps the moving samples visible.
    log_counts = bool(np.max(counts) > 0.5 * speed.size)
    if log_counts:
        ax.set_yscale("log")
    if limited:
        n_beyond = int(np.sum(speed > p999))
        unit = "" if rec.position_units == "arbitrary" else f" {rec.position_units}/s"
        ax.text(
            0.97,
            0.75,
            f"{n_beyond} samples beyond the 99.9th\npercentile ({p999:.3g}"
            f"{unit}) not drawn;\nfastest {fastest:.3g}{unit}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="0.25",
        )
    min_speed = _min_speed(rec)
    if min_speed is not None:
        ax.axvline(min_speed, color="tab:red", label=f"min_speed {min_speed:g}")
        ax.legend(frameon=False)
    ax.set_xlabel(f"speed ({rec.position_units}/s)")
    ax.set_ylabel("samples (log scale)" if log_counts else "samples")
    ax.set_title("Speed")


def _panel_heading(rec: Recording, ax: Axes) -> None:
    if rec.headings is None:
        _missing(ax, "Head direction", "no headings")
        return
    edges = np.linspace(-np.pi, np.pi, 37)
    centers = (edges[:-1] + edges[1:]) / 2
    headings = rec.headings[np.isfinite(rec.headings)]
    if not headings.size:
        _missing(ax, "Head direction", "every heading is NaN")
        return
    counts, _ = np.histogram(headings, bins=edges)
    ax.bar(centers, counts / counts.sum(), width=np.diff(edges), color="0.5")
    ax.set_title("Head direction (0 = East)", pad=14)
    if rec.positions is None or rec.positions.shape[1] < 2:
        return
    from neurospatial.behavior import heading_from_velocity

    min_speed = _min_speed(rec)
    travel = heading_from_velocity(
        rec.times, rec.positions, min_speed=0.0 if min_speed is None else min_speed
    )
    offset = wrap_angle(rec.headings - travel)
    offset = offset[np.isfinite(offset)]
    if not offset.size:
        return
    # Its own radial scale: it is concentrated where the main histogram is flat.
    inset = ax.inset_axes((0.78, 0.82, 0.28, 0.28), projection="polar")
    off_counts, _ = np.histogram(offset, bins=edges)
    inset.bar(
        centers, off_counts / off_counts.sum(), width=np.diff(edges), color="tab:red"
    )
    running = "while running" if min_speed is not None else "all samples"
    inset.set_title(f"heading - travel\n({running})", fontsize=7)
    inset.set_yticklabels([])
    inset.set_xticklabels([])


def _panel_units(rec: Recording, ax: Axes) -> None:
    if rec.n_units == 0:
        _missing(ax, "Units", "no units")
        return
    regions = rec.units["brain_region"].fillna("unlabelled").astype(str)
    per_region = regions.value_counts().sort_index()
    ax.bar(per_region.index, per_region.to_numpy(), width=0.5, color="0.4")
    ax.set_xlim(-1, max(len(per_region), 2))
    ax.set_ylabel("units")
    ax.set_title("Units per region; spikes per unit (inset)")
    inset = ax.inset_axes((0.55, 0.5, 0.42, 0.42))
    inset.hist(rec.units["n_spikes"].to_numpy(), bins=20, color="tab:blue")
    inset.set_xlabel("spikes per unit", fontsize=7)
    inset.tick_params(labelsize=7)


#: Brain regions whose units are sorted by place-field peak on a 1-D track.
HIPPOCAMPAL = re.compile(r"hippocamp|ca1|ca3|(?<![a-z])dg(?![a-z])", re.IGNORECASE)

#: The raster's unit axis when its rows are sorted by place-field peak.
SORTED_LABEL = "units, sorted by place-field peak"


def is_hippocampal(rec: Recording) -> bool:
    """Whether the recording's brain regions name the hippocampus.

    True when ``rec.brain_regions`` or a unit's ``brain_region`` mentions
    the hippocampus, CA1, CA3 or DG, in any case (``"lCA1"`` counts).
    """
    regions = [*rec.brain_regions, *rec.units["brain_region"].dropna().astype(str)]
    return any(HIPPOCAMPAL.search(region) for region in regions)


def place_field_order(rec: Recording) -> NDArray[np.intp] | None:
    """Raster rows by each unit's rate-map peak along a 1-D track.

    Only for hippocampal recordings (:func:`is_hippocampal`) on a 1-D
    environment (a track, linearized graph or 1-D grid); None otherwise,
    and the raster keeps the file's order.

    The rate maps are :func:`~neurospatial.compute_spatial_rates` on
    ``rec.env`` with its defaults and ``params["min_speed"]`` where set. A
    unit's peak is the bin of its highest finite rate, at that bin's
    distance along the track; a unit with no finite rate above zero has no
    peak and goes last, in file order (the sort is stable).

    Returns
    -------
    ndarray of intp, shape (n_units,) or None
        Indices into ``rec.spike_times``, first row first.
    """
    env = rec.env
    if (
        env is None
        or rec.positions is None
        or rec.n_units < 2
        or not (env.is_linearized_track or env.n_dims == 1)
        or not is_hippocampal(rec)
    ):
        return None
    import warnings

    from neurospatial import compute_spatial_rates
    from neurospatial.environment import _track_extents

    with warnings.catch_warnings():
        # Sorting rows does not rely on what these warn about (dropped
        # samples, silent units); the analyses that do warn on their own.
        warnings.simplefilter("ignore")
        rates = np.asarray(
            compute_spatial_rates(
                env,
                rec.spike_times,
                rec.times,
                rec.positions,
                min_speed=_min_speed(rec),
            ).firing_rates,
            dtype=np.float64,
        )
    extents = _track_extents(env)
    assert extents is not None
    along = (extents[0] + extents[1]) / 2
    finite = np.where(np.isfinite(rates), rates, -np.inf)
    peak_bin = np.argmax(finite, axis=1)
    has_peak = finite[np.arange(len(rates)), peak_bin] > 0
    peaks = np.where(has_peak, along[peak_bin], np.inf)
    return np.argsort(peaks, kind="stable").astype(np.intp)


def _panel_time(rec: Recording, ax: Axes) -> None:
    title = f"First {WINDOW_SECONDS:g} s of tracking with spikes"
    ax.set_title(title)
    if rec.positions is None:
        _missing(ax, title, "no positions")
        return
    start, stop = _default_time_range(rec)
    ax.axis("off")
    top = ax.inset_axes((0.0, 0.55, 1.0, 0.4))
    bottom = ax.inset_axes((0.0, 0.0, 1.0, 0.5), sharex=top)
    mask = (rec.times >= start) & (rec.times < stop)
    for dim, name in enumerate("xyz"[: rec.positions.shape[1]]):
        top.plot(rec.times[mask], rec.positions[mask, dim], lw=0.8, label=name)
    top.set_ylabel(rec.position_units or "", fontsize=8)
    top.legend(frameon=False, fontsize=7, ncols=3, loc="upper right")
    top.tick_params(labelbottom=False)
    order = rec._raster_order
    rows = np.arange(rec.n_units) if order is None else order
    for row, unit in enumerate(rows):
        train = rec.spike_times[unit]
        inside = train[(train >= start) & (train < stop)]
        bottom.vlines(inside, row - 0.4, row + 0.4, color="k", lw=0.5)
    bottom.set_xlim(start, stop)
    # Every unit's row, those silent in the window too.
    bottom.set_ylim(-0.5, rec.n_units - 0.5)
    _label_from_session_start(rec, bottom, start, stop)
    bottom.set_ylabel("unit" if order is None else SORTED_LABEL, fontsize=8)


def _label_from_session_start(
    rec: Recording, ax: Axes, start: float, stop: float
) -> None:
    """Tick a time axis in seconds from the first sample; the data stay as stored.

    Artists keep the recording's clock (a Unix clock would otherwise print as
    an offset such as "+1.625935e9"); only the tick labels count from
    ``rec.times[0]``, which the axis label gives.
    """
    from matplotlib.ticker import MaxNLocator

    origin = float(rec.times[0])
    relative = np.asarray(
        MaxNLocator(nbins=6).tick_values(start - origin, stop - origin), dtype=float
    )
    relative = relative[(relative >= start - origin) & (relative <= stop - origin)]
    ax.set_xticks(origin + relative, [f"{value:g}" for value in np.round(relative, 9)])
    ax.set_xlabel(f"time from session start (s); start = {origin:.3f} s")


def plot_overview(rec: Recording, fig: Figure | None = None) -> Figure:
    """Draw the six-panel overview of a session.

    Parameters
    ----------
    rec : Recording
        The session.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into, which should be empty; a new one when None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. A panel whose data the dataset does not have says "not in
        this dataset" instead of drawing empty axes.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    panels: tuple[tuple[Callable[[Recording, Axes], None], dict[str, str]], ...] = (
        (_panel_environment, {}),
        (_panel_occupancy, {}),
        (_panel_speed, {}),
        (_panel_heading, {"projection": "polar"}),
        (_panel_units, {}),
        (_panel_time, {}),
    )
    for index, (draw, options) in enumerate(panels):
        draw(rec, fig.add_subplot(grid[divmod(index, 3)], **options))
    fig.suptitle(f"{rec.dataset} / {rec.session}")
    return fig


def _default_time_range(rec: Recording) -> tuple[float, float]:
    """The first ``WINDOW_SECONDS`` where tracking, spikes and analysis overlap.

    It starts at the later of the first tracked sample and the first spike
    of any unit, so a session whose neural recording starts after its
    tracking (as in ``forli_2025``) opens on spikes; then at the first
    analysed epoch (``params["analysis_epochs"]``, the bat's flights) at or
    after that time, so it opens on the behaviour the dataset analyses
    rather than on, say, a bat resting on its perch. Without spikes, or when
    they start after the last tracked sample, the first rule starts at the
    first tracked sample; with no analysed epoch after that time, the second
    rule is skipped.
    """
    if rec.positions is not None:
        tracked = rec.times[tracked_mask(rec.positions)]
    else:
        tracked = rec.times
    if not tracked.size:
        tracked = rec.times
    start = float(tracked[0])
    first_spike = min((float(t[0]) for t in rec.spike_times if t.size), default=None)
    if first_spike is not None and start < first_spike < tracked[-1]:
        start = first_spike
    rows = rec._analysis_epochs
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    ahead = rows[(rows[:, 1] > start) & (rows[:, 0] < tracked[-1])]
    if len(ahead):
        start = max(start, float(ahead[0, 0]))
    return start, start + WINDOW_SECONDS


def animation(
    rec: Recording,
    *,
    time_range: tuple[float, float] | None = None,
    trail_length: int | None = None,
    video: bool = False,
) -> Animation:
    """Build the interactive or movie scene of a session.

    Parameters
    ----------
    rec : Recording
        The session.
    time_range : (float, float), optional
        Seconds [start, stop) to show, on the recording's clock. Default: the
        first 60 s where tracking and spikes overlap, from the later of the
        first tracked sample and the first spike, moved to the first analysed
        epoch (``params["analysis_epochs"]``) at or after it. The scene has one frame per tracking sample,
        or 30 frames a second when tracking is faster; the overlays keep their
        own samples.
    trail_length : int, optional
        Length of the path trail behind the animal, in frames.
    video : bool, default False
        Draw ``rec.video`` below the data, placed by its calibration and
        timed by its frame times; occupancy is then at most half opaque
        (``VIDEO_FIELD_OPACITY``) and see-through where low.

    Returns
    -------
    Animation
        Occupancy of the whole session over the environment, with the path,
        headings, spike raster, events and epochs of the window on top.

    Raises
    ------
    InputError
        If the session has no positions, or ``time_range`` holds no samples;
        with ``video=True``, if the session has no video or no calibration.
    """
    from neurospatial.animation import (
        Animation,
        AnimationField,
        EpochOverlay,
        EventOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
        SpikeRasterOverlay,
        VideoOverlay,
    )

    if rec.env is None or rec.positions is None:
        have = [
            name
            for name, value in (
                ("headings", rec.headings),
                ("spike_times", rec.spike_times),
            )
            if value is not None and len(value)
        ]
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session} has no positions, so there is no "
                "environment to animate.",
                why="the scene draws the animal's path over the environment.",
                fix="use what this session has: "
                + " and ".join(f"rec.{h}" for h in have)
                + " (for example rec.plot_overview() for the panels it can draw).",
            )
        )
    start, stop = (
        _default_time_range(rec) if time_range is None else map(float, time_range)
    )
    window = (rec.times >= start) & (rec.times < stop)
    if window.sum() < 2:
        raise InputError(
            _format_error(
                f"time_range ({start:g}, {stop:g}) holds {int(window.sum())} "
                f"tracking samples of {rec.times[0]:g} to {rec.times[-1]:g} s.",
                why="a scene needs at least two samples.",
                fix="pass a time_range inside the recording, in seconds.",
            )
        )
    frame_times = rec.times[window]
    if np.median(np.diff(frame_times)) < 1.0 / MAX_FRAME_RATE:
        # Tracking faster than a viewer can play: draw 30 frames a second,
        # through the window's last sample (and never at or past its stop), so
        # the clock ends within one frame of the window and export(time_range=)
        # accepts the window itself.
        step = 1.0 / MAX_FRAME_RATE
        end = min(stop, float(frame_times[-1]) + step)
        frame_times = np.arange(frame_times[0], end, step)
        frame_times = frame_times[frame_times < stop]
    if len(frame_times) < 2:
        raise InputError(
            _format_error(
                f"time_range ({start:g}, {stop:g}) gives {len(frame_times)} frame "
                f"at {MAX_FRAME_RATE:g} frames a second.",
                why="a scene needs at least two frames.",
                fix=f"pass a time_range at least {2 / MAX_FRAME_RATE:.3f} s long "
                "inside the recording.",
            )
        )
    # A field row is held until the next row, so a static field is two equal
    # rows at the scene's first and last frame.
    # A track's bins are drawn along it; camera frames need the plane, so over
    # a video a track session is drawn on a grid over the track's footprint,
    # with the track as its along-track panel.
    env, track = rec.env, None
    if video and rec.env.is_linearized_track:
        env, track = _footprint_grid(rec), rec.env
    occupancy = occupancy_field(rec, env)
    field = AnimationField(
        np.tile(occupancy, (2, 1)),
        frame_times[[0, -1]],
        max_gap=None,
        label="occupancy (s)",
        max_opacity=VIDEO_FIELD_OPACITY if video else 1.0,
    )
    # Fetched once the window is known to hold a scene.
    overlays: list[Any] = [_video_overlay(rec, VideoOverlay)] if video else []
    overlays += [
        PositionOverlay(
            rec.positions,
            rec.times,
            trail_length=trail_length,
            units=rec.position_units,
        )
    ]
    if rec.headings is not None:
        extent = max(hi - lo for lo, hi in env.dimension_ranges or ((0.0, 1.0),))
        overlays.append(
            HeadDirectionOverlay(rec.headings, rec.times, length=0.05 * extent)
        )
    if rec.n_units:
        overlays.append(
            SpikeRasterOverlay(
                rec.spike_times,
                order=rec._raster_order,
                unit_ids=rec.units.index.to_numpy(),
            )
        )
    points = {k: v for k, v in rec.events.items() if v.ndim == 1 and v.size}
    if points:
        overlays.append(EventOverlay.from_trajectory(points, rec.positions, rec.times))
    intervals = {
        **{k: v for k, v in rec.epochs.items() if len(v)},
        **{f"event: {k}": v for k, v in rec.events.items() if v.ndim == 2 and len(v)},
    }
    # Only intervals the scene's window meets get a row, so their names fit.
    intervals = {
        name: rows
        for name, rows in intervals.items()
        if np.any((rows[:, 0] < stop) & (rows[:, 1] > start))
    }
    for index, (name, rows) in enumerate(intervals.items()):
        overlays.append(
            EpochOverlay(
                rows, label=name, color=_EPOCH_COLORS[index % len(_EPOCH_COLORS)]
            )
        )
    return Animation(
        env,
        {"occupancy": field},
        frame_times=frame_times,
        overlays=overlays,
        title=f"{rec.dataset} / {rec.session}",
        show_regions=bool(env.regions) and env.n_dims == 2,
        track=track,
    )


def _footprint_grid(rec: Recording) -> Environment:
    """A 2-D grid over the box enclosing a track's footprint, its bin size.

    The bin size is ``params["bin_size"]``, else the track's median bin
    length; units and frame are the track's.
    """
    import shapely

    from neurospatial import Environment

    assert rec.env is not None
    bounds = rec.env.arena.domain_bounds()
    if bounds is None:
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session}: the track has no width, so there "
                "is no plane to draw its video in.",
                why="camera frames are placed in 2-D; a track's bins lie along it.",
                fix="build the track with track_width=, or call rec.animation() "
                "without video=True.",
            )
        )
    bin_size = rec.params.get("bin_size") or float(np.median(rec.env.bin_sizes))
    return Environment.from_polygon(
        shapely.box(*bounds[0], *bounds[1]),
        bin_size=float(bin_size),
        units=rec.env.units,
        frame=rec.env.frame,
        name=f"{rec.env.name} footprint",
    )


def _video_overlay(rec: Recording, overlay: type[Any]) -> Any:
    """``rec.video`` as a ``VideoOverlay`` below the field, fetched once.

    Refuses a session without a video or without a calibration: a frame
    stretched over the environment's extent would be placed wrong.
    """
    if rec.video is None:
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session} has no video to draw.",
                why="rec.video is None: the dataset publishes no camera video "
                "for this session that the loader reads (see rec.watch_outs).",
                fix="call rec.animation() without video=True.",
            )
        )
    if rec.video.calibration is None:
        tried = "\n".join(f"  - {line}" for line in rec.video.calibration_tried)
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session}: no calibration maps the video's "
                f"pixels to the environment. Sources tried:\n{tried}",
                why="without one the frames cannot be placed under the "
                "positions; stretching them over the environment's extent "
                "would put the camera's view in the wrong place.",
                fix="call rec.animation() without video=True; to draw the video, "
                "measure landmarks on a frame (rec.video.fetch() gives the "
                "file) and pass a VideoOverlay with calibrate_video(env, path, "
                "landmarks_px=..., landmarks_env=...) to your own Animation.",
            )
        )
    from neurospatial._optional import require_optional

    require_optional("cv2", extra="video", purpose="drawing a session's video")
    path = rec.video.fetch()
    return overlay(
        path,
        calibration=rec.video.calibration,
        times=rec.video.frame_times,
        alpha=1.0,
        z_order="below",
    )
