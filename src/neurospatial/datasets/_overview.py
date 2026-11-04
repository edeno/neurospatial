"""The overview figure and the interactive scene of a loaded session."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from neurospatial.animation import Animation
    from neurospatial.datasets._recording import Recording

__all__: list[str] = []

#: Seconds of tracking the overview's time panel and the animation open on.
WINDOW_SECONDS = 60.0

#: Frames per second of the scene's clock for faster tracking (viewers play at most 60).
MAX_FRAME_RATE = 30.0

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


def _speed(rec: Recording) -> NDArray[np.float64]:
    from neurospatial.behavior import compute_speed

    assert rec.positions is not None
    return np.asarray(compute_speed(rec.times, rec.positions), dtype=float)


def _wrap(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def occupancy_field(rec: Recording) -> NDArray[np.float64]:
    """Seconds in each bin over the whole session, NaN where never visited.

    Parameters
    ----------
    rec : Recording
        A session with positions and an environment.

    Returns
    -------
    ndarray, shape (n_bins,)
        ``rec.env.occupancy(rec.times, rec.positions)`` with unvisited bins
        (zero seconds) set to NaN, so they draw as missing rather than as zero.
    """
    assert rec.env is not None
    assert rec.positions is not None
    occupancy: NDArray[np.float64] = rec.env.occupancy(rec.times, rec.positions)
    return np.where(occupancy > 0, occupancy, np.nan)


def _panel_environment(rec: Recording, ax: Axes) -> None:
    title = "Environment and path"
    if rec.env is None or rec.positions is None:
        _missing(ax, title, "no positions")
        return
    if rec.env.n_dims == 3:
        _note(ax, title, "3-D environment:\nopen rec.animation()")
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
        ax.text(
            0.5,
            0.97,
            "path not drawn for 1-D positions",
            ha="center",
            va="top",
            transform=ax.transAxes,
            color="0.35",
        )
    ax.set_title(title)


def _panel_occupancy(rec: Recording, ax: Axes) -> None:
    if rec.env is None or rec.positions is None:
        _missing(ax, "Occupancy", "no positions")
        return
    if rec.env.n_dims == 3:
        _note(ax, "Occupancy", "3-D environment:\nopen rec.animation()")
        return
    rec.env.plot_field(
        occupancy_field(rec), ax=ax, colorbar_label="s", nan_color="lightgray"
    )
    ax.set_title("Occupancy (gray: never visited)")


def _panel_speed(rec: Recording, ax: Axes) -> None:
    if rec.positions is None:
        _missing(ax, "Speed", "no positions")
        return
    speed = _speed(rec)
    speed = speed[np.isfinite(speed)]
    ax.hist(speed, bins=60, color="0.4")
    min_speed = _min_speed(rec)
    if min_speed is not None:
        ax.axvline(min_speed, color="tab:red", label=f"min_speed {min_speed:g}")
        ax.legend(frameon=False)
    ax.set_xlabel(f"speed ({rec.position_units}/s)")
    ax.set_ylabel("samples")
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
    offset = _wrap(rec.headings - travel)
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


def _panel_time(rec: Recording, ax: Axes) -> None:
    ax.set_title(f"First {WINDOW_SECONDS:g} s of tracking")
    if rec.positions is None:
        _missing(ax, f"First {WINDOW_SECONDS:g} s of tracking", "no positions")
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
    for row, train in enumerate(rec.spike_times):
        inside = train[(train >= start) & (train < stop)]
        bottom.vlines(inside, row - 0.4, row + 0.4, color="k", lw=0.5)
    bottom.set_xlim(start, stop)
    bottom.set_xlabel("time (s)")
    bottom.set_ylabel("unit", fontsize=8)


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
    """The first ``WINDOW_SECONDS`` of tracking, from the first tracked sample."""
    if rec.positions is not None:
        tracked = np.all(np.isfinite(rec.positions), axis=1)
        start = float(rec.times[tracked][0]) if tracked.any() else float(rec.times[0])
    else:
        start = float(rec.times[0])
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
        first 60 s of tracking. The scene has one frame per tracking sample,
        or 30 frames a second when tracking is faster; the overlays keep their
        own samples.
    trail_length : int, optional
        Length of the path trail behind the animal, in frames.
    video : bool, default False
        Draw ``rec.video`` below the data, placed by its calibration and
        timed by its frame times; occupancy is then see-through where low.

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
    camera = _video_overlay(rec, VideoOverlay) if video else None
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
        # Tracking faster than a viewer can play: draw 30 frames a second.
        frame_times = np.arange(frame_times[0], frame_times[-1], 1.0 / MAX_FRAME_RATE)
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
    occupancy = occupancy_field(rec)
    field = AnimationField(
        np.tile(occupancy, (2, 1)),
        frame_times[[0, -1]],
        max_gap=None,
        label="occupancy (s)",
    )
    overlays: list[Any] = [] if camera is None else [camera]
    overlays += [
        PositionOverlay(
            rec.positions,
            rec.times,
            trail_length=trail_length,
            units=rec.position_units,
        )
    ]
    if rec.headings is not None:
        extent = max(hi - lo for lo, hi in rec.env.dimension_ranges or ((0.0, 1.0),))
        overlays.append(
            HeadDirectionOverlay(rec.headings, rec.times, length=0.05 * extent)
        )
    if rec.n_units:
        overlays.append(
            SpikeRasterOverlay(rec.spike_times, unit_ids=rec.units.index.to_numpy())
        )
    points = {k: v for k, v in rec.events.items() if v.ndim == 1 and v.size}
    if points:
        overlays.append(EventOverlay.from_trajectory(points, rec.positions, rec.times))
    intervals = {
        **{k: v for k, v in rec.epochs.items() if len(v)},
        **{f"event: {k}": v for k, v in rec.events.items() if v.ndim == 2 and len(v)},
    }
    for index, (name, rows) in enumerate(intervals.items()):
        overlays.append(
            EpochOverlay(
                rows, label=name, color=_EPOCH_COLORS[index % len(_EPOCH_COLORS)]
            )
        )
    return Animation(
        rec.env,
        {"occupancy": field},
        frame_times=frame_times,
        overlays=overlays,
        title=f"{rec.dataset} / {rec.session}",
        show_regions=bool(rec.env.regions) and rec.env.n_dims == 2,
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
