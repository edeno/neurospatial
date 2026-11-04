"""Valero et al. (2022): mouse CA1 on a linear track, filmed from above (DANDI 000568).

What was recorded: CA1 units of CaMKIIa-Cre x Ai32 mice (channelrhodopsin in
CaMKIIa neurons) probed with optogenetic pulses, over sessions of five epochs
on one clock: one epoch on a PVC linear track "110 cm long, 6.35 cm wide"
(the description of each file's position series), the others off it. Position
is one tracked point from an overhead Basler camera, in cm, every 1/30 s.
The files also hold ripples, sleep states (wake, NREM, REM) and up/down
states.

Sessions are DANDI 000568's (version 0.230705.1633, CC-BY-4.0) files without
the raw voltage (``-no-raw-data``, 5 to 53 MB; the copies with it are 1 to
54 GB), named without that suffix. Each file's video, an ``.avi`` beside it on
DANDI, is ``rec.video``: fetched only on request (46 MB for the default). The
default session, ``sub-fCamk2_ses-fCamk2-201014-sess3``, is the one the
survey of public datasets loaded: 35 units, the track from 7,674 to 9,941 s.

Sources read: DANDI 000568's metadata, the files, the paper's Crossref
record (title, authors, volume, pages) and the paper's main text (the author
manuscript, PubMed Central PMC9632609). Its Materials and methods, in the
Supplementary Materials, were not available to read, so no number here
comes from its Methods; the numbers quoted from the main text say so.

What the loader converts:

- positions: the file's ``SpatialSeriesRaw`` as stored, in cm (its unit).
  The head is tracked, so samples spread beyond the 6.35 cm track width.
- times: the position timestamps. The video ``ImageSeries`` holds the same
  frames (as many as position samples) on a clock 22.55 to 23.12 s earlier;
  the overlay follows the position clock, since the spikes are on it (an
  inferred ``Conversion`` and a watch-out).
- epochs: the file's epochs, those holding the tracking as ``track`` (one
  row, or two in sessions that run the track twice) and the others as
  ``epoch_<k>``, their place in the file's table (the ``Conversion`` names
  each one's ``epoch_name``); the stretches of each track epoch without
  positions, before its first and after its last sample, as
  ``track_before_tracking`` and ``track_after_tracking``; ``SleepStates`` by
  label as ``wake``, ``nrem`` and ``rem``.
- events: ``ripples``, the ``RippleTimeIntervals`` rows, [start, stop) in s.
- ``units["brain_region"]``: CA1, the location of every electrode in the
  file; the units table names none.
- headings: none. One point is tracked.

The declared environment is the stated 110 cm track as a one-edge track
graph along the tracked samples' principal axis, centred between their
0.5th and 99.5th percentiles along it (the file states no origin: its
reference frame is "Arbitrary, camera"), in 2.5 cm bins. Its ``track_width``
is a placement tolerance measured from the samples, not the stated width.

The file states no pixel size and stores no pixel tracking, so the
default session's video is calibrated by a robust affine from the LED
detected in the video to the file's positions (a Huber fit with a 2 cm cut,
then least squares over the detections within 5 cm of it;
``scripts/fit_video_led_calibration.py``, run once, stored in the
registry). The overlay is then aligned with the tracking by construction: it
shows the camera's view of each moment, not a check of the tracking or of
the declared track, which the dark video does not show. Other sessions
refuse ``rec.animation(video=True)`` until the fit is run for them. The
scene of the first 20 s over the video is drawn on a grid over the track's
footprint (``scene.env``), with the track as its along-track panel:

>>> from neurospatial.datasets import load
>>> rec = load("valero_2022")  # doctest: +SKIP
>>> start = rec.times[0]  # doctest: +SKIP
>>> scene = rec.animation(time_range=(start, start + 20), video=True)  # doctest: +SKIP
>>> scene.export("valero_video.mp4", fps=30)  # doctest: +SKIP

Your own decoded posterior goes into the same scene as a field over
``scene.env``'s bins, with its overlays and track. That means decoding on
``scene.env``, a 2-D grid over the box around the track, not on the track
graph ``rec.env``: a posterior over the track's bins lies along the track
and cannot be drawn over the video's 2-D frames (draw it with
``rec.animation()`` instead, without the video):

>>> from neurospatial.animation import Animation, AnimationField
>>> mine = Animation(  # doctest: +SKIP
...     scene.env,
...     {"posterior": AnimationField(result.posterior, result.times)},
...     frame_times=scene.frame_times,
...     overlays=scene.overlays,
...     track=scene.track,
... )

Analysis choices (``params``):

- ``bin_size`` 2.5 cm, the loader's choice (no source read states one);
- ``min_speed`` None: no source read states a running threshold;
- ``dt`` 0.25 s, the decoding bin, and ``train_epochs`` / ``test_epochs``,
  alternating 60 s blocks of the tracked span.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    interval_rows,
    nwb_unit_table,
    tracked_mask,
)
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry
    from neurospatial.datasets._video import VideoSource

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": "most CA1 pyramidal neurons on the track are place cells",
    "quote": (
        '"With a standard definition of \u201cplace field\u201d (materials and '
        "methods) (18), more than half of the pyramidal neurons were "
        "classified as place cells [73% and 71% of light-responsive and "
        'nonresponsive neurons, respectively ...]"'
    ),
    "source": (
        "Valero et al. 2022, Science 375, 570-574, main text (author "
        "manuscript, PMC9632609), the paragraph on place cells; fig. S11"
    ),
    "paper": "73% and 71% of pyramidal neurons",
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": None,
    "why": (
        "Three things are missing. The place-field definition is in the "
        "paper's Materials and methods, in the Supplementary Materials, "
        "which were not read. The share is of pyramidal neurons, and the "
        "file's units table holds no cell type (clu_id and unit_name only). "
        'And each track session interleaves "three blocks of 10 baseline '
        'runs" with "two blocks of 40 to 50 stimulation runs" (main text), '
        "where light unmasks place fields in non-place cells, the paper's "
        "own finding; the default file holds no stimulation times (its "
        "stimulus group is empty, as in 26 of the 28 files; two store "
        "optogenetic series that are not loaded), so its baseline runs "
        "cannot be told apart."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

#: The stated track, cm.
LENGTH, WIDTH = 110.0, 6.35
BIN_SIZE = 2.5
DT = 0.25
BLOCK = 60.0
#: The track is placed between these quantiles of the samples along it.
END_QUANTILES = (0.005, 0.995)
#: The placement tolerance is twice this quantile of the samples' distance
#: from the track, and at least the stated width and one bin.
TOLERANCE_QUANTILE = 0.99
_QUANTILE_REASON = (
    "The 99th percentile, so that 1% of the samples are left out: the 95th "
    "leaves out 5%, head samples just beside the track."
)

_BEHAVIOR = "processing/behavior/LinearMazePositionTracking"
_POSITION = f"{_BEHAVIOR}/SpatialSeriesRaw"
_VIDEO = "ImageSeriesTrackingVideo1"
_STATES = {"WAKEstate": "wake", "NREMstate": "nrem", "REMstate": "rem"}

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-fCamk2_ses-fCamk2-201014-sess3": (
        r"runs of 11 or more identical position samples",
        r"lie outside the declared environment",
        r"lie farther than half the stated",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-fCamk2_ses-fCamk2-201014-sess3": (
        "The file's session_start_time, 2016-03-15, is not the recording date: "
        "the epoch names (fCamk2_201014_091141 to fCamk2_201014_122743) give "
        "2020-10-14. Times are seconds from the first epoch's start.",
        "The track epoch starts 22.6 s before the first position sample "
        "(7,674.30 s against 7,696.89 s), so the first 22.6 s of the epoch "
        "has spikes and no positions (epochs['track_before_tracking']); the "
        "file does not say why.",
        "The spikes are on the position clock, not the video's: the 35 units' "
        "spatial information (compute_spatial_rates with method='binned' on "
        "this environment, the whole session) averages 0.84 bits/spike at the "
        "position timestamps, 0.36 with the positions moved 22.84 s earlier "
        "onto the video's clock, and 0.30 to 0.43 with them moved 10 or 40 s "
        "either way (checked on 2026-11-03).",
        "The file's SleepStates score NREM during parts of the track epoch "
        "(for example 9,723 to 9,939 s, while the mouse runs along the track "
        "at 9,914 s): select running by speed, not by epochs['wake'].",
    ),
}

#: Added to every session's watch-outs, not only the ones checked by hand.
_STIMULATION = (
    'Each track session interleaves "three blocks of 10 baseline runs" with '
    '"two blocks of 40 to 50 stimulation runs" (main text of Valero et al. '
    "2022), and light unmasks place fields in non-place cells, the paper's own "
    "finding. Place fields and rate maps computed on the whole track epoch "
    "mix baseline and stimulation runs. "
)


def _stimulation_watch_out(stimulus_names: list[str]) -> str:
    """The baseline-versus-stimulation warning, by what the file's stimulus group holds."""
    if not stimulus_names:
        return (
            _STIMULATION
            + "The file's stimulus group is empty (as in 26 of the 28 files), "
            "so it holds no stimulation times and the blocks cannot be separated."
        )
    return (
        _STIMULATION
        + f"The file's stimulus group holds {len(stimulus_names)} optogenetic "
        f"series ({', '.join(stimulus_names)}), which are not loaded; read "
        "them with pynwb to look for the stimulation runs. Nothing here "
        "separates the blocks."
    )


_UNLOADED = (
    "The file also holds UpDownStatesTimeIntervals (up and down states) and "
    "the ripples' peak times, which are not loaded; read them with pynwb."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 000568 session into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file, or its URL to read by byte range.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the position series is not in cm, or no epoch holds the tracking.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_position, read_units

    source = os.fspath(path_or_stream)
    with open_nwb(source) as nwb:
        position = read_position(nwb, position_name=_POSITION)
        units = read_units(nwb)
        stored = nwb.processing["behavior"]["LinearMazePositionTracking"]
        series = stored.spatial_series["SpatialSeriesRaw"]
        stated_unit = f"unit {series.unit!r}, conversion {series.conversion:g}"
        description, frame = series.description, series.reference_frame
        epochs_table = nwb.epochs.to_dataframe()
        states = _table(nwb, "behavior", "SleepStates")
        ripples = _table(nwb, "ecephys", "RippleTimeIntervals")
        regions = nwb.electrodes.to_dataframe()["location"].dropna().unique()
        video = nwb.acquisition.get(_VIDEO)
        stimulus_names = sorted(nwb.stimulus)
        video_files = [] if video is None else [str(f) for f in video.external_file[:]]
        video_times = (
            None if video is None else np.asarray(video.timestamps[:], np.float64)
        )
        subject = nwb.subject.subject_id
    if position.units != "cm":
        msg = f"{entry.session}: {_POSITION} is in {position.units!r}, not cm"
        raise ValueError(msg)
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64)
    epochs, epoch_names = _epochs(entry.session, times, epochs_table, states)
    env, placement, distance, width_quantile = _declared_track(
        positions, entry.session, frame
    )
    # Units name no electrode: a region holds for every unit only when every
    # electrode is in it.
    unit_table, _ = nwb_unit_table(
        units, region=regions[0] if len(regions) == 1 else None
    )
    camera, video_lines = _video(entry, source, times, video_files, video_times)
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"mouse {subject}",
        apparatus=entry.info.apparatus,
        brain_regions=tuple(regions) or entry.info.brain_regions,
        clock=(
            "the acquisition clock, seconds from the first epoch's start (the "
            "file's session_start_time is not the recording date)"
        ),
        times=times,
        spike_times=tuple(units.spike_times),
        units=unit_table,
        env=env,
        env_kind="declared",
        track_width_quantile=width_quantile,
        env_source=(
            f"the stated {LENGTH:g} cm x {WIDTH:g} cm PVC linear track (the "
            f"position series' description), a one-edge track graph, "
            f"{BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "one point from an overhead camera (the file does not say which part "
            "of the head or headstage)"
        ),
        epochs=epochs,
        events={} if ripples is None else {"ripples": interval_rows(ripples)},
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": None,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                f"positions: {_POSITION} as stored, cm",
                "stated",
                f"the file's series: {stated_unit} (read_position applies the "
                f"conversion); {description!r}",
            ),
            Conversion(
                "epochs: intervals/epochs, those holding the tracking as 'track' "
                "and the others as 'epoch_<k>', their place in the table "
                f"({epoch_names}); the stretches of each track epoch before its "
                "first and after its last position sample as "
                "'track_before_tracking' and 'track_after_tracking'; "
                "processing/behavior/SleepStates by label as 'wake', 'nrem' "
                "and 'rem'",
                "stated",
                "the file's epochs table, position timestamps and SleepStates labels",
            ),
            Conversion(
                "units['brain_region']: the location of every electrode in the "
                "file, for every unit (missing where the electrodes are in more "
                "than one region)",
                "inferred",
                "the electrodes table gives the locations "
                f"{', '.join(regions) or 'none'}; the units table names no "
                "electrode or location",
            ),
            *placement,
            *video_lines[0],
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *video_lines[1],
            _stimulation_watch_out(stimulus_names),
            _beyond_track(distance),
            _UNLOADED,
        ),
        video=camera,
    )


def _table(nwb: Any, module: str, name: str) -> Any:
    """``processing/<module>/<name>`` as a DataFrame, or None where absent."""
    if (
        module not in nwb.processing
        or name not in nwb.processing[module].data_interfaces
    ):
        return None
    return nwb.processing[module][name].to_dataframe()


def _epochs(
    session: str, times: NDArray[np.float64], table: Any, states: Any
) -> tuple[dict[str, NDArray[np.float64]], str]:
    """The file's epochs, the tracked ones as ``track``, and sleep states.

    Epochs holding position samples are the rows of ``track`` (one or two:
    some sessions run the track twice, with sleep between). Each other
    epoch is ``epoch_<k>``, its place in the file's epochs table, from 1.
    The stretches of each track epoch before its first and after its last
    position sample, where it has no positions, are the rows of
    ``track_before_tracking`` and ``track_after_tracking``.

    Returns
    -------
    epochs : dict of str to ndarray, shape (n, 2)
    names : str
        Which file ``epoch_name`` each key is, for the conversion.
    """
    rows = interval_rows(table)
    names = table.sort_values("start_time", kind="stable")["epoch_name"].astype(str)
    first = np.searchsorted(times, rows[:, 0], side="left")
    stop = np.searchsorted(times, rows[:, 1], side="left")
    holds = stop > first
    if not holds.any():
        msg = (
            f"{session}: no epoch holds the tracking from {times[0]:.6g} to "
            f"{times[-1]:.6g} s"
        )
        raise ValueError(msg)
    if int((stop - first)[holds].sum()) != len(times):
        msg = f"{session}: some position samples lie in no epoch"
        raise ValueError(msg)
    epochs: dict[str, NDArray[np.float64]] = {}
    labels = []
    for index, (name, row, held) in enumerate(zip(names, rows, holds, strict=True)):
        key = "track" if held else f"epoch_{index + 1}"
        labels.append(f"{key} = {name}")
        if not held:
            epochs[key] = row[None, :]
    epochs["track"] = rows[holds]
    before = [
        [row[0], times[k]]
        for row, k in zip(rows[holds], first[holds], strict=True)
        if times[k] > row[0]
    ]
    after = [
        [times[k - 1], row[1]]
        for row, k in zip(rows[holds], stop[holds], strict=True)
        if times[k - 1] < row[1]
    ]
    if before:
        epochs["track_before_tracking"] = np.array(before)
    if after:
        epochs["track_after_tracking"] = np.array(after)
    for label, key in _STATES.items():
        if states is not None:
            epochs[key] = interval_rows(states[states["label"] == label])
    return epochs, ", ".join(labels)


def _declared_track(
    positions: NDArray[np.float64], session: str, frame: str | None
) -> tuple[
    Environment, tuple[Conversion, Conversion], NDArray[np.float64], float | None
]:
    """The stated track along the samples' principal axis, and how it was placed.

    Returns
    -------
    env : Environment
    conversions : (Conversion, Conversion)
        The placement and the placement tolerance.
    distance : ndarray, shape (n_tracked,)
        Each tracked sample's distance from the track, cm.
    width_quantile : float or None
        ``TOLERANCE_QUANTILE`` when it set ``track_width``; None when the
        stated width or one bin floored it.
    """
    import networkx as nx

    from neurospatial import Environment

    tracked = positions[tracked_mask(positions)]
    centre = tracked.mean(axis=0)
    _, _, axes = np.linalg.svd(tracked - centre, full_matrices=False)
    axis = axes[0] * np.sign(axes[0][np.argmax(np.abs(axes[0]))])
    along = (tracked - centre) @ axis
    low, high = np.quantile(along, END_QUANTILES)
    middle = centre + axis * (low + high) / 2
    ends = middle + np.outer([-LENGTH / 2, LENGTH / 2], axis)
    graph = nx.Graph()
    for name, point in zip(("start", "end"), ends, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    graph.add_edge("start", "end", distance=LENGTH)
    distance = nearest_segment(tracked, ends[:1], ends[1:]).distance
    width, width_quantile, evidence = placement_tolerance(
        distance,
        TOLERANCE_QUANTILE,
        floor=max(WIDTH, BIN_SIZE),
        measured="the tracked point's distance from the track",
        reason=_QUANTILE_REASON,
    )
    env = Environment.from_graph(
        graph,
        [("start", "end")],
        bin_size=BIN_SIZE,
        edge_spacing=0.0,
        units="cm",
        name=f"valero_2022 {session}",
        track_width=width,
    )
    span = float(high - low)
    beyond = float(np.mean(np.abs(along - (low + high) / 2) > LENGTH / 2))
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    placement = Conversion(
        f"environment: the stated {LENGTH:g} cm track from ({ends[0][0]:.4g}, "
        f"{ends[0][1]:.4g}) to ({ends[1][0]:.4g}, {ends[1][1]:.4g}) cm, along the "
        f"tracked samples' principal axis at {angle:.1f} degrees, centred between "
        "their 0.5th and 99.5th percentiles along it",
        "inferred",
        f"the file states no origin or rotation (reference frame {frame!r}); the "
        f"samples span {span:.1f} cm between those percentiles, "
        f"{span / LENGTH - 1:+.1%} of the stated length, and {beyond:.1%} of them "
        "lie beyond the track's ends",
    )
    tolerance = Conversion(
        f"environment: track_width {width:.3g} cm, a placement tolerance, not the "
        f"stated {WIDTH:g} cm width: twice the tracked samples' "
        f"{TOLERANCE_QUANTILE:.0%} quantile distance from the track",
        "inferred",
        evidence,
    )
    return env, (placement, tolerance), distance, width_quantile


def _beyond_track(distance: NDArray[np.float64]) -> str:
    """The share of tracked samples beyond half the stated width of the track."""
    beyond = distance > WIDTH / 2
    return (
        f"{beyond.mean():.1%} of tracked samples lie farther than half the stated "
        f"{WIDTH:g} cm width from the track's centre line (the 95th percentile "
        f"is {np.quantile(distance, 0.95):.3g} cm). This is tracking spread, not "
        "misfit: the tracked point is on the head, which swings past the edge "
        "of a narrow track and projects outward when raised. params gives no "
        "min_speed (no source read states a running threshold), so the share "
        "is not split into stopped and moving samples. The positions are kept "
        "as stored."
    )


def _video(
    entry: SessionEntry,
    source: str,
    times: NDArray[np.float64],
    files: list[str],
    video_times: NDArray[np.float64] | None,
) -> tuple[VideoSource | None, tuple[tuple[Conversion, ...], tuple[str, ...]]]:
    """``rec.video``, with the conversions and watch-outs it adds."""
    from neurospatial.datasets._video import (
        CALIBRATION_SOURCES,
        VideoSource,
        led_fit_watch_out,
        mirror_watch_out,
        video_calibration,
        video_frame_times,
    )

    if not files or video_times is None:
        return None, ((), ("This session publishes no video.",))
    if len(files) > 1:
        return None, (
            (),
            (
                f"The session's video is split across {len(files)} files "
                f"({', '.join(Path(f).name for f in files)}), which the video "
                "overlay does not join: rec.video is None.",
            ),
        )
    (external,) = files
    assert entry.asset_path is not None  # every session is a DANDI file
    asset_path = str(Path(entry.asset_path).parent / external)
    found = [v for v in entry.videos if v.asset_path == asset_path]
    if not found:
        msg = (
            f"{entry.session}: the file's {_VIDEO} names {external!r}, which the "
            "registry does not list beside it; rebuild the registry"
        )
        raise ValueError(msg)
    (asset,) = found
    frame_times, times_basis, offset = video_frame_times(video_times, times)
    calibration, basis, tried = video_calibration(
        bin_size=BIN_SIZE,
        landmarks=asset.landmarks,
        led_fit=asset.led_fit,
        absent=(
            f"the file's {_VIDEO} states no pixel size or transform, and its "
            "position series' reference frame is 'Arbitrary, camera'",
            f"the file stores the positions in cm only ({_POSITION}), no pixel "
            "tracking",
            "the dataset's registry holds no landmarks for this video",
            "the dataset's registry holds no LED fit for this video (run "
            "scripts/fit_video_led_calibration.py for this session)",
        ),
    )
    streamed = source.startswith(("https://", "http://"))
    camera = VideoSource(
        dataset=entry.info.dataset,
        external_file=external,
        sha256=asset.sha256,
        size_bytes=asset.size_bytes,
        frame_times=frame_times,
        frame_times_basis=times_basis,
        calibration=calibration,
        calibration_basis=basis,
        calibration_tried=tried,
        asset_id=asset.asset_id,
        cache=None if streamed else Path(source).parents[1],
    )
    conversions = (times_basis,) if basis is None else (times_basis, basis)
    watch_outs: tuple[str, ...] = () if offset is None else (offset,)
    if calibration is None:
        watch_outs += (
            "The video is not calibrated, so rec.animation(video=True) refuses: "
            "the file states no pixel size and stores no pixel tracking, and "
            "neither landmarks nor an LED fit are stored for this video (see "
            "rec.video.calibration_tried).",
        )
    else:
        if asset.led_fit is not None and len(tried) == len(CALIBRATION_SOURCES) - 1:
            period = float(np.median(np.diff(frame_times)))
            watch_outs += (led_fit_watch_out(asset.led_fit, period),)
        if (mirrored := mirror_watch_out(calibration)) is not None:
            watch_outs += (mirrored,)
    return camera, (conversions, watch_outs)
