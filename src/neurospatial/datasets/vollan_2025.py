"""Vollan et al. (2025): rat MEC and hippocampus in open fields and on linear tracks (EBRAINS).

What was recorded: Neuropixels units in the medial entorhinal
cortex-parasubiculum region (MEC) and the hippocampus of Long Evans rats,
foraging in an open field or shuttling on a 200 cm linear track. Head
position and head direction come from OptiTrack motion capture of markers
on the implant, projected onto the horizontal plane, at 100 Hz. The authors
sorted MEC units into grid modules on each rat's open-field session and
store each module's grid scores in the files.

Where it lives: EBRAINS dataset "Left-right-alternating theta sweeps in the
entorhinal-hippocampal spatial map (v1)", doi:10.25493/R5FR-EDG
(CC-BY-SA-4.0), a public bucket read without a login. One MATLAB v7 file per
rat, recording and session type, holding one struct, ``Dsession``, described
in the dataset's ``documentation/mat_files.md``. The loader offers every file
of ``navigation/of`` (open field, 31 files) and ``navigation/lt`` (linear
track, 7 files), 101 to 852 MB each; a session is one file, named
``<of|lt>_<rat>_<recording>``. Files are read with ``mat-io``, which reads
their MATLAB ``string`` fields (unit ids, cell types).

Sources read: the paper (Vollan, Gardner, Moser & Moser 2025, Nature 639,
995, Methods), the dataset's ``mat_files.md`` and the files.

What the loader converts:

- ``x``, ``y``: "head x-position relative to arena center (m)"
  (``mat_files.md``), times 100 to cm.
- ``hd`` ("2D head direction (azimuth)", unit and zero direction not stated)
  lies in [-π, π]; minus the direction of movement while running it has a
  circular mean within 6 degrees of 0 in every file, so it is kept as
  allocentric radians (an inferred ``Conversion`` with the measured offset).
- ``t``: "seconds elapsed since the beginning of data acquisition"
  (``mat_files.md``). "All timeseries data has been 'speed-filtered' to
  discard samples when the animal's locomotion speed was below 5 cm/s": the
  removed samples leave gaps in ``t``, which are kept as gaps (rebuilding
  times from ``dt`` would pull every later sample earlier than its spikes).
- units of every recorded area (``units.mec``, ``units.hc``), with Kilosort
  labels ``good`` and ``mua`` both kept, as the authors shared them. The
  authors' functional class (``cellType``) is ``units["label"]``; grid-module
  membership (``acorrCluId``, ``isGrid``) is ``units["module"]`` and
  ``units["is_grid"]`` and, with each module's grid scores, in ``params``.
  A unit's ``acorrCluId`` k (0 for none) is the k-th entry of
  ``unitAcorrClus.grid``, whose ``unitIds`` list the same units.
- ``epochs``: each listed session's ``validTimeRanges``, under its name,
  clipped to the recording (the first to the last sample or spike): the
  authors truncated awake sessions to a whole number of 100 s, so a listed
  session can end after the data, and in two files a listed session lies
  wholly before them and gives no epoch (``clip_epochs_to_recording``).

The declared environment, by the file's session type:

- ``open_field``: the Methods' "square open-field box with a floor size of
  150 x 150 cm", placed on the samples' minimum rotated rectangle;
- ``open_field_light`` and ``open_field_darkness``: the Methods' "circular
  arena 150 cm in diameter", placed on the samples' minimum bounding circle.
  The Methods use it for "one exceptional case" of foraging in a dark room
  and decode that session with "a succeeding recording session with room
  lights on (same arena)" (Extended Data Fig. 9k); no source read names the
  rats or files, so the circle is inferred from the session type and the
  samples' round shape, given in the ``Conversion``;
- ``linear_track``: the Methods' "200 cm linear track", a straight track
  along the samples' principal axis, centred between their 0.5th and 99.5th
  percentiles along it. No width is stated, so the track is built with a
  placement tolerance measured from the samples (``track_width``).

Every placement is fitted to the samples (an inferred ``Conversion``): the
file's (0, 0) is the midrange of its samples, not always the arena's centre.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm, the Methods' rate-map bin;
- ``min_speed`` None: the data are already speed-filtered at 5 cm/s;
- ``max_gap`` 0.015 s, one and a half sampling intervals, so a speed-filter
  gap is not bridged as occupancy;
- ``dt`` 0.25 s and ``train_epochs`` / ``test_epochs``, alternating 60 s
  blocks, for decoding;
- the authors' grid modules: ``module`` and ``is_grid`` per unit,
  ``ks2_labels``, and per module (``module_ids``) its median unit grid score
  (``module_median_grid_score``), the grid score of its median
  autocorrelogram (``module_grid_score``) and its spacing in cm
  (``module_spacing_cm``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._arenas import (
    PLACEMENT_TRIM,
    fitted_circle,
    fitted_rectangle,
    shape_fill,
)
from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    clip_epochs_to_recording,
    heading_vs_movement,
    tracked_mask,
)
from neurospatial.datasets._tracks import placement_tolerance

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The comparison with the authors' module grid scores, attempted on the
#: default session and not met by module 3
#: (``tests/datasets/test_vollan_2025.py`` recomputes it).
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "the median unit grid score of each of the authors' three grid "
        "modules in this session, which the file stores from the authors' own "
        "rate maps and scorer: 1.114, 0.986 and 0.685 for modules 1 to 3"
    ),
    "quote": (
        '"Spatial autocorrelations and grid scores were calculated as '
        "described previously, based on the individual cells' rate maps\" "
        "(Methods, \u201cRate maps and angular tuning curves\u201d), on 2.5 x 2.5 cm "
        "rate maps smoothed by a cross-validated Gaussian kernel (1 cm < "
        "sigma < 50 cm); the file's unitAcorrClus.grid(k).medianGridScore is the "
        '"median gridness score of units in this cluster" (mat_files.md)'
    ),
    "source": (
        "Vollan et al. 2025, Nature 639, 995, Methods; the dataset's "
        "documentation/mat_files.md; of_26018_2.mat"
    ),
    "paper": 0.685,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": {
        "value": 0.185,
        "criterion": (
            "every module's median within 0.15 of the authors'; a tolerance we "
            "chose after the survey had reported these medians, so not a blind "
            "criterion. Modules 1 and 2 meet it (1.176 against 1.114, 0.981 "
            "against 0.986); module 3 does not (0.185 against 0.685, the value "
            "above)"
        ),
        "parameters": (
            "of_26018_2, the 233 units the authors place in a grid module "
            "(48, 90 and 95); compute_spatial_rates on rec.env (the declared "
            "150 cm circle, 2.5 cm bins), method='binned' with a 5 cm "
            "bandwidth (our choice), max_gap=params['max_gap']; "
            "neurospatial.encoding.grid_score of spatial_autocorrelation per "
            "unit, the median over each module's units (one module 2 unit's "
            "score is undefined and left out)"
        ),
        "differs": (
            "module 3 has the largest spacing (93 cm by the authors): "
            "spatial_autocorrelation leaves undefined every lag where fewer "
            "than a quarter of the visited bins overlap, and the ring of "
            "peaks its grid score reads lies at such lags in a 150 cm arena "
            "(the survey's FINDINGS, grid-score section, G-006)",
            "the authors smooth each map with its own cross-validated kernel "
            "(1 to 50 cm); ours is one 5 cm kernel",
            "the authors' grid score follows their reference 37, whose text "
            "was not read; neurospatial's grid_score may differ in its "
            "annulus and rotations",
        ),
    },
    "why": (
        "neurospatial's grid score does not reproduce the authors' module 3 "
        "(the largest spacing) on this session: 0.185 against 0.685, while "
        "modules 1 and 2 agree within 0.07. The gap is the autocorrelogram's "
        "overlap rule, an open question for the grid-score code, which this "
        "dataset records but does not change."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("matio",)

BUCKET = (
    "https://data-proxy.ebrains.eu/api/v1/buckets/"
    "d-4080b78d-edc5-4ae4-8144-7f6de79930ea"
)
#: The stated arenas, cm: the open-field box's side, the circular arena's
#: diameter, the linear track's length.
SIDE = 150.0
DIAMETER = 150.0
TRACK = 200.0
BIN_SIZE = 2.5
#: The authors' speed filter, cm/s (mat_files.md).
SPEED_FILTER = 5.0
DT = 0.25
BLOCK = 60.0
#: Sampling intervals in a stretch of samples, times the file's ``dt``.
MAX_GAP_STEPS = 1.5
#: Steps faster than this (cm/s) count as running in the heading check.
RUNNING = 10.0
#: Heading minus the direction of movement, beyond which the heading's
#: frame is not taken as verified: the circular mean (rad) and median spread.
MAX_HEADING_OFFSET = 0.2
MAX_HEADING_SPREAD = np.radians(45.0)
#: Distance (cm) between the fitted arena centre and the file's (0, 0) above
#: which the file's "relative to arena center" is reported as not holding:
#: two bins.
CENTRE_TOLERANCE = 2 * BIN_SIZE
#: The quantile of the samples' distance from the track that half its
#: placement tolerance is.
TRACK_WIDTH_QUANTILE = 0.99
#: The percentiles of the samples along the track's axis it is centred between.
TRACK_TRIM = 0.005
_PAPER = "the Methods of Vollan et al. 2025"
_DOCS = "the dataset's mat_files.md"
_ARENAS = {
    "open_field": "square",
    "open_field_light": "circle",
    "open_field_darkness": "circle",
    "linear_track": "track",
}
_AREAS = {"mec": "MEC-parasubiculum", "hc": "hippocampus"}

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "of_26018_2": (
        r"times has 173 gaps longer than 0.02 s",
        r"end 352 s before the listed end",
        r"epochs\['open_field_light_1'\] \[3449, 5400\] s ends 352 s after it",
    ),
    "lt_27764_1": (
        r"times has 132 gaps longer than 0.02 s",
        r"of the samples lie outside every listed session",
        r"runs of 11 or more identical position samples",
        r"samples farther from the last believable sample",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "of_26018_2": (
        "The arena is inferred to be the Methods' 150 cm circular arena: the "
        "file's one session is of type open_field_light and its samples are "
        "round, but the paper names neither the rat nor the file, and the "
        "Methods give the box (150 x 150 cm) for every other open-field "
        "recording.",
    ),
    "lt_27764_1": (
        "The track is drawn along the samples' principal axis; the paper "
        "states its length (200 cm) but no width or placement, and the "
        "samples reach past its ends where the rat turns at the reward "
        "ports.",
    ),
}


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one ``Dsession`` file into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached ``.mat`` file.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If ``hd`` is not radians in [-π, π], the file's session types name
        no arena the loader knows or more than one, or a grid module holds
        more units than the file states for it.
    """
    from neurospatial._angles import wrap_angle

    raw = read_session(Path(path_or_stream))
    times = raw["t"]
    positions = np.column_stack([raw["x"], raw["y"]]) * 100.0
    headings = raw["hd"]
    finite = headings[np.isfinite(headings)]
    if finite.size and (finite.min() < -np.pi - 1e-6 or finite.max() > np.pi + 1e-6):
        msg = (
            f"{entry.session}: hd runs from {finite.min():.3g} to "
            f"{finite.max():.3g}, not radians in [-pi, pi]; recheck its unit"
        )
        raise ValueError(msg)
    offset, spread = heading_vs_movement(times, positions, headings, min_speed=RUNNING)
    kinds = {_ARENAS.get(kind) for kind in raw["session_types"]}
    if len(kinds) != 1 or None in kinds:
        msg = (
            f"{entry.session}: session types {raw['session_types']} name "
            f"{'no' if None in kinds else 'more than one'} arena the loader "
            f"knows ({sorted(_ARENAS)})"
        )
        raise ValueError(msg)
    (kind,) = kinds
    assert kind is not None
    env, env_source, placement, width_quantile, centre = _declared(
        kind, positions, entry.session
    )
    train, test = alternating_blocks(float(times[0]), float(times[-1]), BLOCK)
    dt = raw["dt"]
    clipped = clip_epochs_to_recording(
        raw["sessions"],
        times,
        raw["spike_times"],
        source="the file's sessions table (validTimeRanges)",
    )
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {entry.session.split('_')[1]}",
        apparatus=_apparatus(kind),
        brain_regions=tuple(dict.fromkeys(raw["units"]["brain_region"])),
        clock=(
            "the acquisition clock, seconds since the start of data acquisition "
            f"(mat_files.md), from {times[0]:.2f} s"
        ),
        times=times,
        spike_times=raw["spike_times"],
        units=raw["units"],
        env=env,
        env_kind="declared",
        env_source=env_source,
        track_width_quantile=width_quantile,
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: retroreflective markers on the implant, tracked in 3-D by "
            f"OptiTrack and projected onto the horizontal plane ({_PAPER})"
        ),
        headings=np.asarray(wrap_angle(headings), dtype=np.float64),
        heading_source=(
            f"the file's hd, '2D head direction (azimuth)' ({_DOCS}), from the "
            "motion capture"
        ),
        epochs=clipped.epochs,
        params={
            "bin_size": BIN_SIZE,
            "min_speed": None,
            "max_gap": MAX_GAP_STEPS * dt,
            "dt": DT,
            "train_epochs": train,
            "test_epochs": test,
            "module": raw["units"]["module"].to_numpy(),
            "is_grid": raw["units"]["is_grid"].to_numpy(),
            "ks2_labels": raw["units"]["ks2_label"].to_numpy(),
            "module_ids": raw["module_ids"],
            "module_median_grid_score": raw["module_median_grid_score"],
            "module_grid_score": raw["module_grid_score"],
            "module_spacing_cm": raw["module_spacing_cm"],
        },
        conversions=(
            Conversion(
                "positions: x, y in m, times 100 to cm",
                "stated",
                f"{_DOCS}: x, y 'head x-position relative to arena center (m)'",
            ),
            _heading_conversion(offset, spread),
            Conversion(
                "times: t as stored, with the speed filter's gaps",
                "stated",
                f"{_DOCS}: t 'times (s) of head tracking data samples', 'All "
                "timeseries data has been speed-filtered to discard samples when "
                f"the animal's locomotion speed was below 5 cm/s'; dt {dt:g} s",
            ),
            Conversion(
                f"units: {len(raw['spike_times'])} from "
                + ", ".join(
                    f"units.{area} ({n})" for area, n in raw["area_counts"].items()
                )
                + "; units['label'] is cellType, units['module'] acorrCluId (0 "
                "for none), units['is_grid'] isGrid",
                "stated",
                f"{_DOCS}: cellType 'functional class assigned to this unit', "
                "acorrCluId 'numerical ID of the parent grid-module cluster', "
                "isGrid 'whether cell is an identified grid cell'; each module's "
                "unitIds list the units whose acorrCluId is its index",
            ),
            Conversion(
                "epochs: each listed session's validTimeRanges, under its name",
                "stated",
                f"{_DOCS}: sessions, 'list of individual sessions (time epochs) "
                "included'; listed: "
                + "; ".join(
                    f"{name} {ranges.tolist()}"
                    for name, ranges in raw["sessions"].items()
                ),
            ),
            *(() if clipped.conversion is None else (clipped.conversion,)),
            *placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_heading_watch_out(offset, spread),
            _speed_filter_gaps(times, raw["spike_times"], dt),
            *_outside_sessions(times, raw["sessions"]),
            *(() if clipped.watch_out is None else (clipped.watch_out,)),
            *_truncated(times, raw["sessions"], dt),
            *_off_centre(centre),
            *_modules(raw, kind),
        ),
    )


def read_session(path: Path) -> dict[str, Any]:
    """The arrays of one ``Dsession`` file: tracking in m and rad, units, sessions, modules.

    Parameters
    ----------
    path : Path
        A ``navigation/<type>/<rat>_<recording>.mat`` file.

    Returns
    -------
    dict
        ``t``, ``x``, ``y``, ``hd`` (n_time,); ``dt``; ``spike_times`` (one
        sorted array per unit); ``units``, the unit table; ``area_counts``;
        ``session_types``; ``sessions``, each listed session's name to its
        ``validTimeRanges`` (n, 2); and per grid module ``module_ids``,
        ``module_median_grid_score``, ``module_grid_score`` and
        ``module_spacing_cm``.

    Raises
    ------
    ValueError
        If a grid module holds more units than its ``nUnits``, or a unit's
        ``acorrCluId`` names no module.
    """
    import matio
    import pandas as pd

    d = matio.load_from_mat(str(path))["Dsession"]

    def series(name: str) -> NDArray[np.float64]:
        return np.asarray(_value(d[name]), dtype=np.float64).ravel()

    units = _value(d["units"])
    trains: list[NDArray[np.float64]] = []
    rows: list[dict[str, Any]] = []
    area_counts: dict[str, int] = {}
    for area in units.dtype.names:
        table = _value(units[area])
        if not isinstance(table, np.ndarray) or table.size == 0:
            area_counts[area] = 0
            continue
        fields = set(table.dtype.names)
        area_counts[area] = table.size
        for unit in table.ravel():
            train = np.asarray(_value(unit["spikeTimes"]), dtype=np.float64).ravel()
            trains.append(train)
            rows.append(
                {
                    "unit_id": _text(unit["id"]),
                    "brain_region": _AREAS.get(area, area),
                    "label": _text(unit["cellType"]) if "cellType" in fields else None,
                    "ks2_label": _text(unit["ks2Label"]),
                    "module": _integer(unit["acorrCluId"])
                    if "acorrCluId" in fields
                    else 0,
                    "is_grid": bool(_integer(unit["isGrid"]))
                    if "isGrid" in fields
                    else False,
                }
            )
    table = pd.DataFrame(rows).set_index("unit_id")
    sessions = _value(d["sessions"])
    listed = {
        _text(name): np.asarray(_value(ranges), dtype=np.float64).reshape(-1, 2)
        for name, ranges in zip(
            sessions["name"].ravel(), sessions["validTimeRanges"].ravel(), strict=True
        )
    }
    modules = _grid_modules(_value(d["unitAcorrClus"]), table["module"].to_numpy())
    return {
        "t": series("t"),
        "x": series("x"),
        "y": series("y"),
        "hd": series("hd"),
        "dt": float(np.asarray(_value(d["dt"])).ravel()[0]),
        "spike_times": tuple(trains),
        "units": table,
        "area_counts": area_counts,
        "session_types": [_text(kind) for kind in sessions["type"].ravel()],
        "sessions": listed,
        **modules,
    }


def _grid_modules(clusters: Any, module: NDArray[np.int64]) -> dict[str, Any]:
    """Each grid module's id, grid scores and spacing; empty where the file has none."""
    names = getattr(getattr(clusters, "dtype", None), "names", None) or ()
    grid = _value(clusters["grid"]) if "grid" in names else None
    if grid is None or not getattr(grid.dtype, "names", None):
        if np.any(module != 0):
            msg = "units name grid modules, but unitAcorrClus holds none"
            raise ValueError(msg)
        empty = np.empty(0, dtype=np.float64)
        return {
            "module_ids": np.empty(0, dtype=np.int64),
            "module_median_grid_score": empty,
            "module_grid_score": empty,
            "module_spacing_cm": empty,
        }

    def per_module(name: str) -> NDArray[np.float64]:
        return np.array(
            [float(np.asarray(_value(m[name])).ravel()[0]) for m in grid.ravel()]
        )

    sizes = per_module("nUnits").astype(int)
    found = np.array([np.sum(module == k) for k in range(1, grid.size + 1)])
    # Modules are defined on the open-field session; another session's file
    # can hold fewer of their units, never more.
    if np.any(found > sizes) or np.any(module > grid.size) or np.any(module < 0):
        msg = (
            f"units' acorrCluId counts {found.tolist()} exceed the module sizes "
            f"{sizes.tolist()}, or name a module the file does not have"
        )
        raise ValueError(msg)
    return {
        "module_ids": np.arange(1, grid.size + 1),
        "module_median_grid_score": per_module("medianGridScore"),
        "module_grid_score": per_module("gridScore"),
        "module_spacing_cm": per_module("gridSpacingMeters") * 100.0,
        "module_sizes": sizes,
        "module_session_type": _text(clusters["sessionType"]),
    }


def _value(field: Any) -> Any:
    """Unwrap mat-io's nested 1 x 1 cells to the array or scalar inside."""
    while isinstance(field, np.ndarray) and field.dtype == object and field.size == 1:
        field = field.ravel()[0]
    return field


def _text(field: Any) -> str:
    """A MATLAB string or char field as a Python string."""
    return str(np.asarray(_value(field)).ravel()[0])


def _integer(field: Any) -> int:
    """A MATLAB numeric scalar field as a Python int."""
    return int(np.asarray(_value(field)).ravel()[0])


def _apparatus(kind: str) -> str:
    """The session's apparatus, from its arena kind."""
    return {
        "square": "150 x 150 cm open-field box",
        "circle": "circular open arena, 150 cm in diameter",
        "track": "200 cm linear track",
    }[kind]


def _declared(
    kind: str, positions: NDArray[np.float64], session: str
) -> tuple[Environment, str, tuple[Conversion, ...], float | None, NDArray[np.float64]]:
    """The stated arena of ``kind`` placed on the samples.

    Returns
    -------
    env : Environment
    env_source : str
    conversions : tuple of Conversion
        The placement, and for a track its tolerance.
    width_quantile : float or None
        For a track whose tolerance the samples set, ``TRACK_WIDTH_QUANTILE``.
    centre : ndarray, shape (2,)
        The arena's fitted centre, cm.
    """
    if kind == "track":
        return _declared_track(positions, session)
    import shapely
    from shapely import affinity

    from neurospatial import Environment

    rectangle, circle = shape_fill(positions)
    shape = (
        f"the samples' convex hull fills {rectangle:.0%} of their minimum "
        f"rectangle and {circle:.0%} of their minimum circle (a filled square "
        "gives about 100% and 64%, a filled circle 79% and 100%)"
    )
    if kind == "square":
        centre, angle, sides = fitted_rectangle(positions)
        half = SIDE / 2
        polygon = affinity.translate(
            affinity.rotate(
                shapely.box(-half, -half, half, half), angle, use_radians=True
            ),
            *centre,
        )
        source = (
            f"the stated {SIDE:g} x {SIDE:g} cm open-field box ({_PAPER}: 'a square "
            "open-field box with a floor size of 150 x 150 cm'), "
            f"{BIN_SIZE:g} cm bins"
        )
        placement = Conversion(
            f"environment: the stated {SIDE:g} x {SIDE:g} cm square centred at "
            f"({centre[0]:.3g}, {centre[1]:.3g}) cm, rotated "
            f"{np.degrees(angle):+.2f} deg from the file's axes; the samples' own "
            "sides against the stated side: "
            + ", ".join(f"{s:.4g} cm, {s / SIDE - 1:+.1%}" for s in sides),
            "inferred",
            "the centre and rotation are those of the minimum rotated rectangle "
            f"around the tracked samples, less the {4 * PLACEMENT_TRIM:.1%} "
            f"farthest from their median; {shape}",
        )
    else:
        centre, diameter = fitted_circle(positions)
        polygon = shapely.Point(*centre).buffer(DIAMETER / 2, quad_segs=16)
        source = (
            f"the stated circular arena {DIAMETER:g} cm in diameter ({_PAPER}: 'a "
            "circular arena 150 cm in diameter'), inferred for this session "
            f"type, {BIN_SIZE:g} cm bins"
        )
        placement = Conversion(
            f"environment: a circle {DIAMETER:g} cm in diameter centred at "
            f"({centre[0]:.3g}, {centre[1]:.3g}) cm; the samples' own diameter "
            f"{diameter:.4g} cm, {diameter / DIAMETER - 1:+.1%} of it",
            "inferred",
            f"{_PAPER} use the circular arena for a dark-room session and the "
            "lit session after it, and the box for the other open-field "
            "recordings, without naming rats or files; this session's type is "
            f"not plain open_field and {shape}. The centre is that of the "
            "samples' minimum bounding circle, less the "
            f"{4 * PLACEMENT_TRIM:.1%} farthest from their median",
        )
    env = Environment.from_polygon(
        polygon, bin_size=BIN_SIZE, units="cm", name=f"vollan_2025 {session}"
    )
    return env, source, (placement,), None, centre


def _declared_track(
    positions: NDArray[np.float64], session: str
) -> tuple[Environment, str, tuple[Conversion, ...], float | None, NDArray[np.float64]]:
    """The stated 200 cm track along the samples' principal axis."""
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._tracks import nearest_segment

    samples = positions[tracked_mask(positions)]
    mean, axis, along, (low, high) = principal_axis(samples)
    middle = mean + 0.5 * (low + high) * axis
    start, end = middle - TRACK / 2 * axis, middle + TRACK / 2 * axis
    graph = nx.Graph()
    graph.add_node("start", pos=(float(start[0]), float(start[1])))
    graph.add_node("end", pos=(float(end[0]), float(end[1])))
    graph.add_edge("start", "end", distance=TRACK)
    distance = nearest_segment(samples, start[None], end[None]).distance
    width, width_quantile, evidence = placement_tolerance(
        distance,
        TRACK_WIDTH_QUANTILE,
        floor=BIN_SIZE,
        measured=(
            "no source read states the track's width. The tracked head's "
            "distance from the track"
        ),
        reason=(
            "The samples reach past the track's ends where the rat turns at "
            "the reward ports, so the tolerance also places most of those."
        ),
    )
    env = Environment.from_graph(
        graph,
        [("start", "end")],
        bin_size=BIN_SIZE,
        edge_spacing=0.0,
        units="cm",
        name=f"vollan_2025 {session}",
        track_width=width,
    )
    beyond = float(np.mean(np.abs(along - 0.5 * (low + high)) > TRACK / 2))
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    source = (
        f"the stated {TRACK:g} cm linear track ({_PAPER}: 'a 200 cm linear "
        "track'), as a straight track graph, width not stated, "
        f"{BIN_SIZE:g} cm bins"
    )
    conversions = (
        Conversion(
            f"environment: a straight {TRACK:g} cm track from ({start[0]:.3g}, "
            f"{start[1]:.3g}) to ({end[0]:.3g}, {end[1]:.3g}) cm, at {angle:.1f} "
            "deg along the samples' principal axis",
            "inferred",
            f"{_PAPER} state the length; the file states no placement. The track "
            f"is centred between the samples' {TRACK_TRIM:.1%} and "
            f"{1 - TRACK_TRIM:.1%} percentiles along the axis, which span "
            f"{high - low:.4g} cm; {beyond:.1%} of tracked samples lie beyond "
            "its ends",
        ),
        Conversion(
            f"environment: track_width {width:.3g} cm, a placement tolerance, "
            "not the track's width: twice the tracked samples' "
            f"{TRACK_WIDTH_QUANTILE:.0%} quantile distance from the track",
            "inferred",
            evidence,
        ),
    )
    return env, source, conversions, width_quantile, middle


def principal_axis(
    samples: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[float, float]
]:
    """The samples' principal axis and where they lie along it.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, 2)
        Tracked positions.

    Returns
    -------
    mean : ndarray, shape (2,)
    axis : ndarray, shape (2,)
        Unit vector, its larger component positive.
    along : ndarray, shape (n_samples,)
        Each sample's coordinate along ``axis`` from ``mean``.
    ends : (float, float)
        The ``TRACK_TRIM`` and ``1 - TRACK_TRIM`` quantiles of ``along``.

    Examples
    --------
    >>> line = np.column_stack([np.linspace(0, 6, 601), np.linspace(0, 8, 601)])
    >>> mean, axis, along, ends = principal_axis(line)
    >>> np.round(axis, 6).tolist(), round(ends[1] - ends[0], 3)
    ([0.6, 0.8], 9.9)
    """
    mean = samples.mean(axis=0)
    _, _, axes = np.linalg.svd(samples - mean, full_matrices=False)
    axis = axes[0] if axes[0][np.argmax(np.abs(axes[0]))] > 0 else -axes[0]
    along = (samples - mean) @ axis
    low, high = np.quantile(along, [TRACK_TRIM, 1 - TRACK_TRIM])
    return mean, axis, along, (float(low), float(high))


def _heading_conversion(offset: float, spread: float) -> Conversion:
    """How ``hd`` became ``headings``, with its check against movement."""
    verified = abs(offset) <= MAX_HEADING_OFFSET and spread <= MAX_HEADING_SPREAD
    return Conversion(
        "headings: hd in [-pi, pi] rad, kept; taken as allocentric in the "
        "positions' frame (0 = East, counter-clockwise)",
        "inferred",
        f"{_DOCS} states no unit or zero for hd; minus the direction of movement "
        f"above {RUNNING:g} cm/s it has a circular mean of "
        f"{np.degrees(offset):+.1f} deg and a median spread of "
        f"{np.degrees(spread):.1f} deg"
        + (
            ", so it points along the path"
            if verified
            else ", so its frame is not verified (see Watch out)"
        ),
    )


def _heading_watch_out(offset: float, spread: float) -> tuple[str, ...]:
    """A watch-out when ``hd`` does not follow the direction of movement."""
    if abs(offset) <= MAX_HEADING_OFFSET and spread <= MAX_HEADING_SPREAD:
        return ()
    return (
        "The heading frame is not verified by movement: hd minus the direction "
        f"of movement above {RUNNING:g} cm/s has a circular mean of "
        f"{np.degrees(offset):+.1f} deg and a median spread of "
        f"{np.degrees(spread):.1f} deg (a frame in the positions' axes gives a "
        f"mean within {np.degrees(MAX_HEADING_OFFSET):.0f} deg of 0 and a spread "
        f"under {np.degrees(MAX_HEADING_SPREAD):.0f} deg). The values are kept "
        "as stored.",
    )


def _speed_filter_gaps(
    times: NDArray[np.float64], spike_times: tuple[NDArray[np.float64], ...], dt: float
) -> str:
    """The gaps the authors' speed filter left in ``times``, and the spikes in them."""
    steps = np.diff(times)
    bar = 2 * dt
    long = steps > bar
    spikes = np.concatenate([np.empty(0), *spike_times])
    after = np.searchsorted(times, spikes, side="right")
    inside = (after > 0) & (after < times.size)
    in_gap = np.zeros(spikes.size, dtype=bool)
    in_gap[inside] = long[after[inside] - 1] & (
        spikes[inside] > times[after[inside] - 1] + dt
    )
    return (
        f"times has {int(long.sum())} gaps longer than {bar:g} s "
        f"({steps[long].sum():.0f} s of the {times[-1] - times[0]:.0f} s span, the "
        f"longest {steps.max():.1f} s): the authors removed every sample slower "
        f"than {SPEED_FILTER:g} cm/s ({_DOCS}), and the gaps are kept. "
        f"{int(in_gap.sum())} of {spikes.size} spikes lie inside them. "
        "compute_spatial_rates bridges a gap shorter than its max_gap (1 s by "
        "default) as occupancy and warns that the times look boolean-masked; "
        "pass max_gap=params['max_gap'] so the removed time is not counted, and "
        "no min_speed (the data are already filtered)."
    )


def _outside_sessions(
    times: NDArray[np.float64], sessions: dict[str, NDArray[np.float64]]
) -> tuple[str, ...]:
    """A watch-out when samples lie outside every listed session."""
    rows = np.vstack(list(sessions.values()))
    inside = np.zeros(times.size, dtype=bool)
    for start, stop in rows:
        inside |= (times >= start) & (times <= stop)
    if inside.all():
        return ()
    listed = "; ".join(
        f"{name} " + ", ".join(f"{a:g} to {b:g} s" for a, b in ranges)
        for name, ranges in sessions.items()
    )
    outside = times[~inside]
    return (
        f"{np.mean(~inside):.1%} of the samples lie outside every listed session "
        f"({listed}), from {outside.min():.1f} to {outside.max():.1f} s. They are "
        "kept, with their spikes; the file does not say which session they "
        "belong to.",
    )


def _truncated(
    times: NDArray[np.float64], sessions: dict[str, NDArray[np.float64]], dt: float
) -> tuple[str, ...]:
    """A watch-out when the samples end well before the last listed session does."""
    end = float(np.vstack(list(sessions.values()))[:, 1].max())
    short = end - float(times[-1])
    if short <= 1.0:
        return ()
    held = times.size * dt
    whole = bool(np.isclose(held % 100.0, 0.0) or np.isclose(held % 100.0, 100.0))
    return (
        f"The samples end {short:.0f} s before the listed end of the last session "
        f"({end:g} s). The file holds {times.size} samples, {held:g} s at dt"
        + (
            f", a whole number of 100 s, as {_PAPER} describe: awake sessions "
            "were 'truncated in length to the nearest multiple of 100 s ... by "
            "trimming the last part of the behaviour session'."
            if whole
            else "."
        ),
    )


def _off_centre(centre: NDArray[np.float64]) -> tuple[str, ...]:
    """A watch-out when the fitted arena is not centred on the file's (0, 0)."""
    offset = float(np.hypot(*centre))
    if offset <= CENTRE_TOLERANCE:
        return ()
    return (
        f"The arena placed on the samples is centred at ({centre[0]:.1f}, "
        f"{centre[1]:.1f}) cm, {offset:.1f} cm from the file's (0, 0), though "
        f"{_DOCS} gives x and y 'relative to arena center'. Positions are kept "
        "as stored; the declared environment follows the samples.",
    )


def _modules(raw: dict[str, Any], kind: str) -> tuple[str, ...]:
    """What the file's grid modules are, for this session."""
    ids = raw["module_ids"]
    if not ids.size:
        return (
            "The file holds no grid modules (unitAcorrClus is empty): "
            "units['module'] is 0 and units['is_grid'] False for every unit.",
        )
    if kind != "track":
        return ()
    counts = [int(np.sum(raw["units"]["module"] == k)) for k in ids]
    return (
        f"The grid modules were defined on the rat's open-field session "
        f"(unitAcorrClus.sessionType {raw['module_session_type']!r}), and their "
        "grid scores in params are from that session's maps, not this track's. "
        f"This file holds {counts} of their {raw['module_sizes'].tolist()} units.",
    )
