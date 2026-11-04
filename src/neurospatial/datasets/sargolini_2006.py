"""Sargolini et al. (2006): rat medial entorhinal cortex in a 1 m box (DANDI 000582).

What was recorded: units in medial entorhinal cortex (MEC) of Long Evans rats
running in a 1 x 1 m enclosure (DANDI 000582, 118 sessions, CC-BY-4.0).
DANDI's description says the authors recorded from each principal cell layer
of MEC; the files' experiment_description says the sample includes
conjunctive and head-direction cells from layers III and V, and each unit's
histology names its layer (MEC LII, LIII, LV or LVI in the 12 sessions
opened). The authors' readme says sessions are normally 10 minutes. Position is one
tracking LED (``SpatialSeriesLED1``), sampled every 0.02 s (50 Hz); some files
also hold the second LED (``SpatialSeriesLED2``).

The default session, ``sub-10073_ses-17010302``, is the one pynapple's DANDI
tutorial streams: 8 layer II units. ``sub-10884_ses-03080402``, used by
pynapple's null-distribution tutorial, is also checked by hand.

What the loader converts:

- positions: the file stores cm with ``unit="meters"`` and
  ``conversion=0.01``; ``read_position`` returns metres, multiplied by 100
  to give cm;
- headings: none. Neither the file nor the authors' readme says which of two
  LEDs is at the front, so head direction is not computed even where both
  are stored.

The declared environment is the stated 100 x 100 cm square, placed at the
midrange of the tracked samples, axes as in the file (which states no
origin), with 2.5 cm bins. How far each side of the samples' range falls
short of or past 100 cm is in its ``Conversion``. In the 10-minute sessions
opened so far the positions are centred on their own midrange and the
longer axis spans exactly 100 cm, so the coordinates were already fitted to
the box.

Analysis choices (``params``):

- ``bin_size`` 2.5 cm: 40 bins across the box, as in the spatial-information
  table pynapple's tutorial prints for ``sub-10884_ses-03080402``;
- ``dt`` 0.25 s, the decoding bin;
- ``min_speed`` None: no speed filter, as in that table;
- ``train_epochs`` and ``test_epochs``: alternating 60 s blocks, so decoding
  is always tested on held-out time.

Of the 12 sessions opened, two are 1200 s long and span about 1.5 m and 2 m
(``sub-11265_ses-16030601``, ``sub-11207_ses-24060501``), though their
session_description states the 1 x 1 m enclosure. No source read states the
box they were recorded in, and the units check refuses them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import Conversion, Recording

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = ["REQUIRES", "WATCH_OUTS", "load_session"]

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

#: The side of the stated enclosure, in cm.
SIDE = 100.0
BIN_SIZE = 2.5
DT = 0.25
BLOCK = 60.0
_LED1, _LED2 = "SpatialSeriesLED1", "SpatialSeriesLED2"
_CM_PER = {"m": 100.0, "cm": 1.0, "mm": 0.1}
#: Below this many cm, two coordinates are equal up to float rounding.
_ROUNDING = 1e-6

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "sub-10073_ses-17010302": (
        "The file's experiment_description says the sample is from layers III "
        "and V, but this session's units table gives every unit's histology as "
        "MEC LII; units['brain_region'] follows the units table.",
    ),
    "sub-10884_ses-03080402": (
        "The first of the 30000 position samples, at 0 s, is NaN (not tracked); "
        "it is kept as NaN.",
    ),
}

_SMOOTHED = (
    "The authors smoothed the positions with a moving mean filter to remove "
    "tracking jitter (the authors' readme, readme_sargolini2006.txt, NIRD "
    "doi:10.11582/2014.00003), so speeds and held-sample counts are "
    "those of smoothed positions."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one DANDI 000582 session into a :class:`Recording`.

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
        If the position series is in a unit other than m, cm or mm.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_position, read_units

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        position = read_position(nwb, position_name=_LED1)
        units = read_units(nwb)
        stored = nwb.processing["behavior"]["Position"].spatial_series
        led = stored[_LED1]
        stated_unit = f"unit {led.unit!r}, conversion {led.conversion:g}"
        has_second_led = _LED2 in stored
        subject = nwb.subject.subject_id
        started = nwb.session_start_time
    if position.units not in _CM_PER:
        msg = (
            f"{entry.session}: {_LED1} is in {position.units!r}, not a length "
            "unit the loader converts to cm (m, cm or mm)"
        )
        raise ValueError(msg)
    scale = _CM_PER[position.units]
    times = np.asarray(position.times, dtype=np.float64)
    positions = np.asarray(position.positions, dtype=np.float64) * scale
    env, placement = _declared_square(positions, entry.session, position.frame)
    assert units.unit_table is not None  # read_units always reads the table
    table = units.unit_table.copy()
    table["brain_region"] = table.get("histology")
    table["label"] = None
    regions = tuple(dict.fromkeys(table["brain_region"].dropna()))
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    note = ", a placeholder" if started.year == 1900 else ""
    clock = (
        "the file's clock, seconds from its session start time "
        f"({started:%Y-%m-%d}{note})"
    )
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"rat {subject}",
        apparatus=entry.info.apparatus,
        brain_regions=regions or entry.info.brain_regions,
        clock=clock,
        times=times,
        spike_times=tuple(np.asarray(t, dtype=np.float64) for t in units.spike_times),
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {SIDE:g} x {SIDE:g} cm square (the 1 x 1 m enclosure of "
            "the authors' readme and the file's session_description), "
            f"{BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=f"one tracking LED, the file's {_LED1}",
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": None,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            Conversion(
                f"positions: cm = {_LED1} in {position.units} (read_position "
                f"applies the file's conversion) x {scale:g}",
                "stated",
                f"the file's {_LED1}: {stated_unit}",
            ),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _led_watch_out(has_second_led),
            _SMOOTHED,
            *_fitted_coordinates(positions),
        ),
    )


def _declared_square(
    positions: NDArray[np.float64], session: str, frame: str | None
) -> tuple[Environment, Conversion]:
    """The stated square at the tracked samples' midrange, and how it was placed.

    The placement records each axis's residual: the tracked samples' range on
    that axis against the stated side, negative where the animal kept off a
    wall, positive where samples reach past it.

    The files state no origin (their reference frame is "(0,0) is not
    known") and no rotation, so the square keeps the file's axes and is
    centred where the tracked samples' range is centred.
    """
    import shapely

    from neurospatial import Environment

    tracked = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = tracked.min(axis=0), tracked.max(axis=0)
    centre = (low + high) / 2
    residuals = ", ".join(
        f"{axis} {span:.4g} cm, {span / SIDE - 1:+.1%}"
        for axis, span in zip("xy", high - low, strict=True)
    )
    half = SIDE / 2
    square = shapely.box(*(centre - half), *(centre + half))
    env = Environment.from_polygon(
        square, bin_size=BIN_SIZE, units="cm", name=f"sargolini_2006 {session}"
    )
    placement = Conversion(
        f"environment: the stated {SIDE:g} x {SIDE:g} cm square centred at "
        f"({centre[0]:.3g}, {centre[1]:.3g}) cm, sides along the file's x and y; "
        f"the samples' range against the stated side: {residuals}",
        "inferred",
        f"the file states no origin or rotation (reference frame {frame!r}); "
        "the centre is the midrange of the tracked samples, x "
        f"{low[0]:.4g} to {high[0]:.4g} cm and y {low[1]:.4g} to {high[1]:.4g} cm",
    )
    return env, placement


def _led_watch_out(has_second_led: bool) -> str:
    """Why there are no headings, for a file with one or two LEDs."""
    if has_second_led:
        return (
            f"No headings, though the file also stores the second LED ({_LED2}): "
            "neither the file nor the authors' readme says which LED is at the "
            f"front, so head direction is not computed. Positions are {_LED1}'s."
        )
    return (
        f"No headings: the file stores only the first LED ({_LED1}), though its "
        "session_description says two are given, so head direction, and the "
        "paper's head-direction and conjunctive cells, cannot be found from it."
    )


def _fitted_coordinates(positions: NDArray[np.float64]) -> tuple[str, ...]:
    """A watch-out when the positions are centred and scaled to the stated side."""
    tracked = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = tracked.min(axis=0), tracked.max(axis=0)
    span = float(np.max(high - low))
    if not (np.all(np.abs(low + high) < _ROUNDING) and abs(span - SIDE) < _ROUNDING):
        return ()
    return (
        "The positions are centred on their own midrange ((max + min) / 2 is "
        f"(0, 0) cm) and their longer axis spans exactly {SIDE:g} cm, the side of "
        "the box: the coordinates were centred and scaled to the box before "
        "publication, so they do not show where the walls were, and the span "
        "cannot reveal a scale error.",
    )
