"""Peyrache et al. (2015): mouse anterior thalamus head-direction cells, wake and sleep.

What was recorded: units in the anterior thalamus of mice, mainly the
antero-dorsal nucleus (ADn), while they foraged in a 53 x 46 cm arena for
30 to 45 minutes between two sleep sessions of about 2 hours (the paper's
Methods). Head direction is the orientation of a blue and a red LED on the
head, filmed at 30 frames/s and resampled at 39 Hz by the acquisition
system. Spikes were sorted with KlustaKwik and adjusted by hand.

The default session, ``Mouse32-140822``, is the file pynapple's and NeMoS's
head-direction tutorials download: OSF ``jb2gd``, 36.6 MB, in the
"Pynapple-files" project (G. Viejo), which states no license. It holds 49
units (31 with location ``adn``, the 31 the pynapple tutorial selects, and 18
``thalamus``), the wake and sleep epochs, slow-wave and REM sleep intervals,
and head direction ``ry`` during the wake epoch only. It has no x/y
position, so the recording has no positions and no environment.

DANDI 000056 (CC-BY-4.0) holds the dataset's other sessions, but not
``Mouse32-140822``, in a layout this loader does not read: raw blue and red
LED positions in camera pixels (``unit`` "meters" with ``conversion`` NaN),
with no head-direction series, no wake or sleep epochs and no unit
locations. They are not listed.

Sources read: the paper (Peyrache, Lacroix, Petersen & Buzsáki 2015, the
author manuscript on PubMed Central, PMC4376557), pynapple's head-direction
tutorial, the OSF project and file metadata, DANDI 000056's metadata, and
the files.

What the loader converts:

- headings: ``ry`` (radians, 0 to 2π) wrapped to [-π, π]. The file states no
  reference frame and the paper says only that heading is the orientation
  of two LEDs, so the package frame (0 = East, counter-clockwise) is assumed:
  an inferred ``Conversion``. With no positions, heading cannot be compared
  with the direction of movement. Tuning curve shapes, widths and decoding
  do not depend on the frame; absolute preferred directions do.
- epochs: ``wake`` and ``sleep`` from the epochs table's tags; ``sws`` and
  ``rem`` from their own tables; times as stored, in seconds.

Analysis choices (``params``):

- ``analysis_epochs``: the wake epoch, the only time with tracked head
  direction;
- ``dt`` 0.1 s, the decoding bin of pynapple's head-direction tutorial.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._checks import _count, gap_stats
from neurospatial.datasets._recording import Conversion, Recording, interval_rows

if TYPE_CHECKING:
    from neurospatial.datasets._catalog import SessionEntry

__all__ = ["REQUIRES", "WATCH_OUTS", "WORKED_RESULT", "load_session"]

#: The paper's finding this dataset reproduces: level A, a number with the
#: paper's stated parameters (``tests/datasets/test_worked_results.py``).
WORKED_RESULT: dict[str, Any] = {
    "status": "reproduced",
    "level": "A",
    "finding": "the share of ADn units that are head-direction cells",
    "quote": (
        '"Neurons were classified as HD cells for concentration parameter '
        "exceeding 1, peak firing rates greater than 1Hz and probability of "
        'non-uniform distribution smaller than 0.001." ... "In the thalamus, '
        "when considering only cell ensembles recorded on silicon probe "
        "shanks where at least one HD cell was detected, the average "
        'proportion of HD cells was 71%"'
    ),
    "source": (
        "Peyrache et al. 2015, Methods, Data analysis and statistics, "
        '"Head direction cells" (author manuscript, PMC4376557)'
    ),
    "paper": 0.71,
    "ours": 0.645,
    "criterion": (
        "|ours - 0.71| <= 0.163: 71% is an average over sessions with no "
        "stated spread, so the tolerance is two binomial standard deviations "
        "for one session of 31 units, 2 * sqrt(0.71 * 0.29 / 31); fixed "
        "before the run"
    ),
    "parameters": (
        "20 of the 31 units with location 'adn', in the wake epoch: "
        "compute_directional_rates with 1 degree bins and a 6 degree "
        "Gaussian (the paper's); concentration above 1 as mean vector "
        "length above I1(1) / I0(1) = 0.446; peak rate above 1 Hz; "
        "stats.rayleigh_test on the headings at the unit's spikes, each the "
        "nearest tracked sample's, p < 0.001. LED dropouts (\"Times when one "
        'of the LEDs was not detected were excluded from the analysis") are '
        "the 64 sample intervals longer than 1.5 sampling periods "
        "(0.0384 s; the others are 0.025 to 0.026 s): left out of the tuning "
        "curves (max_gap) and their spikes out of the Rayleigh test"
    ),
    "unmatched": (
        "the denominator: the file's 'adn' units, not the units of shanks "
        "with at least one HD cell (the recording holds no shank)",
        "the von Mises fit: the paper does not say how it fitted; ours is "
        "the maximum-likelihood fit to the tuning curve read as a "
        "distribution of angles, whose concentration kappa has A1(kappa) "
        "equal to the curve's mean vector length",
        "the peak rate: the smoothed curve's, not the fitted von Mises's",
        "the Rayleigh test: the paper does not say on what; ours is on the "
        "headings at the spikes, not on the occupancy-normalized field",
    ),
    "attempted": None,
    "why": None,
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("pynwb",)

DT = 0.1
#: Heading samples further apart than this are a gap in the tracking, s.
GAP = 1.0
_PAPER = "the Methods of Peyrache et al. 2015, PMC4376557"

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "Mouse32-140822": (
        "18 of the 49 units have location 'thalamus', not 'adn'; the pynapple "
        "tutorial keeps the 31 'adn' units (unit ids 4 to 34) for head-direction "
        "analyses, so select units['brain_region'] == 'adn' to match it.",
        "This session is not in DANDI 000056 (which holds Mouse32-140820), so "
        "the OSF file is its only public copy and cannot be cross-checked "
        "against a second archive.",
    ),
}


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read the OSF ``Mouse32-140822.nwb`` into a heading-only :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached NWB file.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.
    """
    from neurospatial.datasets._fetch import open_nwb
    from neurospatial.io.nwb import read_head_direction, read_intervals, read_units
    from neurospatial.stats import wrap_angle

    with open_nwb(os.fspath(path_or_stream)) as nwb:
        heading = read_head_direction(nwb)
        series = nwb.acquisition["CompassDirection"]["ry"]
        stated_unit = f"unit {series.unit!r}"
        units = read_units(nwb)
        tagged = read_intervals(nwb, interval_name="epochs").intervals_table()
        epochs = {
            tag: interval_rows(tagged[[tag in tags for tags in tagged["tags"]]])
            for tag in ("wake", "sleep")
        }
        for name in ("sws", "rem"):
            epochs[name] = interval_rows(
                read_intervals(nwb, interval_name=name).intervals_table()
            )
        started = nwb.session_start_time
    times = np.asarray(heading.times, dtype=np.float64)
    raw = np.asarray(heading.headings, dtype=np.float64)
    assert units.unit_table is not None  # read_units always reads the table
    table = units.unit_table[[]].copy()
    table["brain_region"] = units.unit_table["location"]
    table["label"] = None
    regions = tuple(dict.fromkeys(table["brain_region"].dropna()))
    return Recording(
        **entry.identity(path_or_stream),
        animal="mouse Mouse32",
        apparatus=entry.info.apparatus,
        brain_regions=regions or entry.info.brain_regions,
        clock=(
            "the file's clock, seconds from its session start time "
            f"({started:%Y-%m-%d}, the date the file was written, not the "
            "recording's 2014-08-22)"
        ),
        times=times,
        spike_times=tuple(np.asarray(t, dtype=np.float64) for t in units.spike_times),
        units=table,
        tracking=(
            "head direction: the orientation of a blue and a red LED on the head, "
            f"filmed at 30 frames/s and resampled at 39 Hz ({_PAPER})"
        ),
        headings=np.asarray(wrap_angle(raw), dtype=np.float64),
        heading_source="the file's head direction, ry, during the wake epoch",
        epochs=epochs,
        params={"analysis_epochs": epochs["wake"], "dt": DT},
        conversions=(
            Conversion(
                "headings: ry wrapped from [0, 2π) to [-π, π] "
                "(neurospatial.stats.wrap_angle)",
                "stated",
                f"the file's ry: {stated_unit}, values "
                f"{raw.min():.3g} to {raw.max():.3g}",
            ),
            Conversion(
                "headings: allocentric, 0 = East and counter-clockwise, the "
                "package convention; not rotated or mirrored",
                "inferred",
                "assumed: the file's reference_frame is empty, the paper says "
                f"only that heading is the orientation of two LEDs ({_PAPER}), "
                "and with no positions in the file heading cannot be compared "
                "with the direction of movement",
            ),
            Conversion(
                "epochs: 'wake' and 'sleep' from the tags of the epochs table; "
                "'sws' and 'rem' from their own tables",
                "stated",
                "the file's intervals: epochs (3 rows tagged wake or sleep), sws "
                "and rem",
            ),
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            "No positions: the file has head direction only, so there is no "
            "environment (the paper's 53 x 46 cm arena is not in it) and no "
            "speed; spatial maps and decoding of position are not possible.",
            _frame_watch_out(),
            *_gaps(times),
        ),
    )


def _frame_watch_out() -> str:
    return (
        "The heading frame is assumed, not stated: 0 = East and counter-clockwise "
        "(the package convention). Tuning-curve shapes, widths and decoding "
        "error do not depend on it; absolute preferred directions, and comparing "
        "them with another session's or a cue's, do."
    )


def _gaps(times: NDArray[np.float64]) -> tuple[str, ...]:
    """The tracking gaps longer than ``GAP`` seconds, with their numbers."""
    if (gaps := gap_stats(times, GAP)) is None:
        return ()
    n, total, longest = gaps
    return (
        f"Head direction has {_count(n, 'gap')} longer than {GAP:g} s "
        f"({total:.1f} s in all, the longest {longest:.2f} s), "
        "where the file has no sample. Spikes in a gap have no heading; nothing "
        "is filled in.",
    )
