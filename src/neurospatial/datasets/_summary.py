"""The text printed for a loaded recording: what it is and what to watch for.

Six sections, in this order: identity (paper, animal, session, apparatus,
archive, license, the file's size and URL, citation), time (duration, clock, sampling rate, epochs),
tracking (what was tracked, tracked share, longest gap, speed with its
fastest step, head-direction coverage), environment, neurons, and "Watch out", which is always printed in
full. The ranges are computed from the arrays; the rest comes from the
recording's own fields.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from neurospatial.datasets._recording import Recording

__all__ = ["heading_coverage", "summary_text"]

_LABEL = 12
_WIDTH = 78
#: Width of the head-direction bins whose coverage the summary reports.
HEADING_BIN_DEGREES = 10


def summary_text(rec: Recording) -> str:
    """The full summary of ``rec``, the text ``repr(rec)`` shows.

    Parameters
    ----------
    rec : Recording

    Returns
    -------
    str
        The identity lines, then the labelled sections Time, Epochs,
        Tracking, Environment, Neurons and Watch out, one line each or more.
    """
    lines = [
        *_identity(rec),
        *_labelled("Time", _time(rec)),
        *_labelled("Epochs", [_epochs(rec)]),
        *_labelled("Tracking", _tracking(rec)),
        *_labelled("Environment", _environment(rec)),
        *_labelled("Neurons", _neurons(rec)),
        *_watch_out(rec),
    ]
    return "\n".join(lines)


def heading_coverage(headings: NDArray[np.float64]) -> float:
    """The share of 10° head-direction bins holding at least one sample.

    Parameters
    ----------
    headings : ndarray, shape (n_time,)
        Radians in [-π, π]; NaN samples are ignored.

    Returns
    -------
    float
        From 0 (no finite heading) to 1 (every bin of the circle visited).
    """
    n_bins = 360 // HEADING_BIN_DEGREES
    finite = headings[np.isfinite(headings)]
    # -π and π are one direction: wrap before binning.
    turns = np.mod(finite + np.pi, 2 * np.pi) / (2 * np.pi)
    bins = np.floor(turns * n_bins).astype(int)
    return np.unique(np.clip(bins, 0, n_bins - 1)).size / n_bins


def _labelled(label: str, lines: list[str]) -> list[str]:
    """``lines`` after a label column, wrapped, continuation lines indented."""
    out = []
    for index, line in enumerate(lines):
        first = label.ljust(_LABEL) if index == 0 else " " * _LABEL
        out.extend(
            textwrap.wrap(
                line,
                width=_WIDTH,
                initial_indent=first,
                subsequent_indent=" " * (_LABEL + 2),
            )
        )
    return out


def _identity(rec: Recording) -> list[str]:
    from neurospatial.datasets._load import _size

    regions = ", ".join(rec.brain_regions) or "brain region not stated"
    citation = textwrap.shorten(rec.citation, width=_WIDTH - 22, placeholder=" ...")
    lines = [
        f"{rec.paper}, {rec.animal}, session {rec.session} ({rec.dataset})",
        f"{rec.apparatus}; {regions}. {rec.provenance.archive}, {rec.license}",
        f"Source: a {_size(rec.provenance.size_bytes)} file at {rec.provenance.url}",
        f"Cite: {citation} -> rec.citation",
    ]
    # A URL is never broken across lines, so it can be copied whole.
    return [
        wrapped
        for line in lines
        for wrapped in textwrap.wrap(
            line, _WIDTH, break_long_words=False, break_on_hyphens=False
        )
    ]


def _time(rec: Recording) -> list[str]:
    sampled = "position" if rec.positions is not None else "heading"
    return [
        f"{rec.duration:.0f} s from the first to the last sample, in seconds on "
        f"{rec.clock}; {sampled} sampled at {rec.sampling_rate:.4g} Hz"
    ]


def _epochs(rec: Recording) -> str:
    if not rec.epochs:
        return "none named in this file"
    parts = []
    for name, rows in rec.epochs.items():
        if len(rows) == 1:
            parts.append(f"{name} ({rows[0, 0]:.6g} to {rows[0, 1]:.6g} s)")
        else:
            total = float(np.sum(rows[:, 1] - rows[:, 0]))
            parts.append(f"{name} ({len(rows)} rows, {total:.0f} s in all)")
    return "; ".join(parts)


def _tracking(rec: Recording) -> list[str]:
    if rec.positions is None:
        lines = [f"{rec.tracking} (heading only; no positions in this dataset)"]
    else:
        lines = [f"{rec.tracking} ({rec.n_dims}-D, {rec.position_units})"]
    lines.append(
        f"tracked {rec.tracked_fraction:.1%} of samples, longest gap "
        f"{_quantity(rec.longest_gap, 's')}"
    )
    if rec.positions is not None:
        unit = rec._repr_unit("median_speed") or "arbitrary units/s"
        fastest = rec._speed_quantiles[3]
        lines.append(
            f"speed median {_quantity(rec.median_speed, unit)}; 95th percentile "
            f"{_quantity(rec.speed_p95, unit)}; fastest step "
            f"{_quantity(fastest, unit)}"
        )
    if rec.video is not None:
        placed = (
            "calibrated"
            if rec.video.calibration is not None
            else "no calibration, so it is not drawn"
        )
        lines.append(
            f"video: {len(rec.video.frame_times)} frames, "
            f"{rec.video.size_bytes / 1e6:.3g} MB, fetched on request; {placed}"
            " -> rec.animation(video=True)"
        )
    if rec.headings is None:
        lines.append("head direction: not in this dataset")
    else:
        lines.append(
            f"head direction covers {heading_coverage(rec.headings):.0%} of "
            f"{HEADING_BIN_DEGREES}° bins ({rec.heading_source})"
        )
    return lines


def _quantity(value: float, unit: str) -> str:
    """``value`` to three significant figures and its unit; ``"nan"`` alone."""
    return f"{value:.3g} {unit}" if np.isfinite(value) else f"{value:.3g}"


def _environment(rec: Recording) -> list[str]:
    if rec.env is None:
        return ["none: this dataset has no positions"]
    return [f"{rec.env_kind}: {rec.env_source}", "-> rec.env.plot()"]


def _neurons(rec: Recording) -> list[str]:
    units = rec.units
    regions = ", ".join(rec.brain_regions) or "brain region not stated"
    n_fire = int(units["fires_in_epochs"].sum())
    n_silent = rec.n_units - n_fire
    return [
        f"{rec.n_units} units ({regions})",
        f"{n_fire} {'fires' if n_fire == 1 else 'fire'} in the analysed epochs; "
        f"{n_silent} {'has' if n_silent == 1 else 'have'} no spikes there",
        f"area per unit: {_counts(units['brain_region'])}",
        f"authors' labels: {_counts(units['label'])}",
    ]


def _counts(column: pd.Series) -> str:
    """``"CA1 120, CA3 40"`` for a unit column; ``"not in the file"`` if empty."""
    counts = column.dropna().value_counts(sort=True)
    if counts.empty:
        return "not in the file"
    return ", ".join(f"{value} {count}" for value, count in counts.items())


def _watch_out(rec: Recording) -> list[str]:
    title = "Watch out"
    if not rec.verified:
        title += "   Not yet checked by hand. Known pitfalls and automatic findings:"
    lines = [title]
    if not rec.watch_outs:
        lines.append("  none known")
    for item in rec.watch_outs:
        lines.extend(
            textwrap.wrap(
                item, width=_WIDTH, initial_indent="  ! ", subsequent_indent="    "
            )
        )
    return lines
