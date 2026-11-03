"""The printed summary of a recording."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurospatial.datasets._summary import heading_coverage, summary_text

LOADER_NOTE = (
    "headings are the direction of movement, not of the head: they say "
    "nothing about head scanning"
)

EXPECTED = """\
Synthetic et al. (2026), simulated rat s1, session square (synthetic_2026)
simulated apparatus; CA1. Synthetic 0, CC0-1.0
Cite: Synthetic A, Simulated B (2026) A simulated session. ... -> rec.citation
Time        60 s tracked, in seconds on the simulation's clock, from 0 s;
              position sampled at 100 Hz
Epochs      run (0 to 60 s)
Tracking    simulated head position (2-D, cm)
            tracked 100.0% of samples, longest gap 0.01 s
            speed median 7.06 cm/s; 95th percentile 13.7 cm/s
            head direction covers 100% of 10° bins (direction of movement of
              the simulated path)
Environment declared: 1 m square box (simulation)
            -> rec.env.plot()
Neurons     6 units (CA1)
            5 fire in the analysed epochs; 1 have no spikes there
            area per unit: CA1 6
            authors' labels: pyramidal 6
Watch out   Not yet checked by hand. Known pitfalls and automatic findings:
  ! headings are the direction of movement, not of the head: they say nothing
    about head scanning"""

SECTIONS = ("Time ", "Epochs ", "Tracking ", "Environment ", "Neurons ", "Watch out")


@pytest.fixture
def checked(square_recording):
    trains = list(square_recording.spike_times)
    trains[3] = np.empty(0)
    return dataclasses.replace(
        square_recording, spike_times=tuple(trains), watch_outs=(LOADER_NOTE,)
    )


def test_summary_text(checked):
    text = repr(checked)
    assert text == EXPECTED
    assert text == summary_text(checked)
    starts = [
        next(i for i, line in enumerate(text.splitlines()) if line.startswith(label))
        for label in SECTIONS
    ]
    assert starts == sorted(starts)
    assert text.splitlines()[0].startswith("Synthetic et al. (2026)")
    assert "Watch out" in text.splitlines()[starts[-1]]


def test_verified_session_has_no_unchecked_note(checked):
    verified = dataclasses.replace(checked, verified=True)
    assert "Not yet checked by hand" not in repr(verified)
    assert "Not yet checked by hand" in repr(checked)


def test_heading_only_summary(heading_recording):
    text = repr(heading_recording)
    assert "heading only; no positions in this dataset" in " ".join(text.split())
    assert "Environment none: this dataset has no positions" in text
    assert "speed" not in text
    assert text.endswith(
        "Watch out   Not yet checked by hand. Known pitfalls and "
        "automatic findings:\n  none known"
    )


def test_track_summary_names_missing_headings(track_recording):
    text = repr(track_recording)
    assert "head direction: not in this dataset" in text
    assert "Epochs      none named in this file" in text


def test_missing_unit_columns_say_so(square_recording):
    units = square_recording.units
    units["brain_region"] = None
    rec = dataclasses.replace(square_recording, units=units)
    assert "area per unit: not in the file" in repr(rec)


def test_html_repr_is_the_text(checked):
    assert checked._repr_html_().startswith("<pre>Synthetic et al. (2026)")


@pytest.mark.parametrize(
    ("headings", "coverage"),
    [
        (np.linspace(-np.pi, np.pi, 1000), 1.0),
        (np.full(10, 0.05), 1 / 36),
        (np.array([np.nan, 0.05, -np.pi, np.pi]), 2 / 36),
        (np.array([np.nan]), 0.0),
    ],
)
def test_heading_coverage(headings, coverage):
    assert heading_coverage(headings) == pytest.approx(coverage)
