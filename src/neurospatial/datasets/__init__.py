"""Public spatial neuroscience datasets, downloaded once and checked.

``catalog()`` lists every session of every dataset; ``load(dataset)`` gives
one session (the dataset's default unless ``session=`` names another), and
``fetch_all(dataset)`` downloads them all. A loaded session is a
:class:`Recording`: the arrays neurospatial's functions take (``times``,
``positions``, ``spike_times``, ...), the conversions the loader applied
(:class:`Conversion`), where the file came from (:class:`Provenance`), and
every known pitfall of the session. Where the dataset publishes the
session's video, ``rec.video`` is a :class:`VideoSource`: the file, its
frame times and its pixel-to-cm calibration, fetched only on request. Where
it publishes the LFP, ``rec.lfp`` is an :class:`LFPSource`: where it is, what
each channel is, and its unit and clock, also fetched only on request. Where it keeps every detected spike's waveform,
``rec.marks`` is a :class:`SpikeMarks`: each spike's time, amplitudes and
cluster, the input of a clusterless decoder. Printing a recording shows its full summary. A
session whose spikes, events or epochs disagree with the tracking clock, or
whose positions disagree with the stated apparatus, raises
:class:`DatasetCheckError` instead.

Importing this module needs no optional dependency; the loaders ask for the
``datasets`` extra (``pip install 'neurospatial[datasets]'``) when they need it.
"""

from __future__ import annotations

from neurospatial.datasets._catalog import catalog
from neurospatial.datasets._errors import DatasetCheckError, DownloadWarning
from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._load import fetch_all, load
from neurospatial.datasets._marks import SpikeMarks
from neurospatial.datasets._recording import Conversion, Provenance, Recording
from neurospatial.datasets._video import VideoSource

__all__ = [
    "Conversion",
    "DatasetCheckError",
    "DownloadWarning",
    "LFPSource",
    "Provenance",
    "Recording",
    "SpikeMarks",
    "VideoSource",
    "catalog",
    "fetch_all",
    "load",
]


def __dir__() -> list[str]:
    """List the public names."""
    return sorted(__all__)
