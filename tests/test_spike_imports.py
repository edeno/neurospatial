"""Shared spike helpers sit below the encoding, decoding and events packages."""

import subprocess
import sys

import pytest


def _loaded_after(statement: str, forbidden: tuple[str, ...]) -> str:
    """Run ``statement`` in a fresh interpreter and fail if a package loaded."""
    code = f"""
import sys
import numpy as np
{statement}
for prefix in {forbidden!r}:
    assert not any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    ), prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    return result.stderr if result.returncode else ""


@pytest.mark.parametrize("module", ["neurospatial._spikes", "neurospatial._time_bins"])
def test_shared_spike_helpers_import_no_analysis_package(module):
    error = _loaded_after(
        f"import importlib; importlib.import_module({module!r})",
        ("neurospatial.encoding", "neurospatial.decoding", "neurospatial.events"),
    )
    assert not error, error


def test_events_import_does_not_load_encoding_or_decoding():
    error = _loaded_after(
        "import neurospatial.events",
        ("neurospatial.encoding", "neurospatial.decoding"),
    )
    assert not error, error


@pytest.mark.parametrize(
    "call",
    [
        "population_peri_event_histogram("
        "[np.array([0.1, 0.4])], [0.5], (-0.5, 0.5), bin_size=0.1)",
        "peri_event_histogram(np.array([0.1, 0.4]), [0.5], (-0.5, 0.5), bin_size=0.1)",
    ],
)
def test_peri_event_histograms_do_not_load_decoding(call):
    error = _loaded_after(
        "from neurospatial.events import "
        f"peri_event_histogram, population_peri_event_histogram\n{call}",
        ("neurospatial.encoding", "neurospatial.decoding"),
    )
    assert not error, error
