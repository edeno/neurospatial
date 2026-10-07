"""The README's simulated place field recovers its printed ground truth."""

import re

import numpy as np

from .test_executable_docs import _run_document


def test_readme_simulation_recovers_printed_peak(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _run_document("README.md", opt_in=False, seeded=False)
    output = capsys.readouterr().out
    true_match = re.search(r"True field center: \[([^\]]+)\]", output)
    peak_match = re.search(r"Detected peak: \[([^\]]+)\]", output)
    assert true_match and peak_match, output
    center = np.fromstring(true_match[1].replace(",", " "), sep=" ")
    peak = np.fromstring(peak_match[1].replace(",", " "), sep=" ")
    # Recover the known synthetic center to within one 5 cm demonstration bin.
    np.testing.assert_allclose(peak, center, rtol=0, atol=5.0)
