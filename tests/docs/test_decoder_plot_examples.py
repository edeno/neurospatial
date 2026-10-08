"""Tutorial posterior overlays use the decoder's spatial and temporal axes."""

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    "examples/20_bayesian_decoding.py",
    "examples/20_bayesian_decoding.ipynb",
    "docs/examples/20_bayesian_decoding.py",
    "docs/examples/20_bayesian_decoding.ipynb",
]


def _posterior_cell(path, actual_name):
    source = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".ipynb"):
        cells = [
            "".join(cell["source"])
            for cell in json.loads(source)["cells"]
            if cell["cell_type"] == "code"
        ]
    else:
        # Percent-format markdown cells contain only comments, so their empty
        # syntax trees cannot match the plotting cell selected below.
        cells = re.split(r"^# %%[^\n]*$", source, flags=re.MULTILINE)
    for cell_source in cells:
        tree = ast.parse(cell_source)
        uses_actual = any(
            isinstance(node, ast.Name) and node.id == actual_name
            for node in ast.walk(tree)
        )
        plots_result = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "result"
            and node.func.attr == "plot"
            for node in ast.walk(tree)
        )
        if uses_actual and plots_result:
            return cell_source
    raise AssertionError(f"No posterior overlay cell for {actual_name} in {path}")


@pytest.mark.parametrize("path", EXAMPLES)
@pytest.mark.parametrize("actual_name", ["actual_track", "actual_positions"])
@pytest.mark.parametrize("gapped", [False, True], ids=["continuous", "gapped"])
def test_tutorial_actual_overlay_matches_perfect_posterior(path, actual_name, gapped):
    env = Environment.from_samples(
        np.linspace(0, 100, 21)[:, None], bin_size=5.0, units="cm"
    )
    # Exact known physical positions correspond to bins 4, 16, 8 and 12.
    actual = np.array([[20.0], [80.0], [40.0], [60.0]])
    expected_bins = np.array([4, 16, 8, 12])
    np.testing.assert_array_equal(env.bin_at(actual), expected_bins)
    times = (
        np.array([10.05, 10.15, 20.05, 20.15]) if gapped else 10.05 + np.arange(4) * 0.1
    )
    posterior = np.zeros((len(times), env.n_bins))
    posterior[np.arange(len(times)), expected_bins] = 1.0
    result = DecodingResult(posterior, env, times)
    namespace = {
        "np": np,
        "plt": plt,
        "env": env,
        "result": result,
        actual_name: actual,
        "time_bin_centers": times,
        "dt": 0.1,
        "COLORS": {"cyan": "cyan"},
    }
    try:
        exec(compile(_posterior_cell(path, actual_name), path, "exec"), namespace)
        ax = namespace["ax"]
        overlay = ax.lines[-1]
        expected_x = np.arange(len(times)) if gapped else times
        np.testing.assert_array_equal(overlay.get_ydata(), expected_bins)
        np.testing.assert_array_equal(overlay.get_xdata(), expected_x)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected_bins)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), expected_x)
        assert "bin" in ax.get_ylabel().lower()
        assert ax.get_xlim()[0] <= expected_x[0]
        assert ax.get_xlim()[1] >= expected_x[-1]
        # Physical outputs and source arrays stay in centimeters on the original clock.
        np.testing.assert_array_equal(result.map_position, actual)
        np.testing.assert_array_equal(result.posterior, posterior)
        np.testing.assert_array_equal(result.times, times)
    finally:
        plt.close("all")
