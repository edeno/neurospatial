"""Record deterministic affine capability and numerical probes as JSON.

Run with ``uv run python scripts/environment_affine_baseline.py > base.json``.
Repeat on a later phase and compare the files. Timings and process peak RSS are
observations, not performance guarantees; RSS is in bytes on macOS.
"""

from __future__ import annotations

import json
import resource
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.ops.transforms import AffineND


def record() -> dict[str, object]:
    """Probe a full grid with exact, deterministic affine maps."""
    start = time.perf_counter()
    env = Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool),
        grid_edges=(np.arange(4.0), np.arange(5.0)),
        connect_diagonal_neighbors=True,
    )
    field = 2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
    query = np.array([[1.25, 1.75]])
    result: dict[str, object] = {}
    for name, linear in (
        ("native", np.eye(2)),
        ("rotation", np.array([[0.0, -1.0], [1.0, 0.0]])),
        ("reflection", np.diag([-1.0, 1.0])),
        ("shear", np.array([[1.0, 0.3], [0.0, 1.0]])),
    ):
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        transform = AffineND(matrix)
        child = env if name == "native" else env.apply_transform(transform)
        values: dict[str, object] = {
            "ids": child.bin_at(transform(env.bin_centers)).tolist(),
            "sizes": child.bin_sizes.tolist(),
            "faces": child._binning.faces.available,
        }
        try:
            values["linear"] = child.interpolate(
                field, transform(query), mode="linear"
            ).tolist()
        except NotImplementedError as exc:
            values["linear"] = type(exc).__name__
        if child._binning.faces.available:
            values["smooth"] = child.smooth(field, bandwidth=0.75).tolist()
        ax = child.plot_field(field, colorbar=False)
        values["renderer"] = [type(item).__name__ for item in ax.collections]
        plt.close(ax.figure)
        result[name] = values
    result["seconds"] = time.perf_counter() - start
    result["peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


if __name__ == "__main__":
    print(json.dumps(record(), indent=2))
