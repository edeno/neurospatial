"""The real-viewer benchmark's lazy workload and reporting remain reproducible.

Native viewer runs are intentionally performed through the script, with a real
display/GPU, rather than replacing the rendering paths with test doubles.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def scene_benchmark():
    path = Path(__file__).parents[2] / "scripts" / "benchmark_animation_scenes.py"
    spec = importlib.util.spec_from_file_location(
        "scene_benchmark_animation_scenes", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lazy_workload_is_reproducible_and_counts_actual_rows(scene_benchmark):
    centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    first = scene_benchmark.CountingFieldSource(centers, frames=100_000, seed=42)
    second = scene_benchmark.CountingFieldSource(centers, frames=100_000, seed=42)
    assert first.reads == second.reads == 0
    np.testing.assert_equal(first[71], second[71])
    assert first.reads == second.reads == 1
    np.testing.assert_equal(first[2:4], np.stack([second[2], second[3]]))
    assert first.reads == second.reads == 3
    assert first[:0].shape == (0, 4)
    assert first.reads == 3
    assert np.isnan(first[0][0])


@pytest.mark.parametrize("dimensions", [2, 3])
def test_same_scene_workload_stays_lazy_in_both_dimensions(scene_benchmark, dimensions):
    args = scene_benchmark.parser().parse_args(
        [
            "--backend",
            "napari",
            "--dimensions",
            str(dimensions),
            "--grid-size",
            "4",
            "--frames",
            "12",
            "--panels",
        ]
    )
    scene, source = scene_benchmark.make_scene(args)
    with scene:
        assert source.reads == 0
        assert scene.env.n_bins == 4**dimensions - 1
        assert scene.frame_times[0] == 100
        row = scene.fields["field"].sample(float(scene.frame_times[3]))
        assert row.shape == (scene.env.n_bins,)
        assert source.reads == 1
        assert np.isnan(row[0])


def test_latency_report_contains_percentiles_without_playback_fps(scene_benchmark):
    result = scene_benchmark.timing_summary([1, 2, 3, 4, 100])
    assert result["count"] == 5
    assert result["median_ms"] == 3
    assert result["p99_ms"] > result["p95_ms"] > result["median_ms"]
    assert "fps" not in result
    assert scene_benchmark.timing_summary([])["median_ms"] is None
