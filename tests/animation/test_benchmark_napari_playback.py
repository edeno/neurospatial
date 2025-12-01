"""Tests for scripts/benchmark_napari_playback.py.

This test module validates the benchmark script:
- Command-line argument parsing
- Synthetic data generation
- Overlay selection functionality
- Timing metrics output
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Path to the benchmark script
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
BENCHMARK_SCRIPT = SCRIPTS_DIR / "benchmark_napari_playback.py"


class TestBenchmarkScriptExists:
    """Tests that the benchmark script exists and can be imported."""

    def test_script_exists(self):
        """Benchmark script file should exist."""
        assert BENCHMARK_SCRIPT.exists(), (
            f"Expected benchmark script at {BENCHMARK_SCRIPT}"
        )

    def test_script_is_python(self):
        """Script should be valid Python (syntax check)."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(BENCHMARK_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script has syntax errors: {result.stderr}"


class TestBenchmarkArgumentParsing:
    """Tests for command-line argument parsing."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_has_argument_parser(self, benchmark_module):
        """Module should have create_argument_parser or parse_args function."""
        assert hasattr(benchmark_module, "create_argument_parser") or hasattr(
            benchmark_module, "main"
        ), "Module should have argument parsing functionality"

    def test_overlay_selection_args(self, benchmark_module):
        """Parser should support overlay selection arguments."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            # Parse with all overlay flags
            args = parser.parse_args(
                [
                    "--position",
                    "--bodyparts",
                    "--head-direction",
                    "--events",
                    "--timeseries",
                ]
            )

            assert args.position is True
            assert args.bodyparts is True
            assert args.head_direction is True
            assert args.events is True
            assert args.timeseries is True
        else:
            pytest.skip("create_argument_parser not found")

    def test_frames_argument(self, benchmark_module):
        """Parser should support --frames argument."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            args = parser.parse_args(["--frames", "500"])
            assert args.frames == 500
        else:
            pytest.skip("create_argument_parser not found")

    def test_playback_frames_argument(self, benchmark_module):
        """Parser should support --playback-frames argument."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            args = parser.parse_args(["--playback-frames", "50"])
            assert args.playback_frames == 50
        else:
            pytest.skip("create_argument_parser not found")

    def test_all_overlays_flag(self, benchmark_module):
        """Parser should support --all-overlays flag."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            args = parser.parse_args(["--all-overlays"])
            assert args.all_overlays is True
        else:
            pytest.skip("create_argument_parser not found")


class TestBenchmarkDataGeneration:
    """Tests for synthetic data generation."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_generate_benchmark_data_function(self, benchmark_module):
        """Module should have generate_benchmark_data function."""
        assert hasattr(benchmark_module, "generate_benchmark_data"), (
            "Module should have generate_benchmark_data function"
        )

    def test_generate_benchmark_data_returns_dict(self, benchmark_module):
        """generate_benchmark_data should return a dict with expected keys."""
        if not hasattr(benchmark_module, "generate_benchmark_data"):
            pytest.skip("generate_benchmark_data not found")

        result = benchmark_module.generate_benchmark_data(
            n_frames=50,
            grid_size=20,
            seed=42,
        )

        assert isinstance(result, dict)
        assert "env" in result
        assert "fields" in result
        assert "frame_times" in result

    def test_generate_benchmark_data_correct_shapes(self, benchmark_module):
        """Generated data should have correct shapes."""
        if not hasattr(benchmark_module, "generate_benchmark_data"):
            pytest.skip("generate_benchmark_data not found")

        n_frames = 50
        result = benchmark_module.generate_benchmark_data(
            n_frames=n_frames,
            grid_size=20,
            seed=42,
        )

        assert result["fields"].shape[0] == n_frames
        assert len(result["frame_times"]) == n_frames


class TestOverlayCreation:
    """Tests for overlay creation based on flags."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_create_selected_overlays_function(self, benchmark_module):
        """Module should have create_selected_overlays function."""
        assert hasattr(benchmark_module, "create_selected_overlays"), (
            "Module should have create_selected_overlays function"
        )

    def test_position_overlay_creation(self, benchmark_module):
        """Should create PositionOverlay when position=True."""
        if not hasattr(benchmark_module, "create_selected_overlays"):
            pytest.skip("create_selected_overlays not found")

        from neurospatial import Environment
        from neurospatial.animation.overlays import PositionOverlay

        # Create a simple environment
        positions = np.random.rand(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=1.0)

        overlays = benchmark_module.create_selected_overlays(
            env=env,
            n_frames=50,
            seed=42,
            position=True,
            bodyparts=False,
            head_direction=False,
            events=False,
            timeseries=False,
        )

        position_overlays = [o for o in overlays if isinstance(o, PositionOverlay)]
        assert len(position_overlays) == 1

    def test_event_overlay_creation(self, benchmark_module):
        """Should create EventOverlay when events=True."""
        if not hasattr(benchmark_module, "create_selected_overlays"):
            pytest.skip("create_selected_overlays not found")

        from neurospatial import Environment
        from neurospatial.animation.overlays import EventOverlay

        # Create a simple environment
        positions = np.random.rand(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=1.0)

        overlays = benchmark_module.create_selected_overlays(
            env=env,
            n_frames=50,
            seed=42,
            position=False,
            bodyparts=False,
            head_direction=False,
            events=True,
            timeseries=False,
        )

        event_overlays = [o for o in overlays if isinstance(o, EventOverlay)]
        assert len(event_overlays) == 1

    def test_timeseries_overlay_creation(self, benchmark_module):
        """Should create TimeSeriesOverlay when timeseries=True."""
        if not hasattr(benchmark_module, "create_selected_overlays"):
            pytest.skip("create_selected_overlays not found")

        from neurospatial import Environment
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Create a simple environment
        positions = np.random.rand(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=1.0)

        overlays = benchmark_module.create_selected_overlays(
            env=env,
            n_frames=50,
            seed=42,
            position=False,
            bodyparts=False,
            head_direction=False,
            events=False,
            timeseries=True,
        )

        ts_overlays = [o for o in overlays if isinstance(o, TimeSeriesOverlay)]
        assert len(ts_overlays) == 1


class TestTimingMetricsOutput:
    """Tests for timing metrics output."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_print_timing_metrics_function(self, benchmark_module):
        """Module should have print_timing_metrics function."""
        assert hasattr(benchmark_module, "print_timing_metrics"), (
            "Module should have print_timing_metrics function"
        )

    def test_timing_metrics_output_format(self, benchmark_module, capsys):
        """Timing metrics should be printed in a readable format."""
        if not hasattr(benchmark_module, "print_timing_metrics"):
            pytest.skip("print_timing_metrics not found")

        # Create sample timing data
        timing_data = {
            "setup_time": 0.5,
            "frame_times_ms": [10.0, 12.0, 8.0, 15.0, 11.0],
            "total_frames": 5,
            "overlays_enabled": ["position", "events"],
        }

        benchmark_module.print_timing_metrics(timing_data)

        captured = capsys.readouterr()
        output = captured.out

        # Check that key metrics are printed
        assert "Setup" in output or "setup" in output.lower()
        assert "Frame" in output or "frame" in output.lower()


class TestHelpMessage:
    """Tests for script help message."""

    def test_help_flag(self):
        """Script should support --help flag."""
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_help_includes_overlay_options(self):
        """Help should describe overlay selection options."""
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_text = result.stdout.lower()

        # Should mention overlay options
        assert "position" in help_text
        assert "overlay" in help_text


class TestEdgeCases:
    """Tests for edge cases and default behavior."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_default_overlay_behavior(self, benchmark_module):
        """When no overlays specified, should return empty list."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 20
        env = Environment.from_samples(positions, bin_size=1.0)

        # Call with all flags False (simulates no CLI args)
        overlays = benchmark_module.create_selected_overlays(
            env=env,
            n_frames=50,
            seed=42,
            position=False,
            bodyparts=False,
            head_direction=False,
            events=False,
            timeseries=False,
        )

        # Should return empty list (main() adds position, not this function)
        assert len(overlays) == 0

    def test_trajectory_boundary_reflection(self, benchmark_module):
        """Trajectory should stay within bounds after reflection."""
        if not hasattr(benchmark_module, "_generate_smooth_trajectory"):
            pytest.skip("_generate_smooth_trajectory not found")

        rng = np.random.default_rng(42)
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]
        n_frames = 100

        trajectory = benchmark_module._generate_smooth_trajectory(
            n_frames, dim_ranges, rng
        )

        # All points must be within bounds
        assert trajectory.shape == (n_frames, 2)
        assert np.all(trajectory[:, 0] >= 0.0)
        assert np.all(trajectory[:, 0] <= 10.0)
        assert np.all(trajectory[:, 1] >= 0.0)
        assert np.all(trajectory[:, 1] <= 10.0)

    def test_headless_mode_argument(self, benchmark_module):
        """Parser should support --headless mode."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            args = parser.parse_args(["--headless"])
            assert args.headless is True
        else:
            pytest.skip("create_argument_parser not found")

    def test_no_playback_argument(self, benchmark_module):
        """Parser should support --no-playback mode."""
        if hasattr(benchmark_module, "create_argument_parser"):
            parser = benchmark_module.create_argument_parser()
            args = parser.parse_args(["--no-playback"])
            assert args.no_playback is True
        else:
            pytest.skip("create_argument_parser not found")

    def test_timing_metrics_dataclass(self, benchmark_module):
        """TimingMetrics should be a valid dataclass with expected fields."""
        if hasattr(benchmark_module, "TimingMetrics"):
            metrics = benchmark_module.TimingMetrics(
                setup_time_s=0.5,
                frame_times_ms=[10.0, 12.0, 15.0],
                total_frames=3,
                overlays_enabled=["position", "events"],
            )

            assert metrics.setup_time_s == 0.5
            assert len(metrics.frame_times_ms) == 3
            assert metrics.total_frames == 3
            assert "position" in metrics.overlays_enabled
        else:
            pytest.skip("TimingMetrics not found")


class TestTimerContextManager:
    """Tests for the timer context manager."""

    @pytest.fixture
    def benchmark_module(self):
        """Import the benchmark module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_napari_playback", BENCHMARK_SCRIPT
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load benchmark script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_napari_playback"] = module
        spec.loader.exec_module(module)
        return module

    def test_timer_measures_elapsed_time(self, benchmark_module):
        """Timer should measure elapsed time correctly."""
        if not hasattr(benchmark_module, "timer"):
            pytest.skip("timer not found")

        import time

        with benchmark_module.timer() as t:
            time.sleep(0.01)  # Sleep 10ms

        # Should measure at least 10ms (with some tolerance)
        assert t["elapsed"] >= 0.009  # 9ms tolerance
        assert t["elapsed"] < 0.1  # Should be well under 100ms


class TestScriptIntegration:
    """Integration tests for full script execution."""

    @pytest.mark.slow
    def test_script_runs_in_headless_mode(self):
        """Script should complete successfully in headless mode."""
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--frames",
                "10",
                "--playback-frames",
                "5",
                "--headless",
                "--position",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should exit successfully
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Should print setup time
        assert "Setup" in result.stdout or "setup" in result.stdout.lower()

    @pytest.mark.slow
    def test_script_with_all_overlays(self):
        """Script should handle --all-overlays flag."""
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--frames",
                "10",
                "--all-overlays",
                "--headless",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
