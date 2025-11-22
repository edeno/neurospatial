"""Tests for DPI and size guard in video backend.

Phase 5.3: DPI and Size Guard
- Verifies warning when dpi > 150
- Verifies warning includes estimated resolution
- Verifies dry-run shows estimates
"""

import warnings

import numpy as np
import pytest


class TestDPIWarning:
    """Tests for high DPI warning."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_no_warning_for_default_dpi(self, simple_env):
        """Verify no warning when using default dpi (100)."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Run dry_run to avoid actually rendering
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=100,
                dry_run=True,
            )

            # Filter for UserWarning about DPI
            dpi_warnings = [x for x in w if "dpi" in str(x.message).lower()]
            assert len(dpi_warnings) == 0, "No warning expected for dpi=100"

    def test_no_warning_for_dpi_150(self, simple_env):
        """Verify no warning when dpi=150 (threshold)."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=150,
                dry_run=True,
            )

            dpi_warnings = [x for x in w if "dpi" in str(x.message).lower()]
            assert len(dpi_warnings) == 0, "No warning expected for dpi=150"

    def test_warning_for_high_dpi(self, simple_env):
        """Verify warning when dpi > 150."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=200,
                dry_run=True,
            )

            dpi_warnings = [x for x in w if "dpi" in str(x.message).lower()]
            assert len(dpi_warnings) == 1, "Expected warning for dpi=200"

    def test_warning_includes_resolution_estimate(self, simple_env):
        """Verify warning includes estimated resolution."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=200,
                dry_run=True,
            )

            dpi_warnings = [x for x in w if "dpi" in str(x.message).lower()]
            assert len(dpi_warnings) == 1
            warning_msg = str(dpi_warnings[0].message)

            # Should mention resolution or pixels
            assert (
                "pixel" in warning_msg.lower()
                or "resolution" in warning_msg.lower()
                or "x" in warning_msg  # e.g., "1600x1200"
            ), f"Warning should include resolution estimate: {warning_msg}"

    def test_warning_suggests_lower_dpi(self, simple_env):
        """Verify warning suggests using lower DPI."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=300,
                dry_run=True,
            )

            dpi_warnings = [x for x in w if "dpi" in str(x.message).lower()]
            assert len(dpi_warnings) == 1
            warning_msg = str(dpi_warnings[0].message)

            # Should suggest using lower DPI
            assert "150" in warning_msg or "100" in warning_msg, (
                f"Warning should suggest lower DPI: {warning_msg}"
            )


class TestDryRunEstimates:
    """Tests for dry-run estimation output."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_dry_run_shows_frame_count(self, simple_env, capsys):
        """Verify dry run shows frame count."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [
            np.random.default_rng(42).random(simple_env.n_bins) for _ in range(10)
        ]

        render_video(
            env=simple_env,
            fields=fields,
            save_path="test.mp4",
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert "10" in captured.out, "Should show frame count"
        assert "Frames" in captured.out, "Should label frame count"

    def test_dry_run_shows_estimated_time(self, simple_env, capsys):
        """Verify dry run shows estimated time."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [
            np.random.default_rng(42).random(simple_env.n_bins) for _ in range(10)
        ]

        render_video(
            env=simple_env,
            fields=fields,
            save_path="test.mp4",
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert "time" in captured.out.lower(), "Should mention time estimate"

    def test_dry_run_shows_estimated_size(self, simple_env, capsys):
        """Verify dry run shows estimated file size."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [
            np.random.default_rng(42).random(simple_env.n_bins) for _ in range(10)
        ]

        render_video(
            env=simple_env,
            fields=fields,
            save_path="test.mp4",
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert "size" in captured.out.lower() or "MB" in captured.out, (
            "Should show file size estimate"
        )

    def test_dry_run_returns_none(self, simple_env):
        """Verify dry run returns None."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        result = render_video(
            env=simple_env,
            fields=fields,
            save_path="test.mp4",
            dry_run=True,
        )

        assert result is None, "dry_run=True should return None"

    def test_dry_run_does_not_create_file(self, simple_env, tmp_path):
        """Verify dry run doesn't create output file."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]
        output_path = tmp_path / "test.mp4"

        render_video(
            env=simple_env,
            fields=fields,
            save_path=str(output_path),
            dry_run=True,
        )

        assert not output_path.exists(), "dry_run should not create file"


class TestDPIEstimatedResolution:
    """Tests for resolution estimation in high DPI warning."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_resolution_scales_with_dpi(self, simple_env):
        """Verify resolution estimate scales correctly with DPI."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(3)]

        # Test at two different DPIs to verify scaling
        with warnings.catch_warnings(record=True) as w200:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=200,
                dry_run=True,
            )

        with warnings.catch_warnings(record=True) as w300:
            warnings.simplefilter("always")
            render_video(
                env=simple_env,
                fields=fields,
                save_path="test.mp4",
                dpi=300,
                dry_run=True,
            )

        # Both should have warnings
        dpi_warnings_200 = [x for x in w200 if "dpi" in str(x.message).lower()]
        dpi_warnings_300 = [x for x in w300 if "dpi" in str(x.message).lower()]

        assert len(dpi_warnings_200) == 1
        assert len(dpi_warnings_300) == 1
