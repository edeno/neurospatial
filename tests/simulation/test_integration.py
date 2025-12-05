"""Integration tests for simulation subpackage.

Tests that verify end-to-end workflows combining multiple simulation components.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, compute_place_field
from neurospatial.simulation import (
    boundary_cell_session,
    grid_cell_session,
    linear_track_session,
    open_field_session,
    simulate_session,
    tmaze_alternation_session,
    validate_simulation,
)


class TestSimulateSessionIntegration:
    """Integration tests for simulate_session() with all cell types."""

    def test_simulate_session_place_cells_end_to_end(self, simple_2d_env):
        """Test complete place cell simulation workflow."""
        simple_2d_env.units = "cm"

        # Create session
        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=10,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Verify structure
        assert session.env is simple_2d_env
        assert len(session.models) == 10
        assert len(session.spike_trains) == 10
        assert len(session.ground_truth) == 10
        assert len(session.positions) > 0
        assert len(session.times) > 0

        # Verify spikes were generated
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0

    @pytest.mark.slow
    def test_simulate_session_boundary_cells_end_to_end(self, simple_2d_env):
        """Test complete boundary cell simulation workflow."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=10.0,
            n_cells=10,
            cell_type="boundary",
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 10
        assert len(session.spike_trains) == 10
        assert len(session.ground_truth) == 10

    def test_simulate_session_grid_cells_end_to_end(self, simple_2d_env):
        """Test complete grid cell simulation workflow."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=10,
            cell_type="grid",
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 10
        assert len(session.spike_trains) == 10
        assert len(session.ground_truth) == 10

    def test_simulate_session_mixed_cells_end_to_end(self, simple_2d_env):
        """Test complete mixed cell simulation workflow."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=10,
            cell_type="mixed",
            seed=42,
            show_progress=False,
        )

        # Should have 6 place (60%), 2 boundary (20%), 2 grid (20%)
        from neurospatial.simulation.models import (
            BoundaryCellModel,
            GridCellModel,
            PlaceCellModel,
        )

        n_place = sum(isinstance(m, PlaceCellModel) for m in session.models)
        n_boundary = sum(isinstance(m, BoundaryCellModel) for m in session.models)
        n_grid = sum(isinstance(m, GridCellModel) for m in session.models)

        assert n_place == 6
        assert n_boundary == 2
        assert n_grid == 2


class TestValidateSimulationIntegration:
    """Integration tests for validate_simulation() with place field detection."""

    def test_validate_simulation_detects_place_fields_correctly(self, simple_2d_env):
        """Test that validate_simulation() detects place fields with good accuracy."""
        simple_2d_env.units = "cm"

        # Create session with place cells
        session = simulate_session(
            simple_2d_env,
            duration=60.0,  # Longer duration for better place field detection
            n_cells=5,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Validate
        result = validate_simulation(session)

        # Should have results for all cells
        assert len(result["center_errors"]) == 5
        assert len(result["correlations"]) == 5

        # Check that at least some cells pass validation
        # (not all will pass due to randomness, but most should)
        valid_errors = result["center_errors"][~np.isnan(result["center_errors"])]
        if len(valid_errors) > 0:
            # Mean error should be reasonable (< 4 bin sizes)
            mean_bin_size = np.mean(simple_2d_env.bin_sizes)
            assert np.mean(valid_errors) < 4 * mean_bin_size

    @pytest.mark.slow
    def test_validate_simulation_with_custom_thresholds(self, simple_2d_env):
        """Test validate_simulation() with custom validation thresholds."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=10.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Use strict thresholds
        result = validate_simulation(
            session,
            max_center_error=5.0,  # cm
            min_correlation=0.8,
        )

        # Should have validation results
        assert "passed" in result
        assert isinstance(result["passed"], bool)


class TestPreConfiguredExamplesIntegration:
    """Integration tests verifying all pre-configured examples run without errors."""

    def test_open_field_session_runs_without_errors(self):
        """Test open_field_session() completes successfully."""
        session = open_field_session(duration=10.0, n_place_cells=5, seed=42)

        assert session is not None
        assert len(session.models) == 5
        assert len(session.spike_trains) == 5

    def test_linear_track_session_runs_without_errors(self):
        """Test linear_track_session() completes successfully."""
        session = linear_track_session(
            duration=10.0, track_length=100.0, n_place_cells=5, n_laps=3, seed=42
        )

        assert session is not None
        assert len(session.models) == 5
        assert len(session.spike_trains) == 5

    def test_tmaze_alternation_session_runs_without_errors(self):
        """Test tmaze_alternation_session() completes successfully."""
        session = tmaze_alternation_session(
            duration=30.0, n_trials=4, n_place_cells=5, seed=42
        )

        assert session is not None
        assert len(session.models) == 5
        assert len(session.spike_trains) == 5
        assert "trial_choices" in session.metadata

    def test_boundary_cell_session_runs_without_errors(self):
        """Test boundary_cell_session() completes successfully."""
        session = boundary_cell_session(
            duration=10.0, n_boundary_cells=3, n_place_cells=2, seed=42
        )

        assert session is not None
        assert len(session.models) == 5  # 3 boundary + 2 place
        assert len(session.spike_trains) == 5

    def test_grid_cell_session_runs_without_errors(self):
        """Test grid_cell_session() completes successfully."""
        session = grid_cell_session(duration=10.0, n_grid_cells=5, seed=42)

        assert session is not None
        assert len(session.models) == 5
        assert len(session.spike_trains) == 5


class TestPlaceFieldDetectionAccuracy:
    """Tests for place field detection accuracy as specified in SIMULATION_PLAN.md."""

    @pytest.mark.slow
    def test_place_field_detection_accuracy(self):
        """Test place field detection pipeline works end-to-end."""
        # Use pre-configured session for reliable test
        # Increased max_rate to 50 Hz allows shorter duration (3x faster)
        session = open_field_session(
            duration=40.0, n_place_cells=5, seed=42, max_rate=50.0
        )

        # Test that we can detect place fields from simulated data
        env = session.env
        positions = session.positions
        times = session.times
        spike_trains = session.spike_trains

        # Try to detect place fields for all cells
        detected_fields = []
        for spike_times in spike_trains:
            # Threshold of 5 spikes ensures reliable detection
            # With 40s duration and 50 Hz max_rate, detectable cells have >5 spikes
            if len(spike_times) > 5:
                # Compute place field
                rate_map = compute_place_field(
                    env, spike_times, times, positions, smoothing_method="diffusion_kde"
                )

                # Find peak (detected center)
                peak_bin = np.argmax(rate_map)
                detected_center = env.bin_centers[peak_bin]
                detected_fields.append(detected_center)

        # Assert that place field detection works for at least some cells
        # With 40s, 50 Hz max_rate, and 5 cells, should detect at least 2 fields
        assert len(detected_fields) >= 2, (
            f"Only detected {len(detected_fields)} fields out of 5 cells. "
            f"Spike counts: {[len(st) for st in spike_trains]}. "
            "Place field detection may not be working."
        )

        # Verify detected centers are within environment bounds
        for detected_center in detected_fields:
            # All coordinates should be within [0, arena_size]
            assert all(0 <= coord <= 110 for coord in detected_center), (
                f"Detected center {detected_center} outside environment bounds"
            )

    def test_place_field_detection_with_longer_duration(self):
        """Test that longer recordings improve detection accuracy."""
        rng = np.random.default_rng(42)
        bin_size = 2.0
        data = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(data, bin_size=bin_size)
        env.units = "cm"

        from neurospatial.simulation.models import PlaceCellModel
        from neurospatial.simulation.spikes import generate_poisson_spikes
        from neurospatial.simulation.trajectory import simulate_trajectory_ou

        # Create known place cell
        center = env.bin_centers[len(env.bin_centers) // 2]
        model = PlaceCellModel(env, center=center, width=15.0, max_rate=30.0)

        # Short duration
        positions_short, times_short = simulate_trajectory_ou(
            env, duration=30.0, seed=42
        )
        rates_short = model.firing_rate(positions_short, times_short)
        spikes_short = generate_poisson_spikes(rates_short, times_short, seed=42)

        # Long duration
        positions_long, times_long = simulate_trajectory_ou(
            env, duration=180.0, seed=42
        )
        rates_long = model.firing_rate(positions_long, times_long)
        spikes_long = generate_poisson_spikes(rates_long, times_long, seed=42)

        # Detect centers
        if len(spikes_short) > 0:
            rate_map_short = compute_place_field(
                env,
                spikes_short,
                times_short,
                positions_short,
                smoothing_method="diffusion_kde",
            )
            detected_short = env.bin_centers[np.argmax(rate_map_short)]
            error_short = np.linalg.norm(center - detected_short)
        else:
            error_short = np.inf

        if len(spikes_long) > 0:
            rate_map_long = compute_place_field(
                env,
                spikes_long,
                times_long,
                positions_long,
                smoothing_method="diffusion_kde",
            )
            detected_long = env.bin_centers[np.argmax(rate_map_long)]
            error_long = np.linalg.norm(center - detected_long)
        else:
            error_long = np.inf

        # Longer duration should have lower error (or at least not worse)
        # Allow some tolerance due to randomness
        assert error_long <= error_short + bin_size


class TestSimulationReproducibility:
    """Integration tests for reproducibility across simulation pipeline."""

    def test_full_pipeline_reproducibility(self, simple_2d_env):
        """Test that entire simulation pipeline is reproducible with same seed."""
        simple_2d_env.units = "cm"

        seed = 42

        # Run twice with same seed
        session1 = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            cell_type="place",
            seed=seed,
            show_progress=False,
        )

        session2 = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            cell_type="place",
            seed=seed,
            show_progress=False,
        )

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

        # Ground truth should be identical
        for key in session1.ground_truth:
            for param_key in session1.ground_truth[key]:
                val1 = session1.ground_truth[key][param_key]
                val2 = session2.ground_truth[key][param_key]
                if isinstance(val1, np.ndarray):
                    np.testing.assert_array_equal(val1, val2)
                else:
                    assert val1 == val2
