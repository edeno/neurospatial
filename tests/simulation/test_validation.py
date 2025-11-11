"""Tests for simulation validation module."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.simulation import simulate_session
from neurospatial.simulation.validation import validate_simulation


class TestValidateSimulation:
    """Tests for validate_simulation() function."""

    def test_validate_simulation_with_session(self, simple_2d_env):
        """validate_simulation() should accept SimulationSession."""
        simple_2d_env.units = "cm"

        # Create session with place cells
        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Validate
        result = validate_simulation(session)

        # Check return structure
        assert isinstance(result, dict)
        assert "center_errors" in result
        assert "correlations" in result
        assert "summary" in result
        assert "passed" in result

    def test_validate_simulation_returns_all_fields(self, simple_2d_env):
        """validate_simulation() should return all required fields."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session)

        # Required fields
        assert "center_errors" in result
        assert "correlations" in result
        assert "summary" in result
        assert "passed" in result

        # Check types
        assert isinstance(result["center_errors"], np.ndarray)
        assert isinstance(result["correlations"], np.ndarray)
        assert isinstance(result["summary"], str)
        assert isinstance(result["passed"], bool)

    def test_validate_simulation_center_errors(self, simple_2d_env):
        """validate_simulation() should compute center errors for each cell."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session)

        # Should have one error per cell
        assert len(result["center_errors"]) == 3

        # Non-NaN errors should be non-negative
        valid_errors = result["center_errors"][~np.isnan(result["center_errors"])]
        assert np.all(valid_errors >= 0)

    def test_validate_simulation_correlations(self, simple_2d_env):
        """validate_simulation() should compute correlations between true and detected fields."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session)

        # Should have one correlation per cell
        assert len(result["correlations"]) == 3

        # Non-NaN correlations should be between -1 and 1
        valid_corrs = result["correlations"][~np.isnan(result["correlations"])]
        assert np.all(valid_corrs >= -1)
        assert np.all(valid_corrs <= 1)

    def test_validate_simulation_summary_string(self, simple_2d_env):
        """validate_simulation() should generate summary string."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session)

        # Summary should contain key statistics
        assert "mean" in result["summary"].lower()
        assert "error" in result["summary"].lower()
        assert "correlation" in result["summary"].lower()

    def test_validate_simulation_pass_fail(self, simple_2d_env):
        """validate_simulation() should determine pass/fail."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session)

        # Should return boolean
        assert isinstance(result["passed"], bool)

    def test_validate_simulation_with_thresholds(self, simple_2d_env):
        """validate_simulation() should accept custom thresholds."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Use strict thresholds
        result = validate_simulation(
            session,
            max_center_error=5.0,  # cm
            min_correlation=0.9,
        )

        # Check that thresholds are applied
        assert "passed" in result

    def test_validate_simulation_individual_parameters(self, simple_2d_env):
        """validate_simulation() should accept individual parameters instead of session."""
        simple_2d_env.units = "cm"

        # Create session
        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Validate with individual parameters
        result = validate_simulation(
            env=session.env,
            spike_trains=session.spike_trains,
            positions=session.positions,
            times=session.times,
            ground_truth=session.ground_truth,
        )

        # Should work the same way
        assert "center_errors" in result
        assert "correlations" in result
        assert "passed" in result

    def test_validate_simulation_with_cell_indices(self, simple_2d_env):
        """validate_simulation() should validate only specific cells."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Validate only cells 0, 2, 4
        result = validate_simulation(session, cell_indices=[0, 2, 4])

        # Should only have 3 errors/correlations
        assert len(result["center_errors"]) == 3
        assert len(result["correlations"]) == 3

    def test_validate_simulation_empty_spike_trains(self, simple_2d_env):
        """validate_simulation() should handle cells with no spikes."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=5.0,  # Very short, may have empty trains
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Should not crash even if some cells have no spikes
        result = validate_simulation(session)

        assert "center_errors" in result
        assert "correlations" in result

    def test_validate_simulation_show_plots_false(self, simple_2d_env):
        """validate_simulation() with show_plots=False should not return plots."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session, show_plots=False)

        # Should not have 'plots' key
        assert "plots" not in result or result.get("plots") is None

    def test_validate_simulation_show_plots_true(self, simple_2d_env):
        """validate_simulation() with show_plots=True should return matplotlib figure."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result = validate_simulation(session, show_plots=True)

        # Should have 'plots' key with figure
        assert "plots" in result
        assert result["plots"] is not None

    def test_validate_simulation_place_field_method(self, simple_2d_env):
        """validate_simulation() should accept place field computation method."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Use binned method
        result = validate_simulation(session, method="binned")

        assert "center_errors" in result

    def test_validate_simulation_reproducible(self, simple_2d_env):
        """validate_simulation() should produce same results on same session."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        result1 = validate_simulation(session)
        result2 = validate_simulation(session)

        # Results should be identical
        np.testing.assert_array_equal(
            result1["center_errors"], result2["center_errors"]
        )
        np.testing.assert_array_equal(result1["correlations"], result2["correlations"])

    def test_validate_simulation_invalid_session_type(self):
        """validate_simulation() should raise error for invalid input."""
        with pytest.raises((TypeError, ValueError)):
            validate_simulation("not a session")

    def test_validate_simulation_missing_ground_truth(self, simple_2d_env):
        """validate_simulation() should raise error if ground_truth missing."""
        simple_2d_env.units = "cm"

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            seed=42,
            show_progress=False,
        )

        # Try to validate without ground truth
        with pytest.raises(ValueError, match="ground_truth"):
            validate_simulation(
                env=session.env,
                spike_trains=session.spike_trains,
                positions=session.positions,
                times=session.times,
                # Missing ground_truth
            )
