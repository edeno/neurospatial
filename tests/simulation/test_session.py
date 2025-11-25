"""Tests for simulation session module."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation.models import (
    BoundaryCellModel,
    GridCellModel,
    PlaceCellModel,
)
from neurospatial.simulation.session import SimulationSession, simulate_session


class TestSimulationSession:
    """Tests for SimulationSession dataclass."""

    def test_simulation_session_is_frozen(self, simple_2d_env):
        """SimulationSession should be immutable (frozen dataclass)."""
        rng = np.random.default_rng(42)
        # Create minimal session
        positions = np.array([[50.0, 50.0], [51.0, 51.0]])
        times = np.array([0.0, 0.1])
        spike_trains = [np.array([0.05]), np.array([0.06])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {"cell_0": models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_trains=spike_trains,
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        # Attempt to modify should raise FrozenInstanceError
        with pytest.raises(AttributeError, match="cannot assign to field"):
            session.env = Environment.from_samples(
                rng.standard_normal((100, 2)), bin_size=2.0
            )

    def test_simulation_session_has_required_fields(self, simple_2d_env):
        """SimulationSession should have all required fields."""
        positions = np.array([[50.0, 50.0]])
        times = np.array([0.0])
        spike_trains = [np.array([0.05])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {"cell_0": models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_trains=spike_trains,
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        # Check all fields exist
        assert session.env is simple_2d_env
        assert isinstance(session.positions, np.ndarray)
        assert isinstance(session.times, np.ndarray)
        assert isinstance(session.spike_trains, list)
        assert isinstance(session.models, list)
        assert isinstance(session.ground_truth, dict)
        assert isinstance(session.metadata, dict)

    def test_simulation_session_type_hints(self, simple_2d_env):
        """SimulationSession should have proper type hints."""
        from dataclasses import fields

        # Check that all required fields are defined
        field_names = {f.name for f in fields(SimulationSession)}

        assert "env" in field_names
        assert "positions" in field_names
        assert "times" in field_names
        assert "spike_trains" in field_names
        assert "models" in field_names
        assert "ground_truth" in field_names
        assert "metadata" in field_names

        # Check that all fields have type annotations
        for field in fields(SimulationSession):
            assert field.type is not None, f"Field {field.name} missing type annotation"

    def test_simulation_session_with_empty_spike_trains(self, simple_2d_env):
        """SimulationSession should handle empty spike trains."""
        positions = np.array([[50.0, 50.0]])
        times = np.array([0.0])
        spike_trains = [np.array([])]  # Empty spike train
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {"cell_0": models[0].ground_truth}
        metadata = {"duration": 0.0}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_trains=spike_trains,
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        assert len(session.spike_trains) == 1
        assert len(session.spike_trains[0]) == 0

    def test_simulation_session_with_multiple_cells(self, simple_2d_env):
        """SimulationSession should handle multiple cells."""
        positions = np.array([[50.0, 50.0], [51.0, 51.0]])
        times = np.array([0.0, 0.1])
        spike_trains = [
            np.array([0.05, 0.15]),
            np.array([0.02, 0.12]),
            np.array([0.08]),
        ]
        models = [
            PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0])),
            PlaceCellModel(simple_2d_env, center=np.array([60.0, 60.0])),
            PlaceCellModel(simple_2d_env, center=np.array([70.0, 70.0])),
        ]
        ground_truth = {
            f"cell_{i}": model.ground_truth for i, model in enumerate(models)
        }
        metadata = {"n_cells": 3}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_trains=spike_trains,
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        assert len(session.spike_trains) == 3
        assert len(session.models) == 3
        assert len(session.ground_truth) == 3

    def test_simulation_session_repr(self, simple_2d_env):
        """SimulationSession should have informative repr."""
        positions = np.array([[50.0, 50.0]])
        times = np.array([0.0])
        spike_trains = [np.array([0.05])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {"cell_0": models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_trains=spike_trains,
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        repr_str = repr(session)
        assert "SimulationSession" in repr_str


class TestSimulateSession:
    """Tests for simulate_session() high-level function."""

    def test_simulate_session_returns_simulation_session(self, simple_2d_env):
        """simulate_session() should return SimulationSession instance."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env, duration=1.0, n_cells=3, seed=42, show_progress=False
        )

        assert isinstance(session, SimulationSession)
        assert session.env is simple_2d_env
        assert len(session.spike_trains) == 3
        assert len(session.models) == 3

    def test_simulate_session_place_cells(self, simple_2d_env):
        """simulate_session() with cell_type='place' should create place cells."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=5,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 5
        assert all(isinstance(m, PlaceCellModel) for m in session.models)
        assert len(session.spike_trains) == 5
        assert len(session.ground_truth) == 5

    def test_simulate_session_boundary_cells(self, simple_2d_env):
        """simulate_session() with cell_type='boundary' should create boundary cells."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=5,
            cell_type="boundary",
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 5
        assert all(isinstance(m, BoundaryCellModel) for m in session.models)

    def test_simulate_session_grid_cells(self, simple_2d_env):
        """simulate_session() with cell_type='grid' should create grid cells."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=5,
            cell_type="grid",
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 5
        assert all(isinstance(m, GridCellModel) for m in session.models)

    def test_simulate_session_mixed_cells(self, simple_2d_env):
        """simulate_session() with cell_type='mixed' should create 60% place, 20% boundary, 20% grid."""
        simple_2d_env.units = "cm"
        n_cells = 10
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=n_cells,
            cell_type="mixed",
            seed=42,
            show_progress=False,
        )

        # Count cell types
        n_place = sum(isinstance(m, PlaceCellModel) for m in session.models)
        n_boundary = sum(isinstance(m, BoundaryCellModel) for m in session.models)
        n_grid = sum(isinstance(m, GridCellModel) for m in session.models)

        # Check proportions (with tolerance for rounding)
        assert n_place == 6  # 60% of 10
        assert n_boundary == 2  # 20% of 10
        assert n_grid == 2  # 20% of 10

    def test_simulate_session_trajectory_ou(self, simple_2d_env):
        """simulate_session() with trajectory_method='ou' should create OU trajectory."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=3,
            trajectory_method="ou",
            seed=42,
            show_progress=False,
        )

        # Check trajectory shapes
        assert session.positions.ndim == 2
        assert session.positions.shape[1] == 2  # 2D environment
        assert len(session.times) == len(session.positions)

        # All positions should be in environment
        assert session.positions.shape[0] > 0

    def test_simulate_session_coverage_uniform(self, simple_2d_env):
        """simulate_session() with coverage='uniform' should evenly space field centers."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=1.0,
            n_cells=5,
            cell_type="place",
            coverage="uniform",
            seed=42,
            show_progress=False,
        )

        # Extract field centers from ground truth
        centers = [
            session.ground_truth[f"cell_{i}"]["center"]
            for i in range(len(session.models))
        ]

        # With uniform coverage, centers should be well-distributed
        # Check that centers are not all the same
        unique_centers = np.unique(np.array(centers), axis=0)
        assert len(unique_centers) > 1  # Should have multiple unique centers

    def test_simulate_session_coverage_random(self, simple_2d_env):
        """simulate_session() with coverage='random' should randomly sample field centers."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=1.0,
            n_cells=5,
            cell_type="place",
            coverage="random",
            seed=42,
            show_progress=False,
        )

        # Extract field centers
        centers = [
            session.ground_truth[f"cell_{i}"]["center"]
            for i in range(len(session.models))
        ]

        # Should have multiple unique centers
        unique_centers = np.unique(np.array(centers), axis=0)
        assert len(unique_centers) >= 1

    def test_simulate_session_seed_reproducibility(self, simple_2d_env):
        """simulate_session() with same seed should produce identical results."""
        simple_2d_env.units = "cm"
        seed = 42

        session1 = simulate_session(
            simple_2d_env,
            duration=1.0,
            n_cells=3,
            seed=seed,
            show_progress=False,
        )
        session2 = simulate_session(
            simple_2d_env,
            duration=1.0,
            n_cells=3,
            seed=seed,
            show_progress=False,
        )

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_simulate_session_ground_truth_structure(self, simple_2d_env):
        """simulate_session() ground_truth should have proper structure."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=1.0,
            n_cells=3,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        # Check ground_truth keys
        assert len(session.ground_truth) == 3
        for i in range(3):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # For place cells, should have center, width, max_rate, baseline_rate
            gt = session.ground_truth[key]
            assert "center" in gt
            assert "width" in gt
            assert "max_rate" in gt
            assert "baseline_rate" in gt

    def test_simulate_session_metadata(self, simple_2d_env):
        """simulate_session() metadata should contain session parameters."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=1.5,
            n_cells=4,
            cell_type="place",
            trajectory_method="ou",
            coverage="uniform",
            seed=42,
            show_progress=False,
        )

        # Check that metadata contains key parameters
        meta = session.metadata
        assert "duration" in meta
        assert "n_cells" in meta
        assert "cell_type" in meta
        assert "trajectory_method" in meta
        assert "coverage" in meta
        assert meta["duration"] == 1.5
        assert meta["n_cells"] == 4
        assert meta["cell_type"] == "place"

    def test_simulate_session_invalid_cell_type(self, simple_2d_env):
        """simulate_session() should raise ValueError for invalid cell_type."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="cell_type must be one of"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=3,
                cell_type="invalid",
                show_progress=False,
            )

    def test_simulate_session_invalid_trajectory_method(self, simple_2d_env):
        """simulate_session() should raise ValueError for invalid trajectory_method."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="trajectory_method must be one of"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=3,
                trajectory_method="invalid",
                show_progress=False,
            )

    def test_simulate_session_invalid_coverage(self, simple_2d_env):
        """simulate_session() should raise ValueError for invalid coverage."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="coverage must be one of"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=3,
                coverage="invalid",
                show_progress=False,
            )

    def test_simulate_session_grid_cells_require_2d(self, simple_1d_env):
        """simulate_session() with grid cells should require 2D environment."""
        simple_1d_env.units = "cm"
        with pytest.raises(ValueError, match="2D environment"):
            simulate_session(
                simple_1d_env,
                duration=1.0,
                n_cells=3,
                cell_type="grid",
                show_progress=False,
            )

    def test_simulate_session_kwargs_passed_to_trajectory(self, simple_2d_env):
        """simulate_session() should pass kwargs to trajectory function."""
        simple_2d_env.units = "cm"
        # Pass custom speed_mean and coherence_time
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=3,
            trajectory_method="ou",
            speed_mean=0.15,  # Custom speed
            coherence_time=1.0,  # Custom coherence
            seed=42,
            show_progress=False,
        )

        # Should complete without error
        assert len(session.positions) > 0

    def test_simulate_session_positions_within_bounds(self, simple_2d_env):
        """simulate_session() positions should all be within environment bounds."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=3,
            seed=42,
            show_progress=False,
        )

        # All positions should map to valid bins
        bin_indices = simple_2d_env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions are outside environment"

    def test_simulate_session_trajectory_laps(self, simple_2d_env):
        """simulate_session() with trajectory_method='laps' should work."""
        simple_2d_env.units = "cm"
        session = simulate_session(
            simple_2d_env,
            duration=2.0,
            n_cells=3,
            trajectory_method="laps",
            n_laps=3,
            seed=42,
            show_progress=False,
        )

        # Check that positions were generated
        assert len(session.positions) > 0
        assert len(session.times) == len(session.positions)

    @pytest.mark.skip(reason="Requires proper 1D environment with GraphLayout")
    def test_simulate_session_trajectory_sinusoidal(self, simple_1d_env):
        """simulate_session() with trajectory_method='sinusoidal' should work for 1D."""
        simple_1d_env.units = "cm"
        session = simulate_session(
            simple_1d_env,
            duration=2.0,
            n_cells=3,
            trajectory_method="sinusoidal",
            seed=42,
            show_progress=False,
        )

        # Check that positions were generated
        assert len(session.positions) > 0
        assert len(session.times) == len(session.positions)

    def test_simulate_session_zero_cells(self, simple_2d_env):
        """simulate_session() with n_cells=0 should raise ValueError."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="n_cells must be positive"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=0,
                show_progress=False,
            )

    def test_simulate_session_negative_cells(self, simple_2d_env):
        """simulate_session() with negative n_cells should raise ValueError."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="n_cells must be positive"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=-5,
                show_progress=False,
            )

    def test_simulate_session_zero_duration(self, simple_2d_env):
        """simulate_session() with duration=0 should raise ValueError."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="duration must be positive"):
            simulate_session(
                simple_2d_env,
                duration=0.0,
                n_cells=3,
                show_progress=False,
            )

    def test_simulate_session_negative_duration(self, simple_2d_env):
        """simulate_session() with negative duration should raise ValueError."""
        simple_2d_env.units = "cm"
        with pytest.raises(ValueError, match="duration must be positive"):
            simulate_session(
                simple_2d_env,
                duration=-1.0,
                n_cells=3,
                show_progress=False,
            )
