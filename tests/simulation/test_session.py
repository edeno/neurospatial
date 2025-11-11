"""Tests for simulation session module."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation.models import PlaceCellModel
from neurospatial.simulation.session import SimulationSession


class TestSimulationSession:
    """Tests for SimulationSession dataclass."""

    def test_simulation_session_is_frozen(self, simple_2d_env):
        """SimulationSession should be immutable (frozen dataclass)."""
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
                np.random.randn(100, 2), bin_size=2.0
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
