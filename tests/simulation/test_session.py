"""Tests for simulation session module."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation.models import (
    BorderCellModel,
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
        spike_trains = [np.array([0.05])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {0: models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_times=spike_trains,
            unit_ids=np.arange(len(spike_trains), dtype=np.int64),
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
        positions = np.array([[50.0, 50.0], [51.0, 51.0]])
        times = np.array([0.0, 0.1])
        spike_trains = [np.array([0.05])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {0: models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_times=spike_trains,
            unit_ids=np.arange(len(spike_trains), dtype=np.int64),
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        # Check all fields exist
        assert session.env is simple_2d_env
        assert isinstance(session.positions, np.ndarray)
        assert isinstance(session.times, np.ndarray)
        assert isinstance(session.spike_times, tuple)
        assert isinstance(session.models, tuple)
        assert session.ground_truth[0].keys() == models[0].ground_truth.keys()
        np.testing.assert_array_equal(session.ground_truth[0]["center"], [50.0, 50.0])
        assert dict(session.metadata) == metadata

    def test_simulation_session_type_hints(self, simple_2d_env):
        """SimulationSession should have proper type hints."""
        from dataclasses import fields

        # Check that all required fields are defined
        field_names = {f.name for f in fields(SimulationSession)}

        assert "env" in field_names
        assert "positions" in field_names
        assert "times" in field_names
        assert "spike_times" in field_names
        assert "unit_ids" in field_names
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
        ground_truth = {0: models[0].ground_truth}
        metadata = {"duration": 0.0}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_times=spike_trains,
            unit_ids=np.arange(len(spike_trains), dtype=np.int64),
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        assert len(session.spike_times) == 1
        assert len(session.spike_times[0]) == 0

    def test_simulation_session_with_multiple_cells(self, simple_2d_env):
        """SimulationSession should handle multiple cells."""
        positions = np.array([[50.0, 50.0], [51.0, 51.0]])
        times = np.array([0.0, 0.1])
        spike_trains = [
            np.array([0.05, 0.09]),
            np.array([0.02, 0.07]),
            np.array([0.08]),
        ]
        models = [
            PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0])),
            PlaceCellModel(simple_2d_env, center=np.array([60.0, 60.0])),
            PlaceCellModel(simple_2d_env, center=np.array([70.0, 70.0])),
        ]
        ground_truth = {i: model.ground_truth for i, model in enumerate(models)}
        metadata = {"n_cells": 3}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_times=spike_trains,
            unit_ids=np.arange(len(spike_trains), dtype=np.int64),
            models=models,
            ground_truth=ground_truth,
            metadata=metadata,
        )

        assert len(session.spike_times) == 3
        assert len(session.models) == 3
        assert len(session.ground_truth) == 3

    def test_simulation_session_repr(self, simple_2d_env):
        """SimulationSession should have informative repr."""
        positions = np.array([[50.0, 50.0], [51.0, 51.0]])
        times = np.array([0.0, 0.1])
        spike_trains = [np.array([0.05])]
        models = [PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))]
        ground_truth = {0: models[0].ground_truth}
        metadata = {"duration": 0.1}

        session = SimulationSession(
            env=simple_2d_env,
            positions=positions,
            times=times,
            spike_times=spike_trains,
            unit_ids=np.arange(len(spike_trains), dtype=np.int64),
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        session = simulate_session(
            simple_2d_env, duration=1.0, n_cells=3, seed=42, show_progress=False
        )

        assert isinstance(session, SimulationSession)
        assert session.env is simple_2d_env
        assert len(session.spike_times) == 3
        assert len(session.models) == 3

    def test_simulate_session_place_cells(self, simple_2d_env):
        """simulate_session() with cell_type='place' should create place cells."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        assert len(session.spike_times) == 5
        assert len(session.ground_truth) == 5

    def test_simulate_session_border_cells(self, connected_arena_env):
        """simulate_session() with cell_type='border' should create border cells."""
        session = simulate_session(
            connected_arena_env,
            duration=2.0,
            n_cells=5,
            cell_type="border",
            distance_tolerance=5.0,
            seed=42,
            show_progress=False,
        )

        assert len(session.models) == 5
        assert all(isinstance(m, BorderCellModel) for m in session.models)
        assert all(m.distance_tolerance == 5.0 for m in session.models)

    @pytest.mark.parametrize("cell_type", ["border", "mixed"])
    def test_border_cells_default_to_a_physical_width(
        self, connected_arena_env, cell_type
    ):
        """Without distance_tolerance, border cells use the 9.8 cm default."""
        session = simulate_session(
            connected_arena_env,
            duration=1.0,
            n_cells=5,
            cell_type=cell_type,
            show_progress=False,
            seed=0,
        )
        tolerances = [
            m.distance_tolerance
            for m in session.models
            if hasattr(m, "distance_tolerance")
        ]
        assert tolerances and all(t == pytest.approx(9.8) for t in tolerances)

    def test_simulate_session_grid_cells(self, simple_2d_env):
        """simulate_session() with cell_type='grid' should create grid cells."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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

    def test_model_parameters_use_the_models_names(self, connected_arena_env):
        """grid_spacing, grid_orientation and width reach the models unrenamed."""
        session = simulate_session(
            connected_arena_env,
            duration=1.0,
            n_cells=10,
            cell_type="mixed",
            grid_spacing=30.0,
            grid_orientation=0.2,
            width=7.0,
            seed=0,
            show_progress=False,
        )
        truths = list(session.ground_truth.values())
        grids = [t for t in truths if t["cell_type"] == "grid"]
        places = [t for t in truths if t["cell_type"] == "place"]
        assert grids and places
        assert all(t["grid_spacing"] == 30.0 for t in grids)
        assert all(t["grid_orientation"] == pytest.approx(0.2) for t in grids)
        assert all(t["width"] == 7.0 for t in places)

    @pytest.mark.parametrize(
        ("cell_type", "parameter", "owner"),
        [
            ("place", {"grid_spacing": 30.0}, "grid"),
            ("place", {"grid_orientation": 0.1}, "grid"),
            ("place", {"distance_tolerance": 5.0}, "border"),
            ("grid", {"width": 5.0}, "place"),
            ("border", {"metric": "geodesic"}, "place"),
        ],
    )
    def test_parameters_of_absent_cell_types_are_refused(
        self, connected_arena_env, cell_type, parameter, owner
    ):
        name = next(iter(parameter))
        with pytest.raises(
            ValueError, match=f"{name} applies only to cell_type='{owner}'"
        ) as raised:
            simulate_session(
                connected_arena_env,
                duration=1.0,
                n_cells=2,
                cell_type=cell_type,
                show_progress=False,
                **parameter,
            )
        assert "Fix:" in str(raised.value)

    @pytest.mark.parametrize("old_name", ["spacing", "orientation", "border_width"])
    def test_renamed_parameters_are_not_accepted(self, connected_arena_env, old_name):
        with pytest.raises(TypeError, match=old_name):
            simulate_session(
                connected_arena_env,
                duration=1.0,
                n_cells=2,
                cell_type="mixed",
                show_progress=False,
                **{old_name: 1.0},
            )

    def test_simulate_session_mixed_cells(self, connected_arena_env):
        """simulate_session() with cell_type='mixed' should create 60% place, 20% border, 20% grid."""
        n_cells = 10
        session = simulate_session(
            connected_arena_env,
            duration=2.0,
            n_cells=n_cells,
            cell_type="mixed",
            distance_tolerance=5.0,
            seed=42,
            show_progress=False,
        )

        # Count cell types
        n_place = sum(isinstance(m, PlaceCellModel) for m in session.models)
        n_border = sum(isinstance(m, BorderCellModel) for m in session.models)
        n_grid = sum(isinstance(m, GridCellModel) for m in session.models)

        # Check proportions (with tolerance for rounding)
        assert n_place == 6  # 60% of 10
        assert n_border == 2  # 20% of 10
        assert n_grid == 2  # 20% of 10

    def test_simulate_session_trajectory_ou(self, simple_2d_env):
        """simulate_session() with trajectory_method='ou' should create OU trajectory."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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

    @pytest.mark.parametrize("n_cells", [5, 9, 50])
    def test_simulate_session_coverage_uniform(self, connected_arena_env, n_cells):
        """Uniform centers are distinct and fill every column and row of the inset grid."""
        session = simulate_session(
            connected_arena_env,
            duration=1.0,
            n_cells=n_cells,
            cell_type="place",
            coverage="uniform",
            seed=42,
            show_progress=False,
        )
        centers = np.array([truth["center"] for truth in session.ground_truth.values()])
        assert len(np.unique(centers, axis=0)) == n_cells
        per_axis = int(np.ceil(np.sqrt(n_cells)))
        for axis in range(2):
            columns = np.unique(centers[:, axis])
            # Every grid line along this axis has a center, from 20% to 80%
            # of the 100 cm arena (snapped to 2 cm bins).
            assert len(columns) == per_axis
            assert columns.min() == pytest.approx(20.0, abs=1.0)
            assert columns.max() == pytest.approx(80.0, abs=1.0)

    def test_mixed_place_and_grid_centers_both_cover_the_arena(
        self, connected_arena_env
    ):
        """In mixed mode neither cell type is confined to part of the arena."""
        session = simulate_session(
            connected_arena_env,
            duration=1.0,
            n_cells=50,
            cell_type="mixed",
            seed=3,
            show_progress=False,
        )
        place = np.array(
            [
                t["center"]
                for t in session.ground_truth.values()
                if t["cell_type"] == "place"
            ]
        )
        grid = np.array(
            [
                t["phase_offset"]
                for t in session.ground_truth.values()
                if t["cell_type"] == "grid"
            ]
        )
        for centers in (place, grid):
            for axis in range(2):
                assert centers[:, axis].min() < 35.0
                assert centers[:, axis].max() > 65.0

    def test_simulate_session_coverage_random(self, simple_2d_env):
        """Random centers are seeded draws of bin centers."""

        def centers(seed):
            session = simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=20,
                cell_type="place",
                coverage="random",
                seed=seed,
                show_progress=False,
            )
            return np.array([t["center"] for t in session.ground_truth.values()])

        first, again, other = centers(42), centers(42), centers(43)
        bins = simple_2d_env.bin_at(first)
        np.testing.assert_array_equal(first, simple_2d_env.bin_centers[bins])
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_simulate_session_seed_reproducibility(self, simple_2d_env):
        """simulate_session() with same seed should produce identical results."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        for st1, st2 in zip(session1.spike_times, session2.spike_times, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_simulate_session_mixed_shared_max_rate_reaches_all_cell_types(
        self, connected_arena_env
    ):
        """A non-default ``max_rate`` in mixed mode must reach every cell type.

        Previously the place-cell parameter pop consumed shared rate kwargs
        (``max_rate``/``baseline_rate``/``metric``) before the border and grid
        loops read them, so border/grid cells silently fell back to their
        defaults. The shared kwargs must now reach all three model types.
        """
        custom_max_rate = 37.0  # not equal to any model's default

        session = simulate_session(
            connected_arena_env,
            duration=1.0,
            n_cells=10,  # 6 place, 2 border, 2 grid
            cell_type="mixed",
            distance_tolerance=5.0,
            max_rate=custom_max_rate,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.models import (
            BorderCellModel,
            GridCellModel,
            PlaceCellModel,
        )

        saw_border = saw_grid = saw_place = False
        for model in session.models:
            assert model.max_rate == custom_max_rate, (
                f"{type(model).__name__} did not receive shared max_rate; "
                f"got {model.max_rate}"
            )
            saw_border |= isinstance(model, BorderCellModel)
            saw_grid |= isinstance(model, GridCellModel)
            saw_place |= isinstance(model, PlaceCellModel)

        assert saw_place and saw_border and saw_grid, (
            "mixed session should contain place, border, and grid cells"
        )

    def test_simulate_session_spike_stream_independent_of_trajectory(
        self, simple_2d_env
    ):
        """Spike RNG stream must be decoupled from the trajectory RNG stream.

        Previously ``simulate_session`` passed the same ``seed`` to both
        ``simulate_trajectory_ou`` and ``generate_population_spikes``, so the
        trajectory and the spikes shared a correlated RNG stream (and neuron 0
        drew from the exact same seed the trajectory consumed). After the fix,
        trajectory / field-center / spike seeds are spawned independently from a
        single ``SeedSequence(seed)``.

        We verify decoupling behaviorally: regenerating spikes for the session's
        own models and positions while reusing the *trajectory* seed must NOT
        reproduce the session spike trains, because the session no longer uses
        the trajectory seed for spikes.
        """
        from neurospatial.simulation.spikes import generate_population_spikes

        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        seed = 7

        session = simulate_session(
            simple_2d_env,
            duration=3.0,
            n_cells=4,
            cell_type="place",
            max_rate=40.0,
            width=15.0,
            seed=seed,
            show_progress=False,
        )

        # Reproduce spikes using the trajectory seed (the old collision path).
        collision_spikes = generate_population_spikes(
            session.models,
            session.times,
            session.positions,
            seed=seed,
            show_progress=False,
        )

        # At least one neuron must have produced spikes (otherwise the test is
        # vacuous), and the session's spike trains must differ from the
        # trajectory-seed reproduction for at least one neuron.
        total = sum(len(st) for st in session.spike_times)
        assert total > 0, "no spikes generated; cannot test stream independence"

        differs = any(
            not np.array_equal(a, b)
            for a, b in zip(session.spike_times, collision_spikes, strict=True)
        )
        assert differs, (
            "session spike trains reproduced exactly using the trajectory seed, "
            "indicating the spike RNG stream is still coupled to the trajectory seed"
        )

    def test_simulate_session_ground_truth_structure(self, simple_2d_env):
        """simulate_session() ground_truth should have proper structure."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
            key = i
            assert key in session.ground_truth

            # For place cells, should have center, width, max_rate, baseline_rate
            gt = session.ground_truth[key]
            assert "center" in gt
            assert "width" in gt
            assert "max_rate" in gt
            assert "baseline_rate" in gt

    def test_simulate_session_metadata(self, simple_2d_env):
        """simulate_session() metadata should contain session parameters."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        simple_1d_env = simple_1d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
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

    def test_simulate_session_trajectory_laps(self, connected_arena_env):
        """simulate_session() with trajectory_method='laps' should work."""
        session = simulate_session(
            connected_arena_env,
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
        simple_1d_env = simple_1d_env.with_metadata(units="cm")
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
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        with pytest.raises(ValueError, match="n_cells must be positive"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=0,
                show_progress=False,
            )

    def test_simulate_session_negative_cells(self, simple_2d_env):
        """simulate_session() with negative n_cells should raise ValueError."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        with pytest.raises(ValueError, match="n_cells must be positive"):
            simulate_session(
                simple_2d_env,
                duration=1.0,
                n_cells=-5,
                show_progress=False,
            )

    def test_simulate_session_zero_duration(self, simple_2d_env):
        """simulate_session() with duration=0 should raise ValueError."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        with pytest.raises(ValueError, match="duration must be positive"):
            simulate_session(
                simple_2d_env,
                duration=0.0,
                n_cells=3,
                show_progress=False,
            )

    def test_simulate_session_negative_duration(self, simple_2d_env):
        """simulate_session() with negative duration should raise ValueError."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")
        with pytest.raises(ValueError, match="duration must be positive"):
            simulate_session(
                simple_2d_env,
                duration=-1.0,
                n_cells=3,
                show_progress=False,
            )


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        ({"times": np.array([0.0, 0.1, 0.2])}, "times"),
        ({"ground_truth": {7: {}}}, "ground_truth"),
    ],
)
def test_session_rejects_misaligned_samples_and_ground_truth(
    simple_2d_env, change, detail
):
    """Misaligned holders would silently pair the wrong samples or models."""
    model = PlaceCellModel(simple_2d_env, center=np.array([50.0, 50.0]))
    fields = {
        "env": simple_2d_env,
        "positions": np.array([[50.0, 50.0], [51.0, 51.0]]),
        "times": np.array([0.0, 0.1]),
        "spike_times": [np.array([0.05])],
        "unit_ids": np.array([0], dtype=np.int64),
        "models": [model],
        "ground_truth": {0: model.ground_truth},
        "metadata": {},
    }
    with pytest.raises(ValueError, match=detail):
        SimulationSession(**{**fields, **change})


def test_session_rejects_duplicate_unit_ids(simple_2d_env):
    """Two cells labelled 7 would both be validated against ground_truth[7]."""
    models = [
        PlaceCellModel(simple_2d_env, center=np.array([20.0, 20.0])),
        PlaceCellModel(simple_2d_env, center=np.array([80.0, 80.0])),
    ]
    with pytest.raises(ValueError, match="unique"):
        SimulationSession(
            env=simple_2d_env,
            positions=np.array([[50.0, 50.0], [51.0, 51.0]]),
            times=np.array([0.0, 0.1]),
            spike_times=[np.array([0.05]), np.array([0.06])],
            unit_ids=np.array([7, 7], dtype=np.int64),
            models=models,
            ground_truth={7: models[0].ground_truth},
            metadata={},
        )


def _holder_fields(env):
    model = PlaceCellModel(env, center=np.array([50.0, 50.0]))
    return {
        "env": env,
        "positions": np.array([[50.0, 50.0], [51.0, 51.0], [52.0, 51.0]]),
        "times": np.array([0.0, 0.1, 0.2]),
        "spike_times": [np.array([0.05, 0.15])],
        "unit_ids": np.array([0], dtype=np.int64),
        "models": [model],
        "ground_truth": {0: model.ground_truth},
        "metadata": {"trial_choices": ["left", "right"]},
    }


class TestSessionOwnsItsData:
    """A frozen session cannot be changed through its arrays or mappings."""

    def test_arrays_are_read_only_copies(self, simple_2d_env):
        fields = _holder_fields(simple_2d_env)
        session = SimulationSession(**fields)
        fields["positions"][0, 0] = -1.0
        fields["spike_times"][0][0] = -1.0
        assert session.positions[0, 0] == 50.0
        assert session.spike_times[0][0] == 0.05
        for array in (
            session.positions,
            session.times,
            session.unit_ids,
            session.spike_times[0],
        ):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_spike_times_and_models_cannot_grow(self, simple_2d_env):
        session = SimulationSession(**_holder_fields(simple_2d_env))
        assert isinstance(session.spike_times, tuple)
        assert isinstance(session.models, tuple)
        with pytest.raises(AttributeError):
            session.spike_times.append(np.array([0.1]))  # type: ignore[attr-defined]

    def test_mappings_are_read_only(self, simple_2d_env):
        session = SimulationSession(**_holder_fields(simple_2d_env))
        with pytest.raises(TypeError):
            session.metadata["seed"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            session.ground_truth[1] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            session.ground_truth[0]["width"] = 1.0  # type: ignore[index]
        with pytest.raises(ValueError, match="read-only"):
            session.ground_truth[0]["center"][0] = 0.0
        assert session.metadata["trial_choices"] == ("left", "right")

    def test_pickle_and_deepcopy_keep_the_contents_read_only(self, simple_2d_env):
        import copy
        import pickle

        session = SimulationSession(**_holder_fields(simple_2d_env))
        for restored in (pickle.loads(pickle.dumps(session)), copy.deepcopy(session)):
            np.testing.assert_array_equal(restored.positions, session.positions)
            np.testing.assert_array_equal(
                restored.spike_times[0], session.spike_times[0]
            )
            assert restored.metadata["trial_choices"] == ("left", "right")
            assert not restored.positions.flags.writeable
            assert not restored.spike_times[0].flags.writeable
            with pytest.raises(TypeError):
                restored.ground_truth[0]["width"] = 1.0  # type: ignore[index]

    @pytest.mark.parametrize(
        ("change", "detail"),
        [
            ({"positions": np.zeros((3, 1))}, "positions"),
            ({"times": np.array([0.0, 0.2, 0.1])}, "strictly increasing"),
            ({"spike_times": [np.array([0.05, 0.25])]}, "spike"),
            ({"spike_times": [np.array([-0.01])]}, "spike"),
            ({"unit_ids": np.array([0.7])}, "integer"),
        ],
    )
    def test_refuses_inconsistent_data(self, simple_2d_env, change, detail):
        fields = {**_holder_fields(simple_2d_env), **change}
        with pytest.raises(ValueError, match=detail) as raised:
            SimulationSession(**fields)
        assert "Fix:" in str(raised.value)
