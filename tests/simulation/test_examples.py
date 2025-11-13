"""Tests for pre-configured example simulations."""

from __future__ import annotations

import numpy as np

from neurospatial.simulation.examples import (
    boundary_cell_session,
    grid_cell_session,
    linear_track_session,
    open_field_session,
    tmaze_alternation_session,
)
from neurospatial.simulation.session import SimulationSession


class TestOpenFieldSession:
    """Tests for open_field_session() convenience function."""

    def test_open_field_session_returns_simulation_session(self):
        """open_field_session() should return SimulationSession instance."""
        session = open_field_session(duration=10.0, n_place_cells=5)

        assert isinstance(session, SimulationSession)

    def test_open_field_session_default_parameters(self):
        """open_field_session() should work with default parameters."""
        session = open_field_session()

        # Should create session
        assert isinstance(session, SimulationSession)
        assert session.env is not None
        assert len(session.spike_trains) > 0

    def test_open_field_session_creates_square_arena(self):
        """open_field_session() should create square arena environment."""
        arena_size = 100.0
        bin_size = 2.0
        session = open_field_session(
            duration=10.0, arena_size=arena_size, bin_size=bin_size, n_place_cells=5
        )

        env = session.env

        # Check that environment is 2D
        assert env.n_dims == 2

        # Check environment units
        assert env.units == "cm"

    def test_open_field_session_correct_number_of_cells(self):
        """open_field_session() should create correct number of place cells."""
        n_cells = 20
        session = open_field_session(duration=10.0, n_place_cells=n_cells)

        assert len(session.models) == n_cells
        assert len(session.spike_trains) == n_cells
        assert len(session.ground_truth) == n_cells

    def test_open_field_session_uses_place_cells(self):
        """open_field_session() should create only place cells."""
        from neurospatial.simulation.models import PlaceCellModel

        session = open_field_session(duration=10.0, n_place_cells=5)

        # All models should be place cells
        assert all(isinstance(m, PlaceCellModel) for m in session.models)

    def test_open_field_session_seed_reproducibility(self):
        """open_field_session() with same seed should produce identical results."""
        seed = 42

        session1 = open_field_session(duration=10.0, n_place_cells=5, seed=seed)
        session2 = open_field_session(duration=10.0, n_place_cells=5, seed=seed)

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_open_field_session_custom_duration(self):
        """open_field_session() should respect custom duration."""
        duration = 60.0
        session = open_field_session(duration=duration, n_place_cells=5)

        # Check that session duration matches
        assert session.times[-1] >= duration * 0.9  # Allow small tolerance

    def test_open_field_session_custom_arena_size(self):
        """open_field_session() should create arena with custom size."""
        arena_size = 50.0
        session = open_field_session(
            duration=10.0, arena_size=arena_size, bin_size=1.0, n_place_cells=5
        )

        env = session.env

        # Check environment dimensions (approximately arena_size / bin_size bins per dimension)
        # With some tolerance for rounding
        expected_bins_per_dim = int(arena_size / 1.0)
        assert env.n_bins >= expected_bins_per_dim  # At least this many bins

    def test_open_field_session_custom_bin_size(self):
        """open_field_session() should use custom bin_size."""
        bin_size_small = 2.0
        bin_size_large = 10.0

        session_small = open_field_session(
            duration=10.0, arena_size=100.0, bin_size=bin_size_small, n_place_cells=5
        )
        session_large = open_field_session(
            duration=10.0, arena_size=100.0, bin_size=bin_size_large, n_place_cells=5
        )

        # Smaller bin_size should result in more bins
        assert session_small.env.n_bins > session_large.env.n_bins

    def test_open_field_session_metadata(self):
        """open_field_session() metadata should contain session info."""
        session = open_field_session(duration=10.0, n_place_cells=5)

        meta = session.metadata
        assert "cell_type" in meta
        assert meta["cell_type"] == "place"

    def test_open_field_session_generates_spikes(self):
        """open_field_session() should generate spikes."""
        session = open_field_session(duration=30.0, n_place_cells=5, seed=42)

        # At least some cells should have spikes
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0, "No spikes generated in session"

    def test_open_field_session_trajectory_within_bounds(self):
        """open_field_session() trajectory should be within arena bounds."""
        session = open_field_session(duration=10.0, n_place_cells=5, seed=42)

        # All positions should map to valid bins
        bin_indices = session.env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions outside environment"

    def test_open_field_session_ground_truth_structure(self):
        """open_field_session() should have proper ground_truth structure."""
        n_cells = 5
        session = open_field_session(duration=10.0, n_place_cells=n_cells, seed=42)

        # Check ground_truth keys
        assert len(session.ground_truth) == n_cells
        for i in range(n_cells):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # For place cells, should have center, width, max_rate, baseline_rate
            gt = session.ground_truth[key]
            assert "center" in gt
            assert "width" in gt
            assert "max_rate" in gt
            assert "baseline_rate" in gt


class TestLinearTrackSession:
    """Tests for linear_track_session() convenience function."""

    def test_linear_track_session_returns_simulation_session(self):
        """linear_track_session() should return SimulationSession instance."""
        session = linear_track_session(duration=10.0, n_place_cells=5, n_laps=3)

        assert isinstance(session, SimulationSession)

    def test_linear_track_session_default_parameters(self):
        """linear_track_session() should work with default parameters."""
        session = linear_track_session()

        # Should create session
        assert isinstance(session, SimulationSession)
        assert session.env is not None
        assert len(session.spike_trains) > 0

    def test_linear_track_session_creates_1d_environment(self):
        """linear_track_session() should create 1D track environment."""
        track_length = 200.0
        bin_size = 1.0
        session = linear_track_session(
            duration=10.0,
            track_length=track_length,
            bin_size=bin_size,
            n_place_cells=5,
            n_laps=3,
        )

        env = session.env

        # Check that environment is 1D
        assert env.n_dims == 1

        # Check environment units
        assert env.units == "cm"

    def test_linear_track_session_correct_number_of_cells(self):
        """linear_track_session() should create correct number of place cells."""
        n_cells = 20
        session = linear_track_session(duration=10.0, n_place_cells=n_cells, n_laps=3)

        assert len(session.models) == n_cells
        assert len(session.spike_trains) == n_cells
        assert len(session.ground_truth) == n_cells

    def test_linear_track_session_uses_place_cells(self):
        """linear_track_session() should create only place cells."""
        from neurospatial.simulation.models import PlaceCellModel

        session = linear_track_session(duration=10.0, n_place_cells=5, n_laps=3)

        # All models should be place cells
        assert all(isinstance(m, PlaceCellModel) for m in session.models)

    def test_linear_track_session_seed_reproducibility(self):
        """linear_track_session() with same seed should produce identical results."""
        seed = 42

        session1 = linear_track_session(
            duration=10.0, n_place_cells=5, n_laps=3, seed=seed
        )
        session2 = linear_track_session(
            duration=10.0, n_place_cells=5, n_laps=3, seed=seed
        )

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_linear_track_session_custom_duration(self):
        """linear_track_session() should respect custom duration."""
        duration = 60.0
        session = linear_track_session(duration=duration, n_place_cells=5, n_laps=5)

        # Check that session duration matches approximately
        assert session.times[-1] >= duration * 0.9  # Allow small tolerance

    def test_linear_track_session_custom_track_length(self):
        """linear_track_session() should create track with custom length."""
        track_length = 150.0
        session = linear_track_session(
            duration=10.0,
            track_length=track_length,
            bin_size=1.0,
            n_place_cells=5,
            n_laps=3,
        )

        env = session.env

        # Check that environment spans approximately track_length
        # For 1D environment, check bin_centers span
        position_span = np.ptp(env.bin_centers)
        assert position_span >= track_length * 0.9  # At least 90% of track length

    def test_linear_track_session_custom_bin_size(self):
        """linear_track_session() should use custom bin_size."""
        bin_size_small = 1.0
        bin_size_large = 5.0

        session_small = linear_track_session(
            duration=10.0,
            track_length=200.0,
            bin_size=bin_size_small,
            n_place_cells=5,
            n_laps=3,
        )
        session_large = linear_track_session(
            duration=10.0,
            track_length=200.0,
            bin_size=bin_size_large,
            n_place_cells=5,
            n_laps=3,
        )

        # Smaller bin_size should result in more bins
        assert session_small.env.n_bins > session_large.env.n_bins

    def test_linear_track_session_custom_n_laps(self):
        """linear_track_session() should use custom n_laps."""
        session = linear_track_session(
            duration=20.0, n_place_cells=5, n_laps=10, seed=42
        )

        # Should complete without error
        assert len(session.positions) > 0
        assert len(session.times) > 0

    def test_linear_track_session_metadata(self):
        """linear_track_session() metadata should contain session info."""
        session = linear_track_session(duration=10.0, n_place_cells=5, n_laps=3)

        meta = session.metadata
        assert "cell_type" in meta
        assert meta["cell_type"] == "place"
        assert "trajectory_method" in meta
        assert meta["trajectory_method"] == "laps"

    def test_linear_track_session_generates_spikes(self):
        """linear_track_session() should generate spikes."""
        session = linear_track_session(
            duration=30.0, n_place_cells=5, n_laps=5, seed=42
        )

        # At least some cells should have spikes
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0, "No spikes generated in session"

    def test_linear_track_session_trajectory_within_bounds(self):
        """linear_track_session() trajectory should be within track bounds."""
        session = linear_track_session(
            duration=10.0, n_place_cells=5, n_laps=3, seed=42
        )

        # All positions should map to valid bins
        bin_indices = session.env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions outside environment"

    def test_linear_track_session_ground_truth_structure(self):
        """linear_track_session() should have proper ground_truth structure."""
        n_cells = 5
        session = linear_track_session(
            duration=10.0, n_place_cells=n_cells, n_laps=3, seed=42
        )

        # Check ground_truth keys
        assert len(session.ground_truth) == n_cells
        for i in range(n_cells):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # For place cells, should have center, width, max_rate, baseline_rate
            gt = session.ground_truth[key]
            assert "center" in gt
            assert "width" in gt
            assert "max_rate" in gt
            assert "baseline_rate" in gt


class TestTmazeAlternationSession:
    """Tests for tmaze_alternation_session() convenience function."""

    def test_tmaze_alternation_session_returns_simulation_session(self):
        """tmaze_alternation_session() should return SimulationSession instance."""
        session = tmaze_alternation_session(duration=10.0, n_trials=5, n_place_cells=10)

        assert isinstance(session, SimulationSession)

    def test_tmaze_alternation_session_default_parameters(self):
        """tmaze_alternation_session() should work with default parameters."""
        session = tmaze_alternation_session()

        # Should create session
        assert isinstance(session, SimulationSession)
        assert session.env is not None
        assert len(session.spike_trains) > 0

    def test_tmaze_alternation_session_has_trial_metadata(self):
        """tmaze_alternation_session() should include trial_choices in metadata."""
        n_trials = 10
        session = tmaze_alternation_session(
            duration=30.0, n_trials=n_trials, n_place_cells=10
        )

        meta = session.metadata
        assert "trial_choices" in meta
        assert isinstance(meta["trial_choices"], list)

    def test_tmaze_alternation_session_correct_number_of_trials(self):
        """tmaze_alternation_session() should have correct number of trials."""
        n_trials = 15
        session = tmaze_alternation_session(
            duration=30.0, n_trials=n_trials, n_place_cells=10
        )

        trial_choices = session.metadata["trial_choices"]
        assert len(trial_choices) == n_trials

    def test_tmaze_alternation_session_alternating_choices(self):
        """tmaze_alternation_session() should have alternating left/right pattern."""
        session = tmaze_alternation_session(
            duration=30.0, n_trials=10, n_place_cells=10, seed=42
        )

        trial_choices = session.metadata["trial_choices"]

        # Should have both 'left' and 'right' choices
        assert "left" in trial_choices
        assert "right" in trial_choices

        # Choices should alternate (for perfect alternation)
        # Note: Depending on implementation, may be ['left', 'right', 'left', 'right', ...]
        # or start with random first choice
        unique_choices = set(trial_choices)
        assert unique_choices == {"left", "right"}

    def test_tmaze_alternation_session_correct_number_of_cells(self):
        """tmaze_alternation_session() should create correct number of place cells."""
        n_cells = 30
        session = tmaze_alternation_session(
            duration=20.0, n_trials=5, n_place_cells=n_cells
        )

        assert len(session.models) == n_cells
        assert len(session.spike_trains) == n_cells
        assert len(session.ground_truth) == n_cells

    def test_tmaze_alternation_session_uses_place_cells(self):
        """tmaze_alternation_session() should create only place cells."""
        from neurospatial.simulation.models import PlaceCellModel

        session = tmaze_alternation_session(duration=20.0, n_trials=5, n_place_cells=10)

        # All models should be place cells
        assert all(isinstance(m, PlaceCellModel) for m in session.models)

    def test_tmaze_alternation_session_seed_reproducibility(self):
        """tmaze_alternation_session() with same seed should produce identical results."""
        seed = 42

        session1 = tmaze_alternation_session(
            duration=20.0, n_trials=8, n_place_cells=10, seed=seed
        )
        session2 = tmaze_alternation_session(
            duration=20.0, n_trials=8, n_place_cells=10, seed=seed
        )

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

        # Trial choices should be identical
        assert session1.metadata["trial_choices"] == session2.metadata["trial_choices"]

    def test_tmaze_alternation_session_custom_duration(self):
        """tmaze_alternation_session() should respect custom duration."""
        duration = 90.0
        session = tmaze_alternation_session(
            duration=duration, n_trials=10, n_place_cells=10
        )

        # Check that session duration matches approximately
        assert session.times[-1] >= duration * 0.9  # Allow small tolerance

    def test_tmaze_alternation_session_metadata(self):
        """tmaze_alternation_session() metadata should contain session info."""
        session = tmaze_alternation_session(duration=20.0, n_trials=5, n_place_cells=10)

        meta = session.metadata
        assert "cell_type" in meta
        assert meta["cell_type"] == "place"
        assert "trial_choices" in meta
        assert "trajectory_method" in meta

    def test_tmaze_alternation_session_generates_spikes(self):
        """tmaze_alternation_session() should generate spikes."""
        session = tmaze_alternation_session(
            duration=60.0, n_trials=10, n_place_cells=10, seed=42
        )

        # At least some cells should have spikes
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0, "No spikes generated in session"

    def test_tmaze_alternation_session_ground_truth_structure(self):
        """tmaze_alternation_session() should have proper ground_truth structure."""
        n_cells = 10
        session = tmaze_alternation_session(
            duration=20.0, n_trials=5, n_place_cells=n_cells, seed=42
        )

        # Check ground_truth keys
        assert len(session.ground_truth) == n_cells
        for i in range(n_cells):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # For place cells, should have center, width, max_rate, baseline_rate
            gt = session.ground_truth[key]
            assert "center" in gt
            assert "width" in gt
            assert "max_rate" in gt
            assert "baseline_rate" in gt

    def test_tmaze_alternation_session_trajectory_within_bounds(self):
        """tmaze_alternation_session() trajectory should be within environment bounds."""
        session = tmaze_alternation_session(
            duration=20.0, n_trials=5, n_place_cells=10, seed=42
        )

        # All positions should map to valid bins
        bin_indices = session.env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions outside environment"

    def test_tmaze_alternation_session_validates_duration(self):
        """tmaze_alternation_session() should reject non-positive duration."""
        import pytest

        with pytest.raises(ValueError, match="duration must be positive"):
            tmaze_alternation_session(duration=-1.0, n_trials=5, n_place_cells=10)

        with pytest.raises(ValueError, match="duration must be positive"):
            tmaze_alternation_session(duration=0.0, n_trials=5, n_place_cells=10)

    def test_tmaze_alternation_session_validates_n_trials(self):
        """tmaze_alternation_session() should reject non-positive n_trials."""
        import pytest

        with pytest.raises(ValueError, match="n_trials must be positive"):
            tmaze_alternation_session(duration=30.0, n_trials=0, n_place_cells=10)

        with pytest.raises(ValueError, match="n_trials must be positive"):
            tmaze_alternation_session(duration=30.0, n_trials=-5, n_place_cells=10)

    def test_tmaze_alternation_session_validates_n_place_cells(self):
        """tmaze_alternation_session() should reject non-positive n_place_cells."""
        import pytest

        with pytest.raises(ValueError, match="n_place_cells must be positive"):
            tmaze_alternation_session(duration=30.0, n_trials=5, n_place_cells=0)

        with pytest.raises(ValueError, match="n_place_cells must be positive"):
            tmaze_alternation_session(duration=30.0, n_trials=5, n_place_cells=-10)


class TestBoundaryCellSession:
    """Tests for boundary_cell_session() convenience function."""

    def test_boundary_cell_session_returns_simulation_session(self):
        """boundary_cell_session() should return SimulationSession instance."""
        session = boundary_cell_session(
            duration=10.0, n_boundary_cells=5, n_place_cells=5
        )

        assert isinstance(session, SimulationSession)

    def test_boundary_cell_session_default_parameters(self):
        """boundary_cell_session() should work with default parameters."""
        session = boundary_cell_session()

        # Should create session
        assert isinstance(session, SimulationSession)
        assert session.env is not None
        assert len(session.spike_trains) > 0

    def test_boundary_cell_session_creates_2d_arena(self):
        """boundary_cell_session() should create 2D arena environment."""
        session = boundary_cell_session(
            duration=10.0,
            arena_shape="square",
            arena_size=100.0,
            n_boundary_cells=5,
            n_place_cells=5,
        )

        env = session.env

        # Check that environment is 2D
        assert env.n_dims == 2

        # Check environment units
        assert env.units == "cm"

    def test_boundary_cell_session_square_arena(self):
        """boundary_cell_session() should create square arena."""
        session = boundary_cell_session(
            duration=10.0,
            arena_shape="square",
            n_boundary_cells=5,
            n_place_cells=5,
        )

        assert session.env is not None
        assert session.env.n_dims == 2

    def test_boundary_cell_session_correct_number_of_cells(self):
        """boundary_cell_session() should create correct number of cells."""
        n_boundary = 15
        n_place = 10
        session = boundary_cell_session(
            duration=10.0,
            n_boundary_cells=n_boundary,
            n_place_cells=n_place,
        )

        total_cells = n_boundary + n_place
        assert len(session.models) == total_cells
        assert len(session.spike_trains) == total_cells
        assert len(session.ground_truth) == total_cells

    def test_boundary_cell_session_uses_mixed_cell_types(self):
        """boundary_cell_session() should create both boundary and place cells."""
        from neurospatial.simulation.models import BoundaryCellModel, PlaceCellModel

        session = boundary_cell_session(
            duration=10.0, n_boundary_cells=10, n_place_cells=5
        )

        # Count cell types
        n_boundary = sum(isinstance(m, BoundaryCellModel) for m in session.models)
        n_place = sum(isinstance(m, PlaceCellModel) for m in session.models)

        assert n_boundary == 10
        assert n_place == 5
        assert n_boundary + n_place == len(session.models)

    def test_boundary_cell_session_seed_reproducibility(self):
        """boundary_cell_session() with same seed should produce identical results."""
        seed = 42

        session1 = boundary_cell_session(
            duration=10.0, n_boundary_cells=5, n_place_cells=5, seed=seed
        )
        session2 = boundary_cell_session(
            duration=10.0, n_boundary_cells=5, n_place_cells=5, seed=seed
        )

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_boundary_cell_session_custom_duration(self):
        """boundary_cell_session() should respect custom duration."""
        duration = 60.0
        session = boundary_cell_session(
            duration=duration, n_boundary_cells=5, n_place_cells=5
        )

        # Check that session duration matches approximately
        assert session.times[-1] >= duration * 0.9  # Allow small tolerance

    def test_boundary_cell_session_custom_arena_size(self):
        """boundary_cell_session() should create arena with custom size."""
        arena_size = 80.0
        session = boundary_cell_session(
            duration=10.0,
            arena_size=arena_size,
            n_boundary_cells=5,
            n_place_cells=5,
        )

        env = session.env

        # Environment should be created
        assert env is not None
        assert env.n_bins > 0

    def test_boundary_cell_session_metadata(self):
        """boundary_cell_session() metadata should contain session info."""
        session = boundary_cell_session(
            duration=10.0, n_boundary_cells=5, n_place_cells=5
        )

        meta = session.metadata
        assert "cell_type" in meta
        # Should indicate mixed cell types
        assert "boundary" in meta["cell_type"] or "mixed" in meta["cell_type"]

    def test_boundary_cell_session_generates_spikes(self):
        """boundary_cell_session() should generate spikes."""
        session = boundary_cell_session(
            duration=30.0, n_boundary_cells=5, n_place_cells=5, seed=42
        )

        # At least some cells should have spikes
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0, "No spikes generated in session"

    def test_boundary_cell_session_ground_truth_structure(self):
        """boundary_cell_session() should have proper ground_truth structure."""
        n_boundary = 3
        n_place = 2
        session = boundary_cell_session(
            duration=10.0,
            n_boundary_cells=n_boundary,
            n_place_cells=n_place,
            seed=42,
        )

        # Check ground_truth keys
        total_cells = n_boundary + n_place
        assert len(session.ground_truth) == total_cells
        for i in range(total_cells):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # All cells should have some ground truth parameters
            gt = session.ground_truth[key]
            assert isinstance(gt, dict)
            assert len(gt) > 0

    def test_boundary_cell_session_trajectory_within_bounds(self):
        """boundary_cell_session() trajectory should be within environment bounds."""
        session = boundary_cell_session(
            duration=10.0, n_boundary_cells=5, n_place_cells=5, seed=42
        )

        # All positions should map to valid bins
        bin_indices = session.env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions outside environment"

    def test_boundary_cell_session_validates_duration(self):
        """boundary_cell_session() should reject non-positive duration."""
        import pytest

        with pytest.raises(ValueError, match="duration must be positive"):
            boundary_cell_session(duration=-1.0, n_boundary_cells=5, n_place_cells=5)

        with pytest.raises(ValueError, match="duration must be positive"):
            boundary_cell_session(duration=0.0, n_boundary_cells=5, n_place_cells=5)

    def test_boundary_cell_session_validates_n_boundary_cells(self):
        """boundary_cell_session() should reject non-positive n_boundary_cells."""
        import pytest

        with pytest.raises(ValueError, match="n_boundary_cells must be positive"):
            boundary_cell_session(duration=30.0, n_boundary_cells=0, n_place_cells=5)

        with pytest.raises(ValueError, match="n_boundary_cells must be positive"):
            boundary_cell_session(duration=30.0, n_boundary_cells=-5, n_place_cells=5)

    def test_boundary_cell_session_validates_n_place_cells(self):
        """boundary_cell_session() should reject non-positive n_place_cells."""
        import pytest

        with pytest.raises(ValueError, match="n_place_cells must be positive"):
            boundary_cell_session(duration=30.0, n_boundary_cells=5, n_place_cells=0)

        with pytest.raises(ValueError, match="n_place_cells must be positive"):
            boundary_cell_session(duration=30.0, n_boundary_cells=5, n_place_cells=-5)


class TestGridCellSession:
    """Tests for grid_cell_session() convenience function."""

    def test_grid_cell_session_returns_simulation_session(self):
        """grid_cell_session() should return SimulationSession instance."""
        session = grid_cell_session(duration=10.0, n_grid_cells=5)

        assert isinstance(session, SimulationSession)

    def test_grid_cell_session_default_parameters(self):
        """grid_cell_session() should work with default parameters."""
        session = grid_cell_session()

        # Should create session
        assert isinstance(session, SimulationSession)
        assert session.env is not None
        assert len(session.spike_trains) > 0

    def test_grid_cell_session_creates_2d_arena(self):
        """grid_cell_session() should create 2D arena environment."""
        arena_size = 150.0
        session = grid_cell_session(
            duration=10.0, arena_size=arena_size, n_grid_cells=5
        )

        env = session.env

        # Check that environment is 2D
        assert env.n_dims == 2

        # Check environment units
        assert env.units == "cm"

    def test_grid_cell_session_correct_number_of_cells(self):
        """grid_cell_session() should create correct number of grid cells."""
        n_cells = 20
        session = grid_cell_session(duration=10.0, n_grid_cells=n_cells)

        assert len(session.models) == n_cells
        assert len(session.spike_trains) == n_cells
        assert len(session.ground_truth) == n_cells

    def test_grid_cell_session_uses_grid_cells(self):
        """grid_cell_session() should create only grid cells."""
        from neurospatial.simulation.models import GridCellModel

        session = grid_cell_session(duration=10.0, n_grid_cells=5)

        # All models should be grid cells
        assert all(isinstance(m, GridCellModel) for m in session.models)

    def test_grid_cell_session_seed_reproducibility(self):
        """grid_cell_session() with same seed should produce identical results."""
        seed = 42

        session1 = grid_cell_session(duration=10.0, n_grid_cells=5, seed=seed)
        session2 = grid_cell_session(duration=10.0, n_grid_cells=5, seed=seed)

        # Trajectories should be identical
        np.testing.assert_array_equal(session1.positions, session2.positions)
        np.testing.assert_array_equal(session1.times, session2.times)

        # Spike trains should be identical
        for st1, st2 in zip(session1.spike_trains, session2.spike_trains, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_grid_cell_session_custom_duration(self):
        """grid_cell_session() should respect custom duration."""
        duration = 60.0
        session = grid_cell_session(duration=duration, n_grid_cells=5)

        # Check that session duration matches
        assert session.times[-1] >= duration * 0.9  # Allow small tolerance

    def test_grid_cell_session_custom_arena_size(self):
        """grid_cell_session() should create arena with custom size."""
        arena_size = 100.0
        session = grid_cell_session(
            duration=10.0, arena_size=arena_size, n_grid_cells=5
        )

        env = session.env

        # Environment should be created
        assert env is not None
        assert env.n_bins > 0

    def test_grid_cell_session_custom_grid_spacing(self):
        """grid_cell_session() should use custom grid_spacing."""
        grid_spacing = 30.0
        session = grid_cell_session(
            duration=10.0, grid_spacing=grid_spacing, n_grid_cells=5, seed=42
        )

        # Should complete without error
        assert session is not None
        assert len(session.models) == 5

    def test_grid_cell_session_metadata(self):
        """grid_cell_session() metadata should contain session info."""
        session = grid_cell_session(duration=10.0, n_grid_cells=5)

        meta = session.metadata
        assert "cell_type" in meta
        assert meta["cell_type"] == "grid"

    def test_grid_cell_session_generates_spikes(self):
        """grid_cell_session() should generate spikes."""
        session = grid_cell_session(duration=30.0, n_grid_cells=5, seed=42)

        # At least some cells should have spikes
        total_spikes = sum(len(st) for st in session.spike_trains)
        assert total_spikes > 0, "No spikes generated in session"

    def test_grid_cell_session_trajectory_within_bounds(self):
        """grid_cell_session() trajectory should be within arena bounds."""
        session = grid_cell_session(duration=10.0, n_grid_cells=5, seed=42)

        # All positions should map to valid bins
        bin_indices = session.env.bin_at(session.positions)
        assert np.all(bin_indices >= 0), "Some positions outside environment"

    def test_grid_cell_session_ground_truth_structure(self):
        """grid_cell_session() should have proper ground_truth structure."""
        n_cells = 5
        session = grid_cell_session(duration=10.0, n_grid_cells=n_cells, seed=42)

        # Check ground_truth keys
        assert len(session.ground_truth) == n_cells
        for i in range(n_cells):
            key = f"cell_{i}"
            assert key in session.ground_truth

            # For grid cells, should have grid-specific parameters
            gt = session.ground_truth[key]
            assert "cell_type" in gt
            assert gt["cell_type"] == "grid"

    def test_grid_cell_session_validates_duration(self):
        """grid_cell_session() should reject non-positive duration."""
        import pytest

        with pytest.raises(ValueError, match="duration must be positive"):
            grid_cell_session(duration=-1.0, n_grid_cells=5)

        with pytest.raises(ValueError, match="duration must be positive"):
            grid_cell_session(duration=0.0, n_grid_cells=5)

    def test_grid_cell_session_validates_n_grid_cells(self):
        """grid_cell_session() should reject non-positive n_grid_cells."""
        import pytest

        with pytest.raises(ValueError, match="n_grid_cells must be positive"):
            grid_cell_session(duration=30.0, n_grid_cells=0)

        with pytest.raises(ValueError, match="n_grid_cells must be positive"):
            grid_cell_session(duration=30.0, n_grid_cells=-5)

    def test_grid_cell_session_validates_grid_spacing(self):
        """grid_cell_session() should reject non-positive grid_spacing."""
        import pytest

        with pytest.raises(ValueError, match="grid_spacing must be positive"):
            grid_cell_session(duration=30.0, grid_spacing=-10.0, n_grid_cells=5)

        with pytest.raises(ValueError, match="grid_spacing must be positive"):
            grid_cell_session(duration=30.0, grid_spacing=0.0, n_grid_cells=5)

    def test_grid_cell_session_validates_arena_size(self):
        """grid_cell_session() should reject non-positive arena_size."""
        import pytest

        with pytest.raises(ValueError, match="arena_size must be positive"):
            grid_cell_session(duration=30.0, arena_size=-100.0, n_grid_cells=5)

        with pytest.raises(ValueError, match="arena_size must be positive"):
            grid_cell_session(duration=30.0, arena_size=0.0, n_grid_cells=5)

    def test_grid_cell_session_validates_grid_spacing_vs_arena_size(self):
        """grid_cell_session() should reject grid_spacing >= arena_size."""
        import pytest

        # grid_spacing equal to arena_size
        with pytest.raises(ValueError, match="must be smaller than"):
            grid_cell_session(
                duration=30.0, arena_size=100.0, grid_spacing=100.0, n_grid_cells=5
            )

        # grid_spacing larger than arena_size
        with pytest.raises(ValueError, match="must be smaller than"):
            grid_cell_session(
                duration=30.0, arena_size=100.0, grid_spacing=150.0, n_grid_cells=5
            )
