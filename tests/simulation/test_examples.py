"""Tests for pre-configured example simulations."""

from __future__ import annotations

import numpy as np

from neurospatial.simulation.examples import open_field_session
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
