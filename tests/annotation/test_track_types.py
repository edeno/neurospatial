"""Tests for track graph annotation type definitions."""

from __future__ import annotations

from typing import get_args


class TestTrackGraphMode:
    """Tests for TrackGraphMode type alias."""

    def test_import(self) -> None:
        """Type alias should be importable from module."""
        from neurospatial.annotation._track_types import TrackGraphMode

        assert TrackGraphMode is not None

    def test_literal_values(self) -> None:
        """Type alias should have expected literal values."""
        from neurospatial.annotation._track_types import TrackGraphMode

        # Get the literal values from the type alias
        values = get_args(TrackGraphMode)

        assert "add_node" in values
        assert "add_edge" in values
        assert "delete" in values
        assert len(values) == 3

    def test_valid_literal_assignment(self) -> None:
        """Valid literal strings should be assignable to TrackGraphMode variables.

        This is a runtime test that the type exists and can be used.
        Mypy will verify the actual type checking at static analysis time.
        """
        from neurospatial.annotation._track_types import TrackGraphMode

        # These assignments should work at runtime (mypy checks at static time)
        mode1: TrackGraphMode = "add_node"
        mode2: TrackGraphMode = "add_edge"
        mode3: TrackGraphMode = "delete"

        assert mode1 == "add_node"
        assert mode2 == "add_edge"
        assert mode3 == "delete"
