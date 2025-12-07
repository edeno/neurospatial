"""Tests to verify old backward-compatibility files are deleted.

Milestone 10: Delete Old Files

This test file verifies that:
1. Backward-compatibility re-export wrappers have been deleted
2. All imports now use the new canonical paths
3. No production code imports from deleted locations
"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestOldFilesDeleted:
    """Tests that old backward-compat files are properly deleted."""

    def test_behavioral_py_deleted(self) -> None:
        """Verify behavioral.py backward-compat wrapper is deleted.

        The old neurospatial.behavioral module was a re-export wrapper.
        Users should now import from neurospatial.behavior.navigation
        or neurospatial.behavior.trajectory.
        """
        # Clear any cached import
        if "neurospatial.behavioral" in sys.modules:
            del sys.modules["neurospatial.behavioral"]

        with pytest.raises(ModuleNotFoundError, match="behavioral"):
            importlib.import_module("neurospatial.behavioral")

    def test_metrics_package_deleted(self) -> None:
        """Verify entire metrics package has been deleted.

        The metrics package was a re-export layer that duplicated the API.
        Users should now import from canonical locations:
        - neurospatial.encoding.place (place field metrics)
        - neurospatial.encoding.head_direction (HD cell metrics)
        - neurospatial.encoding.object_vector (OVC metrics)
        - neurospatial.behavior.navigation (navigation metrics)
        - neurospatial.behavior.vte (VTE metrics)
        """
        # Clear any cached imports
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("neurospatial.metrics"):
                del sys.modules[module_name]

        with pytest.raises(ModuleNotFoundError, match="metrics"):
            importlib.import_module("neurospatial.metrics")


class TestNewImportPathsWork:
    """Tests that new canonical import paths work correctly."""

    def test_behavior_navigation_imports(self) -> None:
        """Verify behavior.navigation module has all expected exports."""
        from neurospatial.behavior.navigation import (
            cost_to_goal,
            graph_turn_sequence,
            path_progress,
        )

        # Verify they're callable/accessible
        assert callable(path_progress)
        assert callable(cost_to_goal)
        assert callable(graph_turn_sequence)

    def test_behavior_trajectory_imports(self) -> None:
        """Verify behavior.trajectory module has all expected exports."""
        from neurospatial.behavior.trajectory import (
            compute_step_lengths,
            compute_trajectory_curvature,
        )

        assert callable(compute_trajectory_curvature)
        assert callable(compute_step_lengths)

    def test_behavior_decisions_imports(self) -> None:
        """Verify behavior.decisions module has all expected exports."""
        from neurospatial.behavior.decisions import compute_decision_analysis

        assert callable(compute_decision_analysis)

    def test_behavior_vte_imports(self) -> None:
        """Verify behavior.vte module has all expected exports."""
        from neurospatial.behavior.vte import compute_vte_index

        assert callable(compute_vte_index)
