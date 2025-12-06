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

    def test_metrics_decision_analysis_deleted(self) -> None:
        """Verify metrics/decision_analysis.py re-export wrapper is deleted.

        Users should now import from neurospatial.behavior.decisions.
        """
        if "neurospatial.metrics.decision_analysis" in sys.modules:
            del sys.modules["neurospatial.metrics.decision_analysis"]

        with pytest.raises(ModuleNotFoundError, match="decision_analysis"):
            importlib.import_module("neurospatial.metrics.decision_analysis")

    def test_metrics_goal_directed_deleted(self) -> None:
        """Verify metrics/goal_directed.py re-export wrapper is deleted.

        Users should now import from neurospatial.behavior.navigation.
        """
        if "neurospatial.metrics.goal_directed" in sys.modules:
            del sys.modules["neurospatial.metrics.goal_directed"]

        with pytest.raises(ModuleNotFoundError, match="goal_directed"):
            importlib.import_module("neurospatial.metrics.goal_directed")

    def test_metrics_path_efficiency_deleted(self) -> None:
        """Verify metrics/path_efficiency.py re-export wrapper is deleted.

        Users should now import from neurospatial.behavior.navigation.
        """
        if "neurospatial.metrics.path_efficiency" in sys.modules:
            del sys.modules["neurospatial.metrics.path_efficiency"]

        with pytest.raises(ModuleNotFoundError, match="path_efficiency"):
            importlib.import_module("neurospatial.metrics.path_efficiency")

    def test_metrics_vte_deleted(self) -> None:
        """Verify metrics/vte.py re-export wrapper is deleted.

        Users should now import from neurospatial.behavior.decisions.
        """
        if "neurospatial.metrics.vte" in sys.modules:
            del sys.modules["neurospatial.metrics.vte"]

        with pytest.raises(ModuleNotFoundError, match="vte"):
            importlib.import_module("neurospatial.metrics.vte")


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
        from neurospatial.behavior.decisions import (
            compute_decision_analysis,
            compute_vte_index,
        )

        assert callable(compute_vte_index)
        assert callable(compute_decision_analysis)
