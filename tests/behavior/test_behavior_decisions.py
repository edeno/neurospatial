"""Tests for behavior/decisions.py module (new location).

This test file verifies that all decision and VTE functions are importable from
the new location and work correctly. These tests follow TDD - written before
the implementation is moved.

Functions being moved:
- From metrics/decision_analysis.py:
  - PreDecisionMetrics, DecisionBoundaryMetrics, DecisionAnalysisResult
  - compute_decision_analysis, compute_pre_decision_metrics
  - decision_region_entry_time, detect_boundary_crossings
  - distance_to_decision_boundary, extract_pre_decision_window
  - geodesic_voronoi_labels, pre_decision_heading_stats, pre_decision_speed_stats

- From metrics/vte.py:
  - VTETrialResult, VTESessionResult
  - compute_vte_index, compute_vte_trial, compute_vte_session
  - classify_vte, head_sweep_from_positions, head_sweep_magnitude
  - integrated_absolute_rotation, normalize_vte_scores
  - wrap_angle (re-exported from stats.circular)
"""

import numpy as np
import pytest

# =============================================================================
# Constants for parametrized import tests
# =============================================================================

# Decision analysis classes (non-callable)
DECISION_CLASSES = [
    "PreDecisionMetrics",
    "DecisionBoundaryMetrics",
    "DecisionAnalysisResult",
]

# Decision analysis functions (callable)
DECISION_FUNCTIONS = [
    "compute_decision_analysis",
    "compute_pre_decision_metrics",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "geodesic_voronoi_labels",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
]

# VTE classes (non-callable)
VTE_CLASSES = [
    "VTETrialResult",
    "VTESessionResult",
]

# VTE functions (callable)
VTE_FUNCTIONS = [
    "compute_vte_index",
    "compute_vte_trial",
    "compute_vte_session",
    "classify_vte",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "integrated_absolute_rotation",
    "normalize_vte_scores",
]


# =============================================================================
# Parametrized import tests
# =============================================================================


class TestDecisionAnalysisImports:
    """Test that decision analysis functions are importable from behavior.decisions."""

    @pytest.mark.parametrize("class_name", DECISION_CLASSES)
    def test_class_importable_from_decisions(self, class_name: str) -> None:
        """Verify {class_name} is importable from behavior.decisions."""
        from neurospatial.behavior import decisions

        cls = getattr(decisions, class_name, None)
        assert cls is not None, f"{class_name} not found in behavior.decisions"

    @pytest.mark.parametrize("func_name", DECISION_FUNCTIONS)
    def test_function_importable_from_decisions(self, func_name: str) -> None:
        """Verify {func_name} is importable from behavior.decisions."""
        from neurospatial.behavior import decisions

        func = getattr(decisions, func_name, None)
        assert func is not None, f"{func_name} not found in behavior.decisions"
        assert callable(func), f"{func_name} should be callable"


class TestVTEImports:
    """Test that VTE functions are importable from behavior.vte."""

    @pytest.mark.parametrize("class_name", VTE_CLASSES)
    def test_class_importable_from_vte(self, class_name: str) -> None:
        """Verify {class_name} is importable from behavior.vte."""
        from neurospatial.behavior import vte

        cls = getattr(vte, class_name, None)
        assert cls is not None, f"{class_name} not found in behavior.vte"

    @pytest.mark.parametrize("func_name", VTE_FUNCTIONS)
    def test_function_importable_from_vte(self, func_name: str) -> None:
        """Verify {func_name} is importable from behavior.vte."""
        from neurospatial.behavior import vte

        func = getattr(vte, func_name, None)
        assert func is not None, f"{func_name} not found in behavior.vte"
        assert callable(func), f"{func_name} should be callable"

    def test_wrap_angle_importable_from_stats_circular(self) -> None:
        """Test wrap_angle is importable from stats.circular."""
        from neurospatial.stats.circular import wrap_angle

        assert callable(wrap_angle)


class TestBehaviorModuleExports:
    """Test that all functions are exported from behavior/__init__.py."""

    @pytest.mark.parametrize("name", DECISION_CLASSES + DECISION_FUNCTIONS)
    def test_decision_analysis_exports_from_behavior(self, name: str) -> None:
        """Verify {name} is exported from behavior module."""
        from neurospatial import behavior

        obj = getattr(behavior, name, None)
        assert obj is not None, f"{name} not exported from behavior module"

    @pytest.mark.parametrize("name", VTE_CLASSES + VTE_FUNCTIONS)
    def test_vte_exports_from_behavior(self, name: str) -> None:
        """Verify {name} is exported from behavior module."""
        from neurospatial import behavior

        obj = getattr(behavior, name, None)
        assert obj is not None, f"{name} not exported from behavior module"


class TestDecisionAnalysisFunctionality:
    """Test that decision analysis functions work correctly from new location."""

    def test_pre_decision_metrics_dataclass(self):
        """Test PreDecisionMetrics can be instantiated and has required fields."""
        from neurospatial.behavior.decisions import PreDecisionMetrics

        result = PreDecisionMetrics(
            mean_speed=15.0,
            min_speed=2.0,
            heading_mean_direction=0.5,
            heading_circular_variance=0.3,
            heading_mean_resultant_length=0.7,
            window_duration=1.0,
            n_samples=30,
        )

        assert result.mean_speed == pytest.approx(15.0)
        assert result.min_speed == pytest.approx(2.0)
        assert result.heading_mean_direction == pytest.approx(0.5)
        assert result.heading_circular_variance == pytest.approx(0.3)
        assert result.heading_mean_resultant_length == pytest.approx(0.7)
        assert result.window_duration == pytest.approx(1.0)
        assert result.n_samples == 30

    def test_pre_decision_metrics_suggests_deliberation(self):
        """Test suggests_deliberation method."""
        from neurospatial.behavior.decisions import PreDecisionMetrics

        # High variance + low speed -> deliberation
        result = PreDecisionMetrics(
            mean_speed=5.0,
            min_speed=0.5,
            heading_mean_direction=0.0,
            heading_circular_variance=0.7,
            heading_mean_resultant_length=0.3,
            window_duration=1.0,
            n_samples=30,
        )

        assert result.suggests_deliberation(
            variance_threshold=0.5, speed_threshold=10.0
        )

    def test_decision_boundary_metrics_dataclass(self):
        """Test DecisionBoundaryMetrics can be instantiated."""
        from neurospatial.behavior.decisions import DecisionBoundaryMetrics

        goal_labels = np.array([0, 0, 0, 1, 1, 1])
        distance_to_boundary = np.array([10.0, 5.0, 1.0, 1.0, 5.0, 10.0])
        crossing_times = [2.5]
        crossing_directions = [(0, 1)]

        result = DecisionBoundaryMetrics(
            goal_labels=goal_labels,
            distance_to_boundary=distance_to_boundary,
            crossing_times=crossing_times,
            crossing_directions=crossing_directions,
        )

        assert len(result.goal_labels) == 6
        assert result.n_crossings == 1

    def test_decision_analysis_result_dataclass(self):
        """Test DecisionAnalysisResult can be instantiated."""
        from neurospatial.behavior.decisions import (
            DecisionAnalysisResult,
            PreDecisionMetrics,
        )

        pre = PreDecisionMetrics(
            mean_speed=15.0,
            min_speed=2.0,
            heading_mean_direction=0.5,
            heading_circular_variance=0.3,
            heading_mean_resultant_length=0.7,
            window_duration=1.0,
            n_samples=30,
        )

        result = DecisionAnalysisResult(
            entry_time=5.0,
            pre_decision=pre,
            boundary=None,
            chosen_goal=1,
        )

        assert result.entry_time == pytest.approx(5.0)
        assert result.pre_decision is pre
        assert result.chosen_goal == 1

    def test_extract_pre_decision_window(self):
        """Test extract_pre_decision_window function."""
        from neurospatial.behavior.decisions import extract_pre_decision_window

        positions = np.column_stack([np.linspace(0, 100, 100), np.zeros(100)])
        times = np.linspace(0, 10, 100)

        window_pos, window_times = extract_pre_decision_window(
            positions, times, entry_time=5.0, window_duration=2.0
        )

        assert window_times[0] >= 2.9
        assert window_times[-1] < 5.0
        assert len(window_pos) == len(window_times)

    def test_pre_decision_speed_stats(self):
        """Test pre_decision_speed_stats function."""
        from neurospatial.behavior.decisions import pre_decision_speed_stats

        n_samples = 101
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)

        mean_speed, min_speed = pre_decision_speed_stats(positions, times)

        assert mean_speed > 0
        assert min_speed >= 0


class TestVTEFunctionality:
    """Test that VTE functions work correctly from new location."""

    def test_vte_trial_result_dataclass(self):
        """Test VTETrialResult can be instantiated."""
        from neurospatial.behavior.vte import VTETrialResult

        result = VTETrialResult(
            head_sweep_magnitude=2.5,
            z_head_sweep=1.2,
            mean_speed=15.0,
            min_speed=2.0,
            z_speed_inverse=0.8,
            vte_index=1.0,
            is_vte=True,
            window_start=0.0,
            window_end=1.0,
        )

        assert result.head_sweep_magnitude == 2.5
        assert result.idphi == 2.5  # alias
        assert result.z_idphi == 1.2  # alias
        assert result.is_vte is True

    def test_vte_session_result_dataclass(self):
        """Test VTESessionResult can be instantiated."""
        from neurospatial.behavior.vte import VTESessionResult

        session = VTESessionResult(
            trial_results=[],
            mean_head_sweep=1.5,
            std_head_sweep=1.0,
            mean_speed=22.5,
            std_speed=7.5,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

        assert session.mean_head_sweep == 1.5
        assert session.mean_idphi == 1.5  # alias
        assert session.std_idphi == 1.0  # alias

    def test_head_sweep_magnitude_basic(self):
        """Test head_sweep_magnitude function."""
        from neurospatial.behavior.vte import head_sweep_magnitude

        # Constant heading
        headings = np.zeros(20)
        result = head_sweep_magnitude(headings)
        assert result == 0.0

        # Alternating headings
        headings = np.array([0, np.pi / 4, 0, np.pi / 4, 0, np.pi / 4])
        result = head_sweep_magnitude(headings)
        expected = 5 * np.pi / 4
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_integrated_absolute_rotation_alias(self):
        """Test integrated_absolute_rotation is alias for head_sweep_magnitude."""
        from neurospatial.behavior.vte import (
            head_sweep_magnitude,
            integrated_absolute_rotation,
        )

        assert integrated_absolute_rotation is head_sweep_magnitude

    def test_compute_vte_index(self):
        """Test compute_vte_index function."""
        from neurospatial.behavior.vte import compute_vte_index

        # Equal weighting
        result = compute_vte_index(1.0, 1.0, alpha=0.5)
        assert result == 1.0

        # Head sweep only
        result = compute_vte_index(2.0, 0.5, alpha=1.0)
        assert result == 2.0

    def test_classify_vte(self):
        """Test classify_vte function."""
        from neurospatial.behavior.vte import classify_vte

        assert classify_vte(1.0, threshold=0.5) is True
        assert classify_vte(0.3, threshold=0.5) is False

    def test_normalize_vte_scores(self):
        """Test normalize_vte_scores function."""
        from neurospatial.behavior.vte import normalize_vte_scores

        head_sweeps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        speeds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        z_head_sweeps, _z_speed_inverse = normalize_vte_scores(head_sweeps, speeds)

        # Z-scores should have mean 0 and std 1
        np.testing.assert_allclose(np.mean(z_head_sweeps), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(z_head_sweeps), 1.0, atol=1e-10)

    def test_wrap_angle(self):
        """Test wrap_angle function (re-exported from stats.circular)."""
        from neurospatial.stats.circular import wrap_angle

        # Already in range
        np.testing.assert_allclose(wrap_angle(np.array([0.0])), [0.0])

        # Wrap positive overflow
        result = wrap_angle(np.array([2 * np.pi]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_head_sweep_from_positions(self):
        """Test head_sweep_from_positions function."""
        from neurospatial.behavior.vte import head_sweep_from_positions

        # Straight line
        n_samples = 20
        times = np.linspace(0, 2, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.ones(n_samples) * 50]
        )

        result = head_sweep_from_positions(positions, times, min_speed=5.0)
        assert result < 0.1  # Low for straight line

    def test_compute_vte_trial(self):
        """Test compute_vte_trial function."""
        from neurospatial.behavior.vte import VTETrialResult, compute_vte_trial

        n_samples = 30
        times = np.linspace(0, 3, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 50, n_samples), np.ones(n_samples) * 50]
        )

        result = compute_vte_trial(
            positions=positions,
            times=times,
            entry_time=2.0,
            window_duration=1.0,
            min_speed=1.0,
        )

        assert isinstance(result, VTETrialResult)
        assert result.z_head_sweep is None  # Single trial
        assert result.window_start == 1.0
        assert result.window_end == 2.0


class TestAllSymbolsExported:
    """Test that __all__ contains all expected symbols."""

    @pytest.mark.parametrize("name", DECISION_CLASSES + DECISION_FUNCTIONS)
    def test_decisions_module_exports(self, name: str) -> None:
        """Verify {name} is in decisions module __all__."""
        from neurospatial.behavior import decisions

        assert hasattr(decisions, name), f"Missing export: {name}"

    @pytest.mark.parametrize("name", VTE_CLASSES + VTE_FUNCTIONS)
    def test_vte_module_exports(self, name: str) -> None:
        """Verify {name} is in vte module __all__."""
        from neurospatial.behavior import vte

        assert hasattr(vte, name), f"Missing export: {name}"
