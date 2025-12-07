"""Tests for segmentation functions and submodule imports."""


def test_detect_goal_directed_runs_from_behavior():
    """Test that detect_goal_directed_runs is importable from behavior submodule."""
    from neurospatial.behavior import detect_goal_directed_runs

    # Verify it's callable
    assert callable(detect_goal_directed_runs)


def test_detect_runs_between_regions_from_behavior():
    """Test that detect_runs_between_regions is importable from behavior submodule."""
    from neurospatial.behavior import detect_runs_between_regions

    # Verify it's callable
    assert callable(detect_runs_between_regions)


def test_segment_by_velocity_from_behavior():
    """Test that segment_by_velocity is importable from behavior submodule."""
    from neurospatial.behavior import segment_by_velocity

    # Verify it's callable
    assert callable(segment_by_velocity)


def test_all_segmentation_functions_in_behavior_all():
    """Test that all three functions are in behavior.__all__."""
    from neurospatial import behavior

    assert "detect_goal_directed_runs" in behavior.__all__
    assert "detect_runs_between_regions" in behavior.__all__
    assert "segment_by_velocity" in behavior.__all__
