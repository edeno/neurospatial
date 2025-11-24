"""Tests for segmentation functions and public API exports."""


def test_detect_goal_directed_runs_exported():
    """Test that detect_goal_directed_runs is exported from main package."""
    from neurospatial import detect_goal_directed_runs

    # Verify it's callable
    assert callable(detect_goal_directed_runs)


def test_detect_runs_between_regions_exported():
    """Test that detect_runs_between_regions is exported from main package."""
    from neurospatial import detect_runs_between_regions

    # Verify it's callable
    assert callable(detect_runs_between_regions)


def test_segment_by_velocity_exported():
    """Test that segment_by_velocity is exported from main package."""
    from neurospatial import segment_by_velocity

    # Verify it's callable
    assert callable(segment_by_velocity)


def test_all_segmentation_functions_in_all():
    """Test that all three functions are in __all__."""
    import neurospatial

    assert "detect_goal_directed_runs" in neurospatial.__all__
    assert "detect_runs_between_regions" in neurospatial.__all__
    assert "segment_by_velocity" in neurospatial.__all__
