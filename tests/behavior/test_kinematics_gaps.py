"""Kinematic intervals preserve observations without bridging recording gaps."""

import numpy as np

from neurospatial.environment.trajectory import observed_interval_mask


def test_interval_velocity_has_physical_units_and_masks_gap():
    from neurospatial.behavior._kinematics import interval_velocity

    times = np.array([0.0, 0.2, 0.4, 2.4, 2.6])
    positions = np.column_stack([[0.0, 2.0, 6.0, 100.0, 101.0], np.zeros(5)])
    mask = observed_interval_mask(times, max_gap=0.5, epochs=None)
    result = interval_velocity(times, positions, mask)
    assert result.shape == (4, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [[10, 0], [20, 0], [np.nan, np.nan], [5, 0]])


def test_interval_velocity_is_empty_without_intervals():
    from neurospatial.behavior._kinematics import interval_velocity

    for n_samples in (0, 1):
        result = interval_velocity(
            np.zeros(n_samples), np.zeros((n_samples, 2)), np.zeros(0, dtype=bool)
        )
        assert result.shape == (0, 2)
