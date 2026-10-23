"""Tests for movement and body heading.

``heading_from_velocity`` (direction of the smoothed velocity) and
``heading_from_body_orientation`` (nose-tail vector).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_velocity_heading_uses_actual_intervals():
    """Unsmoothed velocity is the centered secant over real elapsed times."""
    from neurospatial._intervals import observed_interval_mask
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    times = np.array([0.0, 0.1, 0.3])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 4.0]])
    headings, speed = _movement_heading_and_speed(
        times,
        positions,
        observed_interval_mask(times, max_gap=0.5, epochs=None),
        smooth_window=0.0,
    )
    velocity = np.array(
        [
            (positions[1] - positions[0]) / 0.1,
            (positions[2] - positions[0]) / 0.3,
            (positions[2] - positions[1]) / 0.2,
        ]
    )
    np.testing.assert_allclose(headings, np.arctan2(velocity[:, 1], velocity[:, 0]))
    np.testing.assert_allclose(speed, np.hypot(velocity[:, 0], velocity[:, 1]))
    np.testing.assert_allclose(speed[[0, 2]], [10.0, 20.0])  # one-sided ends


def test_velocity_smoothing_is_separate_per_run(two_epoch_recording):
    from scipy.ndimage import gaussian_filter1d

    from neurospatial._intervals import observed_interval_mask
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    r = two_epoch_recording
    headings, speed = _movement_heading_and_speed(
        r.times,
        r.positions,
        observed_interval_mask(r.times, max_gap=0.5, epochs=None),
        smooth_window=0.04,  # two 50 Hz samples
    )
    for run in (slice(0, 5000), slice(5000, 10000)):
        velocity = np.gradient(r.positions[run], r.times[run], axis=0)
        velocity = gaussian_filter1d(velocity, 2.0, axis=0, mode="nearest")
        expected_heading = np.arctan2(velocity[:, 1], velocity[:, 0])
        expected_speed = np.hypot(velocity[:, 0], velocity[:, 1])
        np.testing.assert_allclose(
            headings[run], expected_heading, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(speed[run], expected_speed, rtol=1e-9, atol=1e-12)


def test_heading_from_velocity_ignores_pause(two_epoch_recording):
    from neurospatial.behavior import heading_from_velocity

    r = two_epoch_recording
    # Unsmoothed, the run's last velocity is its one-sided final difference.
    headings = heading_from_velocity(r.times, r.positions, smooth_window=0.0)
    previous = r.positions[4999] - r.positions[4998]
    jump = r.positions[5000] - r.positions[4999]
    expected = np.arctan2(previous[1], previous[0])
    np.testing.assert_allclose(headings[4999], expected, rtol=1e-12, atol=0)
    jump_heading = np.arctan2(jump[1], jump[0])
    assert not np.any(np.isclose(headings, jump_heading, rtol=1e-12, atol=1e-12))


def test_heading_from_velocity_isolated_sample_nan(two_epoch_recording):
    from neurospatial.behavior import heading_from_velocity

    r = two_epoch_recording
    selected = [0, 1, 2, 4999, 5000, 5001, 5002]  # 4999 sits alone
    headings = heading_from_velocity(r.times[selected], r.positions[selected])
    assert np.isnan(headings[3])
    assert np.isfinite(headings[[0, 1, 2, 4, 5, 6]]).all()


def test_slow_samples_have_no_heading_in_either_run(two_epoch_recording):
    from neurospatial.behavior import heading_from_velocity

    r = two_epoch_recording
    times = r.times[np.r_[np.arange(6), 5000 + np.arange(6)]]
    positions = np.c_[
        np.r_[0, 1, 2, 2, 2, 2, np.full(6, 20)], np.r_[np.zeros(6), np.arange(6)]
    ]
    # Smoothed speed in the first run falls from 32.7 to 13.7 cm/s as the
    # animal stops; samples 4-5 are below 20 cm/s. The second run moves North
    # at 50 cm/s throughout.
    headings = heading_from_velocity(times, positions, min_speed=20.0)
    np.testing.assert_allclose(headings[:4], 0.0, atol=0)
    assert np.isnan(headings[4:6]).all()
    np.testing.assert_allclose(headings[6:], np.pi / 2, atol=0)


def test_run_without_moving_anchors_stays_nan(two_epoch_recording):
    from neurospatial.behavior import heading_from_velocity

    r = two_epoch_recording
    times = r.times[np.r_[np.arange(6), 5000 + np.arange(6)]]
    positions = np.c_[
        np.r_[np.zeros(6), np.full(6, 20)], np.r_[np.zeros(6), np.arange(6)]
    ]
    headings = heading_from_velocity(times, positions, min_speed=5.0)
    assert np.isnan(headings[:6]).all()
    np.testing.assert_allclose(headings[6:], np.pi / 2, atol=0)


@pytest.fixture
def joined_recordings():
    """Two 30 Hz recordings joined on one clock at 60 s.

    The first runs East at 5 cm/s and stands still for its last second; the
    second starts 40 cm further west and runs North at 5 cm/s. The position
    jump at the join points West.
    """
    t_first = np.arange(0.0, 60.0, 1 / 30)
    t_second = 60.0 + np.arange(0.0, 60.0, 1 / 30)
    x_first = 10.0 + 5.0 * np.minimum(t_first, 59.0)
    first = np.c_[x_first, np.full_like(t_first, 50.0)]
    second = np.c_[
        np.full_like(t_second, x_first[-1] - 40.0), 50.0 + 5.0 * (t_second - 60)
    ]
    times = np.r_[t_first, t_second]
    positions = np.r_[first, second]
    break_before = np.zeros(times.size, dtype=bool)
    break_before[t_first.size] = True
    return times, positions, break_before, t_first.size


def _is_west(headings):
    return np.cos(headings) < -0.5


def test_heading_break_before_matches_separate_recordings(joined_recordings):
    from neurospatial.behavior import heading_from_velocity

    times, positions, break_before, n_first = joined_recordings
    joined = heading_from_velocity(
        times, positions, min_speed=2.0, break_before=break_before
    )
    first = heading_from_velocity(times[:n_first], positions[:n_first], min_speed=2.0)
    second = heading_from_velocity(times[n_first:], positions[n_first:], min_speed=2.0)
    np.testing.assert_allclose(
        joined, np.r_[first, second], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    near_join = np.abs(times - 60.0) < 2.0
    assert not _is_west(joined[near_join]).any()
    # The first recording stands still for its last second: no heading there.
    assert np.isnan(joined[n_first - 15 : n_first]).all()
    np.testing.assert_allclose(
        joined[n_first:][near_join[n_first:]], np.pi / 2, atol=1e-9
    )


def test_heading_without_break_before_bridges_the_join(joined_recordings):
    """Without a declared break the jump reads as westward running.

    Touching ``epochs`` rows merge into one window, so they do not separate
    the recordings either; ``break_before`` is what prevents the bridge.
    """
    from neurospatial.behavior import heading_from_velocity

    times, positions, _, _ = joined_recordings
    near_join = np.abs(times - 60.0) < 2.0
    bridged = heading_from_velocity(times, positions, min_speed=2.0)
    assert _is_west(bridged[near_join]).sum() > 10
    touching = np.array([[0.0, 60.0], [60.0, 121.0]])
    np.testing.assert_array_equal(
        heading_from_velocity(times, positions, min_speed=2.0, epochs=touching),
        bridged,
    )


@pytest.mark.parametrize(
    "break_before",
    [np.zeros(5, dtype=bool), np.zeros(6, dtype=int), np.zeros((6, 1), dtype=bool)],
)
def test_heading_rejects_bad_break_before(break_before):
    from neurospatial.behavior import heading_from_velocity

    positions = np.c_[np.arange(6.0), np.zeros(6)]
    with pytest.raises(ValueError, match="break_before") as caught:
        heading_from_velocity(np.arange(6) * 0.1, positions, break_before=break_before)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize(
    ("times", "labels_why"),
    [
        (np.array(0.1), True),
        (np.zeros((3, 1)), True),
        (np.array([0.0, 0.1]), True),
        # The shared repeated-timestamp message states why without a label.
        (np.array([0.0, 0.0, 0.1]), False),
        (np.array([0.2, 0.1, 0.0]), True),
        (np.array([0.0, np.nan, 0.2]), True),
        (np.array([0.0, np.inf, 0.2]), True),
        (np.array(["zero", "one", "two"]), True),
    ],
)
def test_heading_from_velocity_rejects_bad_times(times, labels_why):
    from neurospatial.behavior import heading_from_velocity

    positions = np.c_[[0.0, 1.0, 2.0], np.zeros(3)]
    with pytest.raises(ValueError) as caught:
        heading_from_velocity(times, positions)
    message = str(caught.value)
    assert "times" in message
    assert ("Why:" in message) is labels_why
    assert "Fix:" in message


@pytest.mark.parametrize(
    "positions",
    [np.arange(4.0), np.arange(4.0)[:, None], np.array(0.0), np.zeros((4, 2, 2))],
)
def test_heading_rejects_position_shapes_before_broadcast(positions):
    from neurospatial.behavior import heading_from_velocity

    with pytest.raises(ValueError) as caught:
        heading_from_velocity(np.arange(4) * 0.1, positions)
    message = str(caught.value)
    assert "positions" in message
    assert "shape" in message
    assert "Why:" in message
    assert "Fix:" in message


class TestHeadingFromVelocity:
    """Tests for heading_from_velocity function."""

    def test_smooth_trajectory(self):
        """Heading from smooth trajectory matches expected direction."""
        from neurospatial.behavior import heading_from_velocity

        # Straight line moving East
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([t * 10, np.zeros_like(t)])  # x = 0 to 100

        headings = heading_from_velocity(positions=positions, times=t)

        # All headings should be 0 (East), except possibly at boundaries
        assert_allclose(headings[10:-10], 0.0, atol=0.1)

    def test_heading_northward(self):
        """Heading is π/2 for northward motion."""
        from neurospatial.behavior import heading_from_velocity

        # Straight line moving North
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([np.zeros_like(t), t * 10])

        headings = heading_from_velocity(positions=positions, times=t)

        # All headings should be π/2 (North)
        assert_allclose(headings[10:-10], np.pi / 2, atol=0.1)

    def test_stationary_periods_have_no_heading(self):
        """Samples slower than min_speed get NaN; moving samples keep East."""
        from neurospatial.behavior import heading_from_velocity

        # Moving, then stationary, then moving again
        n = 100
        positions = np.zeros((n, 2))

        # First 30 points: moving East
        positions[:30, 0] = np.arange(30)
        # Points 30-70: stationary at x=30
        positions[30:70, 0] = 30
        # Points 70-100: moving East again
        positions[70:, 0] = 30 + np.arange(30)

        headings = heading_from_velocity(
            positions=positions,
            times=np.arange(len(positions)) * 0.1,
            min_speed=0.5,
            smooth_window=0.2,
        )

        # The smoothed speed is below 0.5 at samples 34-66 (exactly 0 at 23 of
        # them); those have no heading, and every other sample is East.
        slow = np.zeros(n, dtype=bool)
        slow[34:67] = True
        assert np.isnan(headings[slow]).all()
        assert_allclose(headings[~slow], 0.0, atol=1e-12)

    def test_all_speeds_low_raises_by_default(self):
        """Every sample below min_speed raises (fail loud), not silent all-NaN.

        Regression: returning an all-NaN heading array flowed silently into
        egocentric / object-vector analyses as a false negative (e.g. a slow
        recording -> is_object_vector_cell False). The all-below case must fail
        loud by default and only return NaN when allow_all_nan is opted into.
        """
        from neurospatial.behavior import heading_from_velocity

        # Barely moving trajectory -> every speed below the threshold.
        positions = np.random.RandomState(42).randn(100, 2) * 0.001

        with pytest.raises(ValueError, match=r"(?i)min_speed|speed"):
            heading_from_velocity(
                positions=positions,
                times=np.arange(len(positions)) * 0.1,
                min_speed=10.0,
            )

        # Opt-in escape hatch: warns and returns the all-NaN array.
        with pytest.warns(UserWarning, match="speed"):
            headings = heading_from_velocity(
                positions=positions,
                times=np.arange(len(positions)) * 0.1,
                min_speed=10.0,
                allow_all_nan=True,
            )
        assert np.all(np.isnan(headings))

    def test_still_recording_fix_does_not_suggest_lowering_zero_min_speed(self):
        """At min_speed=0 the only remedies are allow_all_nan or a pose heading."""
        from neurospatial.behavior import heading_from_velocity

        positions = np.zeros((10, 2))
        times = np.arange(10) * 0.1
        with pytest.raises(ValueError, match="or zero") as caught:
            heading_from_velocity(times, positions)
        fix = str(caught.value).split("Fix:")[1]
        assert "lower min_speed" not in fix
        assert "allow_all_nan=True" in fix
        assert "heading_from_body_orientation" in fix

        with pytest.raises(ValueError) as caught:
            heading_from_velocity(times, positions, min_speed=2.0)
        assert "lower min_speed" in str(caught.value).split("Fix:")[1]

    def test_single_sample_has_no_heading(self):
        """One sample has a position but no movement, so its heading is NaN."""
        from neurospatial.behavior import heading_from_velocity

        positions = np.array([[0.0, 0.0]])  # Only 1 point

        headings = heading_from_velocity(positions=positions, times=np.array([0.0]))
        assert headings.shape == (1,)
        assert np.isnan(headings).all()

    def test_empty_input_raises(self):
        from neurospatial.behavior import heading_from_velocity

        with pytest.raises(ValueError, match="no samples"):
            heading_from_velocity(np.array([]), np.empty((0, 2)))

    def test_repeated_timestamp_raises(self):
        from neurospatial.behavior import heading_from_velocity

        positions = np.c_[np.arange(4.0), np.zeros(4)]
        with pytest.raises(ValueError, match="strictly increasing"):
            heading_from_velocity(np.array([0.0, 0.1, 0.1, 0.2]), positions)

    def test_smoothing_reduces_noise(self):
        """Larger smoothing sigma reduces heading noise."""
        from neurospatial.behavior import heading_from_velocity

        # Noisy trajectory moving mostly East
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 100)
        noise = rng.normal(0, 0.5, size=(100, 2))
        positions = np.column_stack([t * 10, np.zeros_like(t)]) + noise

        headings_no_smooth = heading_from_velocity(
            positions=positions, times=np.arange(len(positions)) * 0.1, smooth_window=0
        )
        headings_smooth = heading_from_velocity(
            positions=positions,
            times=np.arange(len(positions)) * 0.1,
            smooth_window=0.5,
        )

        # Smoothed version should have less variance
        var_no_smooth = np.nanvar(headings_no_smooth[10:-10])
        var_smooth = np.nanvar(headings_smooth[10:-10])
        assert var_smooth < var_no_smooth


def _facing(angles):
    """Unit nose positions at ``angles`` (radians) with the tail at the origin."""
    angles = np.asarray(angles, dtype=float)
    return np.column_stack([np.cos(angles), np.sin(angles)]), np.zeros((angles.size, 2))


class TestHeadingFromBodyOrientation:
    """Body heading: nose-tail direction, short gaps filled inside one recording."""

    def test_northward(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.full(50, np.pi / 2))
        headings = heading_from_body_orientation(np.arange(50) / 30, nose, tail)
        assert headings.shape == (50,)
        assert_allclose(headings, np.pi / 2, atol=1e-12)

    def test_short_gap_filled(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(6))
        nose[2:4] = np.nan
        headings = heading_from_body_orientation(np.arange(6) * 0.1, nose, tail)
        np.testing.assert_array_equal(headings, np.zeros(6))

    def test_interpolation_is_linear_in_time(self):
        from neurospatial.behavior import heading_from_body_orientation

        times = np.array([0.0, 0.1, 0.4, 0.5])
        nose, tail = _facing([0.0, 0.0, 0.0, np.pi / 2])
        nose[1:3] = np.nan
        headings = heading_from_body_orientation(times, nose, tail)
        # By sample index the fill would be pi/6 and pi/3.
        assert_allclose(
            headings, [0.0, np.pi / 10, 2 * np.pi / 5, np.pi / 2], atol=1e-12
        )

    def test_long_gap_is_nan(self):
        from neurospatial.behavior import heading_from_body_orientation

        times = np.arange(10) * 0.1
        nose, tail = _facing(np.zeros(10))
        nose[2:8] = np.nan  # anchors 1 and 8 are 0.7 s apart
        headings = heading_from_body_orientation(times, nose, tail)
        assert np.isnan(headings[2:8]).all()
        assert np.isfinite(headings[[0, 1, 8, 9]]).all()

        unlimited = heading_from_body_orientation(times, nose, tail, max_gap=None)
        np.testing.assert_array_equal(unlimited, np.zeros(10))

    def test_recording_break_is_not_bridged(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(6))
        nose[3] = np.nan
        breaks = np.zeros(6, dtype=bool)
        breaks[4] = True
        headings = heading_from_body_orientation(
            np.arange(6) * 0.1, nose, tail, break_before=breaks
        )
        np.testing.assert_array_equal(headings, [0.0, 0.0, 0.0, np.nan, 0.0, 0.0])

    def test_epoch_edge_is_not_bridged(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(10))
        nose[4] = np.nan
        headings = heading_from_body_orientation(
            np.arange(10) * 0.1, nose, tail, epochs=[(0.0, 0.45)]
        )
        np.testing.assert_array_equal(headings[:4], np.zeros(4))
        assert np.isnan(headings[4:]).all()

    def test_no_extrapolation_past_the_ends(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(6))
        nose[0] = nose[-1] = np.nan
        headings = heading_from_body_orientation(np.arange(6) * 0.1, nose, tail)
        np.testing.assert_array_equal(headings, [np.nan, 0, 0, 0, 0, np.nan])

    def test_zero_length_body_vector_is_missing(self):
        """arctan2(0, 0) = 0 (East) is not a direction; the frame is filled."""
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.full(5, np.pi / 2))
        nose[2] = tail[2]
        headings = heading_from_body_orientation(np.arange(5) * 0.1, nose, tail)
        assert_allclose(headings, np.pi / 2, atol=1e-12)

    def test_fill_across_pi_takes_the_shorter_arc(self):
        from neurospatial.behavior import heading_from_body_orientation

        angles = np.linspace(0.9 * np.pi, 1.1 * np.pi, 100)
        nose, tail = _facing(angles)
        nose[45:55] = np.nan
        headings = heading_from_body_orientation(np.arange(100) / 100, nose, tail)
        expected = (angles + np.pi) % (2 * np.pi) - np.pi
        diff = np.angle(np.exp(1j * (headings - expected)))
        assert_allclose(diff, 0.0, atol=1e-12)

    def test_gap_free_is_the_nose_tail_direction(self):
        from neurospatial.behavior import heading_from_body_orientation

        rng = np.random.default_rng(7)
        nose = rng.normal(size=(200, 2)) * 10
        tail = rng.normal(size=(200, 2)) * 10
        headings = heading_from_body_orientation(np.arange(200) / 30, nose, tail)
        np.testing.assert_array_equal(
            headings, np.arctan2(nose[:, 1] - tail[:, 1], nose[:, 0] - tail[:, 0])
        )

    def test_extra_columns_are_ignored(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.full(4, np.pi / 2))
        nose3 = np.column_stack([nose, np.arange(4.0)])
        tail3 = np.column_stack([tail, np.zeros(4)])
        headings = heading_from_body_orientation(np.arange(4) * 0.1, nose3, tail3)
        assert_allclose(headings, np.pi / 2, atol=1e-12)

    def test_all_samples_outside_epochs_warn_and_are_nan(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(5))
        with pytest.warns(UserWarning, match="no missing frame is filled"):
            headings = heading_from_body_orientation(
                np.arange(5) * 0.1, nose, tail, epochs=[(5.0, 6.0)]
            )
        assert np.isnan(headings).all()

    def test_all_intervals_excluded_keeps_observed_headings(self):
        """At 1 Hz every interval exceeds max_gap: no fill, observed frames kept."""
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.linspace(-1.0, 1.0, 10))
        with pytest.warns(UserWarning) as record:
            headings = heading_from_body_orientation(np.arange(10.0), nose, tail)
        message = str(record[0].message)
        assert "so no missing frame is filled" in message
        assert "heading is NaN" not in message
        assert_allclose(headings, np.linspace(-1.0, 1.0, 10), atol=1e-12)

        nose[4] = np.nan
        with pytest.warns(UserWarning, match="no missing frame is filled"):
            headings = heading_from_body_orientation(np.arange(10.0), nose, tail)
        assert np.isnan(headings[4])
        assert np.isfinite(np.delete(headings, 4)).all()

    def test_no_observed_sample_raises(self):
        from neurospatial.behavior import heading_from_body_orientation

        nose, tail = _facing(np.zeros(5))
        nose[:] = np.nan
        nose[2] = tail[2]  # a coincident pair is not observed either
        with pytest.raises(ValueError) as error:
            heading_from_body_orientation(np.arange(5) * 0.1, nose, tail)
        message = str(error.value)
        assert message.startswith("heading_from_body_orientation()")
        assert "heading_from_velocity" in message

    @pytest.mark.parametrize("case", ["empty", "tail_shape", "decreasing", "1d"])
    def test_invalid_inputs_raise_naming_the_call(self, case):
        from neurospatial.behavior import heading_from_body_orientation

        times = np.arange(5) * 0.1
        nose, tail = _facing(np.zeros(5))
        if case == "empty":
            times, nose, tail = times[:0], nose[:0], tail[:0]
        elif case == "tail_shape":
            tail = tail[:, :1]
        elif case == "decreasing":
            times = times[::-1]
        else:
            nose, tail = nose[:, 0], tail[:, 0]
        with pytest.raises(ValueError, match="heading_from_body_orientation") as error:
            heading_from_body_orientation(times, nose, tail)
        if case == "empty":
            what, why, fix = str(error.value).split("\n")
            assert what == (
                "heading_from_body_orientation() got no samples: times, nose and "
                "tail are empty."
            )
            assert why == "Why: there is no observation to measure."
            assert fix.startswith("Fix: pass at least one sample")


class TestHeadingFromVelocityWestward:
    """Pin the westward-+pi convention and ±pi stability across brief stops."""

    def test_heading_from_velocity_westward_returns_pi_not_negative_pi(self):
        """Purely westward motion returns +pi (per arctan2), never -pi."""
        from neurospatial.behavior import heading_from_velocity

        # x decreasing from 100 to 0 in equal steps, constant y.
        x = np.linspace(100.0, 0.0, 11)
        y = np.full_like(x, 50.0)
        positions = np.column_stack([x, y])

        headings = heading_from_velocity(
            positions=positions, times=np.arange(len(positions)) * 0.1, min_speed=5.0
        )

        assert np.allclose(headings, np.pi, atol=1e-6)

    def test_heading_at_pi_boundary_with_brief_stop(self):
        """Westward samples around a brief stop are +pi; the stop has no heading.

        Unsmoothed, the three samples in the middle of the stop have zero
        velocity. They are NaN, and no moving sample flips to -pi.
        """
        from neurospatial.behavior import heading_from_velocity

        # Westward, a 5-sample stop (constant x), then westward again.
        x = np.array([100, 90, 80, 70, 70, 70, 70, 70, 60, 50, 40, 30], dtype=float)
        y = np.full_like(x, 50.0)
        positions = np.column_stack([x, y])

        headings = heading_from_velocity(
            positions=positions,
            times=np.arange(len(positions)) * 0.1,
            min_speed=5.0,
            smooth_window=0.0,
        )

        stopped = np.zeros(len(x), dtype=bool)
        stopped[4:7] = True
        assert np.isnan(headings[stopped]).all()
        assert np.allclose(headings[~stopped], np.pi, atol=1e-6)
        # Explicitly: no sample collapsed to -pi.
        assert not np.any(np.isclose(headings, -np.pi, atol=1e-6))


class TestHeadingFromVelocityGuards:
    """heading_from_velocity must reject non-increasing timestamps and non-finite positions."""

    def test_heading_from_velocity_rejects_nonincreasing_times(self):
        """Duplicate and descending timestamps raise instead of NaN/flipped headings."""
        from neurospatial.behavior import heading_from_velocity

        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        with pytest.raises(ValueError, match="times"):
            heading_from_velocity(
                positions=positions, times=np.arange(len(positions)) * 0.0
            )

        with pytest.raises(ValueError, match="times"):
            heading_from_velocity(
                positions=positions, times=np.arange(len(positions)) * -0.1
            )

    @pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
    def test_heading_from_velocity_nonfinite_rows_have_no_heading(self, missing):
        """A row with a non-finite coordinate is missing: NaN there, East elsewhere."""
        from neurospatial.behavior import heading_from_velocity

        positions = np.c_[np.arange(6.0), np.zeros(6)]
        positions[2, 0] = missing
        headings = heading_from_velocity(
            np.arange(6) * 0.1, positions, smooth_window=0.0
        )
        np.testing.assert_array_equal(headings, [0.0, 0.0, np.nan, 0.0, 0.0, 0.0])

    def test_heading_from_velocity_descending_times_would_flip(self):
        """The timestamp guard blocks descending times that would rotate headings by pi.

        For an eastward trajectory ascending times give heading ~= 0;
        descending times would silently negate the velocity and return ~= pi.
        """
        from neurospatial.behavior import heading_from_velocity

        positions = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])

        headings = heading_from_velocity(
            positions=positions, times=np.arange(len(positions)) * 0.1, min_speed=1.0
        )
        assert np.allclose(headings[:-1], 0.0, atol=1e-8)

        # Descending times (which would return ~= pi) must instead raise.
        with pytest.raises(ValueError, match="times"):
            heading_from_velocity(
                positions=positions,
                times=np.arange(len(positions)) * -0.1,
                min_speed=1.0,
            )


@pytest.mark.parametrize("turn", [np.pi / 2, np.pi - 0.1, np.pi])
def test_heading_interpolation_is_uniform_on_shorter_arc(turn):
    from neurospatial.behavior.trajectory import _interpolate_heading_circular

    headings = np.array([0.0, np.nan, np.nan, np.nan, turn])
    filled = _interpolate_heading_circular(np.arange(5.0), headings, np.isnan(headings))
    np.testing.assert_allclose(
        filled[1:-1], turn * np.array([0.25, 0.5, 0.75]), atol=1e-12
    )


def test_heading_interpolation_across_pi_wrap():
    from neurospatial.behavior.trajectory import _interpolate_heading_circular

    headings = np.array([np.pi - 0.1, np.nan, -np.pi + 0.1])
    filled = _interpolate_heading_circular(np.arange(3.0), headings, np.isnan(headings))
    assert abs(filled[1]) == pytest.approx(np.pi, abs=1e-12)


def test_antipodal_turn_has_no_jump_to_zero():
    from neurospatial.behavior.trajectory import _interpolate_heading_circular

    headings = np.array([np.pi / 2, np.nan, np.nan, np.nan, -np.pi / 2])
    filled = _interpolate_heading_circular(np.arange(5.0), headings, np.isnan(headings))
    np.testing.assert_allclose(filled[1:-1], [np.pi / 4, 0.0, -np.pi / 4], atol=1e-12)


def test_heading_from_velocity_stop_has_no_heading():
    from neurospatial.behavior import compute_speed, heading_from_velocity

    # East, a 0.3 s stop, then West: the stop's direction is not observed, so
    # it gets no heading instead of one interpolated between East and West.
    positions = np.column_stack([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], np.zeros(7)])
    times = np.arange(len(positions)) * 0.1
    speed = compute_speed(times, positions)
    np.testing.assert_allclose(
        speed, [8.204, 4.977, 1.480, 1.480, 4.977, 8.204, 9.684], atol=6e-4
    )
    heading = heading_from_velocity(times, positions, min_speed=2.0)
    np.testing.assert_allclose(
        heading, [0.0, 0.0, np.nan, np.nan, np.pi, np.pi, np.pi], atol=1e-12
    )


def test_heading_from_velocity_zero_speed_has_no_heading():
    """arctan2(0, 0) = 0 (East) is not a direction; zero speed gives NaN."""
    from neurospatial.behavior import compute_speed, heading_from_velocity

    positions = np.c_[[0.0, 1.0, 1.0, 1.0, 2.0], np.zeros(5)]
    times = np.arange(5) * 0.1
    np.testing.assert_allclose(
        compute_speed(times, positions, smooth_window=0.0), [10, 5, 0, 5, 10]
    )
    heading = heading_from_velocity(times, positions, smooth_window=0.0)
    np.testing.assert_array_equal(heading, [0.0, 0.0, np.nan, 0.0, 0.0])


def test_heading_from_velocity_nan_row_is_missing_not_bridged():
    from neurospatial.behavior import heading_from_velocity

    times = np.arange(8) * 0.1
    positions = np.c_[np.arange(8.0), np.zeros(8)]
    positions[3] = np.nan
    heading = heading_from_velocity(times, positions, smooth_window=0.0)
    np.testing.assert_array_equal(heading, [0, 0, 0, np.nan, 0, 0, 0, 0])

    # With rows 3 and 5 missing, sample 4 touches no observed interval.
    positions[5] = np.nan
    heading = heading_from_velocity(times, positions, smooth_window=0.0)
    np.testing.assert_array_equal(heading, [0, 0, 0, np.nan, np.nan, np.nan, 0, 0])


def test_heading_from_velocity_nan_rows_match_separate_runs():
    """Missing rows split the path into runs analyzed as separate recordings."""
    from neurospatial.behavior import heading_from_velocity

    times = np.arange(200) / 50  # s, 50 Hz
    x = 10.0 * np.minimum(times, 2.0)  # cm: East for 2 s ...
    y = 10.0 * np.maximum(times - 2.0, 0.0)  # ... then North
    positions = np.c_[x, y]
    missing = np.arange(98, 103)
    positions[missing] = np.nan
    joined = heading_from_velocity(times, positions)
    before, after = slice(0, 98), slice(103, 200)
    expected = np.full(200, np.nan)
    expected[before] = heading_from_velocity(times[before], positions[before])
    expected[after] = heading_from_velocity(times[after], positions[after])
    np.testing.assert_array_equal(joined, expected)
    assert np.isfinite(joined[before]).all() and np.isfinite(joined[after]).all()


def test_nan_anchor_not_propagated():
    from neurospatial.behavior.trajectory import _interpolate_heading_circular

    headings = np.array([0.0, 0.0, np.nan, np.pi / 2])
    fill = np.array([False, True, False, False])
    filled = _interpolate_heading_circular(np.arange(4.0), headings, fill)
    assert np.isfinite(filled[1])
    assert np.isnan(filled[2])


@pytest.mark.parametrize("short", ["nose", "tail"])
def test_body_orientation_length_mismatch_names_the_keypoint(short):
    from neurospatial.behavior import heading_from_body_orientation

    nose, tail = _facing(np.zeros(6))
    keypoints = {"nose": nose, "tail": tail}
    keypoints[short] = keypoints[short][:4]
    with pytest.raises(ValueError) as excinfo:
        heading_from_body_orientation(np.arange(6) * 0.1, **keypoints)
    message = str(excinfo.value)
    assert f"{short} has 4" in message
    assert "times has 6" in message
    assert "positions" not in message
    assert "Why:" in message and "Fix:" in message
