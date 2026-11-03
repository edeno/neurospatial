from __future__ import annotations

import numpy as np
import pytest
from _formats import read_res_clu, read_whl, two_led_heading, whl_times
from _record import heading_vs_movement


def test_whl_reader_marks_untracked_as_nan(tmp_path):
    path = tmp_path / "s.whl"
    path.write_text(
        "10.5 20.0 12.5 20.0\n"
        "-1 -1 13.0 21.0\n"  # front LED lost
        "11.0 -1 13.5 21.5\n"  # one coordinate lost loses the LED
        "-1 -1 -1 -1\n"  # both lost
        "12.0 22.0 14.0 22.0\n"
    )
    frames = read_whl(path)
    expected = np.array(
        [
            [10.5, 20.0, 12.5, 20.0],
            [np.nan, np.nan, 13.0, 21.0],
            [np.nan, np.nan, 13.5, 21.5],
            [np.nan, np.nan, np.nan, np.nan],
            [12.0, 22.0, 14.0, 22.0],
        ]
    )
    np.testing.assert_array_equal(frames, expected)
    np.testing.assert_allclose(whl_times(3, 20000.0), [0.0, 0.0256, 0.0512])


def test_whl_reader_refuses_an_unpaired_column(tmp_path):
    path = tmp_path / "s.whl"
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="3 columns"):
        read_whl(path)


def test_two_led_heading_frame():
    back = np.array([[0.0, 0.0], [5.0, 5.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    front = back + np.array(
        [[3.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -4.0], [np.nan, np.nan]]
    )
    np.testing.assert_allclose(
        two_led_heading(front, back)[:4], [0.0, np.pi / 2, np.pi, -np.pi / 2]
    )
    assert np.isnan(two_led_heading(front, back)[4])


def test_two_led_heading_has_no_direction_where_the_leds_coincide():
    # One blob given to both LEDs: arctan2(0, 0) would say 0 (east).
    back = np.array([[5.0, 5.0], [5.0, 5.0], [0.0, 0.0]])
    front = np.array([[5.0, 5.0], [6.0, 5.0], [0.0, 0.0]])
    np.testing.assert_array_equal(two_led_heading(front, back), [np.nan, 0.0, np.nan])


def test_two_led_heading_matches_movement_only_with_leds_in_order():
    # An animal running a circle with the front LED ahead of the back LED.
    times = np.arange(0.0, 20.0, 0.025)
    angle = 0.5 * times
    centre = 50.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    ahead = np.column_stack([-np.sin(angle), np.cos(angle)])
    front, back = centre + 2.0 * ahead, centre - 2.0 * ahead
    mean, spread = heading_vs_movement(
        times, centre, two_led_heading(front, back), min_speed=10.0
    )
    assert abs(mean) < 0.05
    assert spread < 0.05
    swapped, _ = heading_vs_movement(
        times, centre, two_led_heading(back, front), min_speed=10.0
    )
    assert abs(abs(swapped) - np.pi) < 0.05


def test_res_clu_pairs_spikes_with_clusters(tmp_path):
    (tmp_path / "s.res.1").write_text("100\n250\n400\n")
    (tmp_path / "s.clu.1").write_text("4\n2\n0\n3\n")
    samples, clusters = read_res_clu(tmp_path / "s.res.1", tmp_path / "s.clu.1")
    np.testing.assert_array_equal(samples, [100, 250, 400])
    np.testing.assert_array_equal(clusters, [2, 0, 3])
    (tmp_path / "s.clu.2").write_text("3\n2\n")
    with pytest.raises(ValueError, match="1 spikes"):
        read_res_clu(tmp_path / "s.res.1", tmp_path / "s.clu.2")
