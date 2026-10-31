"""Unit tests for the shared input-validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial._validation import validate_finite, validate_lengths


def test_swapped_times_positions_names_swap():
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError) as caught:
        validate_times_positions(np.zeros((3, 2)), np.arange(2.0), call="f")
    message = str(caught.value)
    assert "did you pass positions before times" in message
    assert "f(..., times, positions, ...)" in message
    assert "same length" in message
    assert "got shape (3, 2)" in message
    assert "Why:" in message and "Fix:" in message
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 2


def test_swapped_one_dimensional_arguments_name_swap():
    """Linear-track positions passed as times, and times passed as positions."""
    from neurospatial._validation import validate_times_positions

    track = [5.0, 7.0, 6.0, 9.0]
    with pytest.raises(ValueError, match="did you pass positions before times"):
        validate_times_positions(track, [0.0, 0.1, 0.2, 0.3], call="f")


@pytest.mark.parametrize(
    "times, positions",
    [
        ([0.0, np.nan, 0.2], [0.0, 1.0, 2.0]),
        ([0.0, 0.1, 0.2], [0.0, 1.0, 2.0, 3.0]),
        ([0.0, 0.2, 0.1], [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        ([0.0, 0.2, 0.1], [0.0, np.nan, 2.0]),
    ],
    ids=[
        "nan time, track positions",
        "length mismatch",
        "unsorted, 2-D positions",
        "unsorted, positions with nan",
    ],
)
def test_swap_is_not_suggested_without_evidence(times, positions):
    """Positions that never decrease are normal on a linear track."""
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError) as caught:
        validate_times_positions(times, positions, call="f")
    message = str(caught.value)
    assert "positions before times" not in message
    assert "Fix: pass times as a sorted 1-D array" in message


def test_unsorted_times_reports_first_decrease():
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError) as caught:
        validate_times_positions([0, 1, 0.5, 2], np.zeros((4, 2)), call="f")
    assert "index 1" in str(caught.value)
    assert "1.0 -> 0.5" in str(caught.value)


@pytest.mark.parametrize("times", [[np.nan], [np.inf], [-np.inf]])
def test_single_nonfinite_timestamp_rejected(times):
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError, match=r"1 non-finite.*index 0"):
        validate_times_positions(times, [[0.0]], call="f")


def test_times_positions_preserves_duplicates_and_missing_positions():
    from neurospatial._validation import validate_times_positions

    t, p = validate_times_positions([0, 0, 1], [[0], [np.nan], [2]], call="f")
    np.testing.assert_array_equal(t, [0, 0, 1])
    np.testing.assert_array_equal(p, [[0], [np.nan], [2]])
    assert t.dtype == p.dtype == np.float64


def _refusing_calls():
    """Entry points of every analysis family that take sampled coordinates."""
    from neurospatial import Environment
    from neurospatial.behavior import compute_speed
    from neurospatial.encoding import compute_directional_rate, compute_spatial_rate

    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 10, (500, 2)), bin_size=2.0)
    times = np.arange(20) * 0.1
    positions = rng.uniform(1, 9, (20, 2))
    positions[4, 1] = np.inf
    headings = np.zeros(20)
    headings[4] = -np.inf
    spikes = np.array([0.5, 1.0])
    return {
        "from_samples": lambda: Environment.from_samples(positions, bin_size=2.0),
        "occupancy": lambda: env.occupancy(times, positions),
        "spatial_rate": lambda: compute_spatial_rate(env, spikes, times, positions),
        "directional_rate": lambda: compute_directional_rate(spikes, times, headings),
        "compute_speed": lambda: compute_speed(times, positions),
    }


@pytest.mark.parametrize(
    "call",
    ["from_samples", "occupancy", "spatial_rate", "directional_rate", "compute_speed"],
)
def test_infinite_samples_are_refused_everywhere(call):
    """NaN is the only missing-sample marker; inf is refused, not dropped."""
    with pytest.raises(
        ValueError, match=r"1 infinite value\(s\) in (positions|headings)"
    ):
        _refusing_calls()[call]()


def test_numeric_conversion_errors_are_named_together():
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError) as caught:
        validate_times_positions(["bad-time"], [["bad-position"]], call="f")
    message = str(caught.value)
    assert "times" in message and "positions" in message
    assert "bad-time" in message and "bad-position" in message
    assert "Why:" in message and "Fix:" in message
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 2


@pytest.mark.parametrize(
    "times, positions, other_problem",
    [
        ([1, 0], [["bad"], ["bad"]], "monotonically non-decreasing"),
        (["bad"], np.zeros((1, 2, 1)), "got shape (1, 2, 1)"),
    ],
)
def test_conversion_errors_keep_other_argument_problems(
    times, positions, other_problem
):
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError) as caught:
        validate_times_positions(times, positions, call="f")
    assert other_problem in str(caught.value)
    assert "numeric" in str(caught.value)


def test_positions_without_coordinates_are_refused():
    """(n, 0) positions have no coordinates; no sample can be valid."""
    from neurospatial._validation import validate_times_positions

    with pytest.raises(ValueError, match=r"got shape \(3, 0\)") as caught:
        validate_times_positions([0.0, 0.1, 0.2], np.zeros((3, 0)), call="f")
    assert "n_dims >= 1" in str(caught.value)


def test_compute_speed_refuses_positions_without_coordinates():
    from neurospatial.behavior import compute_speed

    with pytest.raises(ValueError, match=r"got shape \(10, 0\)"):
        compute_speed(np.arange(10) * 0.1, np.zeros((10, 0)))


class TestValidateFinite:
    """Tests for ``validate_finite``."""

    def test_clean_finite_input_returns_float64_preserving_values(self) -> None:
        result = validate_finite([1.0, 2.5, -3.0], name="x")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.5, -3.0]))

    def test_accepts_list_input_and_returns_float64(self) -> None:
        result = validate_finite([0, 1, 2], name="counts")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))

    def test_raises_on_positive_inf(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.inf, 3.0], name="x")

    def test_raises_on_negative_inf(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, -np.inf, 3.0], name="x")

    def test_raises_on_nan_by_default(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.nan, 3.0], name="x")

    def test_allows_nan_when_requested(self) -> None:
        result = validate_finite([1.0, np.nan, 3.0], name="x", allow_nan=True)
        assert result.dtype == np.float64
        assert np.isnan(result[1])
        assert result[0] == 1.0
        assert result[2] == 3.0

    def test_still_raises_on_inf_even_when_allow_nan(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.inf, np.nan], name="x", allow_nan=True)

    def test_still_raises_on_neg_inf_even_when_allow_nan(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([np.nan, -np.inf], name="x", allow_nan=True)

    def test_error_message_contains_name_count_and_first_index(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_finite([1.0, 2.0, np.inf, np.inf], name="velocity")
        message = str(exc_info.value)
        assert "velocity" in message
        # Two offending values.
        assert "2 non-finite value(s)" in message
        # First offending value is at index 2.
        assert "index 2" in message

    def test_error_message_reports_first_nan_index(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_finite([0.0, 0.0, np.nan], name="signal")
        message = str(exc_info.value)
        assert "signal" in message
        assert "1 non-finite value(s)" in message
        assert "index 2" in message

    def test_works_on_2d_array_first_offending_index_via_flat(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, np.inf]])
        with pytest.raises(ValueError) as exc_info:
            validate_finite(arr, name="grid")
        message = str(exc_info.value)
        assert "grid" in message
        # Flattened index of the inf (row 1, col 1) is 3.
        assert "flat index 3" in message

    def test_one_dimensional_index_is_not_called_flat(self) -> None:
        with pytest.raises(ValueError, match=r"first at index 1:"):
            validate_finite([0.0, np.inf], name="x", allow_nan=True)

    def test_2d_clean_input_preserved(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_finite(arr, name="grid")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_raises_valueerror_exact_type(self) -> None:
        # Ensure the raised exception is exactly ValueError, not a subclass.
        with pytest.raises(ValueError):
            validate_finite([np.nan], name="x")


class TestValidateLengths:
    """Tests for ``validate_lengths``."""

    def test_single_entry_passes(self) -> None:
        # A single-entry mapping can never mismatch; should not raise.
        validate_lengths({"only": np.array([1.0, 2.0, 3.0])})

    def test_all_equal_multi_entry_passes(self) -> None:
        s = np.array([0.1, 0.2, 0.3])
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # positions is 2-D but has len 3, matching s and t.
        validate_lengths({"spike_times": s, "times": t, "positions": p})

    def test_empty_mapping_passes(self) -> None:
        # No arrays -> no mismatch possible.
        validate_lengths({})

    def test_raises_on_length_mismatch_with_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "spike_times": np.array([0.1, 0.2, 0.3]),
                    "times": np.array([0.0, 1.0]),
                }
            )
        message = str(exc_info.value)
        assert "Length mismatch" in message
        assert "spike_times=3" in message
        assert "times=2" in message

    def test_length_one_among_longer_is_mismatch_not_broadcastable(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "scalar_like": np.array([5.0]),
                    "times": np.array([0.0, 1.0, 2.0]),
                }
            )
        message = str(exc_info.value)
        assert "Length mismatch" in message
        assert "scalar_like=1" in message
        assert "times=3" in message

    def test_message_lists_each_name_and_length(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "a": np.array([0.0, 1.0]),
                    "b": np.array([0.0, 1.0, 2.0]),
                    "c": np.array([0.0]),
                }
            )
        message = str(exc_info.value)
        assert "a=2" in message
        assert "b=3" in message
        assert "c=1" in message

    def test_message_says_why_and_how_to_fix(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths({"angles": np.zeros(4), "weights": np.zeros(3)})
        lines = str(exc_info.value).splitlines()
        assert lines[0] == "Length mismatch: angles=4, weights=3."
        assert lines[1].startswith("Why: ")
        assert lines[-1].startswith("Fix: ")

    def test_compares_first_axis_of_two_dimensional_arrays(self) -> None:
        with pytest.raises(ValueError, match="times=3, positions=2"):
            validate_lengths({"times": np.zeros(3), "positions": np.zeros((2, 3))})

    def test_raises_valueerror_exact_type(self) -> None:
        with pytest.raises(ValueError):
            validate_lengths({"a": np.array([1.0]), "b": np.array([1.0, 2.0])})


def test_non_pynapple_t_attribute_is_not_read_as_times() -> None:
    """Only pynapple objects are read by ``.t``: an xarray DataArray with a
    ``t`` dimension exposes its index there, not the event times."""
    xr = pytest.importorskip("xarray")
    from neurospatial.events import time_to_nearest_event

    times = np.array([10.0, 20.0, 30.0])
    events = xr.DataArray([10.0, 20.0, 30.0], dims="t")
    np.testing.assert_array_equal(time_to_nearest_event(times, events), [0, 0, 0])
