"""Tests for directional place field computation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neurospatial import Environment
from neurospatial.encoding.spatial import (
    DirectionalPlaceFields,
    compute_directional_place_fields,
    compute_spatial_rate,
)


def test_label_epochs_do_not_join_segments():
    times = np.arange(20) / 10
    positions = np.c_[np.linspace(1, 19, 20)]
    env = Environment.from_samples(np.c_[np.linspace(0, 20, 41)], bin_size=2.0)
    labels = np.array(["A"] * 10 + [None] * 3 + ["A"] * 7)
    result = compute_directional_place_fields(
        env, np.array([0.05, 1.05, 1.55]), times, positions, labels, method="binned"
    )
    assert result.occupancy["A"].sum() == pytest.approx(1.6, abs=1e-12)


def test_directional_place_fields_record_spike_window():
    times = np.arange(20) / 10
    positions = np.c_[np.linspace(1, 19, 20)]
    env = Environment.from_samples(np.c_[np.linspace(0, 20, 41)], bin_size=2.0)
    labels = np.array(["A"] * 10 + [None] * 3 + ["A"] * 7)
    result = compute_directional_place_fields(
        env,
        np.array([0.05, 1.05, 1.55]),
        times,
        positions,
        labels,
        method="binned",
        spike_window=(1.0, 2.0),
    )
    np.testing.assert_array_equal(result.spike_window, [[1.0, 2.0]])
    assert result.spike_window_assumed is False
    assert result.summary()["spike_window"] == [[1.0, 2.0]]
    assert result.summary()["spike_window_assumed"] is False
    assert result.occupancy["A"].sum() == pytest.approx(0.6, abs=1e-12)


def _gapped_recording():
    """40 samples at 10 Hz with a lost frame (sample 12) and a 1 s gap after 30."""
    rng = np.random.default_rng(3)
    times = np.arange(40) / 10
    positions = np.linspace(5, 95, 40)[:, None]
    positions[12] = np.nan
    times[31:] += 1.0  # 1 s gap after sample 30
    spikes = np.sort(rng.uniform(0, times[-1], 60))
    env = Environment.from_samples(np.linspace(0, 100, 101)[:, None], bin_size=10.0)
    return env, spikes, times, positions


@pytest.mark.parametrize("method", ["binned", "diffusion_kde"])
def test_label_windows_match_hand_written_epochs(method):
    """Per-label maps equal the estimator run on hand-written label windows."""
    env, spikes, times, positions = _gapped_recording()
    labels = np.array(["A"] * 10 + [None] * 5 + ["B"] * 10 + ["A"] * 15)
    kwargs = {"method": method, "bandwidth": 10.0, "max_gap": 0.5}
    fields = compute_directional_place_fields(
        env, spikes, times, positions, labels, **kwargs
    )
    windows = {"A": [(0.0, 1.0), (2.5, 4.9)], "B": [(1.5, 2.5)]}
    assert fields.labels == ("A", "B")
    for label, epochs in windows.items():
        ref = compute_spatial_rate(
            env, spikes, times, positions, epochs=epochs, **kwargs
        )
        assert_array_equal(fields.firing_rates[label], ref.firing_rate)
        assert_array_equal(fields.occupancy[label], ref.occupancy)


_OTHER_BETWEEN_A = ["A"] * 10 + ["other"] * 5 + ["A"] * 25
_NONE_BETWEEN_A = ["A"] * 10 + [None] * 5 + ["A"] * 25


def _assert_same_fields(actual, expected):
    assert actual.labels == expected.labels
    for label in expected.labels:
        assert_array_equal(actual.firing_rates[label], expected.firing_rates[label])
        assert_array_equal(actual.occupancy[label], expected.occupancy[label])


def test_other_is_an_ordinary_label():
    """The label "other" gets its own map unless passed as ``missing_label``."""
    env, spikes, times, positions = _gapped_recording()
    labels = np.array(_OTHER_BETWEEN_A, dtype=object)
    kwargs = {"method": "binned", "bandwidth": 10.0}

    fields = compute_directional_place_fields(
        env, spikes, times, positions, labels, **kwargs
    )
    assert fields.labels == ("A", "other")
    other = compute_spatial_rate(
        env, spikes, times, positions, epochs=[(1.0, 1.5)], **kwargs
    )
    assert_array_equal(fields.occupancy["other"], other.occupancy)

    declared = compute_directional_place_fields(
        env, spikes, times, positions, labels, missing_label="other", **kwargs
    )
    unlabeled = np.array(_NONE_BETWEEN_A, dtype=object)
    _assert_same_fields(
        declared,
        compute_directional_place_fields(
            env, spikes, times, positions, unlabeled, **kwargs
        ),
    )
    assert declared.labels == ("A",)
    assert_array_equal(declared.firing_rates["A"], fields.firing_rates["A"])
    assert_array_equal(declared.occupancy["A"], fields.occupancy["A"])


def test_directional_place_fields_record_missing_label():
    """The declared missing label is kept on the result and survives copying."""
    import copy
    import pickle

    env, spikes, times, positions = _gapped_recording()
    labels = np.array(_OTHER_BETWEEN_A, dtype=object)
    kwargs = {"method": "binned", "bandwidth": 10.0}
    declared = compute_directional_place_fields(
        env, spikes, times, positions, labels, missing_label="other", **kwargs
    )
    assert declared.missing_label == "other"
    assert declared.summary()["missing_label"] == "other"
    for restored in (copy.deepcopy(declared), pickle.loads(pickle.dumps(declared))):
        assert restored.missing_label == "other"
    default = compute_directional_place_fields(
        env, spikes, times, positions, labels, **kwargs
    )
    assert default.missing_label is None
    assert default.summary()["missing_label"] is None


def test_none_labels_are_left_out():
    """None between two stretches of "A" gives the hand-written "A" windows."""
    env, spikes, times, positions = _gapped_recording()
    labels = np.array(_NONE_BETWEEN_A, dtype=object)
    kwargs = {"method": "binned", "bandwidth": 10.0}
    fields = compute_directional_place_fields(
        env, spikes, times, positions, labels, **kwargs
    )
    assert fields.labels == ("A",)
    expected = compute_spatial_rate(
        env, spikes, times, positions, epochs=[(0.0, 1.0), (1.5, 4.9)], **kwargs
    )
    assert_array_equal(fields.firing_rates["A"], expected.firing_rate)
    assert_array_equal(fields.occupancy["A"], expected.occupancy)


def test_missing_labels_are_excluded():
    times = np.arange(20) / 10
    positions = np.c_[np.linspace(1, 19, 20)]
    env = Environment.from_samples(np.c_[np.linspace(0, 20, 41)], bin_size=2.0)
    labels = np.array(["A"] * 8 + [None] * 2 + ["B"] * 5 + [np.nan] * 2 + ["A"] * 3)
    result = compute_directional_place_fields(
        env, np.array([0.05, 1.55]), times, positions, labels, method="binned"
    )
    assert result.labels == ("A", "B")
    assert result.occupancy["A"].sum() == pytest.approx(0.8 + 0.2, abs=1e-12)
    assert result.occupancy["B"].sum() == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        ([(0.0, 0.5)], 0.5),
        ([(1.5, 2.0)], 0.4),
        # Spans the unlabeled stretch: intersecting gives 0.5 + 0.3 s, while
        # replacing the label windows would give the full 1.1 s.
        ([(0.5, 1.6)], 0.8),
    ],
)
def test_caller_epochs_intersect_label_windows(epochs, expected):
    """Caller epochs restrict each direction's windows; neither replaces the other."""
    times = np.arange(20) / 10
    positions = np.c_[np.linspace(1, 19, 20)]
    env = Environment.from_samples(np.c_[np.linspace(0, 20, 41)], bin_size=2.0)
    labels = np.array(["A"] * 10 + [None] * 3 + ["A"] * 7)
    result = compute_directional_place_fields(
        env,
        np.array([0.05, 1.05, 1.55]),
        times,
        positions,
        labels,
        method="binned",
        epochs=epochs,
    )
    assert result.occupancy["A"].sum() == pytest.approx(expected, abs=1e-12)


class TestDirectionalPlaceFieldsDataclass:
    """Tests for the DirectionalPlaceFields dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test basic creation of DirectionalPlaceFields."""
        fields = {
            "A→B": np.array([1.0, 2.0, 3.0]),
            "B→A": np.array([3.0, 2.0, 1.0]),
        }
        labels = ("A→B", "B→A")

        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = fields
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=labels,
        )

        assert tuple(result.firing_rates) == tuple(fields)
        for label, rate in fields.items():
            np.testing.assert_array_equal(result.firing_rates[label], rate)
        assert result.labels == labels

    def test_dataclass_is_frozen(self) -> None:
        """Test that dataclass is immutable (frozen)."""
        fields = {"A→B": np.array([1.0, 2.0, 3.0])}
        labels = ("A→B",)

        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = fields
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=labels,
        )

        # Should raise FrozenInstanceError when trying to modify
        with pytest.raises(AttributeError):
            result.labels = ("B→A",)  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.firing_rates = {}  # type: ignore[misc]

    def test_labels_is_tuple(self) -> None:
        """Test that labels preserves iteration order as tuple."""
        fields = {
            "first": np.array([1.0]),
            "second": np.array([2.0]),
            "third": np.array([3.0]),
        }
        labels = ("first", "second", "third")

        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = fields
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=labels,
        )

        assert isinstance(result.labels, tuple)
        assert result.labels == ("first", "second", "third")

    def test_fields_is_mapping(self) -> None:
        """Test that fields is a mapping from string labels to arrays."""
        fields = {"A→B": np.array([1.0, 2.0])}
        labels = ("A→B",)

        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = fields
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=labels,
        )

        # Should support dict-like access
        assert "A→B" in result.firing_rates
        assert_array_equal(result.firing_rates["A→B"], np.array([1.0, 2.0]))

    def test_empty_fields(self) -> None:
        """Test creation with empty fields."""
        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = {}
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=(),
        )

        assert len(result.firing_rates) == 0
        assert len(result.labels) == 0

    def test_single_direction(self) -> None:
        """Test with a single direction label."""
        fields = {"forward": np.array([1.0, 2.0, 3.0, 4.0])}
        labels = ("forward",)

        # Test stub: DirectionalPlaceFields requires firing_rates,
        # occupancy, env, labels. For pure dataclass
        # construction tests we synthesize the new fields with
        # zero occupancy and a tiny synthetic env.
        _fields_for_dpf = fields
        _occ_for_dpf = {k: np.zeros_like(v) for k, v in _fields_for_dpf.items()}
        _env_for_dpf = Environment.from_samples(
            np.linspace(0, 10, 11)[:, None], bin_size=1.0
        )
        result = DirectionalPlaceFields(
            firing_rates=_fields_for_dpf,
            occupancy=_occ_for_dpf,
            env=_env_for_dpf,
            labels=labels,
        )

        assert len(result.firing_rates) == 1
        assert len(result.labels) == 1
        assert result.labels[0] == "forward"


class TestComputeDirectionalPlaceFields:
    """Tests for the compute_directional_place_fields function."""

    @pytest.fixture
    def sample_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        positions = np.column_stack(
            [np.linspace(0, 100, 200), np.linspace(0, 100, 200)]
        )
        return Environment.from_samples(positions, bin_size=10.0)

    @pytest.fixture
    def sample_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample trajectory data."""
        times = np.linspace(0, 20, 200)  # 20 seconds
        positions = np.column_stack(
            [np.linspace(0, 100, 200), np.linspace(0, 100, 200)]
        )
        return times, positions

    def test_constant_labels_equals_compute_spatial_rate(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """If all labels are the same, result equals compute_spatial_rate."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 10.0, 15.0])

        # All labels the same
        labels = np.full(len(times), "forward", dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        # Compare with the canonical spatial-rate interface.
        expected = compute_spatial_rate(
            sample_env,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=10.0,
        ).firing_rate

        assert len(result.firing_rates) == 1
        assert "forward" in result.firing_rates
        assert result.labels == ("forward",)
        # Should be numerically close
        assert_array_almost_equal(result.firing_rates["forward"], expected, decimal=5)

    def test_two_directions_partition(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Two non-overlapping directions produce independent fields."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 12.0, 15.0])

        # First half is "A", second half is "B"
        labels = np.array(
            ["A"] * 100 + ["B"] * 100, dtype=object
        )  # 200 total like times

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert len(result.firing_rates) == 2
        assert "A" in result.firing_rates
        assert "B" in result.firing_rates
        assert set(result.labels) == {"A", "B"}

        # Each field should have shape (n_bins,)
        assert result.firing_rates["A"].shape == (sample_env.n_bins,)
        assert result.firing_rates["B"].shape == (sample_env.n_bins,)

    def test_none_label_excluded(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Samples labeled None are excluded from results."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 15.0])

        # Mix of "forward", unlabeled, and "backward"
        labels = np.array(
            ["forward"] * 50 + [None] * 100 + ["backward"] * 50, dtype=object
        )

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert result.labels == ("backward", "forward")
        assert "forward" in result.firing_rates
        assert "backward" in result.firing_rates

    def test_no_spikes_returns_zero_or_nan_fields(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Empty spike train produces zero/NaN fields."""
        times, positions = sample_trajectory
        spike_times = np.array([])  # No spikes

        labels = np.full(len(times), "forward", dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert "forward" in result.firing_rates
        # Field should be all zeros or NaN (depending on occupancy)
        field = result.firing_rates["forward"]
        assert np.all(np.isnan(field) | (field == 0))

    def test_result_structure(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """DirectionalPlaceFields has correct structure."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0, 10.0])

        labels = np.array(["A"] * 100 + ["B"] * 100, dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        # Result should be DirectionalPlaceFields
        assert isinstance(result, DirectionalPlaceFields)

        # fields should be a mapping
        assert hasattr(result.firing_rates, "__getitem__")

        # labels should be a tuple
        assert isinstance(result.labels, tuple)

        # All fields should have correct shape
        for label in result.labels:
            assert result.firing_rates[label].shape == (sample_env.n_bins,)

    def test_length_mismatch_raises_error(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Raises ValueError if direction_labels length doesn't match times."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0])

        # Wrong length labels
        wrong_labels = np.array(["A", "B", "C"], dtype=object)

        with pytest.raises(ValueError, match="direction_labels"):
            compute_directional_place_fields(
                sample_env,
                spike_times,
                times,
                positions,
                wrong_labels,
                method="binned",
                bandwidth=10.0,
            )

    def test_all_none_labels_returns_empty(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """If every label is None, returns empty fields."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0, 10.0])

        labels = np.full(len(times), None, dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert len(result.firing_rates) == 0
        assert len(result.labels) == 0

        # to_dataframe() on this empty result must not crash on pd.concat([]);
        # it returns an empty frame carrying the documented column schema.
        df = result.to_dataframe()
        assert len(df) == 0
        for col in ("direction", "bin", "coord_0", "firing_rate", "occupancy"):
            assert col in df.columns
