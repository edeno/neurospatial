"""Known field recording support survives scene construction and export sampling."""

import copy
import pickle
from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.overlays import PositionData
from neurospatial.decoding import DecodingResult


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


@pytest.mark.parametrize(
    "options", [{}, {"max_gap": None}, {"max_gap": None, "epochs": (-1, 1)}]
)
def test_decoding_known_recording_gaps_survive_animation_options(env, options):
    result = DecodingResult(
        np.full((2, env.n_bins), 1 / env.n_bins),
        env,
        np.array([0.05, 0.25]),
        spike_window=np.array([[0.0, 0.1], [0.2, 0.3]]),
    )
    with (
        pytest.warns(UserWarning, match="exact samples only"),
        Animation.from_decoding(result, **options) as scene,
    ):
        field = scene.fields["posterior"]
        assert field.values is result.posterior
        np.testing.assert_array_equal(field.epochs, result.spike_window)
        np.testing.assert_array_equal(field.sample_valid, [True, True])
        np.testing.assert_array_equal(field.interval_valid, [False])
        np.testing.assert_array_equal(
            scene.sample_fields(0.05)["posterior"], result.posterior[0]
        )
        np.testing.assert_array_equal(
            scene.sample_fields(0.25)["posterior"], result.posterior[1]
        )
        assert np.isnan(scene.sample_fields(0.15)["posterior"]).all()
        # Holding is conservative: the whole source interval spans a pause.
        assert np.isnan(scene.sample_fields(0.075)["posterior"]).all()


def test_decoding_user_epochs_can_only_narrow_known_recording_support(env):
    result = DecodingResult(
        np.full((4, env.n_bins), 1 / env.n_bins),
        env,
        np.array([0, 0.1, 0.2, 0.3]),
        spike_window=np.array([[0, 0.15], [0.2, 0.4]]),
    )
    with Animation.from_decoding(result, max_gap=None, epochs=(0.05, 0.3)) as scene:
        np.testing.assert_array_equal(
            scene.fields["posterior"].epochs, [[0.05, 0.15], [0.2, 0.3]]
        )
        np.testing.assert_array_equal(
            scene.fields["posterior"].sample_valid, [False, True, True, False]
        )
        assert np.isnan(scene.sample_fields(0)["posterior"]).all()
        assert np.isnan(scene.sample_fields(0.3)["posterior"]).all()
    with pytest.raises(ValueError, match=r"does not overlap.*known recording support"):
        Animation.from_decoding(result, epochs=(1, 2))


def test_field_half_open_epochs_gate_exact_samples_and_held_intervals():
    times = np.array([-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    values = np.arange(len(times), dtype=float)[:, None]
    field = AnimationField(values, times, max_gap=None, epochs=[[0, 0.1], [0.2, 0.3]])
    np.testing.assert_array_equal(
        field.sample_valid, [False, True, True, False, False, True, True, False]
    )
    for query in [-0.1, -0.05, 0.1, 0.15, 0.175, 0.3]:
        assert np.isnan(field.sample(query)).all()
    for query, value in [
        (0, 1),
        (0.025, 1),
        (0.05, 2),
        (0.075, 2),
        (0.2, 5),
        (0.225, 5),
        (0.275, 6),
    ]:
        np.testing.assert_array_equal(field.sample(query), [value])


@pytest.mark.parametrize(
    "operation", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_epochs_owned_immutable_after_copy_and_pickle(operation):
    windows = np.array([[0.0, 0.1], [0.2, 0.3]])
    with pytest.warns(UserWarning, match="exact samples only"):
        field = AnimationField(np.array([[1.0], [2.0]]), [0.05, 0.25], epochs=windows)
    restored = operation(field)
    windows[0, 1] = 99
    np.testing.assert_array_equal(restored.epochs, [[0, 0.1], [0.2, 0.3]])
    with pytest.raises(FrozenInstanceError):
        restored.epochs = np.array([[0, 99]])
    with pytest.raises(ValueError):
        restored.epochs.flags.writeable = True
    assert np.isnan(restored.sample(0.15)).all()


@pytest.mark.parametrize("windows", [[], [(0, 0)], [(1, 0)], [(0, np.inf)]])
def test_invalid_field_epochs_use_canonical_validation(windows):
    with pytest.raises(ValueError, match="epochs"):
        AnimationField(np.array([[1.0], [2.0]]), [0, 0.1], epochs=windows)


def test_field_scales_ignore_samples_outside_known_support():
    field = AnimationField(
        np.array([[-1e9], [1.0], [3.0], [1e9]]), [-0.1, 0, 0.1, 0.2], epochs=(0, 0.2)
    )
    assert field.color_limits()[0] >= 1
    assert field.color_limits()[1] <= 3
    from neurospatial.animation._prepared import field_scale

    assert field_scale(field, "rate", see_through=True).opaque_from == 2


@dataclass(frozen=True)
class FrozenPosition:
    times: np.ndarray
    data: PositionData
    calls: list
    interp: str = "linear"

    def convert_to_data(self, frame_times, n_frames, env):
        self.calls.append(np.array(frame_times, copy=True))
        return self.data


class ReadonlyPosition:
    def __init__(self, times, data):
        self._times = times
        self._data = data
        self.calls = []

    @property
    def times(self):
        return self._times

    @property
    def interp(self):
        return "linear"

    def convert_to_data(self, frame_times, n_frames, env):
        self.calls.append(np.array(frame_times, copy=True))
        return self._data


@pytest.mark.parametrize("adapter_type", [FrozenPosition, ReadonlyPosition])
def test_scene_accepts_readonly_adapters_and_owns_prepared_metadata(env, adapter_type):
    times = np.array([10.0, 10.1, 10.2])
    data = PositionData(
        np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]), "red", 5, None, times=times
    )
    if adapter_type is FrozenPosition:
        adapter = FrozenPosition(times, data, [])
    else:
        adapter = ReadonlyPosition(times, data)
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[10, 10.2], overlays=[adapter]
    ) as scene:
        # The adapter is called once, on the scene clock.
        assert len(adapter.calls) == 1
        np.testing.assert_array_equal(adapter.calls[0], [10, 10.2])
        # Mutation of the adapter or the data it returned cannot alter the scene.
        times[:] = -1
        data.data[:] = 99
        data.color = "blue"
        owned = scene._overlay_data.positions[0]
        np.testing.assert_array_equal(owned.times, [10, 10.1, 10.2])
        np.testing.assert_allclose(owned.data, [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        assert owned.color == "red"
        scene.overlay_data([10.05, 10.15])
        assert len(adapter.calls) == 1


@pytest.mark.parametrize("order", [None, [1, 0]])
def test_builtin_raster_snapshot_preserves_mixed_public_ids_and_rate_identity(
    env, order
):
    from neurospatial.animation import SpikeRasterOverlay
    from neurospatial.encoding.spatial import SpatialRatesResult

    labels = np.empty(3, dtype=object)
    labels[:] = [1, "1", "b"]
    rates = SpatialRatesResult(
        np.vstack([np.full(env.n_bins, value) for value in [10.0, 99.0, 20.0]]),
        np.ones(env.n_bins),
        env,
        "binned",
        2.0,
        unit_ids=labels,
    )
    raster = SpikeRasterOverlay(
        [np.array([0.0]), np.array([0.1])],
        unit_ids=[1, "b"],
        rates=rates,
        order=order,
    )
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.1], overlays=[raster]
    ) as scene:
        public = scene.overlays[0].unit_ids
        assert public.tolist() == [1, "b"]
        assert isinstance(public[0], int)
        assert isinstance(public[1], str)
        assert not public.flags.writeable
        prepared = scene.overlay_data().spike_rasters[0]
        expected_ids, expected_rates = (
            ([1, "b"], [10, 20]) if order is None else (["b", 1], [20, 10])
        )
        assert prepared.unit_ids.tolist() == expected_ids
        np.testing.assert_array_equal(prepared.rates[:, 0], expected_rates)


@pytest.mark.parametrize("labels, expected_kind", [([1, 2], "i"), (["a", "b"], "U")])
def test_builtin_raster_snapshot_keeps_homogeneous_public_id_dtype(
    env, labels, expected_kind
):
    from neurospatial.animation import SpikeRasterOverlay

    raster = SpikeRasterOverlay([np.array([0.0]), np.array([0.1])], unit_ids=labels)
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.1], overlays=[raster]
    ) as scene:
        assert scene.overlays[0].unit_ids.dtype.kind == expected_kind
        assert scene.overlays[0].unit_ids.tolist() == labels
