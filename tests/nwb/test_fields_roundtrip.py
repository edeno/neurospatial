"""Round-trip and lazy-read tests for NWB spatial-rate and behavior I/O.

Covers Phase 3.3:

- ``write_spatial_rates`` / ``read_place_field`` population round-trip with a
  unit axis (``unit_ids`` and ``unit_table`` preserved).
- ``lazy=True`` reads on ``read_position`` / ``read_pose`` / ``read_units``
  return handles that materialize on slice, while the eager default path is
  byte-for-byte unchanged.

All tests are gated on pynwb being installed (``skipif(not HAS_PYNWB)`` via
``importorskip``); the ``tests/nwb`` conftest additionally marks them
``integration``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip every test here when pynwb is absent (HAS_PYNWB gate).
pynwb = pytest.importorskip("pynwb")

from datetime import datetime  # noqa: E402
from uuid import uuid4  # noqa: E402


def _make_rates_result(env, *, unit_ids=None, unit_table=None, n_units=3, seed=0):
    """Build a SpatialRatesResult directly (no compute) for round-trip tests."""
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    firing_rates = rng.uniform(0, 10, (n_units, env.n_bins))
    occupancy = rng.uniform(0, 5, env.n_bins)
    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        smoothing_method="diffusion_kde",
        bandwidth=5.0,
        unit_ids=unit_ids,
        unit_table=unit_table,
    )


class TestSpatialRatesRoundTrip:
    """write_spatial_rates -> read_place_field population round-trip."""

    def test_full_roundtrip_equal(self, empty_nwb, sample_environment):
        """firing_rates/occupancy/unit_ids/unit_table/smoothing/bandwidth equal."""
        from neurospatial.encoding.spatial import SpatialRatesResult
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        unit_ids = np.array([7, 11, 42])
        unit_table = pd.DataFrame(
            {"region": ["CA1", "CA1", "CA3"], "quality": [0.9, 0.8, 0.7]}
        )
        result = _make_rates_result(env, unit_ids=unit_ids, unit_table=unit_table)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert isinstance(back, SpatialRatesResult)
        np.testing.assert_array_equal(
            np.asarray(back.firing_rates), np.asarray(result.firing_rates)
        )
        np.testing.assert_array_equal(
            np.asarray(back.occupancy), np.asarray(result.occupancy)
        )
        np.testing.assert_array_equal(
            np.asarray(back.unit_ids), np.asarray(result.unit_ids)
        )
        pd.testing.assert_frame_equal(back.unit_table, result.unit_table)
        assert back.smoothing_method == result.smoothing_method
        assert back.bandwidth == result.bandwidth
        assert back.env.n_bins == env.n_bins

    def test_unit_ids_and_table_preserved(self, empty_nwb, sample_environment):
        """Non-default unit_ids and a non-trivial unit_table survive exactly."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        unit_ids = np.array([7, 11, 42])
        unit_table = pd.DataFrame(
            {"region": ["CA1", "CA1", "CA3"], "quality": [0.9, 0.8, 0.7]}
        )
        result = _make_rates_result(env, unit_ids=unit_ids, unit_table=unit_table)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert list(back.unit_ids) == [7, 11, 42]
        assert list(back.unit_table["region"]) == ["CA1", "CA1", "CA3"]
        np.testing.assert_array_equal(
            back.unit_table["quality"].to_numpy(), [0.9, 0.8, 0.7]
        )

    def test_roundtrip_without_unit_table(self, empty_nwb, sample_environment):
        """No unit_table -> reads back as None; default unit_ids preserved."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)  # default unit_ids, no table

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert back.unit_table is None
        np.testing.assert_array_equal(back.unit_ids, np.arange(3))

    def test_reconstructs_env_when_not_passed(self, empty_nwb, sample_environment):
        """With env=None, a minimal env is reconstructed from stored bin_centers."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb)  # env=None

        assert back.env.n_bins == env.n_bins
        np.testing.assert_allclose(back.env.bin_centers, env.bin_centers)

    def test_overwrite_semantics(self, empty_nwb, sample_environment):
        """Duplicate name raises without overwrite; replaces with overwrite=True."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        first = _make_rates_result(env, seed=0)
        write_spatial_rates(empty_nwb, first)

        second = _make_rates_result(env, seed=99)
        with pytest.raises(ValueError, match="already exists"):
            write_spatial_rates(empty_nwb, second)

        write_spatial_rates(empty_nwb, second, overwrite=True)
        back = read_place_field(empty_nwb, env=env)
        np.testing.assert_array_equal(
            np.asarray(back.firing_rates), np.asarray(second.firing_rates)
        )

    def test_firing_rates_shape_validation(self, empty_nwb, sample_environment):
        """Wrong firing_rates n_bins raises a param-named ValueError."""
        from neurospatial.encoding.spatial import SpatialRatesResult
        from neurospatial.io.nwb import write_spatial_rates

        env = sample_environment
        rng = np.random.default_rng(0)
        bad = SpatialRatesResult(
            firing_rates=rng.uniform(0, 10, (3, env.n_bins + 1)),
            occupancy=rng.uniform(0, 5, env.n_bins),
            env=env,
            smoothing_method="binned",
            bandwidth=5.0,
        )
        with pytest.raises(ValueError, match="firing_rates"):
            write_spatial_rates(empty_nwb, bad)

    def test_occupancy_shape_validation(self, empty_nwb, sample_environment):
        """Wrong occupancy length raises a param-named ValueError."""
        from neurospatial.encoding.spatial import SpatialRatesResult
        from neurospatial.io.nwb import write_spatial_rates

        env = sample_environment
        rng = np.random.default_rng(0)
        bad = SpatialRatesResult(
            firing_rates=rng.uniform(0, 10, (3, env.n_bins)),
            occupancy=rng.uniform(0, 5, env.n_bins + 1),
            env=env,
            smoothing_method="binned",
            bandwidth=5.0,
        )
        with pytest.raises(ValueError, match="occupancy"):
            write_spatial_rates(empty_nwb, bad)

    def test_disk_roundtrip(self, sample_environment, tmp_path):
        """Population rates survive a real write -> close -> reopen cycle."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        from ._nwb_helpers import create_roundtrip_nwb

        env = sample_environment
        unit_ids = np.array([7, 11, 42])
        unit_table = pd.DataFrame(
            {"region": ["CA1", "CA1", "CA3"], "quality": [0.9, 0.8, 0.7]}
        )
        result = _make_rates_result(env, unit_ids=unit_ids, unit_table=unit_table)

        nwbfile = create_roundtrip_nwb()
        write_spatial_rates(nwbfile, result)

        nwb_path = tmp_path / "rates.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)
        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            back = read_place_field(reopened, env=env)
            np.testing.assert_allclose(
                np.asarray(back.firing_rates), np.asarray(result.firing_rates)
            )
            np.testing.assert_allclose(
                np.asarray(back.occupancy), np.asarray(result.occupancy)
            )
            np.testing.assert_array_equal(np.asarray(back.unit_ids), unit_ids)
            assert list(back.unit_table["region"]) == ["CA1", "CA1", "CA3"]


class TestReadPositionLazy:
    """lazy=True on read_position returns h5py-backed handles."""

    def test_lazy_returns_handle_equal_to_eager(
        self, sample_nwb_with_position, tmp_path
    ):
        import h5py
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_position

        nwb_path = tmp_path / "pos.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(sample_nwb_with_position)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            pos_lazy, ts_lazy = read_position(nwb, lazy=True)
            pos_eager, ts_eager = read_position(nwb, lazy=False)

            # The lazy handle is NOT a plain fully-loaded ndarray.
            assert not isinstance(pos_lazy, np.ndarray)
            assert isinstance(pos_lazy, h5py.Dataset) or type(
                pos_lazy
            ).__module__.startswith("h5py")

            # Materializing the handle equals the eager read, byte-for-byte.
            np.testing.assert_array_equal(np.asarray(pos_lazy), pos_eager)
            np.testing.assert_array_equal(np.asarray(ts_lazy), ts_eager)

    def test_eager_default_unchanged(self, sample_nwb_with_position, tmp_path):
        """lazy=False (default) equals a directly-materialized reference."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_position

        nwb_path = tmp_path / "pos.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(sample_nwb_with_position)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            pos_default, ts_default = read_position(nwb)  # default lazy=False
            series = nwb.processing["behavior"]["Position"]["position"]
            raw_pos = np.asarray(series.data[:], dtype=np.float64)
            raw_ts = np.asarray(series.timestamps[:], dtype=np.float64)

            assert isinstance(pos_default, np.ndarray)
            np.testing.assert_array_equal(pos_default, raw_pos)
            np.testing.assert_array_equal(ts_default, raw_ts)


class TestReadUnitsLazy:
    """lazy=True on read_units defers per-unit materialization."""

    @pytest.fixture
    def nwb_with_units(self):
        from pynwb import NWBFile

        nwbfile = NWBFile(
            session_description="units lazy test",
            identifier=str(uuid4()),
            session_start_time=datetime.now().astimezone(),
        )
        nwbfile.add_unit(spike_times=[0.1, 0.5, 1.2, 3.4], id=10)
        nwbfile.add_unit(spike_times=[0.3, 0.9], id=20)
        nwbfile.add_unit(spike_times=[2.0, 2.5, 2.7, 4.1, 5.0], id=30)
        return nwbfile

    def test_lazy_handles_materialize_to_eager(self, nwb_with_units, tmp_path):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_units

        nwb_path = tmp_path / "units.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwb_with_units)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            lazy_trains, lazy_ids = read_units(nwb, lazy=True)
            eager_trains, eager_ids = read_units(nwb, lazy=False)

            np.testing.assert_array_equal(lazy_ids, eager_ids)
            assert len(lazy_trains) == len(eager_trains)
            for lazy_h, eager in zip(lazy_trains, eager_trains, strict=True):
                # Not a plain, fully-loaded ndarray until materialized.
                assert not isinstance(lazy_h, np.ndarray)
                np.testing.assert_array_equal(np.asarray(lazy_h), eager)

    def test_eager_default_unchanged(self, nwb_with_units):
        from neurospatial.io.nwb import read_units

        trains, ids = read_units(nwb_with_units)  # default lazy=False
        np.testing.assert_array_equal(ids, [10, 20, 30])
        assert all(isinstance(t, np.ndarray) for t in trains)
        np.testing.assert_array_equal(trains[0], [0.1, 0.5, 1.2, 3.4])


class TestReadPoseLazy:
    """lazy=True on read_pose returns h5py-backed bodypart handles."""

    def test_lazy_bodyparts_materialize_to_eager(self, sample_nwb_with_pose, tmp_path):
        pytest.importorskip("ndx_pose")
        import h5py
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_pose

        nwb_path = tmp_path / "pose.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(sample_nwb_with_pose)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            bp_lazy, ts_lazy, _ = read_pose(nwb, lazy=True)
            bp_eager, ts_eager, _ = read_pose(nwb, lazy=False)

            assert set(bp_lazy) == set(bp_eager)
            for name, handle in bp_lazy.items():
                assert not isinstance(handle, np.ndarray)
                assert isinstance(handle, h5py.Dataset) or type(
                    handle
                ).__module__.startswith("h5py")
                np.testing.assert_array_equal(np.asarray(handle), bp_eager[name])
            np.testing.assert_array_equal(np.asarray(ts_lazy), ts_eager)
