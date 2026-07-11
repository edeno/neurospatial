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
        method="diffusion_kde",
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
        assert back.method == result.method
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

    def test_env_none_restores_connected_env(self, empty_nwb, sample_environment):
        """env=None restores the PERSISTED env with its connectivity intact."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb)  # env=None -> persisted env

        assert back.env.n_bins == env.n_bins
        np.testing.assert_allclose(back.env.bin_centers, env.bin_centers)

        # The restored env is NOT a degenerate, edgeless env: it carries the
        # original connectivity, so graph ops work and neighbors match.
        assert back.env.connectivity.number_of_edges() > 0
        assert env.connectivity.number_of_edges() == (
            back.env.connectivity.number_of_edges()
        )
        # neighbors() matches the original for several bins, and at least one is
        # genuinely non-empty (proving connectivity was restored, not fabricated).
        checked_bins = [i for i in (0, 1, 2, env.n_bins - 1) if 0 <= i < env.n_bins]
        for i in checked_bins:
            assert set(back.env.neighbors(i)) == set(env.neighbors(i))
        assert any(len(back.env.neighbors(i)) > 0 for i in checked_bins)

    def test_env_none_without_persisted_env_raises(self, empty_nwb, sample_environment):
        """env=None AND no persisted env -> clear ValueError (no fabrication)."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        # Drop the persisted environment so neither source is available.
        del empty_nwb.scratch["spatial_rates_environment"]

        with pytest.raises(ValueError, match="no attached environment"):
            read_place_field(empty_nwb)  # env=None

    def test_mismatched_explicit_env_raises(self, empty_nwb, sample_environment):
        """A wrong-n_bins explicit env= raises instead of silently attaching."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)
        write_spatial_rates(empty_nwb, result)

        # A clearly coarser env has a different (smaller) n_bins.
        rng = np.random.default_rng(1)
        other = Environment.from_samples(rng.uniform(0, 100, (500, 2)), bin_size=25.0)
        assert other.n_bins != env.n_bins

        with pytest.raises(ValueError, match="n_bins"):
            read_place_field(empty_nwb, env=other)

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

        # overwrite=True cleanly replaced ALL companions: the persisted env is
        # still readable (exactly one copy) and env=None restores it.
        back_env_none = read_place_field(empty_nwb)
        assert back_env_none.env.n_bins == env.n_bins
        np.testing.assert_array_equal(
            np.asarray(back_env_none.firing_rates), np.asarray(second.firing_rates)
        )

    def test_orphaned_occupancy_blocks_write_without_overwrite(
        self, empty_nwb, sample_environment
    ):
        """A pre-existing orphan {name}_occupancy blocks the write, leaving no table."""
        from pynwb import TimeSeries

        from neurospatial.io.nwb import write_spatial_rates
        from neurospatial.io.nwb._core import _get_or_create_processing_module

        env = sample_environment
        result = _make_rates_result(env)

        # Simulate a half-written state: an orphan occupancy companion with the
        # table itself absent.
        analysis = _get_or_create_processing_module(
            empty_nwb, "analysis", "Analysis results"
        )
        analysis.add(
            TimeSeries(
                name="spatial_rates_occupancy",
                data=np.zeros((1, env.n_bins)),
                unit="seconds",
                timestamps=[0.0],
            )
        )

        with pytest.raises(ValueError, match="already exists"):
            write_spatial_rates(empty_nwb, result)  # overwrite=False

        # The write aborted BEFORE any mutation: no new table was created.
        assert "spatial_rates" not in analysis.data_interfaces

    def test_orphaned_occupancy_overwrite_replaces_cleanly(
        self, empty_nwb, sample_environment
    ):
        """overwrite=True replaces a stale orphan occupancy and reads back clean."""
        from pynwb import TimeSeries

        from neurospatial.io.nwb import read_place_field, write_spatial_rates
        from neurospatial.io.nwb._core import _get_or_create_processing_module

        env = sample_environment
        result = _make_rates_result(env)

        analysis = _get_or_create_processing_module(
            empty_nwb, "analysis", "Analysis results"
        )
        # Stale orphan with a WRONG length (would corrupt a name-only pairing).
        analysis.add(
            TimeSeries(
                name="spatial_rates_occupancy",
                data=np.zeros((1, env.n_bins + 5)),
                unit="seconds",
                timestamps=[0.0],
            )
        )

        write_spatial_rates(empty_nwb, result, overwrite=True)
        back = read_place_field(empty_nwb, env=env)
        np.testing.assert_array_equal(
            np.asarray(back.occupancy), np.asarray(result.occupancy)
        )

    def test_wrong_length_occupancy_raises_at_read(self, empty_nwb, sample_environment):
        """A companion occupancy of wrong length raises at READ time (FIX 3)."""
        from pynwb import TimeSeries

        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)
        write_spatial_rates(empty_nwb, result)

        # Corrupt the occupancy companion to a wrong length after the write.
        analysis = empty_nwb.processing["analysis"]
        del analysis.data_interfaces["spatial_rates_occupancy"]
        analysis.add(
            TimeSeries(
                name="spatial_rates_occupancy",
                data=np.zeros((1, env.n_bins + 3)),
                unit="seconds",
                timestamps=[0.0],
            )
        )

        with pytest.raises(ValueError, match="Occupancy companion"):
            read_place_field(empty_nwb, env=env)

    def test_wrong_table_name_raises_clear_error(self, empty_nwb, sample_environment):
        """A name pointing at a non-spatial-rates table raises a clear error (L2)."""
        from hdmf.common import DynamicTable, VectorData

        from neurospatial.io.nwb import read_place_field
        from neurospatial.io.nwb._core import _get_or_create_processing_module

        analysis = _get_or_create_processing_module(
            empty_nwb, "analysis", "Analysis results"
        )
        analysis.add(
            DynamicTable(
                name="not_rates",
                description="just a plain description, not JSON metadata",
                columns=[VectorData(name="x", description="d", data=np.arange(3))],
            )
        )

        with pytest.raises(ValueError, match="not a spatial-rates table"):
            read_place_field(empty_nwb, name="not_rates", env=sample_environment)

    def test_reserved_unit_table_columns_raise(self, empty_nwb, sample_environment):
        """A unit_table column named unit_id/firing_rate raises a clear error (L3)."""
        from neurospatial.io.nwb import write_spatial_rates

        env = sample_environment
        bad_table = pd.DataFrame({"unit_id": [1, 2, 3], "region": ["a", "b", "c"]})
        result = _make_rates_result(env, unit_table=bad_table)

        with pytest.raises(ValueError, match="reserved"):
            write_spatial_rates(empty_nwb, result)

    def test_copy_on_write_decouples_from_result(self, empty_nwb, sample_environment):
        """Mutating result arrays after write does not change what was written (L1)."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)
        original_rates = np.array(result.firing_rates, copy=True)
        original_occ = np.array(result.occupancy, copy=True)

        write_spatial_rates(empty_nwb, result)

        # Mutate the live result's arrays in place after the write.
        np.asarray(result.firing_rates)[:] = -999.0
        np.asarray(result.occupancy)[:] = -1.0

        back = read_place_field(empty_nwb, env=env)
        np.testing.assert_array_equal(np.asarray(back.firing_rates), original_rates)
        np.testing.assert_array_equal(np.asarray(back.occupancy), original_occ)

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
            method="binned",
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
            method="binned",
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

    def test_disk_roundtrip_env_none_restores_connectivity(
        self, sample_environment, tmp_path
    ):
        """On disk, env=None restores the env WITH connectivity (neighbors match)."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        from ._nwb_helpers import create_roundtrip_nwb

        env = sample_environment
        result = _make_rates_result(env)

        nwbfile = create_roundtrip_nwb()
        write_spatial_rates(nwbfile, result)

        nwb_path = tmp_path / "rates_env.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)
        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            back = read_place_field(reopened)  # env=None -> persisted env

            assert back.env.n_bins == env.n_bins
            assert back.env.connectivity.number_of_edges() > 0
            assert (
                back.env.connectivity.number_of_edges()
                == env.connectivity.number_of_edges()
            )
            checked = [i for i in (0, 1, env.n_bins - 1) if 0 <= i < env.n_bins]
            for i in checked:
                assert set(back.env.neighbors(i)) == set(env.neighbors(i))
            assert any(len(back.env.neighbors(i)) > 0 for i in checked)


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

    def test_lazy_length_mismatch_raises_like_eager(self, empty_nwb):
        """lazy=True validates positions/timestamps lengths like the eager path."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        behavior_module = empty_nwb.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        series = SpatialSeries(
            name="position",
            data=np.zeros((10, 2)),
            timestamps=np.arange(10) / 30.0,
            reference_frame="test",
            unit="cm",
        )
        position.add_spatial_series(series)
        behavior_module.add(position)

        # pynwb validates data/timestamps agreement at construction, so simulate
        # a corrupt/externally-written series by shrinking data after the fact.
        series.fields["data"] = np.zeros((8, 2))

        with pytest.raises(ValueError, match="Length mismatch"):
            read_position(empty_nwb, lazy=True)


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

    def test_lazy_bodypart_length_mismatch_raises_like_eager(self, empty_nwb):
        """Two bodyparts of differing length raise on lazy read, like the eager path."""
        pytest.importorskip("ndx_pose")
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

        from neurospatial.io.nwb import read_pose

        behavior_module = empty_nwb.create_processing_module(
            name="behavior", description="Behavior data"
        )
        skeleton = Skeleton(
            name="test_skeleton",
            nodes=["a", "b"],
            edges=np.array([[0, 1]], dtype=np.uint8),
        )
        # "a" (alphabetically first) supplies the shared timestamps (50); "b" is
        # shorter (40), so its length disagrees with the shared timestamps.
        pose = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=[
                PoseEstimationSeries(
                    name="a",
                    data=np.ones((50, 2)),
                    confidence=np.ones(50),
                    timestamps=np.arange(50) / 30.0,
                    reference_frame="test",
                    unit="cm",
                ),
                PoseEstimationSeries(
                    name="b",
                    data=np.ones((40, 2)),
                    confidence=np.ones(40),
                    timestamps=np.arange(40) / 30.0,
                    reference_frame="test",
                    unit="cm",
                ),
            ],
            skeleton=skeleton,
            source_software="Test",
        )
        behavior_module.add(pose)
        empty_nwb.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
        empty_nwb.processing["Skeletons"].add(skeleton)

        with pytest.raises(ValueError, match="Length mismatch"):
            read_pose(empty_nwb, lazy=True)
