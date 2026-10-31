"""Round-trip and lazy-read tests for NWB spatial-rate and behavior I/O.

Covers population field storage and lazy behavior reads:

- ``write_spatial_rates`` / ``read_spatial_rates`` population round-trip with a
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
    """write_spatial_rates -> read_spatial_rates population round-trip."""

    @pytest.mark.parametrize("explicit_ids", [False, True])
    @pytest.mark.parametrize("spike_window", [None, [[0.0, 0.5], [1.0, 2.0]]])
    def test_disk_roundtrip_preserves_decode_pairing_and_coverage(
        self, empty_nwb, tmp_path, explicit_ids, spike_window
    ):
        """Persistence preserves label semantics, recording windows and decoding."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.decoding import BayesianDecoder
        from neurospatial.encoding import SpatialRatesResult, SpikeTrains
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = Environment.from_grid_mask(
            np.ones(2, dtype=bool), grid_edges=(np.arange(3.0),)
        )
        rates = SpatialRatesResult(
            firing_rates=np.array([[20.0, 1.0], [1.0, 20.0]]),
            occupancy=np.ones(2),
            env=env,
            method="binned",
            bandwidth=0.0,
            unit_ids=np.arange(2) if explicit_ids else None,
            spike_window=None if spike_window is None else np.array(spike_window),
        )
        # Explicit IDs pair labelled spikes by label; generated IDs pair a plain
        # list by position and refuse labelled spikes, before and after a file.
        trains = [np.array([0.05, 0.75, 1.05]), np.array([])]
        labelled = SpikeTrains(trains, unit_ids=[1, 0])
        spikes = labelled if explicit_ids else trains[::-1]
        times = np.arange(21) / 10
        expected = BayesianDecoder.from_rates(rates, dt=0.1).predict(
            spikes, times, spike_window=rates.spike_window
        )

        write_spatial_rates(empty_nwb, rates)
        path = tmp_path / "rates.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(empty_nwb)
        with NWBHDF5IO(str(path), "r") as io:
            restored = read_spatial_rates(io.read())

        assert restored._unit_ids_generated == rates._unit_ids_generated
        assert restored.spike_window_assumed == rates.spike_window_assumed
        if spike_window is not None:
            np.testing.assert_array_equal(restored.spike_window, rates.spike_window)
            assert not restored.spike_window.flags.writeable
            np.testing.assert_array_equal(restored[0].spike_window, rates.spike_window)

        restored_decoder = BayesianDecoder.from_rates(restored, dt=0.1)
        actual = restored_decoder.predict(
            spikes, times, spike_window=restored.spike_window
        )
        if not explicit_ids:
            with pytest.raises(ValueError, match="unit labels"):
                restored_decoder.predict(
                    labelled, times, spike_window=restored.spike_window
                )
        np.testing.assert_array_equal(actual.times, expected.times)
        np.testing.assert_allclose(actual.posterior, expected.posterior)

    @pytest.mark.parametrize(
        "key",
        [
            "bandwidth",
            "n_units",
            "unit_table_columns",
            "occupancy_name",
            "unit_ids_generated",
            "min_occupancy",
            "n_spikes_outside_intervals",
            "analysis_settings",
            "spike_window",
        ],
    )
    def test_missing_metadata_key_is_refused(self, empty_nwb, sample_environment, key):
        """A key the writer always stores is never filled with a guess on read.

        Defaults would read generated labels as explicit, drop the unit table,
        or report full spike-recording coverage.
        """
        import json
        import re

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        write_spatial_rates(empty_nwb, _make_rates_result(sample_environment))
        table = empty_nwb.processing["analysis"]["spatial_rates"]
        meta = json.loads(table.description)
        meta.pop(key)
        table.fields["description"] = json.dumps(meta)
        message = (
            f"Corrupt spatial-rates record 'spatial_rates': missing metadata key "
            f"'{key}'."
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            read_spatial_rates(empty_nwb)

    def test_unit_count_disagreeing_with_table_is_refused(
        self, empty_nwb, sample_environment
    ):
        """The recorded unit count must match the stored rows."""
        import json

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        write_spatial_rates(empty_nwb, _make_rates_result(sample_environment))
        table = empty_nwb.processing["analysis"]["spatial_rates"]
        meta = json.loads(table.description)
        meta["n_units"] = 4
        table.fields["description"] = json.dumps(meta)
        with pytest.raises(ValueError, match="records n_units=4 but stores 3 rows"):
            read_spatial_rates(empty_nwb)

    @pytest.mark.parametrize(
        ("key", "value"),
        [("unit_ids_generated", "false"), ("spike_window", [[1.0, 0.0]])],
    )
    def test_invalid_provenance_metadata_raises(
        self, empty_nwb, sample_environment, key, value
    ):
        """Corrupt metadata must not silently change label pairing or coverage."""
        import json

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        write_spatial_rates(empty_nwb, _make_rates_result(sample_environment))
        table = empty_nwb.processing["analysis"]["spatial_rates"]
        meta = json.loads(table.description)
        meta[key] = value
        table.fields["description"] = json.dumps(meta)
        with pytest.raises(ValueError, match=key):
            read_spatial_rates(empty_nwb)

    def test_min_occupancy_and_support_survive_roundtrip(
        self, empty_nwb, sample_environment
    ):
        """The occupancy threshold, not a settings dict, defines read-back support."""
        import dataclasses

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        result = dataclasses.replace(
            _make_rates_result(sample_environment), min_occupancy=2.5
        )
        assert result.analysis_settings is None
        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb, env=sample_environment)
        assert back.min_occupancy == 2.5
        np.testing.assert_array_equal(back.spatial_support, result.spatial_support)
        assert not back.spatial_support.all()
        assert back.n_spikes_outside_intervals is None

    def test_spike_accounting_survives_roundtrip(self, empty_nwb, sample_environment):
        import dataclasses

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        result = dataclasses.replace(
            _make_rates_result(sample_environment),
            n_spikes_outside_intervals=np.array([0, 4, 17]),
        )
        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb, env=sample_environment)
        np.testing.assert_array_equal(back.n_spikes_outside_intervals, [0, 4, 17])
        assert back[2].n_spikes_outside_intervals == 17

    def test_full_roundtrip_equal(self, empty_nwb, sample_environment):
        """firing_rates/occupancy/unit_ids/unit_table/smoothing/bandwidth equal."""
        from neurospatial.encoding.spatial import SpatialRatesResult
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        unit_ids = np.array([7, 11, 42])
        unit_table = pd.DataFrame(
            {"region": ["CA1", "CA1", "CA3"], "quality": [0.9, 0.8, 0.7]}
        )
        result = _make_rates_result(env, unit_ids=unit_ids, unit_table=unit_table)

        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb, env=env)

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
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        unit_ids = np.array([7, 11, 42])
        unit_table = pd.DataFrame(
            {"region": ["CA1", "CA1", "CA3"], "quality": [0.9, 0.8, 0.7]}
        )
        result = _make_rates_result(env, unit_ids=unit_ids, unit_table=unit_table)

        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb, env=env)

        assert list(back.unit_ids) == [7, 11, 42]
        assert list(back.unit_table["region"]) == ["CA1", "CA1", "CA3"]
        np.testing.assert_array_equal(
            back.unit_table["quality"].to_numpy(), [0.9, 0.8, 0.7]
        )

    def test_roundtrip_without_unit_table(self, empty_nwb, sample_environment):
        """No unit_table -> reads back as None; default unit_ids preserved."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)  # default unit_ids, no table

        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb, env=env)

        assert back.unit_table is None
        np.testing.assert_array_equal(back.unit_ids, np.arange(3))

    def test_env_none_restores_connected_env(self, empty_nwb, sample_environment):
        """env=None restores the PERSISTED env with its connectivity intact."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        back = read_spatial_rates(empty_nwb)  # env=None -> persisted env

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
            assert set(back.env.neighbors(bins=i)) == set(env.neighbors(bins=i))
        assert any(len(back.env.neighbors(bins=i)) > 0 for i in checked_bins)

    def test_env_none_without_persisted_env_raises(self, empty_nwb, sample_environment):
        """env=None AND no persisted env -> clear ValueError (no fabrication)."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        # Drop the persisted environment so neither source is available.
        del empty_nwb.scratch["spatial_rates_environment"]

        with pytest.raises(ValueError, match="no attached environment"):
            read_spatial_rates(empty_nwb)  # env=None

    def test_mismatched_explicit_env_raises(self, empty_nwb, sample_environment):
        """A wrong-n_bins explicit env= raises instead of silently attaching."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)
        write_spatial_rates(empty_nwb, result)

        # A clearly coarser env has a different (smaller) n_bins.
        rng = np.random.default_rng(1)
        other = Environment.from_samples(rng.uniform(0, 100, (500, 2)), bin_size=25.0)
        assert other.n_bins != env.n_bins

        with pytest.raises(ValueError, match="n_bins"):
            read_spatial_rates(empty_nwb, env=other)

    def test_same_size_env_with_different_units_raises(
        self, empty_nwb, sample_environment
    ):
        """A same-size env= cannot relabel the stored rates' units."""
        from neurospatial import IncompatibleEnvironmentError
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        write_spatial_rates(empty_nwb, _make_rates_result(env))
        relabelled = env.with_metadata(units="m")
        assert relabelled.n_bins == env.n_bins

        with pytest.raises(IncompatibleEnvironmentError, match="units"):
            read_spatial_rates(empty_nwb, env=relabelled)

    def test_same_size_env_with_shifted_bins_raises(
        self, empty_nwb, sample_environment
    ):
        """A same-size env= with other bin coordinates cannot relabel the maps."""
        from neurospatial import Environment, IncompatibleEnvironmentError
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        write_spatial_rates(empty_nwb, _make_rates_result(env))
        positions = np.random.default_rng(42).uniform(0, 100, (1000, 2))
        shifted = Environment.from_samples(
            positions + 1000.0, bin_size=5.0, units="cm", frame="test_session"
        )
        assert shifted.n_bins == env.n_bins

        with pytest.raises(IncompatibleEnvironmentError, match="bin coordinates"):
            read_spatial_rates(empty_nwb, env=shifted)

    def test_equivalent_env_is_accepted_and_attached(
        self, empty_nwb, sample_environment
    ):
        """An env= equal to the saved one is verified and attached as passed."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        write_spatial_rates(empty_nwb, _make_rates_result(env))
        equivalent = env.with_metadata()

        back = read_spatial_rates(empty_nwb, env=equivalent)
        assert back.env is equivalent

    def test_explicit_env_without_persisted_env_raises(
        self, empty_nwb, sample_environment
    ):
        """env= cannot be verified when the file saved no environment."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        write_spatial_rates(empty_nwb, _make_rates_result(env))
        del empty_nwb.scratch["spatial_rates_environment"]

        with pytest.raises(ValueError, match="cannot verify"):
            read_spatial_rates(empty_nwb, env=env)

    def test_overwrite_semantics(self, empty_nwb, sample_environment):
        """Duplicate name raises without overwrite; replaces with overwrite=True."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        first = _make_rates_result(env, seed=0)
        write_spatial_rates(empty_nwb, first)

        second = _make_rates_result(env, seed=99)
        with pytest.raises(ValueError, match="already exists"):
            write_spatial_rates(empty_nwb, second)

        write_spatial_rates(empty_nwb, second, overwrite=True)
        back = read_spatial_rates(empty_nwb, env=env)
        np.testing.assert_array_equal(
            np.asarray(back.firing_rates), np.asarray(second.firing_rates)
        )

        # overwrite=True cleanly replaced ALL companions: the persisted env is
        # still readable (exactly one copy) and env=None restores it.
        back_env_none = read_spatial_rates(empty_nwb)
        assert back_env_none.env.n_bins == env.n_bins
        np.testing.assert_array_equal(
            np.asarray(back_env_none.firing_rates), np.asarray(second.firing_rates)
        )

    def test_failed_overwrite_restores_the_original_record(
        self, empty_nwb, sample_environment, monkeypatch
    ):
        """A failed replacement leaves the original record readable."""
        from neurospatial.io.nwb import (
            _environment,
            read_spatial_rates,
            write_spatial_rates,
        )

        env = sample_environment
        original = _make_rates_result(env, seed=0)
        write_spatial_rates(empty_nwb, original)

        def fail(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(_environment, "write_environment", fail)
        with pytest.raises(ValueError) as caught:
            write_spatial_rates(
                empty_nwb, _make_rates_result(env, seed=1), overwrite=True
            )
        message = str(caught.value)
        assert "RuntimeError('disk full')" in message
        assert "the pre-existing record was restored" in message
        monkeypatch.undo()

        back = read_spatial_rates(empty_nwb)
        np.testing.assert_array_equal(back.firing_rates, original.firing_rates)

    def test_failed_restore_is_reported_not_claimed(
        self, empty_nwb, sample_environment, monkeypatch
    ):
        """When an original cannot be put back, the error names it."""
        from neurospatial.io.nwb import _environment, write_spatial_rates

        env = sample_environment
        write_spatial_rates(empty_nwb, _make_rates_result(env, seed=0))

        def fail(*args, **kwargs):
            raise RuntimeError("disk full")

        def refuse_restore(obj):
            raise RuntimeError("cannot re-add")

        monkeypatch.setattr(_environment, "write_environment", fail)
        monkeypatch.setattr(empty_nwb, "add_scratch", refuse_restore)
        with pytest.raises(ValueError) as caught:
            write_spatial_rates(
                empty_nwb, _make_rates_result(env, seed=1), overwrite=True
            )
        message = str(caught.value)
        assert "restored" not in message.replace("could not be restored", "")
        assert "could not be restored: ['spatial_rates_environment']" in message
        assert "Do not write this NWBFile" in message
        assert "spatial_rates_environment" not in empty_nwb.scratch

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

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates
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
        back = read_spatial_rates(empty_nwb, env=env)
        np.testing.assert_array_equal(
            np.asarray(back.occupancy), np.asarray(result.occupancy)
        )

    def test_rates_on_two_environments_keep_their_own_bin_centers(
        self, empty_nwb, sample_environment
    ):
        """Each record carries its own environment; no shared bin_centers."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env_a = sample_environment
        positions = np.random.default_rng(42).uniform(0, 100, (1000, 2))
        env_b = Environment.from_samples(positions + 500.0, bin_size=5.0, units="m")
        assert env_b.n_bins == env_a.n_bins
        write_spatial_rates(empty_nwb, _make_rates_result(env_a), name="session_a")
        write_spatial_rates(empty_nwb, _make_rates_result(env_b), name="session_b")

        # A file-wide bin_centers dataset would describe only the first
        # environment and silently mislabel the second record's bins.
        assert "bin_centers" not in empty_nwb.processing["analysis"].data_interfaces
        back_a = read_spatial_rates(empty_nwb, name="session_a")
        back_b = read_spatial_rates(empty_nwb, name="session_b")
        np.testing.assert_array_equal(back_a.env.bin_centers, env_a.bin_centers)
        np.testing.assert_array_equal(back_b.env.bin_centers, env_b.bin_centers)
        assert back_b.env.units == "m"

    def test_wrong_length_occupancy_raises_at_read(self, empty_nwb, sample_environment):
        """A companion occupancy of wrong length raises at READ time."""
        from pynwb import TimeSeries

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

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
            read_spatial_rates(empty_nwb, env=env)

    def test_wrong_table_name_raises_clear_error(self, empty_nwb, sample_environment):
        """A name pointing at a non-spatial-rates table raises a clear error."""
        from hdmf.common import DynamicTable, VectorData

        from neurospatial.io.nwb import read_spatial_rates
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
            read_spatial_rates(empty_nwb, name="not_rates", env=sample_environment)

    def test_reserved_unit_table_columns_raise(self, empty_nwb, sample_environment):
        """A unit_table column named unit_id/firing_rate raises a clear error."""
        from neurospatial.io.nwb import write_spatial_rates

        env = sample_environment
        bad_table = pd.DataFrame({"unit_id": [0, 1, 2], "region": ["a", "b", "c"]})
        result = _make_rates_result(env, unit_table=bad_table)

        with pytest.raises(ValueError, match="reserved"):
            write_spatial_rates(empty_nwb, result)

    def test_copy_on_write_decouples_from_result(self, empty_nwb, sample_environment):
        """Result arrays cannot be mutated after write; the written data matches."""
        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        result = _make_rates_result(env)
        original_rates = np.array(result.firing_rates, copy=True)
        original_occ = np.array(result.occupancy, copy=True)

        write_spatial_rates(empty_nwb, result)

        # The live result's arrays are read-only, so nothing can change them
        # after the write.
        with pytest.raises(ValueError, match="read-only"):
            np.asarray(result.firing_rates)[:] = -999.0
        with pytest.raises(ValueError, match="read-only"):
            np.asarray(result.occupancy)[:] = -1.0

        back = read_spatial_rates(empty_nwb, env=env)
        np.testing.assert_array_equal(np.asarray(back.firing_rates), original_rates)
        np.testing.assert_array_equal(np.asarray(back.occupancy), original_occ)

    def _write_rates_file(self, path, env, *, seed):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import write_spatial_rates

        from ._nwb_helpers import create_roundtrip_nwb

        pytest.importorskip("ndx_events")
        pytest.importorskip("ndx_pose")
        nwbfile = create_roundtrip_nwb()
        write_spatial_rates(nwbfile, _make_rates_result(env, seed=seed))
        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwbfile)

    def test_overwrite_in_file_opened_for_append_is_refused(
        self, sample_environment, tmp_path
    ):
        """pynwb cannot delete written objects: refuse before touching the file."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        path = tmp_path / "rates.nwb"
        self._write_rates_file(path, env, seed=0)
        original = _make_rates_result(env, seed=0)

        with NWBHDF5IO(str(path), "r+") as io:
            nwbfile = io.read()
            with pytest.raises(
                NotImplementedError,
                match="Cannot replace the spatial rates 'spatial_rates'",
            ):
                write_spatial_rates(
                    nwbfile, _make_rates_result(env, seed=1), overwrite=True
                )
            # Nothing was removed: the record still reads back unchanged.
            back = read_spatial_rates(nwbfile)
            np.testing.assert_array_equal(back.firing_rates, original.firing_rates)

    def test_overwrite_in_read_file_exports_the_new_record(
        self, sample_environment, tmp_path
    ):
        """Replace in a file opened with "r", then export to a new path."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

        env = sample_environment
        path = tmp_path / "rates.nwb"
        exported = tmp_path / "rates_replaced.nwb"
        self._write_rates_file(path, env, seed=0)
        replacement = _make_rates_result(env, seed=1)

        with NWBHDF5IO(str(path), "r") as io:
            nwbfile = io.read()
            write_spatial_rates(nwbfile, replacement, overwrite=True)
            with NWBHDF5IO(str(exported), "w") as out:
                out.export(src_io=io, nwbfile=nwbfile)

        with NWBHDF5IO(str(exported), "r") as io:
            back = read_spatial_rates(io.read())
            np.testing.assert_array_equal(back.firing_rates, replacement.firing_rates)
            np.testing.assert_array_equal(back.occupancy, replacement.occupancy)
            np.testing.assert_array_equal(back.env.bin_centers, env.bin_centers)

    def test_disk_roundtrip(self, sample_environment, tmp_path):
        """Population rates survive a real write -> close -> reopen cycle."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

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
            back = read_spatial_rates(reopened, env=env)
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

        from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

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
            back = read_spatial_rates(reopened)  # env=None -> persisted env

            assert back.env.n_bins == env.n_bins
            assert back.env.connectivity.number_of_edges() > 0
            assert (
                back.env.connectivity.number_of_edges()
                == env.connectivity.number_of_edges()
            )
            checked = [i for i in (0, 1, env.n_bins - 1) if 0 <= i < env.n_bins]
            for i in checked:
                assert set(back.env.neighbors(bins=i)) == set(env.neighbors(bins=i))
            assert any(len(back.env.neighbors(bins=i)) > 0 for i in checked)


@pytest.mark.parametrize("method", ["binned", "diffusion_kde"])
def test_unvisited_bins_stay_missing_on_disk(tmp_path, method):
    """NaN rates in unvisited bins survive a file round trip as NaN, not zero."""
    from pynwb import NWBHDF5IO

    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rates
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    from ._nwb_helpers import create_roundtrip_nwb

    pytest.importorskip("ndx_events")
    pytest.importorskip("ndx_pose")
    # A 4 x 4 grid of 10 cm bins; the animal only visits the 2 x 2 corner.
    edges = np.arange(5.0) * 10.0
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(edges, edges), units="cm"
    )
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 120.0, 1 / 30)
    positions = rng.uniform(0.0, 20.0, (times.size, 2))
    spikes = [np.sort(rng.uniform(0, 120, 200)), np.sort(rng.uniform(0, 120, 50))]
    rates = compute_spatial_rates(
        env, spikes, times, positions, method=method, bandwidth=10.0
    )
    visited = np.all(env.bin_centers < 20.0, axis=1)
    assert visited.sum() == 4
    np.testing.assert_array_equal(rates.spatial_support, visited)
    assert np.isnan(rates.firing_rates[:, ~visited]).all()

    nwbfile = create_roundtrip_nwb()
    write_spatial_rates(nwbfile, rates)
    path = tmp_path / "quadrant.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(str(path), "r") as io:
        back = read_spatial_rates(io.read())

    np.testing.assert_array_equal(
        np.isnan(back.firing_rates), np.isnan(rates.firing_rates)
    )
    np.testing.assert_array_equal(back.firing_rates, rates.firing_rates)
    np.testing.assert_array_equal(back.spatial_support, rates.spatial_support)
    np.testing.assert_array_equal(back.occupancy, rates.occupancy)
    assert back.min_occupancy == rates.min_occupancy


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
            position_data = read_position(nwb, lazy=True)
            pos_lazy = position_data.positions
            ts_lazy = position_data.times
            position_data = read_position(nwb, lazy=False)
            pos_eager = position_data.positions
            ts_eager = position_data.times

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
            position_data = read_position(nwb)
            pos_default = position_data.positions
            ts_default = position_data.times  # default lazy=False
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
            unit_data = read_units(nwb, lazy=True)
            lazy_trains = unit_data.spike_times
            lazy_ids = unit_data.unit_ids
            unit_data = read_units(nwb, lazy=False)
            eager_trains = unit_data.spike_times
            eager_ids = unit_data.unit_ids

            np.testing.assert_array_equal(lazy_ids, eager_ids)
            assert len(lazy_trains) == len(eager_trains)
            for lazy_h, eager in zip(lazy_trains, eager_trains, strict=True):
                # Not a plain, fully-loaded ndarray until materialized.
                assert not isinstance(lazy_h, np.ndarray)
                np.testing.assert_array_equal(np.asarray(lazy_h), eager)

    def test_eager_default_unchanged(self, nwb_with_units):
        from neurospatial.io.nwb import read_units

        unit_data = read_units(nwb_with_units)
        trains = unit_data.spike_times
        ids = unit_data.unit_ids  # default lazy=False
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
