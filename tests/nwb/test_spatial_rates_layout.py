"""Where spatial rates live in an NWB file and how they share an environment."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pynwb")

pytestmark = pytest.mark.nwb


def _rates(env, *, seed=0, n_units=3, unit_ids=None):
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    return SpatialRatesResult(
        firing_rates=rng.uniform(0, 10, (n_units, env.n_bins)),
        occupancy=rng.uniform(0, 5, env.n_bins),
        env=env,
        method="diffusion_kde",
        bandwidth=5.0,
        unit_ids=unit_ids,
    )


def _environment_records(nwbfile) -> list[str]:
    return sorted(
        name
        for name, obj in nwbfile.analysis.items()
        if "bin_centers" in getattr(obj, "colnames", ())
    )


def _write_and_reopen(nwbfile, path):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return NWBHDF5IO(str(path), "r")


def test_rates_and_companions_are_written_to_the_analysis_group(
    empty_nwb, sample_environment
):
    """NWB processing modules hold intermediate data; results go in /analysis."""
    from neurospatial.io.nwb import write_spatial_rates

    write_spatial_rates(empty_nwb, _rates(sample_environment), name="ca1")

    assert set(empty_nwb.analysis) == {"ca1", "ca1_occupancy", "ca1_environment"}
    assert "analysis" not in empty_nwb.processing
    meta = json.loads(empty_nwb.analysis["ca1"].description)
    assert meta["schema_version"] == "3.0"
    assert meta["environment_name"] == "ca1_environment"


def test_rates_on_one_environment_share_a_single_record(
    empty_nwb, sample_environment, tmp_path
):
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    first, second = _rates(env, seed=0), _rates(env, seed=1)
    write_spatial_rates(empty_nwb, first, name="ca1", environment_name="open_field")
    record = empty_nwb.analysis["open_field"]
    write_spatial_rates(empty_nwb, second, name="ca3", environment_name="open_field")

    assert _environment_records(empty_nwb) == ["open_field"]
    assert empty_nwb.analysis["open_field"] is record

    with _write_and_reopen(empty_nwb, tmp_path / "shared.nwb") as io:
        nwbfile = io.read()
        assert _environment_records(nwbfile) == ["open_field"]
        for name, written in (("ca1", first), ("ca3", second)):
            back = read_spatial_rates(nwbfile, name=name)
            np.testing.assert_array_equal(back.firing_rates, written.firing_rates)
            np.testing.assert_array_equal(back.env.bin_centers, env.bin_centers)
            np.testing.assert_array_equal(back.env.edge_pairs, env.edge_pairs)
            assert back.env.units == "cm"
            assert sorted(back.env.regions) == sorted(env.regions)


def test_rates_link_to_an_environment_saved_with_write_environment(
    empty_nwb, sample_environment
):
    from neurospatial.io.nwb import (
        read_spatial_rates,
        write_environment,
        write_spatial_rates,
    )

    write_environment(empty_nwb, sample_environment, name="open_field")
    record = empty_nwb.analysis["open_field"]
    write_spatial_rates(
        empty_nwb, _rates(sample_environment), environment_name="open_field"
    )

    assert empty_nwb.analysis["open_field"] is record
    assert _environment_records(empty_nwb) == ["open_field"]
    back = read_spatial_rates(empty_nwb)
    np.testing.assert_array_equal(back.env.bin_centers, sample_environment.bin_centers)


@pytest.mark.parametrize("change", ["bins", "regions"])
def test_a_different_environment_under_a_shared_name_is_refused(
    empty_nwb, sample_environment, change
):
    """Reusing the record must lose nothing, so even other regions refuse."""
    from neurospatial import Environment
    from neurospatial.io.nwb import write_spatial_rates

    env = sample_environment
    if change == "bins":
        other = Environment.from_samples(
            np.random.default_rng(1).uniform(0, 100, (1000, 2)),
            bin_size=10.0,
            units="cm",
            frame="test_session",
        )
    else:
        other = env.with_regions({})
    write_spatial_rates(empty_nwb, _rates(env), name="ca1", environment_name="arena")
    before = dict(empty_nwb.analysis)

    with pytest.raises(ValueError, match="stores a different environment") as error:
        write_spatial_rates(
            empty_nwb, _rates(other), name="ca3", environment_name="arena"
        )
    assert "used by spatial rates ['ca1']" in str(error.value)
    assert "Fix: pass environment_name= a new name" in str(error.value)
    assert dict(empty_nwb.analysis) == before


def test_overwrite_replaces_an_environment_only_its_own_rates_use(
    empty_nwb, sample_environment
):
    """Recomputing on a new binning and overwriting replaces the old environment."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    coarse = env.rebin(2)
    write_spatial_rates(empty_nwb, _rates(env))
    write_spatial_rates(empty_nwb, _rates(coarse), overwrite=True)

    back = read_spatial_rates(empty_nwb)
    np.testing.assert_array_equal(back.env.bin_centers, coarse.bin_centers)
    assert _environment_records(empty_nwb) == ["spatial_rates_environment"]


def test_overwrite_never_replaces_an_environment_other_rates_use(
    empty_nwb, sample_environment
):
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    write_spatial_rates(empty_nwb, _rates(env), name="ca1", environment_name="arena")
    write_spatial_rates(empty_nwb, _rates(env), name="ca3", environment_name="arena")

    with pytest.raises(ValueError, match=r"used by spatial rates \['ca1', 'ca3'\]"):
        write_spatial_rates(
            empty_nwb,
            _rates(env.rebin(2)),
            name="ca1",
            environment_name="arena",
            overwrite=True,
        )
    np.testing.assert_array_equal(
        read_spatial_rates(empty_nwb, name="ca3").env.bin_centers, env.bin_centers
    )


def test_overwrite_and_export_from_disk_keeps_the_shared_record(
    empty_nwb, sample_environment, tmp_path
):
    """The documented replace-on-disk path: open with mode 'r', overwrite, export."""
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    write_spatial_rates(empty_nwb, _rates(env, seed=0))
    replacement = _rates(env, seed=5)
    exported = tmp_path / "exported.nwb"
    with _write_and_reopen(empty_nwb, tmp_path / "source.nwb") as io:
        nwbfile = io.read()
        write_spatial_rates(nwbfile, replacement, overwrite=True)
        with NWBHDF5IO(str(exported), "w") as out:
            out.export(src_io=io, nwbfile=nwbfile)
    with NWBHDF5IO(str(exported), "r") as io:
        back = read_spatial_rates(io.read())
    np.testing.assert_array_equal(back.firing_rates, replacement.firing_rates)
    np.testing.assert_array_equal(back.env.bin_centers, env.bin_centers)


def test_overwrite_in_a_file_opened_for_writing_is_refused(
    empty_nwb, sample_environment, tmp_path
):
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import write_spatial_rates

    write_spatial_rates(empty_nwb, _rates(sample_environment))
    path = tmp_path / "rates.nwb"
    _write_and_reopen(empty_nwb, path).close()
    with NWBHDF5IO(str(path), "r+") as io:
        nwbfile = io.read()
        with pytest.raises(NotImplementedError, match="mode='r'"):
            write_spatial_rates(
                nwbfile, _rates(sample_environment, seed=2), overwrite=True
            )


@pytest.mark.parametrize(
    ("environment_name", "match"),
    [
        ("spatial_rates", "three different /analysis names"),
        ("quality", "not an environment record"),
    ],
)
def test_environment_name_must_name_a_free_or_environment_slot(
    empty_nwb, sample_environment, environment_name, match
):
    from hdmf.common import DynamicTable, VectorData

    from neurospatial.io.nwb import write_spatial_rates

    empty_nwb.add_analysis(
        DynamicTable(
            name="quality",
            description="Sorting quality",
            columns=[VectorData(name="snr", description="SNR", data=[4.0])],
        )
    )
    with pytest.raises(ValueError, match=match):
        write_spatial_rates(
            empty_nwb, _rates(sample_environment), environment_name=environment_name
        )
    assert "spatial_rates" not in empty_nwb.analysis


def test_rates_never_replace_an_environment_record(empty_nwb, sample_environment):
    from neurospatial.io.nwb import write_environment, write_spatial_rates

    write_environment(empty_nwb, sample_environment, name="arena")
    record = empty_nwb.analysis["arena"]
    with pytest.raises(ValueError, match="/analysis/arena is an environment record"):
        write_spatial_rates(
            empty_nwb, _rates(sample_environment), name="arena", overwrite=True
        )
    assert empty_nwb.analysis["arena"] is record


def test_rates_from_an_earlier_layout_are_refused(empty_nwb, sample_environment):
    """Rates an earlier version put in the 'analysis' processing module."""
    from hdmf.common import DynamicTable, VectorData

    from neurospatial.io.nwb import read_spatial_rates

    module = empty_nwb.create_processing_module("analysis", "Analysis results")
    module.add(
        DynamicTable(
            name="spatial_rates",
            description=json.dumps({"schema_version": "2.6", "method": "binned"}),
            columns=[VectorData(name="firing_rate", description="Hz", data=[[1.0]])],
        )
    )
    with pytest.raises(
        ValueError, match="in the processing module 'analysis'"
    ) as error:
        read_spatial_rates(empty_nwb)
    assert "Fix: recompute the rates" in str(error.value)


def test_missing_rates_list_the_saved_rates(empty_nwb, sample_environment):
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    write_spatial_rates(empty_nwb, _rates(sample_environment), name="ca1")
    with pytest.raises(KeyError, match=r"Saved spatial rates: \['ca1'\]"):
        read_spatial_rates(empty_nwb, name="ca3")


def _add_units(nwbfile, ids):
    for unit_id in ids:
        nwbfile.add_unit(id=unit_id, spike_times=[0.1 * unit_id, 1.0 + unit_id])


@pytest.mark.parametrize("on_disk", [False, True])
def test_rates_link_each_row_to_its_units_table_row(
    empty_nwb, sample_environment, tmp_path, on_disk
):
    """With matching ids, a DynamicTableRegion joins rates to nwbfile.units."""
    from hdmf.common import DynamicTableRegion

    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    _add_units(empty_nwb, [3, 7, 11, 20])
    rates = _rates(sample_environment, unit_ids=[11, 3, 20])
    write_spatial_rates(empty_nwb, rates)

    def check(nwbfile):
        table = nwbfile.analysis["spatial_rates"]
        link = table["units"]
        assert isinstance(link, DynamicTableRegion)
        assert link.table is nwbfile.units
        np.testing.assert_array_equal(link.data[:], [2, 0, 3])
        unit_table_ids = np.asarray(nwbfile.units.id[:])
        np.testing.assert_array_equal(unit_table_ids[link.data[:]], [11, 3, 20])
        back = read_spatial_rates(nwbfile)
        np.testing.assert_array_equal(back.unit_ids, [11, 3, 20])
        np.testing.assert_array_equal(back.firing_rates, rates.firing_rates)
        assert back.unit_table is None

    if not on_disk:
        check(empty_nwb)
        return
    with _write_and_reopen(empty_nwb, tmp_path / "linked.nwb") as io:
        check(io.read())


@pytest.mark.parametrize(
    ("units", "unit_ids"),
    [
        (None, [0, 1, 2]),  # no units table
        ([0, 1], [0, 1, 2]),  # id 2 is not a unit of the file
        ([0, 1, 2], None),  # generated ids are positions, not identities
        ([0, 1, 2], ["a", "b", "c"]),  # labels, not table ids
    ],
    ids=["no_units_table", "mismatch", "generated", "labels"],
)
def test_rates_without_matching_units_keep_unit_id_only(
    empty_nwb, sample_environment, units, unit_ids
):
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    if units is not None:
        _add_units(empty_nwb, units)
    rates = _rates(sample_environment, unit_ids=unit_ids)
    write_spatial_rates(empty_nwb, rates)

    assert "units" not in empty_nwb.analysis["spatial_rates"].colnames
    back = read_spatial_rates(empty_nwb)
    np.testing.assert_array_equal(back.unit_ids, rates.unit_ids)
    assert back._unit_ids_generated == (unit_ids is None)


def test_unit_table_column_named_units_is_reserved(empty_nwb, sample_environment):
    import pandas as pd

    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.io.nwb import write_spatial_rates

    rates = _rates(sample_environment)
    rates = SpatialRatesResult(
        firing_rates=rates.firing_rates,
        occupancy=rates.occupancy,
        env=sample_environment,
        method="diffusion_kde",
        bandwidth=5.0,
        unit_table=pd.DataFrame({"units": ["CA1", "CA1", "CA3"]}),
    )
    with pytest.raises(ValueError, match=r"reserved column name\(s\) \['units'\]"):
        write_spatial_rates(empty_nwb, rates)


def _shifted(env):
    """Same bins and units, every center moved 1000 cm: same n_bins, other places."""
    from neurospatial.ops.transforms import translate

    return env.apply_transform(translate(1000.0, 0.0))


@pytest.mark.parametrize("route", ["write_environment", "to_nwb"])
@pytest.mark.parametrize("change", ["shifted", "relabelled"])
def test_environment_writers_never_replace_a_record_rates_use(
    empty_nwb, sample_environment, route, change
):
    """Replacing it would silently relabel the bins of the rates that use it."""
    from neurospatial.io.nwb import (
        read_spatial_rates,
        write_environment,
        write_spatial_rates,
    )

    env = sample_environment
    other = _shifted(env) if change == "shifted" else env.with_metadata(units="m")
    write_spatial_rates(empty_nwb, _rates(env), name="ca1", environment_name="arena")
    write_spatial_rates(empty_nwb, _rates(env), name="ca3", environment_name="arena")
    record = empty_nwb.analysis["arena"]

    with pytest.raises(ValueError, match="spatial rates use it") as error:
        if route == "to_nwb":
            other.to_nwb(empty_nwb, name="arena", overwrite=True)
        else:
            write_environment(empty_nwb, other, name="arena", overwrite=True)
    assert "['ca1', 'ca3']" in str(error.value)
    assert "Fix:" in str(error.value)
    assert empty_nwb.analysis["arena"] is record
    np.testing.assert_array_equal(
        read_spatial_rates(empty_nwb, name="ca1").env.bin_centers, env.bin_centers
    )


def test_rewriting_the_same_environment_a_record_rates_use_is_allowed(
    empty_nwb, sample_environment
):
    from neurospatial.io.nwb import (
        read_spatial_rates,
        write_environment,
        write_spatial_rates,
    )

    write_spatial_rates(empty_nwb, _rates(sample_environment), environment_name="arena")
    write_environment(empty_nwb, sample_environment, name="arena", overwrite=True)
    back = read_spatial_rates(empty_nwb)
    np.testing.assert_array_equal(back.env.bin_centers, sample_environment.bin_centers)


@pytest.mark.parametrize("with_env", [False, True])
def test_reader_refuses_rates_whose_environment_record_was_swapped(
    empty_nwb, sample_environment, with_env
):
    """Any route that swaps the record (here, direct pynwb edits) is caught on read."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates
    from neurospatial.io.nwb._environment import _environment_record

    env = sample_environment
    write_spatial_rates(empty_nwb, _rates(env), environment_name="arena")
    del empty_nwb.analysis["arena"]
    empty_nwb.add_analysis(_environment_record(_shifted(env), "arena")[0])

    with pytest.raises(ValueError, match="changed since these rates were written"):
        read_spatial_rates(empty_nwb, env=env if with_env else None)


def test_overwrite_removes_the_default_environment_record_it_no_longer_uses(
    empty_nwb, sample_environment
):
    from neurospatial.io.nwb import write_environment, write_spatial_rates

    env = sample_environment
    write_environment(empty_nwb, env, name="open_field")
    write_spatial_rates(empty_nwb, _rates(env), name="ca1")
    assert "ca1_environment" in empty_nwb.analysis

    write_spatial_rates(
        empty_nwb,
        _rates(env, seed=1),
        name="ca1",
        environment_name="open_field",
        overwrite=True,
    )
    assert _environment_records(empty_nwb) == ["open_field"]


@pytest.mark.parametrize("old_name", ["standalone", "shared_default"])
def test_overwrite_keeps_an_environment_record_saved_or_used_elsewhere(
    empty_nwb, sample_environment, old_name
):
    from neurospatial.io.nwb import (
        read_spatial_rates,
        write_environment,
        write_spatial_rates,
    )

    env = sample_environment
    if old_name == "standalone":
        write_environment(empty_nwb, env, name="arena")
        write_spatial_rates(
            empty_nwb, _rates(env), name="ca1", environment_name="arena"
        )
        kept = "arena"
    else:
        write_spatial_rates(empty_nwb, _rates(env), name="ca1")
        write_spatial_rates(
            empty_nwb, _rates(env), name="ca3", environment_name="ca1_environment"
        )
        kept = "ca1_environment"
    write_spatial_rates(
        empty_nwb,
        _rates(env, seed=1),
        name="ca1",
        environment_name="other",
        overwrite=True,
    )
    assert kept in _environment_records(empty_nwb)
    assert "other" in _environment_records(empty_nwb)
    if old_name == "shared_default":
        read_spatial_rates(empty_nwb, name="ca3")


def test_rates_with_a_bin_centers_unit_column_are_not_environment_records(
    empty_nwb, sample_environment
):
    """Records are identified by their declared format, not a column name."""
    import pandas as pd

    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.io.nwb import (
        read_environment,
        write_environment,
        write_spatial_rates,
    )

    base = _rates(sample_environment)
    rates = SpatialRatesResult(
        firing_rates=base.firing_rates,
        occupancy=base.occupancy,
        env=sample_environment,
        method="diffusion_kde",
        bandwidth=5.0,
        unit_table=pd.DataFrame({"bin_centers": [1.0, 2.0, 3.0]}),
    )
    write_spatial_rates(empty_nwb, rates, name="ca1")

    with pytest.raises(KeyError) as missing:
        read_environment(empty_nwb, name="absent")
    assert "saved environments: ['ca1_environment']" in str(missing.value)
    with pytest.raises(ValueError, match="not an environment record"):
        write_environment(empty_nwb, sample_environment, name="ca1", overwrite=True)
    with pytest.raises(ValueError, match="/analysis/ca1 is not an environment record"):
        read_environment(empty_nwb, name="ca1")
