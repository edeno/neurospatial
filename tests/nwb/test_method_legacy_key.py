"""NWB backward-compat: the estimator key rename reads legacy files.

``write_spatial_rates`` stores the estimator under the metadata key ``"method"``
(schema >= 2.0). Files written by earlier versions used ``"smoothing_method"``
(schema 1.x); ``read_place_field`` must still read those via a defensive
fallback on the old key.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Skip every test here when pynwb is absent (HAS_PYNWB gate).
pytest.importorskip("pynwb")

from neurospatial.io.nwb._fields import (
    DEFAULT_ANALYSIS_MODULE,
    DEFAULT_SPATIAL_RATES_NAME,
)


def _make_rates_result(env):
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(0)
    return SpatialRatesResult(
        firing_rates=rng.uniform(0, 10, (3, env.n_bins)),
        occupancy=rng.uniform(0, 5, env.n_bins),
        env=env,
        method="diffusion_kde",
        bandwidth=5.0,
    )


def test_nwb_reads_legacy_key(empty_nwb, sample_environment):
    """A table carrying the legacy ``"smoothing_method"`` key still reads."""
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_rates_result(env)

    # Write with the current writer (stores key "method"), then rewrite the
    # table's description JSON to the legacy schema-1.x form to simulate an old
    # file: rename the key back to "smoothing_method" and drop schema_version to
    # "1.0". (fields[...] override bypasses hdmf's read-only description.)
    write_spatial_rates(empty_nwb, result)
    table = empty_nwb.processing[DEFAULT_ANALYSIS_MODULE][DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    assert meta["method"] == "diffusion_kde"  # writer used the new key
    meta["smoothing_method"] = meta.pop("method")
    meta["schema_version"] = "1.0"
    table.fields["description"] = json.dumps(meta)

    # The fallback path reconstructs the estimator from the legacy key.
    back = read_place_field(empty_nwb, env=env)
    assert back.method == "diffusion_kde"
    np.testing.assert_array_equal(
        np.asarray(back.firing_rates), np.asarray(result.firing_rates)
    )


def test_nwb_writes_new_key_and_roundtrips(empty_nwb, sample_environment):
    """A freshly written table uses the ``"method"`` key and round-trips."""
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_rates_result(env)

    write_spatial_rates(empty_nwb, result)
    table = empty_nwb.processing[DEFAULT_ANALYSIS_MODULE][DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    assert "method" in meta and "smoothing_method" not in meta
    assert meta["schema_version"] == "2.0"

    back = read_place_field(empty_nwb, env=env)
    assert back.method == "diffusion_kde"
