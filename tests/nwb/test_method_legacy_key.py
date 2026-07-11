"""NWB estimator-key contract: current ``"method"`` key, no legacy fallback.

``write_spatial_rates`` stores the estimator under the metadata key ``"method"``
(schema ``2.0``). The rename is a clean break with no back-compatibility shim:
tables carrying only the old ``"smoothing_method"`` key (schema ``1.x``) are
rejected, not silently read.
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


def test_nwb_legacy_key_rejected(empty_nwb, sample_environment):
    """A table carrying only the legacy ``"smoothing_method"`` key does not read.

    No silent fall-through: an old-schema table is rejected with a clear error
    rather than reconstructed.
    """
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_rates_result(env)

    # Write with the current writer (stores key "method"), then rewrite the
    # table's description JSON to the legacy schema-1.x form (rename the key back
    # to "smoothing_method", drop schema_version). fields[...] override bypasses
    # hdmf's read-only description.
    write_spatial_rates(empty_nwb, result)
    table = empty_nwb.processing[DEFAULT_ANALYSIS_MODULE][DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    meta["smoothing_method"] = meta.pop("method")
    meta["schema_version"] = "1.0"
    table.fields["description"] = json.dumps(meta)

    with pytest.raises(ValueError, match="method"):
        read_place_field(empty_nwb, env=env)


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
