"""Saved environments keep their layout label, derivation history and choices."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.ops.transforms import translate


def _grid():
    positions = np.random.default_rng(0).uniform(0, 20, (800, 2))
    return Environment.from_samples(positions, bin_size=2.0)


def _half(env):
    return env.subset(mask=np.arange(env.n_bins) < env.n_bins // 2)


def _save_and_read(env, how, tmp_path: Path):
    """Save ``env`` with this version; return the saved metadata and the read-back."""
    if how == "dict":
        saved = json.loads(json.dumps(to_dict(env)))
        return saved, from_dict(saved)
    if how == "file":
        to_file(env, tmp_path / "env")
        saved = json.loads((tmp_path / "env.json").read_text())
        return saved, from_file(tmp_path / "env")
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="current labels",
        identifier="current labels",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    saved = json.loads(nwbfile.scratch["spatial_environment"].description)
    return saved, read_environment(nwbfile)


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
def test_current_records_keep_their_labels_history_and_choices(how, tmp_path):
    env = _half(_grid()).apply_transform(translate(1.0, 2.0))
    saved, restored = _save_and_read(env, how, tmp_path)
    assert saved["layout_type"] == "RegularGrid"
    assert saved["binning"]["kind"] == "RegularGrid"
    assert saved["derivations"] == ["subset", "apply_transform"]
    assert "layout_category" not in saved
    assert restored.layout_type == env.layout_type
    assert restored.derivations == env.derivations
    assert restored.layout_parameters == env.layout_parameters


def test_malformed_derivations_are_refused():
    record = to_dict(_grid())
    record["derivations"] = "subset"
    with pytest.raises(ValueError, match="derivations must be a list"):
        from_dict(record)
