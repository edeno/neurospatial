"""Expected storage corruption produces contextual, actionable value errors."""

from datetime import datetime, timezone

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout import _codecs


def _environment():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )


@pytest.mark.parametrize("part", ["parameters", "locator"])
def test_cyclic_dictionary_metadata_has_a_contextual_value_error(part):
    record = _environment().to_dict()
    if part == "parameters":
        record["layout_parameters"]["cycle"] = record["layout_parameters"]
    else:
        locator = {"codec": "TransformedLocator", "codec_version": 1}
        locator["base"] = locator
        record["binning"]["locator"] = locator
    with pytest.raises(
        ValueError, match=r"Invalid environment data.*dictionary"
    ) as error:
        Environment.from_dict(record)
    assert "Fix:" in str(error.value)


@pytest.mark.parametrize(
    "metadata",
    ["{", "[1, 2]", "[" * 2000 + "0" + "]" * 2000],
    ids=["syntax", "root_type", "depth"],
)
def test_malformed_file_metadata_has_a_contextual_value_error(tmp_path, metadata):
    prefix = tmp_path / "environment"
    _environment().to_file(prefix)
    prefix.with_suffix(".json").write_text(metadata)
    with pytest.raises(ValueError, match="Invalid environment metadata") as error:
        Environment.from_file(prefix)
    assert "Fix:" in str(error.value)


@pytest.mark.parametrize(
    "archive", [b"PK\x03\x04broken-archive", b"not-an-array"], ids=["zip", "bytes"]
)
def test_malformed_numpy_archive_has_a_contextual_value_error(tmp_path, archive):
    prefix = tmp_path / "environment"
    _environment().to_file(prefix)
    prefix.with_suffix(".npz").write_bytes(archive)
    with pytest.raises(ValueError, match="Invalid environment array archive") as error:
        Environment.from_file(prefix)
    assert "Fix:" in str(error.value)


def test_numpy_array_cannot_masquerade_as_a_numpy_archive(tmp_path):
    prefix = tmp_path / "environment"
    _environment().to_file(prefix)
    with prefix.with_suffix(".npz").open("wb") as stream:
        np.save(stream, np.ones(2))
    with pytest.raises(ValueError, match="Invalid environment array archive"):
        Environment.from_file(prefix)


@pytest.mark.parametrize(
    "metadata", ["[1, 2]", "[" * 2000 + "0" + "]" * 2000], ids=["root_type", "depth"]
)
def test_malformed_nwb_metadata_has_a_contextual_value_error(metadata):
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        "malformed", "malformed", datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    write_environment(nwbfile, _environment())
    nwbfile.scratch["spatial_environment"].fields["description"] = metadata
    with pytest.raises(ValueError, match="Invalid environment metadata") as error:
        read_environment(nwbfile)
    assert "Fix:" in str(error.value)


def test_unrelated_runtime_errors_are_not_hidden_as_input_corruption(monkeypatch):
    record = _environment().to_dict()

    def broken_decoder(*args, **kwargs):
        raise RuntimeError("implementation failure")

    monkeypatch.setattr(_codecs, "decode", broken_decoder)
    with pytest.raises(RuntimeError, match="implementation failure"):
        Environment.from_dict(record)
