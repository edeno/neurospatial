"""Expected storage corruption produces contextual, actionable value errors."""

from datetime import datetime, timezone
from struct import unpack_from
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.geometry import _codecs


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


def test_non_utf8_file_metadata_has_a_contextual_value_error(tmp_path):
    prefix = tmp_path / "environment"
    _environment().to_file(prefix)
    prefix.with_suffix(".json").write_bytes(b"\xff")
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


def test_corrupt_compressed_array_has_a_contextual_value_error(tmp_path):
    prefix = tmp_path / "environment"
    _environment().to_file(prefix)
    path = prefix.with_suffix(".npz")
    with ZipFile(path) as archive:
        entry = archive.infolist()[0]
    assert entry.compress_type == ZIP_DEFLATED
    payload = bytearray(path.read_bytes())
    name_length, extra_length = unpack_from("<HH", payload, entry.header_offset + 26)
    start = entry.header_offset + 30 + name_length + extra_length
    payload[start : start + 2] = b"\xff\xff"
    path.write_bytes(payload)
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


# Every key the readers decode is written by every save, so a missing one means
# the record is incomplete; the readers must not substitute a default.
_DECODED_KEYS = [
    "name",
    "n_dims",
    "n_bins",
    "is_linearized_track",
    "layout_parameters",
    "derivations",
    "length_scale",
    "binning",
    "arena",
    "regions",
    "landmarks",
    "coordinate_kind",
]


@pytest.mark.parametrize("built_environment", ["polygon"], indirect=True)
@pytest.mark.parametrize("key", _DECODED_KEYS)
def test_record_without_a_saved_entry_is_refused(corrupted_environment_round_trip, key):
    def drop(metadata, arrays):
        del metadata[key]

    with pytest.raises(ValueError, match=rf"the record has no {key!r} entry") as error:
        corrupted_environment_round_trip(drop)
    assert "Fix:" in str(error.value)


def test_record_without_regions_is_refused_rather_than_losing_them():
    from neurospatial.regions import Region

    env = _environment()
    env = env.with_regions(
        {"reward": Region(name="reward", kind="point", data=np.array([0.5, 0.5]))}
    )
    record = env.to_dict()
    assert Environment.from_dict(record).regions.keys() == {"reward"}
    del record["regions"]
    with pytest.raises(ValueError, match="the record has no 'regions' entry"):
        Environment.from_dict(record)


def test_non_scalar_layout_parameter_is_refused_rather_than_dropped():
    record = _environment().to_dict()
    record["layout_parameters"]["bin_size"] = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(
        ValueError, match=r"layout_parameters\['bin_size'\] is not a build choice"
    ):
        Environment.from_dict(record)


@pytest.mark.parametrize(
    ("key", "value", "dtype", "message"),
    [
        (
            "binning/connections/all_pairs",
            [[0.5, 1.0]] + [[0, 1]] * 5,
            None,
            "must contain integer indices",
        ),
        ("binning/connections/closed_by", [300] * 6, "|i1", "integers outside int8"),
        (
            "binning/locator/active_mask",
            [[1, 1], [1, 1]],
            None,
            "must contain booleans",
        ),
        ("bin_centers", [["a", "b"]] * 4, None, "must contain numerical values"),
        ("bin_centers", [[0.0, 0.0]] * 4, "<U1", "unsupported dtype"),
    ],
    ids=["fractional_index", "out_of_range", "non_boolean_mask", "text", "dtype"],
)
def test_dictionary_arrays_are_decoded_strictly(key, value, dtype, message):
    """JSON lists are not cast silently: a cast would change indices or masks."""
    record = _environment().to_dict()
    record["arrays"][key] = value
    if dtype is not None:
        record["array_specs"][key]["dtype"] = dtype
    with pytest.raises(ValueError, match=rf"array {key!r} .*{message}") as error:
        Environment.from_dict(record)
    assert "Invalid arrays in the environment dictionary" in str(error.value)
    assert "Fix:" in str(error.value)


def test_payload_that_is_not_a_binning_is_refused():
    record = _environment().to_dict()
    record["binning"] = record["binning"]["locator"]
    for key in list(record["array_specs"]):
        if key.startswith("binning/locator/"):
            renamed = "binning/" + key.removeprefix("binning/locator/")
            record["array_specs"][renamed] = record["array_specs"][key]
            record["arrays"][renamed] = record["arrays"][key]
    with pytest.raises(ValueError, match="the binning payload is not a Binning"):
        Environment.from_dict(record)


def _bump(key):
    def corrupt(metadata, arrays):
        metadata[key] += 1

    return corrupt


def _flip_linearized(metadata, arrays):
    metadata["is_linearized_track"] = not metadata["is_linearized_track"]


def _shift_array(key):
    def corrupt(metadata, arrays):
        arrays[key] = arrays[key] + 1.0

    return corrupt


def _non_finite_center(metadata, arrays):
    arrays["bin_centers"][0, 0] = np.nan


def _grid_shape(metadata, arrays):
    metadata["grid_shape"] = [shape + 1 for shape in metadata["grid_shape"]]


def _dimension_ranges(metadata, arrays):
    (low, high), *rest = metadata["dimension_ranges"]
    metadata["dimension_ranges"] = [[low, high + 1.0], *rest]


def _active_mask(metadata, arrays):
    arrays["active_mask"][...] = ~arrays["active_mask"]


@pytest.mark.parametrize("built_environment", ["mask"], indirect=True)
@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_bump("n_bins"), "stored bin counts/dimensions differ from the binning"),
        (_bump("n_dims"), "stored bin counts/dimensions differ from the binning"),
        (_flip_linearized, "stored is_linearized_track differs from the binning"),
        (_non_finite_center, "stored bin_centers contain non-finite coordinates"),
        (
            _shift_array("bin_centers"),
            "stored bin_centers differ from the authoritative binning centers",
        ),
        (_shift_array("grid_edges_0"), "stored grid_edges_0 differs from the binning"),
        (_grid_shape, "stored grid_shape differs from the binning"),
        (_active_mask, "stored active_mask differs from the binning"),
        (_dimension_ranges, "stored dimension_ranges differ from the binning"),
    ],
    ids=[
        "n_bins",
        "n_dims",
        "linearized",
        "non_finite_centers",
        "centers",
        "grid_edges",
        "grid_shape",
        "active_mask",
        "dimension_ranges",
    ],
)
def test_descriptions_that_disagree_with_the_binning_are_refused(
    corrupted_environment_round_trip, corrupt, message
):
    with pytest.raises(ValueError, match=message) as error:
        corrupted_environment_round_trip(corrupt)
    assert "Fix:" in str(error.value)


def test_framed_grid_cannot_carry_world_grid_edges():
    """A sheared grid's axes are in its source frame, not world coordinates."""
    from neurospatial.ops.transforms import AffineND

    shear = np.eye(3)
    shear[0, 1] = 0.5
    framed = _environment().apply_transform(AffineND(shear))
    assert framed.binning.grid_frame is not None
    record = framed.to_dict()
    assert not any(key.startswith("grid_edges_") for key in record["arrays"])
    record["arrays"]["grid_edges_0"] = [0.0, 1.0, 2.0]
    record["array_specs"]["grid_edges_0"] = {"shape": [3], "dtype": "<f8"}
    with pytest.raises(
        ValueError, match="Framed grid source axes cannot be stored as world grid_edges"
    ):
        Environment.from_dict(record)
