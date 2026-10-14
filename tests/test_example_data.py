"""The real-data example can be fetched safely and read without pickle."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "data" / "load_bandit_data.py"
    spec = importlib.util.spec_from_file_location("load_bandit_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recording_arrays():
    """Keep column dtypes, nonconsecutive graph labels and an empty unit visible."""
    return {
        "format_version": np.asarray(1, dtype=np.int64),
        "position_time": np.array([10.0, 10.1, 10.2]),
        "position_index_name": np.asarray("time"),
        "position_columns": np.array(
            ["head_position_x", "head_position_y", "linear_position", "patch_id"]
        ),
        "position/head_position_x": np.array([0.0, 1.0, 2.0]),
        "position/head_position_y": np.array([2.0, 2.0, 2.0]),
        "position/linear_position": np.array([5.0, 6.0, 7.0]),
        "position/patch_id": np.array([1, 1, 2], dtype=np.int32),
        "spike_times": np.array([10.01, 10.03, 10.15]),
        "spike_offsets": np.array([0, 2, 2, 3]),
        "track_nodes": np.array([7, 2, 9]),
        "track_node_positions": np.array([[0.0, 2.0], [5.0, 2.0], [5.0, 8.0]]),
        "track_edges": np.array([[7, 2], [2, 9]]),
        "track_edge_distance": np.array([5.0, 6.0]),
        "track_edge_id": np.array([4, 8]),
        "linear_edge_order": np.array([[9, 2], [2, 7]]),
        "linear_edge_spacing": np.asarray(15.0),
    }


def test_load_recording_reconstructs_original_dictionary(
    helper, tmp_path, recording_arrays
):
    np.savez(tmp_path / helper.BANDIT_DATA_FILE, **recording_arrays)
    result = helper.load_neural_recording_from_files(tmp_path, helper.BANDIT_EPOCH)

    assert set(result) == {
        "position_info",
        "spike_times",
        "track_graph",
        "linear_edge_order",
        "linear_edge_spacing",
    }
    expected = pd.DataFrame(
        {
            column: recording_arrays[f"position/{column}"]
            for column in recording_arrays["position_columns"]
        },
        index=pd.Index(recording_arrays["position_time"], name="time"),
    )
    pd.testing.assert_frame_equal(result["position_info"], expected)
    spikes = result["spike_times"]
    assert isinstance(spikes, list)
    assert len(spikes) == 3
    np.testing.assert_array_equal(spikes[0], [10.01, 10.03])
    np.testing.assert_array_equal(spikes[1], [])
    np.testing.assert_array_equal(spikes[2], [10.15])
    assert list(result["track_graph"]) == [7, 2, 9]
    assert result["track_graph"].nodes[7]["pos"] == (0.0, 2.0)
    assert result["track_graph"][2][9] == {"distance": 6.0, "edge_id": 8}
    assert result["linear_edge_order"] == [(9, 2), (2, 7)]
    assert result["linear_edge_spacing"] == 15.0


@pytest.mark.parametrize("version", [2, 1.0, [1]])
def test_load_rejects_unsupported_schema(helper, tmp_path, recording_arrays, version):
    recording_arrays["format_version"] = np.asarray(version)
    np.savez(tmp_path / helper.BANDIT_DATA_FILE, **recording_arrays)
    with pytest.raises(ValueError, match="format_version"):
        helper.load_neural_recording_from_files(tmp_path, helper.BANDIT_EPOCH)


@pytest.mark.parametrize("offsets", [[1, 3], [0, 2], [0, 3, 2, 3], [0.0, 3.0], []])
def test_load_rejects_offsets_that_lose_or_mispair_spikes(
    helper, tmp_path, recording_arrays, offsets
):
    recording_arrays["spike_offsets"] = np.asarray(offsets)
    np.savez(tmp_path / helper.BANDIT_DATA_FILE, **recording_arrays)
    with pytest.raises(ValueError, match="spike_offsets"):
        helper.load_neural_recording_from_files(tmp_path, helper.BANDIT_EPOCH)


def test_load_never_unpickles_object_arrays(helper, tmp_path, recording_arrays):
    recording_arrays["position/head_position_x"] = np.array([{}, {}, {}], dtype=object)
    np.savez(tmp_path / helper.BANDIT_DATA_FILE, **recording_arrays)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        helper.load_neural_recording_from_files(tmp_path, helper.BANDIT_EPOCH)


def test_missing_local_recording_explains_explicit_download(
    helper, tmp_path, monkeypatch
):
    no_network = Mock(side_effect=AssertionError("A local loader must not download"))
    monkeypatch.setattr(helper.urllib.request, "urlopen", no_network)
    with pytest.raises(
        FileNotFoundError, match=r"uv run python data/load_bandit_data\.py"
    ):
        helper.load_neural_recording_from_files(tmp_path, helper.BANDIT_EPOCH)
    no_network.assert_not_called()


@pytest.fixture
def download(helper, monkeypatch):
    payload = b"a small stand-in for the published NPZ bytes"
    monkeypatch.setattr(
        helper, "BANDIT_DATA_SHA256", hashlib.sha256(payload).hexdigest()
    )
    response = io.BytesIO(payload)
    response.headers = {"Content-Length": str(len(payload))}
    request = Mock(return_value=response)
    monkeypatch.setattr(helper.urllib.request, "urlopen", request)
    return payload, response, request


def test_fetch_publishes_only_verified_bytes(helper, tmp_path, download):
    payload, _, request = download
    destination = tmp_path / "new data directory"
    path = helper.fetch_bandit_data(destination)
    assert path == destination / helper.BANDIT_DATA_FILE
    assert path.read_bytes() == payload
    assert list(destination.iterdir()) == [path]
    request.assert_called_once_with(helper.BANDIT_DATA_URL, timeout=60)


def test_fetch_reuses_verified_cache_without_network(helper, tmp_path, download):
    payload, _, request = download
    path = tmp_path / helper.BANDIT_DATA_FILE
    path.write_bytes(payload)
    assert helper.fetch_bandit_data(tmp_path) == path
    request.assert_not_called()


def test_fetch_preserves_different_existing_file(helper, tmp_path, download):
    _, _, request = download
    path = tmp_path / helper.BANDIT_DATA_FILE
    path.write_bytes(b"a user's different local file")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        helper.fetch_bandit_data(tmp_path)
    assert path.read_bytes() == b"a user's different local file"
    request.assert_not_called()


def test_fetch_removes_checksum_mismatch(helper, tmp_path, download, monkeypatch):
    monkeypatch.setattr(helper, "BANDIT_DATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        helper.fetch_bandit_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_removes_truncated_download(helper, tmp_path, download):
    payload, response, _ = download
    response.headers["Content-Length"] = str(len(payload) + 1)
    with pytest.raises(ConnectionError, match="Download stopped"):
        helper.fetch_bandit_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_removes_interrupted_download(helper, tmp_path, download):
    _, response, _ = download
    response.read = Mock(side_effect=[b"partial bytes", OSError("connection lost")])
    with pytest.raises(OSError, match="connection lost"):
        helper.fetch_bandit_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_does_not_overwrite_file_created_during_download(
    helper, tmp_path, download, monkeypatch
):
    output = tmp_path / helper.BANDIT_DATA_FILE
    link = helper.os.link

    def concurrent_file(source, destination):
        output.write_bytes(b"another process's recording")
        link(source, destination)

    monkeypatch.setattr(helper.os, "link", concurrent_file)
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        helper.fetch_bandit_data(tmp_path)
    assert output.read_bytes() == b"another process's recording"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("custom", [False, True])
def test_cli_download_destination(helper, tmp_path, monkeypatch, capsys, custom):
    args = ["load_bandit_data.py"]
    expected = Path(helper.__file__).resolve().parent
    if custom:
        args += ["--data-path", str(tmp_path)]
        expected = tmp_path
    fetch = Mock(return_value=expected / helper.BANDIT_DATA_FILE)
    monkeypatch.setattr(helper, "fetch_bandit_data", fetch)
    monkeypatch.setattr(sys, "argv", args)
    helper.main()
    fetch.assert_called_once_with(expected)
    assert "Verified" in capsys.readouterr().out
