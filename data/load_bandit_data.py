"""Fetch and read the published J16 spatial-bandit example recording.

The version-1 NPZ archive contains plain numeric and string arrays and is
loaded with ``allow_pickle=False``.
Only :func:`fetch_bandit_data` (or this module's CLI) accesses the network.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from itertools import pairwise
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

BANDIT_EPOCH = "j1620210710_02_r1"
BANDIT_DATA_FILE = f"{BANDIT_EPOCH}_figure04_inputs.npz"
BANDIT_DATA_VERSION = "1.0"
BANDIT_DATA_DOI = "10.5281/zenodo.23020757"
BANDIT_DATA_SHA256 = "60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3"
BANDIT_DATA_URL = (
    f"https://zenodo.org/api/records/23020757/files/{BANDIT_DATA_FILE}/content"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_existing(path: Path) -> Path:
    if _file_sha256(path) != BANDIT_DATA_SHA256:
        raise FileExistsError(
            f"{path} exists with a different SHA-256 from the published recording. "
            "Move it aside before downloading; it will not be overwritten."
        )
    return path


def fetch_bandit_data(data_path: str | Path) -> Path:
    """Download and verify version 1.0 of the J16 example data from Zenodo.

    Parameters
    ----------
    data_path : str or Path
        Destination directory, created if needed. An existing verified file is
        reused without accessing the network; a different file is never replaced.

    Returns
    -------
    Path
        Path to ``j1620210710_02_r1_figure04_inputs.npz`` (about 75 MB).

    Raises
    ------
    FileExistsError
        If a different file already uses the destination filename.
    ValueError
        If the downloaded bytes do not match the pinned SHA-256 checksum.
    OSError
        If the download fails or its announced length is not received. Partial
        files are removed, and never appear under the final filename.
    """
    data_path = Path(data_path)
    output = data_path / BANDIT_DATA_FILE
    if output.exists():
        return _verified_existing(output)
    data_path.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=data_path, prefix=f".{BANDIT_DATA_FILE}.", suffix=".part", delete=False
        ) as stream:
            partial = Path(stream.name)
            digest = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(BANDIT_DATA_URL, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
                    received += len(block)
                announced = response.headers.get("Content-Length")
            if announced is not None and received != int(announced):
                raise ConnectionError(
                    f"Download stopped after {received} of {announced} bytes. "
                    "Run the download again."
                )
            if digest.hexdigest() != BANDIT_DATA_SHA256:
                raise ValueError(
                    f"Downloaded recording has SHA-256 {digest.hexdigest()}; "
                    f"expected {BANDIT_DATA_SHA256}. Run the download again."
                )
        # A hard link publishes the complete file atomically without replacing a
        # file that another process created while the transfer was in progress.
        try:
            os.link(partial, output)
        except FileExistsError:
            return _verified_existing(output)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return output


def load_neural_recording_from_files(
    data_path: str | Path,
    animal_date_epoch: str,
) -> dict[str, Any]:
    """Read a local recording NPZ into the example's original dictionary format.

    Parameters
    ----------
    data_path : str or Path
        Directory containing ``{animal_date_epoch}_figure04_inputs.npz``. Run
        ``uv run python data/load_bandit_data.py`` to fetch the published epoch.
        This loader never downloads data automatically.
    animal_date_epoch : str
        Recording identifier, e.g. ``"j1620210710_02_r1"``.

    Returns
    -------
    dict
        ``position_info`` is a time-indexed DataFrame (seconds; position in cm).
        ``spike_times`` is a list of per-unit timestamp arrays (seconds), in the
        archive's order, including empty trains. Curated unit IDs are absent
        from the archive; list indices are positional references only.
        ``track_graph`` is a NetworkX graph with node ``pos`` and edge
        ``distance``/``edge_id`` attributes. ``linear_edge_order`` is a list of
        node pairs; ``linear_edge_spacing`` is a scalar in cm.

    Notes
    -----
    Reads schema version 1 with ``allow_pickle=False``. The older five-file
    pickle/joblib export is no longer used.
    """
    path = Path(data_path) / f"{animal_date_epoch}_figure04_inputs.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Recording not found: {path}. Download the published J16 recording "
            "with `uv run python data/load_bandit_data.py --data-path <directory>`, "
            "or point data_path at a directory containing the NPZ."
        )
    with np.load(path, allow_pickle=False) as arrays:
        version = arrays["format_version"]
        if (
            version.shape != ()
            or not np.issubdtype(version.dtype, np.integer)
            or version.item() != 1
        ):
            raise ValueError(
                "Unsupported recording format_version; expected scalar integer 1."
            )

        times = arrays["position_time"]
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "position_time must be nonempty, finite and strictly increasing."
            )
        columns = arrays["position_columns"].tolist()
        if not isinstance(columns, list) or len(columns) != len(set(columns)):
            raise ValueError(
                "position_columns must be a vector of unique column names."
            )
        required = {"head_position_x", "head_position_y", "linear_position"}
        if not required.issubset(columns):
            raise ValueError(
                f"position_columns is missing {sorted(required - set(columns))}."
            )
        position_info = pd.DataFrame(
            {column: arrays[f"position/{column}"] for column in columns},
            index=pd.Index(times, name=str(arrays["position_index_name"]) or None),
        )

        concatenated = np.asarray(arrays["spike_times"], dtype=np.float64)
        offsets = arrays["spike_offsets"]
        if (
            concatenated.ndim != 1
            or offsets.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or not len(offsets)
            or offsets[0] != 0
            or offsets[-1] != len(concatenated)
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(
                "spike_offsets must be a nondecreasing integer vector starting at 0 "
                "and ending at the length of the one-dimensional spike_times array."
            )
        spike_times = [
            concatenated[start:stop].copy() for start, stop in pairwise(offsets)
        ]

        nodes = arrays["track_nodes"]
        positions = arrays["track_node_positions"]
        edges = arrays["track_edges"]
        distances = arrays["track_edge_distance"]
        edge_ids = arrays["track_edge_id"]
        edge_order = arrays["linear_edge_order"]
        if (
            nodes.ndim != 1
            or positions.ndim != 2
            or len(nodes) != len(positions)
            or edges.ndim != 2
            or edges.shape[1] != 2
            or distances.shape != (len(edges),)
            or edge_ids.shape != (len(edges),)
            or edge_order.ndim != 2
            or edge_order.shape[1] != 2
        ):
            raise ValueError("Track graph arrays have incompatible shapes.")
        graph = nx.Graph()
        for node, position in zip(nodes, positions, strict=True):
            graph.add_node(int(node), pos=tuple(position))
        for (start, stop), distance, edge_id in zip(
            edges, distances, edge_ids, strict=True
        ):
            if int(start) not in graph or int(stop) not in graph:
                raise ValueError(
                    "track_edges refers to a node absent from track_nodes."
                )
            graph.add_edge(
                int(start), int(stop), distance=float(distance), edge_id=int(edge_id)
            )
        linear_edge_order = [(int(start), int(stop)) for start, stop in edge_order]
        if any(not graph.has_edge(*edge) for edge in linear_edge_order):
            raise ValueError(
                "linear_edge_order refers to an edge absent from track_edges."
            )
        spacing = float(arrays["linear_edge_spacing"])
        if not np.isfinite(spacing) or spacing < 0:
            raise ValueError("linear_edge_spacing must be finite and nonnegative.")

    return {
        "position_info": position_info,
        "spike_times": spike_times,
        "track_graph": graph,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": spacing,
    }


def main() -> None:
    """Download the pinned example recording, or verify its local cached copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: the directory containing this helper).",
    )
    args = parser.parse_args()
    print(f"Verified {fetch_bandit_data(args.data_path)}")


if __name__ == "__main__":
    main()
