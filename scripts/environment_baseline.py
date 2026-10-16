"""Record and compare Environment outputs to check that a refactor changes nothing.

Every probe runs on ten environment types (8- and 4-connected grids, a
polygon arena, a holed grid mask, a pixel mask, hexagonal, a branching track,
a triangular mesh, and circular and open polar grids), each also read back
from a dict, a file and NWB, and copied. A probe stores its output as a float
array, or the exception type and message it raised, so refusals are compared
too, and a probe that yields text (warnings, plot labels) stores the text. The
signature and docstring of every public member of the common Environment
are recorded as well. Cartesian and polar numerical cases remain separate.

Usage
-----
Record on the base commit and on the change, then compare::

    uv run python scripts/environment_baseline.py record base.npz
    uv run python scripts/environment_baseline.py record new.npz
    uv run python scripts/environment_baseline.py compare base.npz new.npz

``compare`` exits non-zero and lists every probe whose output, error, or
presence differs. Outputs must match exactly.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import sys
import tempfile
import warnings
import zlib
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment
from neurospatial.encoding._smoothing import resolve_spatial_bandwidth, smooth_rate_map
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout
from neurospatial.ops.basis import heat_kernel_wavelet_basis
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.transforms import translate

_ERROR_SUFFIX = "#err"
_API_PREFIX = "api/"


def _base_environments() -> Iterator[tuple[str, Any]]:
    positions = np.random.default_rng(0).uniform(0, 30, (4000, 2))
    yield "grid8", Environment.from_samples(positions, bin_size=3.0)
    yield (
        "grid4",
        Environment.from_samples(
            positions, bin_size=3.0, connect_diagonal_neighbors=False
        ),
    )
    yield "polygon", Environment.from_polygon(Point(15, 15).buffer(15), bin_size=3.0)
    holed = np.ones((10, 10), dtype=bool)
    holed[3:6, 4:7] = False
    yield (
        "mask",
        Environment.from_grid_mask(
            holed, grid_edges=(np.linspace(0, 30, 11), np.linspace(0, 30, 11))
        ),
    )
    yield "pixels", Environment.from_pixel_mask(holed, pixel_size=3.0)
    yield (
        "hex",
        Environment.from_layout(
            create_layout(
                "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 30)]
            )
        ),
    )
    track = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (40.0, 0.0)]):
        track.add_node(node, pos=pos)
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        track.add_edge(u, v, distance=20.0)
    yield (
        "track",
        Environment.from_graph(
            track, edge_order=[(0, 1), (1, 2), (1, 3)], edge_spacing=0.0, bin_size=2.0
        ),
    )
    yield (
        "mesh",
        Environment.from_layout(
            create_layout(
                "TriangularMesh",
                boundary_polygon=Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]),
                point_spacing=4.0,
            )
        ),
    )
    yield (
        "polar",
        Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 6,
        ),
    )
    yield (
        "polar_open",
        Environment.from_polar_egocentric(
            distance_range=(2.0, 30.0),
            angle_range=(-np.pi / 2, np.pi / 2),
            distance_bin_size=4.0,
            angle_bin_size=np.pi / 8,
            circular_angle=False,
        ),
    )


def _nwb_round_trip(env: Any) -> Any:
    from pynwb import NWBFile

    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = NWBFile(
        session_description="baseline",
        identifier="baseline",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    return read_environment(nwbfile)


def _variants(name: str, env: Any) -> Iterator[tuple[str, Any]]:
    yield name, env
    yield f"{name}|dict", from_dict(to_dict(env))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "env"
        to_file(env, path)
        yield f"{name}|file", from_file(path)
    yield f"{name}|nwb", _nwb_round_trip(env)


def _as_array(value: Any) -> np.ndarray:
    """Flatten a probe result into one float array, or keep text as text."""
    if isinstance(value, str):
        return np.array(value)
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=float)
    if hasattr(value, "bin_centers") and hasattr(value, "connectivity"):
        edges = sorted(
            (min(u, v), max(u, v), d.get("distance", np.nan))
            for u, v, d in value.connectivity.edges(data=True)
        )
        return np.concatenate(
            [np.asarray(value.bin_centers, dtype=float).ravel(), np.ravel(edges)]
        )
    if is_dataclass(value):
        return np.concatenate(
            [
                np.ravel(np.asarray(getattr(value, f.name), dtype=float))
                for f in fields(value)
            ]
        )
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        lengths = np.array([len(part) for part in value], dtype=float)
        return np.concatenate([lengths, *[np.asarray(p, dtype=float) for p in value]])
    return np.asarray(value, dtype=float)


def _record(out: dict[str, np.ndarray], key: str, probe: Callable[[], Any]) -> None:
    try:
        out[key] = _as_array(probe())
    except Exception as exc:  # refusals are recorded, not raised
        out[key + _ERROR_SUFFIX] = np.array(f"{type(exc).__name__}: {exc}")


def _warnings_text(call: Callable[[], Any]) -> str:
    """Run ``call`` and return the messages of the warnings it emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return "\n".join(str(w.message) for w in caught)


def _plot_labels(env: Any, field: np.ndarray) -> str:
    """Axis labels and aspect that ``plot_field`` chooses for ``env``."""
    fig, ax = plt.subplots()
    try:
        env.plot_field(field, ax=ax)
        return f"{ax.get_xlabel()} | {ax.get_ylabel()} | {ax.get_aspect()}"
    finally:
        plt.close(fig)


def _inputs(name: str, env: Any) -> dict[str, Any]:
    """Probe inputs drawn once from the base environment, shared by its variants."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    centers = np.asarray(env.bin_centers, dtype=float)
    n_bins = centers.shape[0]
    spread = float(np.median(np.ptp(centers, axis=0))) / 50.0
    inside = centers[rng.integers(0, n_bins, 40)] + rng.normal(0, spread, (40, 2))
    outside = centers.max(axis=0) + 100.0 + np.arange(3)[:, None]
    walk = [0]
    for _ in range(299):
        neighbors = sorted(env.connectivity.neighbors(walk[-1]))
        walk.append(int(rng.choice(neighbors)) if neighbors else walk[-1])
    times = np.arange(300) * 0.1
    times[150:] += 2.0  # a recording gap longer than the default max_gap
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    mid = (lo + hi) / 2.0
    return {
        "points": np.vstack([inside, outside]),
        "field": np.sin(np.arange(n_bins) * 0.37),
        "times": times,
        "positions": centers[walk] + rng.normal(0, spread, (300, 2)),
        "bins": np.asarray(walk, dtype=np.int32),
        "speed": np.abs(rng.normal(5.0, 3.0, 300)),
        "half": np.arange(n_bins) < n_bins // 2,
        "polygon": box(lo[0], lo[1], mid[0], mid[1]),
        "point": centers[n_bins // 3],
    }


def _probe(out: dict[str, np.ndarray], tag: str, env: Any, x: dict[str, Any]) -> None:
    n_last = env.n_bins - 1
    probes: dict[str, Callable[[], Any]] = {
        # Geometry and diffusion.
        "geometry": lambda: env,
        "boundary_bins": lambda: env.boundary_bins,
        "ncomp": lambda: env._diffusion_geometry.n_components,
        "smooth": lambda: env.smooth(x["field"], 3.0),
        "diffuse": lambda: env.diffuse(x["field"], 3.0),
        "grad": lambda: gradient(env, x["field"]),
        "divgrad": lambda: divergence(env, gradient(env, x["field"])),
        "wavelet": lambda: heat_kernel_wavelet_basis(
            env, centers=np.array([0, env.n_bins // 2]), scales=[1.0, 3.0]
        ),
        "gaussian_kde": lambda: smooth_rate_map(
            env,
            np.abs(x["field"]) * 10.0,
            np.ones(env.n_bins),
            method="gaussian_kde",
            bandwidth=3.0,
        ),
        "default_bandwidth": lambda: resolve_spatial_bandwidth(env, None),
        "bandwidth_warning": lambda: _warnings_text(
            lambda: resolve_spatial_bandwidth(env, 1e6)
        ),
        "plot_labels": lambda: _plot_labels(env, x["field"]),
        # Graph queries.
        "neighbors": lambda: env.neighbors(0),
        "path_between": lambda: env.path_between(0, n_last),
        "reachable_from": lambda: env.reachable_from(0, radius=2, metric="hops"),
        "rings": lambda: env.rings(0, hops=2),
        "components": lambda: env.components(),
        "distance_to_geodesic": lambda: env.distance_to([0], metric="geodesic"),
        "distance_to_euclidean": lambda: env.distance_to([0], metric="euclidean"),
        # Coordinate queries.
        "bin_at": lambda: env.bin_at(x["points"]),
        "contains": lambda: env.contains(x["points"]),
        "distance_between": lambda: env.distance_between(
            env.bin_centers[0], env.bin_centers[n_last]
        ),
        "neighbors_of_point": lambda: env.neighbors(env.bin_centers[n_last]),
        "interpolate_nearest": lambda: env.interpolate(x["field"], x["points"]),
        "interpolate_linear": lambda: env.interpolate(
            x["field"], x["points"], mode="linear"
        ),
        "to_linear": lambda: env.to_linear(x["positions"][:20]),
        "linear_to_nd": lambda: env.linear_to_nd(env.to_linear(x["positions"][:20])),
        # Trajectories.
        "occupancy_start": lambda: env.occupancy(x["times"], x["positions"]),
        "occupancy_linear": lambda: env.occupancy(
            x["times"], x["positions"], time_allocation="linear"
        ),
        "occupancy_speed": lambda: env.occupancy(
            x["times"], x["positions"], speed=x["speed"], min_speed=4.0
        ),
        "occupancy_smoothed": lambda: env.occupancy(
            x["times"], x["positions"], bandwidth=3.0
        ),
        "bin_sequence": lambda: env.bin_sequence(x["times"], x["positions"]),
        "bin_sequence_all": lambda: env.bin_sequence(
            x["times"], x["positions"], dedup=False, outside_value=None
        ),
        "bin_sequence_with_runs": lambda: env.bin_sequence_with_runs(
            x["times"], x["positions"]
        ),
        "transitions_bins": lambda: env.transitions(x["bins"]),
        "transitions_trajectory": lambda: env.transitions(
            times=x["times"], positions=x["positions"]
        ),
        "transitions_random_walk": lambda: env.transitions(method="random_walk"),
        "transitions_diffusion": lambda: env.transitions(
            method="diffusion", bandwidth=3.0
        ),
        # Regions.
        "bins_in_point_region": lambda: env.bins_in_region("baseline_point"),
        "bins_in_polygon_region": lambda: env.bins_in_region("baseline_box"),
        "region_mask": lambda: env.region_mask(["baseline_point", "baseline_box"]),
        "region_membership": lambda: env.region_membership(),
        # Environments derived from this one.
        "rebin": lambda: env.rebin(2),
        "subset": lambda: env.subset(bins=x["half"]),
        "subset_inverted_polygon": lambda: env.subset(
            polygon=x["polygon"], invert=True
        ),
        "apply_transform": lambda: env.apply_transform(translate(5.0, -3.0)),
    }
    for mode in ("average", "transition", "density"):
        probes[f"kernel_{mode}"] = lambda mode=mode: env.compute_kernel(
            3.0, mode=mode, cache=False
        )
    for key, probe in probes.items():
        _record(out, f"{tag}/{key}", probe)


def _record_api(out: dict[str, np.ndarray]) -> None:
    """Record each public member's kind, signature (without self) and docstring."""
    for cls in (Environment,):
        for name in sorted(n for n in dir(cls) if not n.startswith("_")):
            member = inspect.getattr_static(cls, name)
            function = (
                getattr(member, "__func__", None)
                or getattr(member, "func", None)
                or getattr(member, "fget", None)
                or member
            )
            signature = ""
            if inspect.isfunction(function):
                full = inspect.signature(function)
                parameters = list(full.parameters.values())[1:]
                signature = str(full.replace(parameters=parameters))
            text = f"{type(member).__name__}{signature}\n{inspect.getdoc(function)}"
            out[f"{_API_PREFIX}{cls.__name__}.{name}"] = np.array(text)


def record(path: Path) -> None:
    """Run every probe and save the outputs to ``path``."""
    warnings.simplefilter("ignore")
    out: dict[str, np.ndarray] = {}
    _record_api(out)
    for name, base in _base_environments():
        x = _inputs(name, base)
        for variant_name, variant in list(_variants(name, base)):
            for tag, env in (
                (variant_name, variant),
                (f"{variant_name}|copy", copy.deepcopy(variant)),
            ):
                regions = type(env.regions)(env.regions.values())
                regions.add("baseline_point", point=x["point"])
                regions.add("baseline_box", polygon=x["polygon"])
                env = env.with_regions(regions)
                _probe(out, tag, env, x)
    np.savez(path, **out)
    n_errors = sum(key.endswith(_ERROR_SUFFIX) for key in out)
    print(f"{len(out)} probes recorded to {path} ({n_errors} raised)")


def compare(base_path: Path, new_path: Path) -> int:
    """Print every probe that differs and return the number of differences."""
    base, new = np.load(base_path), np.load(new_path)
    differences = []
    for key in sorted(set(base.files) | set(new.files)):
        if key not in new.files:
            differences.append(f"{key}: missing from {new_path}")
        elif key not in base.files:
            differences.append(f"{key}: missing from {base_path}")
        elif base[key].dtype.kind in "US" or new[key].dtype.kind in "US":
            if str(base[key]) != str(new[key]):
                differences.append(f"{key}:\n  - {base[key]}\n  + {new[key]}")
        elif base[key].shape != new[key].shape or not np.array_equal(
            base[key], new[key], equal_nan=True
        ):
            differences.append(f"{key}: values differ")
    print("\n".join(differences) or f"{len(base.files)} probes identical")
    return len(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("record").add_argument("out", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("base", type=Path)
    comparison.add_argument("new", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        record(args.out)
    else:
        sys.exit(1 if compare(args.base, args.new) else 0)


if __name__ == "__main__":
    main()
