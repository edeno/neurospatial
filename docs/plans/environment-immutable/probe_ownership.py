"""Capture Stage B ownership behavior without asserting the old bugs are correct.

Run before and after the implementation with the same isolated environment::

    uv run --no-sync python docs/plans/environment-immutable/probe_ownership.py

The JSON records mutation attempts separately from numerical data. Expected
ownership changes can then be distinguished from geometry changes.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from neurospatial import Environment, Region, Regions
from neurospatial.environment.polar import EgocentricPolarEnvironment


def attempt(operation: Callable[[], Any]) -> str:
    """Record successful edits or the exception that prevents them."""
    try:
        operation()
    except (AttributeError, TypeError, ValueError) as error:
        return type(error).__name__
    return "allowed"


def environment() -> Environment:
    """Make a fixed four-bin environment with a named point region."""
    base = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    return Environment(
        name="ownership probe",
        layout=base.layout,
        layout_type_used=base.layout_type,
        layout_params_used=base.layout_parameters,
        regions=Regions([Region("goal", "point", np.array([0.25, 0.25]))]),
        arena=base.arena,
    )


def record() -> dict[str, Any]:
    """Probe caller aliases, restoration paths and public analysis caches."""
    output: dict[str, Any] = {}
    point = np.array([0.25, 0.25])
    region = Region("goal", "point", point)
    point[0] = 0.75
    output["point_after_caller_edit"] = np.asarray(region.data).tolist()
    for label, restored in (
        ("original", region),
        ("copy", copy.copy(region)),
        ("deepcopy", copy.deepcopy(region)),
        ("pickle", pickle.loads(pickle.dumps(region))),
    ):
        output[f"region/{label}/writeable"] = bool(restored.data.flags.writeable)
        output[f"region/{label}/setflags"] = attempt(
            lambda restored=restored: restored.data.setflags(write=True)
        )

    env = environment()
    variants = {
        "original": env,
        "copy": copy.copy(env),
        "shallow": env.copy(deep=False),
        "deep": env.copy(deep=True),
        "deepcopy": copy.deepcopy(env),
        "pickle": pickle.loads(pickle.dumps(env)),
        "dict": Environment.from_dict(env.to_dict()),
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "environment"
        env.to_file(path)
        variants["file"] = Environment.from_file(path)
        from pynwb import NWBHDF5IO, NWBFile

        nwb = NWBFile(
            "ownership probe", "stage-b", datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        env.to_nwb(nwb, name="environment")
        nwb_path = Path(directory) / "environment.nwb"
        with NWBHDF5IO(nwb_path, "w") as stream:
            stream.write(nwb)
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as stream:
            variants["nwb"] = Environment.from_nwb(
                stream.read(), scratch_name="environment"
            )

    for label, restored_env in variants.items():
        output[f"environment/{label}/centers"] = restored_env.bin_centers.tolist()
        output[f"environment/{label}/region_coordinates"] = np.asarray(
            restored_env.regions["goal"].data
        ).tolist()
        output[f"environment/{label}/units_assignment"] = attempt(
            lambda restored_env=restored_env: setattr(restored_env, "units", "cm")
        )
        output[f"environment/{label}/region_edit"] = attempt(
            lambda restored_env=restored_env: restored_env.regions.set(
                "goal", Region("goal", "point", [1.5, 1.5])
            )
        )

    env = environment()
    kernel = env.compute_kernel(0.5)
    output["cache/kernel_edit"] = attempt(lambda: kernel.fill(0))
    output["cache/kernel_sum_after_export_edit"] = float(env.compute_kernel(0.5).sum())
    env = environment()
    operator = env.get_differential_operator()
    output["cache/operator_edit"] = attempt(
        lambda: setattr(operator, "data", np.zeros_like(operator.data))
    )
    output["cache/operator_norm_after_export_edit"] = float(
        np.linalg.norm(env.get_differential_operator().toarray())
    )

    polar = EgocentricPolarEnvironment.create(
        (1, 3), (-np.pi, np.pi), 1, np.pi, circular_angle=False
    )
    output["polar/circular_before"] = bool(polar._angular_is_circular)
    output["polar/builder_edit"] = attempt(
        lambda: polar.layout._build_params_used.__setitem__("circular_angle", True)
    )
    polar.clear_cache()
    output["polar/circular_after_builder_edit"] = bool(polar._angular_is_circular)
    return output


if __name__ == "__main__":
    print(json.dumps(record(), indent=2, sort_keys=True))
