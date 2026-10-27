"""Every public ``max_gap`` shares one calibrated default, and results record it."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np

import neurospatial
from neurospatial import Environment
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial.encoding import compute_spatial_rate


def _public_callables():
    seen = set()
    for info in pkgutil.walk_packages(neurospatial.__path__, "neurospatial."):
        if any(part.startswith("_") for part in info.name.split(".")[1:]):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # optional extras
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or id(obj) in seen:
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                seen.add(id(obj))
                yield f"{info.name}.{name}", obj
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if inspect.isfunction(member) and not attr.startswith("_"):
                        yield f"{info.name}.{name}.{attr}", member


def test_every_public_max_gap_defaults_to_the_shared_constant():
    checked, wrong = 0, []
    for name, obj in _public_callables():
        try:
            parameter = inspect.signature(obj).parameters.get("max_gap")
        except (TypeError, ValueError):
            continue
        if parameter is None or parameter.default is inspect.Parameter.empty:
            continue
        checked += 1
        if parameter.default != DEFAULT_MAX_GAP:
            wrong.append(f"{name}: {parameter.default!r}")
    assert checked > 50
    assert wrong == []


def test_the_default_is_recorded_as_the_value_used():
    times = np.arange(0, 20, 1 / 30)
    positions = np.column_stack([50 + 40 * np.sin(times), 50 + 40 * np.cos(times)])
    env = Environment.from_samples(positions, bin_size=10.0)
    result = compute_spatial_rate(env, np.array([1.0, 2.0]), times, positions)
    assert result.analysis_settings["max_gap"] == DEFAULT_MAX_GAP == 1.0
