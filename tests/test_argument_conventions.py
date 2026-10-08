"""Public function signatures follow the shared input conventions."""

import importlib
import inspect

from test_public_api_snapshot import NAMESPACES


def public_functions():
    """Iterate distinct public functions across the snapshot namespaces."""
    seen = set()
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            function = getattr(module, name)
            if inspect.isfunction(function) and function not in seen:
                seen.add(function)
                yield function, inspect.signature(function).parameters


def test_times_before_positions():
    offenders = []
    for function, parameters in public_functions():
        required = [
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        if (
            "times" in required
            and "positions" in required
            and required.index("times") > required.index("positions")
        ):
            offenders.append(function.__name__)
    assert not offenders, sorted(offenders)


def test_no_spike_trains_parameter_name():
    offenders = [
        (function.__name__, name)
        for function, parameters in public_functions()
        for name in parameters
        if name in {"spike_trains", "trains", "trajectory"}
    ]
    assert not offenders, sorted(offenders)


def test_env_is_first():
    offenders = []
    for function, parameters in public_functions():
        positional = [
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if "env" in positional and positional[0] != "env":
            first = positional[0]
            if not first.startswith("position_bins") and first not in {
                "trials",
                "nwbfile",
            }:
                offenders.append(function.__name__)
    assert not offenders, sorted(offenders)
