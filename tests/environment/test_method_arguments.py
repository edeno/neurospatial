"""Method calls in an earlier form name the keyword to use, and its code runs.

Earlier versions took a query's target and the options of the field and plot
methods by position. Today the target is ``bins=``, ``points=`` or
``region=`` and the options are keyword-only, so an old call fails. Each case
makes an old call, checks that the error names the keyword, then runs the
call its ``Fix:`` line shows with the names the caller had and compares the
result with the call written by hand.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment, Regions


@pytest.fixture(scope="module")
def names() -> dict[str, Any]:
    """An arena with a reward region, a field and the caller's other names."""
    positions = np.random.default_rng(0).uniform(0.0, 40.0, (4000, 2))
    regions = Regions()
    regions.add("reward", point=(30.0, 30.0))
    env = Environment.from_samples(positions, bin_size=4.0, regions=regions)
    field = np.linspace(0.0, 1.0, env.n_bins)
    return {
        "np": np,
        "env": env,
        "field": field,
        "fields": field,
        "ax": None,
        "mask": env.bin_centers[:, 0] < 20.0,
        "track": Environment.linear_track(
            endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, track_width=8.0
        ),
    }


@dataclass(frozen=True)
class Case:
    """An old call, the keyword its error names, and the call written by hand.

    ``env`` names the environment the call is made on.
    """

    refuse: Callable[[dict[str, Any]], Any]
    keyword: str
    expected: Callable[[dict[str, Any]], Any]
    env: str = "env"


CASES: dict[str, Case] = {
    "distance_to(bins)": Case(
        lambda n: n["env"].distance_to([3]),
        "bins=[3]",
        lambda n: n["env"].distance_to(bins=[3]),
    ),
    "distance_to(region name)": Case(
        lambda n: n["env"].distance_to("reward", metric="euclidean"),
        "region='reward'",
        lambda n: n["env"].distance_to(region="reward", metric="euclidean"),
    ),
    "distance_to(targets=bins)": Case(
        lambda n: n["env"].distance_to(targets=[3, 4]),
        "bins=[3, 4]",
        lambda n: n["env"].distance_to(bins=[3, 4]),
    ),
    "neighbors(bin)": Case(
        lambda n: n["env"].neighbors(5),
        "bins=5",
        lambda n: n["env"].neighbors(bins=5),
    ),
    "neighbors(point)": Case(
        lambda n: n["env"].neighbors([10.0, 10.0]),
        "points=[[10.0, 10.0]]",
        lambda n: n["env"].neighbors(points=[[10.0, 10.0]]),
    ),
    "path_between(bin, bin)": Case(
        lambda n: n["env"].path_between(0, 7),
        "bins=[0, 7]",
        lambda n: n["env"].path_between(bins=[0, 7]),
    ),
    "path_between(source_active_bin_idx=)": Case(
        lambda n: n["env"].path_between(
            source_active_bin_idx=0, target_active_bin_idx=7
        ),
        "bins=[0, 7]",
        lambda n: n["env"].path_between(bins=[0, 7]),
    ),
    "distance_between(point, point, edge_weight)": Case(
        lambda n: n["env"].distance_between(
            np.array([5.0, 5.0]), np.array([30.0, 30.0]), None
        ),
        "points=[[5.0, 5.0], [30.0, 30.0]], edge_weight=None",
        lambda n: n["env"].distance_between(
            points=[[5.0, 5.0], [30.0, 30.0]], edge_weight=None
        ),
    ),
    "reachable_from(bin)": Case(
        lambda n: n["env"].reachable_from(0, radius=2),
        "bins=0, radius=2",
        lambda n: n["env"].reachable_from(bins=0, radius=2),
    ),
    "rings(center_bin=)": Case(
        lambda n: n["env"].rings(center_bin=0, hops=2),
        "bins=0, hops=2",
        lambda n: n["env"].rings(bins=0, hops=2),
    ),
    "subset(bins=mask)": Case(
        lambda n: n["env"].subset(bins=n["mask"]),
        "mask=",
        lambda n: n["env"].subset(mask=n["mask"]),
    ),
    "subset(mask)": Case(
        lambda n: n["env"].subset(n["mask"]),
        "mask=",
        lambda n: n["env"].subset(mask=n["mask"]),
    ),
    "smooth(field, bandwidth)": Case(
        lambda n: n["env"].smooth(n["field"], 5.0),
        "bandwidth=5.0",
        lambda n: n["env"].smooth(n["field"], bandwidth=5.0),
    ),
    "diffuse(fields, bandwidth)": Case(
        lambda n: n["env"].diffuse(n["fields"], 5.0, mode="average"),
        "bandwidth=5.0, mode='average'",
        lambda n: n["env"].diffuse(n["fields"], bandwidth=5.0, mode="average"),
    ),
    "compute_kernel(bandwidth)": Case(
        lambda n: n["env"].compute_kernel(5.0),
        "bandwidth=5.0",
        lambda n: n["env"].compute_kernel(bandwidth=5.0),
    ),
    "compute_gaussian_kernel(bandwidth)": Case(
        lambda n: n["env"].compute_gaussian_kernel(5.0),
        "bandwidth=5.0",
        lambda n: n["env"].compute_gaussian_kernel(bandwidth=5.0),
    ),
    "diffusion_accuracy(bandwidth)": Case(
        lambda n: n["env"].diffusion_accuracy(5.0),
        "bandwidth=5.0",
        lambda n: n["env"].diffusion_accuracy(bandwidth=5.0),
    ),
    "plot(ax, show_regions)": Case(
        lambda n: n["env"].plot(n["ax"], True),
        "show_regions=True",
        lambda n: n["env"].plot(n["ax"], show_regions=True),
    ),
    "plot_1d(ax, layout_plot_kwargs)": Case(
        lambda n: n["track"].plot_1d(n["ax"], {}),
        "layout_plot_kwargs={}",
        lambda n: n["track"].plot_1d(n["ax"], layout_plot_kwargs={}),
        env="track",
    ),
    "plot_field(field, ax, cmap, vmin)": Case(
        lambda n: n["env"].plot_field(n["field"], n["ax"], "magma", 0.0),
        "cmap='magma', vmin=0.0",
        lambda n: n["env"].plot_field(n["field"], n["ax"], cmap="magma", vmin=0.0),
    ),
}


def _same(first: Any, second: Any) -> None:
    if isinstance(first, Environment):
        np.testing.assert_array_equal(first.bin_centers, second.bin_centers)
        np.testing.assert_array_equal(first.edge_pairs, second.edge_pairs)
    elif isinstance(first, list) and first and isinstance(first[0], np.ndarray):
        assert len(first) == len(second)
        for one, other in zip(first, second, strict=True):
            np.testing.assert_array_equal(one, other)
    elif hasattr(first, "get_xlim"):
        assert first.get_xlim() == second.get_xlim()
        assert len(first.get_children()) == len(second.get_children())
    else:
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name", sorted(CASES))
def test_old_call_names_the_keyword_and_the_shown_call_runs(name, names):
    case = CASES[name]
    with pytest.raises(TypeError) as caught:
        case.refuse(names)
    message = str(caught.value)
    fix = message.splitlines()[-1]
    assert fix.startswith("Fix: ")
    assert case.keyword in fix, fix
    # The shown call, with a long value named the way the caller named it.
    code = fix.split("as in ", 1)[1].replace("<ndarray>", "mask")
    try:
        _same(eval(code, {**names, "env": names[case.env]}), case.expected(names))
    finally:
        plt.close("all")


@pytest.mark.parametrize("method", sorted({name.split("(")[0] for name in CASES}))
def test_guarded_methods_publish_their_own_signature(method):
    member = getattr(Environment, method)
    signature = inspect.signature(member)
    assert signature == inspect.signature(inspect.unwrap(member))
    assert next(iter(signature.parameters)) == "self"


def test_mixed_targets_name_the_keywords_without_a_call(names):
    with pytest.raises(TypeError) as caught:
        names["env"].path_between(0, [10.0, 10.0])
    fix = str(caught.value).splitlines()[-1]
    assert "bins=" in fix and "points=" in fix
    assert "as in" not in fix


def test_to_nwb_by_position_names_the_keyword(names):
    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    nwbfile = pynwb.NWBFile(
        session_description="arena",
        identifier="arena",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(TypeError) as caught:
        names["env"].to_nwb(nwbfile, "arena")
    fix = str(caught.value).splitlines()[-1]
    assert fix.endswith("as in env.to_nwb(nwbfile, name='arena')")
    eval(fix.split("as in ", 1)[1], {**names, "nwbfile": nwbfile})
    assert "arena" in nwbfile.scratch
