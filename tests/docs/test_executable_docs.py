"""Markdown examples execute with reader-like state and explicit opt-ins."""

import builtins
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import neurospatial
from neurospatial import Environment

ROOT = Path(__file__).resolve().parents[2]
RUN_ALL = {
    "README.md": False,
    "docs/getting-started/quickstart.md": False,
    "docs/user-guide/population-analysis.md": False,
    "docs/user-guide/regions.md": False,
}
OPT_IN = (
    "docs/getting-started/core-concepts.md",
    "docs/user-guide/composite-environments.md",
    "docs/user-guide/simulated-mazes.md",
    "docs/user-guide/environments.md",
    "docs/user-guide/layout-engines.md",
    "docs/user-guide/spatial-analysis.md",
    "docs/user-guide/alignment.md",
    "docs/user-guide/animation.md",
    "docs/user-guide/interoperability.md",
    "docs/user-guide/spike-field-primitives.md",
    "docs/user-guide/trajectory-and-behavioral-analysis.md",
    "docs/user-guide/behavior-around-events.md",
    "docs/user-guide/video-annotation.md",
    "docs/user-guide/workflows.md",
    "docs/user-guide/neuroscience-metrics.md",
    "docs/user-guide/signal-processing-primitives.md",
    "docs/migration/v0.6.md",
    "docs/migration/environment-unified.md",
    "docs/migration/environment-geometry.md",
    "docs/migration/environment-architecture.md",
)
SETUPS = {
    "docs_video_annotation_use_results": """import numpy as np
from types import SimpleNamespace

from neurospatial import Environment

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)
times = np.arange(len(positions), dtype=float) / 30.0  # 30 Hz tracking
env = Environment.from_samples(positions, bin_size=1.0)
result = SimpleNamespace(environment=env, regions={})
""",
    "docs_trajectory_region_crossings": """import numpy as np
from shapely.geometry import Point

from neurospatial import Environment

positions = np.column_stack([
    np.linspace(0.0, 100.0, 80),
    np.full(80, 50.0),
])
env = Environment.from_samples(positions, bin_size=5.0)
from neurospatial import Regions
region_builder = Regions()
region_builder.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))
env = env.with_regions(region_builder)

trajectory = np.array([
    [10.0, 50.0],
    [30.0, 50.0],
    [50.0, 50.0],
    [70.0, 50.0],
    [50.0, 50.0],
    [30.0, 50.0],
])
position_bins = env.bin_at(trajectory)
times = np.arange(len(trajectory)) * 0.1  # 10 Hz tracking
""",
    "workflows_decode_session_summary_streaming": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
from neurospatial import Environment
from neurospatial.simulation import (
    PlaceCellModel,
    generate_population_spikes,
    simulate_trajectory_ou,
)
env = Environment.from_samples(
    np.linspace(0.0, 100.0, 51).reshape(-1, 1), bin_size=2.0
)
env = env.with_metadata(units="cm")
positions, times = simulate_trajectory_ou(
    env, duration=120.0, dt=0.02, speed_mean=15.0, seed=0, speed_units="cm"
)
cells = [
    PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, seed=i)
    for i, c in enumerate(np.linspace(5.0, 95.0, 15))
]
spike_times = generate_population_spikes(
    cells, times, positions, seed=0, show_progress=False
)
""",
    "workflows_batch_processing_compute_spatial_rates": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
from neurospatial import Environment
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)
env = Environment.from_samples(
    np.linspace(0.0, 100.0, 51).reshape(-1, 1), bin_size=2.0
)
env = env.with_metadata(units="cm")
positions, times = simulate_trajectory_ou(
    env, duration=120.0, dt=0.02, speed_mean=15.0, seed=0, speed_units="cm"
)

def load_all_neurons():
    cells = [
        PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, seed=i)
        for i, c in enumerate(np.linspace(5.0, 95.0, 5))
    ]
    return {
        f"unit_{i}": generate_poisson_spikes(
            cell.firing_rate(positions), times, seed=i
        )
        for i, cell in enumerate(cells)
    }
""",
}
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[ \t]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.M | re.S,
)
_MARKER = re.compile(
    r"<!--\s*docs-test:\s*(?P<kind>run|skip|raises)\b(?P<arg>[^>]*?)\s*-->"
)


@dataclass(frozen=True)
class DocBlock:
    path: str
    line: int
    code: str
    kind: str | None
    arg: str


def collect_blocks(text: str, path: str) -> list[DocBlock]:
    blocks = []
    for match in _FENCE.finditer(text):
        preceding = text[: match.start()].splitlines()
        above = preceding[-1] if preceding else ""
        mark = _MARKER.fullmatch(above.strip())
        blocks.append(
            DocBlock(
                path,
                text.count("\n", 0, match.start()) + 2,
                textwrap.dedent(match["body"]),
                mark["kind"] if mark else None,
                mark["arg"].strip() if mark else "",
            )
        )
    return blocks


# Later blocks of these guides continue from the guide's first block; reading
# that block from the document keeps the setup identical to what readers run.
_BEHAVIOR_GUIDE = "docs/user-guide/behavior-around-events.md"
SETUPS["docs_behavior_around_reward"] = collect_blocks(
    (ROOT / _BEHAVIOR_GUIDE).read_text(encoding="utf-8"), _BEHAVIOR_GUIDE
)[0].code
_ANIMATION_GUIDE = "docs/user-guide/animation.md"
SETUPS["docs_animation_quick_start"] = collect_blocks(
    (ROOT / _ANIMATION_GUIDE).read_text(encoding="utf-8"), _ANIMATION_GUIDE
)[0].code
_WORKFLOWS_GUIDE = "docs/user-guide/workflows.md"
SETUPS["workflows_outbound_inbound"] = collect_blocks(
    (ROOT / _WORKFLOWS_GUIDE).read_text(encoding="utf-8"), _WORKFLOWS_GUIDE
)[0].code


def _fixture_namespace() -> dict[str, object]:
    rng = np.random.default_rng(0)
    times = np.arange(1800) / 30.0
    positions = (
        50 + 40 * np.c_[np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    spikes = np.sort(rng.uniform(0, times[-1], 300))
    velocity = np.gradient(positions, times, axis=0)
    return {
        "np": np,
        "Environment": Environment,
        "rng": rng,
        "times": times,
        "positions": positions,
        "env": env,
        "spike_times": spikes,
        "spike_times_list": [
            spikes,
            np.sort(rng.uniform(0, times[-1], 150)),
            np.sort(rng.uniform(0, times[-1], 450)),
        ],
        "headings": np.arctan2(velocity[:, 1], velocity[:, 0]),
        "object_positions": np.array([[50.0, 50.0], [75.0, 25.0]]),
        "reward_times": np.array([10.0, 25.0, 40.0]),
        "fields": rng.uniform(size=(20, env.n_bins)),
        "frame_times": np.arange(20) / 30.0,
        "trajectory": positions[:20],
    }


def _run_document(path: str, *, opt_in: bool, seeded: bool) -> int:
    shared = {}
    executed = 0
    for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
        location = f"{path}:{block.line}"
        if block.kind == "skip":
            assert block.arg, f"{location}: 'docs-test: skip' needs a reason"
            continue
        if opt_in and block.kind not in {"run", "raises"}:
            continue
        namespace = _fixture_namespace() if seeded else shared
        if setup := re.search(r"setup=(\w+)", block.arg):
            assert setup[1] in SETUPS, f"{location}: unknown setup {setup[1]}"
            exec(SETUPS[setup[1]], namespace)
        code = compile("\n" * (block.line - 1) + block.code, str(ROOT / path), "exec")
        try:
            if block.kind == "raises":
                assert block.arg, f"{location}: 'docs-test: raises' needs an exception"
                name = block.arg.split()[0]
                expected = getattr(builtins, name, None) or getattr(neurospatial, name)
                with pytest.raises(expected):
                    exec(code, namespace)
            else:
                exec(code, namespace)
        finally:
            plt.close("all")
        executed += 1
    return executed


@pytest.mark.parametrize("path", sorted(RUN_ALL))
def test_documented_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=False, seeded=RUN_ALL[path]) > 0


@pytest.mark.parametrize("path", OPT_IN)
def test_opted_in_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=True, seeded=True) > 0


def test_collector_markers():
    source = (
        "# Examples\n```python\nx = 1\n```\n"
        "<!-- docs-test: skip external data -->\n```python\nmissing()\n```\n"
        "<!-- docs-test: raises ValueError -->\n```python\nraise ValueError('bad')\n```\n"
        "<!-- docs-test: run setup=sample -->\n```python\ny = x + 1\n```\n"
        "  ```python\n  z = 3\n  ```\n"
    )
    assert collect_blocks(source, "example.md") == [
        DocBlock("example.md", 3, "x = 1\n", None, ""),
        DocBlock("example.md", 7, "missing()\n", "skip", "external data"),
        DocBlock("example.md", 11, "raise ValueError('bad')\n", "raises", "ValueError"),
        DocBlock("example.md", 15, "y = x + 1\n", "run", "setup=sample"),
        DocBlock("example.md", 18, "z = 3\n", None, ""),
    ]


def test_collector_does_not_attach_a_marker_across_blank_lines():
    source = "<!-- docs-test: skip external data -->\n\n```python\nx = 1\n```\n"
    assert collect_blocks(source, "example.md")[0].kind is None


@pytest.fixture
def example_document(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "example.md"


def test_skip_requires_a_reason(example_document):
    example_document.write_text("<!-- docs-test: skip -->\n```python\nmissing()\n```\n")
    with pytest.raises(AssertionError, match="needs a reason"):
        _run_document("example.md", opt_in=False, seeded=False)


def test_unseeded_blocks_share_reader_state(example_document):
    example_document.write_text(
        "```python\nvalues = []\n```\n```python\nvalues.append(7)\nassert values == [7]\n```\n"
    )
    assert _run_document("example.md", opt_in=False, seeded=False) == 2


def test_seeded_blocks_have_fresh_arrays_and_regions(example_document):
    example_document.write_text(
        "```python\nfrom neurospatial import Regions\ntimes[:] = 0\n"
        "region_builder = Regions(env.regions.values())\n"
        "region_builder.add('temporary', point=(50., 50.))\n"
        "env = env.with_regions(region_builder)\n```\n"
        "```python\nassert np.all(np.diff(times) > 0)\nassert 'temporary' not in env.regions\n```\n"
    )
    assert _run_document("example.md", opt_in=False, seeded=True) == 2


def test_opted_in_documents_check_raises_markers(example_document):
    example_document.write_text(
        "```python\nunmarked()\n```\n"
        "<!-- docs-test: raises ValueError -->\n```python\nx = 1\n```\n"
    )
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        _run_document("example.md", opt_in=True, seeded=True)


def test_marker_setups_are_known_and_used():
    used = set()
    for path in set(RUN_ALL) | set(OPT_IN):
        for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
            if block.kind == "skip":
                assert block.arg, f"{path}:{block.line}: skip needs a reason"
            if match := re.search(r"setup=(\w+)", block.arg):
                used.add(match[1])
    assert used == set(SETUPS), f"Unknown or unused setups: {used ^ set(SETUPS)}"
