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
    "CLAUDE.md": True,
}
OPT_IN = (
    ".claude/QUICKSTART.md",
    "docs/user-guide/alignment.md",
    "docs/user-guide/animation.md",
    "docs/user-guide/interoperability.md",
    "docs/user-guide/trajectory-and-behavioral-analysis.md",
    "docs/user-guide/video-annotation.md",
    "docs/user-guide/workflows.md",
    "docs/migration/v0.6.md",
)
SETUPS = {
    "docs_animation_quick_start": """import os
import numpy as np
os.environ["MPLBACKEND"] = "Agg"

from neurospatial import Environment

positions = np.column_stack([
    np.linspace(0.0, 20.0, 60),
    5.0 + np.sin(np.linspace(0.0, 2.0 * np.pi, 60)),
])
times = np.linspace(0.0, 10.0, len(positions))
spikes = [np.array([1.0, 3.0, 6.0]) for _ in range(30)]

def _check_animate_fields(self, fields, *, frame_times, **kwargs):
    assert len(frame_times) == len(fields)
    return kwargs.get("save_path")

Environment.animate_fields = _check_animate_fields
""",
    "docs_video_annotation_use_results": """import numpy as np
from types import SimpleNamespace

from neurospatial import Environment

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)
times = np.arange(len(positions), dtype=float)
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
env.regions.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))

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
    "quickstart_vte_session": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.behavior.segmentation import Trial

# Animal runs back and forth along x through a central decision zone.
seg = np.linspace(0.0, 100.0, 200)
xs = np.concatenate([seg, seg[::-1]])
ys = 50.0 + 2.0 * np.sin(np.linspace(0.0, 8 * np.pi, len(xs)))
positions = np.column_stack([xs, ys])
times = np.linspace(0.0, 40.0, len(positions))

env = Environment.from_samples(positions, bin_size=4.0)
env.regions.add("center", polygon=box(40.0, 30.0, 60.0, 70.0))

trials = [
    Trial(start_time=0.0, end_time=20.0, start_region="start",
          end_region="goal", success=True),
    Trial(start_time=20.0, end_time=40.0, start_region="goal",
          end_region="start", success=True),
]
""",
    "quickstart_ovc_classify_single": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np

from neurospatial import Environment
from neurospatial.ops.egocentric import heading_from_velocity

t = np.linspace(0.0, 20.0, 400)
positions = np.column_stack([
    50.0 + 30.0 * np.cos(2 * np.pi * t / 20.0),
    50.0 + 30.0 * np.sin(2 * np.pi * t / 20.0),
])
times = t
env = Environment.from_samples(positions, bin_size=4.0)
headings = heading_from_velocity(times, positions, min_speed=1.0)
object_positions = np.array([[50.0, 30.0], [80.0, 60.0]])
spike_times = times[::7]
""",
    "quickstart_view_classify": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import compute_view_rate, compute_view_rates
from neurospatial.ops.egocentric import heading_from_velocity

t = np.linspace(0.0, 20.0, 400)
positions = np.column_stack([
    50.0 + 30.0 * np.cos(2 * np.pi * t / 20.0),
    50.0 + 30.0 * np.sin(2 * np.pi * t / 20.0),
])
times = t
env = Environment.from_samples(positions, bin_size=4.0)
headings = heading_from_velocity(times, positions, min_speed=1.0)
single = compute_view_rate(
    env, times[::7], times, positions, headings, view_distance=10.0
)
batch = compute_view_rates(
    env, [times[::7], times[::9], times[::11]], times, positions,
    headings, view_distance=10.0,
)
""",
    "quickstart_circular_basis_metrics": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np

rng = np.random.default_rng(0)
n = 2000
head_direction_angles = rng.uniform(-np.pi, np.pi, n)
rate = 2.0 + 1.5 * np.cos(head_direction_angles - 0.7)
spike_counts = rng.poisson(rate)
""",
    "quickstart_overlay_block": """import os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np

from neurospatial import Environment

n = 30
trajectory = np.column_stack([
    np.linspace(0.0, 20.0, n),
    5.0 + np.sin(np.linspace(0.0, 2.0 * np.pi, n)),
])
traj1 = trajectory
traj2 = trajectory + np.array([1.0, 1.0])
nose_traj = trajectory + np.array([0.5, 0.0])
body_traj = trajectory
tail_traj = trajectory - np.array([0.5, 0.0])

positions = trajectory
env = Environment.from_samples(positions, bin_size=2.0)
fields = [np.zeros(env.n_bins) for _ in range(n)]
frame_times = np.arange(n) / 30.0

def _noop_animate_fields(self, fields, *, frame_times, overlays=None, **kwargs):
    assert len(frame_times) == len(fields)
    if overlays:
        for ov in overlays:
            ov.convert_to_data(
                np.asarray(frame_times, dtype=float), len(fields), self
            )
    return kwargs.get("save_path")

Environment.animate_fields = _noop_animate_fields
""",
    "quickstart_events_glm_regressors": """import numpy as np

sample_times = np.linspace(0.0, 60.0, 600)
reward_times = np.array([5.0, 12.0, 23.0, 41.0, 55.0])
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
env.units = "cm"
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
env.units = "cm"
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
        if opt_in and block.kind != "run":
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
    monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=False, seeded=RUN_ALL[path]) > 0


@pytest.mark.parametrize("path", OPT_IN)
def test_opted_in_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)
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
        "```python\ntimes[:] = 0\nenv.regions.add('temporary', point=(50., 50.))\n```\n"
        "```python\nassert np.all(np.diff(times) > 0)\nassert 'temporary' not in env.regions\n```\n"
    )
    assert _run_document("example.md", opt_in=False, seeded=True) == 2


def test_marker_setups_are_known_and_used():
    used = set()
    for path in set(RUN_ALL) | set(OPT_IN):
        for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
            if block.kind == "skip":
                assert block.arg, f"{path}:{block.line}: skip needs a reason"
            if match := re.search(r"setup=(\w+)", block.arg):
                used.add(match[1])
    assert used == set(SETUPS), f"Unknown or unused setups: {used ^ set(SETUPS)}"


def test_stubs_do_not_leak(tmp_path):
    original = Environment.animate_fields
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)
        monkeypatch.chdir(tmp_path)
        _run_document("docs/user-guide/animation.md", opt_in=True, seeded=True)
        assert Environment.animate_fields is not original
    assert Environment.animate_fields is original
