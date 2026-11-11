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
    "docs/user-guide/compute-on-trials.md": False,
    "docs/user-guide/population-analysis.md": False,
    "docs/user-guide/reactivation-methods.md": False,
    "docs/user-guide/regions.md": False,
    "docs/user-guide/replay-methods.md": False,
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
    "docs/user-guide/datasets.md",
    "docs/datasets/bhattarai_2020.md",
    "docs/datasets/carey_2019.md",
    "docs/datasets/comrie_2026.md",
    "docs/datasets/forli_2025.md",
    "docs/datasets/gardner_2022.md",
    "docs/datasets/grieves_2020.md",
    "docs/datasets/grosmark_2016.md",
    "docs/datasets/huszar_2022.md",
    "docs/datasets/huszar_2026.md",
    "docs/datasets/igata_2021.md",
    "docs/datasets/kanter_2025.md",
    "docs/datasets/peyrache_2015.md",
    "docs/datasets/peyrache_rsc.md",
    "docs/datasets/pfeiffer_2013.md",
    "docs/datasets/sargolini_2006.md",
    "docs/datasets/shipley_2026.md",
    "docs/datasets/valero_2022.md",
    "docs/datasets/vollan_2025.md",
    "docs/user-guide/workflows.md",
    "docs/user-guide/neuroscience-metrics.md",
    "docs/user-guide/signal-processing-primitives.md",
    "docs/migration/v0.6.md",
    "docs/migration/environment-unified.md",
    "docs/migration/environment-geometry.md",
    "docs/migration/environment-architecture.md",
)
SETUPS = {
    "docs_datasets_session": """import numpy as np
import pandas as pd

from neurospatial.behavior import heading_from_velocity
from neurospatial.datasets import Provenance, Recording
from neurospatial.simulation import open_field_session

# A 30 s simulated box session stands in for load(...) so the fences run offline.
_sim = open_field_session(duration=30.0, n_place_cells=6, rng=0)
rec = Recording(
    dataset="simulated_2026",
    session="box",
    paper="Simulated et al. (2026)",
    animal="simulated rat",
    apparatus="simulated 1 m box",
    brain_regions=("CA1",),
    license="CC0-1.0",
    citation="Simulated A (2026) A simulated session.",
    provenance=Provenance(
        archive="Simulated 0",
        url="file:///simulated/session.npz",
        sha256="0" * 64,
        size_bytes=1024,
        neurospatial_version="0.0",
        fetched_at="2026-11-03T00:00:00+00:00",
    ),
    clock="the simulation's clock, from 0 s",
    times=_sim.times,
    spike_times=tuple(_sim.spike_times),
    units=pd.DataFrame(
        {"brain_region": "CA1", "label": "pyramidal"},
        index=pd.Index(np.arange(len(_sim.spike_times)), name="unit_id"),
    ),
    env=_sim.env,
    env_kind="declared",
    env_source="1 m square box (simulation)",
    positions=_sim.positions,
    position_units="cm",
    tracking="simulated head position",
    headings=heading_from_velocity(_sim.times, _sim.positions),
    heading_source="direction of movement of the simulated path",
    epochs={"run": np.array([[0.0, 30.0]])},
)
""",
    "docs_video_annotation_use_results": """import numpy as np
from types import SimpleNamespace

from neurospatial import Environment

side = np.linspace(0.0, 10.0, 25, endpoint=False)
corners = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
positions = np.concatenate(
    [a + (b - a) * side[:, None] / 10.0 for a, b in zip(corners, np.roll(corners, -1, axis=0))]
)  # one lap around a 10 x 10 arena
times = np.arange(len(positions), dtype=float) / 30.0  # 30 Hz tracking
env = Environment.from_samples(positions, bin_size=1.0)
result = SimpleNamespace(environment=env, regions={})
""",
    "docs_trajectory_region_crossings": """import numpy as np
from shapely.geometry import Point

from neurospatial import Environment

track = np.column_stack([
    np.linspace(0.0, 100.0, 80),
    np.full(80, 50.0),
])
env = Environment.from_samples(track, bin_size=5.0)
from neurospatial import Regions
region_builder = Regions()
region_builder.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))
env = env.with_regions(region_builder)

positions = np.array([
    [10.0, 50.0],
    [30.0, 50.0],
    [50.0, 50.0],
    [70.0, 50.0],
    [50.0, 50.0],
    [30.0, 50.0],
])
times = np.arange(len(positions)) * 0.1  # 10 Hz tracking
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
    env, duration=120.0, dt=0.02, speed_mean=15.0, rng=0, speed_units="cm"
)
cells = [
    PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, rng=i)
    for i, c in enumerate(np.linspace(5.0, 95.0, 15))
]
spike_times = generate_population_spikes(
    cells, times, positions, rng=0, show_progress=False
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
    env, duration=120.0, dt=0.02, speed_mean=15.0, rng=0, speed_units="cm"
)

def load_all_neurons():
    cells = [
        PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, rng=i)
        for i, c in enumerate(np.linspace(5.0, 95.0, 5))
    ]
    return {
        f"unit_{i}": generate_poisson_spikes(
            cell.firing_rate(times, positions), times, rng=i
        )
        for i, cell in enumerate(cells)
    }
""",
}
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[ \t]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_MARKER = re.compile(
    r"<!--\s*docs-test:\s*(?P<kind>run|skip|raises)\b(?P<arg>[^>]*?)\s*-->"
)
# The message text is literal, not a regular expression.
_RAISES = re.compile(r'(?P<name>\w+)(?:\s+match="(?P<match>[^"]+)")?')


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
_TRAJECTORY_GUIDE = "docs/user-guide/trajectory-and-behavioral-analysis.md"
# The labels and lost-tracking examples continue from the trials detected
# from positions.
SETUPS["docs_trials_from_positions"] = next(
    block.code
    for block in collect_blocks(
        (ROOT / _TRAJECTORY_GUIDE).read_text(encoding="utf-8"), _TRAJECTORY_GUIDE
    )
    if "trials = detect_trials(env, times, positions, regions=list(wells))"
    in block.code
)
_WORKFLOWS_GUIDE = "docs/user-guide/workflows.md"
_WORKFLOWS_BLOCKS = collect_blocks(
    (ROOT / _WORKFLOWS_GUIDE).read_text(encoding="utf-8"), _WORKFLOWS_GUIDE
)
SETUPS["workflows_outbound_inbound"] = _WORKFLOWS_BLOCKS[0].code


def _workflows_block(marker: str) -> str:
    """The one block of the workflows guide containing ``marker``."""
    matching = [block.code for block in _WORKFLOWS_BLOCKS if marker in block.code]
    assert len(matching) == 1, f"{len(matching)} workflow blocks contain {marker!r}"
    return matching[0]


# Workflow 9 continues block by block: each setup is the blocks before it.
SETUPS["workflows_route_environment"] = _workflows_block("make_w_maze().env_track")
SETUPS["workflows_route_recording"] = "\n".join(
    [SETUPS["workflows_route_environment"], _workflows_block("def run(start, goal")]
)
SETUPS["workflows_route_trials"] = "\n".join(
    [
        SETUPS["workflows_route_recording"],
        _workflows_block("trials = detect_trials(track"),
    ]
)
SETUPS["workflows_route_progress"] = "\n".join(
    [SETUPS["workflows_route_trials"], _workflows_block("progress = path_progress(")]
)


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
                raises = _RAISES.fullmatch(block.arg)
                assert raises, (
                    f"{location}: write 'docs-test: raises <Exception>' with an "
                    f'optional match="<text in the message>"; got {block.arg!r}'
                )
                name = raises["name"]
                expected = getattr(builtins, name, None) or getattr(neurospatial, name)
                match = None if raises["match"] is None else re.escape(raises["match"])
                with pytest.raises(expected, match=match):
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


def test_raises_marker_matches_message_text(example_document):
    marker = '<!-- docs-test: raises ValueError match="bad (x.y)" -->\n'
    example_document.write_text(
        marker + "```python\nraise ValueError('bad (x.y)')\n```\n"
    )
    assert _run_document("example.md", opt_in=False, seeded=False) == 1
    example_document.write_text(
        marker + "```python\nraise ValueError('bad (xzy)')\n```\n"
    )
    with pytest.raises(AssertionError, match="Regex pattern did not match"):
        _run_document("example.md", opt_in=False, seeded=False)


def test_raises_marker_refuses_unknown_text(example_document):
    example_document.write_text(
        "<!-- docs-test: raises ValueError bad -->\n```python\nraise ValueError\n```\n"
    )
    with pytest.raises(AssertionError, match="optional match="):
        _run_document("example.md", opt_in=False, seeded=False)


def test_marker_setups_are_known_and_used():
    used = set()
    for path in set(RUN_ALL) | set(OPT_IN):
        for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
            if block.kind == "skip":
                assert block.arg, f"{path}:{block.line}: skip needs a reason"
            if match := re.search(r"setup=(\w+)", block.arg):
                used.add(match[1])
    assert used == set(SETUPS), f"Unknown or unused setups: {used ^ set(SETUPS)}"
