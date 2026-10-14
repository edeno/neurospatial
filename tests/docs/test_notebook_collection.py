"""The tutorial collection has one catalog, valid prerequisites and synced sources."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

jupytext = pytest.importorskip("jupytext")
nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SOURCES = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def test_core_collection_follows_nine_user_workflows():
    assert [source.stem for source in SOURCES] == [
        "01_getting_started",
        "02_spatial_rate_maps",
        "03_trajectories_and_behavior",
        "04_real_data_analysis",
        "05_simulation_and_validation",
        "06_position_decoding",
        "07_peri_event_analysis",
        "08_visualization",
        "09_nwb_io",
    ]


def _markdown(path: Path) -> str:
    return "\n\n".join(
        cell.source
        for cell in jupytext.read(path).cells
        if cell.cell_type == "markdown"
    )


def _local_links(markdown: str) -> list[str]:
    # A copied command or illustrative Markdown fence does not create a link.
    prose = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    return [
        unquote(urlsplit(target).path)
        for target in LINK.findall(prose)
        if not urlsplit(target).scheme and urlsplit(target).path
    ]


def _metadata(path: Path) -> tuple[str, str, tuple[str, ...]]:
    markdown = _markdown(path)
    title = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    estimates = re.findall(
        r"^\*\*Estimated time\*\*:\s*([^\n]+)", markdown, re.MULTILINE
    )
    prerequisites = re.search(
        r"^\*\*Prerequisites\*\*:\s*(.*?)(?:\n\s*\n|\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert title, f"{path.name} needs a notebook title"
    assert len(estimates) == 1, f"{path.name} needs one estimated time"
    assert prerequisites, f"{path.name} needs explicit prerequisites (or None)"
    dependencies = tuple(
        target
        for target in _local_links(prerequisites.group(1))
        if target.endswith(".ipynb")
    )
    return title.group(1), estimates[0].rstrip("."), dependencies


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.stem)
def test_notebook_pair_and_docs_mirror_agree(source):
    notebook = source.with_suffix(".ipynb")
    nbformat.validate(nbformat.read(notebook, as_version=4))
    expected = [
        (cell.cell_type, cell.source.strip()) for cell in jupytext.read(source).cells
    ]
    actual = [
        (cell.cell_type, cell.source.strip()) for cell in jupytext.read(notebook).cells
    ]
    assert actual == expected, f"Run jupytext --sync {source.relative_to(ROOT)}"
    for path in (source, notebook):
        assert path.read_bytes() == (ROOT / "docs/examples" / path.name).read_bytes(), (
            "Run python docs/sync_notebooks.py"
        )


def test_catalog_covers_every_notebook_and_matches_its_prerequisites():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    catalog = readme.split("<!-- notebook-catalog:start -->", 1)[1].split(
        "<!-- notebook-catalog:end -->", 1
    )[0]
    rows = [line for line in catalog.splitlines() if re.match(r"\| \[\d\d\.", line)]
    assert len(rows) == len(SOURCES)
    for source in SOURCES:
        title, estimate, dependencies = _metadata(source)
        filename = source.with_suffix(".ipynb").name
        primary = f"| [{source.name[:2]}. {title}]({filename}) |"
        row = next(line for line in rows if line.startswith(primary))
        assert estimate in row
        row_dependencies = tuple(_local_links(row))[1:]
        assert row_dependencies == dependencies
    index = (ROOT / "docs/examples/index.md").read_text(encoding="utf-8")
    assert '--8<-- "examples/README.md"' in index


def test_prerequisites_are_available_and_acyclic():
    prerequisites = {
        source.with_suffix(".ipynb").name: _metadata(source)[2] for source in SOURCES
    }

    def visit(notebook, ancestors):
        branch = [*ancestors, notebook]
        assert notebook not in ancestors, f"Circular prerequisites: {branch}"
        assert notebook in prerequisites, f"Unknown prerequisite {notebook}"
        for dependency in prerequisites[notebook]:
            visit(dependency, branch)

    for notebook in prerequisites:
        visit(notebook, [])


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.stem)
def test_notebook_links_work_in_a_local_checkout(source):
    for target in _local_links(_markdown(source)):
        assert (source.parent / target).exists(), (
            f"{source.name}: missing link target {target}"
        )


def test_ci_executes_every_notebook_once():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/test_notebooks.yml").read_text(encoding="utf-8")
    )
    groups = workflow["jobs"]["execute-notebooks"]["strategy"]["matrix"]["include"]
    selected = [
        path
        for group in groups
        for pattern in group["notebooks"].split()
        for path in ROOT.glob(pattern)
    ]
    expected = {source.with_suffix(".ipynb") for source in SOURCES}
    assert set(selected) == expected
    assert len(selected) == len(expected), (
        "A notebook is listed in more than one CI group"
    )
