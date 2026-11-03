"""Run the operation registry on datasets and write ``datasets/results/``.

Usage::

    MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple \\
        python datasets/run.py [--datasets KEY ...] [--ops OP_ID ...] [--render-only]

Writes one ``results/<key>.json`` per dataset, then regenerates
``results/matrix.md`` (status, runtime and peak memory per dataset and
operation) and ``results/warnings.md`` (warnings on clean data and hazard
probes, joined with the verdicts in ``verdicts.json``).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from _calibration import captured, probe_hazards
from _record import check_expectations, validate_recording
from operations import OPERATIONS, Outcome

LOADERS = {
    "sargolini_000582": "load_sargolini_000582",
    "peyrache_mouse32": "load_peyrache_mouse32",
    "figure8_000552": "load_figure8_000552",
    "rosenberg_labyrinth": "load_rosenberg_labyrinth",
    "yartsev_bat_demo": "load_yartsev_bat_demo",
    "miniscope_002030": "load_miniscope_002030",
    "giocomo_000405": "load_giocomo_000405",
    "giocomo_001701": "load_giocomo_001701",
    "gardner_2022_grid": "load_gardner_2022_grid",
    "vollan_lt": "load_vollan_lt",
    "vollan_of": "load_vollan_of",
    "grosmark_000044_circular": "load_grosmark_000044_circular",
    "grosmark_000044_linear": "load_grosmark_000044_linear",
    "jadhav_000447": "load_jadhav_000447",
    "hc11_achilles": "load_hc11_achilles",
    "hc3_ec013_669": "load_hc3_ec013_669",
    "hc2_ec013_527": "load_hc2_ec013_527",
    "hc28_hpa": "load_hc28_hpa",
    "hc31_anm190422": "load_hc31_anm190422",
    "hc18_train261": "load_hc18_train261",
    "grieves_2020_lattice": "load_grieves_2020_lattice",
    "watermaze_10248158": "load_watermaze_10248158",
    "olafsdottir_5566548": "load_olafsdottir_5566548",
    "takigawa_7114733": "load_takigawa_7114733",
    "hafting_2008_nird": "load_hafting_2008_nird",
    "j16_23020757": "load_j16_23020757",
    "agarwal_24757638": "load_agarwal_24757638",
    "shipley_30902972": "load_shipley_30902972",
    "rsc_000987": "load_rsc_000987",
    "ca1ca3_001880": "load_ca1ca3_001880",
    "gonzalez_001695": "load_gonzalez_001695",
    "neurolab_001754": "load_neurolab_001754",
    "valero_000568": "load_valero_000568",
    "aeryjones_000165": "load_aeryjones_000165",
    "dyads_001845": "load_dyads_001845",
    "senzai_000003": "load_senzai_000003",
    "kanter_2025": "load_kanter_2025",
}
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VERDICTS = HERE / "verdicts.json"
_STATUS_CODE = {
    "ok": "ok",
    "workaround": "wk",
    "missing": "miss",
    "error": "err",
    "n/a": "–",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _versions() -> dict[str, str]:
    installed = {
        dist.metadata["Name"].lower(): dist.version
        for dist in importlib.metadata.distributions()
    }
    versions = {
        package: installed.get(package, "not installed")
        for package in (
            "neurospatial",
            "numpy",
            "scipy",
            "pynwb",
            "pynapple",
            "opexebo",
        )
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    versions["git_commit"] = commit.stdout.strip()
    return versions


def run_dataset(
    key: str,
    op_ids: Sequence[str] | None = None,
    operations: tuple[Any, ...] = OPERATIONS,
) -> dict[str, Any]:
    """Load one dataset, run the registry and the hazard probes, write its JSON."""
    module = importlib.import_module(LOADERS[key])
    rec, load_error, load_warnings = captured(module.load)
    if load_error is not None:
        raise load_error  # a loader that fails is a harness bug, not a table cell
    validate_recording(rec)
    return run_recording(rec, module, op_ids, operations, load_warnings)


def run_recording(
    rec: Any,
    module: Any,
    op_ids: Sequence[str] | None = None,
    operations: tuple[Any, ...] = OPERATIONS,
    load_warnings: list[dict[str, Any]] | None = None,
    results: Path = RESULTS,
) -> dict[str, Any]:
    """Run ``operations`` and the hazard probes on a loaded recording.

    With ``op_ids``, only those operations are re-run and their entries are
    merged into the dataset's existing results file, so the other cells keep
    their last results.
    """
    path = results / f"{rec.dataset}.json"
    previous = (
        json.loads(path.read_text())["operations"]
        if op_ids is not None and path.exists()
        else {}
    )
    record: dict[str, Any] = {
        "dataset": rec.dataset,
        "session": rec.session,
        "source": rec.source,
        "license": rec.license,
        "conversions": list(rec.conversions),
        "expectations": check_expectations(rec, getattr(module, "EXPECTATIONS", {})),
        "versions": _versions(),
        "load_warnings": load_warnings or [],
        "operations": dict(previous),
    }
    cache: dict[str, Any] = {}
    for op in operations:
        if op_ids is not None and op.op_id not in op_ids:
            continue
        if missing := [name for name in op.requires if getattr(rec, name) is None]:
            record["operations"][op.op_id] = {
                "status": "n/a",
                "note": f"needs {', '.join(missing)}",
                "warnings": [],
            }
            continue
        tracemalloc.start()
        start = time.perf_counter()
        outcome, error, caught = captured(op.run, rec, cache)
        if error is not None:
            outcome = Outcome("error", note=f"{type(error).__name__}: {error}")
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["operations"][op.op_id] = {
            **asdict(outcome),
            "warnings": caught,
            "seconds": round(seconds, 3),
            "peak_mb": round(peak / 1e6, 1),
        }
    record["hazards"] = probe_hazards(module, rec, record["operations"], operations)
    results.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_jsonable) + "\n")
    return record


def _cell(entry: dict[str, Any], cell_gap: str | None = None) -> str:
    text = _STATUS_CODE[entry["status"]]
    if gap := entry.get("gap") or (cell_gap if entry["status"] != "ok" else None):
        text += f" {gap}"
    if entry.get("warnings"):
        text += f" ⚠{len(entry['warnings'])}"
    return text


def render_matrix(
    records: Sequence[dict[str, Any]], cell_gaps: dict[str, Any] | None = None
) -> str:
    """Render the dataset × operation status and cost tables.

    ``cell_gaps`` maps ``"<dataset>/<op>"`` to ``{"gap": "G-###"}`` for cells
    that are not ``ok`` and carry no gap of their own (errors, and
    workarounds whose gap depends on the dataset); the gap entry is written
    after the run.
    """
    cell_gaps = cell_gaps or {}
    op_ids = [op.op_id for op in OPERATIONS]
    head = "| dataset | expectations | " + " | ".join(op_ids) + " |"
    rule = "| --- " * (len(op_ids) + 2) + "|"
    lines = [
        "# Dataset × operation matrix",
        "",
        "Generated by `datasets/run.py`; do not edit. Cells: `ok`, `wk G-###`"
        " (workaround), `miss G-###`, `err G-###` (the message is in the JSON),"
        " `–` (n/a)."
        " `⚠N`: N distinct warning messages (see `warnings.md`).",
        "",
        head,
        rule,
    ]
    for record in records:
        expectations = record["expectations"]
        passed = sum(row["passed"] for row in expectations)
        cells = [
            _cell(
                record["operations"][op_id],
                cell_gaps.get(f"{record['dataset']}/{op_id}", {}).get("gap"),
            )
            if op_id in record["operations"]
            else ""
            for op_id in op_ids
        ]
        lines.append(
            f"| {record['dataset']} | {passed}/{len(expectations)} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Runtime (s) / peak traced memory (MB)",
        "",
        head.replace(" expectations |", ""),
        "| --- " * (len(op_ids) + 1) + "|",
    ]
    for record in records:
        cells = []
        for op_id in op_ids:
            entry = record["operations"].get(op_id, {})
            cells.append(
                f"{entry['seconds']} / {entry['peak_mb']}" if "seconds" in entry else ""
            )
        lines.append(f"| {record['dataset']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Memory is from `tracemalloc`: NumPy allocations are traced, JAX device"
        " memory is not.",
        "",
    ]
    return "\n".join(lines)


def _md(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summary(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    return ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in summary.items()
    )


def warning_key(row: dict[str, Any]) -> str:
    """Verdict key of a clean-data warning row."""
    return f"{row['origin']} {row['category']}: {row['template']}"


def hazard_key(dataset: str, row: dict[str, Any]) -> str:
    """Verdict key of a hazard-probe row."""
    return f"{dataset}/{row['hazard']}/{row['op']}"


def render_warnings(records: Sequence[dict[str, Any]], verdicts: dict[str, Any]) -> str:
    """Render the warnings-on-clean-data and hazard-probe tables.

    ``verdicts`` has a ``"warnings"`` section keyed by ``warning_key`` and a
    ``"hazards"`` section keyed by ``hazard_key``.
    """
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        sources = [("load", record["load_warnings"])] + [
            (op_id, entry.get("warnings", []))
            for op_id, entry in record["operations"].items()
        ]
        for where, rows in sources:
            for row in rows:
                group = groups.setdefault(
                    warning_key(row),
                    {**row, "where": [], "total": 0, "max_per_call": 0},
                )
                group["where"].append(f"{record['dataset']}:{where}")
                group["total"] += row["count"]
                group["max_per_call"] = max(group["max_per_call"], row["count"])

    def verdict_cells(section: str, key: str, response: str = "") -> str:
        if response == "not probed":
            return "– |  | "
        verdict = verdicts.get(section, {}).get(key)
        if verdict is None:
            return "unreviewed |  | "
        return f"{verdict['verdict']} | {verdict.get('gap') or ''} | {_md(verdict.get('reason', ''), 200)}"

    lines = [
        "# Warning calibration",
        "",
        "Generated by `datasets/run.py`; do not edit. Verdicts come from"
        " `datasets/verdicts.json`. A hazard probe counts as `warned` only for"
        " warnings the clean run of the same operation did not emit.",
        "",
        "## Warnings on correctly loaded data",
        "",
        "| category | origin | example | where | total | max per call | verdict | gap | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, group in sorted(groups.items()):
        lines.append(
            f"| {group['category']} | {group['origin']} | {_md(group['example'])} |"
            f" {', '.join(sorted(set(group['where'])))} | {group['total']} |"
            f" {group['max_per_call']} | {verdict_cells('warnings', key)} |"
        )
    lines += [
        "",
        "## Hazard probes",
        "",
        "| dataset | hazard | op | response | new message | clean warnings repeated |"
        " clean | with hazard | verdict | gap | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        for row in record.get("hazards", []):
            lines.append(
                f"| {record['dataset']} | {row['hazard']} | {row['op']} | {row['response']} |"
                f" {_md(row['message'])} | {row.get('repeated_clean_warnings', 0)} |"
                f" {_md(_summary(row['clean_summary']))} |"
                f" {_md(_summary(row['hazard_summary']))} |"
                f" {verdict_cells('hazards', hazard_key(record['dataset'], row), row['response'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def render(results: Path = RESULTS, verdicts_path: Path = VERDICTS) -> None:
    """Regenerate ``matrix.md`` and ``warnings.md`` from every results JSON."""
    records = [json.loads(path.read_text()) for path in sorted(results.glob("*.json"))]
    verdicts = json.loads(verdicts_path.read_text()) if verdicts_path.exists() else {}
    (results / "matrix.md").write_text(render_matrix(records, verdicts.get("cells")))
    (results / "warnings.md").write_text(render_warnings(records, verdicts))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=sorted(LOADERS), default=None)
    parser.add_argument(
        "--ops", nargs="+", choices=[op.op_id for op in OPERATIONS], default=None
    )
    parser.add_argument(
        "--render-only", action="store_true", help="only regenerate the Markdown tables"
    )
    args = parser.parse_args(argv)
    if not args.render_only:
        for key in args.datasets or LOADERS:
            print(f"running {key}")
            record = run_dataset(key, args.ops)
            statuses = {
                op: entry["status"] for op, entry in record["operations"].items()
            }
            print(f"  {statuses}")
    render()


if __name__ == "__main__":
    main()
