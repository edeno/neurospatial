# Designs

[← back to PLAN.md](PLAN.md)

The code in `datasets/` is canonical once a phase has shipped; the samples here
record the design and may lag it (phase 1 found that `spatial_information`,
`grid_scores`, `border_scores` and `mean_vector_lengths` are methods, that
`DecodingResult.error_against` replaces the harness interpolation, and that
`_warnings` shadows a CPython module, hence `_calibration.py`).

- [Fetch and cache](#fetch-and-cache)
- [Record validation](#record-validation)
- [Expectations](#expectations)
- [Operation registry](#operation-registry)
- [Driver](#driver)
- [Warning calibration](#warning-calibration)
- [Reference checks](#reference-checks)
- [Loader recipe](#loader-recipe)

## Fetch and cache

`datasets/_fetch.py`. Generalizes `data/load_bandit_data.py` (pinned SHA-256,
temp file in the destination directory, never overwrite a mismatched file).

```python
"""Download public dataset files once into a local cache and verify them."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "datasets"


def cache_root(cache: Path | None = None) -> Path:
    """Return the cache directory: ``cache``, ``$NEUROSPATIAL_DATA_DIR`` or ``data/datasets``."""
    root = Path(cache or os.environ.get("NEUROSPATIAL_DATA_DIR", DEFAULT_CACHE))
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing(dest: Path, sha256: str) -> Path | None:
    if not dest.exists():
        return None
    if file_sha256(dest) != sha256:
        msg = f"{dest} exists with a different SHA-256 from the pinned file. Move it aside to download again."
        raise FileExistsError(msg)
    return dest


def _install(tmp: Path, dest: Path, sha256: str) -> Path:
    actual = file_sha256(tmp)
    if actual != sha256:
        tmp.unlink()
        msg = f"{dest.name}: downloaded SHA-256 {actual} does not match the pinned {sha256}."
        raise ValueError(msg)
    tmp.replace(dest)
    return dest


def _download_to_temp(dest: Path, write) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            write(tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path


def fetch(url: str, name: str, *, sha256: str, subdir: str, cache: Path | None = None) -> Path:
    """Return a verified local copy of ``url``, downloading it once."""
    dest = cache_root(cache) / subdir / name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp):
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


def fetch_zip_member(url: str, member: str, *, sha256: str, subdir: str, cache: Path | None = None) -> Path:
    """Extract one member of a remote zip with HTTP range reads; zipfile checks its CRC-32."""
    import fsspec

    dest = cache_root(cache) / subdir / Path(member).name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp):
        with (
            fsspec.open(url, "rb", block_size=4 << 20) as remote,
            zipfile.ZipFile(remote) as archive,
            archive.open(member) as source,
        ):
            shutil.copyfileobj(source, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


def dandi_asset_url(asset_id: str) -> str:
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
```

Pinning: for DANDI, take the SHA-256 from the asset metadata
(`DandiAPIClient().get_dandiset(id, version).get_asset(asset_id).get_raw_metadata()["digest"]["dandi:sha2-256"]`)
and record the published version, not `draft`, whenever one exists. For Zenodo,
check the downloaded file's MD5 against the record's `files[].checksum`, then
pin the SHA-256. For OSF, compare against NeMoS's pooch registry
(`src/nemos/fetch/fetch_data.py` in the NeMoS repo) when the file is listed
there. Zip members are verified by zipfile's CRC-32; pin their SHA-256 after
the first extraction. Grieves 2020 (later batch) uses Deflate64, which `zipfile`
cannot read; that needs `inflate64` and is out of scope until then.

## Record validation

`datasets/_record.py` holds `Recording` ([contract](shared-contracts.md#recording))
and `validate_recording`, which collects every violation of the contract
invariants and raises one `ValueError` listing them all.

## Expectations

Also in `datasets/_record.py`. A small fixed set of measurements, plus
free-form predicates.

```python
MEASURES: dict[str, Callable[[Recording], Any]] = {
    "n_units": lambda r: None if r.spike_times is None else len(r.spike_times),
    "n_cells": lambda r: None if r.activity is None else r.activity.shape[1],
    "n_dims": lambda r: None if r.positions is None else r.positions.shape[1],
    "sampling_rate_hz": lambda r: None if r.times is None else float(1.0 / np.median(np.diff(r.times))),
    "duration_s": lambda r: None if r.times is None else float(r.times[-1] - r.times[0]),
    "position_min": lambda r: None if r.positions is None else np.nanmin(r.positions, axis=0),
    "position_max": lambda r: None if r.positions is None else np.nanmax(r.positions, axis=0),
    "position_nan_fraction": lambda r: None if r.positions is None else float(np.isnan(r.positions).any(axis=1).mean()),
    "epoch_names": lambda r: sorted(r.epochs),
    "event_names": lambda r: sorted(r.events),
}


def check_expectations(rec: Recording, expectations: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate pre-registered expectations; never raises on a failed check."""
    rows = []
    for key, expected in expectations.items():
        if callable(expected):
            rows.append({"key": key, "passed": bool(expected(rec)), "detail": "predicate"})
            continue
        if key not in MEASURES:
            msg = f"Unknown expectation {key!r}; use one of {sorted(MEASURES)} or a predicate."
            raise KeyError(msg)
        actual = MEASURES[key](rec)
        if isinstance(expected, tuple) and len(expected) == 2 and actual is not None:
            low, high = expected
            values = np.asarray(actual, dtype=float)
            passed = bool(np.all((values >= np.asarray(low)) & (values <= np.asarray(high))))
        else:
            passed = actual is not None and np.array_equal(np.asarray(actual), np.asarray(expected))
        rows.append({"key": key, "passed": passed, "detail": f"expected {expected!r}, got {actual!r}"})
    return rows
```

A failed expectation is a finding: the loader author decides whether the
expectation, the loader, or neurospatial is wrong, and logs a gap if it's not
the expectation. Never edit an expectation to make it pass without a gap entry
or a commit message explaining the source of the original belief.

## Operation registry

`datasets/operations.py`. Phase 1 operations; later phases append their own.
The per-recording `cache` dict shares the environment and rates across
operations, so the first operation that needs them pays the cost (order the
registry so `env.from_samples` and `rates.spatial` come first).

```python
import numpy as np
from neurospatial import Environment
from neurospatial.decoding import BayesianDecoder, median_decoding_error
from neurospatial.encoding import (
    border_score,
    compute_directional_rates,
    compute_spatial_rates,
    grid_score,
    spatial_autocorrelation,
)


def _env(rec, cache):
    if "env" not in cache:
        cache["env"] = Environment.from_samples(
            rec.positions, bin_size=rec.params["bin_size"], units=rec.position_units, name=rec.dataset
        )
    return cache["env"]


def _rates(rec, cache):
    if "rates" not in cache:
        cache["rates"] = compute_spatial_rates(
            _env(rec, cache), list(rec.spike_times), rec.times, rec.positions,
            min_speed=rec.params.get("min_speed"),
            epochs=rec.params.get("analysis_epochs"),
        )
    return cache["rates"]


def env_from_samples(rec, cache):
    env = _env(rec, cache)
    return Outcome("ok", {"n_bins": env.n_bins, "n_dims": env.n_dims})


def occupancy(rec, cache):
    occ = _env(rec, cache).occupancy(rec.times, rec.positions)
    return Outcome("ok", {"total_s": float(np.nansum(occ)), "frac_bins_visited": float(np.mean(occ > 0))})


def rates_spatial(rec, cache):
    rates = _rates(rec, cache)
    return Outcome("ok", {"n_units": len(rec.spike_times), "frac_nan_bins": float(np.isnan(np.asarray(rates.firing_rates)).mean())})


def spatial_information(rec, cache):
    info = np.asarray(_rates(rec, cache).spatial_information)
    return Outcome("ok", {"median_bits_per_spike": float(np.nanmedian(info)), "n_finite": int(np.isfinite(info).sum())})


def place_screen(rec, cache):
    return Outcome("ok", {"n_place_screen": int(np.sum(_rates(rec, cache).classify()))})


def grid_scores(rec, cache):
    env, rates = _env(rec, cache), _rates(rec, cache)
    scores = np.array([grid_score(spatial_autocorrelation(env, rate)) for rate in np.asarray(rates.firing_rates)])
    return Outcome("ok", {"median": float(np.nanmedian(scores)), "max": float(np.nanmax(scores))})


def border_scores(rec, cache):
    env, rates = _env(rec, cache), _rates(rec, cache)
    scores = np.array([border_score(env, rate) for rate in np.asarray(rates.firing_rates)])
    return Outcome("ok", {"median": float(np.nanmedian(scores)), "max": float(np.nanmax(scores))})


def hd_tuning(rec, cache):
    tuning = compute_directional_rates(list(rec.spike_times), rec.times, rec.headings)
    cache["hd_tuning"] = tuning
    return Outcome("ok", {"n_units": len(rec.spike_times)})


def decode_position(rec, cache):
    train, test = rec.params.get("train_epochs"), rec.params.get("test_epochs")
    if train is None or test is None:
        return Outcome("n/a", note="params has no train_epochs/test_epochs (held-out decoding only)")
    env = _env(rec, cache)
    decoder = BayesianDecoder(env).fit(list(rec.spike_times), rec.times, rec.positions, epochs=train)
    result = decoder.predict(list(rec.spike_times), rec.times, epochs=test)
    actual = np.column_stack([np.interp(result.times, rec.times, column) for column in rec.positions.T])
    error = median_decoding_error(env, result.map_position, actual)
    return Outcome("ok", {"median_error": float(error), "units": rec.position_units, "n_bins": len(result.times)})
```

Notes for the executor:

- Check the exact shape of `result.times` (bin centres vs edges) in
  `src/neurospatial/decoding/_result.py:337` before interpolating; if it is
  `(n, 2)` edges, interpolate at the centres. `np.interp` across NaN gaps is a
  harness choice: mask bins whose nearest tracked sample is NaN instead of
  interpolating across them.
- `classify()` with default thresholds is a threshold screen, not a verdict
  (CLAUDE.md "Scientific conventions"); the table calls it `cells.place_screen`.
- An operation returns `workaround` only by its own decision (with a gap id);
  exceptions are left to the driver.

Phase-1 registry, in order:

| op_id | function | requires |
| --- | --- | --- |
| `env.from_samples` | `env_from_samples` | `positions` |
| `env.occupancy` | `occupancy` | `positions` |
| `rates.spatial` | `rates_spatial` | `positions`, `spike_times` |
| `rates.spatial_information` | `spatial_information` | `positions`, `spike_times` |
| `cells.place_screen` | `place_screen` | `positions`, `spike_times` |
| `cells.grid_score` | `grid_scores` | `positions`, `spike_times` |
| `cells.border_score` | `border_scores` | `positions`, `spike_times` |
| `hd.tuning` | `hd_tuning` | `headings`, `spike_times` |
| `decode.position` | `decode_position` | `positions`, `spike_times` |
| `io.nwb_roundtrip` | `nwb_roundtrip` | `positions`, `spike_times` |

`nwb_roundtrip`: build a minimal `pynwb.NWBFile(session_description=rec.session,
identifier=rec.dataset, session_start_time=datetime(2000, 1, 1, tzinfo=UTC))`,
call `neurospatial.io.nwb.write_environment(nwbfile, env)` and
`write_spatial_rates(nwbfile, rates)`, write with `pynwb.NWBHDF5IO` to a
`tempfile.TemporaryDirectory`, read back with `read_environment` and
`read_spatial_rates`, and return `ok` only if `np.allclose(..., equal_nan=True)`
holds for firing rates and occupancy and the bin centres match.

## Driver

`datasets/run.py`:

```python
"""Run the operation registry on datasets and write datasets/results/."""

LOADERS = {
    "sargolini_000582": "load_sargolini_000582",
    "peyrache_mouse32": "load_peyrache_mouse32",
}
RESULTS = Path(__file__).resolve().parent / "results"


def run_dataset(key: str, op_ids: Sequence[str] | None) -> dict[str, Any]:
    module = importlib.import_module(LOADERS[key])
    rec, load_error, load_warnings = captured(module.load)
    if load_error is not None:
        raise load_error  # a loader that fails is a harness bug, not a table cell
    validate_recording(rec)
    record = {
        "dataset": key,
        "session": rec.session,
        "source": rec.source,
        "conversions": list(rec.conversions),
        "expectations": check_expectations(rec, module.EXPECTATIONS),
        "versions": _versions(),
        "load_warnings": load_warnings,  # e.g. from neurospatial.io.nwb readers
        "operations": {},
    }
    cache: dict[str, Any] = {}
    for op in OPERATIONS:
        if op_ids is not None and op.op_id not in op_ids:
            continue
        if missing := [name for name in op.requires if getattr(rec, name) is None]:
            record["operations"][op.op_id] = {"status": "n/a", "note": f"needs {', '.join(missing)}"}
            continue
        tracemalloc.start()
        start = time.perf_counter()
        outcome, exc, caught = captured(op.run, rec, cache)
        if exc is not None:
            outcome = Outcome("error", note=f"{type(exc).__name__}: {exc}")
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["operations"][op.op_id] = {
            **asdict(outcome), "warnings": caught, "seconds": round(seconds, 3), "peak_mb": round(peak / 1e6, 1)
        }
    record["hazards"] = probe_hazards(module, rec, record["operations"])
    (RESULTS / f"{key}.json").write_text(json.dumps(record, indent=2, default=_jsonable))
    return record
```

- `_versions()` records `neurospatial.__version__`, numpy, scipy, pynwb,
  pynapple and the current git commit (`git rev-parse HEAD`).
- `_jsonable` converts numpy scalars/arrays to Python values.
- After running, regenerate `results/matrix.md` from every `results/*.json`:
  one row per dataset, one column per op_id, cell text `ok`, `wk G-###`,
  `miss G-###`, `err`, or `–` (n/a), followed by `⚠N` when the operation
  emitted N distinct warning templates, plus a column `expectations` showing
  `passed/total`. A second table lists `seconds` and `peak_mb` per cell.
  Also regenerate `results/warnings.md` (see [Warning calibration](#warning-calibration)).
- CLI: `python datasets/run.py [--datasets KEY ...] [--ops OP_ID ...]`;
  default is every dataset and every op.
- `tracemalloc` sees NumPy allocations but not JAX device memory; the matrix
  footer says so.
- Run with `MPLBACKEND=Agg` (see the `nwb-docs-test-plt-show-hang` memory).

## Warning calibration

Checks warnings in both directions. **Too aggressive:** every warning emitted
while loading or running operations on correctly loaded data is a candidate
false alarm. **Not aggressive enough:** each recording is re-run with one known
defect re-introduced (a *hazard*); an operation that accepts it without a
warning or error is a candidate silent hazard. Both live in
`datasets/_calibration.py`.

```python
import re
import warnings
from collections import Counter
from dataclasses import replace

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def template(message: str) -> str:
    """Group warnings that differ only in the numbers they report."""
    return _NUMBER.sub("#", message)


def captured(fn, *args):
    """Call fn(*args); return (value, exception or None, distinct warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = fn(*args), None
        except Exception as exc:  # the driver records every failure
            value, error = None, exc
    counts = Counter((w.category.__name__, template(str(w.message))) for w in caught)
    example = {}
    for w in caught:
        example.setdefault((w.category.__name__, template(str(w.message))), str(w.message))
    rows = [
        {"category": category, "template": text, "example": example[category, text], "count": n}
        for (category, text), n in counts.items()
    ]
    return value, error, rows


# Applied to every recording they fit: (applies?, make hazardous recording).
GENERIC_HAZARDS = {
    "headings_in_degrees": (
        lambda r: r.headings is not None,
        lambda r: replace(r, headings=np.degrees(r.headings)),
    ),
    "missing_positions_as_zero": (
        lambda r: r.positions is not None and bool(np.isnan(r.positions).any()),
        lambda r: replace(r, positions=np.nan_to_num(r.positions, nan=0.0)),
    ),
    "spike_times_in_ms": (
        lambda r: r.spike_times is not None,
        lambda r: replace(r, spike_times=tuple(t * 1e3 for t in r.spike_times)),
    ),
    # Off by 100: metres labelled cm, or cm labelled metres. params["bin_size"]
    # stays in the labelled unit, as a user who missed the error would leave it.
    "length_unit_off_by_100": (
        lambda r: r.positions is not None and r.position_units in ("cm", "m"),
        lambda r: replace(r, positions=r.positions * (0.01 if r.position_units == "cm" else 100.0)),
    ),
}
PROBE_OPS = ("env.from_samples", "env.occupancy", "rates.spatial", "hd.tuning")


def probe_hazards(module, rec, clean_operations):
    hazards = {name: make for name, (applies, make) in GENERIC_HAZARDS.items() if applies(rec)}
    hazards.update(getattr(module, "HAZARDS", {}))
    rows = []
    for name, make in hazards.items():
        bad, cache = make(rec), {}  # no validate_recording: the hazard is the point
        for op in OPERATIONS:
            if op.op_id not in PROBE_OPS or any(getattr(bad, f) is None for f in op.requires):
                continue
            outcome, error, caught = captured(op.run, bad, cache)
            rows.append({
                "hazard": name,
                "op": op.op_id,
                "response": "raised" if error else "warned" if caught else "silent",
                "message": f"{type(error).__name__}: {error}" if error else [c["example"] for c in caught],
                "clean_summary": clean_operations.get(op.op_id, {}).get("summary"),
                "hazard_summary": None if outcome is None else outcome.summary,
            })
    return rows
```

`results/warnings.md` (regenerated from `results/*.json`) has two tables:

1. **Warnings on clean data** — one row per (category, template): example
   message, the datasets and operations (or `load`) where it fired, total
   count, the largest count from a single operation call, and the verdict.
2. **Hazard probes** — one row per (dataset, hazard, op): response, message,
   and the clean vs hazardous summary side by side (a silent response with a
   changed summary means a quietly different result), and the verdict.

Verdicts are hand-written in `datasets/warning_verdicts.json`, keyed by
template (table 1) or `"<dataset>/<hazard>/<op>"` (table 2), each
`{"verdict": ..., "gap": "G-###" | null, "reason": ...}`. The renderer joins
them in and prints `unreviewed` for keys without one, so verdicts survive
regeneration. Verdict vocabulary:

| Table | Verdict | Meaning | Gap kind |
| --- | --- | --- | --- |
| 1 | `warranted` | the data really has the condition, and the message says what failed, why, and how to fix it | none |
| 1 | `false alarm` | the condition is harmless for this data; `reason` says why | `false-alarm` |
| 1 | `unactionable` | true, but the message does not say what to do | `error-message` |
| 1 | `noisy` | warranted, but repeated within one call (count > 1 for a single fact) | `friction` |
| 2 | `caught` | raised or warned, with an actionable message | none |
| 2 | `caught, poor message` | raised or warned, but the message would not lead a user to the cause | `error-message` |
| 2 | `silent` | accepted without comment | `silent-hazard` |
| 2 | `undetectable` | silent, but no rule on values could catch it (`reason` says why, e.g. both value ranges are plausible) | `docs` if the docs should warn instead |
| 2 | `caught upstream` | silent here, but an earlier operation in the same workflow already warned or raised for this hazard | none |
| 1 | `third-party` | emitted by another package (origin `third-party`), not neurospatial | none |

Judge a `false alarm` against the data, not the threshold: open the samples
that triggered it before deciding. A threshold change is proposed only in
phase 4, with the counts from every dataset.

## Reference checks

`datasets/reference_checks.py`, writing `results/reference_checks.md`. Each
check fixes the conventions of both tools, reports numbers side by side, and
lists every remaining convention difference. Agreement thresholds below are
investigation triggers, not pass/fail tests.

1. **Spatial information vs pynapple (DANDI 000582, `sub-10884_ses-03080402`).**
   Reproduce the pynapple null-distribution tutorial setup exactly: read its
   source at the pinned pynapple version
   (`doc/examples/tutorial_null_distributions.md` in pynapple-org/pynapple),
   which uses `units`, `SpatialSeriesLED1`, 40×40 bins over [−50, 50] cm.
   neurospatial side:
   `Environment.from_samples(positions, bin_size=2.5, dimension_ranges=[(-50, 50), (-50, 50)], infer_active_bins=False, units="cm")`
   and `compute_spatial_rates(..., method="binned")`, then
   `rates.spatial_information` (bits/spike) and
   `neurospatial.encoding.information_per_second(firing_rate, occupancy)` (bits/s).
   Compare per unit against (a) pynapple run in-process on the same arrays and
   (b) the tutorial's printed table (unit 0: 2.797138 bits/s, 1.389261
   bits/spike; the other five values are in
   `docs/research/research_notes/Open spatial neuroscience datasets/tool_tutorial_datasets.md`).
   With conventions aligned, expect relative difference < 1e-6; otherwise list
   each difference (bin edges, empty-bin handling, which LED, units, epoch) and
   how far it moves the number.
2. **Grid score vs opexebo (same session, all units).** Feed the *same* binned
   rate map to `neurospatial.encoding.grid_score(spatial_autocorrelation(env, rate))`
   and to opexebo's autocorrelogram + grid-score functions (read their
   signatures in the installed opexebo source; do not guess). Report per-unit
   pairs, Pearson r, and max |diff|; investigate any |diff| > 0.1 and record
   the cause (annulus choice, NaN handling, rotation set).
3. **Head-direction decoding vs pynapple (Mouse32-140822).** Reproduce the
   pynapple HD tutorial (`doc/examples/tutorial_HD_dataset.md`): ADn units,
   wake epoch, 61 bins over [0, 2π), `decode_bayes` with 100 ms bins. Check
   the structural numbers it prints (49 units, 31 ADn, 19,590 decoded samples).
   Compute both tools' median absolute circular error against the tracked
   heading. If neurospatial has no public path to decode heading, record
   `decode.heading` as `missing` with a gap entry (check
   `docs/user-guide/` for a documented heading-decoding recipe first) and
   still report pynapple's error as the target.

## Loader recipe

Every loader follows these steps; the phase files list per-dataset specifics.

1. Pin files and checksums (see [Fetch and cache](#fetch-and-cache)).
2. Open with the most direct reader: `pynwb.NWBHDF5IO(path, "r", load_namespaces=True)` for NWB;
   `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` for MATLAB v5;
   `h5py` for v7.3; `matio.load_from_mat` for MATLAB `table` objects;
   `np.load(..., allow_pickle=False)` for npz.
3. For NWB, first try neurospatial's readers (`neurospatial.io.nwb.read_position`,
   `read_head_direction`, `read_units`, `read_intervals`, `read_pose`). If one fails
   or returns something wrong, keep the failure as a gap entry and read the
   data with pynwb/h5py directly. Before declaring anything absent, walk the
   whole file by `neurodata_type` (`h5py.File.visititems`), not only standard paths.
4. Check every stored `unit`/`conversion`/`rate` against the value ranges; when
   they disagree, trust the ranges, record the conversion, and log a `data-quirk` gap.
5. Return the `Recording`; `if __name__ == "__main__":` calls `load()`,
   `validate_recording`, `check_expectations`, and prints a short summary
   (n units, duration, sampling rate, position extent, epochs, conversions).
