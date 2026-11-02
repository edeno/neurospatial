# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Recording](#recording) — what every loader returns
- [Loader module](#loader-module) — what every `datasets/load_<key>.py` exposes
- [Operation and Outcome](#operation-and-outcome) — what every operation returns
- [Gap-log entry](#gap-log-entry) — format of `datasets/GAPS.md` entries
- [Folder layout](#folder-layout)

## Recording

`datasets/_record.py`. A frozen dataclass of plain arrays in the units and
conventions neurospatial expects. It is harness-internal; it is not a proposal
for a neurospatial `Session`.

```python
@dataclass(frozen=True)
class Recording:
    dataset: str                      # registry key, e.g. "sargolini_000582"
    session: str                      # file/session identifier within the dataset
    source: str                       # DOI or URL of the exact file(s) loaded
    license: str                      # e.g. "CC-BY-4.0"; "unstated" if none
    times: NDArray[np.float64] | None = None        # (n_time,) s, strictly increasing
    positions: NDArray[np.float64] | None = None    # (n_time, n_dims)
    position_units: str | None = None               # "cm" | "m" | "pixel" | "arbitrary"
    headings: NDArray[np.float64] | None = None     # (n_time,) rad; allocentric 0 = East, +π/2 = North
    spike_times: tuple[NDArray[np.float64], ...] | None = None  # per unit, s, sorted
    unit_ids: NDArray[Any] | None = None            # (n_units,), unique
    activity: NDArray[np.float64] | None = None     # (n_frames, n_cells) imaging
    activity_times: NDArray[np.float64] | None = None  # (n_frames,) s
    activity_kind: str | None = None                # "deconvolved" | "dff" | "events" | "binned_rate"
    epochs: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)  # name -> (n, 2) [start, stop) s
    events: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)  # name -> (n,) or (n, 2) s
    geometry: Mapping[str, Any] = field(default_factory=dict)  # arena facts from the source: shape, size, bounds, track graph
    params: Mapping[str, Any] = field(default_factory=dict)    # analysis choices: bin_size, dt, min_speed, train/test epochs
    conversions: tuple[str, ...] = ()                # every transformation applied, one line each
```

Invariants (checked by `validate_recording`, [designs.md#record-validation](designs.md#record-validation)):

- `times` is set exactly when `positions` or `headings` is (a head-direction recording can have headings without positions); `positions.shape[0] == times.shape[0]`.
- `headings`, when set, has the same length as `times` and lies in `[-π, π]` (wrap with `neurospatial`'s convention, not modulo 2π in place).
- Missing samples are NaN, never 0 or a sentinel. Replacing a sentinel (e.g. VR −500) with NaN is a conversion and gets a gap entry.
- `spike_times[i]` is sorted and finite; `len(spike_times) == len(unit_ids)`.
- `activity` and `activity_times` are both set or both `None`.
- `geometry` holds only facts the source states (paper, readme, file metadata); inferred values are prefixed `inferred_`.
- `params` holds the analysis choices the driver uses, chosen by the loader author and justified in a comment next to each value. Required keys when the data supports them: `bin_size` (in `position_units`), `dt` (decoding bin, s). Optional: `min_speed`, `analysis_epochs` ((n, 2) s; restricts rate maps, e.g. to running or flight), `train_epochs`, `test_epochs` ((n, 2) s; held-out decoding), `replay_dt` (s; event decoding bin), `nwb_path` (local cached path of the NWB file the recording came from; lets the `io.nwb_readers` operation test neurospatial's readers against the loader's range-checked arrays).
- `conversions` lists every unit change, axis reorder, sentinel replacement, clock alignment or column selection, e.g. `"positions: cm from stored meters (conversion=0.01)"`.

Do not weaken: no field may hold a quantity in a different unit than documented above; a loader that cannot produce a field leaves it `None` (the operations that need it become `n/a`).

## Loader module

Each dataset is one file `datasets/load_<key>.py` exposing exactly:

```python
DATASET: dict[str, str]   # {"key", "title", "source", "license", "download_size"}
EXPECTATIONS: dict[str, Any]  # pre-registered checks, committed before the first run
HAZARDS: dict[str, Callable[[Recording], Recording]]  # optional; see below

def load(cache: Path | None = None) -> Recording: ...

if __name__ == "__main__":  # standalone: fetch, load, validate, print a summary
    ...
```

- `load` fetches through `datasets/_fetch.py` only, with a pinned SHA-256 for every file.
- `EXPECTATIONS` keys are free-form but each is checked by `check_expectations` ([designs.md#expectations](designs.md#expectations)); supported forms: exact value, `(low, high)` range, or a callable predicate on the `Recording`.
- `EXPECTATIONS` lands in its own commit before the commit that first runs the driver on that dataset, so history shows it was written first.
- The module docstring states: what was recorded, where the data lives, what the loader converts, and every analysis choice in `params` with its justification.
- `HAZARDS` (optional) maps a short name to a function that undoes one correction the loader made for a quirk this dataset really has, returning the recording as a user who missed the quirk would pass it (e.g. the stored "mm" values left unconverted, a −500 sentinel left in place of NaN). Each entry names the `conversions` line it undoes in a comment. The harness also applies [generic hazards](designs.md#warning-calibration) to every recording, so a loader lists only dataset-specific ones. Hazardous recordings bypass `validate_recording`: the point is to see what neurospatial does with them.

## Operation and Outcome

`datasets/operations.py`.

```python
Status = Literal["ok", "workaround", "missing", "error", "n/a"]

@dataclass(frozen=True)
class Outcome:
    status: Status
    summary: Mapping[str, float | int | str] = field(default_factory=dict)  # small scalars for the table
    note: str = ""
    gap: str | None = None       # "G-###"; required when status is "workaround" or "missing"

@dataclass(frozen=True)
class Operation:
    op_id: str                   # stable column name, e.g. "rates.spatial"
    requires: tuple[str, ...]    # Recording fields that must be non-None, else "n/a"
    run: Callable[[Recording, dict[str, Any]], Outcome]  # second arg: per-recording cache (env, rates, ...)
```

- `ok`: neurospatial did it through the public API with no harness-side adaptation.
- `workaround`: it worked only after harness code adapted inputs or outputs; `note` says what, `gap` points to the entry.
- `missing`: the operation knowingly cannot be expressed with the public API (the op decides this explicitly, it is not inferred from an exception).
- `error`: an exception escaped; the driver fills `note` with `"<Type>: <message>"`. Every `error` cell gets a gap entry when the table is reviewed (the error message's quality is itself a finding).
- `n/a`: a required field is `None`.
- "Wrong" is not an operation status: disagreement with expectations or a reference tool is reported in the expectations and reference-check outputs, then logged as a gap.

## Gap-log entry

`datasets/GAPS.md`, one entry per finding, numbered in order of discovery:

```markdown
## G-007: Head direction labelled degrees but stored in radians is read as degrees
- Datasets: giocomo_001701
- Operation / step: loader (read_head_direction)
- Kind: wrong-result | missing-feature | friction | error-message | docs | performance | data-quirk | false-alarm | silent-hazard
- Evidence: <numbers, file:line in the loader, table cell, reference-check output>
- Harness workaround: <datasets/load_x.py:NN, or "none">
- Proposed change: <one line, or "undecided">
- Status: open | promoted (<PR link>) | won't fix (<reason>)
```

`data-quirk` entries describe the source data, not neurospatial; they matter
when the question is whether neurospatial should detect or refuse the quirk.
`false-alarm` entries are warnings that fired on correctly loaded data;
`silent-hazard` entries are hazards neurospatial accepted without a warning or
error (see [warning calibration](designs.md#warning-calibration)).

## Folder layout

```text
datasets/
  README.md               # how to run, cache location, how to add a dataset
  GAPS.md                 # gap log (hand-written, tracked)
  _fetch.py               # download + verify + cache
  _record.py              # Recording + validate_recording + check_expectations
  _calibration.py            # warning capture, generic hazards, hazard probes
  verdicts.json   # hand-written verdicts joined into results/warnings.md
  operations.py           # operation registry
  reference_checks.py     # cross-tool comparisons
  run.py                  # driver: datasets x operations -> results/
  load_<key>.py           # one per dataset
  results/                # tracked, regenerated: <key>.json, matrix.md, warnings.md, reference_checks.md
  tests/                  # offline tests on simulated recordings
```

`datasets/` is not a Python package (no `__init__.py`), so it cannot shadow the
PyPI `datasets` package; scripts are run by path and import siblings because the
script directory is on `sys.path`. `datasets/tests/conftest.py` inserts the
`datasets/` directory on `sys.path` for the tests.
