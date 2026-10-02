# Phase 6c — Plain data holders; `Session` deleted

**Requires:** Phase 6b.

[← back to PLAN.md](PLAN.md) · [executing](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Read [executing.md](executing.md) first: branch and PR workflow, definition of done, CHANGELOG-per-commit, and what to do when the plan and reality disagree. This file holds only what is specific to Phase 6c.

This phase (decision 4: raw arrays are the only input form; loaders and simulators return plain data holders whose attributes are passed explicitly):

- makes simulator output a frozen holder whose attribute names are the analysis parameter names;
- makes the NWB readers return frozen, non-iterable holders, and keeps the units' acquisition windows;
- deletes `main`'s second data bundle (`Session`, `load_session`, `recording.py`) and the `PositionLike` machinery that only it still uses;
- gives `validate_simulation` and `plot_session_summary` one holder-taking form each.

Regenerate the API snapshot (Phase 6a) at the end and put its diff in the PR.

**Inputs to read first** (verified on `main` at `da631a47`; earlier phases shift line numbers):

- [src/neurospatial/recording.py](../../../../src/neurospatial/recording.py) — `Position` (:103), `Session` (:148, a bundle no analysis function accepts), `Session.from_nwb` (:382; reads units and position at :454-456), `load_session` (:589). It imports `PositionLike`/`as_times_positions` (:56-57).
- [src/neurospatial/_typing.py](../../../../src/neurospatial/_typing.py) — `PositionLike` (:58), `_is_position_like` (:210), `as_times_positions` (:221). After Phase 6b their only users are `recording.py` and `io/pynapple.py::from_pynapple` (:100-103).
- [src/neurospatial/simulation/session.py:17](../../../../src/neurospatial/simulation/session.py) — `SimulationSession` fields `env, positions, times, spike_trains, models, ground_truth, metadata` (:121-127); `ground_truth` keyed `"cell_{i}"` (:93 docstring, :531, `examples.py:738, 763, 992`); there is no `unit_ids`.
- [src/neurospatial/simulation/validation.py](../../../../src/neurospatial/simulation/validation.py) — `validate_simulation` (:18) has a dual form on `main`: `validate_simulation(session=None, *, env=None, spike_trains=None, positions=None, times=None, ground_truth=None, cell_indices=None, method=…, max_center_error=None, min_correlation=None, show_plots=False, **kwargs)`, branching at :176 and building `f"cell_{cell_idx}"` keys at :257. `plot_session_summary(session, cell_ids=None, figsize=(15, 10))` (:446; selection at :566-587).
- [src/neurospatial/io/nwb/_behavior.py:34](../../../../src/neurospatial/io/nwb/_behavior.py) — `read_position` returns `(positions, timestamps)` (data before time), `read_head_direction` (:301) returns `(angles, timestamps)`; [io/nwb/_units.py:70](../../../../src/neurospatial/io/nwb/_units.py) `read_units` returns `(spike_trains, unit_ids)`. Internal callers: `io/nwb/_environment.py:1101`, `io/nwb/_overlays.py:73, :221`, `recording.py:454-456`. Docstring unpackings: `_behavior.py:102, :108, :345`, `_core.py:55`, `io/nwb/__init__.py:45`.
- [.github/workflows/test_nwb.yml:40](../../../../.github/workflows/test_nwb.yml) — runs `uv run pytest tests/nwb tests/test_recording.py -n 0 -q`.
- **Files earlier phases already changed** (search by symbol):
  - `io/nwb/_behavior.py`, `_pose.py`, `_environment.py`: Phase 1 (NWB environment geometry) and Phase 2a Tasks 3 and 4 (`data_from_series` conversion/offset, unit normalization, the `"cm"` warning). The holder change must keep `data_from_series` on every read path.
  - `simulation/models/place_cells.py`, `simulation/validation.py`: Phase 2a Task 5 (default width, `max_center_error`). Phase 6b renamed `validate_simulation(spike_trains=)` to `spike_times=`; this phase removes that keyword form.
  - `simulation/models/object_vector_cells.py`: Phase 5a (`direction_frame`).
  - `_intervals.py`: Phase 3a's `as_intervals` and `intersect_intervals`.
  - The root `__init__.py`: Phase 6a kept `Session` and `load_session` in `_LAZY_ATTRS`, its `TYPE_CHECKING` block and `__all__` until this phase.

**Contracts referenced:**

- [Overview decision 4](overview.md#settled-design-decisions) — no `Session` bundle, no dual-form overloads. A holder-taking function is not a dual form: the holder *is* the data, and its single signature takes it.
- [Time-window semantics → Defaults](shared-contracts.md#time-window-semantics) — loaders preserve acquisition windows (`spike_window`).
- [Error-message contract](shared-contracts.md#error-message-contract) — rule 1: no keyword slot that is required in one call form and forbidden in another.

## Tasks

**6c.1 `SimulationSession` becomes a plain holder.** It keeps its name and becomes:

```python
@dataclass(frozen=True)
class SimulationSession:
    """Simulated recording; pass its attributes straight into analyses.

    >>> rates = compute_spatial_rates(sim.env, sim.spike_times, sim.times,
    ...                               sim.positions, unit_ids=sim.unit_ids)
    """

    env: Environment
    spike_times: list[NDArray[np.float64]]   # one (n_spikes_i,) array per unit, seconds
    unit_ids: NDArray[np.int64]              # (n_units,), np.arange(n_units)
    times: NDArray[np.float64]               # (n_time,), seconds
    positions: NDArray[np.float64]           # (n_time, n_dims), env units
    models: list[NeuralModel]                # models[i] generated spike_times[i]
    ground_truth: dict[int, dict[str, Any]]  # keyed by unit_id
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n = len(self.spike_times)
        if not (len(self.unit_ids) == len(self.models) == n):
            raise ValueError(
                f"spike_times has {n} units, unit_ids {len(self.unit_ids)}, models "
                f"{len(self.models)}; these must match one-to-one.\n"
                "Fix: build unit_ids as np.arange(len(spike_times)).")
```

- **Callers:** update `simulate_session` (`session.py:531, 545`), its docstring (:93), and every `examples.py` builder (:738-798, :992-1027): `ground_truth` keys `"cell_{i}"` → `int` `unit_id`.

**6c.2 `validate_simulation` and `plot_session_summary` take the holder only.**

- `validate_simulation(session, *, unit_ids=None, method="diffusion_kde", max_center_error=None, min_correlation=None, show_plots=False, **kwargs)`. `session` is an ordinary required positional argument. The raw keyword form (`env`, `spike_times`, `positions`, `times`, `ground_truth`) is deleted: it needed the same data the holder carries plus `ground_truth`, which only a simulator produces, and its `session=None` slot made every argument conditionally required. `**kwargs` keeps forwarding to `compute_spatial_rate` (`validation.py:247`), as now.
- `plot_session_summary(session, *, unit_ids=None, figsize=(15, 10))`.
- `unit_ids=` (replacing `cell_indices` / `cell_ids`) selects units **by label** from `session.unit_ids`; `None` means all (validation) or the first six (plot, as now). An unknown label raises `ValueError` listing the unknown labels and the valid range, with a `Fix:` line.
- Internally, read `session.ground_truth[unit_id]` (no more `f"cell_{i}"` at `validation.py:257`).
- Tests: `tests/simulation/test_validation_sim.py` uses the holder form everywhere except the raw-form test at :195-220, which is deleted; `cell_indices=[0, 2, 4]` (:236) becomes `unit_ids=[0, 2, 4]`.

**6c.3 Delete `Session`, `load_session` and `recording.py`.**

- Delete `src/neurospatial/recording.py` (`Position`, `Session`, `load_session`) and `tests/test_recording.py`. No analysis function accepts `Session`, so it is a second bundle in a design with none.
- **CI:** `.github/workflows/test_nwb.yml:40` runs `uv run pytest tests/nwb tests/test_recording.py -n 0 -q`. Remove `tests/test_recording.py` from that command in the same commit; otherwise the NWB job fails on a missing path.
- Root `__init__.py`: remove `Session` and `load_session` from `__all__`, `_LAZY_ATTRS` and the `TYPE_CHECKING` block (Phase 6a), and from `tests/test_lazy_imports.py` and `tests/typing/top_level_lazy_exports.py` if Phase 6a listed them there.
- Phase 4b's `tests/docs/test_docstring_sections.py` has `_EXEMPT = frozenset({"neurospatial.load_session"})` and asserts every exempt name still resolves. Delete that entry in the same commit.
- **`PositionLike` machinery, now unused** (Phase 6b removed it from every analysis signature): delete `_typing.PositionLike`, `_is_position_like` and `as_times_positions`, and their entries in `_typing.__all__`. `io/pynapple.py::from_pynapple` keeps returning `(times, positions)` for a `Tsd`/`TsdFrame`; inline its coercion there (`np.asarray(obj.t, dtype=np.float64)` and `.values`, falling back to `.d`, as `as_times_positions` did). Delete the `as_times_positions` tests in `tests/test_typing.py` (:92-131) and keep the rest of that file.
- `grep -rn "recording\|load_session\|PositionLike\|as_times_positions" src tests docs examples README.md CLAUDE.md .claude/` must return nothing outside `docs/plans/`, `docs/reviews/` and `CHANGELOG.md`.

**6c.4 NWB readers return frozen, non-iterable holders.** An old `positions, t = read_position(...)` unpack then fails loudly with `TypeError` instead of swapping:

- `read_position → NWBPosition(times, positions)`;
- `read_head_direction → NWBHeadDirection(times, headings)`;
- `read_units → NWBUnits(spike_times, unit_ids, obs_intervals, spike_window)`.

Define them as `@dataclass(frozen=True)` in `io/nwb/_holders.py` (no `__iter__`, so tuple unpacking raises `TypeError`), export them from `io.nwb`, and update the internal callers and docstring unpackings listed in Inputs. `read_position(..., lazy=True)` keeps its lazy arrays inside the holder. Update every test, doc and example that unpacks a reader: `tests/nwb/test_behavior.py`, `test_units.py`, `test_environment.py`, `test_overlays.py`, `test_fields_roundtrip.py`, `docs/user-guide/interoperability.md`, `examples/27_loading_from_nwb.py`, `.claude/QUICKSTART.md`, `.claude/ADVANCED.md`, `.claude/TROUBLESHOOTING.md`.

**6c.5 Units keep their acquisition windows** ([time-window semantics, Defaults](shared-contracts.md#time-window-semantics): loaders preserve acquisition windows). On `main`, `read_units` (`io/nwb/_units.py:70-186`) reads only `spike_times` and ignores the NWB `obs_intervals` column. Add both fields:

- `obs_intervals: list[NDArray[np.float64]] | None` holds one `(n_i, 2)` array per read unit (`np.asarray(units[row, "obs_intervals"])`), or `None` when `"obs_intervals" not in units.colnames`.
- `spike_window: NDArray[np.float64] | None` is the intersection of the read units' `obs_intervals`, folded with `_intervals.intersect_intervals` after `as_intervals`, or `None` when the column is absent. It is the time *every* read unit was observed, so a shared occupancy never includes time when a unit could not fire.
- The `read_units` docstring says to read units with different coverage in separate calls (`unit_ids=`) when the intersection is too short. An empty intersection is stored as shape `(0, 2)`; passing it to an analysis raises `as_intervals`'s "no rows" contract error.
- A probe with pynwb shows the column round-trips as `units[i, "obs_intervals"]`: unit 7 `[[0, 100], [1100, 1200]]` and unit 11 `[[10, 1150]]` give the intersection `[[10, 100], [1100, 1150]]`.

**The NWB recipe** that replaces `Session.from_nwb`:

```python
units = read_units(f)
pos = read_position(f)
rates = compute_spatial_rates(
    read_environment(f), units.spike_times, pos.times, pos.positions,
    unit_ids=units.unit_ids, spike_window=units.spike_window,
)
```

The result's `spike_window_assumed` is then `False` whenever the file recorded `obs_intervals`.

**6c.6 Regenerate the API snapshot.** `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py`. Expected diff: root −2 (`Session`, `load_session`), `io.nwb` +3 (`NWBPosition`, `NWBHeadDirection`, `NWBUnits`), changed lines for `validate_simulation`, `plot_session_summary` and the three readers; 420 lines in all. These are expectations; the diff is the truth. Then run Phase 4b's executable docs, the full suite, the slow tests (`-m "slow and not napari"`) and the NWB job locally (`uv run pytest tests/nwb -n 0 -q` after `uv sync --all-extras`, plus Phase 4b's `tests/docs/test_flagship_docstrings.py -m nwb`).

**6c.7 Documentation** (part of this PR; each commit adds its own CHANGELOG bullet per [executing.md](executing.md), and this task checks they are all present):

- `README.md`: the project-tree line `recording.py    # Session bundle and NWB session loader` at :612.
- `.claude/QUICKSTART.md` (:1012-1019 `read_position` unpacking), `.claude/ADVANCED.md`, `.claude/TROUBLESHOOTING.md`, `.claude/API_REFERENCE.md` (`Session`, `load_session`, the holders).
- `docs/user-guide/interoperability.md`: the `Session` sections (:9, :32-104 and onward) become the explicit-attribute recipes; the NWB recipe passes `spike_window=units.spike_window`.
- `docs/api/index.md` (`Session`, `load_session`).
- `examples/15_simulation_workflows.py` (`session.spike_trains`, `ground_truth['cell_0']` at :121-127) and `examples/27_loading_from_nwb.py`. Sync them with `uv run jupytext --sync` and `uv run python docs/sync_notebooks.py`.
- `CHANGELOG.md` `[Unreleased]`: `### Removed` `Session`, `load_session`, `neurospatial.recording`, `validate_simulation`'s keyword form; `### Changed` (breaking) `SimulationSession.spike_times`/`unit_ids` and integer `ground_truth` keys, the reader holders, `validate_simulation(unit_ids=)`/`plot_session_summary(unit_ids=)`; `### Added` `NWBUnits.obs_intervals`/`spike_window` read from the NWB `obs_intervals` column.

## Deliberately not in this phase

- **Headings or HD cells in `simulate_session`.** The holder gains no `headings` until a simulator produces them.
- **The whole-session NWB round trip** (decision 7). The recipe above reads; nothing writes a session.
- **Reloaded non-grid NWB environments lacking finite-volume geometry** (overview Non-Goals).
- **Changing the `NeuralModel.firing_rate(positions, times=None, headings=None)` protocol.** It is the simulator's internal model interface.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/simulation/test_session_holder.py::test_attributes_feed_analysis` | `sim = open_field_session(duration=60, seed=0)`; `compute_spatial_rates(sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids).unit_ids` equals `np.arange(len(sim.spike_times))`; `set(sim.ground_truth) == set(sim.unit_ids)` |
| `…::test_mismatched_lengths_raise` | `unit_ids` of wrong length → `ValueError` with `Fix:` |
| `…::test_holder_is_frozen` | assigning `sim.times = …` raises `FrozenInstanceError` |
| `tests/simulation/test_validation_sim.py::test_select_by_label` | `validate_simulation(sim, unit_ids=[0, 2, 4])` validates exactly those three; `unit_ids=[99]` raises `ValueError` naming `99` with `Fix:`; `validate_simulation(env=…)` raises `TypeError` (no such keyword) |
| `tests/simulation/test_validation_sim.py::test_plot_session_summary_labels` | `plot_session_summary(sim, unit_ids=[1, 3])` draws two rate panels |
| `tests/nwb/test_reader_holders.py::test_read_position_holder` (pynwb extra) | `pos = read_position(f)`; `pos.times.ndim == 1`, `pos.positions.shape == (n, 2)`; `a, b = read_position(f)` raises `TypeError`. Conversion and offset are still applied (`pos.positions` equals `data * conversion + offset`) |
| `tests/nwb/test_reader_holders.py::test_read_head_direction_holder` (pynwb extra) | `hd.times`, `hd.headings` shapes; unpacking raises `TypeError` |
| `tests/nwb/test_reader_holders.py::test_read_units_spike_window` (pynwb extra) | Units 7 (`obs_intervals=[[0, 100], [1100, 1200]]`) and 11 (`[[10, 1150]]`): `units.obs_intervals[0]` equals the first, and `units.spike_window` equals `[[10, 100], [1100, 1150]]`. A file without the column gives `obs_intervals is None` and `spike_window is None`. Passing `spike_window=units.spike_window` to `compute_spatial_rates` gives `spike_window_assumed is False` |
| `tests/test_typing.py` (trimmed) | passes without the `as_times_positions` tests |
| `tests/io_tests/test_pynapple.py` (pynapple job) | `from_pynapple(tsdframe)` still returns `(times, positions)` with the inlined coercion |
| `tests/test_public_api_snapshot.py` | passes with the regenerated file |
| `.github/workflows/test_nwb.yml` | the command no longer names `tests/test_recording.py`; the job passes on the PR (`gh pr checks`) |
| Phase 4b executable-docs test and `NWB_FLAGSHIP` doctests | pass with every doc updated in 6c.7 |

## Fixtures

- **Simulation:** `open_field_session(duration=60, seed=0)`.
- **NWB:** reuse the existing `tests/nwb` pynwb fixtures; add one file with an `obs_intervals` units column for 6c.5.

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- Every task is implemented as specified; `validate_simulation` and `plot_session_summary` each have one signature with no conditionally required keyword.
- The "Deliberately not in this phase" list is honored.
- Validation slice tests pass; NWB tests ran with the extra installed (check the `-rs` skip summary).
- Tests aren't trivial: they exercise the asserted behavior, not tautologies (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are actually removed (`recording.py`, `PositionLike`, `as_times_positions`, `"cell_{i}"` keys, the `test_recording.py` workflow reference).
- User-facing documentation listed as tasks is updated, not deferred.
