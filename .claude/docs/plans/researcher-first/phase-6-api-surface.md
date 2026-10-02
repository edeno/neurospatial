# Phase 6 — A curated, snapshot-guarded API surface with one argument convention

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

This phase:

- shrinks each namespace to user-facing names and deletes `main`'s second data bundle (`Session`);
- makes every signature follow the input conventions, with a times check that names a swap;
- makes simulator and NWB output plain holders whose attribute names are the analysis parameter names;
- replaces the export-pinning tests with the one API snapshot.

It lands after Phase 4: Phase 4's executable-docs test is what catches a curated-away name that a document still uses.

**Inputs to read first** (verified on `main` at `da631a47`; earlier phases shift line numbers):

- [src/neurospatial/__init__.py](../../../../src/neurospatial/__init__.py) — root `__all__` (28 names), the eager `bin_spikes_in_time` import, and `_LAZY_ATTRS` (`SpikeTrains`, `restrict`, `Session`, `load_session`, `BayesianDecoder`).
- [src/neurospatial/recording.py](../../../../src/neurospatial/recording.py) — `Position` (:103), `Session` (:148, a bundle no analysis function accepts), `Session.from_nwb` (:382), `load_session` (:589).
- [src/neurospatial/encoding/spatial.py:2542](../../../../src/neurospatial/encoding/spatial.py) — `compute_spatial_rate(env, spike_times, times: NDArray | PositionLike, positions=None)`, the dual form (adapter at :2874; same in `compute_spatial_rates` :3026, `decode_session(_summary)` (`decoding/session.py:95, :682`), and `BayesianDecoder.fit`/`predict`/`predict_summary`/`score`, where `times: ArrayLike | PositionLike` sits at `decoding/estimator.py:281, :407, :451, :501`). There is no `fit_predict`.
- [src/neurospatial/simulation/session.py:17](../../../../src/neurospatial/simulation/session.py) — `SimulationSession` fields `env, positions, times, spike_trains, models, ground_truth, metadata` (:121-127); `ground_truth` keyed `"cell_{i}"` (:531, `examples.py:738,763,992`); there is no `unit_ids`.
- [src/neurospatial/io/nwb/_behavior.py:34](../../../../src/neurospatial/io/nwb/_behavior.py) — `read_position` returns `(positions, timestamps)` (data before time), `read_head_direction` (:301) returns `(angles, timestamps)`; [io/nwb/_units.py:70](../../../../src/neurospatial/io/nwb/_units.py) `read_units` returns `(spike_trains, unit_ids)`. Internal callers: `io/nwb/_environment.py:1101`, `io/nwb/_overlays.py:73,221`.
- [src/neurospatial/_validation.py](../../../../src/neurospatial/_validation.py) — `validate_finite`, `validate_lengths`; [environment/trajectory.py:338-378](../../../../src/neurospatial/environment/trajectory.py) — `env.occupancy`'s swap-aware times check (the model for 6.4).
- [tests/test_sparse_init_exports.py](../../../../tests/test_sparse_init_exports.py), [tests/test_lazy_imports.py](../../../../tests/test_lazy_imports.py), [tests/test_package_imports.py](../../../../tests/test_package_imports.py) — existing export tests.
- **Files earlier phases already changed** (search by symbol):
  - `decoding/estimator.py`: Phase 1 Task 6 (`_unit_ids_supplied`, `_align_to_fitted_units`) and Phase 3b (`epochs`/`spike_window` replace `epoch`). 6.5 builds on both.
  - `events/alignment.py`: Phase 2 Task 8 (spike-group input) and Phase 3b (event filtering). 6.4 renames the parameter only.
  - `behavior/vte.py`, `navigation.py`, `decisions.py`: Phase 2 Tasks 9–11 and Phase 3c (`max_gap`/`epochs`, per-run kinematics). 6.4 reorders their arguments only.
  - `ops/egocentric.py`: Phase 2 Tasks 6 and 10 and Phase 3c (`heading_from_velocity(positions, times, …)`).
  - `io/nwb/_behavior.py`, `_pose.py`, `_environment.py`: Phase 1 Task 10 and Phase 2 Tasks 3 and 12. The holder change in 6.6 must keep `data_from_series` (conversion/offset) on every read path.
  - `simulation/models/place_cells.py`, `simulation/validation.py`: Phase 2 Task 4 (default width, `max_center_error`). 6.6 changes `validate_simulation`'s keys only.
  - `_exceptions.py` and the root `__init__.py`: Phase 4 added `NeurospatialError` to the root `__all__`; keep it. Phase 4's `FLAGSHIP` list and executable docs name public paths; update them for every name moved here.
  - `encoding/*`: Phase 3a (time-window keywords, `spike_window` on results) and Phase 5 (renames, `compute_object_vector_rate(s)`, `has_place_field`, the four `*_significance` functions, `criterion=` keywords).
- Archive pruning, for input only: `git cat-file -p "$(git rev-parse 'archive/public-api-curation-2026-10-01^{commit}'):src/neurospatial/animation/__init__.py"` (and the other `__init__.py` files). Under zsh, quote the `rev:path` argument, because `$A:s…` is a zsh history modifier.

**Contracts referenced:**

- [API snapshot](shared-contracts.md#api-snapshot) — implemented exactly in 6.7.
- [Input conventions](shared-contracts.md#input-conventions) — 6.4–6.6. Do not weaken the times-before-positions rule or the swap-naming error. 6.5 implements the Population-identity rules that Phase 1 left to this phase: `BayesianDecoder.fit(unit_ids=)` and duplicate rejection wherever `unit_ids` are supplied.
- [Error-message contract](shared-contracts.md#error-message-contract) — the swap error lists every problem at once and ends with `Fix:`.

**Designs referenced:** none (designs inline).

## Tasks

**6.1 Remove `main`'s governance remnants.**

- Delete `docs/plans/public-api-curation/` (`PLAN.md`, `TASKS.md`; the head commit `da631a47` added them) and its row in `docs/plans/README.md`.
- Delete `tests/test_sparse_init_exports.py`. It pins root `__all__` plus a removed-names list, and the snapshot replaces it (decision 7).
- Keep `tests/test_lazy_imports.py` (lazy-loading behaviour) and `tests/test_package_imports.py` (each `__all__` name resolves). Update their name lists to 6.2.
- The local `docs/plans/scientific-data-integrity/` and `tests/public_api/` contain only untracked `__pycache__`. Nothing is tracked, so take no action, and do not commit them.

**6.2 Curate namespaces.** Rules:

1. The root holds the core spatial types, the public exceptions, the domain submodules, and the flagship four-array path and its results.
2. A domain namespace exports user-facing functions and types only. Plumbing, normalizers and internal containers leave `__all__`.
3. One object has one name: aliases are deleted, not kept.
4. No exported name equals a sibling submodule.
5. No deprecation shims (decision 1).

The archive's pruning was reviewed and is **not** followed where it removed user-facing names: it dropped the four `is_*_cell` predicates, `heading_from_velocity`, `PositionOverlay` and every simulation model.

| Namespace | `main` | Target | Change |
| --- | --- | --- | --- |
| root | 28 | 31 | **remove** `BayesianDecoder`, `bin_spikes_in_time` (both stay in `decoding`), `SpikeTrains` (stays in `encoding`), `restrict` (stays in `behavior`), `Session`, `load_session` (deleted, 6.6). **Add** (lazy) `compute_spatial_rate`, `compute_spatial_rates`, `SpatialRateResult`, `SpatialRatesResult`, `decode_position`, `DecodingResult`, `peri_event_histogram`, `PeriEventResult`. **Keep** `NeurospatialError`, which Phase 4 exported (the contract requires it at the root; the 31 counts it) |
| encoding | 64¹ | 69 | **remove** `as_spike_trains`, `as_spike_trains_with_ids` (internal normalizers in `encoding/_spikes.py`); **rename** `phase_precession` → `compute_phase_precession` (the function shadows the `encoding.phase_precession` submodule) |
| decoding | 35 | 34 | **remove** `poisson_likelihood` (its own docstring says it under/overflows and to prefer `log_poisson_likelihood`) |
| behavior | 69 | 68 | **remove** `integrated_absolute_rotation` (alias: `behavior/vte.py:316` `= head_sweep_magnitude`) |
| events | 18 | 16 | **remove** `validate_events_dataframe`, `validate_spatial_columns`; rename them `_validate_…` in `events/_core.py` and update call sites |
| ops | 67 | 65 | **remove** `Affine3D` (alias, `ops/transforms.py:1045`; delete the alias) and `clear_kdtree_cache` (cache plumbing; keep the function in `ops/binning.py`) |
| stats | 28² | 29 | none here. Delete the surrogate re-export "for backward compatibility" at `stats/shuffle.py:68-73` and its docstring note at :27-29 |
| simulation | 23 | 23 | signature and field changes only (6.6) |
| io | 6 | 6 | none |
| io.nwb | 22 | 25 | **add** `NWBPosition`, `NWBHeadDirection`, `NWBUnits` (6.6) |
| animation | 31 | 16 | **remove**: <br>• `PositionData`, `BodypartData`, `HeadDirectionData`, `VideoData`, `EventData`, `TimeSeriesData`, `ObjectVectorData` (docstrings say "internal container … should not be instantiated by users"); <br>• `OverlayProtocol` (its `convert_to_data` must return those internal types, so it is not a usable public extension point); <br>• `VideoReaderProtocol` (only used inside `VideoData`); <br>• `SpikeOverlay` (alias `animation/overlays.py:1472` `= EventOverlay`; delete it); <br>• `VideoCalibration` (canonical in `ops`); <br>• `add_scale_bar_to_axes`, `compute_nice_length`, `configure_napari_scale_bar`, `format_scale_label` (renderer helpers; `ScaleBarConfig` stays) |
| regions | 10 | 10 | none |
| layout | 14 | 14 | none |
| annotation | 12 | 12 | none |

¹ The target counts Phase 5's renames (`ObjectVectorRate(s)Result`, net 0) and additions: `compute_object_vector_rate(s)` (+2), `has_place_field` (+1), and `place_cell_significance`, `head_direction_cell_significance`, `object_vector_cell_significance` and `spatial_view_cell_significance` (+4). So the target is 64 − 2 + 7 = 69. ² The target counts Phase 5's `shuffle_spike_times_circular` (+1).

**Totals:** 427 names on `main` across the 14 contract namespaces (the line count of the snapshot rendered on `main`, measured), and 418 at the target.

Root `__init__.py` changes:

- delete the eager `from neurospatial.decoding import bin_spikes_in_time`, which also stops `decoding` loading at import;
- set `_LAZY_ATTRS` to the eight flagship names (`"compute_spatial_rate": ("encoding.spatial", "compute_spatial_rate")`, …, `"PeriEventResult": ("events._core", "PeriEventResult")`);
- add the same eight under `if TYPE_CHECKING:` so mypy and IDEs resolve them;
- rewrite the module docstring's "Core Classes" and "Import Patterns" sections to list the new root. Today they list only the core classes and import `SpikeOverlay`.

Delete every test assertion that a removed name is in an `__all__`, for example `tests/decoding/test_decode_session.py:671-674` (`as_spike_trains`).

**6.3 Remove the remaining deprecation shims** (decision 1; Phase 4 removed the `detect_region_crossings` old-order dispatch and Phase 5 the classifier aliases):

- `ImageMaskLayout`'s `bin_size` alias (`layout/engines/image_mask.py:97-130`);
- `ViewRateResult.peak_view_location` (`encoding/view.py:263`) and `ViewRatesResult.peak_view_location` (:820).

Delete their warning tests.

**6.4 One argument convention, checked.**

Add to `src/neurospatial/_validation.py`:

```python
def validate_times_positions(
    times: ArrayLike, positions: ArrayLike, *, call: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a ``(times, positions)`` pair and name an argument swap.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Sample timestamps in seconds; must be 1-D, finite and non-decreasing.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        One position row per timestamp.
    call : str
        Public function name, used in the message.

    Returns
    -------
    times, positions : ndarray
        float64 arrays; shapes are not changed.

    Raises
    ------
    ValueError
        Listing every problem, with a ``Fix:`` line that names the swap when the
        arguments look swapped.
    """
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    problems: list[str] = []
    if t.ndim != 1:
        problems.append(f"times must be 1-D (n_samples,), got shape {t.shape}.")
    elif t.size > 1:
        finite = np.isfinite(t)
        if not finite.all():
            problems.append(f"times has {int((~finite).sum())} non-finite value(s), "
                            f"first at index {int(np.argmin(finite))}.")
        else:
            down = np.flatnonzero(np.diff(t) < 0)
            if down.size:
                k = int(down[0])
                problems.append(f"times must be non-decreasing; it decreases at "
                                f"{down.size} place(s), first {t[k]!r} -> {t[k + 1]!r} at index {k}.")
    if p.ndim not in (1, 2):
        problems.append(f"positions must be (n_samples, n_dims), got shape {p.shape}.")
    if t.ndim >= 1 and p.ndim >= 1 and len(t) != len(p):
        problems.append(f"times has {len(t)} samples but positions has {len(p)}.")
    if not problems:
        return t, p
    looks_swapped = t.ndim == 2 or (p.ndim == 1 and p.size > 1 and bool(np.all(np.diff(p) >= 0)))
    fix = (f"Fix: call {call}(..., times, positions, ...); the arguments look swapped "
           "(timestamps come first, positions second)."
           if looks_swapped else
           "Fix: pass times as a sorted 1-D array of timestamps in seconds with one "
           "positions row per timestamp.")
    raise ValueError(
        f"Invalid times/positions passed to {call}():\n- " + "\n- ".join(problems)
        + "\nWhy: each interval [times[k], times[k+1]) is weighted by its duration, "
        "so mis-shaped or unsorted timestamps give wrong numbers.\n" + fix
    )
```

Call it first in every public function that takes both. Replace the inline checks in `env.occupancy` (`environment/trajectory.py:338-378`) and `env.bin_sequence` with it. Phase 4 (4.2) gave the encoding validator `encoding/_validation.py::validate_trajectory` its own swap check. Make it call `validate_times_positions` for the times/positions problems and keep only its `n_dims` check, so there is one swap detector.

Signature changes on `main` (decision 1, no shims):

- **`(positions, times)` → `(times, positions)`.** Validate, and update internal call sites, tests and doctests:
  - `behavior` (16): `approach_rate`, `compute_decision_analysis`, `compute_goal_directed_metrics`, `compute_path_efficiency`, `compute_pre_decision_metrics`, `compute_vte_session`, `compute_vte_trial`, `extract_pre_decision_window`, `goal_bias`, `head_sweep_from_positions`, `instantaneous_goal_alignment`, `mean_square_displacement`, `pre_decision_heading_stats`, `pre_decision_speed_stats`, `segment_by_velocity`, `time_efficiency`;
  - `ops.visibility_occupancy(env, positions, headings, times)` → `(env, times, positions, headings)`;
  - `simulation.generate_population_spikes(models, positions, times, *, headings)` → `(models, times, positions, *, headings=None)`;
  - `compute_trajectory_curvature(positions, times=None)` and `heading_direction_labels(positions=None, times=None, …)` keep their order, because `times` is optional there. They call the validator whenever `times` is given, so a swap still raises.
- **`env` first** (the field-metric majority — `field_size`, `rate_map_centroid`, `detect_place_fields`, `border_score` — is already env-first):
  - `encoding.compute_region_coverage(field_bins, env)`;
  - `encoding.field_shape_metrics(firing_rate, field_bins, env)`;
  - `encoding.rate_map_coherence(firing_rate, env)`;
  - `ops.map_points_to_bins(points, env)`;
  - `stats.shuffle_place_fields_circular_2d(encoding_models, env)`;
  - `animation.calibrate_video(video_path, env)`.
  - Segmentation-form functions keep env after the data. These are the functions whose first parameter starts with `position_bins` or is `trials` (`detect_laps`, `trajectory_similarity`, `trials_to_region_arrays`, …).
- **Spike parameter named `spike_times`:**
  - `decoding.bin_spikes_in_time(spike_trains, …)`;
  - `events.population_peri_event_histogram(spike_trains, …)`;
  - `behavior.restrict_spike_trains(trains, epochs)` (parameter only);
  - `simulation.validate_simulation(spike_trains=…)`.
- **Dual form removed** (decision 4). Every `times` parameter becomes a plain 1-D array; no `times` slot accepts a `PositionLike` any more. Which functions need positions follows Phase 3b:
  - **Positions required** (encoding needs tracking): `compute_spatial_rate(s)`, `BayesianDecoder.fit(spike_times, times, positions, *, ...)` and `BayesianDecoder.score(spike_times, times, positions, *, ...)` (ground truth). Their `positions` loses its `=None` default.
  - **No positions** (decoding without tracking): `BayesianDecoder.predict(spike_times, times, *, epochs=None, spike_window=None)` and `predict_summary(spike_times, times, *, time_chunk=1024, epochs=None, spike_window=None)`. Here `times` are the timestamps whose valid runs are tiled with decode bins; Phase 3b documents `times=np.arange(t0, t1, dt)` for spans without tracking.
  - **Positions unless models are given:** `decode_session(_summary)(env, spike_times, times, positions=None, *, ..., encoding_models=None)`. `positions` is needed only to encode, and `predict` reaches these functions through the `encoding_models=` path. With `encoding_models=None` and `positions=None`, raise `ValueError` per the contract: "decode_session needs positions to build encoding models. Fix: pass positions, or encoding_models= from a fitted BayesianDecoder."
  - Delete the `PositionLike` adapter branch (`encoding/spatial.py:2874`) and the estimator and session branches, and delete `_typing.PositionLike` once it is unused.
  - pynapple users pass `tsd.t, tsd.values`.
- **`ops.heading_from_velocity(positions, times, *, …)`** (Phase 3c replaced `dt` with `times` in place) → `(times, positions, *, …)`. Update its internal callers (`behavior/vte.py`, `simulation/spikes.py`), Phase 2's `_velocity_heading_and_speed` if it still takes `(positions, times)`, and the docs Phase 3c updated (CLAUDE.md pattern 7, `.claude/QUICKSTART.md`, `docs/api/index.md`, examples 22, 24 and 25).

Add `tests/test_argument_conventions.py`, a rule test over the snapshot namespaces with no name lists. For every public *function* (not class):

- if `times` and `positions` are both positional without defaults, `times` comes first;
- no parameter is named `spike_trains`, `trains` or `trajectory`;
- a positional `env` is the first parameter, unless the first parameter's name starts with `position_bins`, or is `trials` or `nwbfile`. NWB writers take the file container first, as pynwb does (`write_environment(nwbfile, env, …)`, `write_occupancy`, `write_place_field`); the exception is needed because the snapshot namespaces include `neurospatial.io.nwb`.

The archive branch is the reason this guard checks rules, not names. Its convention lint grew into prose-parsing tests.

**6.5 Unit identity** ([Population identity](shared-contracts.md#input-conventions)).

- **Duplicates raise where labels are supplied.** `neurospatial._results.resolve_unit_ids` is the single validator every plural encoder, both PETH functions and the `SpikeTrains` container call (`encoding/spatial.py`, `directional.py`, `view.py`, `egocentric.py`, `spike_trains.py`, `events/_core.py`, `events/alignment.py`). After its length check, it raises `ValueError` when the labels repeat, listing every repeated label: "unit_ids has duplicate labels [3, 7]; each label must name one unit. Fix: pass unique labels, e.g. unit_ids=np.arange(n_units)." Route each `summary_table(unit_ids=)` relabel through it too. The existing `to_xarray` duplicate check then cannot fire on a result built by a compute function; keep it for directly constructed results.
- **`BayesianDecoder.fit(spike_times, times, positions, *, unit_ids=None, …)`.** The new keyword-only argument comes first among the keywords. Labels are never overridden ([Population identity](shared-contracts.md#input-conventions)), as in the encoders since Phase 1 Task 7. If the input is a labelled group and `unit_ids=` is also passed, they must be identical in the same order, or the call raises listing both. Otherwise whichever is present is used. Resolve with `resolve_unit_ids(unit_ids, n_units, input_ids=extracted_ids, context="BayesianDecoder.fit")` and set Phase 1's `_unit_ids_supplied = unit_ids is not None or extracted_ids is not None`. A decoder fitted with `unit_ids=` then aligns a labelled predict input by label (Phase 1 Task 6). Document the keyword and the pairing rule in `fit`.
- **Internal imports.** 6.2 removes `as_spike_trains`/`as_spike_trains_with_ids` from `encoding.__all__`. Import them from `neurospatial.encoding._spikes` in `decoding/estimator.py` (Phase 1 Task 6), `events/alignment.py` (Phase 2 Task 8) and the three plural rate functions (Phase 1 Task 7).

**6.6 Plain data holders** (decision 4).

`SimulationSession` keeps its name and becomes:

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

- **Callers:** update `simulate_session` (`session.py:531,545`), every `examples.py` builder (:738-798, :992-1027), and `validate_simulation`/`plot_session_summary` (`validation.py:25,190-261`, keys `"cell_{i}"` → `unit_id`).
- **Delete** `recording.py` (`Position`, `Session`, `load_session`) and `tests/test_recording.py`. No analysis function accepts `Session`, so it is a second bundle in a design with none.
- **NWB readers** return frozen, non-iterable holders. An old `positions, t = read_position(...)` unpack then fails loudly with `TypeError` instead of swapping:
  - `read_position → NWBPosition(times, positions)`;
  - `read_head_direction → NWBHeadDirection(times, headings)`;
  - `read_units → NWBUnits(spike_times, unit_ids, obs_intervals, spike_window)`.
  - Define them in `io/nwb/_holders.py`, export them from `io.nwb`, and update the internal callers and doctests listed above.
- **Units keep their acquisition windows** ([time-window semantics, Defaults](shared-contracts.md#time-window-semantics): loaders preserve acquisition windows). On `main`, `read_units` (`io/nwb/_units.py:70-186`) reads only `spike_times` and ignores the NWB `obs_intervals` column. Add both fields:
  - `obs_intervals: list[NDArray[np.float64]] | None` holds one `(n_i, 2)` array per read unit (`np.asarray(units[row, "obs_intervals"])`), or `None` when `"obs_intervals" not in units.colnames`.
  - `spike_window: NDArray[np.float64] | None` is the intersection of the read units' `obs_intervals`, folded with `_intervals.intersect_intervals` after `as_intervals`, or `None` when the column is absent. It is the time *every* read unit was observed, so a shared occupancy never includes time when a unit could not fire.
  - The `read_units` docstring says to read units with different coverage in separate calls (`unit_ids=`) when the intersection is too short. An empty intersection is stored as shape `(0, 2)`; passing it to an analysis raises `as_intervals`'s "no rows" contract error.
  - A probe with pynwb shows the column round-trips as `units[i, "obs_intervals"]`: unit 7 `[[0, 100], [1100, 1200]]` and unit 11 `[[10, 1150]]` give the intersection `[[10, 100], [1100, 1150]]`.
- **The NWB recipe** that replaces `Session.from_nwb`: `units = read_units(f); pos = read_position(f); compute_spatial_rates(read_environment(f), units.spike_times, pos.times, pos.positions, unit_ids=units.unit_ids, spike_window=units.spike_window)`. The result's `spike_window_assumed` is then `False` whenever the file recorded `obs_intervals`.

**6.7 API snapshot.** Add `tests/test_public_api_snapshot.py` exactly as below, then generate `tests/data/public_api.txt` with `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py` and commit both. This code (with the 14 namespaces above) was run against `main`: it rendered 427 lines, identically across two processes, and imported neither `pynwb` nor `napari`, so it runs in the default `--extra dev` CI job. The mismatch path printed a unified diff and the update command.

```python
"""Pin the public API: every ``__all__`` name of every public namespace."""

from __future__ import annotations

import difflib
import importlib
import inspect
import os
import re
from pathlib import Path

NAMESPACES = (
    "neurospatial", "neurospatial.encoding", "neurospatial.decoding",
    "neurospatial.behavior", "neurospatial.events", "neurospatial.ops",
    "neurospatial.stats", "neurospatial.simulation", "neurospatial.io",
    "neurospatial.io.nwb", "neurospatial.animation", "neurospatial.annotation",
    "neurospatial.regions", "neurospatial.layout",
)
SNAPSHOT = Path(__file__).parent / "data" / "public_api.txt"
UPDATE_ENV = "NEUROSPATIAL_UPDATE_API_SNAPSHOT"
UPDATE_CMD = f"{UPDATE_ENV}=1 uv run pytest tests/test_public_api_snapshot.py"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _describe(obj: object) -> str:
    """Return the snapshot suffix for one exported object."""
    if inspect.ismodule(obj):
        return " (module)"
    if inspect.isclass(obj):
        return " (class)"
    if not callable(obj):
        return " (constant)"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(<signature unavailable>)"
    # Annotation text depends on the Python and NumPy versions (NDArray's repr),
    # so it is dropped; names, kinds, defaults and order are what break callers.
    sig = sig.replace(
        parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )
    return _ADDRESS.sub("", str(sig))


def render_public_api() -> str:
    """Render the sorted snapshot text for every namespace in ``NAMESPACES``."""
    lines = []
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            lines.append(f"{namespace}.{name}{_describe(getattr(module, name))}")
    return "\n".join(sorted(lines)) + "\n"


def test_public_api_matches_snapshot() -> None:
    current = render_public_api()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        return
    expected = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if current != expected:
        diff = "".join(difflib.unified_diff(
            expected.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile="tests/data/public_api.txt (committed)",
            tofile="public API (this checkout)",
        ))
        raise AssertionError(
            "The public API differs from tests/data/public_api.txt.\n"
            f"{diff}\nIf this change is intended, regenerate the snapshot and "
            f"commit it:\n    {UPDATE_CMD}"
        )
```

After generating it, run Phase 4's executable-docs test and the full suite (`uv run pytest`, then `uv run pytest -m slow`). Every failure is a doc or test that uses a name changed here: fix the caller, never re-export.

**6.8 Documentation** (part of this PR):

- `README.md`: `generate_population_spikes` call at :213; the project tree line `recording.py` at :612.
- `docs/getting-started/quickstart.md`, and CLAUDE.md:
  - the Canonical Argument Order block, with behavior as `(env, times, positions)`;
  - patterns that import from the root;
  - the v0.6 naming-contract lines that mention removed names.
- `.claude/QUICKSTART.md` (:911-913 `spike_trains`, :1012-1019 `read_position` unpacking, :879 `phase_precession()`) and `.claude/API_REFERENCE.md`.
- `docs/user-guide/interoperability.md` (the `Session` and pynapple `PositionLike` sections become the explicit-attribute recipes; the NWB recipe passes `spike_window=units.spike_window`).
- `docs/api/index.md` and `.claude/API_REFERENCE.md` (`SpikeOverlay` → `EventOverlay`, `Session`, and the other removed names).
- `examples/15_simulation_workflows.py`, `examples/20_bayesian_decoding.py` and other examples using `spike_trains` or the swapped behavior order. Sync them with `uv run jupytext --sync` and `uv run python docs/sync_notebooks.py`.
- `CHANGELOG.md` `[Unreleased]`: one "Breaking" section listing every removal, rename and reorder in 6.2–6.6. It also covers `fit(unit_ids=)` (which raises when it disagrees with a labelled group), the duplicate-label error, `predict`/`predict_summary` taking no positions and a plain `times` array, and `NWBUnits.obs_intervals`/`spike_window` read from the NWB `obs_intervals` column.

## Deliberately not in this phase

- **"Did you mean" `__getattr__` redirects for removed names.** Decision 1, and Phase 4's executable docs catch internal users.
- **Renaming `EgocentricPolarEnvironment`, `view_spatial_information`, or the head-direction family.** These are not convention violations under the contract; they are separate naming work.
- **`env.track` linear frame and linearization properties** (design-review High #11). Phase 2 fixes the W-maze bug; a track frame is new functionality.
- **Assembly results carrying `unit_ids`, and `bin_spikes_in_time` returning labelled counts.** That is output work for Phase 7 or later; this phase only renames the parameter.
- **Headings or HD cells in `simulate_session`.** The holder gains no `headings` until a simulator produces them.
- **Behavior `epochs=`.** Phase 3 adds it; this phase only reorders.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_public_api_snapshot.py::test_public_api_matches_snapshot` | rendered text equals `tests/data/public_api.txt` (418 lines at the target; 427 when rendered on `main`); deleting a line from the file fails with a unified diff and the `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1` command |
| `tests/test_argument_conventions.py::test_times_before_positions` | for all public functions with both parameters required, the rule holds. Run against `main` it fails for exactly the 18 functions reordered in 6.4 (measured). At the start of this phase it also fails for `heading_from_velocity(positions, times)`, which Phase 3c introduced: 19 in all |
| `…::test_no_spike_trains_parameter_name` | no public function parameter is named `spike_trains`/`trains`/`trajectory`. On `main` it fails for exactly `bin_spikes_in_time`, `population_peri_event_histogram`, `restrict_spike_trains`, `validate_simulation` |
| `…::test_env_is_first` | the rule holds. On `main` it fails for exactly the 6 env-first functions in 6.4 (measured with the `nwbfile` exception; without it the three NWB writers also fail) |
| `tests/test_validation.py::test_swapped_times_positions_names_swap` | `validate_times_positions(positions_2d, times_1d, call="f")` raises `ValueError`; message contains `"look swapped"`, `"Fix:"`, and both problems (shape and length are listed together when both apply) |
| `…::test_unsorted_times_reports_first_decrease` | times `[0, 1, 0.5, 2]` → message names index 1 and `1.0 -> 0.5` |
| `tests/behavior/test_argument_order.py::test_old_order_raises` (parametrized over the 16 reordered behavior functions and `ops.heading_from_velocity`, plus `compute_trajectory_curvature(times, positions)` swapped) | calling with the old `(positions, times)` order raises `ValueError` containing `"swapped"` |
| `…::test_1d_column_swap_no_longer_silent` | 1-D env, `x` shape (600, 1), `times` shape (600, 1), the audit case: `compute_path_efficiency(env, x, times, goal)` (old order) raises; on `main` it returned efficiency 4.758 (correct 0.083) |
| `tests/test_unit_identity.py::test_duplicate_unit_ids_raise` (parametrized over `compute_spatial_rates`, `compute_directional_rates`, `compute_view_rates`, `compute_egocentric_rates`, `compute_object_vector_rates`, the four `*_cell_significance` functions, `population_peri_event_histogram`, `BayesianDecoder.fit`) | `unit_ids=[3, 3, 7]` → `ValueError` naming `[3]` with a `Fix:` line |
| `…::test_fit_unit_ids_enable_label_alignment` | `BayesianDecoder(env).fit(trains, t, p, unit_ids=[10, 11, 12])`, then `predict` on a spike-group double keyed `[12, 11, 10]` (reordered trains) equals `predict(trains, t)`; keyed `[10, 11, 13]` → `ValueError` with `missing: [12]`, `unexpected: [13]` |
| `…::test_fit_unit_ids_must_match_group_labels` | `fit(group_keyed_[10, 20], …, unit_ids=[20, 10])` raises `ValueError` whose message contains both `[20, 10]` and `[10, 20]` and a `Fix:` line. `unit_ids=[10, 20]` is accepted, with `unit_ids == [10, 20]`; a plain list with `unit_ids=[20, 10]` gives `[20, 10]` |
| `…::test_predict_takes_no_positions` | `list(inspect.signature(BayesianDecoder.predict).parameters)` is `["self", "spike_times", "times", "epochs", "spike_window"]`, and `predict_summary`'s is the same plus `time_chunk`. `fit` and `score` have a `positions` parameter with no default. `decode_session(env, spikes, t)` with no positions and no `encoding_models` raises `ValueError` with `Fix:`, while `decode_session(env, spikes, t, encoding_models=m)` decodes |
| `tests/simulation/test_session_holder.py::test_attributes_feed_analysis` | `sim = open_field_session(duration=60, seed=0)`; `compute_spatial_rates(sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids).unit_ids` equals `np.arange(len(sim.spike_times))`; `set(sim.ground_truth) == set(sim.unit_ids)` |
| `…::test_mismatched_lengths_raise` | `unit_ids` of wrong length → `ValueError` with `Fix:` |
| `tests/nwb/test_reader_holders.py::test_read_position_holder` (pynwb extra) | `pos = read_position(f)`; `pos.times.ndim == 1`, `pos.positions.shape == (n, 2)`; `a, b = read_position(f)` raises `TypeError` |
| `tests/nwb/test_reader_holders.py::test_read_units_spike_window` (pynwb extra) | Units 7 (`obs_intervals=[[0, 100], [1100, 1200]]`) and 11 (`[[10, 1150]]`): `units.obs_intervals[0]` equals the first, and `units.spike_window` equals `[[10, 100], [1100, 1150]]`. A file without the column gives `obs_intervals is None` and `spike_window is None`. Passing `spike_window=units.spike_window` to `compute_spatial_rates` gives `spike_window_assumed is False` |
| `tests/test_lazy_imports.py` (updated) | `import neurospatial` does not import `neurospatial.decoding`; `neurospatial.decode_position is neurospatial.decoding.decode_position` |
| `tests/test_package_imports.py` (unchanged) | every `__all__` name resolves |
| Phase 4 executable-docs test | passes with every doc updated in 6.8 |

## Fixtures

- **Snapshot:** none; it imports the package.
- **Behavior order tests:** reuse the existing behavior test trajectories (600 samples at 30 Hz on a 100 cm grid env). The 1-D case is the audit script `claim4c`, reproduced inline: `x = (50 + 45·sin(linspace(0, 6π, 600)))[:, None]`, `env = Environment.from_samples(linspace(0, 100, 500)[:, None], bin_size=2.0)`, `goal = [95.0]`.
- **Simulation:** `open_field_session(duration=60, seed=0)`.
- **NWB:** reuse the existing `tests/nwb` pynwb fixtures.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
