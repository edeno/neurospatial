# Phase 4a — Errors that teach

**Requires:** Phases 2a, 2b, 3a, 3b, 3c, 3d and 3e merged (all of Phases 2 and 3; see [overview → Rollout Strategy](overview.md#rollout-strategy)). This phase edits error sites they changed.

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#error-message-contract) · next: [4b](phase-4b-docs-that-run.md)

**Line numbers** below were read on `main` at `da631a47`. Phases 1–3 land first and shift them, so find each site by the quoted message text, not by the number alone.

**Inputs to read first:**

- [src/neurospatial/_exceptions.py](../../../../src/neurospatial/_exceptions.py) — four public classes. It imports `EnvironmentNotFittedError` (`environment/decorators.py:19`) and `GraphValidationError` (`layout/validation.py:28`) to re-export them. That import is the cycle worked around by the function-local import at `environment/queries.py:583-590` (inside `distance_to`). `BinIndexOutOfRangeError(index, *, n_bins)`, `IncompatibleEnvironmentError` and `LayoutNotBuiltError(layout_name, attribute)` are exported but never raised anywhere in `src/`.
- `src/neurospatial/environment/core.py:339-353` — `[E1006]`, the model message. Even it has no `Fix:` line.
- `src/neurospatial/encoding/_validation.py:40-247` — `validate_env_fitted`, `validate_times` (its `context` defaults to `"encoding"`, used by decoding too) and `validate_trajectory`.
- `tests/test_exceptions.py` — the existing exception tests. This phase extends the file; it does not replace it.
- **Files earlier phases already changed** (re-locate every site by its quoted message text):
  - `environment/core.py`, `environment/factories.py`, `layout/engines/graph.py`: Phase 2b Task 1 (calculus operator cache) and Phase 2a Task 1 (`edge_id` numbering).
  - `decoding/posterior.py`, `decoding/estimator.py`: Phase 1 Tasks 2 and 6; Phase 3c (per-run bins).
  - `events/alignment.py`: Phase 2a Task 6 (population PETH input) and Phase 3c (`_keep_observed_events`). The four window checks in 4.3 sit next to that code.
  - `behavior/segmentation.py`: Phase 3d split each detector into a public wrapper and `_<name>_contiguous`. Raise `RegionNotFoundError` in the public wrapper, before the per-run loop.
  - `io/nwb/_behavior.py`: Phase 2a Task 3 (`data_from_series`, lazy refusal).
  - `CHANGELOG.md` has two `## [Unreleased]` headings on `main` (`:3` and `:494`). Append under the first; Phase 4b merges them.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — implemented here. Rules 1–4 apply to every site this phase touches, including the exemption for missing required arguments (4.4, last row).
- [Time-window semantics](shared-contracts.md#time-window-semantics) — not changed here.

**Designs referenced:** none.

## Tasks

### 4.1 Exception base class and hierarchy

- In `_exceptions.py`, add the base class and the shared formatter, and **move** the `EnvironmentNotFittedError` and `GraphValidationError` class bodies in unchanged. `environment/decorators.py` and `layout/validation.py` then import them *from* `_exceptions`, and `_exceptions` imports no `neurospatial` module. This removes the cycle, so replace the function-local `from neurospatial._exceptions import RegionNotFoundError` in `queries.py` (`distance_to`, `:590`) with a top-level import, and delete its comment. (`distance_to`'s other local import, `neurospatial.ops.distance`, is unrelated and stays.)

  ```python
  class NeurospatialError(Exception):
      """Base class for every exception neurospatial defines.

      Each concrete error also inherits a built-in type, listed first, so
      ``except ValueError`` keeps working. ``except NeurospatialError`` catches
      only problems that neurospatial itself detected.
      """


  def _format_error(what: str, *, fix: str, why: str | None = None) -> str:
      """Return ``what``, an optional ``why``, and a final ``Fix:`` line."""
      lines = [what.strip()] + ([why.strip()] if why else []) + [f"Fix: {fix.strip()}"]
      return "\n".join(lines)
  ```

- Bases:
  - `RegionNotFoundError(KeyError, ValueError, NeurospatialError)`. Adding `ValueError` (the precedent is NumPy's `AxisError(ValueError, IndexError)`) lets this one class replace the eight `ValueError("... not found ...")` sites in 4.3 with no change of catch type.
  - `BinIndexOutOfRangeError(ValueError, NeurospatialError)`.
  - `IncompatibleEnvironmentError(ValueError, NeurospatialError)`.
  - `LayoutNotBuiltError(RuntimeError, NeurospatialError)`.
  - `EnvironmentNotFittedError(RuntimeError, NeurospatialError)`.
  - `GraphValidationError(ValueError, NeurospatialError)`.

  `RegionNotFoundError` must also define `__str__` returning `str(self.args[0])`. `KeyError.__str__` comes first in its MRO, so it quotes the message and prints the `Fix:` line as a literal `\n`; verified: an override on the base class is shadowed. Each `__init__` builds its message with `_format_error`. `RegionNotFoundError.__init__(name, *, available=None, argument="region_name")` suggests the closest match with `difflib.get_close_matches(name, available, n=1)`. Its fix is either "pass `{argument}='{match}'`" or "add it first: `env.regions.add('{name}', point=(x, y))` (or `polygon=...`)".
- Export `NeurospatialError` from `neurospatial/__init__.py` (import block at `:223-229`, `__all__` at `:297-305`).
- Give the unused classes real raise sites:
  - `BinIndexOutOfRangeError` replaces the `IndexError` at `environment/queries.py:51`. Update the `Raises` sections of `neighbors` and the other callers of `_resolve_point_or_index`.
  - `IncompatibleEnvironmentError` is raised at `composite.py:104` (`[E1003]`) and `decoding/posterior.py:1636` (encoding-model bins ≠ `env.n_bins`).
  - `LayoutNotBuiltError` replaces the **ten** `RuntimeError` "not built" sites. Find them with `grep -rn "not built" src/neurospatial/layout`; their wording differs ("Layout not built…", "Grid layout not built…", "TriangularMeshLayout is not built…"):
    - `layout/mixins.py:293, :358, :436`;
    - `layout/engines/hexagonal.py:364`;
    - `layout/engines/graph.py:386, :422`;
    - `layout/engines/shapely_polygon.py:191`;
    - `layout/engines/triangular_mesh.py:208, :308, :432`.

    Each site passes its layout class name and the missing attribute. `LayoutNotBuiltError.__init__` builds the one wording, with `Fix: call build() on the layout first, or create the environment with a factory such as Environment.from_samples(positions, bin_size=2.0).`

### 4.2 Mechanical relabel and the domain word

- Rename the label `HOW:` to `Fix:` in the existing WHAT/WHY/HOW messages: 156 occurrences across 22 files, `events/` and `animation/` mostly (`grep -rn "HOW:" src/neurospatial`). This is text only.
- Update the tests that assert the old label (`grep -rn "HOW" tests`): `tests/animation/test_rendering.py:605` (`"HOW:" in error_msg`) and `tests/animation/test_video_overlays.py:1564` (`"HOW" in warning_msg`) assert `"Fix:"` instead. The other hits are comments and docstrings; leave them.
- Make `context` keyword-required, with no `"encoding"` default, in `validate_times`, `validate_spike_times` and `validate_trajectory` (`encoding/_validation.py:76, :116, :187`). Then every caller, including `decoding/session.py:445`, names its own function (contract rule 3).
- `validate_trajectory` gains `n_dims: int | None = None`. It collects every problem into a list and raises once (rule 4), covering:
  - `times` not 1-D;
  - length mismatches;
  - `positions.shape[1] != n_dims`;
  - 1-D `positions` with `n_dims > 1`;
  - **swap detection**: `times.ndim == 2` and `positions.ndim == 1` gives "did you pass positions before times?".

  Pass `n_dims=env.n_dims` from every encoding entry point that has an env. This replaces the deep `layout/helpers/regular_grid.py:682` "Dimensionality mismatch … grid_edges" message that users hit today.

### 4.3 Rewrite the first-run raise sites

Of 454 raise sites in the flagship modules, 1 has a `Fix:` line and 122 give no guidance at all (an AST scan of `raise X("...")`). Rewrite only the sites below, chosen because they are on first-run paths or were found by probes. All use `_format_error`, except the bare-`KeyError` sites (last two rows).

| Site (`da631a47`) | Change |
| --- | --- |
| `environment/core.py:339`, `_exceptions.py` E1004 text (from `decorators.py:97-103`) | Add `Fix: env = Environment.from_samples(positions, bin_size=2.0)`. |
| `environment/factories.py:420` (`positions must be a 2D array`) | Fix: `positions[:, None]` for 1-D data; `positions.T` when the shape is `(n_dims, n_samples)`. |
| `environment/factories.py:189` (`Unknown maze kind`) | List the valid kinds. |
| `environment/queries.py:51, :56, :63` | See 4.4 (length-1 array). |
| `layout/helpers/utils.py:331` (`All 'positions' are NaN`) | Report "N of M rows contain NaN; check for tracking dropouts", with a fix. |
| `layout/helpers/regular_grid.py:378-390` (`[E1002]`) | Fix gives a value, e.g. `bin_size=2.0` (same units as `positions`). |
| `encoding/_validation.py:73` | See 4.4 (non-Environment first argument). |
| `encoding/_validation.py:95, :155, :219-245`; `encoding/_binning.py:813` | Fix lines; covered by the 4.2 rewrite. |
| `encoding/spatial.py:2306`, `directional.py:1603`, `view.py:1063`, `egocentric.py:1183` (`unit_ids has … elements`) | Fix: "pass one label per unit (`len(unit_ids) == {n}`)". |
| `decoding/posterior.py:1519`, `likelihood.py:153` (neuron-count mismatch) | Fix: build counts and models from the same unit list, in the same order. |
| `decoding/posterior.py:216, :1636`; `decoding/_binning.py:192` | Fix: `handle_degenerate='uniform'`; `IncompatibleEnvironmentError` (4.1); "use `dt <= {span}`". |
| `events/alignment.py:131, :282, :437, :586` (four copies of the window check) | Replace with one `_validate_window(window, *, context)` (see 4.4). |
| `behavior/segmentation.py:462, :645, :651, :1244, :1849, :1860, :2339`; `events/regressors.py:963`; `environment/queries.py:592` | `raise RegionNotFoundError(name, available=..., argument="start_region")`, with the right argument name each time. |
| `behavior/segmentation.py:1238, :1242, :1855` | Fix shows the missing argument, e.g. `start_region='home'`. |
| `animation/core.py:328, :333, :508, :556` | Fix: `save_path='out.mp4'`; list the valid backends. |
| `io/files.py:299, :442, :444` | Fix line. Note that `env.to_file(path)` writes both `.json` and `.npz`. |
| `io/nwb/_units.py:171` (`unit_ids not found in the units table`, a `ValueError`) | A separate `Fix:` line naming `unit_ids=` and listing how to see the available ids. |
| `io/nwb/_behavior.py:183, :199, :208` (bare `KeyError`) | Fix names the reader argument (`processing_module=`). These stay bare `KeyError`, whose `str()` quotes the message and escapes newlines, so the fix is the final sentence (`… Fix: pass processing_module='behavior'.`), not a separate line. |
| `encoding/spatial.py:3751, :3817` (`Unknown direction label`, bare `KeyError` in `correlation` and `directionality_index`) | Same treatment as the NWB `KeyError` rows: one message whose final sentence is `Fix: pass one of {known labels}.` |

Representative before/after messages (each "after" message is asserted in the validation slice):

```text
# compute_spatial_rate(spike_times, times, positions)        -- env forgotten
before: EnvironmentNotFittedError: [E1004] compute_spatial_rate() requires the environment to be fully initialized. Ensure it was created with a factory method. …
after:  TypeError: compute_spatial_rate() expects an Environment as its first argument, got ndarray with shape (200,).
        Fix: build one with env = Environment.from_samples(positions, bin_size=2.0), then call compute_spatial_rate(env, spike_times, times, positions).

# compute_spatial_rate(env, spike_times, times, positions[:, 0])   -- 2-D env, 1-D positions
before: ValueError: Dimensionality mismatch: points have 1 dimension(s), but grid_edges has 2 and grid_shape has 2. …
after:  ValueError: compute_spatial_rate: positions has shape (1800,) but env is 2-D, so positions must have shape (n_samples, 2).
        Fix: pass both coordinates, e.g. np.column_stack([x, y]); for a 1-D track, build env from 1-D data (positions[:, None]) or Environment.linear_track(...).

# detect_laps(..., start_region="home") on an env with no regions
before: ValueError: start_region 'home' not in env.regions. Available regions: []
after:  RegionNotFoundError: Region 'home' not found. This environment has no regions.
        Fix: add it first: env.regions.add('home', point=(x, y)) (or polygon=...), then pass start_region='home'.
```

### 4.4 Detect the silent and weak first-run mistakes

| Mistake (probed on `main`) | Today | Change |
| --- | --- | --- |
| `from_samples(positions, bin_size=500)` on 80 cm data | Succeeds silently with a 4-bin env. | In `from_samples` (`factories.py`, just before `cls.from_layout` at `:506`): if `np.all(bin_size >= np.ptp(finite_positions, axis=0))` and some extent is > 0, `warnings.warn(UserWarning)`. The message gives the value, the per-axis extent, and `Fix: bin_size=<max extent / 50>`. Run the check only when `bin_size` is a scalar or has length `n_dims`. A 2-D linear track (200 × 5 cm, `bin_size=5`) must not warn. |
| `bin_size=0.01`, ~99M bins | `ResourceWarning`, which default filters hide (`layout/helpers/utils.py:1181-1188`). | **Decision:** make it a `UserWarning` and add a `Fix:` line ("increase bin_size; this grid has {n_bins:,} bins"). The existing 8 GiB hard ceiling (`regular_grid.py:45, :517`) stays. `ResourceWarning` is not a `UserWarning` subclass, so update the five `pytest.warns(ResourceWarning…)` calls in `tests/layout/helpers/test_memory_safety.py` (`:88, :94, :99, :114, :164`) and the `Warns` section of the docstring at `utils.py:1139, :1170`. |
| `env.neighbors(env.bin_at([x, y]))` | Fails deep inside with "points have 1 dimension(s)… grid_edges". This is CLAUDE.md pattern 1. | In `_resolve_point_or_index` (`queries.py:53`): a shape-`(1,)` input in an env with `n_dims > 1` raises `ValueError`. The message explains that `bin_at` returns an array, with `Fix: env.neighbors(int(bin_idx[0]))` or pass the point. |
| Environment forgotten | Misleading E1004, shown above. | `validate_env_fitted` (`_validation.py:40`) raises `TypeError` when `env` has no `_is_fitted` attribute. The real E1004 stays for half-built envs. |
| `peri_event_histogram(..., window=(-500, 1000))`, a window in ms | Silently returns 60 000 bins. | `_validate_window` warns (`UserWarning`) when `stop - start > 60` s, with `Fix: window=(-0.5, 1.0) if you meant milliseconds`. **Assumption:** 60 s is wider than any realistic peri-event window. |
| `from_samples(positions)` (no `bin_size`); `animate_fields(fields)` (no `frame_times`) | Python's own `TypeError`, which names the mixin (`EnvironmentFactories`, `EnvironmentVisualization`). | **Decision: leave as is**, per the contract's missing-argument exemption. The message already names the missing argument. Rewriting the mixin `__qualname__` would also mislabel `EgocentricPolarEnvironment`, which shares those mixins (MRO verified). A `=None` sentinel that then raises would hide that the argument is required. |

### 4.9a User-facing documentation for errors

Per [executing.md](executing.md#while-you-work), each commit above appends its own `CHANGELOG.md` bullet under the first `## [Unreleased]`. This task checks that these are all present, and writes the error reference page.

- `CHANGELOG.md` `[Unreleased]`:
  - `NeurospatialError` and the hierarchy (stdlib base first, so existing `except` clauses keep working);
  - error messages end with a `Fix:` line (formerly `HOW:`);
  - `RegionNotFoundError` is now raised by segmentation and `regressors`, and is also a `ValueError`;
  - the `BinIndexOutOfRangeError` type change at `neighbors` (was `IndexError`);
  - `LayoutNotBuiltError` and `IncompatibleEnvironmentError` are now raised;
  - a non-Environment first argument raises `TypeError` instead of `[E1004]`;
  - the coarse-`bin_size` and window-span warnings;
  - the large-grid warning is a `UserWarning` (was `ResourceWarning`, hidden by default).
- `docs/errors.md`:
  - a "Catching neurospatial errors" section, showing `except NeurospatialError`, the stdlib-first bases, and the `Fix:` convention;
  - update the E1004 and E1006 entries.

## Deliberately not in this phase

- **Docstring completeness, executable docs, stale examples, README/quickstart/CLAUDE.md edits, `docs/changelog.md`** — Phase 4b. That includes CLAUDE.md's stale "Error: `RuntimeError: Environment must be fitted`" heading and gotcha 2 comment (4b's stale-example table), and removing the `detect_region_crossings` old-order dispatch.
- **Rewriting all 454 raise sites.** Only the 4.3 list and the 156 relabels. A blanket rewrite stalls the phase; later phases follow the contract for code they touch.
- **Population-silence and gap warnings** (Phase 3a); **classifier wording** (Phase 5b); **summary/repr/overwrite fixes** (Phase 7). Phase 7 replaces the `io/files.py:299` overwrite message with a shared helper; rewrite it here anyway, since 4b's docs and this phase's tests exercise it.
- **Archive-only items** that are absent on `main` (verified): `_PopulationTypeError`, `remediation` fields, `TemporalSupport`, and `Session.from_arrays` support errors. Nothing to port.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_exceptions.py::test_every_library_error_is_a_neurospatial_error` | For all 7 classes, `issubclass(cls, NeurospatialError)`, and `cls.__mro__[1]` is the stdlib type listed in 4.1. |
| `test_exceptions.py::test_region_not_found_prints_fix_unquoted` | `str(RegionNotFoundError("hom", available=["home"]))` contains `"\nFix: pass region_name='home'"`; it does not start with `'`; `isinstance(exc, ValueError)` and `isinstance(exc, KeyError)` are both True. |
| `test_exceptions.py::test_no_import_cycle` | Static, with `ast`: `src/neurospatial/_exceptions.py` contains no `Import`/`ImportFrom` of a `neurospatial` module, and `environment/queries.py` has no `ImportFrom` of `neurospatial._exceptions` inside a function body. (A subprocess import test would pass trivially: importing any submodule runs `neurospatial/__init__.py`, which imports `_exceptions` first.) |
| `test_exceptions.py::test_layout_not_built_sites` | For two engines (a grid and `TriangularMeshLayout`), calling a method that needs `build()` on an unbuilt layout raises `LayoutNotBuiltError` (also a `RuntimeError`) with a `Fix:` line. |
| `tests/test_first_run_errors.py::test_missing_env_names_the_call` | `compute_spatial_rate(spikes, times, positions)` raises `TypeError` matching `"expects an Environment"`, and its last line starts with `"Fix: "`. |
| `test_first_run_errors.py::test_1d_positions_on_2d_env` | Raises `ValueError` matching `r"shape \(1800,\).*2-D"` and `"Fix:"`, not `"grid_edges"`. |
| `test_first_run_errors.py::test_swapped_times_positions` | `compute_spatial_rate(env, s, positions, times)` raises with `"did you pass positions before times"`. Several problems in one call are listed in one message: a length mismatch plus a dimension mismatch gives 2 bullet lines. |
| `test_first_run_errors.py::test_neighbors_of_bin_at_output` | `env.neighbors(env.bin_at([[50, 50]]))` raises `ValueError` matching `r"int\(bin_idx\[0\]\)"`. |
| `test_first_run_errors.py::test_coarse_bin_size_warns` | `bin_size=500` on 80 cm data gives exactly one `UserWarning` containing `"bin_size=500"` and `"Fix:"`. `bin_size=2.0` gives none. A 200 × 5 cm track with `bin_size=5.0` gives none. |
| `test_first_run_errors.py::test_large_grid_warning_is_visible` | The grid-size warning category is `UserWarning`, so it is visible under default filters. |
| `test_first_run_errors.py::test_psth_window_in_ms_warns` | `window=(-500, 1000)` warns with `"window=(-0.5, 1.0)"`. `(-1.0, 2.0)` does not warn. `(1.0, -1.0)` raises with `"Fix:"`. |
| `test_first_run_errors.py::test_segmentation_unknown_region` | `detect_laps(..., start_region="home")` on an env without regions raises `RegionNotFoundError` containing `"env.regions.add('home'"`. |
| `test_first_run_errors.py::test_rewritten_sites_teach` (parametrized over the 4.3 table) | Each triggering call's `str(exc)` contains a line starting `"Fix: "`. The two `encoding/spatial.py` bare-`KeyError` sites are checked for `"Fix:"` anywhere. The NWB sites live in `tests/nwb/test_reader_errors.py` instead (same assertions; `_behavior.py`'s three `KeyError`s checked for `"Fix:"` anywhere), because `test_nwb.yml` runs `tests/nwb` with pynwb installed while the default job skips NWB tests. |
| Updated: `tests/layout/helpers/test_memory_safety.py`, `tests/animation/test_rendering.py`, `tests/animation/test_video_overlays.py` | Assert `UserWarning` and `"Fix:"` instead of `ResourceWarning` and `"HOW"` (4.2, 4.4). |

No test in this phase is marked `slow`.

## Fixtures

- A module-scoped fixture in `tests/test_first_run_errors.py`: `times`, 60 s at 30 Hz (1800 samples); `positions = 50 + 40 * [sin(2πt/20), cos(2πt/13)]` (cm); `env = Environment.from_samples(positions, bin_size=4.0, units="cm")`; 300 seeded uniform spike times. The error tests need no real data. Phase 4b's docs fixture uses the same trajectory.

## Definition of done

Everything in [executing.md → Definition of done](executing.md#definition-of-done), plus:

- `uv run --extra docs mkdocs build --strict` (`docs/errors.md` changed).
- `uv run pytest tests/nwb/test_reader_errors.py -n 4 -rs` under `uv sync --all-extras` shows no skips, so the NWB cases actually ran.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the function-local import in `queries.py`, the four duplicated window checks, the `ResourceWarning` category.
- User-facing documentation listed as tasks is updated, not deferred.
