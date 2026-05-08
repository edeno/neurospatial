# v0.4 UX Cleanup — Implementation Plan

**Committed**: 2026-05-08
**Status**: Proposed
**Target release**: v0.4.0 (single cut, hard API break from v0.3.x)
**Source**: [`docs/reviews/UX_REVIEW_2026-05-08.md`](../../reviews/UX_REVIEW_2026-05-08.md) (~162 findings across 11 categories).

## Source

This plan addresses findings from a five-pass UX audit of the repository. The audit had three loud themes:

1. **Onboarding is broken at the front door.** README, `.claude/QUICKSTART.md`, `docs/getting-started/quickstart.md`, and the top-level package docstring all contain copy-paste-failing snippets. CI does not execute README/quickstart code blocks, so these regressions persist while green.
2. **Pervasive parameter-name and argument-order drift.** Distance metric, smoothing bandwidth, and velocity threshold each have 3-4 spellings. `env` placement and the canonical encoding/behavioral signatures are violated in ~20 functions.
3. **Silent wrong-result paths in numerically-sensitive code.** `detect_place_fields` drops fast-firing cells, batch metric functions substitute `nan` per neuron without summary, `bin_at` and `map_points_to_bins` disagree on out-of-env points, polar envs are indistinguishable from Cartesian.

The package is in alpha (v0.3.x) so this plan **does not** carry a deprecation cycle: v0.4.0 ships a single hard break with a CHANGELOG that documents the renames and migration mapping. Every breaking change has an entry in `CHANGELOG.md` "Breaking changes" and one or more migration examples in `docs/migrating-to-v0.4.md`.

## Goals

1. **Make first-run onboarding deterministic.** Every first-run example in `README.md`, `.claude/QUICKSTART.md`, `docs/getting-started/quickstart.md`, the top-level package docstring, and the `examples/` notebooks is either executable as written or explicitly marked as pseudocode / skipped with a reason. CI executes a curated, representative subset on every PR.
2. **Eliminate silent wrong-result paths.** Either the operation succeeds, or it raises an `EnvironmentNotFittedError` / `ValueError` / domain-specific exception with a clear, actionable message. Fallbacks that silently substitute `nan`, `[]`, or "nearest in-env bin" are surfaced via warnings, return-type metadata, or removed.
3. **Make the public surface predictable.** One spelling per concept (`metric`, `bandwidth`, `min_speed`, `env`, `occupancy`); canonical argument order from CLAUDE.md actually enforced; result-class shapes are uniform; `is_X_cell` classifiers and `*Result` containers follow one pattern each.
4. **Fill the highest-value example gaps.** NWB loading, head-direction tuning, and peri-event analysis (PSTH) each get a notebook. The bandit-dataset notebook gracefully degrades on a fresh clone.
5. **Tighten error/warning hygiene.** `EnvironmentNotFittedError` is the single canonical "not fitted" exception. Numeric error messages name units. `warnings.warn` calls use explicit `category=` and `stacklevel=`. Production `print()` calls become `logger.*`.

Out of scope:

- New analysis features (no new encoding/decoding/behavior algorithms).
- Performance work (separate plan; the encoding-cleanup plan already covers the hot paths).
- Repackaging the `Environment` mixin scaffold; this plan only adjusts mixin method signatures, not the mixin design.

## Compat policy and release shape

- **Single v0.4.0 cut.** All milestones land on `main` before the tag. Each milestone is a mergeable PR that keeps tests green; users see one upgrade.
- **No backwards compatibility.** This is the operating constraint for the entire plan. Every renamed kwarg, function, factory, attribute, or method is a clean delete-and-replace. No aliases. No `DeprecationWarning`. No `# kept for compat` shims. No grandfathered call patterns. Old names that survive the rename simply do not exist in v0.4.0. Catching the old name fails with `TypeError: unexpected keyword argument 'X'` or `AttributeError`, which is the desired user signal.
- **The CHANGELOG and migration doc are documentation, not code.** `CHANGELOG.md` "Breaking changes" + `docs/migrating-to-v0.4.md` exist as a migration aid for users updating their own code. The library itself ships zero v0.3-compatibility surface.
- **Migration doc is mandatory output.** Every renamed kwarg, reordered signature, removed function, removed method, and result-class field rename appears in a single migration table. CI fails if the migration doc references a `To` name that doesn't exist in the codebase (cross-check task in M7).

## Milestones

```text
M0  Onboarding hotfixes        (10 tasks, low risk)         pure docs + CI
M1  Stop silent wrong results  (8 tasks, medium risk)       behavior changes
M2  API consolidation          (24 tasks, high risk)        hard break — names/order/returns
M3  Errors and validation      (9 tasks, medium risk)       exception unification
M4  Coordinate-convention safe (5 tasks, medium risk)       polar/linearized/peak_coords
M5  Environment class polish   (9 tasks, medium risk)       mutability/factories/serialization
M6  Examples and docs site     (12 tasks, low-medium risk)  new notebooks + nav
M7  v0.4.0 release prep        (6 tasks, low risk)          CHANGELOG, migration doc, version bump
```

Total: ~83 tasks. Some review findings collapse into one task (e.g., all four parameter-naming families fold into M2 task 2.1).

### Milestone dependencies

```text
M0 ─┬──────────────────────────────────────────────► M7
    │
    └─► M1 ─┬──► M2 ─┬─► M4 ─┐
            │       │        ├──► M5 ─► M6 ─► M7
            │       └─► M3 ──┘        │
            └────────────────────────►┘
```

- **M0** is independent and can ship first; it does not block any later milestone.
- **M1** must land before M6 so new examples don't paper over silent bugs.
- **M2** must land before M3, M4, M5, M6 so error messages, examples, and migration doc reference final names.
- **M3** depends on M2 because some error messages quote parameter names that get renamed.
- **M4, M5** can run in parallel after M2.
- **M6** is last (other than release prep) so notebooks reflect the final API.
- **M7** packages everything for release.

### Milestone gates

Every milestone gates on:

- `uv run pytest -q` is green.
- `uv run ruff check . && uv run ruff format --check .` is green.
- `uv run mypy src/neurospatial/` is no worse than baseline.
- `uv run pytest --doctest-modules src/neurospatial/` is green for any module touched in the milestone.
- Any new public symbol or rename has at least one regression test.
- The changes are reflected in `CHANGELOG.md` (under `## [Unreleased]` until M7 cuts the release).

Milestones M0, M2, M3 additionally gate on the new "smoke-test snippets" CI job introduced in M0 task 0.10.

---

## Milestone details

### M0 — Onboarding hotfixes (Tier 1)

**Goal**: Every visible code block on the user's first-30-minutes path runs under the current API. CI prevents regression.

**Risk**: Low. Pure documentation and CI. No behavior changes in `src/neurospatial`.

**Notable tasks**:

- Fix `data_samples=` → `positions=` in `README.md:148, 225` and any straggler.
- Add required `frame_times=` to all README animation snippets (`README.md:430-440`).
- Fix the four broken snippets in `.claude/QUICKSTART.md` (lines 66, 91, 191, 739).
- Fix the four broken snippets in `docs/getting-started/quickstart.md` (lines 194, 202, 283, 287). Note: line 202 needs more than a kwarg rename — `from_samples()` does not currently support triangular layouts; the example must use a different factory or be removed.
- Fix the broken examples in the top-level package docstring (`src/neurospatial/__init__.py:140, 190`).
- Update `examples/README.md` to document all 22 notebooks; rename the misnamed reference to notebook 08.
- Add a CI job that executes a curated set of README / quickstart examples against a freshly-built environment. Use a small helper (`scripts/test_doc_snippets.py`) driven by an explicit manifest of snippet IDs / line anchors. Markdown fenced blocks and Python docstring doctest examples are extracted through separate parsers; placeholder-only prose examples are either rewritten into runnable examples or explicitly marked as skipped.

This milestone has no dependencies and is the only one that can ship as v0.3.1 if the team wants an immediate user-facing patch.

### M1 — Stop silent wrong results (Tier 2)

**Goal**: No public path silently produces an output that looks plausible but is wrong.

**Risk**: Medium. Some changes alter return-type / warning behavior.

**Notable tasks**:

- **Subset round-trip.** Rewrite `Environment.subset()` to return a `MaskedGrid`, reusing the existing layout serializer instead of adding a one-off `subset` layout kind. The resulting environment must round-trip through `to_file`/`from_file`.
- **`bin_at` vs `map_points_to_bins` divergence.** Standardize trajectory semantics on `-1` for points outside the environment. `Environment.bin_sequence` and `Environment.occupancy` must agree. "Nearest in-env bin" remains available only for explicit interpolation-style queries.
- **Polar envs.** Add an `is_polar: bool` property and a `coordinate_kind: Literal["cartesian", "polar"]` attribute on `Environment`. Make Cartesian-only methods (`distance_to`, `bin_at` on (x, y) input, `plot_field` axis labels) raise a clear `ValueError` for polar envs. Update `from_polar_egocentric` to set the flag.
- **`detect_place_fields`.** Either remove the silent `if mean_rate > max_mean_rate: return []` early-exit or return a richer object that reports `excluded_reason="mean_rate_above_threshold"` so the caller can distinguish "no fields detected" from "cell excluded as putative interneuron". Issue a `UserWarning` regardless.
- **Batch metric `nan` substitution.** `batch_grid_scores`, `batch_border_scores`, and any sibling that swallows `(ValueError, RuntimeError)` must (a) emit a `UserWarning` with the failure count, (b) return a structured result with a `failures: NDArray[bool]` mask in addition to the raw scores.
- **`@check_fitted` / fitted-state validation at entry of public env-consuming functions.** `compute_spatial_rate`, `compute_egocentric_rate`, `compute_view_rate`, `decode_position`. (`compute_directional_rate` does not take an `env` so is not in scope here; see M2 task 2.15.)
- **`spike_times` sortedness check.** Add `_validate_spike_times` to `encoding/_validation.py` that asserts 1-D, finite, sorted, non-negative; call from all four `compute_*_rate(s)` entry points. Same for `decode_position` validating non-negative `spike_counts`.

### M2 — API consolidation (Tier 3, hard breaks)

**Goal**: One spelling per concept, one shape per result class, canonical argument order enforced.

**Risk**: High. This is the single biggest user-visible PR. Land it with the migration doc draft + a complete CHANGELOG section before merging.

**Notable tasks**:

- **Distance-metric kwarg.** Rename `distance_metric`, `distance_type`, `use_geodesic` → `metric` everywhere. For physical distance APIs, standardize legal values to `{"euclidean", "geodesic"}` and replace `{"euclidean", "graph"}` with `{"euclidean", "geodesic"}`. Do **not** collapse graph-hop reachability into Euclidean/geodesic distance: `reachable_from(metric=...)` keeps `{"hops", "geodesic"}` because `"hops"` means graph-edge count and is semantically distinct from weighted path length.
- **Bandwidth kwarg.** Rename `smoothing_sigma`, `kernel_bandwidth` → `bandwidth` everywhere.
- **Velocity-threshold kwarg.** Rename `velocity_threshold`, `speed_threshold`, `threshold` (in `segment_by_velocity`) → `min_speed`.
- **Result-class field naming.** `EgocentricRateResult.ego_env` → `env`; `ViewRateResult.view_occupancy` → `occupancy`. `DirectionalRateResult` remains the documented exception because it is an angular histogram result, not an `Environment`-backed spatial result; keep `bin_centers` / `bin_size` as its domain representation unless a real angular-domain abstraction is introduced in a separate design.
- **`PeriEventResult.firing_rate`.** Convert from method to attribute (cached on construction or computed lazily via `__getattr__`); name preserved across all result classes.
- **`compute_directional_rate` and `is_head_direction_cell`.** These are the only two encoding functions that don't take `env`, by design: their domain is circular heading, not spatial position. Document the exception explicitly in CLAUDE.md and the function docstrings with rationale. Do not add an `Environment` shim for symmetry.
- **Argument order.** Apply CLAUDE.md canonical order across:
  - Encoding (`env` first): `population_coverage`, `detect_place_fields`, `rate_map_centroid`, `field_size`, `field_shift_distance`, `compute_field_emd`, `border_score`.
  - Egocentric ops: `allocentric_to_egocentric`, `egocentric_to_allocentric` reorder to `(positions, headings, targets)`.
  - Behavioral seg: `detect_runs_between_regions` switch to `position_bins` and accept region-spec kwargs only after `*`.
  - Decision/VTE: `compute_decision_analysis`, `compute_vte_session`, `decision_region_entry_time` reorder to canonical and force region kwargs after `*`.
  - Navigation: `path_progress`, `distance_to_region`, `cost_to_goal` reorder to `(position_bins, times, env, *, ...)`.
  - Events: `distance_to_reward` swap `times` and `positions`.
  - Decoding: align `fit_isotonic_trajectory` and `fit_linear_trajectory` to the same order; both should accept `env` as a keyword (`None` allowed iff Cartesian).
- **Required-keyword discipline.** `Environment.animate_fields` accepts `frame_times` as the first positional after `*` (not buried 18 deep). `name` becomes keyword-only across all `Environment.from_*` factories. All simulation entry points and `*CellModel.__init__` get a `*` separator.
- **`is_object_vector_cell` signature.** Rewrite to take raw data `(env, spike_times, times, positions, headings, object_positions, *, ...)` like its sister classifiers, computing the tuning curve internally. Keep a private helper for the precomputed-tuning-curve path if used internally.
- **Function-name disambiguation.** Keep `compute_path_efficiency` (returning `PathEfficiencyResult`). Remove `path_efficiency` (the float-returning sibling) outright. The float is recoverable from `result.efficiency`.
- **Result classes.** Convert `mean_square_displacement`, `grid_orientation`, `segment_by_velocity` to return result-class / dataclass / `list[Run]`. `detect_place_fields` returns a `PlaceFieldsResult` (per M1). `Environment.bin_sequence(return_runs=True)` is split into a separate `bin_sequence_with_runs()` method.
- **Re-export hygiene.** Audit `__all__` lists; remove cross-domain re-exports of `circular_mean`, `rayleigh_test`, `mean_resultant_length`, `compute_egocentric_bearing`, `FieldOfView`, `compute_viewed_location`/`viewshed`, `shuffle_*` so each public symbol has exactly one canonical import path.

### M3 — Errors and validation (Tier 5 + Tier 2 follow-on)

**Goal**: One exception type per failure class. Every validation gap from §5 of the review is closed. Numeric errors carry units.

**Risk**: Medium. Behavior changes for users who catch specific exceptions, but those users are rare in v0.3.x alpha.

**Notable tasks**:

- **`EnvironmentNotFittedError` is the only "not fitted" exception.** Add a free-function variant of `EnvironmentNotFittedError` that accepts a function name (drop the misleading `Environment.X()` formatting in `behavior/navigation.py`). Migrate all manual `if not env._is_fitted: raise RuntimeError/ValueError(...)` sites: `composite.py:73`, `animation/core.py:323-327`, `encoding/population.py:236-240, 388-392`, `ops/transforms.py:1439-1443`, `ops/alignment.py:113, 115`, `io/nwb/_environment.py:185-190`.
- **`@check_fitted` coverage.** Methods that succeed on unfitted envs (`clear_cache`, `components`) and methods that raise plain `AttributeError` (`copy`, `occupancy`, `rebin`, `plot_1d`) get the decorator. `Environment.save` / `Environment.load` are excluded because M5 removes them outright.
- **Custom exception classes.** Export `GraphValidationError`. Add `RegionNotFoundError(KeyError)`, `BinIndexOutOfRangeError(ValueError)`, `IncompatibleEnvironmentError(ValueError)`, `LayoutNotBuiltError(RuntimeError)`. Update raise sites.
- **Units in error messages.** Sweep ~30 `raise ValueError(f"X must be positive, got {x}")` sites in `simulation/`, `encoding/_smoothing.py`, `encoding/grid.py`, `behavior/segmentation.py` to include units (`Hz`, `s`, `cm`, `cm/s`).
- **Stack-context errors.** `encoding/_binning.py` mismatch errors gain a `context: str` arg threaded from public entry points (`compute_spatial_rate`, etc.) so users see "in compute_spatial_rate: times length 100 != positions length 99" rather than a deep helper's plain message.
- **Warning hygiene.** Sweep `stats/circular.py`, `regions/io.py`, `regions/plot.py`, `encoding/_field_metrics.py` for `warnings.warn(...)` without `category=` and without `stacklevel=`. Standardize: data-quality issues → `UserWarning`, numerical fallbacks → `RuntimeWarning`. Always set `stacklevel=2`.
- **Print → logger.** Replace `print(...)` in `animation/backends/video_backend.py`, `html_backend.py`, `widget_backend.py`, `_timing.py`, `simulation/spikes.py:328-331` with module-level `logger.info` / `logger.debug`. Add module-level loggers where missing.
- **Region overwriting.** `Regions.__setitem__` on a duplicate key raises `RegionNotFoundError` (or a sibling `RegionAlreadyExistsError`) instead of warning-and-overwriting. CLAUDE.md "regions are immutable" becomes truthful.
- **Silent fallbacks.** Audit and fix:
  - `encoding/grid.py:1017-1019` (peak_local_max swallow)
  - `encoding/border.py:243-244` (Dijkstra failure → nan)
  - `simulation/trajectory.py:353-356, 394-396, 412-414` (three near-identical bin_at fallbacks)
  - `layout/mixins.py:160-168` (KDTree -1 propagation)
  - `animation/_parallel.py:1471-1472`, `animation/backends/napari_backend.py:2507-2509, 2669-2671`
  Each either re-raises with context, returns a structured result with a failure flag, or stays silent only if the fallback is verifiably correct.

### M4 — Coordinate convention safety (Tier 2/3)

**Goal**: No method silently treats polar coordinates as Cartesian or `(row, col)` as `(x, y)`. Heading conventions are unambiguous in every public docstring.

**Risk**: Medium. `is_1d` rename and `peak_coords` reorder are user-visible.

**Notable tasks**:

- **`is_1d` rename.** Rename `Environment.is_1d` to `Environment.is_linearized_track` (the property currently means "linearized 1D track in a 2D world"). Add a separate boolean `n_dims_intrinsic == 1` if needed. CLAUDE.md, examples, and tests update accordingly.
- **`GridProperties.peak_coords` reorder.** Change from `(row_offset, col_offset)` to `(x_offset, y_offset)` to match the rest of the API. This is a breaking change to a public dataclass; document in CHANGELOG.
- **Heading convention docstrings.** `compute_directional_rate`, `compute_view_rate`, `compute_egocentric_rate`, `is_head_direction_cell`, `is_spatial_view_cell`, `is_object_vector_cell` get an explicit "what does heading=0 mean" line. Decide convention per function (allocentric for inputs is fine; document where the egocentric transform happens).
- **Units validation on `Environment.units`.** Either keep as a free-form string but document it as decorative, or introduce a small `UnitsRegistry` (`{"cm", "m", "mm", "px"}`) and raise on arithmetic between mismatched-units envs in `composite.py`.
- **`simulate_trajectory_ou` units.** Make `speed_units` required (no `None` default); if user passes `speed_units` that doesn't match `bin_size` units, raise. Default speeds in cm/s rather than m/s to match common usage.

### M5 — Environment class polish (Tier 2/3)

**Goal**: The `Environment` class behaves predictably under mutation, factory selection, and serialization.

**Risk**: Medium. Frozen-dataclass route is invasive; the alternative (version counter) is less invasive.

**Notable tasks**:

- **Mutability.** Pick one:
  1. `@dataclass(frozen=True)` on `Environment` — invasive but bulletproof. Mutations route through `with_*` builder methods.
  2. Version counter — increment on `_setup_from_layout` and on any documented mutation path; cached properties verify the version on access.
  Recommend (2) for v0.4.0 because (1) ripples through `copy()`, `subset()`, mixin internals.
- **Factory cleanup.** Make `name` keyword-only across all `from_*` (covered in M2). Add examples to `from_graph` docstring. Rename `from_image` → `from_pixel_mask` and `from_mask` → `from_grid_mask` to make the input contract obvious from the name. The old names are removed outright.
- **Region mutation paths.** Consolidate: `add` raises on duplicate, `update_region` raises on missing, `__setitem__` raises on duplicate (changed in M3 task), `__delitem__` and `remove` both raise on missing. Add `Regions.set(name, ...)` as the idempotent path.
- **`bin_attributes` / `edge_attributes` cost.** Convert from `@cached_property` to methods (`get_bin_attributes()`, `get_edge_attributes()`). Document large-env cost in the method docstring.
- **`differential_operator` cost.** Same treatment. Or keep `@cached_property` but log a warning when the env has > 1e4 bins.
- **`mask_for_region(name)` and `region_mask(name)` reconciliation.** Remove `mask_for_region` outright. `region_mask` is the canonical method (more general — accepts name, list, `Region`, or `Regions`). Standardize on `covers` as the containment predicate (inclusive of bins on the polygon boundary). All callers in `src/`, `tests/`, `examples/`, `docs/` migrate to `region_mask`.
- **`__repr__` and `__str__`.** Fix the `name=None` bug for empty-string names. Add `__str__` returning `info()`. Remove `Environment.save`/`Environment.load` outright (pickle path is replaced by `to_file`/`from_file`); no `DeprecationWarning`.
- **`SubsetLayout` KDTree caching.** Cache the KDTree on the layout instance; invalidate on graph mutation (covered by version counter from task 5.1).

### M6 — Examples and docs site (Tier 4)

**Goal**: The "I have NWB data + spikes; how do I make a place field?" 30-second test passes.

**Risk**: Low-medium. New notebooks add surface area; mkdocs/CI must be updated.

**Notable tasks**:

- **Four new notebooks** (slot 21 is already taken by `21_directional_place_fields`, moved into `examples/` during M0 task 0.7):
  - `examples/23_object_vector_cells.ipynb`
  - `examples/24_head_direction_tuning.ipynb`
  - `examples/25_peri_event_psth.ipynb`
  - `examples/26_loading_from_nwb.ipynb` (uses a small public NWB file from DANDI)
- **Bandit dataset.** Add `data/README.md` (tracked via a `.gitignore` exception) with download instructions for the bandit `.pkl` files (Zenodo or similar). `examples/19_real_data_bandit_task.py` gates on file presence and prints a friendly skip message with the URL.
- **README front door.** Add a "First place field" section after the existing Quickstart that walks user-owned spike-times + tracking arrays to a plotted firing-rate map in <30 lines.
- **Glossary page.** `docs/glossary.md` defines `bin`, `node`, `cell`, `field`, `rate map`, `tuning curve`, `place cell`/`place field`, `egocentric`, `object-vector`, `allocentric`, `view field`, `linearization`. Cross-linked from `core-concepts.md`.
- **mkdocs.yml.** Add the eight orphaned user-guide pages to nav: `differential-operators.md`, `neuroscience-metrics.md`, `rl-primitives.md`, `signal-processing-primitives.md`, `spike-field-primitives.md`, `trajectory-and-behavioral-analysis.md`, `performance.md`, plus the four `docs/*.md` pages flagged in the review.
- **`docs/api/index.md` expansion.** Add sections for `encoding`, `decoding`, `behavior`, `events`, `ops.egocentric`, `ops.visibility`, `ops.basis`, `stats`, `animation`, `io.nwb` with mkdocstrings auto-generation.
- **Examples README rewrite.** Cover all 22 notebooks with description, prerequisites, estimated time, and a task-oriented index ("Goal → notebook").
- **Regenerate notebook outputs.** Re-execute notebooks 15 and 20 (currently 0 outputs). Add a CI job that re-executes a representative notebook on PRs to catch silent regressions in the example surface.
- **Notebook 21 sync.** Move `docs/examples/21_directional_place_fields.{py,ipynb}` to `examples/21_directional_place_fields.{py,ipynb}` (or wherever the canonical home is, consistent with `docs/sync_notebooks.py`).
- **Glossary terminology consistency sweep.** README mixes `position` and `positions`; doctring examples use `data` (animation overlays) where the rest uses `positions`. Apply the glossary's canonical terms to all `.md`, `.py` docstrings, and notebook narrative.

### M7 — v0.4.0 release prep (Tier 5)

**Goal**: Release artifact is shippable. Migration story is documented and verified.

**Risk**: Low.

**Notable tasks**:

- **`docs/migrating-to-v0.4.md`.** Single migration page with one row per breaking change: from-name → to-name + minimal code diff. Cross-check task: a script that confirms each `to-name` actually exists in the codebase. The script does **not** check that `from-name` is absent (since absence is the whole point of the release; the v0.3 names are simply gone).
- **`CHANGELOG.md`** under `## [0.4.0]`: Breaking changes (one entry per renamed kwarg, removed function, reordered signature, removed method), Added (notebooks, glossary, doc-snippet smoke CI), Fixed (the silent-failure list), Changed (renames table by reference to migration doc), Removed (`Environment.save`/`load`, `path_efficiency` float-returning function, all cross-domain re-exports). No "Deprecated" section — there are no deprecations in this release.
- **Version bump.** `pyproject.toml`, `src/neurospatial/__init__.py` if `__version__` is set there, citation strings in `README.md` and `docs/index.md`.
- **Internal version drift.** Audit and fix:
  - README "Tested Dependency Versions" header (currently says v0.2.0)
  - `docs/index.md` citation (currently 0.1.0)
  - `examples/README.md` citation (currently 0.1.0)
  - `src/neurospatial/ops/visibility.py:9` "(v0.4.0+)" — verify accuracy.
- **`examples/_style.py`.** Shared matplotlib styling (Wong palette, fixed figsize). Import at top of every example notebook.
- **Final smoke**: full `uv run pytest`, full `mkdocs build --strict`, full notebook re-execution on a fresh environment, verify `pip install -e .` from a clean checkout produces a working interactive session that runs the README quickstart end-to-end.

---

## Risk and rollout summary

| Milestone | Hard breaks for users? | Estimated PR count |
| --- | --- | --- |
| M0 | No (pure docs) | 1-2 PRs |
| M1 | Behavior changes (silent → loud) | 3-4 PRs |
| M2 | Yes (renames, reorders, removed functions) | 1 large PR + helper PRs |
| M3 | Behavior changes (exception types) | 2-3 PRs |
| M4 | Yes (`is_1d` rename, `peak_coords` reorder) | 1-2 PRs |
| M5 | Yes (factory rename, `save`/`load` removal) | 2-3 PRs |
| M6 | No | 4-6 PRs |
| M7 | No | 1 PR |

Estimated total: 14-22 PRs over the milestone sequence. Most milestones are 1-3 PRs because the changes within a milestone are tightly coupled.

## Open questions for the maintainer

These are decisions the plan defers and that should be settled before the corresponding milestone starts. Defaults below will be applied if not otherwise resolved.

1. **M4 task: `Environment.units` validation.** Free-form string vs strict registry? (Default: lightweight advisory registry for v0.4.0; reject unknown units with a warning, not an error.)
2. **M5 task: mutability strategy.** Frozen dataclass vs version counter? (Default: version counter for v0.4.0.)
3. **M6 task: bandit dataset host.** DANDI, Zenodo, figshare, or repo-internal LFS? (Default: Zenodo with a `data/README.md` pointing to the DOI and a `.gitignore` exception so the README is tracked.)

The previously-open questions on `path_efficiency` deletion and `from_image` / `from_mask` renaming have been resolved as decisions in the plan body (delete and rename, respectively) rather than left open.

## Files most affected (cross-reference)

Concentrating fixes by file (high-touch first):

- `src/neurospatial/encoding/spatial.py` (10+ findings)
- `src/neurospatial/encoding/directional.py` (8)
- `src/neurospatial/encoding/egocentric.py` (10+)
- `src/neurospatial/encoding/_field_metrics.py` (8)
- `src/neurospatial/encoding/grid.py`, `border.py`, `view.py`, `population.py` (~5 each)
- `src/neurospatial/behavior/navigation.py` (12)
- `src/neurospatial/behavior/segmentation.py`, `decisions.py`, `vte.py`, `trajectory.py` (~3 each)
- `src/neurospatial/environment/queries.py` (5)
- `src/neurospatial/environment/visualization.py` (5)
- `src/neurospatial/environment/factories.py` (5)
- `src/neurospatial/environment/transforms.py` (3 high-severity)
- `src/neurospatial/environment/regions.py` (M5 region API)
- `src/neurospatial/decoding/posterior.py`, `metrics.py`, `_result.py`, `trajectory.py`
- `src/neurospatial/events/_core.py`, `regressors.py`, `alignment.py`
- `src/neurospatial/ops/egocentric.py`, `visibility.py`, `distance.py`, `transforms.py`, `alignment.py`
- `src/neurospatial/simulation/trajectory.py`, `spikes.py`, `session.py`, `examples.py`, `models/*.py`
- `src/neurospatial/animation/overlays.py`, `core.py`, `backends/*.py`
- `src/neurospatial/__init__.py` (top-level docstring examples)
- `README.md`, `CLAUDE.md`, `.claude/QUICKSTART.md`, `docs/getting-started/quickstart.md`
- `examples/README.md`, all 22 notebooks
- `mkdocs.yml`, `.github/workflows/docs.yml`, `.github/workflows/test.yml`

See [TASKS.md](TASKS.md) for per-task tracking.
