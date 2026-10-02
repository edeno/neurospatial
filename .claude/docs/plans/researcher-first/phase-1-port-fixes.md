# Phase 1 — Port the verified scientific fixes from the archive branch

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

**Requires:** none.

Branching, commits, the CHANGELOG rule, the definition of done and the PR workflow are in [executing.md](executing.md). This file adds only what is specific to Phase 1.

**Inputs to read first:**

- Archive commits, as reference only (`git show <sha>`): 25175891, 796a06e9, 84ebd4ed, 81fa735e, 5ff803ba, 584bc78d, 8d026744, 1deffc62, 43952bb4, c31b6fb2 (geometry hunks of `io/nwb/_environment.py` only), 251c580e. Never cherry-pick: each one is entangled with archive-only infrastructure or test scaffolding.
- [src/neurospatial/decoding/posterior.py:169-226, 309, 414-417, 444-454, 1652-1656](../../../../src/neurospatial/decoding/posterior.py) — the prior clip, the uniform fallback and a validator comment that describes the clip.
- [src/neurospatial/encoding/phase_precession.py:158, 187-205](../../../../src/neurospatial/encoding/phase_precession.py) — the transfer-function `filtfilt` theta band-pass.
- [src/neurospatial/stats/circular.py:833, 1012, 1331](../../../../src/neurospatial/stats/circular.py) — three `1 - cdf` p-values.
- [src/neurospatial/layout/engines/masked_grid.py:61-108](../../../../src/neurospatial/layout/engines/masked_grid.py) — `build` with no edge validation.
- [src/neurospatial/decoding/estimator.py:278-490](../../../../src/neurospatial/decoding/estimator.py) — `fit` captures `unit_ids` (`:351`, `:399-402`); `predict` (`:436-439`) and `predict_summary` (`:485-488`) pass spikes positionally; `score` (`:583`) goes through `predict`.
- [src/neurospatial/encoding/_spikes.py:159-251](../../../../src/neurospatial/encoding/_spikes.py) — `_looks_like_spike_group` and `as_spike_trains_with_ids`, already used by `compute_spatial_rates` ([spatial.py:3408-3419](../../../../src/neurospatial/encoding/spatial.py)).
- [src/neurospatial/io/nwb/_environment.py:77, 201-210, 229-246, 262-340, 512-520, 571-583, 721-838](../../../../src/neurospatial/io/nwb/_environment.py) — the NWB environment writer, reader and `_ReconstructedLayout`.
- [.github/workflows/test_xarray.yml:36-42](../../../../.github/workflows/test_xarray.yml), [test_nwb.yml](../../../../.github/workflows/test_nwb.yml), [test_pynapple.yml](../../../../.github/workflows/test_pynapple.yml). The default `test.yml` job installs only `--extra dev`. A test that does `importorskip("xarray")` outside the listed files is silently skipped in CI.
- The `on:` blocks of every file in [.github/workflows/](../../../../.github/workflows/) (Task 1). Seven filter on `branches: [main]`; `release.yml` (tags) and `publish.yml` (releases, manual dispatch) have no branch filter.
- [pytest.ini](../../../../pytest.ini) and [pyproject.toml:150-162](../../../../pyproject.toml) (Tasks 1 and 1b). `pytest.ini` exists, so pytest ignores `[tool.pytest.ini_options]` in `pyproject.toml`, including its `timeout = 300`. `pytest-timeout` is not installed either (it is in neither `pyproject.toml` nor `uv.lock`). **No test has a timeout.**
- [src/neurospatial/_results.py:49-103](../../../../src/neurospatial/_results.py) — `resolve_unit_ids`, which accepts duplicate labels on `main`. [encoding/spike_trains.py:131-146](../../../../src/neurospatial/encoding/spike_trains.py) — `SpikeTrains`' own duplicate check (Task 7).

**Contracts referenced:**

- [Input conventions → Population identity](shared-contracts.md#input-conventions). The decoder aligns by label **only when both sides carry caller-supplied labels**; otherwise it pairs by position and requires equal unit counts. A label mismatch raises and lists the missing and unexpected labels. Task 6 implements this. Task 7 implements "labels are never overridden" for the four population encoders (a `unit_ids=` that differs from a labelled input's index raises) and "duplicate `unit_ids` raise `ValueError` at the point they're supplied", in the one resolver every population encoder already calls.
- [Error-message contract](shared-contracts.md#error-message-contract). Every *new* message in this phase follows it (what/why plus a `Fix:` line), so [Phase 4a](phase-4a-errors.md) does not have to rewrite it. Phase 4a adds the `_format_error` helper and `NeurospatialError`; this phase writes the message text directly. Older messages on `main` use `WHY:` / `HOW:` lines (for example `_results.py:401-404` and `SpikeTrains`). Leave them as they are: the two styles coexist until Phase 4a relabels the old ones.

**Designs referenced:** none.

## Tasks

Commits, regression-test-first and the per-commit CHANGELOG bullet follow [executing.md → While you work](executing.md#while-you-work). Phase-specific notes:

- Run a new regression test against the unmodified code with `uv run pytest <nodeid> -n 0`, and record the failure (assertion, exception or hang) in the commit body. Do not use `git stash` for this.
- A guard test that is *expected* to pass before the fix is labelled "guard" in the Validation slice.
- Tasks 1 and 1b come first, in that order. Until Task 1b lands, a default-suite run hangs indefinitely (see Task 1b), so do not run the full suite before it.

0. **Commit the external review.** *(Done in `38552cae` before Phase 1 started; skip.)* Add `docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md` as-is (`docs: add 2026-10-01 repository and mathematical review`). `mkdocs.yml:13` excludes `reviews/`, and lychee checks only the built site, so the file's absolute `file:///` links can't break CI. Do not edit it.

1. **Run CI on this branch (`ci: run workflows on feat/researcher-first`).** This is the first code commit, so every later commit and PR in this plan gets CI. In each file below, add `- feat/researcher-first` under **both** the `push:` and the `pull_request:` `branches:` lists (each list currently holds only `- main`):

   | File | `push` branches | `pull_request` branches |
   | --- | --- | --- |
   | `.github/workflows/test.yml` | :5-6 | :8-9 |
   | `.github/workflows/test_docs.yml` | :5-6 | :8-9 |
   | `.github/workflows/test_notebooks.yml` | :5-6 | :8-9 |
   | `.github/workflows/docs.yml` | :5-6 | :8-9 |
   | `.github/workflows/test_nwb.yml` | :10-11 | :13-14 |
   | `.github/workflows/test_pynapple.yml` | :10-11 | :13-14 |
   | `.github/workflows/test_xarray.yml` | :9-10 | :12-13 |

   - Leave `release.yml` (`tags: v*.*.*`) and `publish.yml` (`release`, `workflow_dispatch`) unchanged: they have no branch filter.
   - Leave the deploy gates in `docs.yml:122` and `:128` (`github.ref == 'refs/heads/main'`) unchanged, so this branch builds the docs but never deploys them.
   - **Run `slow` tests in CI.** `pytest.ini:6` deselects them by default (`-m "not slow and not napari"`), and no workflow runs them today, so a slow-marked test is never enforced. Because nothing ran them, some are stale.
     - **First, run them locally** (after `uv sync --all-extras`, with `ffmpeg` on `PATH`): `uv run pytest -m "slow and not napari" -n 4`. On this branch (`d6f92ace`, src identical to `da631a47`) that gave **10 failed, 93 passed, 3 skipped**. A rerun of the failing files gave 9 failed: `test_connected_component_performance_scipy_vs_graph` is a timing ratio that failed only under load. The 9 reproducible failures:

       | Test | Failure | Cause |
       | --- | --- | --- |
       | `tests/benchmarks/test_performance.py::TestSpatialRateComputationPerformance::test_spatial_rate_{diffusion_kde_small,diffusion_kde_medium,binned_small}` | `ValueError: spike_times must be monotonically non-decreasing` | the test passes unsorted spikes |
       | `tests/benchmarks/test_performance.py::TestMetricComputationPerformance::test_detect_place_fields` | `AttributeError: 'Environment' object has no attribute 'shape'` (`spatial.py:4355`) | stale call signature |
       | `tests/animation/test_video_overlay.py::TestNapariVideoOverlay::{test_video_layer_added,test_video_spatial_alignment,test_video_temporal_sync}` | "Video layer not found" | needs a napari viewer; skipped in CI (no napari or cv2 under `--extra dev`) but runs locally under `--all-extras` |
       | `tests/animation/test_benchmark_napari_playback.py::TestScriptIntegration::test_script_with_all_overlays` | the script subprocess fails | needs napari, as above |
       | `tests/simulation/test_integration.py::TestPlaceFieldDetectionAccuracy::test_place_field_detection_accuracy` | "Only 0 of 5 true centers matched within 8.0 cm" | its tolerance is `2 * mean(env.bin_sizes)`, a bin *area* used as a length; the simulator's default field width has the same bug, which [Phase 2a](phase-2a-main-bugs-geometry-io.md) fixes |

     - **Then, in a separate `test:` commit before the CI commit,** make the selection green. Fix a stale test when the fix is mechanical: sort the spikes, or update the call. Give the two napari-dependent tests the `napari` marker they lack, so `-m "slow and not napari"` deselects them as it does every other GUI test. Otherwise mark the test `@pytest.mark.xfail(strict=True, reason="<cause>; fixed in <phase>")`. `test_place_field_detection_accuracy` gets that xfail, naming Phase 2a; with `strict=True`, a pass after Phase 2a's fix fails the run, which forces the marker's removal. Do not xfail a timing test that fails only under load. Rerun it with `-n 0` and record the result. List every changed test and its reason in the PR description.
     - Add a job `slow-tests` to `.github/workflows/test.yml`. It is Ubuntu-only and uses Python 3.13 (`.python-version`). It installs with `uv sync --extra dev` and installs `ffmpeg` (`sudo apt-get install -y ffmpeg`): the parallel video-export tests are slow-marked and skip without it. It sets `MPLBACKEND: Agg`, and its run step is `uv run pytest -m "slow and not napari" -n 4`. It triggers on the same branches as the other jobs.
     - From then on, any slow-marked test in this plan runs on every PR.
   - **Verify** on the Phase 1 PR: `gh pr checks <number>` lists all seven workflows' checks plus `slow-tests`, all green. Record the check URLs in the PR description. As [executing.md → Opening the PR](executing.md#opening-the-pr) explains, `gh run list --branch feat/researcher-first` does not show PR runs.

**1b. Un-hang the parallel-render test** (`tests/animation/test_video_backend.py:494-531`, test only; `test: stop parallel-render test from hanging`). Do this immediately after Task 1, before any full-suite run.

   - **Bug.** `parallel_render_frames` now uses `executor.submit` and `concurrent.futures`, but the test (in the default selection: no `slow` or `napari` marker) still mocks `ProcessPoolExecutor.map`. On `main` it waits forever. There is no timeout to stop it: `pyproject.toml:162`'s `timeout = 300` is ignored because `pytest.ini` takes precedence, and `pytest-timeout` is not installed. A local default-suite run, and CI's `test` job, therefore hang until killed (on GitHub Actions, the 6-hour job limit).
   - **Fix.** Port 43952bb4: `executor.submit` returns a completed `concurrent.futures.Future`, and the test asserts `submit.call_count == 3` instead of the `map` contract that 26fc8dac removed. Each submitted task still carries `env`, `fields` and `start_frame_idx`.
   - Adding `pytest-timeout` is out of scope; this task removes the hang itself.

2. **Zero prior excludes bins (`posterior.py`).** Re-implement 25175891 against `main`:
   - At `:414-417`, replace the clip with `prior_support = prior_arr > 0` and `with np.errstate(divide="ignore"): log_prior = np.log(prior_arr)`. Initialize `prior_support = None` before the `if prior is not None` block.
   - `_normalize_block` (`:169`) gains `prior_support: NDArray[np.bool_] | None = None`. In the `"uniform"` branch (`:221-224`), when `prior_support` is not None, distribute uniformly over the supported bins of each degenerate row: `supported = np.broadcast_to(prior_support, ll_block.shape)[degenerate_mask]`, then `np.divide(supported, supported.sum(-1, keepdims=True), out=np.full(supported.shape, np.nan), where=...>0)`. A row with no supported bins becomes NaN.
   - Pass `prior_support` at `:444` and `:448-454`. In the chunked loop, slice it by `[start:stop]` only when it is 2-D.
   - Update the docstring: `prior` (`:255-257`, "exact zeros exclude bins; positive probabilities are not floored"), the `"uniform"` bullet (`:266-268`), and the Notes code block (`:309`). Rewrite the stale validator comment at `:1652-1656`, which describes the removed clip.
   - Rewrite `test_zero_prior_bin_is_negligible` (`tests/decoding/test_posterior.py:1454`) as `test_zero_prior_bin_is_excluded`, asserting `== 0.0`. The old test encodes the bug.

3. **Stable theta band-pass (`phase_precession.py`).** Following 796a06e9:
   - Import `sosfiltfilt` at `:158`.
   - Replace `:187-205` with `sos = butter(N=4, Wn=(low, high), btype="bandpass", fs=sampling_rate, output="sos")`, `padlen = 3 * (2 * len(sos) + 1)` (still 27, so the too-short error threshold is unchanged), and `filtered = sosfiltfilt(sos, lfp, padlen=padlen)`.
   - Update the comment at `:168` and the error text at `:197-203` (`filtfilt` → `sosfiltfilt`).

4. **Tail p-values via survival functions (`stats/circular.py`).** 84ebd4ed fixed only `:833`. The same cancellation is present at two more sites on `main`, so fix all three in one commit:
   - `:833` → `chi2.sf(chi2_stat, df=2)`.
   - `:1012` (`circular_circular_correlation`) → `2 * stats.norm.sf(np.abs(ts))`.
   - `:1331` (`_wald_test_magnitude`, reached through `circular_basis_metrics`) → `chi2.sf(wald_stat, df=2)`.

5. **Masked-grid edge validation (`masked_grid.py`).** Insert this before `:101`. It is re-implemented from 81fa735e, but **do not port that commit's tolerance verbatim**. 81fa735e accepts `widths` within `atol = 16·eps·max|edges|`, a bound that grows with the coordinate magnitude whether or not the float spacing there can resolve a bin. It therefore accepts `[1e15, 1e15+1, 1e15+3]` (widths `[1, 2]`, verified). The check below instead bounds each edge's representation error by the actual float spacing at the largest coordinate, `ulp = np.spacing(max|edges|)`:
   - **(a) Precision.** If `4·ulp > 1e-4·w0`, the grid cannot be represented to 1 part in 10⁴ of a bin width. Raise, and tell the caller to subtract an origin offset.
   - **(b) Uniformity.** Every width must satisfy `|w − w0| <= 1e-7·w0 + 4·ulp`. Each edge is within ½ ulp of its exact value, so a width is within 1 ulp and two widths differ by at most 2 ulp; 4 ulp leaves 2× headroom for an edge generator (such as `linspace`) that adds one rounding step.

   Measured with this snippet (probe on `main`, `from_samples` then `subset` of half the bins): at offset 1e7 with bin 0.01, `max|w − w0| = 1.86e-9` against a tolerance of `8.45e-9`, with `4·ulp / (1e-4·w0) = 0.0075`. At offset 1e9 with bin 1.0 the values are `1.19e-7`, `5.77e-7` and `0.0048`. Both are accepted, and `subset` succeeds.

   ```python
   if len(grid_edges) != active_mask.ndim or len(grid_edges) == 0:
       raise ValueError(
           f"grid_edges has {len(grid_edges)} edge arrays but active_mask is "
           f"{active_mask.ndim}-D; one edge array per mask axis is required.\n"
           "Fix: pass grid_edges=(edges_axis0, edges_axis1, ...) matching active_mask.ndim."
       )
   grid_edges = tuple(np.asarray(e, dtype=np.float64) for e in grid_edges)
   for axis, edges in enumerate(grid_edges):
       if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
           raise ValueError(
               f"grid_edges[{axis}] must be a finite 1-D array with >= 2 edges, got "
               f"shape {edges.shape}.\nFix: pass e.g. np.linspace(start, stop, n_bins + 1)."
           )
       widths = np.diff(edges)
       if np.any(widths <= 0):
           raise ValueError(
               f"grid_edges[{axis}] must be strictly increasing; smallest width is "
               f"{widths.min():.6g}.\nFix: sort the edges and drop duplicates."
           )
       w0 = widths[0]
       # Float spacing at the largest coordinate bounds each edge's representation error.
       largest = float(np.max(np.abs(edges)))
       ulp = np.spacing(largest)
       if 4 * ulp > 1e-4 * w0:
           raise ValueError(
               f"grid_edges[{axis}] reaches {largest:.6g}, where float64 resolves only "
               f"{ulp:.3g}; a bin width of {w0:.6g} cannot be represented to 1 part in "
               "10^4 there, so bin widths and volumes would be wrong.\n"
               "Fix: subtract an origin offset before building the environment, e.g. "
               "positions - positions.min(axis=0)."
           )
       deviation = float(np.max(np.abs(widths - w0)))
       allowance = 1e-7 * w0 + 4 * ulp
       if deviation > allowance:
           raise ValueError(
               f"grid_edges[{axis}] must be uniformly spaced: widths differ from the "
               f"first ({w0:.10g}) by up to {deviation:.3g}, more than the {allowance:.3g} "
               "rounding allowance; cell volumes and diffusion face measures assume "
               "one width per axis.\n"
               "Fix: use np.linspace(start, stop, n_bins + 1) for each axis."
           )
   ```

   Docstrings: `masked_grid.py:76-78` and `factories.py:1209-1213` gain "finite, strictly increasing, and uniformly spaced along each axis; axes may differ". Remove the "tracked follow-up" caveats at `environment/fields.py:136-140` and `ops/diffusion.py:1291-1295`, replacing them with "`MaskedGridLayout.build` rejects nonuniform `grid_edges`".

6. **Decoder pairs spike trains with encoding models by the Population-identity rule (`estimator.py`).** The rule: align **by label only when both sides carry caller-supplied labels**. That means `fit` received a labelled group (its index becomes `unit_ids`) **and** the predict input is a labelled group. In every other case pair by position and require equal unit counts. The default `np.arange(n)` that `fit` stores for unlabelled input (`:399-402`) is **not** caller-supplied and never triggers label alignment.
   - **Record where the labels came from.** The `@dataclass(frozen=True)` at `:40` gains, after `unit_ids` (`:175`), the private init field `_unit_ids_supplied: bool = field(default=False, repr=False, compare=False)` (import `field` from `dataclasses`). In `fit`, keep the extracted ids before the `arange` fallback and return `replace(self, encoding_models=firing_rates, unit_ids=unit_ids, _unit_ids_supplied=extracted_ids is not None)`. [Phase 6b](phase-6b-argument-conventions.md) adds `fit(unit_ids=)`, which sets the flag too.
   - **Align.** Add the method below and call it in `predict` (`:439`) and `predict_summary` (`:488`) in place of the raw `spike_times`. `score` inherits it through `predict`. This does **not** port 5ff803ba's exact-sequence rejection: a reordered labelled input is aligned, not rejected.

   ```python
   def _align_to_fitted_units(self, spike_times: SpikeTrainsLike) -> list[NDArray[np.float64]]:
       """Pair each spike train with its encoding model.

       By label when both ``fit`` and this input carried caller-supplied labels;
       otherwise by position, which requires one train per fitted unit.
       """
       from collections import Counter

       from neurospatial.encoding import as_spike_trains_with_ids

       trains, input_ids = as_spike_trains_with_ids(spike_times)
       n_models = self._check_fitted().shape[0]
       if input_ids is not None:
           labels = np.asarray(input_ids).tolist()
           duplicated = [u for u, count in Counter(labels).items() if count > 1]
           if duplicated:
               raise ValueError(
                   f"The spike input repeats unit labels {duplicated}, so a label "
                   "cannot name one spike train.\n"
                   "Fix: pass each unit once (pynapple: check group.index)."
               )
           if self._unit_ids_supplied:
               fitted = np.asarray(self.unit_ids).tolist()
               row = {u: i for i, u in enumerate(labels)}
               fitted_set = set(fitted)
               missing = [u for u in fitted if u not in row]
               unexpected = [u for u in labels if u not in fitted_set]
               if missing or unexpected:
                   raise ValueError(
                       "Spike input unit labels do not match the decoder's fitted "
                       f"unit_ids (missing: {missing}, unexpected: {unexpected}).\n"
                       "Decoding would pair spike trains with the wrong encoding models.\n"
                       "Fix: pass spikes for exactly the fitted units (pynapple: "
                       "group[list(decoder.unit_ids)]), or refit on this input."
                   )
               return [trains[row[u]] for u in fitted]
       if len(trains) != n_models:
           raise ValueError(
               f"Got {len(trains)} spike trains but the decoder was fitted with "
               f"{n_models} units. Without caller-supplied labels on both the fit "
               "and the predict input, trains are paired with encoding models by "
               "position.\n"
               "Fix: pass one spike train per fitted unit in the order used for fit, "
               "or fit and predict with the same labelled TsGroup."
           )
       return trains
   ```

   Docstrings: the `unit_ids` field (`:110-113`; "introspection only" is no longer true) and a new sentence on the pairing rule; the `spike_times` parameter text of `fit`, `predict`, `predict_summary` and `score`; and the `Raises` sections of the last three.

7. **Spike-group input in directional, view and egocentric population rates, and labels that are never overridden.** This is a lighter version of 584bc78d: route the input instead of rejecting it, mirroring `spatial.py:3408-3419`. It also implements the contract rule [Labels are never overridden](shared-contracts.md#input-conventions). On `main`, `compute_spatial_rates` lets an explicit `unit_ids=` silently replace a labelled group's index: a group keyed `[10, 20]` with `unit_ids=[20, 10]` returns `unit_ids == [20, 10]`, so unit 10's spike train now carries label 20 (probe on `main`).
   - **One resolver.** `neurospatial._results.resolve_unit_ids(unit_ids, n_units, *, context="")` gains the keyword `input_ids=None`, the labels the spike input carries. When both `unit_ids` and `input_ids` are given, they must be identical (same length, same order, `np.array_equal`). Otherwise it raises:

     ```python
     if unit_ids is not None and input_ids is not None:
         given, carried = np.asarray(unit_ids), np.asarray(input_ids)
         if given.shape != carried.shape or not np.array_equal(given, carried):
             raise ValueError(
                 f"{context or 'This call'} got unit_ids={given.tolist()}, but the spike "
                 f"input is already labelled {carried.tolist()}.\n"
                 "Why: a labelled input names its own units; a different unit_ids would "
                 "attach another unit's label to each spike train.\n"
                 "Fix: drop unit_ids= to keep the input's labels, or relabel the input "
                 "itself before the call (pynapple: build the TsGroup with the labels "
                 "you want)."
             )
     unit_ids = unit_ids if unit_ids is not None else input_ids
     ```

     The existing 1-D and length checks then run on the result. Callers that pass no `input_ids` and no repeated label are unchanged.
   - **Duplicate labels raise where they're supplied** ([Population identity](shared-contracts.md#input-conventions); re-implements the intent of f362c8f8 inside this one resolver instead of a second `resolve_unique_unit_ids`). After the length check, reject repeated labels. Count them with a hash-based `Counter`, because `np.unique` cannot sort a mixed int/str object array:

     ```python
     duplicated = [label for label, count in Counter(resolved.tolist()).items() if count > 1]
     if duplicated:
         where = f" in {context}" if context else ""
         raise ValueError(
             f"unit_ids must be unique{where}: label(s) {duplicated} are repeated.\n"
             "Why: results index units by label, so a repeated label would name two "
             "spike trains.\n"
             "Fix: pass one distinct label per unit, or omit unit_ids= to number the "
             "units 0..n-1."
         )
     ```

     Every caller inherits the check: the four population encoders, `population_peri_event_histogram` (`events/alignment.py:461`), the `summary_table(unit_ids=)` relabel paths (`spatial.py:1415`, `directional.py:1082`, `view.py:556`, `egocentric.py:622`, `events/_core.py:292`) and `SpikeTrains` (`spike_trains.py:132`). On `main`, `compute_spatial_rates(..., unit_ids=[5, 5])` returns `unit_ids == [5, 5]` without error (probe).
     - **Old paths removed in the same commit.** `SpikeTrains`' own duplicate check (`spike_trains.py:134-146`) is now unreachable; delete it. Its tests (`tests/encoding/test_spike_trains.py:95, 109`) match `"unique"`, which the new message keeps. In Task 6's `_align_to_fitted_units`, replace the inline `Counter` block with `labels = resolve_unit_ids(None, len(trains), input_ids=input_ids, context="BayesianDecoder.predict").tolist()`, so there is one duplicate check. The `to_xarray` guard (`_results.py:388-404`) stays: a result built directly with repeated `unit_ids` never passes through the resolver.
     - When [Phase 6b](phase-6b-argument-conventions.md) adds `BayesianDecoder.fit(unit_ids=)`, it routes that argument through this resolver and so inherits the check; no later phase adds a second duplicate check.
   - **Encoders.** In `compute_directional_rates` (`directional.py:2022, 2044-2052`), `compute_view_rates` (`view.py:1588, 1620-1628`) and `compute_egocentric_rates` (`egocentric.py:1835, 1875-1883`), import `as_spike_trains_with_ids` and call `spike_times_list, extracted_unit_ids = as_spike_trains_with_ids(spike_times)`. Then call `resolve_unit_ids(unit_ids, n_units, input_ids=extracted_unit_ids, context=...)`.
   - In `compute_spatial_rates` (`spatial.py:3416-3419`), replace the `unit_ids if unit_ids is not None else extracted_unit_ids` override and its "An explicit `unit_ids=` always wins" comment with the same call.

   Update the `spike_times` and `unit_ids` docstring lines (`directional.py:1906`, `view.py:1409`, `egocentric.py:1639`, and the `compute_spatial_rates` equivalents). They should say that a pynapple `TsGroup` is accepted, that its index becomes `unit_ids`, and that a `unit_ids=` passed together with a labelled group must equal the group's index.

8. **`to_pynapple` pairs times with values (`io/pynapple.py`).** Following 8d026744:
   - Move `nap = _require_pynapple()` from `:178` to just before `:224`, so validation runs without pynapple.
   - After the length check (`:215-221`), raise on non-finite `times`. Also raise on non-increasing `times`, naming the first offending index pair, with `Fix: order = np.argsort(times, kind="stable"); to_pynapple(times[order], values[order])`.
   - For 2-D `values`, raise when `columns` is given and `len(columns) != values.shape[1]`.
   - Update the docstring (`:149-171`).

9. **Directional NetCDF export with `bandwidth=None` (`directional.py:1131-1135`).** Following 1deffc62 (reference only: take its `directional.py` hunk; its tests live in `test_encoding_directional.py`, which `test_xarray.yml` does not run): emit `attrs["bandwidth"]` only when it is not None, the same rule `spatial.py:1703-1710` already uses. Update the docstring at `:1115`. Put the tests in a new `tests/encoding/test_directional_xarray_interop.py` and add that path to the `pytest` list in `.github/workflows/test_xarray.yml:39-42`. Otherwise CI never runs them.

10. **NWB environment geometry round trip (`io/nwb/_environment.py`).** This takes the geometry hunks of c31b6fb2 and leaves out `lineage_id`, `revision_id` and `_restore_identity`.
   - **Edge orientation (`:201-205`).** For each `(u, v, data)`, swap to `(v, u)` only when `data["vector"]` matches `centers[u] - centers[v]` and not `centers[v] - centers[u]` (both by `np.allclose(atol=1e-9)`). Edges whose vector matches neither (for example, polar wrap-around edges) keep enumeration order. This makes the reader's `pos_v - pos_u` (`_reconstruct_graph`) reproduce the original vector.
   - **Bin measures.** Add `COL_BIN_SIZES = "bin_sizes"` after `:101`. Write `env.bin_sizes`, padded to `n_rows` like `:262-272`, as a `VectorData` after `:336-340`. Read it back when `COL_BIN_SIZES in scratch_data.colnames`.
   - **1-D grid geometry.** When `_extract_grid_data(env)` is None and `env.grid_edges` is non-empty, store `grid_edges` and `grid_shape` lists in the metadata JSON (`:229-246`), and read them with `metadata.get`.
   - **Reconstruction.** Thread `bin_sizes`, `grid_edges` and `grid_shape` through `_reconstruct_environment` (`:571-583` → `:721-727`) into `_ReconstructedLayout.__init__` (`:761`). That sets `self.grid_edges` and `self.grid_shape` instead of None (`:782-783`), and `bin_sizes()` (`:814`) returns the stored array when present, or the KDTree estimate otherwise.
   - **Schema version.** Bump `ENVIRONMENT_SCHEMA_VERSION` (`:77`) to `"1.1"`. Make the mismatch warning at `:512-520` fire only for versions outside `{"1.0", "1.1"}`. A 1.0 file still reads, with estimated measures.

11. *(Moved to Task 1b, so the hang is removed before any full-suite run. The number is kept so that references to Tasks 12 and 13 stay valid.)*

12. **`DecodingResult` owns its data (`decoding/_result.py`).** This is a lighter version of 251c580e. The `@cached_property` accessors (`:106, :136, :155, :179`) go stale if the posterior changes after construction, and on `main` there are two ways that happens:
    - the caller edits the array they passed in, in place;
    - the caller reassigns `result.posterior`.

    A read-only *view* is not enough, because it shares memory with the caller's writable array. A probe of that design shows the failure: after `src[0] = [0.1, 0.9, 0, 0]`, the cached `map_estimate` stays `[0]` while `posterior.argmax(1)` is `[1]`. The result must own its arrays:
    - **Frozen.** Change the decorator at `:25` to `@dataclass(frozen=True, repr=False)`. `@cached_property` still works, because it writes the instance `__dict__` directly. `dataclasses.replace` still works, because it calls `__init__`. Assigning any field raises `dataclasses.FrozenInstanceError`.
    - **The public constructor always copies.** A `base is None` check cannot prove exclusive ownership: a writable view created *before* the array was marked read-only still writes into it. So every array passed to `DecodingResult(...)` is copied and made read-only:

      ```python
      def _read_only_copy(array: Any, dtype: Any = None) -> NDArray[Any]:
          """Return a read-only copy that no caller-held view can alias."""
          owned = np.array(array, dtype=dtype, copy=True)
          owned.flags.writeable = False
          return owned

      def __post_init__(self) -> None:
          object.__setattr__(self, "posterior", _read_only_copy(self.posterior))
          if self.times is not None:
              object.__setattr__(self, "times", _read_only_copy(self.times, np.float64))
      ```

    - **No copy on the trusted internal path.** `decode_position` (`posterior.py:715`) is the only producer on `main`; `decode_session`, `BayesianDecoder.predict` and `score` all reach it. It allocates the posterior itself, and no other reference to it escapes. So it builds the result through a private classmethod that skips `__post_init__`'s posterior copy. `times` is still copied, because `np.asarray(times)` can be the caller's array, and that costs one float per time bin.

      ```python
      @classmethod
      def _from_owned_posterior(cls, posterior: NDArray[Any], **fields: Any) -> "DecodingResult":
          """Build from a posterior this module just allocated (no aliases exist).

          Internal use only: the caller guarantees no other reference to ``posterior``.
          """
          posterior.flags.writeable = False
          obj = cls.__new__(cls)
          object.__setattr__(obj, "posterior", posterior)
          for name, value in fields.items():
              object.__setattr__(obj, name, value)
          if obj.times is not None:
              object.__setattr__(obj, "times", _read_only_copy(obj.times, np.float64))
          return obj
      ```

      `fields` must cover every other dataclass field. A test asserts that `set(dataclasses.fields)` equals the keys passed, so adding a field without updating `decode_position` fails loudly.
    - **Docstring.** Rewrite the Notes paragraph at `:79-81`. It should say that the result is frozen, that `posterior` and `times` are read-only copies the result owns (the constructor always copies, so later edits to the caller's arrays or views cannot reach the result), and that a modified result comes from `dataclasses.replace(result, posterior=new)`, which copies `new`.

13. **User-facing documentation check (no separate commit unless something is missing).** Per [executing.md → CHANGELOG](executing.md#while-you-work), each of Tasks 2–10 and 12 already appended its own bullet under `## [Unreleased]` → `### Fixed` in `CHANGELOG.md` (Task 2 creates the section) and shipped its docstring edits. Tasks 1 and 1b change CI and tests only and get no bullet. Before opening the PR, check that:
    - there is one bullet per Task 2–10 and 12, each stating the user-visible symptom and the new behavior, for example "`decode_position` with a zero in `prior` could return the excluded bin as the MAP; zero-prior bins now get exactly zero posterior";
    - these behavior changes are flagged as such:
      - `BayesianDecoder.predict` now aligns by label when both the fit and the predict input are labelled, and raises on a label mismatch or a unit-count mismatch;
      - the population encoders raise when `unit_ids=` disagrees with a labelled group's index (on `main`, `compute_spatial_rates` silently relabelled);
      - every function that resolves `unit_ids` (the population encoders, `population_peri_event_histogram`, `summary_table(unit_ids=)`) raises on a repeated label (on `main`, `compute_spatial_rates(..., unit_ids=[5, 5])` was accepted);
      - `DecodingResult` is frozen and copies a caller's posterior;
      - `to_pynapple` now raises on unsorted times.

    A missing bullet is added in a `docs: complete phase 1 changelog` commit.

## Deliberately not in this phase

- **Recording-gap handling** (archive commits 2eeca395, aad59e74, a0ff995a, d373cc18, c95789c8, 0d1a89f4, 7b862673, f038a558, 19eaa2b3, e9eee2c5, 4505b564) belongs to Phase 3 ([3a](phase-3a-time-windows-core.md) first). They are entangled with `_temporal.py`.
- **Bugs found on `main` but never fixed on the archive branch** (W-maze linearization, polar plot mirroring, NWB `conversion`/`offset` and unit names, simulated field width, `ops/calculus.py` and `ops/basis.py` scaling, heading interpolation, and the behavior/PETH items) belong to [Phase 2a](phase-2a-main-bugs-geometry-io.md) and [Phase 2b](phase-2b-main-bugs-operators-behavior.md).
- **Lineage/identity hunks of c31b6fb2** (`lineage_id`, `revision_id`, `_restore_identity`, `_thawed`) are a non-goal (overview).
- **b4e69e1a (population PETH memory)** is performance work, which the overview lists as a non-goal. It is not a correctness bug.
- **9bf0051b (silently ignored kwargs)** is an argument-convention change ([Phase 6b](phase-6b-argument-conventions.md)'s territory), not a scientific fix. If it is ported later, keep the `width` → `distance_tolerance` alias.
- **1dae9ff1 / 89816f2d (deprecation-notice text).** These are moot under settled decision 1 (no backwards compatibility): the deprecated forms are removed rather than re-worded.
- **208932ae (lazy-export typing)** belongs with [Phase 6a](phase-6a-namespaces-snapshot.md)'s namespace work and snapshot test.
- **Interop CI matrices (NWB, JAX, xarray version ranges) and `scripts/test_doc_snippets.py`.** The existing `test_nwb`, `test_pynapple` and `test_xarray` jobs already run this phase's tests; executable docs are [Phase 4b](phase-4b-docs-that-run.md). Version-matrix CI is an overview Non-Goal.
- **f362c8f8's separate `resolve_unique_unit_ids` helper and its `WHY:`/`HOW:` message.** Task 7 puts the duplicate check in `resolve_unit_ids` itself.
- **`BayesianDecoder.fit(unit_ids=)`** is [Phase 6b](phase-6b-argument-conventions.md). Task 6 records only whether `fit` received a labelled group.
- **36cdae2e** is not applicable: `tests/nwb/test_fields_pooled.py` on `main` has no such test, and `tests/test_session_nwb_roundtrip_contract.py` does not exist.
- **Making reloaded non-grid environments smoothable.** `env.smooth` and `compute_spatial_rate(method="binned" | "diffusion_kde")` raise `NotImplementedError` for `_ReconstructedLayout` on `main` and remain so. Task 10 fixes stored geometry only. This is an overview Non-Goal.
- **Drop:** 40b626a9, 1190a74c, d5ead759 (per the triage).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_posterior.py::test_zero_prior_bin_is_excluded` | `posterior[0, 2] == 0.0` for prior `[0.5, 0.5, 0]` (`main`: ~1e-10) |
| `test_posterior.py::test_zero_prior_overrides_large_likelihood[time_chunk∈{None,1}, dtype∈{f32,f64}, 1-D/2-D prior]` | ll `[[0,100],[0,1000]]`, prior `[1,0]` → posterior exactly `[[1,0],[1,0]]` |
| `test_posterior.py::test_decode_position_respects_zero_prior` | `decode_position(env_2bin, [[10]], [[1,100]], 0.1, prior=[1,0])`: posterior `[[1,0]]` and `map_position == bin_centers[0]` (`main`: `[[1.99e-6, 0.999998]]`, MAP 1.5) |
| `test_posterior.py::test_small_positive_prior_not_floored` | prior `[1e-100, 1]`, flat ll → posterior equals prior at rtol 1e-12 (`main`: `[1e-10, 1]`) |
| `test_posterior.py::test_degenerate_uniform_respects_prior_support` | all-`-inf` row, prior `[1,0,3]` → `[0.5, 0, 0.5]` (`main`: thirds); all-zero prior → NaN |
| `test_theta_phase.py::test_theta_phase_accurate_at_high_sampling_rates[2000,5000,10000,30000]` | 8 Hz sine: all finite; median abs phase error on t∈(1,4) s < 0.01 rad (`main`: 1.557 / 1.553 rad, then all-NaN at ≥10 kHz) |
| `test_circular_metrics.py::test_circular_linear_pvalue_keeps_tail[100,1000]` | `pval == exp(-n/2)` at rel 1e-12 (`main`: 0.0) |
| `test_circular_metrics.py::test_circular_circular_pvalue_positive` | von Mises x (n=2000) against x + N(0, 0.05): `0 < pval < 1e-20` (`main`: 0.0) |
| `test_circular_metrics.py::test_wald_pvalue_keeps_tail` | `circular_basis_metrics(3, 0, 0.01*I)[2] == exp(-450)` at rel 1e-9 (`main`: 0.0) |
| `test_masked_grid_layout.py::test_rejects_invalid_edges[...]` | edges `[0,1,10]`, `[0,1,2.001]`, `[2,1,0]`, `[0,0,1]`, NaN, inf, 2-D, 1 edge → `ValueError` naming `grid_edges[0]` (`main`: accepted with bin_sizes `[1,1]`, `[-1,-1]`, `[0,0]`) |
| `test_masked_grid_layout.py::test_rejects_unrepresentable_edges` | `[1e15, 1e15+1, 1e15+3]` (widths `[1, 2]`) → the precision `ValueError`, with `"origin offset"` on its `Fix:` line: `4·ulp = 0.5 > 1e-4·w0 = 1e-4`. Both `main` and 81fa735e's `16·eps·max\|edges\|` tolerance accept it (verified) |
| `test_masked_grid_layout.py::test_rejects_nonuniform_at_large_offset` | at offset 1e7, edges `1e7 + [0, 0.01, 0.0201]` (widths `[0.01, 0.0101]`) → the uniformity `ValueError`: `max\|w − w0\| = 1e-4 > 8.45e-9`. A single width off by `+1e-8` also raises (`1.12e-8 > 8.45e-9`); 81fa735e's tolerance accepted deviations up to `3.55e-8` there (verified at `+1e-8` and `+2e-8`) |
| `test_masked_grid_layout.py::test_accepts_fine_bins_at_large_offsets[(1e7,0.01),(1e9,1.0)]` (guard) | `from_samples` then `subset` of half the bins succeeds with `n_bins == keep.sum()` (probe: 1014 of 1014 for both). Measured margins: `max\|w − w0\|` is `1.86e-9` against a tolerance of `8.45e-9` at 1e7, and `1.19e-7` against `5.77e-7` at 1e9 |
| `test_masked_grid_layout.py::test_accepts_uniform_anisotropic_edges` (guard) | axes spaced 0.1 and 3.0 → `bin_sizes == 0.3` |
| `test_estimator.py::test_predict_aligns_reordered_labels` | fit on a spike-group double with ids `[10..17]`, predict on the same group with reversed index → posterior equals the in-order predict (`assert_allclose`, atol 1e-12). Before the fix the input is paired positionally, so they differ |
| `test_estimator.py::test_predict_label_mismatch_lists_labels` | predict ids `[11..18]` after fitting `[10..17]` → `ValueError` whose message contains `missing: [10]` and `unexpected: [18]`; same for `predict_summary` and `score` |
| `test_estimator.py::test_unlabelled_fit_pairs_labelled_input_by_position` (guard) | fit on a plain list of 3 trains; predict on a spike-group double with keys `3, 7, 9` wrapping the same 3 trains → posterior equals the list-input predict (atol 1e-12). Default `arange` labels never trigger label alignment (verified on `main`: equal) |
| `test_estimator.py::test_labelled_fit_pairs_plain_input_by_position` (guard) | fit on the group with ids `[10..17]`; predict on a plain list of the same 8 trains in index order → posterior equals the group predict |
| `test_estimator.py::test_positional_count_mismatch_raises` | fit on 3 plain trains, predict on 2 → `ValueError` naming `2` and `3` with a `Fix:` line (before the fix: a deep "Neuron-count mismatch … Poisson likelihood" message with no fix) |
| `test_estimator.py::test_duplicate_input_labels_raise` | predict on a group whose index is `[10, 10, 12, …]` → `ValueError` listing `[10]` |
| `test_estimator.py::test_predict_plain_arrays_stay_positional` (guard) | list-of-arrays predict equals the pre-change result |
| `test_encoding_{directional,view,egocentric}.py::test_rates_accept_spike_group` | group ids `[101, 202]` → `firing_rates.shape[0] == 2`, `unit_ids == [101, 202]`, rates equal the list-of-arrays call (`main`, directional: shape `(1, 60)`, ids `[0]`) |
| `tests/encoding/test_unit_identity.py::test_unit_ids_must_match_group_labels` (parametrized over `compute_{spatial,directional,view,egocentric}_rates`) | group keyed `[10, 20]` with `unit_ids=[20, 10]` → `ValueError` whose message contains both `[20, 10]` and `[10, 20]`, plus a `Fix:` line. `unit_ids=[10, 20]` (identical) is accepted. A plain list with `unit_ids=[20, 10]` gives `unit_ids == [20, 10]`. `main`, spatial: returns `unit_ids == [20, 10]` without error, so unit 10's 30 spikes carry label 20 |
| `tests/encoding/test_unit_identity.py::test_resolve_unit_ids_input_labels` | `resolve_unit_ids([20, 10], 2, input_ids=[10, 20])` raises; `(None, input_ids=[10, 20])` → `[10, 20]`; `([3, 4], input_ids=None)` → `[3, 4]`; `(["10", "20"], input_ids=[10, 20])` raises, because the labels differ by type |
| `tests/encoding/test_unit_identity.py::test_resolve_unit_ids_rejects_duplicates` | `resolve_unit_ids([7, 7, 9], 3)` → `ValueError` containing `"unique"`, `[7]` and a `Fix:` line (`main`: returns `[7, 7, 9]`); mixed `[1, "a", 1]` → the same `ValueError` listing `[1]`, not a `TypeError`; `(None, 3, input_ids=[4, 4, 5])` → raises listing `[4]` |
| `tests/encoding/test_unit_identity.py::test_population_calls_reject_duplicate_unit_ids` (parametrized over `compute_{spatial,directional,view,egocentric}_rates` and `population_peri_event_histogram`) | `unit_ids=[5, 5]` on two plain trains → `ValueError` listing `[5]` (`main`, spatial: returns `unit_ids == [5, 5]`, probe) |
| `tests/encoding/test_spike_trains.py::test_duplicate_unit_ids_raise`, `::test_mixed_type_duplicate_unit_ids_raise` (guard, existing) | still raise `ValueError` matching `"unique"` after `SpikeTrains`' own check is deleted |
| `test_pynapple.py::test_to_pynapple_rejects_unsorted_times` (no pynapple needed) | `to_pynapple([0,2,1], [10,20,30])` → `ValueError` naming indices 1 and 2 (`main`: `t=[0,1,2]`, `d=[10,20,30]`) |
| `test_pynapple.py::test_to_pynapple_rejects_wrong_column_count` | 2 value columns, 3 labels → `ValueError` (`main`: columns silently become `[0, 1]`) |
| `test_directional_xarray_interop.py::test_none_bandwidth_netcdf_roundtrip` | `"bandwidth" not in attrs`; scipy-engine `to_netcdf` then `load_dataset` round-trips rates, occupancy and coords exactly (`main`: `TypeError`) |
| `tests/nwb/test_environment.py::test_graph_env_roundtrip_preserves_geometry` | Y-track (`from_graph`, bin 3, spacing 10, 82 bins): reloaded `bin_sizes` equal the original exactly (`main`: 2.94 → 1640.25), `grid_edges[0]` equal (`main`: None), 0 of 81 edge vectors flipped (`main`: 81) |
| `tests/nwb/test_environment.py::test_reads_schema_1_0_file` (guard) | a file whose metadata is rewritten to 1.0 without the new column reads with estimated `bin_sizes` and no warning |
| `test_video_backend.py::test_parallel_render_frames_partitioning` | completes in under 5 s; `submit.call_count == 3`; each submitted task carries `env`, `fields` and `start_frame_idx` (`main`: hangs indefinitely; a 90 s run with `-n 0` was killed with no result, and no pytest timeout is active) |
| `uv run pytest -m "slow and not napari" -n 4` (Task 1) | 0 failed. Every test changed to get there is a fix, a `napari` marker or an `xfail(strict=True, reason=...)`, each listed in the PR (`main`: 10 failed, 93 passed, 3 skipped; 9 reproducible) |
| `test_result.py::test_result_isolated_from_caller_array` | `src = [[0.9, 0.1, 0, 0]]`, `r = DecodingResult(posterior=src, env=env_4bin)`, read `r.map_estimate`, then `src[0] = [0.1, 0.9, 0, 0]`. Afterwards `r.posterior[0] == [0.9, 0.1, 0, 0]` and `r.map_estimate == r.posterior.argmax(1) == [0]`. The same holds when `src` is passed as a read-only view of a writeable array: it is copied (`not np.shares_memory`). With the view-only design the probe gave a cached MAP `[0]` and an argmax of `[1]`. Mutating a passed `times` array leaves `r.times` unchanged |
| `test_result.py::test_result_fields_cannot_change` | `r.posterior = np.zeros(...)` raises `dataclasses.FrozenInstanceError` (`main`: allowed); `r.posterior[0, 0] = 0.5` raises `ValueError` ("read-only"); the caller's `src` is still writeable |
| `test_result.py::test_result_copies_owning_array_with_prior_view` | `a = np.array([[0.9, 0.1, 0, 0]]); v = a[:]; a.flags.writeable = False; r = DecodingResult(posterior=a, env=env_4bin)`. Read `r.map_estimate`, then write `v[0] = [0.1, 0.9, 0, 0]`. `r.posterior[0]` is still `[0.9, 0.1, 0, 0]` and `r.map_estimate == [0]`. The public constructor copies even an owning, read-only array, because an earlier writable view can alias it |
| `test_result.py::test_decode_position_posterior_not_copied` (parametrized: default, `time_chunk=7`, `dtype=np.float32`) | Monkeypatch `DecodingResult._from_owned_posterior` to record its `posterior` argument. After `decode_position(...)`, `result.posterior is recorded` and `recorded.flags.writeable is False`, so the trusted path made no copy |
| `test_result.py::test_from_owned_posterior_covers_all_fields` | `decode_position`'s call to `_from_owned_posterior` passes exactly `{f.name for f in dataclasses.fields(DecodingResult)} - {'posterior'}`, so a newly added field can't be silently left unset |
| `test_result.py::test_replace_rebuilds_cache` | `r2 = dataclasses.replace(r, posterior=[[0, 0, 1, 0]])` gives `r2.map_estimate == [2]`, while `r.map_estimate` stays `[0]`; `r2.posterior` does not share memory with the argument |

Run the slice with `uv run pytest <files> -n 0`. Then satisfy [executing.md → Definition of done](executing.md#definition-of-done). Phase-specific extra checks:

- `uv run pytest tests/nwb -n 0` and `uv run pytest tests/encoding/test_directional_xarray_interop.py tests/decoding/test_result.py -n 0` under `uv sync --all-extras`. Check with `-rs` that nothing skips for a missing extra.
- `gh pr checks <number>` shows the `slow-tests` job and the `test_xarray` job (which now lists `test_directional_xarray_interop.py`) green.

## Fixtures

- **Spike-group double.** Move `_FakeTs` and `_FakeTsGroupMapping` from `tests/encoding/test_spatial_adapters.py:62-85` into a `make_spike_group(trains, index)` factory fixture in `tests/conftest.py`, and switch `test_spatial_adapters.py` to it. Tasks 6 and 7 use it. It is a `UserDict` whose iteration yields keys, like a real pynapple `TsGroup`, so no pynapple is needed.
- **Decoder data.** Reuse the module-scoped `sim` fixture (`tests/decoding/test_estimator.py:85`), which is 8 place cells over 40 s. Wrap its trains in the spike-group double with ids `10..17`. The 3-unit cases use its first 3 trains (keys `3, 7, 9` for the labelled input).
- **CI trigger (Task 1).** No test; verified by `gh pr checks <number>` on the Phase 1 PR, with the check URLs in its description.
- **NWB.** Build the Y-track graph in the test, as in the triage's `nwbrt.py`: nodes `(0,0), (0,100), (-50,150), (50,150)`, `edge_order=[(0,1),(1,2),(1,3)]`, `edge_spacing=10`, `bin_size=3`. Write it to `tmp_path` with `NWBHDF5IO`. It runs in `test_nwb.yml`.
- Everything else is synthesized inline: 2-bin `from_grid_mask` env, sines, von Mises samples, fixed seeds.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
