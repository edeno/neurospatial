# Overview — Scope, decisions, integration, risks

[← back to PLAN.md](PLAN.md)

## Background

The branch `feat/public-api-curation` (archived as tag `archive/public-api-curation-2026-10-01`, 172 commits on top of `main` at `da631a47`) made the package correct by construction, at heavy cost:

- About 82% of its 466k added lines were governance and process: plan inventories, census tests and frozen counts.
- 34 of 37 calls that work on `main` raise on that branch.
- `compute_spatial_rates` grew to 25 parameters with no Parameters section in its docstring.
- On continuous data its results were identical to `main`'s.

An October 2026 audit verified a specific list of real scientific bugs, some fixed on that branch and some present on both. This plan rebuilds only the valuable outcomes on a fresh branch from `main`.

The audit reports are evidence for the "why" behind each phase. Executors do not need them to carry out a phase, and the copies live only in a session scratchpad:

- `UX-REVIEW.md`
- `DESIGN-REVIEW.md`
- `branch-triage.md`
- `governance-map.md`
- `bugcheck/BUGCHECK.md`
- `prototype-inferred-support.md`

The repository copy of the external review is `docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md`. It is untracked: Phase 1 commits it.

**Reading archive-branch code:** use `git show archive/public-api-curation-2026-10-01:<path>` or `git show <sha>`. Never merge or cherry-pick wholesale from it. Its commits are entangled with `TemporalSupport`, `Session`, unit-identity scopes and lineage infrastructure, none of which this plan ports.

## Guiding principle

> A rigorously validated operation can still be unnecessarily difficult to use.

**Accuracy is non-negotiable.** The safety comes from *safe defaults*, not from required ceremony:

- Infer what can be inferred correctly, such as gaps from timestamps.
- Make what can't be inferred an explicit, optional, well-named argument.
- When a default could be wrong in a detectable way, detect it and warn with the fix.
- Never silently produce a plausible wrong number.

## Settled design decisions

These were decided by the maintainer on 2026-10-01. Do not reopen them in a phase.

1. **No backwards compatibility.** There are no current users. Replace in place, with no deprecation shims or aliases.
2. **Time windows are plain arrays.** There is no new public time type. Optional `epochs=` and `spike_window=` accept an `(n, 2)` array-like of `[start, stop)` rows or a pynapple `IntervalSet`. See [Time-window semantics](shared-contracts.md#time-window-semantics).
3. **Analyses combining streams use their overlap.** An analysis that uses spikes and position analyzes only time when both were recorded. Position-only analyses (behavior) use all tracked time. Spike-only analyses (PSTH) need no position.
4. **Raw arrays are the only input form.** Every analysis takes `func(env, spike_times, times, positions, ...)` (see [Input conventions](shared-contracts.md#input-conventions)). Simulators and NWB loaders return plain data holders whose attributes are passed explicitly. There is no `Session` bundle, and no dual-form overloads.
5. **Cell-type verdicts keep their current threshold criteria.** Phase 5 documents each criterion's measured bias prominently and adds shuffle significance as an opt-in.
   - **Place cells (revised 2026-10-01 after external review).** Field detection flags 0.5 Hz Poisson noise 20/20, so it is not a cell-type verdict. It keeps its fast behavior under the honest name `has_place_field()`.
   - `is_place_cell()` (free function and method) takes **no default criterion**. The caller passes `criterion="spatial_info"` or `criterion="shuffle"`.
   - Bugs in a classifier, as opposed to its default criterion, are fixed. One example is `label_cell_types` labelling random spikes "border" despite its docstring.
6. **Object-vector cells are allocentric by default, matching Høydal 2019.** `compute_object_vector_rate(s)` and `is_object_vector_cell` compute allocentric maps and need no headings. The egocentric map stays available as `compute_egocentric_rate`, with Wang 2018 and Alexander 2020 citations. The simulator also defaults to allocentric.
7. **The Session/NWB whole-session round trip is dropped.** It has no near-term user. NWB *environment geometry* persistence is a bug fix and is ported in Phase 1.
8. **Governance is replaced by one public-API snapshot test** ([API snapshot](shared-contracts.md#api-snapshot)). There are no inventories, frozen reference counts, tier metadata or plan-document tests.

## Current codebase integration points

These are on `main`, which is the base of this branch:

- `src/neurospatial/environment/trajectory.py:90` — `interval_valid_mask`, the single per-interval validity mask that both occupancy and spike counting apply. Phase 3 extends it; it does not replace it.
- `src/neurospatial/encoding/_binning.py:175-395` — the spike-binning kernel and `_resolve_interval_mask`, which already align the numerator (spikes) and denominator (occupancy). This is the model the other families adopt in Phase 3.
- `src/neurospatial/_exceptions.py` — the public exception module. Phase 4 adds a base class here.
- `src/neurospatial/encoding/spatial.py` — `compute_spatial_rate(s)`, which already have `max_gap=0.5` on `main`.
- `src/neurospatial/encoding/{directional,view,egocentric}.py`, `decoding/`, `behavior/segmentation.py`, `events/` — the families that bridge recording gaps on `main` (Phase 3).

## Goals

- Every advertised call in the README, quickstart, CLAUDE.md and the flagship docstrings runs as written, and CI enforces this.
- No silent wrong numbers: every bug verified in the October 2026 audit is fixed, each with a regression test that fails on `main`.
- Recording gaps are handled correctly in every analysis family by default.
- Errors state what went wrong, why, and the exact fix, in `str(exc)`.
- The public surface is small, discoverable, documented, and guarded by one snapshot test.

## Non-Goals

- Porting `TemporalSupport`, `Session`, unit-identity scopes, `EnvironmentRef` lineage, decoder provenance objects, or any inventory or census tooling from the archive branch.
- Changing classification defaults to shuffle-based (decision 5).
- New analysis features beyond those named in a phase (for example MRF-GAM work, or new cell types).
- Performance work, except where a phase's change would otherwise regress runtime (Phase 5's opt-in shuffle is allowed to be slow).
- Reloaded non-grid NWB environments lacking finite-volume geometry (cannot smooth/compute binned rates) — follow-up when a user needs it.
- CI interop matrices (NWB/JAX/xarray version cells) — follow-up after Phase 4.

## Metrics

- **Executable examples:** a CI test executes the README quick example, `docs/getting-started/quickstart.md`, CLAUDE.md patterns 1–9 (or their successors) and the flagship docstring examples, and all of them pass.
- **Audit bugs:** every bug in the audit list has a regression test that fails on `main` and passes after the fix (phase validation slices).
- **Gap correctness:** on a synthetic two-epoch recording with a 1000 s pause and a true rate of 5 Hz, every rate family reports 5 Hz ± 5%, and the decoder creates no time bins inside the pause (Phase 3).
- **Researcher workflow (checkpoint after Phase 4):** each of the five journeys completes from public documentation alone, without reading source code.
- **Expectation, not a target:** the public surface stays large (Phase 6 cuts about 427 exports to 418). The gains are coherence, discoverability and correct defaults, not a much smaller API. The analytical modules remain substantial.
- **Lines of governance:** 0 lines of inventory or count-freeze tests; the API snapshot is one test file plus one text snapshot.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The default spike window (assumed to equal the position coverage) is wrong when ephys starts after tracking. | Phase 3: when ≥5 units are all silent for ≥60 s inside the analyzed window, warn and name `spike_window=`; the parameter is documented on every spike+position function. |
| Porting fixes drags in archive infrastructure. | Phase 1 ports each fix as a minimal re-implementation against `main`'s code, with the archive commit as reference only. |
| Executable-docs CI becomes slow or flaky. | Phase 4 uses small simulated fixtures (≤60 s of data). They are **not** marked `slow`, because no CI job runs `slow` tests (`pytest.ini` overrides `pyproject.toml`), so slow-marked docs tests would never be enforced. |
| CI does not run on this branch: every workflow triggers only on `main`. | Phase 1's first code task adds `feat/researcher-first` to the workflow triggers, so every later PR gets CI. |
| Namespace curation silently drops something a doc uses. | Phase 6 runs the executable-docs test from Phase 4 after curation; a missing name fails CI. |

## Rollout Strategy

There are nine PRs into `feat/researcher-first` (phases 1, 2, 3a, 3b, 3c, 4, 5, 6 and 7), then one release. The ordering constraints:

- Phase 1 must land first, because it enables CI on this branch.
- Phase 2 can follow in any order after that.
- 3b and 3c each require 3a, and are independent of each other.
- All of Phase 3 must land before Phase 4, because Phase 4 executes examples that depend on the four-array calls working.
- Phase 6 must follow Phase 4, because it relies on the executable-docs guard.

No feature flags are used.

**Researcher-workflow checkpoint after Phase 4** (before Phases 5–7 start). Executable examples prove that calls work; this checkpoint tests whether the design actually reduces researcher effort.
- On the branch, re-run the `ux-review` and `design-review` workflows (`.claude/workflows/`). Use their journeys, working only from the public documentation:
  1. load an NWB file or simulate a session;
  2. select epochs;
  3. compute rate maps for a population;
  4. decode;
  5. produce a summary table and a plot.
- Compare against the October 2026 baseline: the UX review rated the experience CONFUSING, and all five design-review journeys were "painful".
- Record each journey's call count and lines of code, and every place the agent had to read source code.
- Findings become tasks in Phases 5–7, or a new phase, before those phases start. If the journeys are not clearly easier, stop and revisit the design before continuing.

## Open Questions

1. The population-silence warning thresholds (≥5 units, ≥60 s). These are the best current answer; revisit after use on real data.
2. Whether `epochs=` should also be accepted by the behavior functions (position-only). Best answer: yes. Phase 3 adds it where a function consumes `times`.
3. Time windows for the per-sample spike-only helpers: `events.time_to_nearest_event`, `event_count_in_window`, `event_indicator` and `align_spikes_to_events` stay deferred (Phase 3b), because the contract's event rule does not map onto per-sample outputs and masking them would change `int64`/`bool` return dtypes or per-event indexing. Decide before Phase 4 documents them.

## Estimated Effort

- Phase 1: about 1–1.5k lines (mostly tests).
- Phase 2: about 600 lines.
- Phases 3a–3c: about 4–5k lines in total, across about 60 entry points.
- Phase 4: about 1.5k lines, mostly docs.
- Phase 5: about 800 lines.
- Phase 6: about 1k lines changed and net deletions.
- Phase 7: about 400 lines.

These are order-of-magnitude figures for diff sizing only.
