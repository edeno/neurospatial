# Planning Documents

This directory contains implementation plans, task trackers, scratchpads, and backlog notes that do not need to live at the repository root.

Date labels are the first Git commit date for the original artifact, based on `git log --follow --diff-filter=A`. Individual plan/task documents also include a `Committed` label near the title.

Statuses were reviewed on **2026-10-14** against the current source, tests, and
Git history. Status notes in each plan and task tracker distinguish delivered
work from historical checkboxes and remaining proposals.

## Active Plans

| Plan committed | Area | Plan | Tasks | Notes |
| --- | --- | --- | --- | --- |
| 2026-10-14 | Continuous behavior and neural analysis | [continuous-behavior/PLAN.md](continuous-behavior/PLAN.md) | [continuous-behavior/TASKS.md](continuous-behavior/TASKS.md) | **Partially implemented (reviewed 2026-10-20).** Six workstreams: observation support, environment contracts, exploration/crossings, ordered paths, continuous events/features, and conditioned neural comparisons. Delivered: area 1 (#112, #114, #120), area 5 except interval states (#118), and area 6 same-environment comparisons (#119). Area 2 is partly covered by environment stages A–E. Areas 3–4 have not started. `break_before` reaches every function that takes `max_gap` (#121). Statistical/null-model frameworks remain outside scope. |
| 2026-10-13 | Environment representation | [environment-representation/PLAN.md](environment-representation/PLAN.md) | Phase files in the same directory | **Representation phases complete on `feature/environment-representation`.** Stored binning/locators, loading without rebuilding, and derived geometry preservation are implemented and validated. Roadmap stages A–F remain separate; this foundation supports continuous behavior's geometry and correspondence/snapshot slices. |
| 2026-10-14 | Continuous arena representation | [environment-arena/PLAN.md](environment-arena/PLAN.md) | [Validation](environment-arena/validation.md) | **Stage A complete on `feature/environment-arena`, stacked on representation.** Continuous domains, explicit walls, exact planar clipping, arena-preserving rebinning and opt-in linear spatial encoding are implemented and validated. Stages B–F remain separate. |
| 2026-10-15 | Affine geometry and grid frames | [environment-affine/PLAN.md](environment-affine/PLAN.md) | E0–E5 | **Complete: E0–E5 implemented and independently reviewed on 2026-10-15** ([#76](https://github.com/edeno/neurospatial/pull/76)). Transforms preserve exact transformed cells and grid operations; numerical operations are enabled only with validated face geometry. See [validation.md](environment-affine/validation.md). |
| 2026-10-15 | Arena re-binning and physical walls | [environment-rebin/PLAN.md](environment-rebin/PLAN.md) | D0–D3 | **Complete: D0–D3 implemented and validated.** Three stacked consumer branches add owned grid specifications, explicit physical/analysis border scores and wall-aware gaze/visibility. Open as [#72](https://github.com/edeno/neurospatial/pull/72) → [#73](https://github.com/edeno/neurospatial/pull/73) → [#74](https://github.com/edeno/neurospatial/pull/74), based on Stage C #71. |
| 2026-10-14 | Combine owned environments | [environment-combine/PLAN.md](environment-combine/PLAN.md) | C0–C4 | **Complete: C0–C4 implemented and independently reviewed on the local stacked branch.** Owned combining, certified support, path-only default bridges and explicit geometric coupling replace the legacy class. See the validation record for CPU checks and limits; open as [PR #71](https://github.com/edeno/neurospatial/pull/71), stacked on Stage B. |
| 2026-10-14 | Immutable environment values | [environment-immutable/PLAN.md](environment-immutable/PLAN.md) | Phases B0–B4 in the plan | **Complete: B0–B4 implemented and independently reviewed.** Immutable values, explicit bins and installed snapshot adapters are validated on the stacked branch; see the validation record for CPU results and unrun GUI/GPU checks. Stage B is open as [PR #70](https://github.com/edeno/neurospatial/pull/70). |

## Partially Implemented Plans

| Plan committed | Area | Plan | Tasks | Status and remaining work |
| --- | --- | --- | --- | --- |
| 2026-05-07 | Encoding cleanup | [encoding-cleanup/PLAN.md](encoding-cleanup/PLAN.md) | [encoding-cleanup/TASKS.md](encoding-cleanup/TASKS.md) | **Partially implemented.** Core cleanup, JAX/cache work, and legacy removal landed. Task 6.5 distance delegation remains deferred, and before/after benchmark tables remain unfilled; review current APIs before resuming. |
| 2025-11-22 | Napari playback | [napari-playback/PLAN.md](napari-playback/PLAN.md) | [napari-playback/TASKS.md](napari-playback/TASKS.md) | **Partially implemented.** FPS slider, play/pause, Space shortcut, and frame counter exist. Proposed speed presets/dropdown, extra shortcuts, and their tests remain absent. Tasks first committed 2025-11-03. |
| 2025-11-04 | Replay transition matrix | [replay-transition/PLAN.md](replay-transition/PLAN.md) | — | **Partially implemented; remaining scope needs review.** Empirical, random-walk, and diffusion modes exist through `Environment.transitions()` with row-stochastic matrices. Six other utilities remain proposals; the old module/signatures and column-stochastic convention are superseded for delivered modes. |

## Completed / Superseded

| Plan committed | Area | Plan | Tasks | Notes |
| --- | --- | --- | --- | --- |
| 2025-12-05 | Encoding refactor | [encoding-refactor/PLAN.md](encoding-refactor/PLAN.md) | [encoding-refactor/TASKS.md](encoding-refactor/TASKS.md) | **Completed historical implementation.** Merged 2026-05-07 (`e6cc25c0`), with documented JAX grid/border deferrals. Current vocabulary is governed by [public API rules](../../.claude/rules/public-api.md); do not copy old `neuron_id`, classification, or result-class names. |
| 2025-12-06 | Environment refactor | [environment-refactor/PLAN.md](environment-refactor/PLAN.md) | — | Superseded by [environment-representation/PLAN.md](environment-representation/PLAN.md). Do not execute: its frozen ~25-method `Environment`, methods moved to `ops/`, and layout-parameter persistence contradict the current design. |
| 2026-05-08 | v0.4 UX cleanup | [v0.4-ux-cleanup/PLAN.md](v0.4-ux-cleanup/PLAN.md) | [v0.4-ux-cleanup/TASKS.md](v0.4-ux-cleanup/TASKS.md) | **Superseded after the v0.4.0 release.** M0–M6 landed; local tag `v0.4.0` points to `55bc0851`. M7 was partial: migration page/checker never landed. Current API rules and tutorials replace the old release roadmap; do not mark the omitted migration work as completed. |

## Planning Backlog

- [Encoder setup consolidation](encoder-setup/PLAN.md) — committed
  2026-10-14; preserves the eight entry points while sharing private setup.
  Implemented in PR #111 (merged 2026-10-19; status reviewed 2026-10-20). The
  parity evidence is in the PR description only.

| Document committed | Area | Document | Notes |
| --- | --- | --- | --- |
| 2025-12-05 | Advanced behavior metrics | [behavior-metrics/PLAN.md](behavior-metrics/PLAN.md) | **Not implemented; needs scope/API review.** Reachability information, manifold embeddings, diffusion baselines, and geodesic VTE extensions are absent. Existing basic behavior metrics do not complete this plan. [Task tracker](behavior-metrics/TASKS.md). |
| 2025-11-23 | LLM integration | [llm-integration/PLAN.md](llm-integration/PLAN.md) | **Not implemented here.** Python/NWB workflows exist, but the proposed agent facade, registries, MCP server, and playbooks do not. No replacement plan covers that layer. |
| 2025-11-19 | General backlog | [backlog/TODO.md](backlog/TODO.md) | **Historical; partly implemented and partly superseded.** Population/results/visibility entries are stale. Follow current topic plans for overlapping work; remaining statistics, pipelines, and scaling ideas are unapproved backlog. |

## Root Policy

Keep new planning artifacts under `docs/plans/<topic>/` unless they are short-lived notes that will be deleted before commit. Pair task trackers with their associated plan in the same topic directory.
