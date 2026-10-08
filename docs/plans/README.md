# Planning Documents

This directory contains implementation plans, task trackers, scratchpads, and backlog notes that do not need to live at the repository root.

Date labels are the first Git commit date for the original artifact, based on `git log --follow --diff-filter=A`. Individual plan/task documents also include a `Committed` label near the title.

## Active or Recently Active

| Plan committed | Area | Plan | Tasks | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-08 | v0.4 UX cleanup | [v0.4-ux-cleanup/PLAN.md](v0.4-ux-cleanup/PLAN.md) | [v0.4-ux-cleanup/TASKS.md](v0.4-ux-cleanup/TASKS.md) | Repository-wide UX & interface consistency cleanup driven by [`docs/reviews/UX_REVIEW_2026-05-08.md`](../reviews/UX_REVIEW_2026-05-08.md). Eight milestones (M0 onboarding hotfixes → M7 v0.4.0 release). Hard-break release; no deprecation cycle. |
| 2026-05-07 | Encoding cleanup | [encoding-cleanup/PLAN.md](encoding-cleanup/PLAN.md) | [encoding-cleanup/TASKS.md](encoding-cleanup/TASKS.md) | Post-review cleanup of the encoding refactor. Six milestones (M1 consistency → M6 legacy delegation), targeting ~−1500 LOC net. M0 wins already applied on `encoding-refactor-wip`. |
| 2025-11-22 | Napari playback | [napari-playback/PLAN.md](napari-playback/PLAN.md) | [napari-playback/TASKS.md](napari-playback/TASKS.md) | Tasks committed 2025-11-03. Playback interface enhancement plan and task breakdown. |
| 2025-12-05 | Behavior metrics | [behavior-metrics/PLAN.md](behavior-metrics/PLAN.md) | [behavior-metrics/TASKS.md](behavior-metrics/TASKS.md) | Tasks committed 2025-12-05. Advanced behavioral metrics plan. It references an earlier `BEHAV_PLAN.md` prerequisite that is not present in this checkout. |

## Archived / Superseded

| Plan committed | Area | Plan | Tasks | Notes |
| --- | --- | --- | --- | --- |
| 2025-12-05 | Encoding refactor | [encoding-refactor/PLAN.md](encoding-refactor/PLAN.md) | [encoding-refactor/TASKS.md](encoding-refactor/TASKS.md) | Complete historical plan. Superseded for current API vocabulary by the v0.6 API contract in `.claude/docs/plans/ux-v0.6/api-contract.md`; do not copy its old `neuron_id`, `detect_*`, or view-specific peak names into new work. |

## Planning Backlog

| Document committed | Area | Document | Notes |
| --- | --- | --- | --- |
| 2025-12-06 | Environment refactor | [environment-refactor/PLAN.md](environment-refactor/PLAN.md) | Pending follow-up after the encoding refactor. |
| 2025-11-23 | LLM integration | [llm-integration/PLAN.md](llm-integration/PLAN.md) | Tooling and agent-facing integration plan. |
| 2025-11-04 | Replay transition matrix | [replay-transition/PLAN.md](replay-transition/PLAN.md) | Replay-analysis transition matrix construction plan. |
| 2025-11-19 | General backlog | [backlog/TODO.md](backlog/TODO.md) | Older broad backlog notes. |

## Root Policy

Keep new planning artifacts under `docs/plans/<topic>/` unless they are short-lived notes that will be deleted before commit. Pair task trackers with their associated plan in the same topic directory.
