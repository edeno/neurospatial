# SCRATCHPAD: Behavioral Trajectory Metrics Implementation

**Started**: 2025-12-05
**Current Phase**: M2 COMPLETE - Starting M3 (Decision Analysis)

---

## Current Task

**COMPLETED**: M2 - Goal-Directed Metrics

Next: **M3 - Decision Analysis**

---

## Notes

### Session 1 (2025-12-05)

- Reviewed PLAN.md and TASKS.md
- Starting with Milestone 1: Path Efficiency
- Key decisions from plan:
  - Use `metric` parameter (not `distance_type`)
  - Compute `dt = median(diff(times))` for `heading_from_velocity()`
  - Direct circular stats computation (not private functions)

### M1 Completion Notes

- All 24 tests pass
- Implemented 8 functions and 2 dataclasses:
  - `PathEfficiencyResult`, `SubgoalEfficiencyResult`
  - `traveled_path_length`, `shortest_path_length`, `path_efficiency`
  - `time_efficiency`, `angular_efficiency`, `subgoal_efficiency`
  - `compute_path_efficiency`
- Code review: APPROVED with no critical issues
- All exports added to `metrics/__init__.py`
- ruff check: PASSED
- mypy check: PASSED

### M2 Completion Notes (Session 2)

- All 22 tests pass
- Implemented 6 functions and 1 dataclass:
  - `GoalDirectedMetrics`
  - `goal_vector`, `goal_direction`
  - `instantaneous_goal_alignment`, `goal_bias`
  - `approach_rate`, `compute_goal_directed_metrics`
- Code review: APPROVED (after fixing 3 critical issues)
  - Added assert for mypy type guard in approach_rate()
  - Added `strict=True` to zip() in tests
  - Added all exports to metrics/__init__.py
- Key design decisions:
  - Orthogonal movement test uses circular path (not straight line)
  - Uses `heading_from_velocity()` with `dt = median(diff(times))`
  - Stationary periods (below min_speed) return NaN
- ruff check: PASSED (auto-fixed import sorting)
- mypy check: PASSED

---

## Blockers

None currently.

---

## Decisions Made

| Decision | Rationale | Date |
|----------|-----------|------|
| TDD approach | Required by workflow | 2025-12-05 |
| Use `metric` not `distance_type` | Consistency with behavioral.py | 2025-12-05 |
| Map `metric` -> `distance_type` internally | compute_step_lengths uses legacy param | 2025-12-05 |

---

## Files Modified

### Session 1 (M1 - Path Efficiency)

- `src/neurospatial/metrics/path_efficiency.py` - NEW (650 LOC)
- `tests/metrics/test_path_efficiency.py` - NEW (24 tests)
- `src/neurospatial/metrics/__init__.py` - Added exports
- `TASKS.md` - Updated M1 checkboxes
- `SCRATCHPAD.md` - Created

### Session 2 (M2 - Goal-Directed Metrics)

- `src/neurospatial/metrics/goal_directed.py` - NEW (~500 LOC)
- `tests/metrics/test_goal_directed.py` - NEW (22 tests)
- `src/neurospatial/metrics/__init__.py` - Added exports
- `TASKS.md` - Updated M2 checkboxes
- `SCRATCHPAD.md` - Updated
