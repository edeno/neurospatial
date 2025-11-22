# VideoOverlay Implementation Scratchpad

**Started**: 2025-11-22
**Current Phase**: Milestone 3 (Video I/O) - Task 3.1

---

## Dependency Analysis

**Issue Found**: Task I.1 (Update Type Signatures) requires `VideoOverlay` to exist, but `VideoOverlay` is created in Task 2.1. This is a dependency ordering issue in TASKS.md.

**Resolution**: Execute tasks in dependency order:
1. I.3, I.4, I.5, I.6 - Verification and prep tasks (no VideoOverlay needed)
2. I.2 - Fix artist reuse (no VideoOverlay needed)
3. 1.1, 1.2 - Calibration infrastructure (needed by VideoCalibration)
4. 2.1 - Create VideoOverlay dataclass
5. I.1 - Update type signatures (NOW VideoOverlay exists)
6. Continue with 2.2, 3.x, etc.

---

## Session Log

### 2025-11-22

- Read PLAN.md and TASKS.md
- Identified dependency issue: I.1 needs VideoOverlay but it's created in 2.1
- **Completed M0**: I.5, I.3, I.4, I.6, I.2 (integration pre-requisites)
- **Completed M1**: 1.1, 1.2 (calibration infrastructure)
  - `calibrate_from_scale_bar()` - scale bar calibration
  - `calibrate_from_landmarks()` - landmark-based calibration
  - `VideoCalibration` - dataclass with serialization
  - 19 unit tests passing
- **Completed M2 (partial)**: 2.1 - VideoOverlay dataclass
  - All fields implemented: source, calibration, times, alpha, z_order, crop, downsample, interp
  - `__post_init__` validation: alpha bounds, downsample, array shape/dtype/channels
  - Comprehensive NumPy-style docstring with examples
  - Added to `__all__` in main package and animation module
  - 22 unit tests passing
- **Completed I.1**: Update type signatures
  - Updated overlays parameter types in core.py and visualization.py
  - Added VideoOverlay and VideoCalibration exports to animation/__init__.py
- **Completed M2**: 2.2 - VideoData internal container
  - All fields: frame_indices, reader, transform_to_env, env_bounds, alpha, z_order
  - `get_frame()` method returns frame or None for out-of-range
  - Pickle-safety verified for parallel rendering
  - 6 unit tests passing

---

## Current Task

**Working on**: Milestone 3 - Video I/O (Task 3.1)

---

## Blockers

None currently.

---

## Decisions Made

1. Reordering tasks to respect dependencies while following TDD
