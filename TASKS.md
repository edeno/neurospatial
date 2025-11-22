# Video Annotation Feature - Implementation Tasks

**Feature**: Interactive annotation system for defining spatial environments and regions from video frames
**Plan**: See [PLAN.md](PLAN.md) for detailed implementation specifications
**Target Version**: v0.6.0

## Progress Overview

| Milestone | Status | Tasks |
|-----------|--------|-------|
| M1: Module Structure | In Progress | 1/2 |
| M2: Converters | Complete | 2/2 |
| M3: External Import | Not Started | 0/2 |
| M4: Napari Widget | Not Started | 0/2 |
| M5: Main Entry Point | Not Started | 0/2 |
| M6: Tests | In Progress | 2/5 |
| M7: Documentation | Not Started | 0/1 |

---

## Milestone 1: Module Structure and Public API

- [x] **1.1** Create `src/neurospatial/annotation/__init__.py` with public exports (partial - converters only)
- [ ] **1.2** Update `src/neurospatial/__init__.py` with annotation imports

**Verification**: `uv run python -c "from neurospatial.annotation import annotate_video; print('OK')"`

---

## Milestone 2: Converters Module

- [x] **2.1** Create `src/neurospatial/annotation/converters.py`
  - `shapes_to_regions()` - napari shapes -> Regions
  - `env_from_boundary_region()` - Region -> Environment
- [x] **2.2** Verify import works

**Key coordinate transform**: napari (row, col) → video (x, y) pixels → calibration → world (x, y) cm

---

## Milestone 3: External Tool Import Wrappers

- [ ] **3.1** Create `src/neurospatial/annotation/io.py`
  - `regions_from_labelme()` - wrapper for `load_labelme_json`
  - `regions_from_cvat()` - wrapper for `load_cvat_xml`
- [ ] **3.2** Verify imports work

**Dependencies**: Uses existing `regions/io.py` functions with `VideoCalibration.transform_px_to_cm`

---

## Milestone 4: Napari Annotation Widget

- [ ] **4.1** Create `src/neurospatial/annotation/_napari_widget.py`
  - `create_annotation_widget()` - magicgui container
  - `setup_shapes_layer_for_annotation()` - features-based coloring
  - `get_annotation_data()` - extract shapes/names/roles
- [ ] **4.2** Verify import works

**Napari patterns** (per PLAN.md Architecture section):

- Features-based coloring with `pd.Categorical`
- Text labels via `text={'string': '{name}'}`
- Keyboard shortcuts: E=environment, R=region
- `events.data.connect()` for shape-added detection

---

## Milestone 5: Main Entry Point

- [ ] **5.1** Create `src/neurospatial/annotation/core.py`
  - `AnnotationResult` named tuple
  - `annotate_video()` main function
  - `_add_initial_regions()` helper
- [ ] **5.2** Verify end-to-end import works

---

## Milestone 6: Tests

- [x] **6.1** Create `tests/annotation/__init__.py`
- [x] **6.2** Create `tests/annotation/test_converters.py`
- [ ] **6.3** Create `tests/annotation/test_io.py`
- [ ] **6.4** Create `tests/annotation/test_core.py`
- [ ] **6.5** Run test suite and verify passing

**Commands**:

```bash
uv run pytest tests/annotation/ -v           # Annotation tests only
uv run pytest tests/annotation/ -m "not gui" # Skip napari tests in CI
uv run pytest                                # Full suite
```

---

## Milestone 7: Documentation

- [ ] **7.1** Update CLAUDE.md Quick Reference with annotation examples

---

## Final Verification Checklist

Before marking complete:

- [ ] All annotation tests pass: `uv run pytest tests/annotation/ -v`
- [ ] Full test suite passes: `uv run pytest`
- [ ] Linting passes: `uv run ruff check . && uv run ruff format .`
- [ ] Type checking passes: `uv run mypy src/neurospatial/annotation/`
- [ ] Public API imports work:

  ```python
  from neurospatial import annotate_video, regions_from_labelme, regions_from_cvat
  ```

---

## Notes

- GUI tests marked `@pytest.mark.gui` - skip in headless CI with `-m "not gui"`
- Register marker in `pyproject.toml` if not present
- See PLAN.md for complete code specifications
