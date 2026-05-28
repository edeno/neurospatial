# Phase 8 — Mock removal (real-path parity tests)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add at least one real-path parity test per heavily-mocked area, so that CI configurations that include the optional dependency actually exercise real code. The audit found that without Qt, 0 lines of real-widget code were exercised; with Qt, mocked `ipywidgets` still hid the integration. Same story for `napari` (56 of 56 backend tests mock `_create_mock_viewer`) and `pynwb` (`MockTimeSeries`/`MockEventsTable` despite the file already gating on `pytest.importorskip("pynwb")`).

This phase does **not** remove existing mock-heavy tests wholesale — they still document API surface. It adds a real-path test alongside each, gated on `importorskip`, so that integration drift is caught.

**Inputs to read first:**

- [tests/animation/test_widget_backend.py:170-189, 218-233, 263-282](../../../tests/animation/test_widget_backend.py) — current `MagicMock(ipywidgets)` blocks. 67/67 tests are mock-driven.
- [tests/animation/test_napari_backend.py:11, 37-45](../../../tests/animation/test_napari_backend.py) — `_create_mock_viewer()` returning `MagicMock`. 56 tests use it.
- [tests/animation/test_widget_fallback.py:64-80](../../../tests/animation/test_widget_fallback.py) — tests log behavior; never asserts output equivalence between fast-path and fallback.
- [tests/nwb/test_adapters.py:17-44, 155-227](../../../tests/nwb/test_adapters.py) — `MockTimeSeries`, `MockEventsTable`. File already has `pytest.importorskip("pynwb")` at line 14.
- [tests/nwb/conftest.py:71-361](../../../tests/nwb/conftest.py) — real pynwb fixtures available. Reuse rather than rebuild.
- [tests/nwb/test_events.py:216-247, 674](../../../tests/nwb/test_events.py) — `monkeypatch(builtins.__import__)` brittle test of error message.
- [tests/nwb/test_pose.py:689](../../../tests/nwb/test_pose.py) — same monkeypatch pattern.
- [src/neurospatial/animation/backends/widget_backend.py](../../../src/neurospatial/animation/backends/widget_backend.py) — `PersistentFigureRenderer.render(idx)` and `.set_array(idx)`. Phase 8 will test the fast-path / fallback equivalence with real ipywidgets.

## Tasks

### 1. Real-widget parity test in `test_backend_consistency.py`

Phase 4 already added cross-backend pixel parity. Phase 8 extends one of those tests to use **real `ipywidgets`** in the widget arm (currently mocked):

- In [tests/animation/test_backend_consistency.py](../../../tests/animation/test_backend_consistency.py), the `test_video_widget_parity` test (added in Phase 4) currently calls `PersistentFigureRenderer.render(2)` and extracts pixels from the matplotlib Figure directly. Strengthen by also asserting on the `ipywidgets.Image.value` bytes (the actual PNG payload the widget would display).

- Decode the PNG bytes (`from PIL import Image; img = Image.open(io.BytesIO(widget.value)); np.array(img)`) and compare to the video-backend frame pixel-wise. Gate on `pytest.importorskip("ipywidgets")` and `pytest.importorskip("PIL")`.

This catches a real bug class: if `PersistentFigureRenderer` writes to the figure correctly but the widget's `Image.value` serialization is broken, no current test catches it.

### 2. Real-widget fast-path / fallback equivalence

In [tests/animation/test_widget_fallback.py](../../../tests/animation/test_widget_fallback.py): after the existing log-behavior tests, add `TestFastPathFallbackEquivalence`:

- `test_set_array_and_full_redraw_produce_same_pixels`: render frame `idx` two ways:
  1. Full redraw via `PersistentFigureRenderer(allow_set_array=False).render(idx)`.
  2. Optimized `set_array` path via `PersistentFigureRenderer(allow_set_array=True)` after advancing the index by 1.

  Extract pixels from both and assert `np.allclose(fast, slow, atol=2)`. (Argument names may differ — read the source to confirm the flag/method names.)

  This is the only way to verify the fast-path optimization actually produces equivalent output. The audit cited `test_widget_fallback.py:64-80` as testing log behavior only.

### 3. Real-napari path in at least one parity test

In [tests/animation/test_backend_consistency.py](../../../tests/animation/test_backend_consistency.py), `test_video_napari_parity` (added in Phase 4) already calls `viewer.screenshot()` on a real napari `Viewer`. Phase 8 confirms there is no `_create_mock_viewer` shortcut in this path.

If Phase 4's implementation accidentally used the mock viewer (executor should verify): rewrite to use a real `napari.Viewer` gated on `pytest.importorskip("napari")`.

No new test file — just confirm the existing parity test runs without mocks.

### 4. Real `pynwb` containers in `test_adapters.py`

In [tests/nwb/test_adapters.py:17-44, 155-227](../../../tests/nwb/test_adapters.py), replace `MockTimeSeries` and `MockEventsTable` with real `pynwb.TimeSeries` and `pynwb.epoch.TimeIntervals` (or whatever the corresponding real class is — read pynwb source).

The conftest fixtures at `tests/nwb/conftest.py:71-361` already produce real containers; reuse via `@pytest.fixture` injection rather than constructing inline.

Two replacement passes:
- Lines 17-44: `MockTimeSeries` block → import real `pynwb.TimeSeries`, use a fixture from conftest.
- Lines 155-227: `MockEventsTable` block → same.

After: the `pytest.importorskip("pynwb")` at line 14 (already there) gates everything; tests run only when pynwb is available, and exercise the real surface.

### 5. Replace `monkeypatch(builtins.__import__)` import-error tests

In [tests/nwb/test_events.py:216-247, 674](../../../tests/nwb/test_events.py) and [tests/nwb/test_pose.py:689](../../../tests/nwb/test_pose.py), the current pattern monkey-patches `builtins.__import__` to test that `_require_ndx_events()` (or similar) raises a helpful error when the extension is missing.

Replace with one of:
- **Option A (preferred)**: Remove the test entirely. The function under test is `_require_ndx_events()` which raises `ImportError` if the import fails. The error message is a wrapper around the standard `ModuleNotFoundError`; testing the wrapper string by mocking `__import__` is brittle. Confirm the function's `try/except ImportError` path is exercised by other means (a test that runs *without* the extension installed in a CI matrix entry would be ideal — check `.github/workflows/` or `pyproject.toml` test matrix; if no such matrix entry exists, leave the test as a `pytest.skip("requires CI matrix without ndx-events")`).
- **Option B**: Remove the module from `sys.modules` and `sys.path` for the test, then reimport. More surgical than `builtins.__import__` patching but still touches global state. Worse than Option A.

Default to Option A. If executor finds the function under test is otherwise uncovered, document in PR description.

### 6. Drop module-reimport gymnastics in `test_napari_backend.py`

[tests/animation/test_napari_backend.py:82-119](../../../tests/animation/test_napari_backend.py) — deletes from `sys.modules` and reimports `napari_backend` under patched `sys.modules`. This is test-order-dependent and brittle.

Replace with: run the backend in a subprocess (`subprocess.run([sys.executable, "-c", "..."])` ) with `napari` unavailable in the subprocess's environment via a custom `PYTHONPATH`. Or, simpler — gate the test on a CI matrix entry that doesn't install napari, and `pytest.skip` otherwise.

If neither option is practical with the existing CI, **delete the test**. The behavior it asserts (graceful import failure when napari is missing) is already covered by `pytest.importorskip("napari")` at the file level being a no-op when napari is absent.

## Deliberately not in this phase

- **No deletion of the existing 67-test mocked widget backend file.** It still documents API surface; deletion is out of scope for this PR.
- **No new perf / benchmark tests.** Phase 7 already disabled the flaky perf assertions.
- **No new GLM regressor / surrogate / shuffle work.** Phase 3 owned that area.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_video_widget_parity` (Phase 4 test, strengthened) | Real `ipywidgets.Image.value` decoded PNG matches video frame pixel-wise. |
| `tests/animation/test_widget_fallback.py::TestFastPathFallbackEquivalence::test_set_array_and_full_redraw_produce_same_pixels` | `set_array` fast-path and full-redraw produce equivalent pixels. **`importorskip("ipywidgets")`.** |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_video_napari_parity` (Phase 4 test, verified) | Real `napari.Viewer` (no mock) is used in this test. **`importorskip("napari")`.** |
| `tests/nwb/test_adapters.py` (mocks removed) | Tests use real `pynwb.TimeSeries`/`TimeIntervals`, gated on `importorskip("pynwb")`. |
| `tests/nwb/test_events.py`, `test_pose.py` (monkeypatch tests removed or rewritten) | No `monkeypatch(builtins.__import__)` remains in either file. |
| `tests/animation/test_napari_backend.py` (`sys.modules` reimport removed) | Either tests run in subprocess, or are deleted with rationale in PR description. |

## Fixtures

In [tests/nwb/conftest.py](../../../tests/nwb/conftest.py): reuse existing real-container fixtures. If a fixture for a minimal `pynwb.TimeIntervals` doesn't exist, add `simple_time_intervals` fixture matching the test data the `MockEventsTable` was using.

No new fixtures expected for the animation tests — Phase 4's `canonical_parity_fixture` is reused.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no wholesale deletion of mock-heavy test files.
- Validation slice tests pass; importorskip gates work correctly (verify by running with the optional dependency uninstalled in a clean env, or check CI matrix).
- Tests aren't trivial — real-path parity tests extract bytes/pixels from real widgets/viewers and compare, not just "viewer is not None". The fast-path / fallback equivalence test pixel-compares both paths, not just "both ran without error". (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed: `MockTimeSeries`, `MockEventsTable`, `monkeypatch(builtins.__import__)` blocks, the `sys.modules` reimport block in `test_napari_backend.py` (or the test is deleted with rationale).
- User-facing documentation listed as tasks is updated, not deferred (none in this phase — public API behavior is unchanged).
