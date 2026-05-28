# Phase 4 — Animation backend parity

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Rewrite `test_backend_consistency.py` so it actually compares pixel output across the four backends (napari, video, html, widget) instead of exercising them one at a time. Rewire `test_overlay_visual_regression.py` to drive the production renderer instead of its own parallel matplotlib implementation. Dedup the near-duplicate `test_video_overlay.py` (6 tests) into `test_video_overlays.py` (38 tests).

**Inputs to read first:**

- [tests/animation/test_backend_consistency.py](../../../tests/animation/test_backend_consistency.py) (282 lines) — current state. Each test exercises one backend. Need to read to understand the existing fixture and config plumbing so the rewrite can reuse them.
- [tests/animation/test_overlay_visual_regression.py:102-141](../../../tests/animation/test_overlay_visual_regression.py) — `render_field_with_overlays` is a parallel matplotlib implementation that mimics the video backend instead of calling it.
- [src/neurospatial/animation/backends/](../../../src/neurospatial/animation/backends/) — list the actual backend module files. Likely `video_backend.py`, `napari_backend.py`, `html_backend.py`, `widget_backend.py`.
- [src/neurospatial/animation/backends/widget_backend.py](../../../src/neurospatial/animation/backends/widget_backend.py) — `PersistentFigureRenderer.render(idx)` returns the actual frame the widget would display. This is what Phase 4 will pixel-extract from for widget.
- [tests/animation/test_video_overlay.py](../../../tests/animation/test_video_overlay.py) (6 tests) and [tests/animation/test_video_overlays.py](../../../tests/animation/test_video_overlays.py) (38 tests) — read both to identify the 6 tests in `test_video_overlay.py` and check which (if any) duplicate the 38 in `test_video_overlays.py`.
- [tests/animation/baseline/](../../../tests/animation/baseline/) — existing `pytest-mpl` baseline images. Phase 4 keeps these but rewires the renderer that produces them.

## Tasks

### 1. Rewrite `test_backend_consistency.py` for actual cross-backend comparison

Delete the existing one-backend-at-a-time tests in [tests/animation/test_backend_consistency.py](../../../tests/animation/test_backend_consistency.py) (lines 63-282 per the audit). Replace with a single parametrized class `TestBackendPixelParity` that:

1. Defines one canonical fixture: a 32×32 spatial field with a Gaussian bump, 4 frames of synthetic firing-rate data, `vmin=0`, `vmax=10`, `cmap="viridis"`. Low resolution to control rendering variance.
2. Renders the same `(env, fields, frame_times, config)` through each backend:
   - **video**: write `out.mp4`; read with `cv2.VideoCapture` (pattern at `test_video_overlay.py:318`); extract uint8 RGB array of frame 2.
   - **napari**: call `viewer.screenshot()`, extract uint8 RGB.
   - **html**: parse the HTML file, base64-decode the embedded frames (the audit cited `test_backend_consistency.py:85` showing the `const frames = [` template); decode frame 2.
   - **widget**: instantiate `PersistentFigureRenderer`, call `renderer.render(field=fields[2], frame_idx=2)` (signature at [widget_backend.py:399-427](../../../src/neurospatial/animation/backends/widget_backend.py): `render(self, field, frame_idx, overlay_data=None, show_regions=False, region_alpha=0.3) -> bytes`). The return is **image bytes** (PNG or JPEG depending on the renderer's `image_format`), not a matplotlib Figure. Decode via `PIL.Image.open(io.BytesIO(rendered_bytes))` then `np.array(img)` to get uint8 RGB.
3. Performs four pairwise comparisons: `(video, napari)`, `(video, html)`, `(video, widget)`, `(napari, html)`. Each uses `np.testing.assert_allclose(a, b, atol=2)` on uint8 RGB (see Open Question 2 in overview.md — tolerance may need to relax to `atol=5` for napari).
4. Add a separate sub-test that **varies `vmin`/`vmax`** and re-checks pairwise pixel parity. This catches the audit's specific concern (a bug where HTML ignores vmin/vmax silently while napari respects it).
5. Add a separate sub-test that **varies `cmap`** (`"viridis"` vs `"hot"`). Pixel values change, but the change should be identical across backends.

Each backend test must use the existing `pytest.importorskip` pattern (napari, ipywidgets, cv2 all already gated elsewhere in the suite). A backend that can't be loaded is skipped, not failed; but the test for each pair runs only if both backends in the pair load.

Total: ~6 tests in the new file, each comparing at minimum two backends pixel-wise. Replaces ~13 existing single-backend tests.

### 2. Wire `test_overlay_visual_regression.py` to the production renderer

In [tests/animation/test_overlay_visual_regression.py:102-141](../../../tests/animation/test_overlay_visual_regression.py), delete the `render_field_with_overlays` helper that re-implements rendering in matplotlib.

Replace each test that uses it with a call to `video_backend.render_video(...)` writing to a temporary `.mp4`, then read frame 0 with `cv2.VideoCapture` and `imwrite` to PNG for the `pytest-mpl` comparison.

The 7 baseline PNGs in `tests/animation/baseline/` may need to be regenerated (one-time): run the new test with `--mpl-generate-path=tests/animation/baseline` and commit the new images. **Inspect the diff manually before committing — if a baseline changes substantially, it's evidence the parallel implementation was lying.**

Add one task to the PR description: a side-by-side image comparison (old vs new baseline) for each of the 7 images. Reviewers should manually inspect.

### 3. Dedup `test_video_overlay.py` → `test_video_overlays.py`

First, read both files and produce a name-by-name index of the 6 tests in [tests/animation/test_video_overlay.py](../../../tests/animation/test_video_overlay.py).

For each test in `test_video_overlay.py`:
- If a same-named or substantially-identical test exists in `test_video_overlays.py`: delete it from the singular file.
- If the test is genuinely unique (e.g., the cv2-roundtrip test the audit cited at lines 318-382): move it to `test_video_overlays.py`.

After moves, delete `test_video_overlay.py` entirely. Commit the deletion separately within this phase's PR for review clarity.

### 4. Verify `test_rendering_validation.py` against new parity

[tests/animation/test_rendering_validation.py](../../../tests/animation/test_rendering_validation.py) currently has one test (one shape-error). Phase 4 leaves this file as-is — Phase 7 may expand or delete it.

Just confirm the parity tests in Task 1 don't accidentally re-implement what `test_rendering_validation.py` was supposed to test, then move on.

### 5. Document the parity guarantee in the animation README/docstring

[src/neurospatial/animation/__init__.py](../../../src/neurospatial/animation/__init__.py) or whichever module exposes `env.animate_fields(...)`: in the public docstring, add a "Backend parity" section noting that the four backends produce visually equivalent output (verified via `tests/animation/test_backend_consistency.py`). One short paragraph; this is documentation that ships with the public API.

## Deliberately not in this phase

- **No mock removal in `test_widget_backend.py` / `test_napari_backend.py`.** Those 1300+ lines of mock-heavy tests stay until Phase 8.
- **No fixing flaky performance thresholds** in `test_video_performance.py` / `test_benchmarks.py`. Phase 7.
- **No deletion or expansion of `test_rendering_validation.py`** beyond cross-checking with Task 1. Phase 7.
- **No new overlay-type tests.** The existing `test_overlays.py` (125 tests) is left alone here; consolidation is out of scope. Phase 7 may revisit if a clear duplicate set emerges.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_video_napari_parity` | Pixel arrays from video and napari match at `atol=2` for the canonical 32×32 fixture. |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_video_html_parity` | Pixel arrays from video and HTML-embedded frames match. |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_video_widget_parity` | Pixel arrays from video and `PersistentFigureRenderer.render(idx)` match. |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_vmin_vmax_respected_across_backends` | Varying vmin/vmax produces the same pixel change in all backends. |
| `tests/animation/test_backend_consistency.py::TestBackendPixelParity::test_cmap_respected_across_backends` | Varying cmap produces the same pixel change in all backends. |
| `tests/animation/test_overlay_visual_regression.py::*` (7 existing tests, rewired) | Production renderer (`video_backend.render_video` + cv2 read) produces images matching the (possibly regenerated) baselines. |
| `pytest tests/animation/test_video_overlays.py` (post-dedup) | All non-duplicate tests from the deleted `test_video_overlay.py` are present and passing. |

Mark new parity tests with no slow marker (32×32 should be fast) but `@pytest.mark.requires_cv2` if a convention exists for that.

## Fixtures

Create `tests/animation/conftest.py` (or extend existing): a `canonical_parity_fixture` returning `(env, fields, frame_times, config)` at 32×32 resolution, 4 frames, Gaussian bump. Used by all 5 parity tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 7 (perf thresholds) or Phase 8 (mock removal).
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — backend-parity tests actually extract pixels and `assert_allclose` across two backends. No `assert viewer is not None` assertions of the kind the audit found.
- The dedup actually removes `test_video_overlay.py`; no tests are lost (executor produced a name-by-name index in Task 3 — include it in the PR description).
- Baseline images for visual regression are regenerated through the production renderer, manually inspected, and the inspection is recorded in the PR description.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (`render_field_with_overlays` helper in `test_overlay_visual_regression.py`, the singular-named `test_video_overlay.py`).
- User-facing documentation listed as tasks is updated (the backend-parity note in the animation public docstring).
