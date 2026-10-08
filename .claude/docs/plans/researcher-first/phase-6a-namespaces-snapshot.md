# Phase 6a — Curated namespaces and the API snapshot

**Requires:** Phase 5b.

[← back to PLAN.md](PLAN.md) · [executing](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Read [executing.md](executing.md) first: branch and PR workflow, definition of done, CHANGELOG-per-commit, and what to do when the plan and reality disagree. This file holds only what is specific to Phase 6a.

## Checkpoint additions (2026-10-07)

- Make the existing `detect_assemblies`, `assembly_activation`,
  `pairwise_correlations`, `explained_variance_reactivation` and
  `reactivation_strength` discoverable in the API index and a complete public
  workflow. The checkpoint found them through public runtime help only.
- Show simulation → encoding, then two explicit branches: decoding from rate
  maps/counts and population statistics from counts. A posterior is not the
  input to assembly/EV analyses. Preserve control-period context and distinguish
  effect size, significant dimensions and thresholded core members using the
  corrected Phase 4c wording.
- Demonstrate a common nonzero-variance unit selection across all periods,
  carrying the same unit order/IDs. Use a real self-contained example under
  the executable-doc guard. Keep the four-array path; add no mandatory bundle,
  new statistical algorithm or private-helper import.

## Original scope

This phase:

- removes `main`'s governance remnants;
- shrinks each namespace to user-facing names, with one name per object;
- types the root's lazy exports so IDEs and mypy resolve them;
- removes the remaining deprecation shims;
- replaces the export-pinning tests with the one [API snapshot](shared-contracts.md#api-snapshot).

It requires Phase 5b because the snapshot freezes the names Phases 5a and 5b introduce. Phase 4b's executable-docs test is what catches a curated-away name that a document still uses. Phases 6b and 6c each regenerate the snapshot this phase creates.

**Inputs to read first** (verified on `main` at `da631a47`; earlier phases shift line numbers):

- [src/neurospatial/__init__.py](../../../../src/neurospatial/__init__.py) — root `__all__` (28 names on `main`; Phase 4a added `NeurospatialError`), the eager `bin_spikes_in_time` import, and `_LAZY_ATTRS` at :266 (`SpikeTrains`, `restrict`, `Session`, `load_session`, `BayesianDecoder`).
- [tests/test_sparse_init_exports.py](../../../../tests/test_sparse_init_exports.py), [tests/test_lazy_imports.py](../../../../tests/test_lazy_imports.py), [tests/test_package_imports.py](../../../../tests/test_package_imports.py) — existing export tests.
- Archive commit `208932ae` ("fix(api): type lazy top-level exports"), which Phase 1 deferred to this phase: `git show 208932ae -- src/neurospatial/__init__.py tests/typing/top_level_lazy_exports.py tests/test_lazy_imports.py`. Input only; it types `Session`/`load_session`/`BayesianDecoder`, which this plan does not keep at the root.
- Archive pruning, for input only: `git cat-file -p "$(git rev-parse 'archive/public-api-curation-2026-10-01^{commit}'):src/neurospatial/animation/__init__.py"` (and the other `__init__.py` files). Under zsh, quote the `rev:path` argument, because `$A:s…` is a zsh history modifier.
- **Files earlier phases already changed** (search by symbol):
  - `_exceptions.py` and the root `__init__.py`: Phase 4a added `NeurospatialError` to the root `__all__`; keep it.
  - Phase 4b's `FLAGSHIP` list, `tests/docs/test_docstring_sections.py` (covers every **root-exported callable** plus `FLAGSHIP`) and the executable docs. Update them for every name moved here.
  - `encoding/*`: Phases 5a and 5b added `compute_object_vector_rate(s)`, `ObjectVectorRate(s)Result`, `is_egocentric_object_vector_cell`, `has_place_field` and the five `*_significance` functions; `stats` gained `shuffle_spike_times_circular`.

**Contracts referenced:**

- [API snapshot](shared-contracts.md#api-snapshot) — implemented exactly in 6a.6.
- [Overview decisions 1 and 8](overview.md#settled-design-decisions) — no shims; the snapshot replaces governance.

## Tasks

**6a.1 Remove `main`'s governance remnants.**

- Delete `docs/plans/public-api-curation/` (`PLAN.md`, `TASKS.md`; the head commit `da631a47` added them) and its row in `docs/plans/README.md` (:11).
- Delete `tests/test_sparse_init_exports.py`. It pins root `__all__` plus a removed-names list, and the snapshot replaces it (decision 8).
- Keep `tests/test_lazy_imports.py` (lazy-loading behaviour) and `tests/test_package_imports.py` (each `__all__` name resolves). Update their name lists to 6a.2.
- The local `docs/plans/scientific-data-integrity/` and `tests/public_api/` contain only untracked `__pycache__`. Nothing is tracked, so take no action, and do not commit them.

**6a.2 Curate namespaces.** Rules:

1. The root holds the core spatial types, the public exceptions, the domain submodules, and the flagship four-array path and its results.
2. A domain namespace exports user-facing functions and types only. Plumbing, normalizers and internal containers leave `__all__`.
3. One object has one name: aliases are deleted, not kept.
4. No exported name equals a sibling submodule.
5. No deprecation shims (decision 1).

The archive's pruning was reviewed and is **not** followed where it removed user-facing names: it dropped the four `is_*_cell` predicates, `heading_from_velocity`, `PositionOverlay` and every simulation model.

Each removal says exactly what happens to the object. **Drop from `__all__`** means: remove the name from that namespace's `__all__` and its `__init__.py` import; the object stays in its private or owning module for internal use. **Delete** means: the object and its tests go.

| Namespace | `main` | After 6a (expected) | Change |
| --- | --- | --- | --- |
| root | 28 (29 after Phase 4a) | 33 | **Drop from `__all__`/`_LAZY_ATTRS`:** `BayesianDecoder`, `bin_spikes_in_time` (both stay in `decoding`), `SpikeTrains` (stays in `encoding`), `restrict` (stays in `behavior`). **Add** (lazy) `compute_spatial_rate`, `compute_spatial_rates`, `SpatialRateResult`, `SpatialRatesResult`, `decode_position`, `DecodingResult`, `peri_event_histogram`, `PeriEventResult`. **Keep** `NeurospatialError`. **Keep for now** `Session` and `load_session`: Phase 6c deletes them together with `recording.py`, so documents that use them keep running until the replacement recipe lands. |
| encoding | 64 (73 after Phase 5b¹) | 71 | **Drop from `__all__`:** `as_spike_trains`, `as_spike_trains_with_ids` (internal normalizers; they stay in `encoding/_spikes.py`). **Rename** `phase_precession` → `compute_phase_precession` (the function shadows the `encoding.phase_precession` submodule) |
| decoding | 35 | 34 | **Delete** `poisson_likelihood` (`decoding/likelihood.py:221`): its own docstring says it under/overflows and to prefer `log_poisson_likelihood`, and nothing in `src/` calls it. Delete its tests in `tests/decoding/test_likelihood.py` and its mentions in the module docstring (:12) and `log_poisson_likelihood`'s See Also (:118) |
| behavior | 69 | 68 | **Delete** the alias `integrated_absolute_rotation` (`behavior/vte.py:316` `= head_sweep_magnitude`) |
| events | 18 | 16 | **Drop from `__all__` and rename** `validate_events_dataframe`, `validate_spatial_columns` → `_validate_…` in `events/_core.py`; update call sites and tests |
| ops | 67 | 65 | **Delete** the alias `Affine3D` (`ops/transforms.py:1045`). **Drop from `__all__`** `clear_kdtree_cache` (cache plumbing; the function stays in `ops/binning.py`, where its docstring example and `tests/ops/test_binning.py:37` already import it) |
| stats | 28 (29 after Phase 5b²) | 29 | **Delete** the surrogate re-export "for backward compatibility" at `stats/shuffle.py:68-73` and its docstring note at :27-29. Update the 13 imports of `generate_poisson_surrogates` from `neurospatial.stats.shuffle` in `tests/decoding/test_shuffle.py` (:1261–1420) to `neurospatial.stats.surrogates` |
| simulation | 23 | 23 | none here (6b and 6c change signatures and fields) |
| io | 6 | 6 | none |
| io.nwb | 22 | 22 | none here (6c adds the holders) |
| animation | 31 | 16 | **Drop from `__all__`** (they stay where they are, for internal use): <br>• `PositionData`, `BodypartData`, `HeadDirectionData`, `VideoData`, `EventData`, `TimeSeriesData`, `ObjectVectorData` (docstrings say "internal container … should not be instantiated by users"); <br>• `OverlayProtocol` (its `convert_to_data` must return those internal types, so it is not a usable public extension point); <br>• `VideoReaderProtocol` (only used inside `VideoData`); <br>• `VideoCalibration` (canonical in `ops`); <br>• `add_scale_bar_to_axes`, `compute_nice_length`, `configure_napari_scale_bar`, `format_scale_label` (renderer helpers; `ScaleBarConfig` stays). <br>**Delete** the alias `SpikeOverlay` (`animation/overlays.py:1472` `= EventOverlay`) |
| regions | 10 | 10 | none |
| layout | 14 | 14 | none |
| annotation | 12 | 12 | none |

¹ Phase 5a: `compute_object_vector_rate(s)` (+2), `is_egocentric_object_vector_cell` (+1), the `ObjectVectorRate(s)Result` renames (net 0). Phase 5b: `has_place_field` (+1) and the five `*_significance` functions (+5). ² Phase 5b's `shuffle_spike_times_circular`.

**Totals.** The snapshot rendered on `main` has 427 lines (measured twice: the original dry run and this plan's remediation run, same per-namespace counts as the `main` column). Expected: 438 at the start of this phase, 419 after it, and 420 after Phase 6c. These are expectations, not targets. Phases 1–5 may add or move names the table doesn't know about; the regenerated snapshot diff is the truth, and the PR description explains every line of it.

**Internal imports of the dropped normalizers.** Code that imports `as_spike_trains` or `as_spike_trains_with_ids` from the public `neurospatial.encoding` path must switch to `neurospatial.encoding._spikes`: `io/pynapple.py:118`, `recording.py:221` and `:345` (deleted in 6c, but it must import until then), and any call site Phases 1–5 added (`grep -rn "from neurospatial.encoding import.*as_spike_trains" src`). Delete the tests that assert the public import, for example the class `TestAsSpikeTrainsPublic` in `tests/decoding/test_decode_session.py` (about :663-685). `tests/encoding/test_encoding_spikes.py` already imports from `_spikes`.

**Other imports of removed root names.** `tests/encoding/test_spike_trains.py:26` and `tests/test_recording.py:33` (`from neurospatial import SpikeTrains`), `tests/behavior/test_epochs.py:388` (`neurospatial.restrict`), the `SpikeTrains` doctests at `encoding/spike_trains.py:99, :257`, and `docs/user-guide/interoperability.md:170, :198`. Point each at the owning namespace. Delete every test assertion that a removed name is in an `__all__`.

**Root `__init__.py` changes:**

- delete the eager `from neurospatial.decoding import bin_spikes_in_time`, which also stops `decoding` loading at import;
- set `_LAZY_ATTRS` to the eight flagship names (`"compute_spatial_rate": ("encoding.spatial", "compute_spatial_rate")`, …, `"PeriEventResult": ("events._core", "PeriEventResult")`; the existing entries use module paths relative to the package) plus `Session` and `load_session` until 6c;
- rewrite the module docstring's "Core Classes" and "Import Patterns" sections to list the new root. Today they list only the core classes and import `SpikeOverlay`.

**6a.3 Type the lazy root exports** (archive `208932ae`, re-implemented for this root).

- Add an `if TYPE_CHECKING:` block in `src/neurospatial/__init__.py` with an explicit re-export for each entry of `_LAZY_ATTRS` (`from neurospatial.encoding.spatial import compute_spatial_rate as compute_spatial_rate`, …), so mypy, pyright and IDEs resolve the concrete objects while runtime stays lazy (PEP 562).
- Add `tests/typing/top_level_lazy_exports.py`, a static contract: under `if TYPE_CHECKING:`, `assert_type(neurospatial.SpatialRatesResult, type[ConcreteSpatialRatesResult])` for each class, and `assert_type` on a call's return type for each function (for example `assert_type(compute_spatial_rate(env, s, t, p), SpatialRateResult)`).
- CI: `test.yml`'s mypy job runs only `uv run mypy src/neurospatial`. Add a step `uv run mypy tests/typing/top_level_lazy_exports.py`, and add the same command to the type-checking block of the definition of done in this PR's description.
- Extend `tests/test_lazy_imports.py` (as the archive commit did, for the new names): each lazy attribute is absent from the package globals until first access, appears in `dir(neurospatial)`, and is the owning module's object after access; `import neurospatial` does not import `neurospatial.decoding`.

**6a.4 Root result classes pass the docstring-sections test.** Phase 4b's `tests/docs/test_docstring_sections.py` checks every root-exported callable. Its class rule: an Examples section, and every constructor parameter named under Parameters, Other Parameters or Attributes. The four result classes added to the root become subject to it. Measured on `main` with a section parser applying that rule (re-measure after Phases 3–5, which add fields such as `spike_window` and `direction_frame`):

| Class | Constructor parameters | Not documented |
| --- | --- | --- |
| `SpatialRateResult` | 17 | 10 GLM fields: `penalty`, `penalty_weights`, `rank`, `deviance`, `converged`, `n_iter`, `reml_objective`, `reml_at_boundary`, `penalty_selected_by_reml`, `pooled` |
| `SpatialRatesResult` | 18 | 0 |
| `DecodingResult` | 3 | 0 |
| `PeriEventResult` | 7 | 0 |

All four have an Examples section on `main`. Document the missing `SpatialRateResult` fields under Attributes ("set by `method='glm'`; `None` otherwise"), as `SpatialRatesResult` already does. The four functions (`compute_spatial_rate(s)`, `decode_position`, `peri_event_histogram`) already pass on `main` and are in `FLAGSHIP`.

**6a.5 Remove the remaining deprecation shims** (decision 1; Phase 4a removed the `detect_region_crossings` old-order dispatch and Phase 5b the classifier aliases):

- `ImageMaskLayout`'s `bin_size` alias (`layout/engines/image_mask.py:72`, handling at :97-130);
- `ViewRateResult.peak_view_location` (`encoding/view.py:263`) and `ViewRatesResult.peak_view_location` (:820).

Delete their warning tests.

**6a.6 API snapshot.** Add `tests/test_public_api_snapshot.py` exactly as below, then generate `tests/data/public_api.txt` with `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py` and commit both. This code (with the 14 namespaces) was run against `main`: it rendered 427 lines, identically across two processes, and imported neither `pynwb` nor `napari`, so it runs in the default `--extra dev` CI job. The mismatch path printed a unified diff and the update command.

```python
"""Pin the public API: every ``__all__`` name of every public namespace."""

from __future__ import annotations

import difflib
import importlib
import inspect
import os
import re
from pathlib import Path

NAMESPACES = (
    "neurospatial", "neurospatial.encoding", "neurospatial.decoding",
    "neurospatial.behavior", "neurospatial.events", "neurospatial.ops",
    "neurospatial.stats", "neurospatial.simulation", "neurospatial.io",
    "neurospatial.io.nwb", "neurospatial.animation", "neurospatial.annotation",
    "neurospatial.regions", "neurospatial.layout",
)
SNAPSHOT = Path(__file__).parent / "data" / "public_api.txt"
UPDATE_ENV = "NEUROSPATIAL_UPDATE_API_SNAPSHOT"
UPDATE_CMD = f"{UPDATE_ENV}=1 uv run pytest tests/test_public_api_snapshot.py"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _describe(obj: object) -> str:
    """Return the snapshot suffix for one exported object."""
    if inspect.ismodule(obj):
        return " (module)"
    if inspect.isclass(obj):
        return " (class)"
    if not callable(obj):
        return " (constant)"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(<signature unavailable>)"
    # Annotation text depends on the Python and NumPy versions (NDArray's repr),
    # so it is dropped; names, kinds, defaults and order are what break callers.
    sig = sig.replace(
        parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )
    return _ADDRESS.sub("", str(sig))


def render_public_api() -> str:
    """Render the sorted snapshot text for every namespace in ``NAMESPACES``."""
    lines = []
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            lines.append(f"{namespace}.{name}{_describe(getattr(module, name))}")
    return "\n".join(sorted(lines)) + "\n"


def test_public_api_matches_snapshot() -> None:
    current = render_public_api()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        return
    expected = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if current != expected:
        diff = "".join(difflib.unified_diff(
            expected.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile="tests/data/public_api.txt (committed)",
            tofile="public API (this checkout)",
        ))
        raise AssertionError(
            "The public API differs from tests/data/public_api.txt.\n"
            f"{diff}\nIf this change is intended, regenerate the snapshot and "
            f"commit it:\n    {UPDATE_CMD}"
        )
```

After generating it, run Phase 4b's executable-docs test (`uv run pytest tests/docs -n 4`), the full suite (`uv run pytest -n 4 -rs`) and the slow tests (`uv run pytest -m "slow and not napari" -n 4`). Every failure is a doc or test that uses a name changed here: fix the caller, never re-export.

**6a.7 Documentation** (part of this PR; each commit adds its own CHANGELOG bullet per [executing.md](executing.md), and this task checks they are all present):

- `CLAUDE.md`: patterns that import from the root; the v0.6 naming-contract lines that mention removed names; the "Most Common Patterns" imports of `compute_spatial_rate`, `decode_position` and `peri_event_histogram` may now use the root.
- `.claude/QUICKSTART.md` (:879 `phase_precession()` → `compute_phase_precession()`) and `.claude/API_REFERENCE.md` (`SpikeOverlay` → `EventOverlay`, `poisson_likelihood`, `clear_kdtree_cache`, `as_spike_trains`, `integrated_absolute_rotation`, `Affine3D`, the animation containers).
- `docs/api/index.md` (`SpikeOverlay` and the other removed names).
- `docs/user-guide/interoperability.md` (:170, :198 root imports of `SpikeTrains` and `restrict`).
- `CHANGELOG.md` `[Unreleased]` `### Removed` / `### Changed` (breaking): every removal, deletion, rename and root change in 6a.2–6a.5, and the new root exports under `### Added`.

## Deliberately not in this phase

- **Argument order, the swap-detecting validator, `PositionLike` removal and `BayesianDecoder.fit(unit_ids=)`.** Phase 6b.
- **Data holders, deleting `Session`/`load_session`/`recording.py`.** Phase 6c.
- **"Did you mean" `__getattr__` redirects for removed names.** Decision 1, and Phase 4b's executable docs catch internal users.
- **Renaming `EgocentricPolarEnvironment`, `view_spatial_information`, or the head-direction family.** Not convention violations under the contract; separate naming work.
- **Adding `stats.shuffle_*`, `generate_*` or `regions` functions to `FLAGSHIP`.** Phase 4b recorded them as follow-ups.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_public_api_snapshot.py::test_public_api_matches_snapshot` | rendered text equals `tests/data/public_api.txt` (expected 419 lines after this phase; 427 when rendered on `main`); deleting a line from the file fails with a unified diff and the `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1` command |
| `tests/test_lazy_imports.py` (updated) | `import neurospatial` does not import `neurospatial.decoding`; each of the eight flagship names is deferred until first access and `neurospatial.decode_position is neurospatial.decoding.decode_position`; `dir(neurospatial)` lists them |
| `uv run mypy tests/typing/top_level_lazy_exports.py` (CI step) | every `assert_type` holds; removing the `TYPE_CHECKING` block makes it fail (check once by hand and record it in the PR) |
| `tests/test_package_imports.py` (unchanged) | every `__all__` name resolves |
| `tests/decoding/test_shuffle.py` | passes with the surrogate imports pointed at `neurospatial.stats.surrogates` |
| Phase 4b `tests/docs/test_docstring_sections.py` | 0 missing sections and 0 undocumented parameters, now including the four root result classes |
| Phase 4b executable-docs and flagship-docstring tests | pass with every doc updated in 6a.7 |

## Fixtures

None. The snapshot imports the package.

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- Every task is implemented as specified, and each removal did what its row says (drop from `__all__` versus delete).
- The "Deliberately not in this phase" list is honored.
- The snapshot diff in the PR matches the table, and every unexpected line is explained.
- Tests aren't trivial: they exercise the asserted behavior, not tautologies (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are actually removed (aliases, shims, the surrogate re-export, `test_sparse_init_exports.py`).
- User-facing documentation listed as tasks is updated, not deferred.
