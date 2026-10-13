# Phase 3 — Save and load the binning; never rebuild on load

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#file-format)

Files and NWB records store the binning itself. Loading decodes it and builds a
`BinningLayout`. It never runs a layout builder. Old files fail with an
actionable message. This fixes NWB read-back: `boundary_bins` works again, and
tracks can call `to_linear`.

**Inputs to read first:**

- [src/neurospatial/io/files.py:44](../../../src/neurospatial/io/files.py#L44) — `_SCHEMA_VERSION`.
- [src/neurospatial/io/files.py:239-390](../../../src/neurospatial/io/files.py#L239-L390) — `to_file`: the JSON metadata, node-link graph, regions and NPZ arrays.
- [src/neurospatial/io/files.py:392-548](../../../src/neurospatial/io/files.py#L392-L548) — `from_file` and `_restore_environment`, which currently call `_from_saved_layout`.
- [src/neurospatial/io/files.py:550-683](../../../src/neurospatial/io/files.py#L550-L683) — `to_dict` and `from_dict`.
- [src/neurospatial/environment/factories.py:1461-1555](../../../src/neurospatial/environment/factories.py#L1461-L1555) — `_from_saved_layout`, the rebuild path this phase deletes.
- [src/neurospatial/io/nwb/_environment.py:114](../../../src/neurospatial/io/nwb/_environment.py#L114) — `write_environment`.
- [src/neurospatial/io/nwb/_environment.py:516](../../../src/neurospatial/io/nwb/_environment.py#L516) — `read_environment`.
- [src/neurospatial/io/nwb/_environment.py:715-995](../../../src/neurospatial/io/nwb/_environment.py#L715-L995) — `_reconstruct_environment`, `_ReconstructedLayout` and `_reconstruct_graph`; their rebuilt nodes cause the `boundary_bins` defect.
- [src/neurospatial/environment/core.py:1280](../../../src/neurospatial/environment/core.py#L1280) — `copy()`, which must work for facade layouts.
- [docs/contributing.md:130-143](../../contributing.md) — the construction-path paragraph this phase rewrites.

**Contracts referenced:**

- [Codec contract](shared-contracts.md#codec-contract) — implement the codecs for every built-in part. Do not weaken "decoding never runs a builder".
- [BinningLayout facade](shared-contracts.md#binninglayout-facade) — implement it in full.
- [FacesSource](shared-contracts.md#facessource) — add `KnownFaces` and `MissingFaces`.
- [Publication on the environment](shared-contracts.md#publication-on-the-environment) — `binning_from_layout` returns `layout.binning()` for the facade.

**Designs referenced:** [designs.md#file-format](designs.md#file-format), [designs.md#plotting-from-binning-data](designs.md#plotting-from-binning-data) (the facade's `plot` choice).

## Tasks

- **Record the baseline** on the base commit, as in phase 1. Also write one
  `Environment-v1` file and one `"1.0"` NWB record to `/tmp`, for the error test
  later.
- **Add `KnownFaces` and `MissingFaces`** to `layout/_binning.py`.
- **Create `src/neurospatial/layout/_codecs.py`** following the codec contract.
  - Write codecs for `SpatialBins`, `FaceGeometry` (as `KnownFaces`),
    `MissingFaces`, `RectilinearGrid`, `TrackLinearization`, `HexagonShapes`,
    `TriangleShapes`, `Outline` (as WKB bytes in a `uint8` array), the four
    data locators and `Binning`.
  - `TrackLinearization` encodes its graph as node-link JSON in meta, the same
    way `to_file` already encodes graphs.
  - `LayoutLocator` and the delegating `TrackLinearization` (a layout) have no
    codec. Encoding them raises the unregistered-kind `ValueError`, which names
    the layout's class.
- **Create `src/neurospatial/layout/_binning_layout.py`** with `BinningLayout`
  as specified. It satisfies the `LayoutEngine` protocol structurally; add a
  check to `tests/typing/`. `binning_from_layout` returns `layout.binning()`
  when the layout defines it.
- **Rewrite files and dicts** (`io/files.py`) to the
  [v2 format](designs.md#file-format):
  - Add `"layout_category"` (`env.layout_type`) and `"binning"` to the metadata.
  - Faces are built at save time.
  - `_restore_environment` decodes, builds `BinningLayout`, and constructs the
    environment once through the normal constructor, so `_setup_from_layout`
    still validates the graph.
  - Reading `"Environment-v1"` raises `ValueError` with What / Why / Fix.
  - Bump `_SCHEMA_VERSION` to `"Environment-v2"`.
- **Rewrite NWB** (`io/nwb/_environment.py`) to the same payload:
  - Store the meta as one JSON string attribute and the arrays as datasets.
  - Store the node-link graph JSON in place of the bare edge list and weights.
  - Bump the record schema to `"2.0"`. A `"1.0"` record raises `ValueError`
    with What / Why / Fix.
  - Delete `_reconstruct_environment`'s two branches, `_ReconstructedLayout`
    (`:854-995`) and `_reconstruct_graph` (`:997`). Reading builds a
    `BinningLayout`.
- **Delete the rebuild path:**
  - Remove `_from_saved_layout` (`factories.py:1461-1555`), and its helper
    `_center_tolerance` if nothing else uses it; `grep` before deleting.
  - Once `_ReconstructedLayout` is gone, no layout defines
    `_explain_missing_geometry`, so drop that lookup from
    `_missing_geometry_error`. The saved reason text now travels as
    `MissingFaces`.
  - Tests that only exercise the rebuilt-centers comparison (the
    center-tolerance fixes `c46f0273`, `cc2d4762` and `7546ae52`) are replaced
    by round-trip tests.
  - Keep the non-finite-centers rejection from `99474b3d`. It is now enforced
    by `SpatialBins.from_arrays`, re-raised with a message that names the file.
- **Check `copy()`** (`core.py:1280`) on a facade environment. If `copy()` reads
  build parameters, change it to copy the layout object and republish, without
  running a builder.
- **Docs:**
  - Rewrite `docs/contributing.md:130-143` to describe the new path: save the
    binning, decode it, use the facade, and never rebuild.
  - Update the docstrings of `to_file`, `from_file`, `to_dict`, `from_dict`,
    `write_environment` and `read_environment`. They should say what is stored,
    that loading doesn't rebuild, and that older versions are refused.
  - Add a `CHANGELOG.md` entry:
    - **Changed:** the file format is v2 and the NWB environment record is 2.0;
      older versions are refused.
    - **Fixed:** `boundary_bins` and `to_linear` after NWB read-back.
  - Update `.claude/ADVANCED.md` (NWB section) if it describes the stored
    fields.
- **Compare the baseline.** The expected differences are:
  - NWB read-back probes for `boundary_bins` and track `to_linear`, which
    errored before and now match the original environment;
  - any probe that records `type(env.layout).__name__` for read-back
    environments.

  List every differing probe in the PR and verify each one against the
  original environment's output.

## Deliberately not in this phase

- **Saving derived environments' wrapper locators** (`SubsetLocator`,
  `TransformedLocator`) is phase 4. Here subset and transformed environments
  still use `_SubsetLayout` and `TransformedLayout`, whose `LayoutLocator`
  can't be encoded. Saving them raises the unregistered-kind error. Today they
  also can't be reloaded, because the rebuild fails, so this is no regression;
  add a test that pins the error.
- **Third-party codec registration through entry points** is roadmap stage B.
- **Arena data in files** is roadmap stage A.
- **The binning spec for `rebin`.** `rebin` keeps reading the tag and the grid
  (`environment/transforms.py:28-187`), and the facade reports the original tag,
  so loaded regular grids still rebin. Separating the spec from provenance is
  roadmap stage D.
- **A converter for v1 files.** There are no users to migrate.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_codec_round_trip` (parametrized over every codec kind) | `decode(encode(x))` gives `np.array_equal` arrays and identical meta |
| `test_unregistered_kind_refuses` | Encoding a `LayoutLocator` raises `ValueError` naming the layout class and the fix |
| `test_file_round_trip_identical` (parametrized over the ten types × file, dict, NWB; NWB cases marked `nwb`) | After the round trip, every binning array, the locator result on the parity probes, the faces, components, `connectivity` edges with their data, regions, units and frame equal the original's |
| `test_load_never_builds` (parametrized) | With every layout class's `build` patched to raise, `from_file`, `from_dict` and `read_environment` still succeed |
| `test_nwb_read_back_boundary_bins` | `boundary_bins` after NWB read-back equals the original environment's for the grid, polygon, mask, pixel-mask and polar types |
| `test_nwb_read_back_track_to_linear` | `to_linear` after NWB read-back equals the original's on 500 seeded points |
| `test_loaded_environment_plots_like_original` (slow) | Artists from `plot()` and `plot_field()` on a loaded environment equal the original's (same record as phase 2b) |
| `test_loaded_environment_smooths_like_original` (parametrized) | `smooth(field, 5.0, mode=m)` is identical for all three modes |
| `test_old_versions_refused` | The v1 file and the 1.0 NWB record raise `ValueError` whose message names the version and the fix |
| `test_copy_of_loaded_environment` | `loaded.copy()` equals `loaded` on bins, locator results and smoothing |
| `test_derived_environment_save_refused_until_supported` | `to_file` on a translated environment raises the unregistered-kind error (replaced in phase 4) |
| baseline compare (manual) | only the listed differences |
| full suite, doctests, `tests/docs`, `tests/nwb` with the NWB extra, API snapshot, ruff, mypy | all pass |

## Fixtures

- Reuse `built_environment`.
- NWB tests use the existing helpers in `tests/nwb/_nwb_helpers.py` and need
  the NWB extra.
- Commit the v1 file and the 1.0 record from the first task as small fixtures
  under `tests/io_tests/data/` and `tests/nwb/data/`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or an equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored: no scope creep into
  adjacent phases.
- The validation slice tests pass, and slow or integration tests are marked.
- The tests aren't trivial. They exercise the asserted behavior, not
  tautologies: no `assert True`, and no assertions that only check the mock the
  test just configured. Shared setup is in fixtures, not copied between tests.
  (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names and module names don't reference this plan or its
  phases.
- The old paths are gone: `_from_saved_layout`, `_ReconstructedLayout`,
  `_reconstruct_graph` and the `_explain_missing_geometry` lookup.
- The user-facing documentation listed as tasks is updated: the I/O
  docstrings, `CHANGELOG.md` and `docs/contributing.md`.
