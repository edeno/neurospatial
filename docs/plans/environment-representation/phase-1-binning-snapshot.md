# Phase 1 — Publish one binning record per environment

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#binning-from-a-layout)

Every environment publishes a `Binning` (bins, locator, faces source,
components) when it's built. Environment code reads bin sizes, point lookup and
capabilities from it instead of asking the layout. Outputs don't change. The
locator and faces still delegate to the layout. Later phases replace those
delegates with data.

**Inputs to read first:**

- [src/neurospatial/environment/core.py:71-85](../../../src/neurospatial/environment/core.py#L71-L85) — `_PublishedGeometry`. It gains the `binning` field.
- [src/neurospatial/environment/core.py:921-1041](../../../src/neurospatial/environment/core.py#L921-L1041) — `_setup_from_layout`, the single place where geometry is published. The binning is built here.
- [src/neurospatial/environment/core.py:439-446](../../../src/neurospatial/environment/core.py#L439-L446) — the placeholder geometry in `__init__`.
- [src/neurospatial/environment/queries.py:196-239](../../../src/neurospatial/environment/queries.py#L196-L239) — `bin_sizes`, which currently returns `self.layout.bin_sizes()`.
- [src/neurospatial/environment/cartesian.py:131](../../../src/neurospatial/environment/cartesian.py#L131) and [:177](../../../src/neurospatial/environment/cartesian.py#L177) — `bin_at` and `contains`, which call `self.layout.point_to_bin_index`.
- [src/neurospatial/environment/cartesian.py:320-336](../../../src/neurospatial/environment/cartesian.py#L320-L336) and [:405-418](../../../src/neurospatial/environment/cartesian.py#L405-L418) — the `TrackLinearizationProvider` checks in `to_linear` and `linear_to_nd`.
- [src/neurospatial/environment/cartesian.py:1044-1049](../../../src/neurospatial/environment/cartesian.py#L1044-L1049) — the `RectilinearGridProvider` check in `subset`.
- [src/neurospatial/environment/fields.py:57-89](../../../src/neurospatial/environment/fields.py#L57-L89) — `_missing_geometry_error`.
- [src/neurospatial/environment/fields.py:178](../../../src/neurospatial/environment/fields.py#L178) and [:227-262](../../../src/neurospatial/environment/fields.py#L227-L262) — `_interpolate_linear(env.layout, ...)` and its `require_rectilinear_grid` call.
- [src/neurospatial/environment/fields.py:790-844](../../../src/neurospatial/environment/fields.py#L790-L844) — `_face_geometry`, `_require_face_geometry`, `_diffusion_geometry`.
- [src/neurospatial/environment/trajectory.py:360-369](../../../src/neurospatial/environment/trajectory.py#L360-L369) — `require_rectilinear_grid(env.layout, ...)` for linear time allocation.
- [src/neurospatial/layout/_capabilities.py:20-84](../../../src/neurospatial/layout/_capabilities.py#L20-L84) — `RectilinearGrid`, the provider protocols, and `require_rectilinear_grid`, whose error text must be preserved.
- [src/neurospatial/layout/_diffusion.py:28-110](../../../src/neurospatial/layout/_diffusion.py#L28-L110) — `FaceGeometry`, reused unchanged.
- [docs/contributing.md:96-226](../../contributing.md) — the architecture rules this phase updates.
- [scripts/environment_baseline.py:1-19](../../../scripts/environment_baseline.py) — record and compare usage.

**Contracts referenced:**

- [Identity and ownership rules](shared-contracts.md#identity-and-ownership-rules) — all new types.
- [SpatialBins](shared-contracts.md#spatialbins) — implement it in full, including validation.
- [FacesSource](shared-contracts.md#facessource) — `LayoutFaces` only.
- [Locator](shared-contracts.md#locator) — the protocol and `LayoutLocator` only.
- [Binning](shared-contracts.md#binning) — `shapes` and `outline` stay `None` until phase 2b.
- [Publication on the environment](shared-contracts.md#publication-on-the-environment) — do not weaken: no provider `isinstance` checks remain in `src/neurospatial/environment/`.

**Designs referenced:** [designs.md#binning-from-a-layout](designs.md#binning-from-a-layout).

## Tasks

- **Record the baseline first.** On the base commit, run
  `uv run python scripts/environment_baseline.py record /tmp/env-base.npz`.
  Also record construction time, median of 7 runs, for:
  - `Environment.from_samples` on 200,000 uniform points with `bin_size=0.5`
    over a 150 cm square (about 90,000 bins);
  - a hexagonal `from_samples` environment of about 20,000 bins;
  - `linear_track` with `bin_size=0.5` over 1,000 cm.
- **Create `src/neurospatial/layout/_binning.py`** with these parts:
  - `SpatialBins`, with `from_arrays` validation and read-only owned arrays.
  - The `Locator` protocol and `LayoutLocator`.
  - The `FacesSource` protocol and `LayoutFaces`.
    - `LayoutFaces.build()` holds the body of `_face_geometry`
      (`fields.py:798-812`) unchanged.
    - `LayoutFaces.missing_error()` returns the result of the existing
      `_missing_geometry_error`. Move that function from `fields.py:57-89` into
      `_binning.py`, and keep its message text byte-identical.
  - `Binning`.
  - `binning_from_layout`, as in
    [designs.md](designs.md#binning-from-a-layout).

  The module imports only NumPy, NetworkX and `neurospatial.layout` siblings,
  because it is foundation tier.
- **Add `require_rectilinear(binning, operation, *, fix) -> RectilinearGrid`**
  to `layout/_capabilities.py`. It produces the same message as
  `require_rectilinear_grid` (`_capabilities.py:60-84`), using `binning.kind`
  for the layout name. Delete `require_rectilinear_grid`: its only callers are
  `fields.py:259` and `trajectory.py:365`.
- **Publish the binning:**
  - Add `binning: Binning | None` to `_PublishedGeometry` (`core.py:71-85`).
  - Pass `binning=None` in the placeholder (`core.py:439-446`).
  - In `_setup_from_layout`, build it with
    `binning_from_layout(self.layout, connectivity)` after validation
    (`core.py:958-993`), and include it in the `_PublishedGeometry(...)` call
    (`core.py:1018-1025`).
  - Add a read-only property `_binning` to `_BaseEnvironment` that asserts the
    binning isn't `None`.
  - Add `_binning` to `EnvironmentProtocol` and to each consumer protocol in
    `src/neurospatial/environment/_protocols.py` that the routed methods use.
- **Route the environment's internal reads through the binning**:
  - `queries.py:239`: `return self._binning.bins.sizes`.
  - `cartesian.py:131` and `cartesian.py:177`: `self._binning.locator.locate(points_nd)`.
  - `cartesian.py:327-336` and `:409-418`:
    `linearization = self._binning.linearization`. If it's `None`, raise the
    same `AttributeError` text, with `self._binning.kind` replacing
    `self.layout._layout_type_tag`. Otherwise return
    `linearization.to_linear(...)` or `linearization.linear_to_nd(...)`.
  - `cartesian.py:1048`: `is_grid_env = self._binning.rectilinear is not None`.
  - `fields.py:178`: call `_interpolate_linear(env._binning, field, points)`.
    At `fields.py:259`, use `require_rectilinear(binning, ...)`.
  - `trajectory.py:365`: `require_rectilinear(env._binning, ...)`.
  - `fields.py:790-831`: `_face_geometry` returns
    `self._binning.faces.build()` and stays a `versioned_cached_property`, so
    caching and `clear_cache` are unchanged. `_require_face_geometry` raises
    `self._binning.faces.missing_error(operation)` when
    `not self._binning.faces.available`.
  - Remove the now-unused imports of `RectilinearGridProvider`,
    `TrackLinearizationProvider` and `FiniteVolumeGeometryProvider` from
    `environment/cartesian.py` and `environment/fields.py`.
- **Tests:**
  - New `tests/layout/test_binning.py`: the `SpatialBins` invariants and
    ownership, and the identity semantics.
  - New `tests/environment/test_binning_publication.py`: publication parity on
    every built-in environment type (see Fixtures).
- **Docs:**
  - In `docs/contributing.md`, rewrite the paragraph at lines 114-117
    (capability checks) and the paragraph at lines 203-212 ("Layout-specific
    operations ..."). They should now say:
    - the environment publishes one `Binning`, and environment code checks its
      components (`rectilinear`, `linearization`, `faces.available`);
    - layouts are asked about capabilities only by `binning_from_layout`;
    - new code must not use `isinstance(env.layout, ...)` checks.
  - Update lines 145-150 to say that `_face_geometry` builds from
    `env._binning.faces`.
  - Add a `CHANGELOG.md` entry under Unreleased → Changed: "`Environment.bin_sizes`
    now returns a read-only array, like `bin_centers`."
- **Compare the baseline.** Run
  `uv run python scripts/environment_baseline.py record /tmp/env-new.npz` and
  then `compare /tmp/env-base.npz /tmp/env-new.npz`. It must report no
  differences. Re-measure the three construction times; each must be within
  10% of the baseline, or the PR description must report the measured delta and
  its cause.

## Deliberately not in this phase

- **Data-backed locators and `TrackLinearization` data** are phase 2a. Here the
  locator and linearization still delegate to the layout.
- **Bin shapes, outlines and the plotting moves** are phase 2b.
- **The `BinningLayout` facade, codecs and file format** are phase 3.
- **Subset and transform rewrites** are phase 4. `_SubsetLayout` and
  `TransformedLayout` stay, and receive a binning through
  `binning_from_layout` like any layout.
- **The roughly 80 `env.layout` readers in visualization, animation,
  encoding, metrics and NWB** stay as they are. They are migrated at roadmap
  stage F.
- **The `LayoutEngine` protocol (`layout/base.py:83`)** isn't changed.
- **No public name is added or renamed.** `tests/data/public_api.txt` must not
  change.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_spatial_bins_rejects_invalid_arrays` (parametrized) | `from_arrays` raises `ValueError` naming the index for: NaN center, zero or negative size, wrong `sizes` shape, `measure_dim` of 0 or greater than `n_dims`, pair index out of range, self pair, duplicate unordered pair, negative distance |
| `test_spatial_bins_owns_read_only_arrays` | Changing the input arrays after construction doesn't change `bins`; writing to `bins.centers[0, 0]` raises `ValueError` |
| `test_spatial_bins_identity_semantics` | `{bins: 1}` works; two bins built from equal arrays compare unequal |
| `test_binning_matches_layout` (parametrized over the ten environment types) | `centers == env.bin_centers`; `sizes == layout.bin_sizes()`; `pairs` equals `list(env.connectivity.edges())` in order; `pair_distances` equals the edge `"distance"` values; `measure_dim` is 1 for the track and `n_dims` otherwise |
| `test_binning_capabilities_match_layout_protocols` (parametrized) | `rectilinear is not None` iff the layout is a `RectilinearGridProvider`; `linearization is not None` iff it's a `TrackLinearizationProvider`; `faces.available` iff it's a `FiniteVolumeGeometryProvider` |
| `test_binning_published_once` | `env._binning` is the same object on repeated access; it isn't `None` after any factory |
| `test_missing_faces_message_unchanged` | Smoothing on an environment read back from a non-grid NWB record raises the same message as before (`_explain_missing_geometry`, `io/nwb/_environment.py:903`) |
| `test_require_rectilinear_message_unchanged` | `interpolate(mode="linear")` on a hexagonal environment raises the same text as before |
| baseline compare (manual, recorded in the PR) | zero differences across all probes |
| full suite, doctests, `tests/docs`, `tests/test_public_api_snapshot.py`, ruff, mypy | all pass |

## Fixtures

Add a parametrized fixture `built_environment` to
`tests/environment/conftest.py`. It builds the same ten environment types as
`_base_environments` in `scripts/environment_baseline.py:60-127`:
- 8- and 4-connected grids;
- a polygon arena;
- a holed grid mask;
- a pixel mask;
- a hexagonal environment;
- a branching track;
- a triangular mesh;
- circular and open polar grids.

Use small sizes and fixed seeds. Tests must not import from `scripts/`. The
parity probes are generated per environment from its bounding box, with
`np.random.default_rng(0)`.

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
- Old code paths flagged for removal in this phase (`require_rectilinear_grid`
  and the provider imports in `environment/`) are actually removed.
- The user-facing documentation listed as tasks (`docs/contributing.md`,
  `CHANGELOG.md`) is updated, not deferred.
