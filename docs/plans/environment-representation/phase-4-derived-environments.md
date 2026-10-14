# Phase 4 — Subset and transform by deriving a new binning

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#derivation-algorithms)

`subset` and `apply_transform` compute a new binning from the parent's, under the
[derivation rules](shared-contracts.md#derivation-rules). The child keeps every
feature its data still supports. When it can't keep one, it refuses with the
reason. This fixes four problems:
- a translated grid loses smoothing, gradients and plotting;
- a translated grid keeps its old `grid_edges`;
- a hexagonal subset has the wrong bin sizes and can't smooth;
- derived environments can't be saved.

**Inputs to read first:**

- [src/neurospatial/environment/cartesian.py:960-1049](../../../src/neurospatial/environment/cartesian.py#L960-L1049) — `subset` and its grid or graph dispatch.
- [src/neurospatial/environment/cartesian.py:1051](../../../src/neurospatial/environment/cartesian.py#L1051) (to the end of the file) — `apply_transform`, which delegates to `ops.transforms.apply_transform_to_environment`.
- [src/neurospatial/environment/transforms.py:218-347](../../../src/neurospatial/environment/transforms.py#L218-L347) — `_subset_mask` (kept).
- [src/neurospatial/environment/transforms.py:350-687](../../../src/neurospatial/environment/transforms.py#L350-L687) — `_grid_subset`, `_restrict_edges`, `_SubsetLayout` and `_graph_subset`, which this phase replaces.
- [src/neurospatial/ops/transforms.py:1342-1672](../../../src/neurospatial/ops/transforms.py#L1342-L1672) — `apply_transform_to_environment`:
  - graph update, lines 1484-1503;
  - extent update, lines 1505-1531;
  - `TransformedLayout`, lines 1536-1610, which copies the parent's grid attributes unchanged at lines 1558-1561;
  - region transforms, lines 1622-1661.
- [src/neurospatial/ops/transforms.py:96](../../../src/neurospatial/ops/transforms.py#L96) and [:270](../../../src/neurospatial/ops/transforms.py#L270) — `AffineND` and `Affine2D`, for reading the linear part and translation.
- [src/neurospatial/environment/core.py:1002-1007](../../../src/neurospatial/environment/core.py#L1002-L1007) — the point-based fallback for grid fields.

**Contracts referenced:**

- [Derivation rules](shared-contracts.md#derivation-rules) — do not weaken: the faces for non-similarity transforms of non-rectilinear binnings are `MissingFaces`, and non-similarity transforms of tracks are refused.
- [Locator](shared-contracts.md#locator) — `SubsetLocator` and `TransformedLocator`, plus the optional exact `subset` and `transformed` methods on `RectilinearLocator`.
- [Codec contract](shared-contracts.md#codec-contract) — add the wrapper codecs, which nest their base under `.../base/`.
- [BinningLayout facade](shared-contracts.md#binninglayout-facade) — every derived environment uses it.

**Designs referenced:** [designs.md#derivation-algorithms](designs.md#derivation-algorithms).

## Tasks

- **Record the baseline** on the base commit, as in phase 1.
- **Add the wrapper locators.** In `layout/_locators.py`:
  - Add `SubsetLocator` and `TransformedLocator`. `TransformedLocator.locate`
    applies the stored inverse, then calls the base. This is the method
    `TransformedLayout` uses today (`ops/transforms.py:1566-1574`).
  - Add `RectilinearLocator.subset(keep)` (a new mask) and
    `RectilinearLocator.transformed(t)`. The `transformed` method is defined
    only for a diagonal `A` with every entry positive: it returns a locator
    with edges `a_k e + t_k` in the original order. For reflections or any
    other `A` it returns `None`, and the caller wraps the locator instead.
  - Register codecs for both wrappers. A `TransformedLocator` stores the
    inverse matrix as an `(n+1, n+1)` array.
- **Create `src/neurospatial/layout/_derive.py`** with `subset_binning` and
  `transform_binning`, following
  [designs.md](designs.md#derivation-algorithms).
  - Classify the transform by the stated tolerances.
  - Every refusal raises `ValueError` with What / Why / Fix. One example is a
    track under a non-similarity transform: "rebuild the track from
    transformed node positions with `Environment.from_graph`".
  - Every `MissingFaces` reason names the transform class and the lost right
    angle.
- **Rewrite `subset`** (`cartesian.py:1041-1049`):
  - Compute the mask with `_subset_mask` and call `subset_binning`.
  - Build the facade's legacy fields: the renumbered connectivity from the
    existing code (`transforms.py:599-632`); the grid fields from the kept
    `rectilinear`, or the point-based fallback. Drawing limits encompass full
    retained cells: grid edge bounds for rectilinear subsets, active shape
    bounds for hexagons/triangles, and kept centers as the fallback. Center-only
    bounds clip outer cell geometry and collapse a one-cell grid's limits.
  - Construct the environment, and copy the units and frame.
  - Regions are still dropped, as documented.
  - Delete `_grid_subset` (`:350-396`), `_SubsetLayout` (`:438-593`) and
    `_graph_subset` (`:596-687`). Delete `_restrict_edges` (`:399-435`) unless
    the renumbering reuses it.
- **Rewrite `apply_transform_to_environment`:**
  - Keep its validation (`ops/transforms.py:1440-1477`), its graph and extent
    updates (`:1484-1531`) and its region transforms (`:1622-1661`).
  - Replace `TransformedLayout` (`:1536-1610`) with `transform_binning`
    followed by a `BinningLayout`. The grid fields come from the transformed
    `rectilinear`, or from the point-based fallback when the transform drops
    it. This removes the stale `grid_edges`.
- **Update `copy()`** if phase 3 didn't already cover facade layouts with
  wrapper locators. Test it either way.
- **Remove the phase 3 placeholder test**
  `test_derived_environment_save_refused_until_supported`. Replace it with the
  round-trip tests below.
- **Docs:**
  - Rewrite the docstrings of `subset` and `apply_transform`. They should say
    what each keeps, what each refuses and why, and that a point in a dropped
    bin is outside a subset.
  - Add a `CHANGELOG.md` entry:
    - **Fixed:** translated, rotated and scaled environments keep smoothing,
      gradients and plotting; translated grids report moved `grid_edges`;
      hexagonal and mesh subsets keep their true bin sizes and can smooth;
      derived environments can be saved.
    - **Changed:** points in bins dropped by `subset` now map to -1, where they
      used to map to the nearest kept bin within 10 bin spacings. Non-uniform
      scaling of hexagonal or mesh environments disables smoothing, with a
      reason. Non-similarity transforms of tracks are refused.
  - Add a paragraph on derived environments after `docs/contributing.md:188-190`.
    It should say that `subset` and `apply_transform` derive a binning under the
    rules in `layout/_derive.py`, and that new derivations follow those rules
    and never rebuild through a factory.
- **Compare the baseline.** Expect differences in the transform and subset
  probes:
  - smoothing, gradients and plots now succeed for transformed environments;
  - grid edges are translated;
  - hexagonal, mesh and track subset sizes, smoothing and point lookup change;
  - transformed grid `plot_field` now renders with `QuadMesh` instead of scatter;
  - the signatures and docstrings of `subset` and `apply_transform`, which this
    phase rewrites and the baseline script records.

  List every differing probe in the PR and verify each one against the
  invariant tests below.

## Deliberately not in this phase

- **Clipping bins to a region**, which changes bin sizes and faces, is roadmap
  stage A. `subset` selects whole bins only.
- **Exact faces under general affine maps**, which need face normals and
  Nanson's formula, are roadmap stage E. Here those faces are `MissingFaces`.
- **Regular hexagon shapes under non-similarity transforms.** They're dropped,
  and `plot` uses the center-scatter fallback. Triangle vertices transform
  exactly under every invertible affine map and are retained for drawing;
  this does not supply the missing finite-volume faces needed for smoothing.
- **Units changes from calibration transforms** (pixels to cm) belong to the
  coordinate-system work in roadmap stage B. Units are copied unchanged, as
  today.
- **Combining environments** is roadmap stage C.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_subset_keeps_sizes_and_faces` (parametrized over the ten types) | `child.bin_sizes == parent.bin_sizes[keep]`; the child's faces equal the parent's restricted to kept pairs; the hexagonal median size stays 21.65, not 5.0 |
| `test_grid_subset_equals_mask_build` (parametrized: 8- and 4-connected) | A rectilinear subset smooths, interpolates and builds kernels identically to `Environment.from_grid_mask` with the same mask, edges and connectivity, as today's `_grid_subset` path does |
| `test_subset_lookup` (parametrized) | With `parent = parent_env.bin_at(p)`, `expected = np.where(parent >= 0, child_of[np.maximum(parent, 0)], -1)` and `child.bin_at(p) == expected` on the parity probes. Points outside the parent and points in dropped bins both return -1 |
| `test_subset_locator_outside_points` | A point outside the parent (`parent.bin_at(p) == -1`) returns -1 from the child. Never the bin that `child_of[-1]` would give: the fixture keeps the parent's last bin so that bug would show |
| `test_translation_preserves_everything` (parametrized over Cartesian types) | Smoothing in all three modes, the gradient operator, `bin_at(p + t) == bin_at(p)`, `to_linear` (tracks) and plot artists shifted by `t` all match the parent; `grid_edges[k] == parent.grid_edges[k] + t_k` |
| `test_derived_plot_field_keeps_renderer` (parametrized: translated, uniformly scaled and positively axis-scaled grids; translated hexagonal and mesh; grid subset) | `plot_field` keeps the parent's artist class (`QuadMesh`, `PatchCollection` or `PolyCollection`) and doesn't fall back to scatter (`PathCollection`) |
| `test_reflection_keeps_bin_identity` (2×2 grid and the 8-connected fixture, `diag(-1, 1)` and `diag(1, -1)`) | `child.bin_at(T(c)) == arange(n_bins)` for the parent's centers `c`. The child has no rectilinear component, and `interpolate(mode="linear")` refuses. Points on the parent's cell boundaries keep their parent owner: `child.bin_at(T(p)) == parent.bin_at(p)` |
| `test_positive_axis_scaling_fast_path_boundaries` | Under `diag(2, 0.5)` plus a translation, points exactly on the mapped edges (`a_k * e + t_k`) locate to the same bins as the parent's points on `e` |
| `test_similarity_scales_smoothing` (Hypothesis: random rotation, scale `s` in [0.25, 4], translation) | For the grid, hexagonal, mesh and track types, to `rtol=1e-10`: `T(env).smooth(f, s*σ, mode=m)` equals `env.smooth(f, σ, mode=m)` for `m` in {`"average"`, `"transition"`}, and equals `env.smooth(f, σ, mode="density") / s**measure_dim` for density, because density divides by bin size. Also `T(env).bin_at(T(p)) == env.bin_at(p)` |
| `test_axis_scaling_grid_equals_scaled_build` | `diag(2, 0.5)` on a grid smooths identically to `from_grid_mask` with scaled edges (`rtol=1e-12`) |
| `test_axis_scaling_hexagonal_refuses_smoothing` | `smooth` raises `NotImplementedError` whose message names axis-aligned scaling and the lost right angle; `bin_at` still works |
| `test_shear_refuses_smoothing` | A sheared grid refuses smoothing with the reason; `bin_sizes == parent.bin_sizes * abs(det A)` |
| `test_track_non_similarity_refused` | `diag(2, 1)` on a track raises `ValueError`, with the fix naming `Environment.from_graph` |
| `test_derived_round_trip` (parametrized: subset, translation, rotation, axis scaling, shear × file, dict, NWB) | A save/load round trip is identical, including nested wrapper locators |
| `test_subset_of_transformed_and_back` | `env.apply_transform(T).subset(bins=m).apply_transform(T.inverse())` gives `bin_at` and smoothing identical to `env.subset(bins=m)` |
| baseline compare (manual) | only the listed differences |
| full suite, doctests, `tests/docs`, `tests/nwb`, API snapshot, ruff, mypy | all pass |

## Fixtures

- Reuse `built_environment` and the parity probes.
- Hypothesis strategies:
  - rotation angles in [0, 2π);
  - scales log-uniform in [0.25, 4];
  - translations in [-1e3, 1e3].

  Seed them with `@settings(derandomize=True)` so CI is reproducible.

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
- The old paths are gone: `_grid_subset`, `_SubsetLayout`, `_graph_subset` and
  `TransformedLayout`. No code compares `_layout_type_tag` to choose a
  derivation path.
- The user-facing documentation listed as tasks is updated: the `subset` and
  `apply_transform` docstrings, `CHANGELOG.md` and `docs/contributing.md`.
