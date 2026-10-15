# Phase 2a — Data-backed locators and track linearization

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#data-backed-locators)

Point lookup and track linearization become small frozen objects made of arrays.
Each one is the single implementation: the layout's own methods call it. After
this phase a binning no longer needs its layout to find bins or linearize. Phase
3 saves these objects and phase 4 derives new ones. Outputs do not change except that hexagonal lookup now rejects incorrect
coordinate widths, as required by the locator contract (approved by the user).

**Inputs to read first:**

- [src/neurospatial/layout/mixins.py:21-228](../../../src/neurospatial/layout/mixins.py#L21-L228) — `_KDTreeMixin`, including `point_to_bin_index` at line 134, with its non-finite-row and dimension-mismatch handling.
- [src/neurospatial/layout/mixins.py:229-311](../../../src/neurospatial/layout/mixins.py#L229-L311) — `_GridMixin`: `active_mask_inverse_map` (line 251) and `point_to_bin_index` (line 272).
- `src/neurospatial/layout/helpers/regular_grid.py` — `_points_to_regular_grid_bin_ind` (used, not moved).
- [src/neurospatial/layout/engines/hexagonal.py:296-346](../../../src/neurospatial/layout/engines/hexagonal.py#L296-L346) — hexagonal lookup and its warning branch.
- `src/neurospatial/layout/helpers/hexagonal.py` — `_points_to_hex_bin_ind` (used, not moved).
- [src/neurospatial/layout/engines/triangular_mesh.py:192-263](../../../src/neurospatial/layout/engines/triangular_mesh.py#L192-L263) — mesh lookup through `Delaunay.find_simplex`, followed by the boundary-polygon containment rule at lines 249-260. `_full_delaunay_tri` is set at line 145.
- [src/neurospatial/layout/engines/graph.py:328-425](../../../src/neurospatial/layout/engines/graph.py#L328-L425) — `to_linear` and `linear_to_nd`.
- [src/neurospatial/layout/engines/_polar.py](../../../src/neurospatial/layout/engines/_polar.py) — the polar grid is a `MaskedGridLayout`, so it gets `RectilinearLocator` through the mixin.
- `src/neurospatial/layout/_binning.py` — created in phase 1: `binning_from_layout` and its `_locator_of` and `_linearization_of` helpers.

**Contracts referenced:**

- [Locator](shared-contracts.md#locator) — implement `RectilinearLocator`, `HexLocator`, `MeshLocator` and `NearestBinLocator`. Do not weaken the non-finite-row and wrong-width rules.
- [Components](shared-contracts.md#components) — the `TrackLinearization` data class.
- [Identity and ownership rules](shared-contracts.md#identity-and-ownership-rules) — memos (KD-tree, Delaunay) are rebuildable and never stored as data.

**Designs referenced:** [designs.md#data-backed-locators](designs.md#data-backed-locators).

## Tasks

- **Record the baseline** on the base commit, as in phase 1.
- **Create `src/neurospatial/layout/_locators.py`** with the four locator
  classes, as frozen `eq=False` dataclasses.
  - Move the lookup bodies listed in
    [designs.md](designs.md#data-backed-locators) into their `locate` methods
    unchanged.
  - The KD-tree and Delaunay memos are private fields created lazily with
    `object.__setattr__`, and excluded from `repr`.
  - `MeshLocator` checks the rebuilt simplices against the stored ones when its
    memo is built. It also stores the boundary polygon
    (`_boundary_polygon_stored`) and applies the containment rule at
    `triangular_mesh.py:249-260` unchanged, so points in holes or concave
    notches stay at -1.
- **Make each layout call its locator:**
  - `_GridMixin`, `HexagonalLayout` and `TriangularMeshLayout` build their
    locator at the end of `build()` and store it as `self._locator`.
    `GraphLayout` gets `NearestBinLocator` through `_KDTreeMixin`.
  - Add `locator(self) -> Locator` to each, returning `self._locator`.
  - Replace each `point_to_bin_index` body with
    `return self.locator().locate(points)`. Keep the guard that raises
    `LayoutNotBuiltError` before build, and the hexagonal warning branch for
    layouts with no active bins.
  - `_KDTreeMixin._build_kdtree` keeps its signature. It now sets
    `self._locator = NearestBinLocator(points, ...)`, and the
    `_kdtree_nodes_to_bin_indices_map` mapping moves into the locator.
- **Add the data `TrackLinearization`** to `layout/_binning.py`:
  - Move the bodies of `to_linear` (`graph.py:328-352`) and `linear_to_nd`
    (`graph.py:353-425`) into its methods.
  - `GraphLayout.build()` stores `self._linearization`.
  - `GraphLayout.to_linear` and `linear_to_nd` delegate to it.
  - Add `linearization(self) -> TrackLinearization` to `GraphLayout`.
- **Switch the binning to data.** In `binning_from_layout`:
  - `_locator_of(layout)` returns `layout.locator()` when the layout defines
    it, and `LayoutLocator(layout)` otherwise. The fallback covers
    `_SubsetLayout`, `TransformedLayout`, `_ReconstructedLayout` and
    third-party layouts.
  - `_linearization_of(layout)` returns `layout.linearization()` when the
    layout defines it. Otherwise it returns the layout itself if it's a
    `TrackLinearizationProvider`, and `None` if not.
- **Tests:**
  - New `tests/layout/test_locators.py`: parity, invariance and property tests.
  - New `tests/layout/test_track_linearization.py`.
- **Compare the baseline**: zero differences except the approved hexagonal
  wrong-width validation. Measure the median time of
  `env.bin_at` on 1,000,000 seeded points, for the 90,000-bin grid and for the
  track. Each must be within 10% of the base commit, or the PR reports the
  delta.

## Deliberately not in this phase

- **Saving locators** is phase 3. This phase adds no codecs.
- **`SubsetLocator` and `TransformedLocator`** are phase 4.
- **A distance limit for track lookup** is roadmap stage A, the arena. Here
  `NearestBinLocator(max_distance=None)` keeps today's unlimited
  nearest-center behavior, even though it's known to be wrong for tracks.
- **Shapes and plotting** are phase 2b.
- **The stand-in layouts' own `point_to_bin_index` code**
  (`environment/transforms.py:493-531`, `ops/transforms.py:1566-1574`,
  `io/nwb/_environment.py:932`) stays until the phases that delete those
  layouts.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_locator_matches_previous_lookup` (parametrized over the ten environment types) | On the [parity probes](designs.md#data-backed-locators), `env.layout.locator().locate(p)` equals the output recorded from the base commit's `point_to_bin_index`, saved as a fixture array; wrong-width input raises the same `ValueError` text |
| `test_locator_non_finite_rows` (parametrized) | Rows with NaN or inf return -1 and finite rows in the same call are still located |
| `test_locator_bin_centers_map_to_themselves` (parametrized) | `locate(env.bin_centers) == arange(n_bins)` for grid, hexagonal, mesh and polar environments |
| `test_rectilinear_locator_matches_brute_force` (Hypothesis, `@pytest.mark.hypothesis`) | For random sorted edges in 1–3D, random masks and random points, the result equals a brute-force "which cell's half-open box contains the point" search |
| `test_mesh_locator_respects_boundary` | On a mesh built from a polygon with a hole and a concave notch, points inside the hole and inside the notch return -1, matching the recorded base-commit output. Points just inside the boundary return their triangle |
| `test_mesh_locator_rejects_changed_triangulation` | Building the memo from points whose Delaunay simplices differ from the stored ones raises `ValueError` |
| `test_locators_are_identity_hashed_and_read_only` | Locators can be dict keys, and their arrays reject writes |
| `test_track_linearization_matches_previous` | `to_linear` and `linear_to_nd` equal the base commit's outputs on 2,000 seeded points along and near a branching track |
| baseline compare (manual) | zero differences |
| full suite, doctests, `tests/docs`, API snapshot, ruff, mypy | all pass |

## Fixtures

- Reuse the `built_environment` fixture from phase 1.
- Commit the base commit's lookup outputs for the parity probes to
  `tests/layout/data/locator_parity.npz`, generated by a small script run once
  on the base commit. The script is not committed: describe the command in the
  PR.
- The track uses the branching track from the fixture.
- Add a mesh case to `built_environment`: `from_layout("TriangularMesh", ...)` on a polygon with one interior hole and one
  concave notch. Record its parity outputs with the others.

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
- There's exactly one lookup implementation per layout kind: the layouts'
  `point_to_bin_index` methods only delegate.
- No user-facing documentation changes are needed. Confirm that no public
  docstring describes `point_to_bin_index` internals that moved.
