# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Problem

Today a layout engine is two things at once:
- **the recipe:** how bins are built (`build`, `_build_params_used`);
- **the result:** the bins, how points map to bins, and which features exist.

An environment made without a recipe gets a stand-in layout, and the stand-in
loses features. Each defect below was reproduced on 2026-10-13.

| Defect | Cause | Fixed in |
| --- | --- | --- |
| `apply_transform(translate(10, 0))` on a grid can't smooth, take gradients or plot | `TransformedLayout` (`src/neurospatial/ops/transforms.py:1536-1610`) has no faces and no plot | phase 4 |
| After translating by 1000 cm, `grid_edges` still start at −2.47 | `TransformedLayout` copies the parent's grid fields unchanged (`ops/transforms.py:1558-1561`) | phase 4 |
| A hexagonal subset reports a median bin size of 5.0 instead of 21.65, and can't smooth | `_SubsetLayout.bin_sizes` uses mean edge length (`src/neurospatial/environment/transforms.py:533-563`) | phase 4 |
| NWB read-back breaks `boundary_bins`; tracks lose `to_linear` | `_ReconstructedLayout` rebuilds nodes without grid indices and drops the track graph (`src/neurospatial/io/nwb/_environment.py:715-995`) | phase 3 |
| Loading reruns the builder and fails if the centers differ; unregistered layouts can't load | `_from_saved_layout` (`src/neurospatial/environment/factories.py:1461-1555`) | phase 3 |
| A point 470 cm from a 100 cm track maps to bin 49, and `contains` is True | Track lookup is nearest center with no limit (`src/neurospatial/layout/mixins.py:134`) | roadmap stage A |
| `rebin` marks every coarse bin active | It coarsens bins instead of re-binning the space (`environment/transforms.py:28-187`) | roadmap stage D |
| `CompositeEnvironment` repeats the environment API in 1,202 lines | Combining doesn't fit the layout model (`src/neurospatial/composite.py:122`) | roadmap stage C |

## Target design

```text
Environment
├── arena        continuous: coordinate system (units, frame), physical walls,
│                regions, track centerline and width            (roadmap stage A)
├── binning      discrete, built by a layout and then stored as data:
│   ├── bins     SpatialBins — centers, sizes, measure_dim, pairs, distances
│   ├── faces    finite-volume geometry, or the reason it's missing
│   ├── locator  point -> bin
│   └── components  rectilinear grid, track linearization, bin shapes, outline
└── provenance   descriptive record of how the environment was made
```

The design rules, revised after the 2026-10-13 review:

1. **A builder runs once.** After that, every operation reads the stored
   binning. Loading never rebuilds.
2. **A feature exists when its data exists.** Derived environments keep exactly
   the features their data supports. Otherwise they refuse with the reason.
   Wrappers keep point lookup; they can't create faces, shapes or codecs.
3. **Coordinate dimension and measure dimension are separate.** Scaling a track
   by 2 doubles its bin lengths.
4. **Preservation is guaranteed only where it's proven:** translations and
   similarities, plus axis-aligned scaling of axis-aligned grids. Other
   transforms are validated or refused (36.87° counterexample:
   [designs.md](designs.md#transform)).
5. **Selecting whole bins and clipping bins to a shape are different
   operations.** Clipping changes sizes and faces, and arrives with the arena
   (stage A).
6. **Physical walls aren't analysis boundaries.** A subset or an egocentric
   cutoff doesn't create walls (stage A).
7. **Identity semantics, owned read-only arrays, and caches held by the
   environment.** No global caches.

## Current codebase integration points

- `src/neurospatial/environment/core.py:71-85` — `_PublishedGeometry`: gains `binning` (phase 1).
- `src/neurospatial/environment/core.py:921-1041` — `_setup_from_layout`, the only publication point: builds the binning (phase 1).
- `src/neurospatial/environment/queries.py:239`, `cartesian.py:131`, `:177`, `:327-336`, `:409-418`, `:1048`, `fields.py:178`, `:259`, `:790-831`, `trajectory.py:365` — environment-internal readers routed to the binning (phase 1).
- `src/neurospatial/layout/mixins.py:134`, `:272`, `engines/hexagonal.py:296`, `engines/triangular_mesh.py:192`, `engines/graph.py:328-425` — lookup and linearization bodies moved into data objects (phase 2a).
- `src/neurospatial/layout/mixins.py:312`, `engines/shapely_polygon.py:146`, `engines/hexagonal.py:197`, `engines/triangular_mesh.py:276`, `engines/graph.py:178-327`, `environment/visualization.py:483-501`, `:1007-1075` — plotting moved to functions of data (phase 2b).
- `src/neurospatial/io/files.py`, `io/nwb/_environment.py`, `environment/factories.py:1461-1555` — persistence rewritten; the rebuild path is deleted (phase 3).
- `src/neurospatial/environment/transforms.py:350-687`, `ops/transforms.py:1536-1610` — subset and transform rewritten; the stand-in layouts are deleted (phase 4).
- **Untouched in phases 1–4:**
  - the `LayoutEngine` protocol (`layout/base.py:83`);
  - the eight environment mixins and `EnvironmentProtocol`;
  - `CompositeEnvironment`;
  - the polar sibling class;
  - `rebin`;
  - the roughly 80 `env.layout` readers in animation, encoding, metrics and NWB
    writing.

## Scope and dependency policy

### Goals

- Every environment publishes one `Binning`, and environment code reads
  capabilities only from it.
- Every built-in layout provides data-backed lookup, linearization, shapes and
  outlines, each with a single implementation.
- Files and NWB records store the binning, and loading never runs a builder.
- `subset` and `apply_transform` derive a binning under explicit rules, keep
  every feature the data supports, and can be saved.

### Non-Goals

- **The arena and continuous geometry.** That includes changing what "inside"
  means; track lookup stays unlimited until stage A.
- **Region clipping of bins, or coverage modes** (stage A).
- **Immutability of the environment value:** regions, the units setter,
  `with_*` methods (stage B).
- **`combine`** (stage C).
- **Changing `rebin`, border scores or ray casting** (stage D).
- **Exact faces or shapes under general affine maps** (stage E).
- **Removing `LayoutEngine` from `Environment`, the mixins or the host
  protocol, or merging the polar class** (stage F).
- **Any change to public names or signatures.** The API snapshot stays the same
  in every phase.

### Dependency policy

No new runtime dependencies. Hypothesis is already a dev dependency
(`pyproject.toml:58`). `import-linter` arrives as a dev dependency at stage F.

## Metrics

- **Behavior-preserving phases** (1, 2a and 2b): `scripts/environment_baseline.py compare`
  reports zero differences.
- **Phases 3 and 4:** the baseline compare reports only the differences listed
  in the phase, each verified by a test.
- **Lookup parity:** every data-backed locator equals the previous lookup on
  the parity probes.
- **Round trip:** save then load gives `np.array_equal` arrays for every
  environment type, and for every derived kind in phase 4.
- **Physical invariant:** similarity-transformed environments smooth
  identically, after scaling σ (phase 4).
- **Speed:** construction time and `bin_at` on 1,000,000 points stay within 10%
  of the base commit (phases 1 and 2a), or the PR reports the delta.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Building the binning eagerly slows construction of large environments | Measured in phase 1. Faces stay lazy through `FacesSource.build()`. Only arrays the layout already computes are copied. |
| Moved lookup or plot code changes output subtly | The bodies move unchanged, and the layouts delegate. Parity arrays and plot-artist records are committed from the base commit. |
| The facade claims capabilities it doesn't have | The facade defines no capability methods, and environment code reads only the binning (phase 1 contract). |
| Removing v1 reading strands old files | There are no users (confirmed 2026-10-13). The refusal names the fix. |
| Delaunay rebuilt from stored points could triangulate differently | The memo checks rebuilt simplices against the stored ones and raises if they differ. |
| Similarity tolerance misclassifies near-similar transforms | Strict `rtol=1e-12`. A misclassified transform falls into "other", which refuses faces, the safe direction. |
| Per-pair Python loops become hot paths | Derivations are vectorized over the pair arrays. Only the legacy graph renumbering stays a loop, as today. |

## Rollout Strategy

The code is replaced in place, with no deprecation period, because there are no
users. Each phase ships as one PR on its own branch from `main`. Phases 1, 2a
and 2b don't change results. Phases 3 and 4 list their intended changes in
`CHANGELOG.md`.

## Roadmap after phase 4

Each stage needs its own planning pass. Decisions marked **(decide)** must be
made before that stage is planned.

| Stage | Goal | Depends on | Decisions and notes |
| --- | --- | --- | --- |
| A. Arena | Continuous representation: coordinate system, `OpenArena`, `TrackArena`, `MaskArena`, `EgocentricSurround`. `bin_at` becomes "inside the arena, then locate". A tested coverage contract. Separate whole-bin selection from clipping, which updates sizes, faces and components. Physical walls are kept separate from analysis boundaries. | phases 1–4 | **Complete on `feature/environment-arena`.** Track width is required when asserting a physical footprint; abstract graphs and old data report an unknown footprint. Center-based selection, connectivity-blocking walls, general polygon clipping and linear spatial encoding are implemented. See the [Stage A design and validation](../environment-arena/PLAN.md). |
| B. Immutable value | Units and frame set at construction. Immutable regions with `with_regions`. Owned input buffers. `to_networkx()`. `from_bins` that requires explicit geometry for diffusion and clipping. Third-party codecs through entry points. Public contract test kits. | A | [Stage B](../environment-immutable/PLAN.md) is complete and independently reviewed on `feature/environment-immutable`, stacked on the arena branch. See its [validation record](../environment-immutable/validation.md). Custom providers normalize to owned built-in snapshots; original custom records require installed adapters, while canonical resaves use built-in codecs. |
| C. Combine | `Environment.combine` replaces `CompositeEnvironment`. | B | [Stage C](../environment-combine/PLAN.md) is complete and independently reviewed on local `feature/environment-combine`, stacked on Stage B. The ordinary environment API replaces the legacy class. Certified support prevents overlapping/ambiguous components; automatic MNN bridges remain path-only unless geometric conductance is explicit. See its validation record for geometry and test limits. |
| D. Re-binning and walls | `rebin(bin_size=...)` re-bins the arena, using a stored binning spec kept separate from provenance. Border scores and ray casting use physical walls. | A | One PR per consumer. |
| E. General affine | Face normals in the binning. Nanson-formula updates, with an explicit non-orthogonality check. Exact shapes. Rectilinear grids kept in their own frame, so rotations and reflections keep exact lookup, linear interpolation and `QuadMesh` rendering without renumbering bins. | A | Starts from the guaranteed set in [shared-contracts.md#derivation-rules](shared-contracts.md#derivation-rules). |
| F. Remove the bridge | Visualization, animation, encoding, metrics and NWB writing read the binning. Remove `LayoutEngine` from `Environment`, the mixins, `EnvironmentProtocol` and the facade. Put the public convenience API in a layer above the model, numerical and adapter layers, enforced with `import-linter`. One internal representation for polar, with typed Cartesian and polar facades kept at first. | A–C | **(decide)** whether to merge the public polar class, separately from this stage. |

## Open Questions

1. **Read-only `bin_sizes`** (phase 1): accepted as an intended change, matching
   `bin_centers`.
2. **v1 files and 1.0 NWB records** (phase 3): refused with a fix message. There
   are no users, so there's no converter.
3. **Names:** `Arena`, `SpatialBins` and `Binning` are kept, per the review. The
   component field is `rectilinear`, not `grid`, which would collide with
   grid-cell analysis in `encoding/grid.py`.
4. **Track transforms outside the similarity class** (phase 4): refused.
   Revisit at stage E if a need appears.

## Estimated Effort

Net changes, excluding tests:

| Phase | Effort |
| --- | --- |
| 1 | +350 / −60 |
| 2a | +450 / −250, mostly moves |
| 2b | +350 / −300, mostly moves |
| 3 | +650 / −450; deletes the rebuild path and `_ReconstructedLayout` |
| 4 | +550 / −550; deletes `_SubsetLayout`, `_graph_subset`, `_grid_subset` and `TransformedLayout` |

Tests add about 1.5× the source lines in each phase.
