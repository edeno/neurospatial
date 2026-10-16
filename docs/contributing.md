# Contributing

Thank you for your interest in contributing to neurospatial! This guide will help you get started.

## Design Principles

The scientist's experience is neurospatial's highest design priority. Scientific
correctness is part of usability: a convenient call that silently changes the
meaning of an analysis is a usability failure. Scientists should be able to
discover an operation, understand its assumptions, and interpret its result
without reading implementation code.

The long-term scope is the operations of spatial neuroscience: loading and
validating recordings, representing environments, analyzing behavior and neural
activity, evaluating statistical evidence, simulating experiments, visualizing
results, and exporting them. Grow this coverage through workflows that work
together and preserve scientific meaning across steps.

Use these principles when designing and reviewing changes:

- **Start with the scientific task.** Describe what the scientist has and wants
  to learn. Use familiar experimental vocabulary and provide a clear recommended
  path for common tasks, with advanced controls available when needed.
- **Make expected behavior scientifically sound.** Choose defensible defaults
  and document their assumptions. Require an explicit choice when alternatives
  change the scientific question and there is no defensible default. Preserve
  the distinction between missing observations and measured zero activity;
  expose how recording gaps, analysis windows, and insufficient coverage affect
  the result.
- **Make the interface learnable across domains.** Keep names, argument order,
  units, shapes, and result methods consistent. Moving from one neuron to a
  population or from encoding to decoding should reuse what the scientist has
  already learned. Follow the existing public API contracts when extending them.
- **Preserve meaning between operations.** Keep timestamps, coordinate frames,
  spatial support, and unit identities aligned through filtering, analysis,
  plotting, and export. Validate incompatible inputs and explain how to fix
  them instead of silently guessing an alignment or conversion.
- **Make results and recovery understandable.** Label plots and tables with
  quantities and units, avoid relying on color alone, and expose the assumptions
  needed to interpret a result. Distinguish descriptive metrics and candidate
  cell labels from statistical evidence. Errors should identify the problem
  and give a concrete next step.
- **Evaluate complete workflows.** For analysis changes, verify the path from
  realistic inputs to an interpretable result using public APIs and runnable
  examples. Test scientific invariants and relevant edge cases, including gaps,
  empty data, and mismatched identities, rather than checking only that a call
  completes. Judge added functionality by whether scientists can use it correctly
  alongside existing operations.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone **your fork** locally (replace `YOUR_USERNAME`, or use
   `gh repo fork edeno/neurospatial --clone` to fork and clone in one step):

```bash
git clone https://github.com/YOUR_USERNAME/neurospatial.git
cd neurospatial
```

3. Install development dependencies:

```bash
uv sync --extra dev
```

4. Set up pre-commit hooks:

```bash
uv run pre-commit install
```

### Development Workflow

This project uses `uv` for package management. Always prefix commands with `uv run`:

```bash
# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_name -v

# Check code quality
uv run ruff check .

# Format code
uv run ruff format .

# Type check (optional)
uv run mypy src/neurospatial
```

### Environment architecture

`Environment` is a facade over owned data. It holds published geometry,
labels, named regions, a build record and one analysis cache, and its public
methods hand that data to private module functions. This section says where
new code goes and which rules it must keep.

#### Layers

`.importlinter` places every top-level package and module in a layer, and a
package imports only from the layers below it. From highest to lowest:

| Layer | Packages and modules |
| --- | --- |
| Storage, annotation, contract-check kit | `io`, `annotation`, `testing` |
| Rendering, events, simulation | `animation`, `events`, `simulation` |
| Decoding | `decoding` |
| Encoding and behavior | `encoding`, `behavior` |
| Statistics | `stats` |
| Result base classes | `_results` |
| Environment facade | `environment` |
| Builders and numerical operators | `layout`, `ops` |
| Public geometry paths and regions | `regions`, `arena`, `binning`, `extensions` |
| Owned geometry | `geometry` |
| Shared utilities | `_constants`, `_exceptions`, `_intervals`, `_logging`, `_lru`, `_stacklevel`, `_typing`, `_validation` |

Packages in the same row do not import each other, except the shared
utilities, which may. The only upward imports are function-local imports in
the `Environment` methods that hand off to storage or rendering (`to_file`,
`from_file`, `to_dict`, `from_dict`, `to_nwb`, `from_nwb`, `animate_fields`);
each such edge is listed in `ignore_imports`. When a new import breaks the
layers, move the imported code down or the importer up instead of adding an
ignore, and add every new top-level module to a layer.

Two more contracts apply. No module imports from the lazy package root
`neurospatial`, because that hides the real dependency. `geometry` imports no
builder, environment, analysis, storage or rendering package, nor Matplotlib,
pynwb, napari, xarray or pynapple. Type-only imports are excluded from all
contracts. Run `uv run --extra dev lint-imports --no-cache`; CI runs it on
Python 3.13.

#### Where code lives

- `geometry/`: owned records (`Binning`, locators, arenas, coordinate
  descriptors), validation, codecs and pure coordinate math. This includes
  deriving binnings for subsets and affine maps (`_derive.py`), arena
  inference and wall checks (`_arena_binning.py`), point lookup gated by the
  arena (`_lookup.py:locate`), boundary bins for every layout
  (`_boundary_bins.py`), wall-frame evidence (`_wall_frame.py`) and snapshot
  adapters for external providers (`_adapters.py`). Each arena type bounds
  its own domain (`Arena.domain_bounds`).
- `layout/`: layout engines. Each builds its bins and graph and assembles its
  own `Binning`.
- `ops/`: numerical algorithms on explicit geometry and arrays, such as
  linear time allocation on rectangular grids (`_linear_allocation.py`),
  exact allocation on planar polygon cells (`_polygon_allocation.py`), wall
  evidence and wall-hit bin ownership (`_boundaries.py`), diffusion,
  calculus, distances and transforms.
- `environment/`: the `Environment` class in `core.py` and the private
  adapters it calls. Modules that hold only implementation have `_`-prefixed
  names. A public type returned by a public method is importable from a
  public path, such as `neurospatial.environment.BinSequenceWithRuns`.

`neurospatial.binning` and `neurospatial.arena` are the public import paths
for geometry records. Importing `neurospatial.binning` does not load layout
engines, `environment.core` or Matplotlib. There are no compatibility alias
modules.

#### Publication

`environment/_publication.py:publish` is the only code that creates an
`Environment`; `Environment()` itself always raises E1006. `publish` owns and
validates the `Binning` and graph, checks the arena against the bins, removes
the faces that physical walls block and freezes the graph. It then stores one
frozen `_PublishedGeometry` (binning, centers, graph, ranges, grid context,
arena, clip mapping and coordinate descriptor), the name, units and frame, a
snapshot of the regions, a `_Provenance` record and an empty `AnalysisCache`.

Every factory, derivation and reader calls `publish` exactly once per
environment it returns, through one of three entry points:

- `publish_layout(cls, layout, ...)` for a built layout engine: the grid,
  polygon, mask, track and sample factories, `from_layout` and the polar
  factory.
- `publish_derived(parent, derivation, ...)` for `subset`, `clip`, `rebin`,
  `apply_transform` and `combine`.
- `publish(...)` itself for `from_bins` and for the file, dict and NWB
  readers, which decode the stored binning and graph without replaying any
  build parameters.

A published environment never changes. Geometry attributes are read-only
views: arrays are read-only headers over owned buffers, and `env.connectivity`
refuses topology and attribute edits. Graph attribute values are frozen once
(lists become tuples, dicts read-only mappings, sets frozensets, arrays
read-only) and are read without copying; `env.to_networkx()` and
`env.connectivity.copy()` return editable graphs. Equality and hashing are by
identity, and `copy.copy` and `copy.deepcopy` return the same object.
`with_metadata` and `with_regions` return a new object over the same
published geometry and cache.

`_Provenance` records the layout type, its category and the scalar build
choices (`build_choices`): numbers, strings, booleans and flat numeric
sequences such as a per-axis `bin_size`. Input data such as positions, masks,
polygons and graphs is never recorded; each built-in engine names its data
arguments in a `_data_parameters` class attribute. Provenance only describes
how the bins were built. Behavior comes from the owned binning, graph and
arena, so do not branch on `layout_type`, `_layout_type_tag` or build
parameters.

Pickling keeps the published state and drops the cache, which unpickles
empty. It is for passing an environment between processes of the same
version. Storage is `to_file`/`from_file`, `to_dict`/`from_dict` and NWB.

#### Layout engines own their binning

Every layout engine, built-in or custom, implements `binning()`. It returns a
`Binning` assembled from the engine's own state: bins in graph node order,
neighbor pairs in graph edge order with each edge's `"distance"`, point
lookup, native faces computed on first use, and the optional grid, track,
grid-specification, shape and outline components. Built-in engines assemble
it with `layout/_assembly.py:assemble_binning`, so they share one rule for
pair order, bin measures and face directions.
`layout/_construction.py:built_layout_data` takes that record with the
engine's graph, ranges and provenance. It never inspects engine classes or
reads build parameters for geometry. An object without `binning()` needs an
installed snapshot adapter.

A custom engine needs no registration: build it and pass it to
`Environment.from_layout(engine)`. A new built-in engine also needs:

- a `LayoutType` member and a `_LAYOUT_MAP` entry in `layout/factories.py`,
  so `create_layout` can build it by name;
- an export from `layout/__init__.py`;
- a `_data_parameters` class attribute naming its data arguments;
- an entry in `MINIMAL_BUILD_PARAMS` in `tests/layout/test_layout_engine.py`.
  The protocol test skips engines missing from that dict instead of failing.

Polar environments use `layout/engines/_polar.py:_PolarGridLayout`, a masked
grid over (distance, angle) whose cell and face measures are scaled by
radius. It builds the physical polar edge distances and the circular seam
edges, records `circular_angle`, and is built only by
`Environment.from_polar_egocentric`, so it has no `_LAYOUT_MAP` entry.

#### Methods pass explicit components

A public method is a thin entry point. It unpacks owned data and calls one
private module function with explicit components: the published `Binning`,
the graph, arrays, the arena, the coordinate descriptor and, for a function
that caches, the `AnalysisCache`. The function takes the environment only
when it needs the whole facade, and it never calls back into the
environment. Put it in a `_`-prefixed module (for example
`environment/_queries.py`), keep it out of `__all__`, and give it a one-line
NumPy summary plus a `Parameters` section when an argument's meaning is not
clear from its name and type. Keep the method's name, signature, decorators
and docstring when you move an implementation, and fix bugs you find in a
separate change. Warnings use `stacklevel=_external_stacklevel()` so they
point at the scientist's line; errors use `_format_error` to say what failed,
why and how to fix it.

A package that consumes an environment declares a small `Protocol` of the
members it reads, as `ops/_protocols.py` does, so Cartesian and polar
environments satisfy it without casts. `ops` never imports `Environment`,
even for type checking. `tests/typing/ops_consumers.py` and
`tests/typing/finite_volume_consumers.py` check both coordinate variants of
`Environment` against these protocols. `environment/` itself has no consumer
protocols and no mixins.

#### Analysis cache

Every memoized analysis lives in the environment's one `AnalysisCache`
(`environment/_cache.py`): the KD-tree, face and diffusion geometry,
eigenbases, smoothing kernels, gradient and divergence operators, boundary
bins and the bin and edge attribute tables. Name a new cached value with a
module-level `CacheKey` (or a bounded `RecentKey` family) and read it with
`cache.get(key, build)`. Do not use `cached_property` for analysis values.

Each key has its own re-entrant lock, so concurrent first requests compute a
value once. A value that grows in place, such as an eigenbasis, is read and
grown inside `with cache.exclusive(key):`. `with_metadata` and `with_regions`
share their parent's cache because the geometry is identical; derivations,
readers and unpickled environments start with an empty one. `clear_cache()`
takes no arguments and empties the whole cache.

#### Cartesian-only members

There is one runtime `Environment` class. Factories give it a static
coordinate type, `Environment[CartesianSystem]` or
`Environment[EgocentricPolarSystem]`. To make a member Cartesian-only,
annotate `self` as `Environment[CartesianSystem]` for type checkers and
decorate the member with `@cartesian_only` from `environment/_cartesian.py`,
or `@cartesian_only(grid=True)` when it needs a Cartesian grid. For a
property, put the decorator below `@property`.

The decorator is the only record of the capability:

- it refuses polar environments on bound and unbound calls, such as
  `Environment.bin_at(polar_env, points)`;
- `CARTESIAN_OPERATIONS` and `CARTESIAN_GRID_OPERATIONS` in
  `environment/core.py` are derived from it;
- `Environment.__getattribute__` refuses the decorated members on polar
  instances, so `hasattr` is False, and `__dir__` leaves them out.

A method that refuses only one mode on polar bins calls
`require_cartesian(coordinates, "...")` for that mode.
`test_every_public_member_is_classified_for_polar` requires every other
public member to have a stated polar meaning. The attribute hook runs only at
runtime, so negative static contracts in `tests/typing` check that type
checkers still reject Cartesian-only members on polar environments. Shared
code asks the coordinate descriptor about lengths, axes and circularity, or
reads the public `coordinate_kind`; it does not branch on classes or layout
labels.

#### Derivations

`subset`, `clip`, `rebin`, `apply_transform` and `combine` compute a
`_Derivation` record (binning, graph, ranges, provenance, and any replacement
regions or arena) in a private module (`_subset.py`, `_clipping.py`,
`_rebin.py`, `_affine.py`, `_combine.py`), using `geometry/_derive.py` for
the geometry. The method passes the record to `publish_derived`, which keeps
the parent's class, name, units, frame, regions and arena unless the record
or the caller replaces them:

- `subset`, `clip` and `rebin` keep every label and accept `name=`.
- `apply_transform` keeps the name and units, maps the regions and arena with
  the bins, and sets `frame` only from the caller's `frame=` (otherwise
  `None`).
- `combine` prefixes each child's regions with its key and has an empty name
  unless `name=` is given.

A derivation never changes units, never rebuilds through a factory and starts
with a fresh analysis cache. Fields and fitted models do not transfer.

#### Factories

Factories are classmethods on `Environment`; their implementations live in
`environment/_factories.py` and, for polar bins, `environment/_polar.py`.
Data arguments are positional and every option is keyword-only after `*`,
with an explicit signature and no `**kwargs` that change meaning. Cartesian
factories accept `name`, `units`, `frame` and `regions`, and `arena` where an
arena is meaningful. An option that does not apply to the chosen layout or
mode raises an actionable `ValueError` instead of being ignored.
`tests/environment/test_factory_arguments.py` checks these rules; add a new
factory to its `_DATA_ARGUMENTS`.

#### Checking a refactor

`scripts/environment_baseline.py` records every probe's output, refusal,
warning, signature and docstring on ten environment types, each also read
back from a dict, a file and NWB. Record on the base commit before you
change code, record again after, and compare:

```bash
uv run python scripts/environment_baseline.py record base.npz  # on the base commit
uv run python scripts/environment_baseline.py record new.npz   # on your change
uv run python scripts/environment_baseline.py compare base.npz new.npz
```

Outputs must match exactly, or the pull request lists and explains every
intended difference. If you remove an API that the script probes, update the
script in the same change. Also run the tests, doctests, executable
documentation, the public API snapshot, Ruff, mypy on the source and on
`tests/typing`, and `lint-imports`.

### Diffusion geometry providers

Smoothing, diffusion, smoothness bases, gradients and divergence use the owned
face source in `Binning.faces`:

- Native construction captures the data needed to compute faces lazily;
  explicit and restored geometry use validated face records. Gradient and
  divergence index edge fields by `env.edge_pairs`, so publication rejects
  faces whose order, distances or orientation disagree with the bins. Each
  undirected edge defines the scalar, symmetric two-point conductance
  `A / distance` used by the current numerical scheme.
- `FaceGeometry.from_graph` validates those faces into read-only per-edge arrays
  in connectivity edge order. `DiffusionGeometry.from_faces` assembles the
  conductances `W = A / distance`; components come from positive conductances,
  so zero-area (corner-touching) edges and isolated bins form separate
  components.

A layout engine's `binning()` records the inputs of its native face formula;
the faces are computed on first use. Explicit and restored binnings carry
validated `KnownFaces`. Geometry without supported faces carries a
`MissingFaces` reason, which saved binnings keep. Numerical capability comes
only from this faces source, never from a layout name.

`Environment._face_geometry` reads `env._binning.faces` through
`face_geometry(cache, binning)` in `environment/_fields.py`, which caches the
validated faces; the diffusion geometry (`_diffusion_geometry`) and the
gradient and divergence operators derive from it. Builders that need faces
call `require_face_geometry(cache, binning, operation)`, which names the
analysis when the layout has no cell faces. `ops/_types.py` describes the
diffusion and MRF eigenbasis cache dictionaries with `TypedDict` and imports
no scientific libraries at runtime. The dense-operation warning threshold is
`neurospatial._constants.LARGE_KERNEL_THRESHOLD`, shared by diffusion and
Gaussian smoothing.

`Binning.face_normals` optionally supplies an owned `FaceNormals` record.
Cell faces are interfaces between bins; they do not assert physical arena walls. Its
`values` array has shape `(n_edges, n_dims)` in `env.edge_pairs` order: each
positive-area face has a unit normal pointing from the first bin toward the
second; zero-area navigation edges have zero rows. These directions come from
stored cell geometry, not from arbitrary center links. Subsets and physical
walls select the same normal rows as their edges; clipping recomputes directions
only for certified faces. Affine transforms map directions by the inverse
transpose without changing edge orientation. `FaceGeometry` retains its four
fields and tuple behavior.

Normals alone do not enable diffusion. Existing approximate mesh faces keep
their current behavior; the normal need not align with a centroid link.
`MissingFaces` retains no areas, so zero rows there may conservatively suppress
contacts as navigation-only edges. Loading can verify nonzero directions against
available geometry but cannot recover discarded area/activity information.
Numerical operations must continue to honor the missing-face reason. New affine
numerical support requires stored normal authority, source orthogonality and
geometric flux distances, and transformed orthogonality residuals at most 1e-12.
Area vectors transform by `abs(det(L)) * inverse(L).T`; log/scaled arithmetic
protects representability. Native approximate meshes retain their existing
similarity behavior; a normal alone does not certify a new deformation.

`Binning.grid_frame` optionally stores an owned `FramedGrid`: current source
`RectilinearGrid` axes/mask plus their forward world map. Grid consumers read
it through `require_grid_context` and map coordinates explicitly. Keep
`rectilinear` and the environment's world-grid attributes (`grid_edges`,
`grid_shape`, `active_mask`) truthful. Framed interpolation/allocation share
bounded source-plane recovery capped at a quarter of the smallest cell width.
QuadMesh consumers transform corners and reshape values with the source mask.
Partial clipping drops the tensor-grid component. `GridBinningSpec` separately
describes how `rebin(bin_size=...)` builds a new grid and must not replace
current-cell geometry.

Binning codec version 3 stores frames and normals; readers accept version 2
(normals only) and version 1 (neither component). New stored geometry components
require a codec version increment so older readers cannot silently ignore a
capability they do not understand. Environment-v3/NWB 3.0 envelopes are unchanged.

Combined environments retain child binnings instead of inventing physical faces
for navigation bridges. Scalar diffusion assembles the certified child blocks
and adds only explicitly supplied symmetric geometric bridge conductances.
Path-only bridges do not transport the field. A retained child with missing
numerical geometry still prevents whole-environment diffusion; removing it with
a subset allows the certified remainder. Gradient/divergence keep the existing
physical-face contract and refuse combinations with nonphysical bridges.

Publication validates geometry with `geometry/_validation.py`, which checks
positions, ordered faces and distances, measures, locator/component
agreement, frames and immediate clip mappings. Readers validate the storage
format; publication runs the shared geometry checks once. Center lookup is
checked on a detached locator so publication does not warm the saved value's
lookup cache. Validated embedded tracks may revisit identical spatial
centers; center lookup must still choose a bin at that same coordinate. Other
bins retain strict center identity checks. Native mesh and hex logical
indexing is distinct from a framed raster's source coordinates.

Stored node-link graphs use an optional `_neurospatial_reserved_attributes`
sidecar to preserve user attributes named `id`, `source`, `target`, or multigraph
`key`, which NetworkX otherwise overwrites with graph identities. Its rows must
match the graph's identities and contain only these reserved attributes. Records
without the sidecar retain their existing format. Older readers can still read
the topology of new records, but retain their previous limitation of dropping
these reserved user attributes; use a current reader to recover them.

`subset` and `apply_transform` derive a binning through `geometry/_derive.py`.
Subsets select whole bins and restrict ordered faces, locators and components;
`geometry/_derive.py:select_bins` selects bins with their graph and ranges,
and `from_layout(arena=...)` uses it as well. Subsets keep the parent's graph
metadata and edge order rather than rebuilding them from geometry, while
`geometry/_graph.py` projects newly built binnings (explicit bins and
clipping) into standard navigation graph attributes, preserving bin IDs and
pair row IDs. Translations and similarities preserve finite-volume geometry;
diagonal grid scaling retains axis-aligned faces. Other affine maps retain
inverse lookup and cell measures, with a `MissingFaces` reason where right
angles are lost; tracks refuse non-similarities. Triangle drawing vertices
are retained under all invertible affine maps, independently of whether
finite-volume faces remain valid for smoothing. Custom providers must use
registered adapters to supported built-in owned snapshots. Live delegated
lookup and track conversion callbacks are refused.

Both rebin modes live in `environment/_rebin.py` and keep distinct
construction rules. Integer-factor `rebin` requires uniform rectangular cells
and native grid navigation, allowing physical walls to remove connections. It
refuses custom path lengths or topology that cell geometry cannot
reconstruct, keeps its historical grid partitioning, retains native lazy
faces and its recorded grid parameters, and keeps truncation warnings and
complete-face wall constraints. `rebin(bin_size=...)` uses the owned
`GridBinningSpec`, independently of provenance: it re-bins the retained known
arena, preserving the grid anchor and affine frame, and prepares faces for
that frame. Requested sizes are physical lengths along those axes. Subset and
clipping preserve the specification; combining does not invent a shared one.
Unsupported families and records without a specification refuse new-size
re-binning.

`env.arena` describes continuous support independently of the active analysis
bins. Samples, grid masks and image masks supply computational domains without
asserting measured physical walls. Explicit polygons and measured track widths
record known footprints; units, frame and regions remain environment-owned.
Subsets preserve the parent arena, while affine transforms map its geometry
with the bins. Exact lookup applies both the arena gate and locator ownership.

Border and visibility consumers use shared planar wall evidence in
`ops/_boundaries.py`. Physical mode reads asserted footprints and internal walls;
analysis mode is explicit. Analysis cuts never create physical walls. Boundary
contacts and wall-hit bin ownership use retained source frames with locally
bounded inverse rounding. Viewshed bin coverage remains sampled, while first
contacts and cue occlusion use geometric intersections. Unsupported geometry
must refuse instead of silently switching boundary sources.

`clip` intersects planar cells with the crop and known domain. Disconnected
pieces become separate bins with an owned immediate-parent `ClipMapping`.
Current polygon cells and each historical locator's cells/mapping retain their
own immutable snapshots and coordinate frames through subset and transform.
No fields or fitted models transfer automatically. Navigation retains allowed
parent connections; explicit complete-face walls remove both navigation edges
and finite-volume faces. Interior-cut or partially covered walls refuse.
Numerical geometry is published only after convexity, interface and
orthogonality checks; containment and plotting do not depend on certification.

Files and dictionaries use `Environment-v3`; NWB environment records use `3.0`.
They include a required Arena record. Version 2/`2.0` remains readable with an
explicit `UnknownArena`, without inferring physical geometry from parameters,
outlines or masks; a polar environment in such a record is refused because
it has no stored `EgocentricSurround`. Version 1/`1.x` records are refused
with recreation guidance. Pure codecs in `geometry/_codecs.py` and
`geometry/_arena.py` store typed arrays and JSON metadata; polygon WKB bytes
and offsets preserve ordered cells and holes. Shared snapshot references
retain locator ownership without duplicating each record. Bounded KD-tree and
mesh triangle-lookup memos are rebuilt from saved data. Standard NWB
containers reject slash-containing names, so a one-row ragged DynamicTable
stores arrays under safe column names. Its description holds the JSON
metadata and canonical-key/shape/dtype mapping, without padding or a custom
extension.

## Code Standards

### Python Style

- Follow PEP 8 (enforced by ruff)
- Use type hints where practical
- Write descriptive variable names
- Keep functions focused and small

### Docstring Format

**CRITICAL**: All docstrings MUST follow NumPy docstring format:

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        When this error occurs.

    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
```

See [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html) for details.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(environment): add .info() method
fix(regions): correct area calculation for polygons
docs(quickstart): update installation instructions
test(layout): add tests for hexagonal layout
chore(deps): update numpy to 2.3.4
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure: `src/neurospatial/foo.py` → `tests/test_foo.py`
- Use descriptive test names: `test_environment_from_samples_creates_bins()`
- Use pytest fixtures from `tests/conftest.py`

Example test:

```python
def test_environment_bin_at_maps_points_correctly():
    """Test that bin_at correctly maps points to bin indices."""
    env = Environment.from_samples(sample_data, bin_size=2.0)
    points = np.array([[10.0, 10.0]])
    bin_idx = env.bin_at(points)
    assert bin_idx.shape == (1,)
    assert 0 <= bin_idx[0] < env.n_bins
```

### Running Tests

```bash
# All tests
uv run pytest

# Napari GUI tests (requires napari/Qt display support)
uv run pytest -n 0 -m "napari and not slow"

# Specific test file
uv run pytest tests/test_environment.py

# With coverage
uv run pytest --cov=src/neurospatial

# Verbose output
uv run pytest -v

# Stop at first failure
uv run pytest -x
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `uv run pytest`
2. **Check code quality**: `uv run ruff check . && uv run ruff format .`
3. **Update documentation**: Add/update docstrings and docs pages
4. **Add tests**: Ensure new code has test coverage
5. **Update CHANGELOG.md**: Add entry under "Unreleased"

### Submitting

1. Create a feature branch:

```bash
git checkout -b feature/amazing-feature
```

2. Make your changes with clear commit messages

3. Push to your fork:

```bash
git push origin feature/amazing-feature
```

4. Open a Pull Request on GitHub

### PR Description

Include in your PR description:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Breaking Changes**: Any API changes?
- **Related Issues**: Link to related issues

Example:

```markdown
## Summary
Adds `.info()` method to Environment class for printing summary statistics.

## Motivation
Users frequently need to inspect environment properties. This provides
a convenient way to see all key information at once.

## Testing
- Added tests in `test_environment.py`
- Manually tested with various environment types
- All existing tests pass

## Breaking Changes
None

## Related Issues
Closes #42
```

## Documentation

### Building Documentation

```bash
# Install docs dependencies
uv sync --extra docs

# Serve locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Documentation is at [https://edeno.github.io/neurospatial/](https://edeno.github.io/neurospatial/)

### Adding Documentation

- **New features**: Update relevant user guide pages
- **API changes**: Docstrings are auto-generated
- **Examples**: add a notebook to `examples/` as a jupytext `.py`/`.ipynb` pair
  (editing either keeps the pair in sync via jupytext), then run
  `uv run python docs/sync_notebooks.py` to refresh the committed
  `docs/examples/` mirror. CI verifies both are in sync.

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions handles PyPI release

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/edeno/neurospatial/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/edeno/neurospatial/issues)
- **Contact**: Email <eric.denovellis@ucsf.edu>

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment

## Recognition

Contributors will be acknowledged in:

- CHANGELOG.md
- GitHub contributors page
- Future publications using neurospatial

Thank you for contributing to neurospatial!
