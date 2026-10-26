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
methods hand that data to private module functions. This section explains the
design choices behind that shape.

#### Layers

Packages depend downward. Storage, rendering, decoding, encoding and
statistics build on the `Environment` facade. The facade builds on `layout`
(which constructs bins) and `ops` (numerical algorithms), and those build on
`geometry`, which holds the owned records and imports no builder, analysis,
storage or rendering package. `layout` and `ops` are peers. When a new import
points upward, move the imported code down or the importer up. The exceptions
are function-local imports in the `Environment` methods that hand off to
storage or rendering (`to_file`, `to_nwb`, `animate_fields`, ...), which load
those packages only when called. Nothing checks these layers automatically;
review imports by hand.

A package imports another package's names from that package's `__init__`
(for example `from neurospatial.geometry import Binning`), never from its `_`
modules, and reads an `Environment` only through public members. An internal
helper that another package needs keeps its leading underscore where it is
defined, and the package `__init__` re-exports it under that name
(`from neurospatial.ops import _ensure_global_modes`); only `__all__` names are
public API. `geometry` is wholly internal, so its names carry no underscore.
`geometry/` has no import cycles: when two modules need each other, move the
shared definition to a lower module.

`neurospatial.binning` and `neurospatial.arena` are the public import paths
for geometry records. Importing them does not load layout engines,
`environment.core` or Matplotlib.

#### One publisher

`environment/_publication.py:_publish` is the only code that creates an
`Environment`; `Environment()` itself always raises E1006. Funnelling every
factory, derivation and reader through one function means validation runs in
one place, and an environment cannot exist in a half-built state. The three
entry points are `publish_layout` for a built layout engine, `publish_derived`
for `subset`, `clip`, `rebin`, `apply_transform`, `with_track_width` and
`combine`, and `_publish` itself for `from_bins` and the file, dict and NWB readers, which decode stored
geometry without replaying build parameters.

A published environment never changes. The graph is frozen, geometry arrays
are read-only (handed out as `.view()`s of arrays marked with
`setflags(write=False)`; geometry records restore the flag after unpickling
through `geometry._arrays.ReadOnlyState`), public attributes cannot be
assigned, and equality is by identity, so cached analyses and
`copy.copy`/`copy.deepcopy` (which return the same object) are always valid.
`env.to_networkx()` returns an editable copy. `with_metadata` and
`with_regions` return a new object over the same geometry and cache.

The build record (`_Provenance`) only describes how the bins were built, as
scalar choices such as `bin_size`, the derivations applied since
(`env.derivations`) and how long a unit of the choices' lengths is in the
current coordinates (`length_scale`: multiplied by each similarity
`apply_transform`, None after any other transform, reset by `rebin`). It never
holds input data: an engine
declares its data arguments once, with `@capture_build_params(data=(...))` on
its `build` method, and any other argument that is not a scalar or a flat
sequence of scalars is refused when the engine is built. The layout label is
one vocabulary, `LayoutType` in `layout/_labels.py`: an engine's
`layout_type` is a member's value, its binning carries the same label as
`Binning.kind`, `env.layout_type` reads `Binning.kind`, and derivations keep
the parent's label (`combine` reports `"Combined"`). Behavior comes from the
owned binning, graph and arena, so do not branch on `layout_type` or build
parameters.

Pickling keeps the published state and drops the cache. It is for passing an
environment between processes of the same version. Use `to_file`, `to_dict`
or NWB for storage.

#### Layout engines own their binning

Every layout engine, built-in or custom, implements `build()` and
`binning()`, which returns a `Binning` assembled from the engine's own state.
The environment never inspects engine classes or reads build parameters for
geometry, so a custom engine needs no registration: build it and pass it to
`Environment.from_layout(engine)`. Built-in engines compute their neighbor
pairs and pair distances as arrays in `build()`, with vectorized helpers in
`layout/helpers/neighbor_pairs.py` that fix the pair order (it becomes
`env.edge_pairs`, the order of edge fields in gradients and divergence), and
share one rule for bin measures and face directions through
`assemble_binning` in `layout/_assembly.py`. A binning keeps the bins
(`SpatialBins`: centers, measures and each bin's measure dimension, which
can differ between combined components) separate from their neighbor pairs
(`BinConnections`), because they change for different reasons. Walls close
pairs (`is_open`) instead of deleting them, so environments over the same bins
keep one pair order; `connections.pairs`, the open pairs, is `env.edge_pairs`,
and faces and face normals align with it. Engines do not build graphs: a
built-in engine's `connectivity` is a convenience derived on first access
from its binning with `geometry.graph_from_binning`, the same graph an
environment publishes, and editing it changes nothing the engine publishes.
Each built-in engine names the build option that sets its cell size in the
class attribute `size_parameter` (`"bin_size"`, `"hexagon_width"`,
`"point_spacing"`, ...): `from_samples` passes its `bin_size` to that option,
and a refused `rebin` reads its recorded value, through the `LayoutType`
registry in `layout/factories.py`, to suggest a rebuild at the coarser size.
Polar environments use a masked grid over
(distance, angle) with circular seam edges and are built only by
`Environment.from_polar_egocentric`.

#### Methods pass explicit components

A public method is a thin entry point. It unpacks owned data (the `Binning`,
graph, arena, coordinate descriptor and, for a function that caches, the
`AnalysisCache`) and calls a private module function. That function takes the
environment only when it needs the whole facade, and never calls back into the
environment. This keeps the numerical code testable on explicit arrays and
stops the facade from growing back into a mixin tangle. Warnings use
`stacklevel=_external_stacklevel()` so they point at the scientist's line;
errors use `_format_error` to say what failed, why and how to fix it.

A package that consumes an environment declares a small `Protocol` of the
members it reads, as `ops/_protocols.py` does, so Cartesian and polar
environments satisfy it without casts and `ops` never imports `Environment`.
A public `ops` function that scientists call as `f(env, ...)` keeps taking
the environment: it reads the components it needs once at the top (for
example `env.binning`, `env.arena`, `env.connectivity` or
`env.get_diffusion_geometry()`) and passes them to private functions that take
those components. It does not call environment analysis methods from inside
the computation. Two kinds of method may be called at the boundary: a query the
environment answers from a structure it caches (its point lookup, KD-tree or
diffusion eigenbasis), so repeated calls do not rebuild it, for example
`src_env.bin_at`, `src_env.bin_center_tree()` and `dst_env.diffuse` in
`resample_field`; and the environment's own renderer, `plot_field`. Each base
protocol names one to three members; combine bases rather than repeating
members, and do not repeat a method's full signature unless the function calls
it.

#### Analysis cache

Every memoized analysis lives in the environment's one `AnalysisCache`
(`environment/_cache.py`) under a module-level `CacheKey` or `RecentKey`,
read with `cache.get(key, build)`. Do not use `cached_property` for analysis
values: one cache means `clear_cache()` empties everything, the relabelling
methods can share it with their parent, and derivations, readers and
unpickled environments can start with a fresh one. Each key has its own
re-entrant lock so concurrent first requests compute a value once. Cached
values are never edited. A value that grows, such as an eigenbasis that gains
modes, is an immutable record: `cache.update(key, initial, step)` passes the
current record to `step`, which returns its replacement and a result, under
the key's lock, so the growth happens once and readers never see a partial
record.

Locators hold no search structures of their own. The KD-tree, triangle index
or polygon tree a locator searches is built by `locate(points, indexes)` and
kept in `indexes`, a `LookupIndexes` store keyed by the locator. Environment
code passes its `AnalysisCache` (`cache.lookup_index`), so these structures
are locked, shared and cleared like every other analysis. Code that locates
points repeatedly on a binning outside an environment, such as the copy
`env.binning` returns, passes one `CallIndexes` to every call; `None` builds
the structure for that call only.

A scientist's positions reach the bins through `geometry.locate(binning,
arena, positions, indexes, call=...)`, not a locator's own `locate`. It checks
the column count with `require_position_columns` on every path, with or
without a known arena, so the error names `call`, the operation the scientist
called. A method that uses positions before the lookup, or returns early
without one, calls `require_position_columns` first with the same `call`.

#### Cartesian-only members

There is one runtime `Environment` class; factories give it a static
coordinate type, `Environment[CartesianSystem]` or
`Environment[EgocentricPolarSystem]`. Mark a member that is meaningless on
polar bins with `@cartesian_only(why=..., fix=..., example=...)` from
`environment/_cartesian.py`, and annotate `self` as
`Environment[CartesianSystem]`. `why` says what the member does with Cartesian
coordinates, `fix` names what works on a polar environment and `example` is
code for it, which the tests run. The decorator is
the only record of the capability: it refuses polar environments when the
member is called and feeds `CARTESIAN_OPERATIONS`. The member stays visible, so
`hasattr` is True. Track-only members (`to_linear`, `linear_to_nd`,
`with_track_width`) are not marked, though their `self` is
`Environment[CartesianSystem]` because every track is Cartesian: a polar
environment is never a track, so at run time it gets the same "needs a track"
refusal as every other non-track environment. A
Cartesian-only mode of a shared method, or an analysis
function in another package, refuses with
`neurospatial.geometry.require_cartesian(env.coordinate_system, operation,
why=..., fix=...)`. Both raise `CartesianOnlyError`, a `TypeError`, with one
message layout.
Shared code asks the coordinate descriptor about lengths, axes, its `kind`
and which axes wrap (`circular`, `periodic_axes`), and passes the descriptor
itself to geometry validators; it does not branch on descriptor classes or
layout labels. The publisher derives the descriptor from the declared
coordinate kind and the arena: a polar environment's `EgocentricSurround` is
the one record of whether its angle wraps, so the arena and the descriptor
cannot disagree. Every other public member needs a stated polar meaning, which
`test_every_public_member_is_classified_for_polar` enforces.

#### Derivations

`subset`, `clip`, `rebin`, `apply_transform`, `with_track_width` and
`combine` compute a `_Derivation` record in a private module and pass it to `publish_derived`,
which keeps the parent's class, name, units, frame, regions and arena unless
the record or the caller replaces them. A derivation never changes units,
never rebuilds through a factory and starts with a fresh cache, so fields and
fitted models do not transfer. Metadata follows what the operation means:

- `subset`, `clip` and `rebin` keep every label and accept `name=`.
- `apply_transform` keeps the name and units, maps the regions and arena with
  the bins, and sets `frame` only from the caller's `frame=`, because a
  transformed environment is no longer in the old frame.
- `with_track_width` keeps every label and the bins, and replaces only the
  track's arena with one of the given width.
- `combine` prefixes each child's regions with its key and has an empty name
  unless `name=` is given.

#### Factories

Data arguments are positional and every option is keyword-only after `*`,
with an explicit signature and no `**kwargs` that change meaning. An option
that does not apply to the chosen layout or mode raises an actionable
`ValueError` instead of being ignored. Cartesian factories accept `name`,
`units`, `frame` and `regions`, and `arena` where an arena is meaningful.

#### Checking a refactor

`scripts/environment_baseline.py` records every probe's output, refusal,
warning, signature and docstring on ten environment types, each also read
back from a dict, a file and NWB. Record on the base commit before changing
code, record again after, and compare:

```bash
uv run python scripts/environment_baseline.py record base.npz  # on the base commit
uv run python scripts/environment_baseline.py record new.npz   # on your change
uv run python scripts/environment_baseline.py compare base.npz new.npz
```

Outputs must match exactly, or the pull request lists and explains every
intended difference. If you remove an API that the script probes, update the
script in the same change.

### Numerical geometry, storage and derivation details

Reference notes for geometry that numerical code depends on. Smoothing,
diffusion, smoothness bases, gradients and divergence use the owned face
source in `Binning.faces`:

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

Builders that need faces call `require_face_geometry(cache, binning,
operation)` in `environment/_fields.py`, which names the analysis when the
layout has no cell faces.

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
`grid_shape`, `active_mask`) truthful. Framed point lookup, interpolation and
allocation share bounded source-plane recovery capped at a quarter of the
smallest cell width. A `Binning` gives its frame to the affine step of its
locator (`framed_locator`), which then assigns points by the half-open source
cells (`FramedGrid._locate`), so lookup and linear allocation agree on shared
faces.
QuadMesh consumers transform corners and reshape values with the source mask.
Partial clipping drops the tensor-grid component. `GridBinningSpec` separately
describes how `rebin(bin_size=...)` builds a new grid and must not replace
current-cell geometry.

Binning codec version 4 stores the bins and their connections as separate
records, with frames and normals; readers accept only the version they write
and refuse older records with recreation guidance. New stored geometry
components require a codec version increment so older readers cannot silently
ignore a capability they do not understand. Environment-v3/NWB 3.0 envelopes are
unchanged.

Combined environments retain child binnings instead of inventing physical faces
for navigation bridges. Scalar diffusion assembles the certified child blocks
and adds only explicitly supplied symmetric geometric bridge conductances.
Path-only bridges do not transport the field. A retained child with missing
numerical geometry still prevents whole-environment diffusion; removing it with
a subset allows the certified remainder. Gradient/divergence keep the existing
physical-face contract and refuse combinations with nonphysical bridges.

Publication validates geometry once, so readers only validate the storage
format; a reader rebuilds the graph from the binning instead of reading the
stored one. Every publication checks that each bin center locates its own
bin, on a detached locator so publication does not warm the saved value's
lookup cache. Validated embedded tracks may revisit
identical spatial centers; center lookup must still choose a bin at that same
coordinate. Other bins keep strict center identity checks.

Stored node-link graphs use an optional `_neurospatial_reserved_attributes`
sidecar to preserve user attributes named `id`, `source`, `target`, or multigraph
`key`, which NetworkX otherwise overwrites with graph identities. Its rows must
match the graph's identities and contain only these reserved attributes. Records
without the sidecar retain their existing format. Older readers can still read
the topology of new records, but retain their previous limitation of dropping
these reserved user attributes; use a current reader to recover them.

Subsets select whole bins and restrict ordered faces, locators and components.
Like every derivation, they hand publication only the derived binning;
publication rebuilds the navigation graph, grid index and ranges from it,
so a subset's graph follows its kept pairs and keeps the parent's bin IDs in
`source_grid_flat_index`. Translations and similarities preserve finite-volume geometry;
diagonal grid scaling retains axis-aligned faces. Other affine maps retain
inverse lookup and cell measures, with a `MissingFaces` reason where right
angles are lost; tracks refuse non-similarities. Triangle drawing vertices
are retained under all invertible affine maps, independently of whether
finite-volume faces remain valid for smoothing. Objects from outside
neurospatial enter only through an installed snapshot adapter
(`neurospatial.extensions.SnapshotAdapter`), which converts them to owned
records. Live delegated lookup and track conversion callbacks are refused.

Both rebin modes live in `environment/_rebin.py` and keep distinct
construction rules. Integer-factor `rebin` requires uniform rectangular cells
and native grid navigation, allowing physical walls to close connections. It
refuses custom path lengths or topology that cell geometry cannot
reconstruct, keeps its historical grid partitioning, retains native lazy
faces and its recorded grid parameters, and keeps truncation warnings and
complete-face wall constraints. `rebin(bin_size=...)` uses the owned
`GridBinningSpec`, independently of provenance: it re-bins the environment's
bins within the known arena (a subset or clip keeps its selection),
preserving the grid anchor and affine frame, and prepares faces for that
frame. Requested sizes are physical lengths along those axes. Subset and
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
`geometry/_boundaries.py`. Physical mode reads asserted footprints and internal walls;
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
parent connections; explicit complete-face walls close navigation edges and
remove their finite-volume faces, and pieces that meet only along a wall are
closed pairs. Interior-cut or partially covered walls refuse.
Numerical geometry is published only after convexity, interface and
orthogonality checks; containment and plotting do not depend on certification.

Files and dictionaries use `Environment-v3`; NWB environment records use `3.0`.
They include a required Arena record. Version 2/`2.0` remains readable with an
explicit `UnknownArena`, without inferring physical geometry from parameters,
outlines or masks; a polar environment in such a record is refused because
it has no stored `EgocentricSurround`. Version 1/`1.x` records are refused
with recreation guidance. Pure codecs in `geometry/_codecs.py`, including the
arena codec, store typed arrays and JSON metadata; polygon WKB bytes
and offsets preserve ordered cells and holes. Shared snapshot references
retain locator ownership without duplicating each record. Locator search
structures are not stored; they are rebuilt from saved data. Standard NWB
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
    positions = np.array([[10.0, 10.0]])
    bin_idx = env.bin_at(positions)
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
