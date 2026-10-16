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

The construction adapter captures supported native face inputs once when
publishing owned binning. The obsolete finite-volume provider protocol is
removed; numerical capability comes from the stored faces source. Environment operations read the
resulting components. Geometry without supported faces has a `MissingFaces`
reason. Saved binnings use
`KnownFaces` for validated geometry and retain that reason when geometry is absent.

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
`RectilinearGrid` axes/mask plus their forward world map. Migrate consumers via
`require_grid_context` and map coordinates explicitly. Keep `rectilinear` and
legacy world-grid properties truthful. Framed interpolation/allocation share
bounded source-plane recovery capped at a quarter of the smallest cell width.
QuadMesh consumers transform corners and reshape values with the source mask.
Partial clipping drops the tensor-grid component. `GridBinningSpec` separately
describes future rebin construction and must not replace current-cell geometry.

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

Grid, hexagonal, track, mesh, and polar layouts own their geometry assembly.
Polar environments are built on `_PolarGridLayout`, a masked grid over
`(distance, angle)` whose cell and face measures are the grid's scaled by
radius. A new layout or subclass does not need registration in the diffusion
solver. These are internal interfaces, not additional scientist-facing factory
requirements.

`layout.engines._polar` also builds physical polar edge distances and circular
seam edges before the environment is initialized. Its masked-grid build records
`circular_angle` alongside the shared grid parameters.

Factories build native layouts once, then extract owned binning, lookup data
and a lazy numerical face recipe. No builder, layout facade or opaque provider
callback remains in the resulting environment. One frozen `_PublishedGeometry`
owns the binning, coordinate descriptor, arena and grid context; its public
array views reject writes and its graph rejects topology and attribute edits.
`env.connectivity.copy()` supplies an independent editable graph.

`environment/_publication.py:publish` validates and publishes owned data
atomically and starts fresh caches. Every factory, derivation and reader calls
it exactly once per returned environment; `Environment` itself only
allocates. Readers decode binning and graph records directly through this
path, without replaying build parameters. Readers validate the storage
format; publication runs the shared geometry checks once.
`geometry/_validation.py` checks positions, ordered faces and distances, measures,
locator/component agreement, frames and immediate clip mappings. Center lookup
is checked on a detached locator so publication does not warm the saved value's
lookup cache. Validated embedded tracks may revisit identical spatial centers; center
lookup must still choose a bin at that same coordinate. Other bins retain
strict center identity checks.
Native mesh and hex logical indexing is distinct from a framed raster's source
coordinates.
Descriptive layout names and parameters never grant numerical capabilities.

Stored node-link graphs use an optional `_neurospatial_reserved_attributes`
sidecar to preserve user attributes named `id`, `source`, `target`, or multigraph
`key`, which NetworkX otherwise overwrites with graph identities. Its rows must
match the graph's identities and contain only these reserved attributes. Records
without the sidecar retain their existing format. Older readers can still read
the topology of new records, but retain their previous limitation of dropping
these reserved user attributes; use a current reader to recover them.


`subset` and `apply_transform` derive a binning through `geometry/_derive.py` and
publish it through the same owned-data path. Subsets select whole bins and restrict
ordered faces, locators and components. Translations and similarities preserve
finite-volume geometry; diagonal grid scaling retains axis-aligned faces.
Other affine maps retain inverse lookup and cell measures, with a `MissingFaces`
reason where right angles are lost; tracks refuse non-similarities. New
derivations follow these rules and never rebuild through a factory.
Triangle drawing vertices are retained under all invertible affine maps,
independently of whether finite-volume faces remain valid for smoothing.
Custom providers must use registered adapters to supported built-in owned
snapshots. Live delegated lookup and track conversion callbacks are refused.

`env.arena` describes continuous support independently of the active analysis
bins. Samples, grid masks and image masks supply computational domains without
asserting measured physical walls. Explicit polygons and measured track widths
record known footprints; units, frame and regions remain environment-owned.
Subsets preserve the parent arena, while affine transforms map its geometry
with the bins. Exact lookup applies both the arena gate and locator ownership.

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
outlines or masks. Version 1/`1.x` records are refused with recreation guidance.
Pure codecs in `geometry/_codecs.py` and `geometry/_arena.py` store typed arrays and JSON
metadata; polygon WKB bytes and offsets preserve ordered cells and holes.
Shared snapshot references retain locator ownership without duplicating each
record. Bounded KD-tree and mesh triangle-lookup memos are rebuilt from saved data.
Standard NWB containers reject slash-containing names, so a one-row ragged
DynamicTable stores arrays under safe column names. Its description holds the
JSON metadata and canonical-key/shape/dtype mapping, without padding or a
custom extension.

`Environment._face_geometry` builds from `env._binning.faces` and caches the
validated faces once for the immutable environment;
the diffusion snapshot (`_diffusion_geometry`) and the gradient and divergence
operators all derive from it. Builders read it through
`require_face_geometry(cache, binning, operation)` in
`environment/_fields.py`, which names the analysis when the layout has no cell
faces. Private analyses live in one `AnalysisCache`
(`environment/_cache.py`) with a lock per entry, so concurrent first requests
compute each value once. `with_metadata` and `with_regions` share their
parent's cache; derivations, readers and unpickled environments start with an
empty one. `clear_cache()` empties it.

Numerical consumers declare their small environment requirements in
`ops._protocols`, so Cartesian and polar environments satisfy them without
casts to a broad host interface. `ops._types` describes the diffusion and
MRF eigenbasis cache dictionaries with `TypedDict`; it imports no scientific
libraries at runtime. The dense-operation warning threshold is
`neurospatial._constants.LARGE_KERNEL_THRESHOLD`, shared by diffusion and
Gaussian smoothing without an import back into the smoothing adapter.
Every ops function that takes an environment names it by such a protocol; ops
never imports `Environment`, even for type checking. CI type-checks both
coordinate variants of the common Environment against these consumer
protocols in `tests/typing/finite_volume_consumers.py` and
`tests/typing/ops_consumers.py`.

### Owned geometry and builder boundaries

`geometry/` holds immutable binning records, locators, arena evidence, coordinate
semantics, validation and codecs. It does not import environment adapters,
layout builders, numerical solvers, storage backends or rendering libraries.
`neurospatial.binning` and `neurospatial.arena` retain their public import paths.
Import geometry from its package; there are no compatibility aliases.

`layout/` constructs native binnings. `ops/` implements numerical algorithms on
explicit geometry and arrays. These packages are independent peers above
`geometry/`. `environment/` coordinates them and owns cached analyses. Import
contracts enforce these package boundaries without ignored dependency edges.
Importing `neurospatial.binning` does not load native builders, `environment.core`
or Matplotlib. Standalone validation functions retain their public contracts;
publication additionally checks agreement between the graph and owned binning.

Subset requests are validated in `environment/_subset.py`, and the selection of
whole bins with their graph and ranges is `geometry/_derive.py:select_bins`,
which `from_layout(arena=...)` also uses. Both supported rebin calls
live in `environment/_rebin.py`. Factor rebin preserves its historical grid
partitioning rules, while `rebin(bin_size=...)` uses the saved binning
specification and frame. Both publish through the same owned-data path. Shared
time-support calculations live in `_intervals.py`. Linear time allocation on
rectangular grids (the vectorized ray-grid traversal) lives in
`ops/_linear_allocation.py`; exact allocation on planar polygon cells, shared
by `Environment.occupancy` and linear spatial encoding, lives in
`ops/_polygon_allocation.py`. Point-to-bin lookup gated by the arena is
`geometry/_lookup.py:locate`. Inferring an arena for a binning, checking that
they agree and removing wall-blocked faces live in `geometry/_arena_binning.py`;
each arena type bounds its own domain (`Arena.domain_bounds`). Boundary bins
for every layout come from `geometry/_boundary_bins.py`, which asks each locator
for its logical cell grid. Physical wall-frame
evidence lives in `geometry/_wall_frame.py`. Wall-hit bin ownership is an
`ops/_boundaries.py` calculation with explicit geometry inputs.

### Environment architecture

New operations are module functions with explicit inputs: the published
`Binning`, the connectivity graph, arrays, the arena, the coordinate descriptor
and, for a function that caches, the environment's `AnalysisCache`. A function
takes the environment only when it needs the whole facade. A package that
consumes environments declares a small consumer `Protocol` of the members it
reads, as `ops/_protocols.py` does. An environment method is an optional
scientist-facing entry point that unpacks its owned data and delegates to the
function in one call; the function does not call back into the environment.

When extracting an existing implementation, preserve the method's name,
signature, decorators and docstring exactly. Put its implementation in a private
`_<method>` function in a `_`-prefixed module, taking the components it reads
followed by the method's parameters in their original order and keyword-only
positions. Forward keyword arguments by keyword. Functions that read no
environment members take no environment components. Keep builder inputs in
construction adapters.

Move logic unchanged: replace reads of `self` members with the passed
components, replace helper-method calls with module-function calls, and split
long bodies only at existing section comments.
Fix discovered bugs separately. Moved warnings use
`stacklevel=_external_stacklevel()` from `neurospatial._stacklevel` so the warning
still points to the scientist's line. Keep implementation functions private and
out of `__all__`; the public API snapshot stays unchanged. Preserve moved helper
docstrings, and give new private functions a one-line NumPy summary plus a
`Parameters` section when an argument's meaning is not clear from its name and
type.

Geometry uses the [owned publication path](#diffusion-geometry-providers)
described above. Derivations return new environments with fresh analysis caches.
The environment owns its nearest-query tree and differential-operator caches.
Compatibility functions delegate through public query, operator and cache-control
methods. Numerical builders receive explicit centers, trees or face geometry;
they do not inspect or mutate environment caches. `env.binning` is a detached
public snapshot for consumers preparing geometry, and keeps native faces lazy.
`geometry/_graph.py` projects newly constructed binning into standard navigation
graph attributes for explicit bins and clipping. It preserves bin IDs and pair
row IDs. Derivations such as subsets retain their existing graph metadata and
edge ordering rather than rebuilding those annotations from geometry alone.

There is one public runtime `Environment` class. Factories infer its static
Cartesian or egocentric-polar coordinate type. To make a member Cartesian-only,
annotate `self` as `Environment[CartesianSystem]` for type checkers and
decorate it with `@cartesian_only` from `environment/_cartesian.py`, or
`@cartesian_only(grid=True)` when it needs a Cartesian grid. For a property,
put the decorator below `@property`. The decorator is the only record of the
capability. It refuses unbound calls such as `Environment.bin_at(polar_env,
points)`, and `CARTESIAN_OPERATIONS` and `CARTESIAN_GRID_OPERATIONS` in
`environment/core.py` are derived from it. On polar instances,
`Environment.__getattribute__` refuses the decorated members, so `hasattr` is
False. A method that refuses only one mode on polar bins calls
`require_cartesian(coordinates, "...")` for that mode.
`test_every_public_member_is_classified_for_polar` checks that every other
public member has a reviewed polar meaning. The attribute-filtering hook exists
only at runtime: type checkers must still reject missing attributes. Negative
static contracts in `tests/typing` guard these failures with checked ignore
comments.
Shared code asks the owned coordinate descriptor about lengths, axes and
circularity, or uses public `coordinate_kind` for metadata. Do not discriminate
using environment classes or descriptive layout labels.

Environment adapters read components from the owned binning; other packages
use `env.binning`: faces for
smoothing, a rectilinear grid for grid interpolation and native N-D time
allocation, exact planar cells for polygon time allocation, and linearization
for track coordinate conversions. Each layout engine's `binning()` assembles
these components from its own state through `layout/_assembly.py`; the
construction adapter in `layout/_construction.py` takes that record with the
engine's connectivity, ranges and provenance, and never inspects layout
classes or build parameters. Custom engines implement the same `binning()`.
Do not dispatch new behavior by comparing
`_layout_type_tag` strings. Refusals name the operation and suggest a supported
alternative. Integer-factor `rebin` requires uniform rectangular cells and
native grid navigation, allowing physical walls to remove connections. It
refuses custom path lengths or topology that cell geometry cannot reconstruct,
and retains truncation warnings and complete-face wall constraints.
Both rebin modes live in `environment/_rebin.py`; selection of existing bins
lives in `geometry/_derive.py:select_bins`. Rebin construction policies remain
distinct:
factor mode retains native lazy faces and its recorded grid parameters, while
resolution mode prepares faces for the retained affine frame. Both use the
same atomic `publish` function for owned data, walls and
fresh analysis caches.
`rebin(bin_size=...)` uses an owned `GridBinningSpec`, independently of descriptive
provenance. It re-bins the retained known arena, preserving the grid anchor and
affine frame; requested sizes are physical lengths along those axes. Subset and
clipping preserve the specification; combining does not invent a shared one.
Unsupported families and old records without a specification refuse new-size
re-binning. Fields and models do not transfer automatically.

Border and visibility consumers use shared planar wall evidence in
`ops/_boundaries.py`. Physical mode reads asserted footprints and internal walls;
analysis mode is explicit. Analysis cuts never create physical walls. Boundary
contacts and wall-hit bin ownership use retained source frames with locally
bounded inverse rounding. Viewshed bin coverage remains sampled, while first
contacts and cue occlusion use geometric intersections. Unsupported geometry
must refuse instead of silently switching boundary sources.

`Environment` is the only environment dataclass and holds thin forwarding
methods. Algorithms live in private module functions using narrow protocols;
there are no environment mixins or broad host protocol.

Owned geometry, validation, codecs and pure coordinate math live in `geometry/`;
native builders and their construction adapters live in `layout/`. The public
`neurospatial.binning` and `neurospatial.arena` imports remain stable. There
are no compatibility alias modules, and pickles from earlier versions are not
loadable.

Run `uv run lint-imports --no-cache` to enforce `.importlinter`. Environment
adapters sit above independent `layout` and `ops` packages, which consume
`geometry`. The whole geometry package cannot import builders, environments,
analysis, storage or rendering packages. Type-only imports are excluded. Affine
convenience functions call the public environment method through a narrow
consumer protocol; Environment retains construction and analysis caches. No
import exceptions are needed. New geometry modules are covered automatically.
Keep `import-linter` in the dev extra; CI runs it on Python 3.13.

Check refactors with `scripts/environment_baseline.py`. Before changing code,
run `uv run python scripts/environment_baseline.py record base.npz` on the
branch's base commit. After changing code, record `new.npz` the same way, then
run `uv run python scripts/environment_baseline.py compare base.npz new.npz`.
Outputs, refusals, signatures and docstrings must match exactly, or the pull
request must list and verify every intended difference. Also run the suite,
doctests, executable documentation, public API snapshot, Ruff and the source
and static-contract mypy checks.

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
