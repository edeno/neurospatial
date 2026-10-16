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

Smoothing, diffusion, smoothness bases, gradients and divergence all use one
layout capability, defined in `neurospatial.layout._diffusion`:

- `FiniteVolumeGeometryProvider.finite_volume_geometry()` returns a copy of the
  layout's connectivity graph with face measures (`A`) and edge distances
  (`distance`) on every edge, plus cell measures in active-bin order. Gradient
  and divergence index edge fields by `env.connectivity`, so the environment
  rejects a graph whose edge order or orientation differs from it. Each
  undirected edge defines the scalar, symmetric two-point conductance
  `A / distance` used by the current numerical scheme.
- `FaceGeometry.from_graph` validates those faces into read-only per-edge arrays
  in connectivity edge order. `DiffusionGeometry.from_faces` assembles the
  conductances `W = A / distance`; components come from positive conductances,
  so zero-area (corner-touching) edges and isolated bins form separate
  components.

The layout-to-binning factory checks these protocols once when publishing
`env._binning`. Environment operations read the resulting components. A layout that cannot
supply geometry is represented by a `MissingFaces` reason. Saved binnings use
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

Layouts own `connectivity`, `bin_centers` and the grid geometry, and an
environment publishes them read-only: its geometry attributes are views of one
frozen `_PublishedGeometry` that cannot be reassigned. It includes one `Binning`
snapshot with owned bin measures and ordered neighbor pairs. The arrays reject
writes, and both graph topology and its attribute dictionaries reject edits.
Node positions, original grid indices and edge vectors are owned tuples, so
restoration cannot leave mutable coordinate lists aliased by graph copies.
`env.connectivity.copy()` supplies an editable graph for separate analyses.

Stored node-link graphs use an optional `_neurospatial_reserved_attributes`
sidecar to preserve user attributes named `id`, `source`, `target`, or multigraph
`key`, which NetworkX otherwise overwrites with graph identities. Its rows must
match the graph's identities and contain only these reserved attributes. Records
without the sidecar retain their existing format. Older readers can still read
the topology of new records, but retain their previous limitation of dropping
these reserved user attributes; use a current reader to recover them.

`_setup_from_layout` swaps that object whole and advances `_state_version`,
which keys every derived cache, in one step. Factories build native layouts;
readers decode the saved binning and node-link graph into `BinningLayout`, a
protocol-compatible facade. Neither loading nor copying a loaded environment
runs a layout builder. The facade checks graph positions, ordered edges and
distances, face measures, and agreement between locators and their grid, shape
or track components before publication. Its legacy grid fields derive from
those components. Descriptive build parameters are retained for inspection.
`_setup_from_layout` validates the graph, publishes the geometry and clears
analysis caches through the normal constructor. Changing a layout's bins and
calling it again is not supported.

`subset` and `apply_transform` derive a binning through `layout/_derive.py` and
publish it through the same facade. Subsets select whole bins and restrict
ordered faces, locators and components. Translations and similarities preserve
finite-volume geometry; diagonal grid scaling retains axis-aligned faces.
Other affine maps retain inverse lookup and cell measures, with a `MissingFaces`
reason where right angles are lost; tracks refuse non-similarities. New
derivations follow these rules and never rebuild through a factory.
Triangle drawing vertices are retained under all invertible affine maps,
independently of whether finite-volume faces remain valid for smoothing.
Custom layout lookup and track conversions retain their delegated in-memory
behavior through private adapters; saving them still requires a supported codec.

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
Pure codecs in `layout/_codecs.py` and `arena.py` store typed arrays and JSON
metadata; polygon WKB bytes and offsets preserve ordered cells and holes.
Shared snapshot references retain locator ownership without duplicating each
record. Bounded KD-tree and Delaunay memos are rebuilt from saved data.
Standard NWB containers reject slash-containing names, so a one-row ragged
DynamicTable stores arrays under safe column names. Its description holds the
JSON metadata and canonical-key/shape/dtype mapping, without padding or a
custom extension.

`Environment._face_geometry` builds from `env._binning.faces` and caches the
validated faces once per state version;
the diffusion snapshot (`_diffusion_geometry`) and the gradient and divergence
operators all derive from it. Consumers read it through
`_require_face_geometry(operation)`, which names the analysis when the layout
has no cell faces. The existing state-version and `clear_cache()` mechanisms
own every cache lifetime.

Numerical consumers declare their small environment requirements in
`ops._protocols`, so Cartesian and polar environments satisfy them without
casts to the entire mixin interface. `ops._types` describes the diffusion and
MRF eigenbasis cache dictionaries with `TypedDict`; it imports no scientific
libraries at runtime. The dense-operation warning threshold is
`neurospatial._constants.LARGE_KERNEL_THRESHOLD`, shared by diffusion and
Gaussian smoothing without an import back into the smoothing adapter. CI type-checks both concrete environment
types against these consumer protocols in `tests/typing/finite_volume_consumers.py`.

### Environment architecture

New operations are module functions with explicit inputs. When an operation
needs an environment, declare a small consumer `Protocol` listing only the
members it reads in `environment/_protocols.py`; numerical operators use
`ops/_protocols.py`. An environment method is an optional scientist-facing
entry point that delegates to the function in one call.

When extracting an existing implementation, preserve the method's name,
signature, decorators and docstring exactly. Put its implementation in a private
`_<method>` function in the same module, with `env` first and the remaining
parameters in their original order and keyword-only positions. Forward keyword
arguments by keyword. Type `env` with the narrow consumer protocol; functions
that read only the layout take `layout`, and functions that read no environment
members take no `env` argument. Move private helpers into module functions in the
same change and remove their declarations from the mixin host protocol.

Move logic unchanged: rename `self` to `env`, replace helper-method calls with
module-function calls, and split long bodies only at existing section comments.
Fix discovered bugs separately. Moved warnings use
`stacklevel=_external_stacklevel()` from `neurospatial._stacklevel` so the warning
still points to the scientist's line. Keep implementation functions private and
out of `__all__`; the public API snapshot stays unchanged. Preserve moved helper
docstrings, and give new private functions a one-line NumPy summary plus a
`Parameters` section when an argument's meaning is not clear from its name and
type.

Geometry uses the [published geometry and construction path](#diffusion-geometry-providers)
described above: updates go through `_setup_from_layout`, and derived values
use `versioned_cached_property` caches.

An operation that interprets `bin_centers` as Cartesian `(x, y[, z])`
coordinates belongs to `CartesianCoordinates`, composed by `Environment`.
Polar environments do not inherit it. A shared method with a Cartesian mode
calls `self._require_cartesian("<call>")`; a new coordinate system defines this
guard to refuse unsupported calls and is covered by
`test_every_inherited_member_is_classified_for_polar`. Shared code that
measures distance, chooses length scales or names axes asks
`env._coordinate_system`. A new coordinate system defines one
`CoordinateSystem` class with those rules; analysis code does not test the
persistence type tag `_POLAR`.

Layout-specific operations read components from `env._binning`: faces for
smoothing, a rectilinear grid for grid interpolation and native N-D time
allocation, exact planar cells for polygon time allocation, and linearization
for track coordinate conversions. Only the layout-to-binning
factory checks `FiniteVolumeGeometryProvider`, `RectilinearGridProvider` and
`TrackLinearizationProvider` on layouts. A layout supplies a capability when it
defines the corresponding method. Do not dispatch new behavior by comparing
`_layout_type_tag` strings. Refusals name the operation and suggest a supported
alternative. Integer-factor `rebin` retains its existing RegularGrid-family
restriction, truncation warnings and complete-face wall constraints.
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

A mixin is a plain class for one domain; `_BaseEnvironment` is the dataclass.
Mixins keep no state except `versioned_cached_property` caches and annotate
`self` with a protocol checked against the concrete environment types. Mypy
rejects a concrete mixin self type such as `self: "_BaseEnvironment"` because
the erased self type is not a supertype of the mixin class.

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
