# Stage F source and dependency inventory

Recorded on 2026-10-15 at Stage E `936b29ae`. This is planning evidence, not a
claim that these dependencies have been removed. See [PLAN.md](PLAN.md).

## Stored-layout attribute audit

The [access list](layout-accesses.tsv) contains 100 AST attribute nodes named
`layout` or `_layout` under `src/neurospatial`. It includes constructor writes,
public reads and native face-provider reads; these are not 100 distinct
consumers. Dynamic `getattr`/`hasattr` accesses and string/type references need
an additional audit during F0.

| Source family | Attribute nodes | Migration concern |
| --- | ---: | --- |
| animation | 21 | Native/framed grid contexts, QuadMesh updates, Napari raster restrictions, truthful diagnostics |
| encoding | 10 | FFT/logical indexing, masks and segmentation; preserve native hex support |
| environment | 49 | Constructor/publication, owned exports, plotting, metrics, derivation and caches |
| io | 1 | Current record's descriptive layout category; restore paths also construct a facade |
| layout | 11 | Native faces retain a builder; data validation and plotting currently live in the facade module |
| ops | 2 | Environment-transform convenience function reads a layout while deriving binning |
| simulation | 3 | Exposed environment shape/metadata consumers |
| stats | 3 | Circular grid shuffling/context and capability checks |

Reproduce the list with `ast.parse` on each Python source and select
`ast.Attribute` nodes whose `attr` is `layout` or `_layout`; record the filename,
line and `ast.unparse(node)`. Classify each entry before changing it. Legitimate
builder-local/UI accesses need an explicit reason; the completion gate concerns
environment/runtime dependencies, not unrelated third-party `.layout` methods.

## Concrete dependency seams

| Source | Current evidence | Stage F requirement |
| --- | --- | --- |
| [core.py](../../../src/neurospatial/environment/core.py) | `_BaseEnvironment` composes eight mixins; `_PublishedGeometry` stores binning plus duplicate views; `__init__` stores `_layout`; `_setup_from_layout` publishes state | One owned publication path and thin public class; no retained builder/facade |
| [polar.py](../../../src/neurospatial/environment/polar.py) | Separate public class, `_POLAR`, guarded attribute lookup, specialized boundary/circularity handling | Per-instance coordinate policy; preserve refusals and polar metrics during class merge |
| [_coordinates.py](../../../src/neurospatial/environment/_coordinates.py) | Cartesian/polar distance, length and axis-label policies already exist | Reuse these policies as owned instance state; normalize circularity once |
| [_binning.py](../../../src/neurospatial/layout/_binning.py) | `LayoutFaces` stores `layout: LayoutEngine`; `binning_from_layout` mixes builder extraction and data ownership | Closed data-backed lazy face sources; separate construction adapter from geometry data |
| [_ownership.py](../../../src/neurospatial/environment/_ownership.py) | `_OwnedLayout` subclasses `BinningLayout`; accepts native lazy face delegation | Preserve ownership without a layout-shaped export or retained callback |
| [_binning_layout.py](../../../src/neurospatial/layout/_binning_layout.py) | Contains publication validation, legacy grid context and component plotting | Extract reusable data functions before deleting wrapper; retain hex versus framed distinctions |
| [_protocols.py](../../../src/neurospatial/environment/_protocols.py) | Broad `EnvironmentProtocol` exposes `layout` and mixin helpers; narrow consumer protocols already exist | Remove host-only surface; retain/complete narrow scientific consumer protocols |
| [files.py](../../../src/neurospatial/io/files.py) | Saves `_POLAR` as `coordinate_kind`; restores `BinningLayout` and selects a concrete class | Restore one class with validated coordinate data; retain supported wire formats |
| [ops/transforms.py](../../../src/neurospatial/ops/transforms.py) | Affine primitives and high-level environment construction share a module | Separate dependency roles without breaking documented public import paths |
| [_typing.py](../../../src/neurospatial/_typing.py) | Public duck-typed `EnvironmentLike` explains the sibling types | Keep structural input support; remove outdated class assumptions |
| [static contracts](../../../tests/typing/environment_protocol.py) | Tests polar `bin_at` as a static error and both coordinate policies explicitly | Preserve coordinate-specific typing on factory results with one runtime class |

## Planning probes and CI evidence

A standalone `/private/tmp/environment_unified_typing_probe.py` passed strict
mypy on 2026-10-15: one `Environment[C]` runtime class, two factory-inferred
coordinate types, shared operations and a statically refused polar Cartesian
method. It is a small feasibility probe, not a test of the production factory,
method decorators, restoration or all supported interpreters. Integrate those
contracts in F0/F3 before relying on this design.

Sampled existing stack failures:

- [#74 Ubuntu/Python 3.12](https://github.com/edeno/neurospatial/actions/runs/38730118263/job/116405699855):
  `test_regions_add_point_and_remove`, ambiguous NumPy array truth at equality.
- [#73 Ubuntu/Python 3.13](https://github.com/edeno/neurospatial/actions/runs/38730119104/job/116405992811):
  `test_thresholds_attrs_match_resolved_defaults[spatial]`, physical walls unknown.

The open-stack snapshot is `/private/tmp/neurospatial-current-prs.json`; the
new Stage E PR's initial checks are in
`/private/tmp/neurospatial-stage-e-published.json`. These are time-stamped
observations and must be refreshed at implementation/publication gates.

## Planning validation

- All 25 local links in the plan, inventory and updated overview resolve.
- All 100 inventory entries match the source AST at their recorded locations.
- The generic typing prototype passes strict mypy in the current interpreter's
  mode and with `--python-version 3.10`. This does not replace production F3
  typing or runtime tests.
- Repository Ruff passes; all 787 Python files are formatted; whitespace checks
  pass. This branch changes planning documents only, so the full scientific
  suite was not rerun. Stage E's completed local gate remains the parent record.

## Final disposition

The original 100 stored-layout AST reads/writes have been resolved. The final
AST scan of all source modules finds zero attributes named `layout` or `_layout`.
Native builder usage is confined to construction adapters, and environment state
is recursively checked after copy/deepcopy/pickle/dict restoration for retained
builders or callbacks. Closed native face recipes retain only numerical data.
Consumers use owned binning, narrow protocols and the dedicated grid context.

`layout/_construction.py` extracts native data once; `layout/_validation.py` keeps
the removed facade's consistency algorithms; `layout/_plotting_binning.py` keeps
component rendering. The runtime data and numerical dependency boundaries are
enforced by `.importlinter`; its only specific ignored import is the public
transform convenience function's lazy integration edge. The original access
list remains historical evidence rather than a list of current source lines.
