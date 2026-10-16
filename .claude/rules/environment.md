---
paths:
  - "src/neurospatial/environment/**"
  - "src/neurospatial/layout/**"
  - "src/neurospatial/geometry/**"
  - "src/neurospatial/ops/**"
---

# Environment, geometry, layout and ops code

Before a structural change, read the "Environment architecture" and "Diffusion
geometry providers" sections of `docs/contributing.md`. They give the import
layers in `.importlinter`, the single publication path, the analysis cache,
how thin public methods pass explicit components to private functions,
Cartesian-only members, the derivation and factory rules, and the
`scripts/environment_baseline.py` comparison every refactor needs.

- Every environment is created once by `environment/_publication.py`, through
  `publish_layout`, `publish_derived` or `publish` itself. Do not build an
  `Environment` any other way or change one after publication.
- Memoized analyses go in the environment's `AnalysisCache` under a
  module-level `CacheKey` or `RecentKey` in `environment/_cache.py`, never in
  a `cached_property`.
- Mark a Cartesian-only member with `@cartesian_only` (or
  `@cartesian_only(grid=True)`) from `environment/_cartesian.py` and annotate
  `self` as `Environment[CartesianSystem]`.
- A custom layout engine needs only `build(...)` and `binning()`; pass the
  built engine to `Environment.from_layout`. A new built-in engine also needs
  a `LayoutType` member and a `_LAYOUT_MAP` entry in `layout/factories.py`, an
  export from `layout/__init__.py`, a `_data_parameters` class attribute
  naming its data arguments, and an entry in `MINIMAL_BUILD_PARAMS` in
  `tests/layout/test_layout_engine.py`. The protocol-adherence test skips
  engines missing from that dict instead of failing.
- A new factory takes its data positionally and every option by keyword after
  `*`; add it to `_DATA_ARGUMENTS` in
  `tests/environment/test_factory_arguments.py`.
