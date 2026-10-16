---
paths:
  - "src/neurospatial/environment/**"
  - "src/neurospatial/layout/**"
  - "src/neurospatial/ops/**"
---

# Environment, layout and ops code

Before a structural change, read the "Environment architecture" and "Diffusion
geometry providers" sections of `docs/contributing.md`. They define where new
operations live, how thin public methods delegate to domain functions, the published-geometry
construction path, Cartesian versus polar capabilities, and the
`scripts/environment_baseline.py` comparison every refactor needs.

A new layout engine needs more than its engine file: a `LayoutType` member and a
`_LAYOUT_MAP` entry in `layout/factories.py`, an export from
`layout/__init__.py`, and an entry in `MINIMAL_BUILD_PARAMS` in
`tests/layout/test_layout_engine.py`. The protocol-adherence test skips engines
missing from that dict instead of failing.
