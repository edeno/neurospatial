# Phase 21 — Top-level lazy submodule access

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Makes `ns.encoding.<TAB>`, `ns.decoding.<TAB>`, etc. work without eager imports (DESIGN-REVIEW High #5). Autocomplete is the dominant discovery path, and the top-level namespace currently exposes only a fraction of the domains. The project already ships the exact pattern to mirror.

**Inputs to read first:**

- [io/nwb/__init__.py:95-135](../../../../src/neurospatial/io/nwb/__init__.py#L95) — the proven `_LAZY_IMPORTS` dict + `__getattr__` + `__all__` pattern (PEP 562). Copy its shape.
- [src/neurospatial/__init__.py:225-232](../../../../src/neurospatial/__init__.py#L225) — current eager imports (`Environment`, `Region`, `CompositeEnvironment`) and `__all__`; the new `__getattr__` covers the *submodules* (`encoding`, `decoding`, `behavior`, `events`, `ops`, `layout`, `regions`, `stats`, `simulation`, `annotation`, `animation`, `io`).

## Tasks

- Add `__getattr__(name)` to `src/neurospatial/__init__.py` (PEP 562) that, for a known submodule name, does `return importlib.import_module(f"neurospatial.{name}")` and caches it into the module globals; raise `AttributeError` for unknown names (so typos still fail correctly).
- Add `__dir__()` returning the eager exports + the lazy submodule names, so autocomplete and `dir(ns)` reveal every domain.
- Add the submodule names to `__all__` (or a dedicated `_LAZY_SUBMODULES` tuple referenced by both `__getattr__` and `__dir__`), mirroring `io/nwb`'s structure.
- Keep the existing eager imports (`Environment` etc.) — they stay eager; this only adds lazy access to the *submodules*.
- Verify no import cycle is introduced (submodules import lazily only when first accessed). CHANGELOG entry.

## Deliberately not in this phase

- Re-exporting individual submodule-only power functions at top level (`heading_from_velocity`, `spatial_autocorrelation_radial`, etc.) — that naming/discoverability work is phase 22 (Low).
- The `bin_spikes_in_time` / `read_units` top-level exports — those ship with their own phases (14, 15) as eager `__all__` entries.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_lazy_submodule_access` | `import neurospatial as ns; ns.encoding` returns the encoding module without an eager import at package load. |
| `test_dir_lists_submodules` | `dir(ns)` includes every lazy submodule name and the eager exports. |
| `test_unknown_attr_raises` | `ns.nonexistent` raises `AttributeError` (typos not silently swallowed). |
| `test_no_eager_submodule_import` | Importing `neurospatial` does not import, e.g., `neurospatial.animation` until accessed (check `sys.modules`). |

## Fixtures

None — pure import-behavior tests.

## Review

Dispatch `code-reviewer`. Confirm: pattern mirrors `io/nwb`; eager exports unchanged; `__dir__` complete; unknown attributes still raise; no import cycle; the no-eager-import test passes; CHANGELOG updated; no plan/phase references in code/test names.
