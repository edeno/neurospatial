# Stage F validation record

## Parent repairs and F0

See [f0-contracts.md](f0-contracts.md) for reproductions, fixed parent commits,
scientific baseline, native face inputs and migration decisions. The full
matrix reruns remain pending; no platform-wide pass is implied by local results.

## F1 — Native face data and coordinate publication

- Fresh behavior capture: 4,072 probes; comparison to F0 is identical.
- Environment/layout/NWB environment tests: 3,477 passed.
- New owned coordinate/native recipe tests: 29 passed (all built-in families;
  polar 1/2/8 angular bins, open/circular, deepcopy/pickle/dictionary).
- Final extracted polar/shared formula gate: 1,150 passed.
- Source mypy: 246 files pass. Ruff/format and whitespace checked.
- Native faces retain closed numerical recipes, not builders or opaque callable
  providers. Formulas are shared with the native layout construction tools.
  Construction stays lazy; skew warnings remain attached to first numerical use.
- Polar circularity normalizes on publication; current shared graph/coordinate
  operations retain their previous values. The stored layout facade and public
  sibling classes are transitional until their later migration phases.
- Independent review: ownership/coordinate checks passed; final review recorded
  at the phase commit. No full scientific CPU/docs suite is claimed at this gate.
