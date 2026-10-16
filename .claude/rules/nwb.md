---
paths:
  - "src/neurospatial/io/nwb/**"
  - "tests/nwb/**"
  - "docs/user-guide/interoperability.md"
---

# NWB code

- Install with `uv sync --extra dev --extra nwb`. CI runs NWB tests serially:
  `uv run pytest tests/nwb -n 0`.
- Import `ndx_events` and `ndx_pose` before opening an `NWBHDF5IO`. Otherwise
  pynwb builds generic classes for those types, and, for example, `read_events`
  raises "'rewards' is not an EventsTable (got EventsTable)".
- Environment records write schema `3.0`: Arena, explicit walls, exact binning,
  lookup, faces, components, clip mapping, graph attributes and coordinate
  metadata are saved. Loading publishes owned binning data directly and never runs a layout
  builder. Schema `2.0` remains readable with an explicit `UnknownArena`;
  never infer a physical footprint from old outlines, parameters or masks.
  Schemas `1.0` and `1.1` are refused with recreation instructions.
  Standard one-row ragged columns hold
  arrays; one description JSON attribute maps canonical keys to columns,
  shapes and dtypes. Check scientific outputs against the original after read-back.
