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
- Analysis results (environments, rate maps) go in the top-level `/analysis`
  group (`nwbfile.add_analysis`), never `scratch` ("not intended for sharing")
  or a processing module (intermediate data). Readers refuse records that earlier
  versions left in either place, with recreation instructions.
- Environment records write schema `4.0`: Arena, explicit walls, exact binning,
  lookup, faces, components, clip mapping and coordinate metadata are saved.
  The connectivity graph is not stored; loading rebuilds it from the binning
  and never runs a layout builder. Only schema `4.0` is read; every earlier
  schema is refused, never converted or given a guessed Arena. One-row ragged,
  gzip-compressed columns hold arrays, named by canonical key with `/` -> `.`;
  one description JSON attribute maps keys to columns, shapes and dtypes.
  Records are identified by `"format": "neurospatial.Environment"` (rates:
  `"neurospatial.SpatialRates"`) in their description JSON, never by column
  names (a `unit_table` column can be called anything). Check scientific outputs against the original after read-back.
- Spatial rates write schema `3.0` and name their environment record
  (`environment_name`); records naming one environment share it. Reuse needs
  an exact match of the stored payload (not just equal bins). Rates store
  `environment_sha256` and the reader refuses them if the record changed, so
  no write route (`write_environment(overwrite=True)`, `to_nwb`, raw pynwb
  edits) can relabel their bins.
