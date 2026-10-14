The schema 1.1 record was written by commit
`f43649416c29df6b60444f7ad4697ce296cfa8f2`, before the 2.0 persistence rewrite,
using the same small grid as `tests/io_tests/data/environment_v1`.
Its NWB session start is fixed at `2020-01-01T00:00:00Z`.

The 1.0 fixture is a copy of the actual 1.1 file with `schema_version` changed
to `"1.0"` in each JSON metadata-column row. The old format stores this field
in the table metadata column, not as a standalone HDF5 attribute. Both fixtures
were successfully loaded and their centers compared with the original before
changing persistence code.

Capture command (the temporary generator is retained in the validation artifact):

```sh
UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache uv run --no-sync python /private/tmp/neurospatial_capture_legacy_environment_formats.py
```
