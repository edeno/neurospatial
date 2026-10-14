The `environment_v1` JSON/NPZ pair was written by commit
`f43649416c29df6b60444f7ad4697ce296cfa8f2`, before the v2 persistence rewrite.
It is a 2×2 active grid with unit edges, `cm` units, the `world` frame and one
point region. It tests refusal of the old format, rather than synthesizing old
metadata through the new writer.

Capture command (the temporary generator is retained in the validation artifact):

```sh
UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache uv run --no-sync python /private/tmp/neurospatial_capture_legacy_environment_formats.py
```
