# Environment representation validation

Implementation branch: `feature/environment-representation`.
Starting commit: `2672c6432ae1e12d69d8042a10a3de98a210410b`.

Validation uses the existing isolated Python 3.13 environment through
`UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache uv run --no-sync`.
Non-GUI pytest commands use `-p no:napari`: the installed napari plugin tries
to write its theme cache outside the sandbox during unrelated tests.
Matplotlib uses `MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache`.

## Binning snapshot

The original behavior capture contains 4,046 probes, including 678 intentional
exception outcomes. Construction medians on the starting and exact reviewed trees, seven runs:

| Environment | Bins | Before (s) | After (s) | Change |
| --- | ---: | ---: | ---: | ---: |
| Grid | 90,000 | 5.515561 | 5.845155 | +6.0% |
| Hexagonal | 20,064 | 0.543166 | 0.608662 | +12.1% |
| Linear track | 2,000 | 0.021790 | 0.023489 | +7.8% |

The final hexagonal construction median exceeds the 10% target (+12.1%);
the first seven-run capture measured +8.3%. The new work snapshots owned
arrays and validates ordered neighbor pairs. The exact final measurements above
are retained rather than hiding this exception.

The grid and hexagonal timings use the same seeded 200,000 uniform points
in a 150 cm square. The grid uses explicit bounds, `bin_size=0.5` and
`infer_active_bins=False`. Hexagonal width is 1.15 cm. The track runs from
(0, 0) to (1,000, 0) with `bin_size=0.5`. Garbage collection and imports
are outside timed regions. Inputs, raw timing arrays, capture metadata and
reusable scripts are retained in
`/private/tmp/environment-representation-2672c643-20261014T174335Z-ifmh8_0t`.

All 4,046 behavior probes remain identical. The 162-test focused slice, all
597 executable doctests (132 skipped), Ruff, the public API snapshot and mypy
for the full source plus four typing-contract files pass. The first full-suite
run passed 11,283 tests; blocked napari cache writes caused GUI failures, and
its four decoder setup failures were repaired and rerun successfully. A final
environment/layout/docs/NWB/API slice passed all 2,481 tests. The graph-validation
test that mutates protected geometry also fails on the starting commit and is
left unchanged. GUI retry with cache access passed its first 21 tests before
a native VisPy/OpenGL segmentation fault, so GUI validation remains limited.
An initial video shape failure passes alone on both starting and current code;
its full-suite interaction remains unexplained.

Implementation also migrates pickles whose published geometry predates the
binning field, rebuilding the snapshot from the retained layout and invalidating
old geometry caches.

## Approved contract clarifications

- Hexagonal lookup must reject incorrect coordinate widths. The implementation
  enforces the locator width contract, including empty and non-finite inputs.
- Loading checks compare the stored locator with its grid, shape and track
  components, in addition to checking center lookup and graph consistency.
- Reflections keep parent bin identity through inverse lookup; they never reverse
  rectilinear edges.
- Track subsets retain the track graph and absolute linear coordinates while
  restricting their active linear mask to the selected bins.
