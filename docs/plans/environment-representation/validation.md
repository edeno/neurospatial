# Environment representation validation

Implementation branch: `feature/environment-representation`.
Starting commit: `2672c6432ae1e12d69d8042a10a3de98a210410b`.

Validation uses the existing isolated Python 3.13 environment through
`UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache uv run --no-sync`.
Non-GUI pytest commands use `-p no:napari`: the installed napari plugin tries
to write its theme cache outside the sandbox during unrelated tests.
Matplotlib uses `MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache` and
`MPLBACKEND=Agg` so documentation examples cannot block on an interactive window.

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
are retained rather than hiding this exception. An isolated seven-run median
for `binning_from_layout` on that existing hexagonal layout is 0.041723 s,
about 64% of the observed 0.065496 s construction increase. Profiling attributes
most added work to ordered edge enumeration and array materialization; validation
itself is smaller. Residual allocator/run variation was not separated.

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

## Data-backed lookup

Before moving lookup implementations, the parity generator recorded 277 arrays
(576,998 bytes) at `bef7abea` into `tests/layout/data/locator_parity.npz`.
It covers eleven deterministic environments, including a triangular mesh with a
hole and concave notch; boundary, non-finite, empty and wrong-width queries;
and 2,000-point branching/gapped track linearization probes. Generator command:
`uv run --no-sync python /private/tmp/neurospatial_capture_locator_parity.py`
with the cache environment variables documented above. The temporary script is
retained outside the repository.

Seven-run lookup medians before implementation, one million seeded queries:
grid 0.147921 s; track 0.406283 s. Inputs and hashes are retained with the
construction artifacts.

The lookup phase preserves all 4,046 behavior probes. Recorded parity also
verifies the approved coordinate-width exceptions explicitly. The broad
non-GUI suite passed 10,059 tests (25 skipped); the GUI directories and the
confirmed starting-commit graph-mutation failure were excluded. All 597
doctests passed (132 skipped), the 337-test focused slice passed with numerical
RuntimeWarnings promoted to errors, and full-source/typing-contract mypy plus
Ruff passed. Independent review also verified actual starting-commit pickles
for all eleven environments through loading, deepcopy and re-pickling.

The track component protects stored graph attributes as well as topology.
Per-segment spacing is stored immutably and converted back to the external
linearization library's required list format only at the call boundary.

Lookup medians on the exact saved queries, seven runs, one million points:

| Environment | Before (s) | After (s) | Observed change |
| --- | ---: | ---: | ---: |
| Grid, 90,000 bins | 0.147921 | 0.140566 | -5.0% |
| Track, 2,000 bins | 0.406283 | 0.401981 | -1.1% |

Both satisfy the 10% limit. Query and output hashes are identical; these timing
observations do not establish a general performance improvement.

## Shapes and drawing

At `654ac1c5`, the pre-change artist capture recorded 27,310 numeric/string
arrays for eleven environments into `tests/environment/data/plot_artists.npz`.
It includes layout plots, field plots with NaNs and colorbars, and the track's
linear plot. The helper records artists on every axes after drawing. Its 23
72-dpi PNGs are retained in `/private/tmp/neurospatial-plot-parity-20261014/before`.
Capture command: `uv run --no-sync python
/private/tmp/neurospatial_capture_plot_parity.py before`, using the cache
variables above.

The old early 3D field-plot refusal is retained. Legacy NWB tracks that lack
linearization data retain their previous plotting path until the loading phase
restores that data; they do not silently switch to a spatial scatter plot.

All 4,046 baseline probes remain identical. The 27,310 default artist arrays
and 23 PNG pairs match exactly; eight custom-options cases add 9,824 identical
artist arrays and eight identical PNG pairs. The committed fixture is a
lossless packed NPZ (440,236 bytes), preserving every array's dtype, shape and
raw bytes without thousands of separate ZIP headers.

The broad non-GUI run passed 10,122 tests with six obsolete tag/absent-component
assertions, which were updated to test actual component behavior. The final
190-test visualization/publication/components/API slice passes, including the
slow artist comparisons. All 597 doctests pass (132 skipped); full mypy and
Ruff pass. Independent review found and verified the MultiPolygon outline fix;
no remaining findings. Intermediate unreleased private Binning pickle states
are not a supported storage format; starting-commit public environment/result
pickles remain covered by the tested migration.

## Saved binning records

Files and dictionaries use `Environment-v2`; NWB records use schema `2.0`.
All three transports decode the same component payload without a layout builder.
The user approved a standard one-row ragged DynamicTable for NWB arrays, with
one JSON description holding shapes, dtypes and safe column-name mappings.
Real v1 file and NWB 1.0/1.1 fixtures were captured before changing the writer;
the new readers refuse them with recreation instructions.

The independent review found two integrity gaps and confirmed their repairs:
track edge IDs must match graph edge order, and grid sizes must match physical
cell measures, including the radial factor for polar cells. Regression tests
first failed, then passed. Measure checks retain the native grid builder's
documented uniform-spacing rounding rule and accept genuine variable widths;
exact comparisons still protect duplicated stored arrays.

The reviewed persistence slice passes 544 tests with numerical RuntimeWarnings
treated as errors. The broader owned slice previously passed 946 tests; the
updated cross-module slice passes after replacing obsolete rebuild/capability-loss
expectations. The broad CPU run passed 10,069 tests before those updates;
its 14 failures were those obsolete expectations and two missing NWB docstring
examples, now repaired (all four NWB flagship examples pass). The serial full
NWB run passes 564 tests, with nine deselected. Full-source and typing-contract
mypy passes 227 files; Ruff and formatting pass. The prior 595 doctests pass,
and the two restored executable NWB examples are checked separately.

Baseline comparison against `baseline-after-plots.npz` changes 232 NPZ entries,
representing 142 logical probes. Exactly 130 changed NWB scientific/refusal
probes now equal their corresponding factory-built original probes exactly;
the other 12 are persistence docstrings, with signatures and member kinds
unchanged. No native, dictionary or file scientific output changed. The complete
inventory follows. Each operation appears for both `<fixture>|nwb/<operation>`
and `<fixture>|nwb|copy/<operation>`, matching the corresponding native output
or refusal exactly (including `#err` entries).

| Fixture | Changed operations |
| --- | --- |
| `grid4`, `grid8` | `boundary_bins`, `rebin` |
| `hex` | `bin_at`, `contains`, `diffuse`, `divgrad`, `grad`, `interpolate_linear`, `kernel_average`, `kernel_density`, `kernel_transition`, `ncomp`, `occupancy_linear`, `occupancy_smoothed`, `rebin`, `smooth`, `transitions_diffusion`, `wavelet` |
| `mask` | `boundary_bins` |
| `mesh` | `bin_at`, `bin_sequence`, `bin_sequence_all`, `bin_sequence_with_runs`, `contains`, `diffuse`, `divgrad`, `grad`, `interpolate_linear`, `kernel_average`, `kernel_density`, `kernel_transition`, `ncomp`, `occupancy_linear`, `occupancy_smoothed`, `occupancy_speed`, `occupancy_start`, `rebin`, `smooth`, `transitions_diffusion`, `transitions_trajectory`, `wavelet` |
| `pixels`, `polygon` | `boundary_bins`, `rebin` |
| `polar`, `polar_open` | `boundary_bins` |
| `track` | `diffuse`, `divgrad`, `grad`, `interpolate_linear`, `kernel_average`, `kernel_density`, `kernel_transition`, `linear_to_nd`, `ncomp`, `occupancy_linear`, `occupancy_smoothed`, `rebin`, `smooth`, `to_linear`, `transitions_diffusion`, `wavelet` |

The 12 API probes are
`api/{Environment,EgocentricPolarEnvironment}.{from_dict,from_file,from_nwb,to_dict,to_file,to_nwb}`.
The artifact directory retains `baseline-save-diff-inventory.json`, with every
logical probe and its exact native comparison key, and the unabridged
`baseline-save-differences.txt`. No unexpected differences remain.
