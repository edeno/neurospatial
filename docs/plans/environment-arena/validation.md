# Stage A validation record

Stage A is complete on `feature/environment-arena`, stacked on
`feature/environment-representation` at `944c818d`. Independent reviews are
approved and the checks below passed. Stages B–F remain separate work.

## Scientific workflows

- Seeded samples, calibrated image masks and explicit grid masks remain direct
  factory inputs. Their inferred domains do not assert measured physical walls.
- A seeded clipped-grid workflow checks occupancy, independently counted spikes,
  firing rates, normalized decoded posteriors and restored position lookup.
  Default start-based encoding requires both interval endpoints to be observed
  inside; standalone start occupancy only gates the starting sample.
  The workflow explicitly uses the same valid intervals when comparing them.
- A local recorded-data check uses the first 600 tracking rows of Arthur's
  2022-03-24 session: 19.7064626 seconds, 89 active bins, 0.22 cm/pixel
  calibration and requested 2 cm bins. Sample, explicit-grid and calibrated
  image-mask assignments agree; occupancy equals the observed duration. Clipped
  lookup, parent mapping and linear occupancy survive dictionary restoration.
  This ignored local fixture is unavailable to CI; it is not a bundled test.
- The same slice also checks the opt-in linear encoder using deterministic
  synthetic events. Sample, grid-mask and clipped-sample inputs give identical
  single/population rates and match standalone linear occupancy: 19.3961426
  retained seconds, 21 observed runs and 89 bins. The tracking is recorded;
  these event trains are simulated, not a validation on recorded neural spikes.

## Compatibility capture

`scripts/environment_baseline.py` recorded 4,046 probes at the branch base.
The candidate records 4,051 probes, with 173 changed, added or removed keys.
Every behavioral family below appears in the original, dict, file, NWB and
copied variants:

| Changes | Explanation |
| --- | --- |
| 29 API keys | Arena and clip accessors, optional factory arguments, rebin/occupancy documentation and version-3 transport documentation. |
| 16 hexagon and 16 mesh occupancy keys | Linear allocation now uses exact stored planar cells; previous refusal records are replaced by numerical results. |
| 8 mesh nearest-interpolation keys | Points outside supported geometry return missing values. |
| 72 polygon sequence, occupancy and transition keys | Known-domain exterior is excluded. Transition estimation retains its existing refusal of sequences containing outside-bin `-1`. |
| 24 grid/grid-mask linear occupancy keys | Shared polygon allocation changes summation rounding by at most 4.45e-16 seconds; totals are unchanged. |
| 8 image-mask linear occupancy keys | Outside-to-inside entry is retained; this probe gains 0.0976465714 seconds of supported time. |

The key counts include both sides of a result replacing an exception. This is
an intentional behavior change; no claim of unchanged polygon lookup is made.
The API snapshot additionally catalogs the public Arena namespace and its six
classes, plus `time_allocation` on both exported spatial-rate functions and
their aliases, with no unrelated public exports.

To reproduce the behavior comparison, record at each checkout and compare:

```bash
uv run --no-sync python scripts/environment_baseline.py record /tmp/base.npz
uv run --no-sync python scripts/environment_baseline.py record /tmp/candidate.npz
uv run --no-sync python scripts/environment_baseline.py compare /tmp/base.npz /tmp/candidate.npz
```

## Performance method

`scripts/benchmark_environment_arena.py` uses seed 20261014, 20,000 source
positions and one million fixed queries per environment. It reports medians
of five construction and lookup runs after warm-up, plus separate peak Python
allocations measured with `tracemalloc`. The base and candidate run sequentially
with the same isolated interpreter and each checkout's `src` on `PYTHONPATH`.
These are local measurements, not timing assertions or a speed improvement.

| Input | Construction, base → candidate (ms) | Million-point lookup, base → candidate (ms) | Lookup Python peak, base → candidate (MiB) |
| --- | --- | --- | --- |
| Position samples | 344 → 304 | 351 → 851 | 85.3 → 82.9 |
| Grid mask | 345 → 321 | 352 → 823 | 84.4 → 79.5 |
| Polygon | 266 → 234 | 333 → 1,497 | 82.3 → 93.5 |

Other validation processes were active on this machine; the construction
differences are not evidence of a speedup. Arena-gated lookup adds work. The
polygon candidate assigns 497,945 queries versus the base's 505,537 because
the exterior portions of boundary cells are now excluded. Samples and mask
assignments are unchanged (671,644 and 639,651 respectively).

## Verification

Focused regression suites cover [arena data](../../../tests/test_arena.py),
[polygon geometry](../../../tests/layout/test_polygon.py),
[saved geometry validation](../../../tests/layout/test_polygon_validation.py),
[arena integration](../../../tests/environment/test_arena_integration.py),
[rebinning](../../../tests/environment/test_arena_rebin.py),
[wall frames](../../../tests/environment/test_arena_wall_frames.py),
[linear occupancy](../../../tests/environment/test_polygon_occupancy.py),
[linear rates](../../../tests/encoding/test_linear_spatial_rates.py) and
[persistence](../../../tests/io_tests/test_arena_persistence.py).

All Python checks use the existing isolated environment through `uv run
--no-sync`, the Agg Matplotlib backend and writable temporary cache directories.
Native GUI/GPU rendering, slow benchmarks and wall-clock assertions are outside
the CPU validation run.

An existing test,
`tests/test_environment_validation.py::TestValidateEnvironment::test_validates_graph_structure`,
attempts to mutate the published read-only graph and fails before it reaches
its validation assertion. The same failure was reproduced with source imports
from `944c818d`; the test remains unchanged and is explicitly deselected in the
broad candidate environment run.

Broad checks, with focused rechecks after independent review repairs:

| Check | Result |
| --- | --- |
| Environment, layout, ops, IO, decoder, Arena and scientist workflows, after final wall repairs | 5,220 passed, 9 skipped; existing graph-mutation test explicitly deselected |
| Encoding, behavior, simulation and segmentation consumers | 4,122 passed, 7 skipped; later boundary regressions also pass in focused checks |
| NWB, serial | 565 passed, 9 deselected, including an actual linear-rate file/decoder round trip |
| Source doctests, serial | 597 passed, 132 skipped; 47 passed, 9 skipped on changed public modules after documentation updates |
| API/argument conventions and selected documentation contracts | 124 passed |
| Executable guides, snapshot and conventions | 25 passed |
| Strict MkDocs build | Passed |
| Full source and four typing contracts | Mypy passed, 232 files |
| Ruff and formatting across changed Python files | Passed, 52 files |
| Independent final wall-frame gate | 214 passed, with numerical RuntimeWarnings treated as errors |
| Independent final wall-junction/linear-encoding gate | 135 passed, including independent route and wall reversal probes |

Independent review additionally reproduced and required repairs for inherited
boundary ownership, polygon ring-order equivalence, strict center selection,
wall endpoint contacts, transformed wall frames, mismatched mask allocation,
nonmaterializable known domains, polygon precision, clip ancestry and a saved
zero-face value bypassing certification. Final wall regressions cover source
offsets hidden by large-origin rounding, artificial inverse-projection gaps,
genuine retained insets, split T/X junctions, bent walls, and a collinear overlap
followed by a separate crossing. Each confirmed defect has a focused regression;
the final repairs received independent approval.

New wall declarations after clipping/transforms refuse when clearance or face
alignment cannot be resolved at stored precision. Trajectory crossing checks
likewise use the actual stored coordinates, including their rounding. Declare
walls before derivation and recenter/rescale coordinates when necessary. These
are explicit representability limits, not a general-mesh solver or an inference
of unrounded geometry.
