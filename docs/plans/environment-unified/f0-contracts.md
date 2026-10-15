# F0 implementation contracts

Recorded before runtime edits on 2026-10-15. Scientific baseline:
`/private/tmp/environment-stage-f-base.npz` (4,072 probes); environment/layout
suite: 3,285 passed. Deterministic performance command and raw three-repeat
results: `/private/tmp/stage_f_benchmark.py` and
`/private/tmp/stage-f-performance-base.json`. Repeat on the same interpreter,
thread limits and inputs after implementation; no speedup claim is implied.

## Native face inputs

| Family | Minimum numerical inputs | Deferred work |
| --- | --- | --- |
| Cartesian grid / masked grid / image / polygon | Centers, ordered pairs/navigation distances, cell measures, axis widths | Face-axis identification and products |
| Polar grid | Grid inputs plus radius, angular bin count and circularity | Radial arc factor and two-bin circular interface sum |
| Hexagons | Ordered pairs/distances and cell measures | Shared side length from distance / sqrt(3) |
| Triangle mesh | Centers, ordered pairs/distances, cell measures and triangle vertices in original order | Shared edge length and skew warning |
| Graph track | Ordered pairs/navigation distances, cell measures, centers and per-edge junction positions | Along-track flux distances, separately from navigation chord distances |

Extract pure formulas shared by native builders and owned lazy sources. These
inputs do not require retaining a builder, lookup callback or construction
parameters. Missing custom geometry retains its explicit refusal. Native sources
remain lazy; serialization can still materialize KnownFaces.

## API and persistence decisions

- Keep default construction's existing E1006 error; no new unfitted workflow.
  The current layout-bearing constructor remains transitional until F4; final
  construction goes through factories or owned publication, without layout=.
- F3 removes the polar public class/export and its create method. The existing
  from_polar_egocentric factory keeps its arguments and scientific meaning.
- Infer static coordinates from factories using a generic Environment; keep
  shared operations and Cartesian-only self restrictions. Runtime guards cover
  bound and unbound calls, including absence via hasattr on polar instances.
- Normalize polar circularity once: preserve the current one/two-bin saved
  option and three-or-more-bin graph seam rule. Do not reinterpret old files.
- Existing dictionary/file/NWB coordinate_kind and circular_angle fields can
  express the owned descriptor; retain their wire versions and normalize at the
  reader boundary. A new authority would require a new version.
- Current deepcopy/pickle must retain ownership and coordinate state. Old
  pickles naming the removed public polar class are not a promised interchange
  format; supported dictionary/file/NWB readers remain migration paths. Do not
  re-export a second class to make historical pickle names resolve.
- Every source inventory item migrates by capability: tensor geometry uses
  rectilinear/framed context; legacy logical indexing also supports native hex
  and track masks; plotting uses stored shapes and connectivity. Builder-local
  and unrelated UI layout accesses remain justified exceptions.

## Parent CI repairs

Python 3.10 reproduced point-region self-equality raising ambiguous NumPy truth.
An explicit value comparator and a regression passed 312 region tests on 3.10
(5 optional skips), 317 on 3.13, strict typing and independent review. Commit
`dea6eb32` is propagated through PRs #70–#76 without history rewriting.
The #73 threshold-summary test also reproduced the physical-walls refusal;
commit `7c9665ac` selects analysis boundaries explicitly, matching the already
corrected child branch. All 34 summary tests pass. Remote reruns are pending;
these local results do not establish every platform's CI outcome.
