# Responsibility refactor validation

Candidate source: `1099bb6b`, stacked above Stage F `0206fe26`. Subsequent documentation
edits do not change executable code. Findings and owner PRs are
listed in [the tracker](TRACKER.md). Final parent `236a2ef9` adds only the
independently verified rotation-property CI setting; library code is unchanged.

## Setup and checks

The isolated CPU environment uses Python 3.13.5 and `uv.lock`, with `dev`, `nwb`
and `xarray` extras plus CI's `ipywidgets` and `qtpy` dependencies. Its interpreter
is `/private/tmp/neurospatial-review-cpu-env/bin/python`. Each worktree is selected
explicitly through `PYTHONPATH`; Matplotlib uses Agg. BLAS/OpenMP thread counts
are fixed at one.

```bash
PYTHONPATH=src MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests -n4 -q -m 'not slow and not napari and not wallclock'
```

The initial full run passed 13,975 tests and found three stale private-helper
calls in an affine animation test. Production callers already supplied the new
binning argument. Updating that test passed all 70 affine/video checks. The
clean full rerun passed: **13,978 tests, 335 skips**, in 315.87 seconds.

| Gate | Result |
| --- | --- |
| Full CPU suite, clean rerun | 13,978 passed, 335 skipped |
| Source and all static contracts | 289 files passed |
| Repository Ruff, formatting and whitespace | Passed; 813 Python files formatted |
| Package dependency contracts | Two kept, zero broken, no ignored edges |
| Strict MkDocs build | Passed in 211 seconds |
| Built documentation fragments | 8,131 checked, zero invalid |
| CPU source doctests | 581 passed, 126 skipped |
| Optional JAX CPU paths | 318 passed |
| New migration example | Passed through the executable Markdown runner |

CPU doctests exclude three optional GUI/JAX implementation modules and two
examples that construct napari objects. The separate JAX gate uses the existing
isolated environment with JAX installed and disables the napari pytest plugin.
The link check excludes generated API/notebook pages as link sources while
checking links into those pages. Local GUI/GPU execution is not claimed.

## Behavior comparison

The mechanical geometry move preserved 4,004 probes exactly. Later captures
isolate each numerical or API change. The final comparison against the repaired
Stage F parent contains 4,004 parent records and 4,011 candidate records:

- 3,943 common records are exact.
- Twelve API records differ: seven added operations/properties and five
  descriptions or cached-property representations changed intentionally.
- Seventy-two rebin records cover 24 newly supported uniform rectangular
  selections (each replaces an old error record with a result), plus 24 updated
  refusal messages for unsupported geometry.
- Eight mesh Gaussian-rate records differ by at most `3.552713678800501e-15`
  absolute and `5.384886777688641e-16` relative. Direct Gaussian matrices remain
  bit-identical; downstream reductions differ after immutable buffer ownership.
  No other numeric, refusal or text differences remain.

The capture command is `python scripts/environment_baseline.py record FILE`.
Final artifacts under `/private/tmp/` are
`environment-review-{parent-final,structural-final}.npz` and
`environment-review-structural-final-comparison.json`.

The pinned J16 recording and parameters are documented in
[Stage F validation](../environment-unified/validation.md#real-recording).
Both 1,800-sample and 60,000-sample slices preserve all 14 arrays exactly,
including population rates, occupancy, lookup and restored geometry. The larger
slice spans 119.998 seconds with 7,677 input spikes and eight units. This verifies
parity on these slices, rather than biological validity of an estimator.

## Ownership and cost

Independent reviews cover geometry imports and legacy pickle aliases,
publication, subset/rebin, wall ownership, cache cleanup and public operations.
Sparse buffer/header mutation, copy/pickle, Gaussian cache bounds and warning
attribution, and reuse of MRF eigensolves have focused regressions.

Warm public exports share immutable buffers through independent headers. Twenty
diffusion exports allocated a peak 2,800 bytes for 157,604 bytes of backing data.
Measured fixed overhead is approximately 2 microseconds for a Gaussian export
and 7 microseconds for a diffusion export. No repeated matrix or edge-buffer
copy occurs on a cache hit.

In one alternating 40-pair, 1,024-bin single-neuron Gaussian-rate benchmark,
the median increased from 223.4 to 282.9 microseconds (26.6%, 59.5 microseconds).
Buffer alignment also differed, but its causal contribution is unproven. The
factor-rebin navigation check adds approximately 15–24 milliseconds on the
recorded 30-by-30 cases. These are measured costs, not speed improvements.
The API benchmark script is `/private/tmp/review_analysis_api_paired.py`; raw
measurements are `neurospatial-analysis-api-{paired,alternating}.json`.

## Remote CI

The updated parent stack is running or queued. A slow rotation property hit
Hypothesis's data-generation timing health check on the current #73 head; the
replayed CI delay reproduced before a narrow test-only repair and passed after
it. The mathematical assertions and execution deadline remain unchanged. This
repair is published in [PR #82](https://github.com/edeno/neurospatial/pull/82) and propagated through the stack. Baseline PRs #78, #79 and #81 are green; #80 has a remaining
job. Earlier green heads and superseded cancelled runs are not counted as proof
for updated code. The new structural PR requires its own CI run.
