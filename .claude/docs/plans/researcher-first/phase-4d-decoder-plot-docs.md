# Phase 4d — Decoder tutorial plots use matching spatial coordinates

**Requires:** Phase 4c merged. Added by the post-4c checkpoint on 2026-10-07;
the gate remains held until this correction merges and the checkpoint repeats.

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) ·
[overview](overview.md) ·
[checkpoint evidence](../../../../docs/reviews/RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07_POST_4C.md)

Read `executing.md` before implementation. Keep this correction bounded to
the demonstrated tutorial display contradiction and its executable guard.

## Verified defect

Public `DecodingResult.plot` documents spatial-bin indices on the y-axis and
plots the highest-probability bin as the MAP line. Tutorial 20's golden and
manual posterior cells instead overlay actual positions in centimeters and
replace the y-axis label with `Position (cm)`.

Current companion anchors, to relocate by symbols:

- `examples/20_bayesian_decoding.py`: `actual_track` at 248, `result.plot` at
  258, actual overlay at 265–272 and label at 274; manual posterior overlay
  at 440–454.
- The notebook code cells are `d61f1740` and `plot-posterior`; synchronized
  `docs/examples` copies carry the same code. The fixture uses 2 cm bins.

The public-only analytic probe constructs a perfect `(4,21)` one-hot posterior
in a 5 cm environment. Its actual position and `map_position` are 80 cm with
zero numerical decoding error. The plot's MAP y value is bin 16, while the
tutorial-style actual overlay is 80. `env.bin_at(actual)` correctly yields 16.
The checkpoint includes the exact script and before/after image. These numbers
are evidence, not targets to tune to.

## Task 4d.1 Correct the posterior overlays and guard their coordinate handoff

- Write a regression first against the unmodified advertised overlay. Use a
  known perfect posterior and non-unit spatial bins so a cm/bin mismatch cannot
  accidentally pass. Record the unmodified failure in the commit body.
- Correct both golden and manual posterior cells in the Python companion,
  notebook and synchronized docs copies. For the documented bin-index plot,
  convert actual spatial positions with public `env.bin_at` and retain truthful
  bin-axis wording. Keep physical `map_position` and actual coordinates for
  the separate cm-valued accuracy, time-series and scatter calculations.
- Handle the documented horizontal coordinate contract too: continuous decoder
  times use seconds; gapped rows use time-bin indices. Never interpolate across
  a tracking pause or overlay physical session times on a bin-index axis.
  Show one explicit public recipe for continuous and gapped results; do not
  invent a plotting facade or reach into private helpers.
- Put a small self-contained corrected overlay example under the existing
  executable-doc marker guard. Assert actual overlay/MAP alignment for the
  known posterior and truthful x-coordinate alignment for continuous and
  gapped timestamps. Check numeric plotted data rather than freezing prose,
  public export counts or an inventory.
- Search the full tutorial for repeated posterior-overlay/axis claims. Preserve
  valid physical-coordinate plots rather than converting every plot to bins.
- Use the notebook synchronization workflow and execute the real tutorial from
  a temporary directory. Retain a labeled diagnostic figure for review.

Do not change decoding algorithms, posterior arrays, `DecodingResult.plot`'s
documented coordinate contract or scientific defaults. Native 1D rate-result
plotting/accessor parity remains Phase 7 work. This phase does not revisit
Phase 4c's repaired duration or statistical interpretation.

## Validation and review

Run the full definition of done in `executing.md`, including executable docs,
source doctests and strict MkDocs. Add this task's own CHANGELOG bullet.
Verify the corrected overlay against a known posterior with non-unit bins and
both continuous and gapped times; preserve numerical decoding outputs.

Dispatch an independent reviewer before the PR. Check both advertised cells,
all synchronized copies, physical-versus-bin axis meaning, x-axis behavior
across gaps, executable guard and unchanged scientific calculations/defaults.

## After merge

Record Phase 4d merged in `PLAN.md`, then repeat the UX probes and five
documentation-only design journeys before Phase 5a. This correction PR alone
does not pass the checkpoint.
