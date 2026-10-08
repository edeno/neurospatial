# neurospatial — UX review after Phase 4d

## Overall assessment

**NEEDS_POLISH; researcher-workflow checkpoint PASS.** All five public-doc
journeys execute without source recovery. The graph route is smooth and the
other four are workable, improving on the original CONFUSING/all-painful
qualitative baseline. Phase 4d closes the tutorial's cm-versus-bin display
contradiction. See [measurements and reproduction](RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07_POST_4D.md).

Reviewed `d10607a5` on 2026-10-07 with two independently authorized reviewers
and lead repeatability controls. The lead has prior implementation context;
fresh reviewer discovery supplies first-user judgments. No quantitative
call/LOC reduction or interactive accessibility certification is claimed.

| Dimension | Rating |
| --- | --- |
| Interface usability | NEEDS_POLISH |
| Error messages | USER_READY for exercised recovery paths |
| Output formatting | NEEDS_POLISH |
| Workflow friction | NEEDS_POLISH |
| Accessibility | NEEDS_POLISH; static/artifact inspection only |

## First-run experience

**First field: workable.** A fitted factory and four arrays produce a readable
viridis field, with Hz labels and explicit cm coordinates. The repeated NWB
control gives 78 bins, 59.99 s occupancy, 5.321 Hz peak and 1.176 bits/spike.
Real component reads remain usable after file close, and population tables
preserve nondefault unit IDs.

**Mistake recovery: smooth for the exercised mistakes.** Bare construction,
shape mismatches, no-bin/grid diagnostics, dropped-spike clock warnings and
missing batch unit indices explain the next call. Large-grid allocation was
prevented by promoting its warning to an exception. Infeasible lap duration
provides a concrete sample budget and duration/count/pause fixes. No unfitted
object was fabricated through private mutation.

**Result interpretation: workable.** Reprs and tables expose useful metrics;
the repaired decoder cells and published guide now put actual and MAP lines
in the same bin coordinates and preserve physical error calculations.
Continuous/gapped clocks both execute correctly. Summary aggregation names,
single/batch table differences, empty attrs and absent singular xarray still
require Phase 7. Synthetic classifications and in-sample errors are interpreted
as candidate/descriptive outputs rather than biological validation.

## Critical issues

No unresolved checkpoint blocker was found in the exercised journeys.

- [x] Requested simulation duration, timestamps and metadata agree.
- [x] Graph geometry is distinguished from explicit direction labels.
- [x] Activation/reference cutoffs and EV effects are distinguished from significance.
- [x] Tutorial posterior overlays and labels share the correct spatial/time coordinates.

## Confusion points and improvements

- Phase 6a should index the assembly/count-statistics route and common-unit
  selection. Public help currently supplies those steps.
- Phase 6b should join full-window event cohorts and explicit model handoffs;
  point-position validity and full-window PSTH validity serve different purposes.
- Phase 6c should join NWB components, acquisition windows, epochs, units and
  identity in a holder recipe. The current route is executable but spans pages.
- Phase 7 should support native 1D result plotting and align single/batch
  summaries, xarray, column order, units/threshold attrs and safer exports.
- Add continuous posterior heatmap column centering to Phase 7: the four-row
  100 ms probe has a maximum 37.5 ms image-pixel offset, although both plotted
  lines use correct timestamps. Gapped pixel centers agree exactly. This is
  nonblocking presentation work; the checkpoint makes no sub-bin timing claim.

## Accessibility observations

The repeated static default uses viridis and labeled axes/colorbars. Actual
and MAP overlays also differ by dash pattern. Real HTML rasters include
cyan markers and explicit session-time labels across gaps. Sparse frames
trigger the documented minimum-fps cap rather than preserving exact 1× timing.
Numeric HTML colorbars, colorblind overlay defaults, keyboard/screen-reader
interaction and browser behavior remain unverified or later polish. Artifact
inspection is not an interactive accessibility audit.

## Keep

Keep array-first calls, eager NWB component reads, identity-preserving results,
explicit recording windows and documented gaps. Keep the joined graph recipe,
descriptive statistical language and actual-bin/MAP-time posterior recipe.
They make all five current workflows usable without implementation access.
