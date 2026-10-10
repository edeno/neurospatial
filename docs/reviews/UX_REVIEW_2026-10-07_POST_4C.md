# neurospatial — UX review after Phase 4c

## Overall assessment

**NEEDS_POLISH; checkpoint HOLD for a scientific display correction.** The
original qualitative baseline was CONFUSING. The repeated five journeys now
complete from public docs/help with zero implementation reads, and the graph
route is smooth. Merged Phase 4c closes all three earlier blockers. Tutorial
20 still plots physical centimeters over posterior bin indices, however,
making even perfect decoding appear wrong. See
[measurements and reproduction](RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07_POST_4C.md).

Reviewed `cf3fb286` on 2026-10-07 with two independently authorized reviewers
and lead first-use controls. The lead has prior implementation context;
independent discovery supplies the fresh-user judgments.

| Dimension | Rating |
| --- | --- |
| Interface usability | NEEDS_POLISH |
| Error messages | USER_READY |
| Output formatting | NEEDS_POLISH |
| Workflow friction | NEEDS_POLISH |
| Accessibility | NEEDS_POLISH |

## First-run experience

**First field: workable.** The four-array call, fitted factory and default
plot still yield a readable viridis field. The repeated NWB fixture gives
78 bins, 59.99 s occupancy, 5.321 Hz peak and 1.176 bits/spike.

**Mistake recovery: smooth for the exercised mistakes.** Factory E1006, no-bin
diagnostics, dimensionality fixes, grid/time warnings and required animation
timestamps guide the next call. An unfitted object was not manufactured through
private mutation; only public factories and the bare-constructor rejection
were exercised. Infeasible lap duration now provides a concrete sample budget
and corrected duration/count/pause choices.

**Result interpretation: workable, with a new high-priority visual error.**
Reprs and population tables expose meaningful metrics and preserve IDs. The
posterior plot correctly documents bin coordinates, but the tutorial overlays
centimeters and overwrites the label. A perfect 80 cm decode plots its MAP at
bin 16 and the tutorial's actual line at 80. Converting actual positions with
`env.bin_at` aligns the lines; Phase 4d must publish that truthful path.

## Critical issues

- [ ] **Decoder tutorial mixes spatial coordinate systems.** Correct golden
  and manual posterior cells, notebook/docs mirrors and their axis wording;
  check a known perfect posterior with non-unit bins and gapped timestamps.
  This is Phase 4d, before Phase 5a, rather than cosmetic output polish.
- [x] **Lap duration and metadata agree.** Independent 1/2 s recordings retain
  both traversals and return 500/1000 samples ending 0.998/1.998 s.
- [x] **Geometry/direction and activity/significance explanations are corrected.**
  The joined direction recipe recovers planted peaks within one bin. Sparse
  standardized activity and EV/REV are interpreted descriptively with controls.

## Confusion points and improvements

- Phase 6a should expose the assembly/count-statistics branch in navigation.
- Phase 6b should publish one retained-event cohort and explicit endpoint/model
  handoff semantics; current helpers serve different documented purposes.
- Phase 6c should join eager NWB readers, units, analysis/acquisition windows
  and identity without the guide's Session-centered detour.
- Phase 7 retains native 1D rate-result plotting and output parity: singular
  summaries lack batch metrics/xarray, default pandas output hides columns,
  and attrs omit units/thresholds. Graph and ordinary 1D line plots are usable.
- Synthetic OVC and place controls both pass current candidate screens; retain
  Phase 5's bias/frame work and avoid interpreting thresholds as validated labels.

## Accessibility and good patterns

Static maps use viridis, labeled coordinates and numeric Hz scales; graph
comparisons include text direction labels. The fresh NWB HTML export has
explicit session-time labels across excluded intervals and a position overlay.
These artifacts were inspected; no browser, GUI or assistive-technology
certification was performed. Numeric HTML colorbars remain optional planned
polish rather than an asserted capability.

Keep four-array composition, automatic gaps, explicit half-open windows,
shared population occupancy, eager arrays after file close, helpful errors,
stable unit identities and the new complete graph/trial recipe.
