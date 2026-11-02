# Phase 5 — Plots show physical axes, keep NaN distinct, and respect the caller's figure

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/neurospatial/encoding/population.py:521-640](../../../../src/neurospatial/encoding/population.py) — `plot_population_coverage`. Field count is plotted without the `is_observed` mask at `:601` and `:634`.
- [src/neurospatial/decoding/_result.py:900-990](../../../../src/neurospatial/decoding/_result.py) — `DecodingResult.plot`, with the y-label "Spatial bin" at `:943`.
- [src/neurospatial/encoding/spatial.py:4908](../../../../src/neurospatial/encoding/spatial.py) — `DirectionalPlaceFields.plot`, with "Bin index" on the x-axis.
- [src/neurospatial/encoding/directional.py:3221](../../../../src/neurospatial/encoding/directional.py) — `set_theta_zero_location` on a possibly Cartesian axis.
- [src/neurospatial/events/_core.py:1541](../../../../src/neurospatial/events/_core.py) — PSTH `set_title`, which overwrites the user's title.
- [src/neurospatial/layout/_plotting.py](../../../../src/neurospatial/layout/_plotting.py) — `env.plot` titles and hidden ticks.
- [src/neurospatial/regions/plot.py:46](../../../../src/neurospatial/regions/plot.py) — `plot_regions`.
- [src/neurospatial/encoding/spatial.py](../../../../src/neurospatial/encoding/spatial.py) around line 1247 — the "Firing Rate (Hz)" colorbar label to copy.
- `docs/reviews/UX_REVIEW_2026-11-02.md` §5. The audit PNGs were in the audit session's scratchpad. Regenerate them rather than rely on them.

**Contracts referenced:**

- [Argument vocabulary](shared-contracts.md#argument-vocabulary) — `ax` keyword-only (already landed in 3a).

## Tasks

- **NaN in the field count.** In both branches of `plot_population_coverage`
  (`population.py:601`, `:634`), plot
  `np.where(result.is_observed, result.field_count, np.nan)`. Use a discrete
  colormap with integer ticks, and the colorbar label "Fields per bin".
  Remove the 0–1 continuous colorbar from the binary mode.
- **Decoding plot.**
  - 1-D (native grid or linearized track): y-axis is linear position in
    `env.units`, using bin centers as the image extent. Draw a legend for
    "MAP" and "Actual position".
  - 2-D: draw two stacked panels of x(t) and y(t) marginal posteriors. This
    needs an axes pair, so accept `ax=(ax_x, ax_y)` or create a figure when
    `ax is None`. Alternatively, raise with a Fix pointing to
    `env.plot_field(result.posterior[t])`.
  - Pick the marginals; it is the standard figure.
  - Default `colorbar=True`.
- **Directional place fields plot.**
  - 1-D: overlay one line per direction against position in `env.units`.
  - 2-D: one `env.plot_field` panel per direction (`ax=` accepts a sequence;
    create a figure when `None`). Remove the bin-index axis.
- **Polar check.** A shared helper `_require_polar(ax, caller)` in
  `src/neurospatial/_plotting.py` (new private module, or the closest
  existing shared plotting module). When `ax.name != "polar"`, raise
  `ValueError`:
  - **What:** "{caller} draws on a polar axes."
  - **Fix:** "create it with `fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})`,
    or pass `polar=False`."

  Use it in `plot_head_direction_tuning`, `DirectionalRateResult.plot` (when
  `polar=True`), `plot_object_vector_tuning` and `plot_circular_basis_tuning`.
- **Titles.** Set a title only when the function created the axes or the
  caller passed `title=`. Apply this to:
  - `env.plot` (`layout/_plotting.py`, the three `set_title` sites)
  - the PSTH plot (`events/_core.py:1541`)
  - population coverage
  - directional place fields
- **Colorbars.**
  - Label every rate colorbar "Firing rate (Hz)": object-vector,
    boundary-vector, view and egocentric `plot` methods.
  - `env.plot_field` defaults its colorbar label to the field's units when
    known, and otherwise to "".
  - Use one flag name, `colorbar: bool = True`, everywhere. Rename
    `add_colorbar` (`plot_object_vector_tuning`).
- **Axis labels.**
  - Phase precession: the x-label default includes `env.units` when known;
    the y-label follows `angle_unit`.
  - Tuning curve: the x-label uses `variable` and `units`, and raises nothing
    when they are absent.
  - HD polar plots label the radial axis in Hz and move radial tick labels
    off the data (`ax.set_rlabel_position`).
- **`plot_regions`.**
  - `ax.set_aspect("equal")`.
  - Cycle colours from the default property cycle.
  - Put the legend inside the axes (`loc="best"`).
  - Label axes with `env.units` when an environment is given.
- **`env.plot` readability.** Show ticks with units, and keep bin outlines
  thin enough that a 2 cm grid does not render as a solid block. Match
  `plot_field`'s linewidth.
- **Own-axes support.**
  - `plot_session_summary` and `plot_basis_functions` accept `ax=` (an array
    of axes) and create a figure only when `None`.
  - `plot_session_summary` applies `unit_ids` to the raster as well.
- **Docs.** Regenerate any affected figures in `docs/user-guide/` notebooks.
  CHANGELOG "Fixed".

## Deliberately not in this phase

- The fastplotlib/napari backends.
- The population heatmap (`idx=None`) belongs to phase 4.
- Colormap changes; viridis stays.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_population_coverage_count_keeps_unobserved_nan` | The image array has NaN exactly at `~is_observed`. |
| `test_decoding_plot_1d_axis_in_cm` | The y-limits span the bin-center range, the y-label contains `env.units`, and a legend is present. |
| `test_decoding_plot_2d_marginals` | Two axes are returned, with y-labels "x (cm)"/"y (cm)". |
| `test_directional_place_fields_plot_axes` | No "Bin index" label; the x-label contains units. |
| `test_polar_plot_on_cartesian_axes_raises` | `ValueError` whose text contains `projection`. |
| `test_plots_keep_user_title` | For each listed function: `ax.set_title("X")`, call with `ax=ax` → the title is still "X". |
| `test_rate_colorbars_labelled` | For each listed result `plot`, the figure's colorbar label contains "Hz". |
| `test_plot_regions_equal_aspect_and_colors` | `ax.get_aspect() == 1.0`, and patch facecolors are distinct. |
| `test_summary_plots_accept_axes` | `plot_session_summary(session, ax=axes)` draws into the provided axes. |

Run with `MPLBACKEND=Agg`. Build the figures in the tests; never compare
images pixel by pixel.

## Fixtures

Simulators with `rng=0`, plus a small synthetic HD cell and an object-vector
cell (reuse `tests/encoding` fixtures).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
