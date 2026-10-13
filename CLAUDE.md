# CLAUDE.md

neurospatial discretizes spatial environments into bins with connectivity
graphs and builds spatial neuroscience analyses on them: neural encoding,
decoding, behavior, peri-event analysis, simulation, animation and NWB I/O.

## Design priority: the scientist's experience

Scientific correctness is part of usability: the obvious path must preserve the
meaning of the analysis and behave as a scientist expects. The long-term goal is
coherent workflows covering spatial neuroscience, from recorded data through
analysis, visualization and export. Apply the
[design principles](docs/contributing.md#design-principles) to every change:

- Start from the scientific task and familiar experimental vocabulary.
- Make common workflows discoverable through public APIs and runnable examples;
  scientists should not need to read implementation code to use them correctly.
- Keep names, signatures and result semantics consistent across domains and
  between single-unit and population analyses.
- Use defensible defaults and state their assumptions; require an explicit
  scientific choice when no default is defensible.
- Preserve time support, coordinate frames, physical units and unit identity
  across operations. Keep missing observations (NaN) distinct from zero activity.
- Make outputs interpretable and errors actionable: say what failed, why, and
  how to fix it.
- When changing analysis behavior, test scientific invariants and the complete
  workflow, not only that a call completes.

## Commands

Run Python tools through `uv run`. Test and lint tooling lives in the `dev`
extra (`uv add --optional dev <pkg>`), and `uv sync` removes extras you do not
name, so keep it when adding another: `uv sync --extra dev --extra nwb`.

```bash
uv run pytest --doctest-modules src/neurospatial/ -n 0   # doctests run serially
uv run mypy src/neurospatial
uv run mypy tests/typing/top_level_lazy_exports.py tests/typing/finite_volume_consumers.py tests/typing/environment_protocol.py
NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py  # after an intentional API change; review the diff
```

## Scientific conventions

- Angles are radians. Allocentric: 0 = East, +π/2 = North. Egocentric (animal
  frame): 0 = ahead, +π/2 = left, −π/2 = right, ±π = behind. Mixing the frames
  gives plausible but wrong results.
- Missing is not zero. Rate maps keep unsupported bins NaN (`fill_value=None`),
  and filling never changes `spatial_support`. Decode with
  `BayesianDecoder.from_rates(rates)` or pass the rate result itself to
  `decode_position`; bare zero-filled arrays need
  `prior=rates.spatial_support.astype(float)`.
- `min_occupancy` is seconds of occupancy per bin, not a density.
- `has_place_field` detects a field, and result-method cell-type checks are
  threshold screens. A shuffle verdict needs `criterion="shuffle"` on raw arrays.

## Canonical argument order

```text
encoding      f(env, spike_times, times, positions[, headings][, object_positions], *, params)
directional   f(spike_times, times, headings, *, params)      # heading is not a position: no env
egocentric    f(positions, headings, targets)
segmentation  f(position_bins, times, env, *, region_params)
behavior      f(env, times, positions, ...) or f(times, positions, ...)
events        f(spike_times, event_times, window, *, bin_size)
```

Data comes before metadata, and `times` precedes `positions` whenever both are
required. The full public API contract is in `.claude/rules/public-api.md`;
read it before adding or changing a public function or result class.

## Where to look

- Public names and signatures: `tests/data/public_api.txt` (test-enforced).
- Executed, worked examples: `docs/user-guide/` and docstring Examples sections.
- Environment architecture rules: `docs/contributing.md`.
- Path-scoped rules in `.claude/rules/` load when you open matching files: the
  API contract, environment and layout internals, animation, NWB, tests and
  executable Markdown.

User-visible changes get an entry under "Unreleased" in `CHANGELOG.md`.
