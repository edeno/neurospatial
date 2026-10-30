# CLAUDE.md

neurospatial discretizes spatial environments into bins with connectivity
graphs and builds spatial neuroscience analyses on them: neural encoding,
decoding, behavior, peri-event analysis, simulation, animation and NWB I/O.

## Design priority: the scientist's experience

Write tutorials for spatial neuroscientists familiar with hippocampal recording,
place fields, occupancy, running bouts and PSTHs. Use simple technical English.
Start with the experimental question, explain analysis choices, interpret each
main figure, and show how to use the reader's own recording. Introduce package
terms when they are needed for the analysis.

Scientific correctness is part of usability: the obvious path must preserve the
meaning of the analysis and behave as a scientist expects. The long-term goals
are in the [project vision](docs/vision.md). Apply the
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
- Keep what a result needs to be reproduced: input selections, exclusions,
  parameters, versions and random seeds.
- Make outputs interpretable and errors actionable: say what failed, why, and
  how to fix it.
- When changing analysis behavior, test scientific invariants and the complete
  workflow, not only that a call completes. Check numerics independently of the
  code under test, and state what a method estimates and the limits of its
  validation.

## Commands

Run Python tools through `uv run`. Test and lint tooling lives in the `dev`
extra (`uv add --optional dev <pkg>`), and `uv sync` removes extras you do not
name, so keep it when adding another: `uv sync --extra dev --extra nwb`.

```bash
uv run pytest --doctest-modules src/neurospatial/ -n 0   # doctests run serially
uv run mypy src/neurospatial
uv run mypy tests/typing/top_level_lazy_exports.py tests/typing/finite_volume_consumers.py tests/typing/environment_protocol.py tests/typing/spike_inputs.py
NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py  # after an intentional API change; review the diff
```

## Scientific conventions

- Angles are radians. Allocentric: 0 = East, +π/2 = North. Egocentric (animal
  frame): 0 = ahead, +π/2 = left, −π/2 = right, ±π = behind. Mixing the frames
  gives plausible but wrong results.
- Missing is not zero. Rate maps always store unsupported bins as NaN; there
  is no option to store a fill. Decode with
  `BayesianDecoder.from_rates(rates)` or pass the rate result itself to
  `decode_position`; bare zero-filled arrays need
  `prior=rates.spatial_support * env.bin_sizes`.
- A decoding posterior or prior is a probability mass per bin. The default
  prior is uniform over space (mass proportional to `env.bin_sizes`), and the
  MAP bin and credible region rank bins by density, mass / bin size.
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
