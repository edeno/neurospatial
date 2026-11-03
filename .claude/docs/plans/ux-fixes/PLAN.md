# UX fixes Implementation Plan

**Status:** In progress — phase 7 (merged: 1 #257, 2 #258, 3a #259, 3b #262, 4 #266, 5 #264, 6 #265)

These changes fix the problems found in the 2026-11-02 UX audit
(`docs/reviews/UX_REVIEW_2026-11-02.md`):

- Mistakes that run without error and return a plausible but wrong number now
  warn or raise.
- The quickstart and the built-in simulators work out of the box.
- Arguments and defaults use one name and one meaning across the package.
- Result objects and plots follow the documented contract.
- Errors name the right call.
- A first-time user, or an LLM assistant, can find the right function from
  `help()` and the docs.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file
   is self-contained: it lists the files to read, the contracts and designs it
   depends on, the tasks, the validation slice and the fixtures.
2. **Need shared rules (argument names, the discarded-data warning,
   verdict vs screen)?** [shared-contracts.md](shared-contracts.md).
3. **Need the detection algorithms for the silent-result guards?**
   [designs.md](designs.md).
4. **Need broader scope, risks or rollout?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, risks,
  rollout and open questions.
- [shared-contracts.md](shared-contracts.md) — the argument vocabulary, the
  discarded-data warning contract and the verdict/screen split, used by
  several phases.
- [designs.md](designs.md) — detection rules and code for the phase-1 guards.
- Phases (each ships as a separate PR, in this order):
  - [phase-1-silent-result-guards.md](phase-1-silent-result-guards.md) —
    warn or raise on the nine wrong-but-plausible cases.
  - [phase-2-first-run-repair.md](phase-2-first-run-repair.md) — quickstart,
    simulators, 1-D `env.plot`, README and workflow snippets.
  - [phase-3a-argument-names.md](phase-3a-argument-names.md) — one name per
    concept (`rng`, region arguments, `dt`, `ax`, …), enforced by tests.
  - [phase-3b-defaults-and-returns.md](phase-3b-defaults-and-returns.md) —
    harmonized defaults, real values instead of `None`, one return type per
    function, `criterion="shuffle"`.
  - [phase-4-result-contracts.md](phase-4-result-contracts.md) — population
    `plot`, `PlaceFieldsResult`, population significance result, units in
    attrs, properties vs methods.
  - [phase-5-plot-fixes.md](phase-5-plot-fixes.md) — NaN in field counts,
    spatial axes, polar-axis check, titles and colorbars.
  - [phase-6-error-gaps.md](phase-6-error-gaps.md) — wrong-kind inputs, typo
    and choice hints, `InputError`, style cleanup.
  - [phase-7-discoverability.md](phase-7-discoverability.md) — Start Here
    blocks, root `decode_session`, `__dir__`, one public home per name,
    glossary, `compute_` naming rule, `llms.txt`.
