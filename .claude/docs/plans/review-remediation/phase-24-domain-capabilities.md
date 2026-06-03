# Phase 24 — Domain capability gaps: ripple detection & theta phase

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The decoding module is framed around replay, which is defined on sharp-wave-ripple events the library cannot currently produce; and the phase-precession path (O'Keefe & Recce) needs a theta phase it does not provide (DESIGN-REVIEW Med/Low). The resolved decision (below) implements theta phase in-library and scopes ripple detection out to the external `ripple_detection` package.

## Decision (RESOLVED) — split per capability

Resolved by the maintainer (2026-06-02). See [overview Open Questions #4](overview.md#open-questions). The decision splits the two capabilities:

1. **Theta phase — IMPLEMENT in-library.** Implement `encoding.theta_phase` (scipy Hilbert analytic-signal phase on a theta-band filtered LFP). No new hard dependency (`scipy` is already required).
2. **Ripple detection — SCOPE OUT.** Do **not** implement `events.detect_ripples`. Direct users to the external `ripple_detection` package and feed its returned intervals into `peri_event_histogram`. Document this in the `decoding` replay docstrings and a worked example. Do **not** add `ripple_detection` as a hard dependency — reference it as the recommended external tool (optionally as an example/test extra).

Follow the **theta-phase "implement"** task and the **ripple "scope out"** task below accordingly. The movement-heading vs head-direction docstring note is added regardless.

**Inputs to read first:**

- `events/` — the public surface and `intervals` types and `peri_event_histogram`, so the ripple-detection example feeds `ripple_detection`'s intervals through the existing PSTH in the representation the rest of `events` uses.
- `encoding/phase_precession.py` — the consumer of theta phase; read what phase array shape it expects so `theta_phase`'s output is drop-in.
- `pyproject.toml` — confirm `scipy` is already a dependency (it is used elsewhere); do not add new hard deps.

## Tasks — theta phase (IMPLEMENT)

- `encoding.theta_phase(lfp, sampling_rate, *, band=(6, 10)) -> NDArray`: theta-band filter + Hilbert analytic-signal phase, in the convention `phase_precession` expects (verify the consumer). `scipy` only — no new dependency.
- Docstrings + a runnable Example (registered in phase 23). CHANGELOG entry.

## Tasks — ripple detection (SCOPE OUT)

- Do **not** implement `events.detect_ripples`. Instead, document in the `decoding` replay docstrings (and a worked example) that ripple intervals come from the external `ripple_detection` package, and that its returned intervals feed directly into `peri_event_histogram`.
- Do **not** add `ripple_detection` as a hard dependency — reference it as the recommended external tool (optionally an example/test extra).

## Tasks — both paths

- Add a docstring note in `compute_directional_rate`/`is_head_direction_cell` distinguishing velocity-derived **movement heading** from **head direction** (prevents the common methodological mislabel) — regardless of the above.

## Deliberately not in this phase

- Spike-sorting / LFP loading — out of scope; these take an LFP array the user already has.
- Any change to `phase_precession`'s statistics — that's phase 5.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_theta_phase_monotonic` | On a pure theta sinusoid, `theta_phase` increases ~linearly mod 2π and feeds `phase_precession` without reshaping. |
| `test_replay_docstrings_point_to_ripple_detection` | The `decoding` replay docstrings mention that ripple intervals come from the `ripple_detection` package (grep-style doc assertion). |
| `test_ripple_detection_example_feeds_psth` | Skippable (`pytest.importorskip("ripple_detection")`) example test: `ripple_detection`'s returned intervals feed `peri_event_histogram` end-to-end. |

## Fixtures

Synthesized LFP traces (sinusoid + injected burst) in `conftest`; no real data required.

## Review

Dispatch `code-reviewer`. Confirm the gate decision is recorded and the file's tasks match it; no new hard dependency added; any implemented function returns the library-standard type/shape (verified against the consumer); the heading-vs-head-direction note is added regardless of the gate; CHANGELOG updated; no plan/phase references in code/test names.
