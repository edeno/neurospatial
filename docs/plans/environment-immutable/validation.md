# Stage B validation record

Implementation branch: `feature/environment-immutable`.
Behavior-capture base: `cba4f1b093b0ceaba3f800a093952b5b860f1195`.

## B0 — Behavior and ownership capture

The existing isolated Python 3.13 environment is used through `uv run --no-sync`,
with Agg Matplotlib and temporary writable cache directories. No environment
installation or runtime dependency change was needed.

The baseline command recorded **4,051 probes**, including **584 expected
refusals**, before any Stage B production code changes:

```bash
uv run --no-sync python scripts/environment_baseline.py record /tmp/immutable-base.npz
```

The local artifact is
`/private/tmp/environment-immutable-base-cba4f1b0.npz`; it is not a portable
committed fixture. The script covers ten native environments, copied variants,
dictionary/file/NWB restoration, lookup, occupancy, numerical operations,
plotting and public signatures/docstrings.

The separate [ownership capture](probe_ownership.py) records caller-buffer
mutation, point-region copy/deepcopy/pickle, environment shallow/deep copying,
dict/file/actual NWB restoration, metadata/region edits, cached kernels/operators
and two-angle polar layout parameters. Run it with the same interpreter:

```bash
uv run --no-sync python docs/plans/environment-immutable/probe_ownership.py
```

The local JSON artifact is
`/private/tmp/environment-immutable-ownership-base-cba4f1b0.json`. Ownership
changes are intentional; numerical changes unrelated to those mutation attempts
need investigation rather than blanket baseline acceptance.

Independent audits confirmed additional regression requirements:

- Returned sparse operators permit edits and buffer replacement that change
  later gradients/divergence. Returned kernels and attribute DataFrames expose
  cached state; pandas deep copies alone retain nested object-cell aliases.
- Public cached-property assignment can shadow geometry-derived values.
- Rebuilding a retained native layout changes later lazy face geometry without
  changing already published bins. Two-angle polar circularity can similarly
  change after editing retained build parameters and clearing caches.
- Opaque custom locator, track and arena callbacks can retain external mutable
  state. The user approved requiring owned immutable snapshots for these roles.

The B0 API review accepts keyword-only metadata replacement with omitted values
preserved, explicit `None` clearing only nullable units/frame, and whole-region
replacement. Both replacement methods preserve the concrete environment type
and geometry, refresh environment analysis caches, and may share private
geometry-owned lookup memos. The region snapshot keeps the `Regions` read API;
standalone builders remain mutable. Unit labels never imply coordinate scaling.

Each implementation phase still needs regression tests, independent review and
its own validation record before completion. B0 does not establish Stage B
implementation completion or authorize PR publication.
