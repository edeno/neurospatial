# Dataset loaders: design

Status: design, written 2026-11-03 from a conversation with the maintainer.
The implementation plan comes after this is approved.

## Intent

Ship correct, ready-to-analyze loaders for open spatial neuroscience datasets
as part of neurospatial, as a service to the community.

The person it serves is a researcher outside the original lab, for example a
graduate student who has just read Grosmark & Buzsáki 2016 and wants its
data. In their first ten minutes they should be able to:

1. **Load** a session in one call, in physical units, on a clear clock, with
   the apparatus declared, without rediscovering the file's quirks.
2. **Understand** what they loaded: units and their brain areas, what was
   tracked and how well, epochs and events, the conversions made and why, and
   the pitfalls of this dataset.
3. **Reproduce** a small result or known property from the paper, which is
   their first worked analysis and their evidence that the data and the
   package are right.

Every lab that analyzes these datasets rediscovers the same traps: times in
ms, positions in pixels labelled cm, swapped or coincident LEDs, two tracks
stored as one, linearizations that leave out reward areas, head direction
with a mounting offset, apparatus never stated in the file. A loader known
to be correct turns days of that into one call, with the caveats written
down. It is also the package's own evidence that it works on real data.

## Goals

- **Coverage:** every open spatial neuroscience dataset is eventually loaded
  this way: hippocampus, entorhinal, head direction, and the rest (imaging,
  bats, VR, non-rodent). The order below only sets priority.
- **One session at a time:** a loader fetches one session, and fetches a
  whole dataset only when asked. Every session of an archive is offered;
  sessions examined by hand are marked verified.
- **A full summary on print:** printing a loaded recording shows the full
  summary, including its "watch out" list (below).
- **An overview figure:** one call draws the session.
- **One worked result per dataset,** from the paper, where possible.
- **Showcase datasets** (3-D, complex mazes, changing environments, the
  animal's own frame, pose) show what the package can do and prove it works
  where it is hardest.

## Non-goals

- **Re-implementing NWB or pynapple.** The loaders call the package's
  existing readers and exporters (`neurospatial.io`, below) and add only what
  the files lack.
- **Converting archives into NWB files.** That is NeuroConv's job.
- **Mirroring every field of every file,** or re-analysis frameworks for each
  paper.
- **A session object that analysis functions accept.** The settled public
  API takes plain arrays (`f(env, spike_times, times, positions)`); see the
  canonical argument order in `CLAUDE.md`.
- **Redistributing data.** Data is fetched from its source archive.

## Anti-goals: what would make this a failure even if every call works

- **Wrong data:** a conversion without evidence, such as ms read as s,
  pixels labelled cm, or swapped LEDs.
- **Misaligned times:** spikes, tracking, events and epochs not on one
  clock.
- **A wrong environment:** an apparatus misdeclared, or inferred from the
  samples without saying so.
- **Functions that don't do what they claim** on the loaded data.

A quietly wrong loader is worse than none, because the student will publish
on it. Correct outranks many.

## What the maintainer decided

### The first look

Printing a loaded recording shows the full summary, in this order:

1. **Identity:** paper and animal, session and epoch, apparatus, brain
   region, archive, license. `rec.citation` holds the full citation.
2. **Time:** duration, the clock the times are on, position sampling rate,
   and the named epochs (run, sleep, rest).
3. **Tracking:**
   - what was tracked (head LEDs on a headstage, a DLC body part, …);
   - the share of samples tracked and the longest gap;
   - speed ranges (median, 95th percentile);
   - head-direction coverage, and where heading comes from.
4. **Environment:** declared or inferred, with its stated dimensions and
   their source. `rec.env.plot()` draws it.
5. **Neurons:** the number of units, how many fire in this epoch, units per
   brain area where the file says, and the authors' cell-type labels where
   present.
6. **Watch out:** every known pitfall of this dataset, with numbers, for
   example J16's +13° heading offset, or 64% of head samples beyond the
   hallway width (head swings, not misfit). This section is always printed
   in full.

`rec.plot()` draws one overview figure:

- the environment with the path over it;
- occupancy, with unvisited bins shown as missing;
- speed, with the running threshold marked;
- head direction, plus heading minus the direction of travel;
- units per brain area, and spikes per unit;
- a minute of position over time with a spike raster.

A mockup of the summary and figure is in
`.superpowers/brainstorm/*/content/first-look.html` (not committed).

### Attributes: plain arrays in the order functions take them

```python
rec = load_comrie_2026_j16()          # name illustrative
rates = compute_spatial_rates(rec.env, rec.spike_times, rec.times,
                              rec.positions, epochs=rec.epochs["run"])
```

| Attribute | Content |
| --- | --- |
| `rec.env` | the `Environment` |
| `rec.times`, `rec.positions`, `rec.headings` | arrays, shapes `(n_time,)`, `(n_time, n_dims)`, `(n_time,)` |
| `rec.spike_times` | one array per unit |
| `rec.units` | a table: unit id, brain area, the authors' label, spike count |
| `rec.epochs` | named `(n, 2)` arrays |
| `rec.events` | name → times |
| `rec.trials` | the trials table |
| `rec.params` | analysis choices used in the worked result (bin size, speed threshold) |
| `rec.citation` | the citation |
| `rec.conversions` | every conversion applied |

This is option A from the conversation. What the student learns here carries
over to their own data, which they will also have as arrays.

### Building on NWB and pynapple, not beside them

- **NWB datasets:** the loader opens the file (cached or streamed) and calls
  the package's own readers: `io.nwb.read_position`, `read_head_direction`,
  `read_units`, `read_trials`, `read_events`, `read_intervals`, `read_pose`
  and `read_environment`. It adds only the declared environment, the stated
  conversions, the watch-out list and the citation.
- **Reader bugs are fixed in the reader.** When a reader cannot read a real
  file correctly (nonstandard locations, a missing `conversion`, units
  outside the standard path), the fix goes in `io.nwb`, not in a workaround
  inside the loader. Every DANDI loader is therefore a real-file test of the
  readers. This continues the maintainer's earlier decision to improve the
  NWB readers (survey gaps G-011, G-012, G-032).
- **Non-NWB archives** (CRCNS `.mat`, `.whl`, `.res`, Zenodo `.npz`, Axona)
  are converted to the same plain arrays.
- **Export** uses the existing `to_pynapple` and the `io.nwb.write_*`
  functions. There are no new exporters; the docs show the calls.

### Downloads

- **Only the requested session is fetched.** Remote zip members are read in
  place, as the survey's `fsspec` byte-range reads already do. Fetching a
  whole dataset is an explicit call that states the total size first.

### Sessions

A loader offers **every session in its archive.** The automatic checks
(clock, units, ranges, the declared environment) run on each one. Sessions
we have examined by hand are marked **verified** in the summary and carry
the detailed watch-out list. Every other session says "not yet checked by
hand", alongside whatever the automatic checks found.

### Facts the file does not state

When a fact the analysis needs is missing from the file but can be inferred
with stated evidence, the loader **converts and marks it inferred.** The
evidence goes in the summary and the watch-out list. Examples:

- J16's unit is cm, from the speed match;
- hc-3's cm per pixel comes from the tracked span;
- Mouse32's heading frame is assumed.

With no evidence, the loader keeps the raw units (pixels, arbitrary) and
says so. `rec.conversions` marks every conversion as either stated by the
source or inferred, with its evidence.

### Which archive to fetch

When the same data is in several archives, the loader fetches the
**tutorial-familiar file**, the one other tools teach with, so a student's
numbers match those tutorials. For Achilles that is the OSF NWB used by
NeMoS and pynapple; for Mouse32, OSF `jb2gd`.

- The summary shows that file's license as the archive states it ("not
  stated" where it isn't).
- The other archives (for Achilles: DANDI 000044, CRCNS hc-11) serve as the
  cross-check that both copies agree.
- Data is fetched from the archive, never redistributed.

### Which datasets first

Every spatial dataset eventually. The order:

1. **Tutorial staples and our own example.** The 2026-11-02 tool survey
   (`docs/research/research_notes/Open spatial neuroscience
   datasets/tool_tutorial_datasets.md`) found three datasets recurring across
   pynapple, NeMoS, CEBRA and RatInABox:
   - Grosmark & Buzsáki 2016, rat Achilles (CA1 linear track);
   - Peyrache 2015, Mouse32 (head direction);
   - Sargolini 2006 (MEC, DANDI 000582).

   The package's real-data example uses J16 (Comrie et al. 2026, Zenodo
   23020757).
2. **Showcase datasets,** more complicated, to show the package's power and
   prove it works:
   - **3-D:** the Yartsev/Forli bat room; the Grieves 3-D lattice.
   - **Complex mazes:** the Rosenberg labyrinth, the J16 bandit maze, the
     Shipley 8-arm star.
   - **Environment change:** the 000447 novel and familiar W-tracks.
   - **The animal's frame:** egocentric boundary cells (Alexander), object
     data (Kanter).
   - **Pose:** DLC datasets read with `read_pose`.
3. **The rest, by how well known the paper is,** hippocampus first: Mizuseki
   2012 (hc-3), Jadhav 2016 (hc-28), the 000044 sessions, then Hafting, Gardner
   and the other survey loaders, until the catalogue
   (`docs/research/reports/Open spatial neuroscience datasets.md`) is covered.
   This order is a judgment, and the maintainer may reorder it.

### Worked results

One per dataset, proposed from the paper and approved by the maintainer.
Each is cited only from text that was read. There are two levels:

- **a reproduced number** with the paper's stated parameters, where its
  Methods allow;
- **a known property of the data** where an exact number is out of reach.

Each dataset's page says which level it reaches and why, and when a number
differs from the paper, every parameter that could not be matched. The
maintainer expects some to be hard.

## Correctness: how each anti-goal is guarded

1. **Wrong data.** Every conversion is stated with its source. Every number
   the student sees is checked against the file, the paper, an independent
   computation, or planted truth. These are the checks of the survey's
   phase 6 (`.claude/docs/plans/real-data-survey-batches/phase-6-correctness-and-ux.md`).
   A conversion without evidence does not ship.
2. **Misaligned times.** Each loader proves one clock:
   - spikes lie within the tracked span;
   - events land at plausible behavior (well entries at the wells);
   - epochs contain what they name;
   - where a session exists in two archives, both agree (hc-11 against DANDI
     000044 already does).

   A failed check refuses the load; it does not warn.
3. **Wrong environment.** The environment is declared from the stated
   apparatus and judged on misfit evidence, not on tracking spread (survey
   phase 5). An inferred environment says so in the summary.
4. **Functions that don't work.** Each dataset's worked result is a test that
   must keep passing. Planted cells on that dataset's own path and apparatus
   (the phase-6 simulation ladder) check the encoding and behavior functions
   against its real messiness.

## Calls made by me, not the maintainer (the maintainer can veto)

- **The public loaders replace the survey loaders.** `datasets/` (the survey
  harness) imports the public loaders instead of keeping its own copies, so
  there is one loader per dataset. The survey's checks become those loaders'
  evidence.
- **`data/load_bandit_data.py`,** the J16 loader behind
  `examples/04_real_data_analysis.py` and `tests/test_example_data.py`, is
  replaced by the public J16 loader in the same change. The example moves to
  it.
- **Location:** a `neurospatial.datasets` module, behind an optional extra.
  The reader dependencies (DANDI streaming, MATLAB tables, zip byte ranges)
  stay out of the core install, and the module imports them lazily, as the
  package's existing optional extras do.
- **Cache:** the first load saves the session to a per-user cache. The
  location can be changed with an environment variable. Each file is
  verified against a pinned checksum, and later loads work offline. Nothing
  is re-fetched silently.
- **Saving versus streaming:** the default saves the session file whole, so
  the second analysis is instant; streaming is an option for very large
  files. The summary says what was fetched, from where, and its size before
  fetching.
- **CRCNS:** each user's own account, read from environment variables, never
  stored or printed. Without them, the error says where to register and
  which variables to set. The survey's handling already works this way; it
  must also catch every exception around the login, so credentials can never
  reach a traceback (open survey finding).
- **Untrusted data:** never unpickle, pin checksums, check archive members.
  The survey's restricted unpicklers and safe readers carry over unchanged.
- **Watch-out entries are written per dataset in its loader,** with the
  survey's evidence behind each one. Ranges such as speed and tracked share
  are computed when the summary prints.
- **What belongs in the data model.** It is done once, so it covers what the
  survey found the files hold:

  | Attribute | Content |
  | --- | --- |
  | `rec.linearized` | the authors' linearization, where present |
  | `rec.regions` | stated wells, ports, platforms and boxes, as package regions |
  | `rec.units` | includes the authors' cell-type and module labels |
  | `rec.lfp` | optional and loaded only on request (one channel and its rate) |
  | `rec.activity` and its frame times | imaging, so imaging datasets need no new type later |

  A file's own speed and the paper's numbers are not attributes. They are
  evidence used in the checks and on the dataset's page.
- **The clock check,** "spikes lie within the tracked span", is scoped to
  tracked epochs. Sleep without tracking is normal (Mouse32, hc-11,
  000044).
- **No override for a failed check.** A failed check refuses with an
  actionable message: what failed, the numbers, and what would fix it.
  Sessions nobody has examined by hand still load (see Sessions); a session
  whose own checks fail does not.
- **Worked results:**
  - The maintainer approves each one at PR review.
  - A dataset may ship before its worked result exists. The page then says
    "worked result: not yet" or why it is not possible.
  - Any result that does ship must pass.
- **Sequencing with the survey:**
  - Each public loader ships with the checks for its own dataset (the phase-6
    kinds of check that apply to it), not on phase 1–5 evidence alone.
  - Phase 6's cross-cutting work (the simulation ladder, the messiness
    inventory, the UX and figure checks) runs against the public loaders.
  - Phase 6's task 6a is replaced by this module's data model.
- **Testing in CI: deferred** (maintainer, 2026-11-03). The real-data checks
  run locally for now. Committing slices of real data, scheduled jobs and
  CRCNS credentials in CI are decided later.
- **Docs:** one page per dataset, covering what was recorded, units and
  conversions, the watch-out list, the worked result, and how to cite it.
  The survey loaders' docstrings already hold most of this.

## Left to the builder

- Loader naming and registry shape (one function per dataset or per
  session; how sessions are listed), within the public API contract in
  `.claude/rules/public-api.md`.
- The type of `rec` and of `rec.units` and `rec.trials`, within "plain
  arrays and existing package types". Use the package's interval type where
  it exists.
- The cache library (the survey's own fetcher, or `pooch` as NeMoS uses).
- How the summary is laid out, within the order above.
- The order of work within the dataset priority.

## Constraints

- **Public API:** follow `.claude/rules/public-api.md` and the canonical
  argument order. Public names are test-enforced in
  `tests/data/public_api.txt`.
- **Missing is not zero:** NaN stays distinct from zero, angles are radians
  in the package's frames, and physical units are kept throughout.
- **Provenance:** results keep what is needed to reproduce them (inputs,
  exclusions, parameters, versions, seeds).
- **Cite only verified:** a paper number appears only if its text was read.
- **No backward compatibility:** there are no users of the survey loaders
  outside the repo.
- **Docs:** user-facing docs and the CHANGELOG ship with the change.
- **Shipping:** one PR per piece of work, as the maintainer prefers.

## Relationship to the survey work in progress

- **Survey phase 5** (declared environments) and **its pending review
  fixes** finish first. They feed the environment and correctness evidence.
- **Survey phase 6** (checks against a truth) supplies the correctness
  evidence. Its task 6a, loaders handing on the file's truth, is the same
  work as this module's data model. It should be done once, here, rather
  than in the survey harness first.
