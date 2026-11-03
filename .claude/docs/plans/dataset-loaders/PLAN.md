# Dataset loaders Implementation Plan

**Status:** Not started.

**Goal:** Ship `neurospatial.datasets`: correct, ready-to-analyze loaders for open spatial neuroscience datasets. This plan covers the module, its first five datasets, and Valero 2022 (DANDI 000568) as the first dataset with video.

**Architecture:**

- **Built on the existing interfaces.** Loaders use the package's own NWB readers (`neurospatial.io.nwb`) and fetch code moved from the survey harness. They return a `Recording` of plain arrays and an `Environment`, in the order the analysis functions take them.
- **Nothing reinvented.** No new NWB or pynapple interface and no session object that functions accept. Export uses the existing `to_pynapple` and `io.nwb.write_*`.
- **One implementation per dataset.** The survey harness (`datasets/`) imports these loaders instead of keeping its own copies.
- **The decisions** this plan implements are in [DESIGN.md](DESIGN.md), approved 2026-11-03. Read it first; this plan does not restate it.

**Tech stack:** numpy, pandas, pynwb/hdmf (the `nwb` extra), fsspec, h5py, inflate64, scipy.io, mat-io (Python ≥ 3.11), networkx, matplotlib, shapely.

**Out of scope:** every dataset beyond the five listed below. The datasets after these are planned separately, with this module as their base.

## Prerequisites

- **Survey phase 5 is merged:** declared environments, the shared verdict, and the review fixes from the other session, including the CRCNS exception leak and the failing pose test (`datasets/tests/test_operations_pose.py`). This plan moves `datasets/_fetch.py`, so phase 5's fixes to it must land first.
- **Branch from `main` after that merge.**

## Inputs to read first

- [DESIGN.md](DESIGN.md): every user decision. Sections "What the maintainer decided" and "Calls made by me" are binding.
- [.claude/rules/public-api.md](../../../rules/public-api.md):
  - names and argument order (lines 15–59);
  - result-class verbs (lines 61–99): `summary()` is a flat scalar dict whose keys read back; `plot(*, ax=None)` returns the Axes;
  - public homes (lines 116–129).
- [datasets/_fetch.py](../../../../datasets/_fetch.py):
  - `fetch` (line 90), `fetch_zip_member` (158), `_crcns_open` (215, the handler at 227–236), `fetch_crcns` (249), `fetch_crcns_tar_members` (264), `dandi_asset_url` (318), `dandi_stream_url` (323), `open_nwb` (341–360);
  - `DEFAULT_CACHE` (line 36) is relative to the file's own location and changes when the file moves.
- [datasets/_record.py](../../../../datasets/_record.py): `Recording` (lines 20–74), `validate_recording` (77–130), `checked_span` (285), `heading_vs_movement` (232), `alternating_blocks` (217).
- The survey loaders being replaced: `datasets/load_j16_23020757.py`, `load_hc11_achilles.py`, `load_grosmark_000044_linear.py` and `load_grosmark_000044_circular.py` (`load_session` at lines 121–220, the DANDI 000044 path), `load_peyrache_mouse32.py`, `load_sargolini_000582.py`, `load_yartsev_bat_demo.py`. Read each docstring: their conversions and caveats become the watch-out lists.
- [src/neurospatial/io/nwb/](../../../../src/neurospatial/io/nwb/):
  - `read_position` (`_behavior.py:36`), `read_head_direction` (`_behavior.py:435`), `read_units` (`_units.py:85`), `read_intervals` (`_events.py:234`), `read_trials` (`_events.py:1096`);
  - container search order (`_core.py:123`), and `_MissingObjectError(KeyError)` (`_core.py:97`).
- [src/neurospatial/_optional.py:11](../../../../src/neurospatial/_optional.py): `require_optional(module, *, extra, purpose)`.
- [`src/neurospatial/__init__.py:343`](../../../../src/neurospatial/__init__.py): `_LAZY_SUBMODULES` (343–364) and `__all__` (414). A new submodule is added to both.
- [tests/test_public_api_snapshot.py:12](../../../../tests/test_public_api_snapshot.py): `NAMESPACES` (lines 12–38).
- [src/neurospatial/_exceptions.py](../../../../src/neurospatial/_exceptions.py): `NeurospatialError` (24), `_format_error(what, *, fix, why=None)` (69).
- [src/neurospatial/_interval_table.py](../../../../src/neurospatial/_interval_table.py): `Intervals` (327), `Intervals.from_arrays` (1202). Trials use it.
- `data/load_bandit_data.py` (`fetch_bandit_data` at 51, `load_neural_recording_from_files` at 118) and its callers: `examples/04_real_data_analysis.py` (lines 65, 72, 89, 90), `tests/test_example_data.py`, `scripts/profile_napari_animation.py` (lines 51, 81, 84), `README.md:254`, `examples/README.md:59`, `data/README.md`.
- `docs/research/research_notes/Open spatial neuroscience datasets/tool_tutorial_datasets.md` (tutorial files and sessions) and `.claude/docs/plans/real-data-survey-batches/appendix.md` (lines 71–91, the DANDI 000044 asset table).

## Module layout and public surface

```text
src/neurospatial/datasets/
    __init__.py        # __all__, __dir__; lazy: importing it needs no extra
    _recording.py      # Recording, Conversion, Provenance
    _summary.py        # the printed summary text
    _overview.py       # plot_overview
    _catalog.py        # catalog(), the registry reader
    _checks.py         # automatic per-session checks
    _fetch.py          # moved from datasets/_fetch.py
    _errors.py         # DatasetCheckError
    registry/*.json    # one per dataset: sessions, sources, checksums, metadata
    grosmark_2016.py   # one module per dataset, private API: load_session(...)
    peyrache_2015.py
    sargolini_2006.py
    comrie_2026.py
    <bat>.py           # named after the paper the Zenodo record cites (read it)
```

Public names, at `neurospatial.datasets`:

| Name | Signature | Returns |
| --- | --- | --- |
| `catalog` | `catalog() -> pandas.DataFrame` | one row per session. Columns: `dataset`, `session`, `species`, `brain_regions`, `apparatus`, `n_units`, `verified`, `default`, `license`, `archive`, `size_bytes`, `citation` |
| `load` | `load(dataset, session=None, *, cache=None, stream=False) -> Recording` | the session. `session=None` is the dataset's default (tutorial-familiar) session |
| `fetch_all` | `fetch_all(dataset, *, cache=None) -> list[Path]` | downloads every session. Logs the total size first and refuses to start if the cache's disk has less free space than that |
| `Recording` | frozen dataclass | see the attributes below |
| `DatasetCheckError` | `NeurospatialError` subclass | raised when a session fails its checks |

- **One public way.** There is one way to load (`load`) and one to list (`catalog`); no per-dataset public functions.
- **Discovery.** Each dataset's docs page carries what its docstring would have.
- **Naming.** Dataset ids are `<first author>_<year>`; add a suffix only to separate two datasets from one paper.
- **Session ids** are the archive's own (DANDI asset path stem, OSF file stem).

`Recording` attributes (DESIGN.md, "Attributes" and the data-model call):

| Attribute | Type |
| --- | --- |
| `dataset`, `session` | `str` |
| `env` | `Environment`, or `None` when there are no positions (Peyrache: heading only) |
| `env_kind` | `"declared"` or `"inferred"` |
| `times` | `(n_time,)` float64, s |
| `positions` | `(n_time, n_dims)` float64, or `None` |
| `position_units` | `"cm"`, `"m"`, `"pixel"` or `"arbitrary"` |
| `headings` | `(n_time,)` radians, allocentric, or `None` |
| `heading_source` | `str` |
| `spike_times` | `tuple[ndarray, ...]` |
| `units` | `pandas.DataFrame` indexed by `unit_id`. Columns: `brain_region`, `label` (the authors'), `n_spikes`, `fires_in_epochs` |
| `epochs` | `Mapping[str, ndarray (n, 2)]` |
| `events` | `Mapping[str, ndarray]` |
| `trials` | `Intervals` or `None` |
| `linearized` | `(n_time,)` or `None` |
| `activity`, `activity_times` | imaging, or `None` |
| `params` | `Mapping`: the analysis choices used by the worked result, including `train_epochs` and `test_epochs` |
| `conversions` | `tuple[Conversion, ...]`. `Conversion(text, basis, evidence)`, with `basis` in `{"stated", "inferred"}` |
| `watch_outs` | `tuple[str, ...]` |
| `verified` | `bool` |
| `provenance` | `Provenance(archive, url, sha256, size_bytes, neurospatial_version, fetched_at)` |
| `citation` | `str` |
| `n_units`, `duration` | properties |

Methods:

- `rec.load_lfp(channel=None) -> tuple[ndarray, float]`: on request only. It raises `NotImplementedError` for datasets whose loader has no LFP yet, and says which.
- `rec.plot_overview(*, fig=None) -> matplotlib.figure.Figure`. This is not `plot`, because the rules fix `plot(*, ax=None)` to return one Axes, and the overview has six panels.
- `rec.animation(*, time_range=None, trail_length=None, video=False) -> Animation`: an interactive or movie scene of the session, built on `neurospatial.animation` (task 6b); `video=True` fetches the session's video and draws the data over it (task 6c).
- `rec.video -> VideoSource | None`: where the session's video is, its frame times, its calibration and their bases. `None` when the dataset publishes no video. Nothing is downloaded until `rec.animation(video=True)` or `rec.video.fetch()`.
- `rec.summary() -> dict`: flat scalars, every key a property of `rec`.
- `repr(rec)`: the full summary text.

## Tasks

1. **Move the fetch code.** Move `datasets/_fetch.py` to `src/neurospatial/datasets/_fetch.py`.
   - **Cache root.** `cache_root(cache=None)` resolves `cache`, then `$NEUROSPATIAL_DATA_DIR`, then a per-user default: `%LOCALAPPDATA%\neurospatial\datasets` on Windows, `~/.cache/neurospatial/datasets` elsewhere. `platformdirs` is not a dependency, and this needs none.
   - **Imports.** Third-party imports stay lazy, through `require_optional(..., extra="datasets")`.
   - **CRCNS handler.** Catch every exception around `opener.open(...)` in `_crcns_open` and re-raise `OSError` from outside the handler, unless phase 5 already did. Keep `datasets/tests/test_fetch.py`'s credential tests, moved to `tests/datasets/test_fetch.py`. Add one with an `http.client.RemoteDisconnected` raised inside `opener.open` that checks `capture_locals=True` tracebacks contain neither secret.
   - **Survey imports.** Delete `datasets/_fetch.py` and change the survey's imports to `from neurospatial.datasets._fetch import ...`. The survey harness is repo tooling, so a private import there is acceptable.
2. **Packaging.**
   - **Extra.** Add `datasets = ["neurospatial[nwb]", "fsspec[http]>=2024.6", "h5py", "inflate64>=1.0.4", "mat-io>=0.5; python_version >= '3.11'"]` to `[project.optional-dependencies]` (`pyproject.toml:57-102`).
   - **Dependency group.** Reduce the `datasets` group (`pyproject.toml:494-503`) to what only the survey needs (`opexebo`), plus `neurospatial[datasets]`.
   - **Exports.** Add `"datasets"` to `_LAZY_SUBMODULES` and `__all__` in `src/neurospatial/__init__.py`, and `"neurospatial.datasets"` to `NAMESPACES`.
   - **Import without extras.** Importing `neurospatial.datasets` without the extra must work (`tests/test_lazy_imports.py`). Loading a session without it raises the `require_optional` message.
   - **Bat loader on Python 3.10.** It needs `mat-io`, so it raises an `ImportError` that says so.
3. **`Recording`, `Conversion`, `Provenance`.**
   - **Freezing.** Frozen dataclasses. Arrays are made read-only with `ReadOnlyArraysMixin` (`_results.py:274`), calling `_own_arrays()` last in `__post_init__`.
   - **Validation in `__post_init__`.** It requires:
     - `times` strictly increasing;
     - `positions` with first dimension `n_time`;
     - headings in [−π, π];
     - unique `units` ids, matching `len(spike_times)`;
     - `(n, 2)` epochs with start < stop;
     - every `Conversion.basis` either stated or inferred, with non-empty evidence when inferred.

     Failures raise `InputError` with `_format_error`.
   - **Summary and repr.** `summary()` keys (each a property): `n_units`, `duration`, `n_dims`, `sampling_rate`, `tracked_fraction`, `longest_gap`, `median_speed`, `speed_p95`, `verified`. Declare their units in `_repr_units`: s, Hz, s, {units}/s.
4. **The automatic checks (`_checks.py`).** They run on every session before `load` returns, and any failure raises `DatasetCheckError` with what failed, the numbers, and what would fix it. There is no override.
   - **Clock:** spikes lie within the tracked span on tracked epochs (an untracked sleep epoch is not tracked); events lie within the recording; epochs lie within the recording and don't overlap within one name.
   - **Units:** the position span is within the dataset's stated range, using `checked_span`'s logic; the sampling rate matches the stated rate within 1%.
   - **Missing is not zero:** the share of samples at exactly (0, 0) is not above 0.1%. Lost tracking stored as zeros fails.
   - **Frozen and bridged tracking:** runs of ≥ 10 identical position samples, and steps faster than the dataset's stated physical speed limit, are counted. They go into `watch_outs` as numbers, not refusals, because some apparatus has the animal still. The survey review found this pattern in Valero 000568 and 001880.
   - **Unit inventory:** units with no spikes in the analysed epochs are counted in `units.fires_in_epochs` and reported.

   Tests use synthetic recordings, each failing exactly one check, and assert the message names it.
5. **Summary text (`_summary.py`).**
   - **Content.** Renders the six sections in the order DESIGN.md gives ("The first look"). The ranges are computed from the arrays: tracked share, longest gap, speed median and 95th percentile, and head-direction coverage as the share of 10° bins with a sample. The rest comes from the registry and the loader: identity, environment source, watch-outs.
   - **Unverified sessions** print "Not yet checked by hand", followed by the automatic findings.
   - **Mockup.** Use the J16 mockup (`.superpowers/brainstorm/*/content/first-look.html`, local and not committed) as the target layout.
   - **Test.** A snapshot test runs on a synthetic recording.
6. **Overview figure (`_overview.py`).** `plot_overview(*, fig=None)` draws six panels:
   - the environment with the path: `env.plot` plus a scatter of positions;
   - occupancy from `env.occupancy`, with NaN for unvisited bins drawn as missing (use the `nan_color` of `plot_field`);
   - a speed histogram with `params["min_speed"]` marked;
   - a head-direction polar histogram, plus heading minus the direction of travel while running;
   - units per brain region, and spikes per unit;
   - 60 s of position over time with a raster.

   A panel with no data says "not in this dataset" instead of drawing empty axes. Test by reading the artists back: the occupancy image equals `env.occupancy`'s values, and its NaNs are masked.
6b. **Interactive view and movie** (`_overview.py`).
    `rec.animation(*, time_range=None, trail_length=None)` returns an
    `Animation` (`src/neurospatial/animation/scene.py:290`,
    `Animation(env, fields, *, frame_times=None, overlays=None, title=None, window_seconds=2.0, show_regions=False, ...)`).

    - **The scene:**
      - **field:** occupancy over the environment, from `env.occupancy`, as a single static field stream. NaN marks unvisited bins.
      - **overlays:**
        - `PositionOverlay(positions, times, trail_length=..., units=rec.position_units)` (`animation/overlays.py:217`, fields at 337–347);
        - `HeadDirectionOverlay` if `headings` (`overlays.py:636`);
        - `SpikeRasterOverlay(spike_times, unit_ids=...)` (`overlays.py:2007`, fields at 2062–2068);
        - `EventOverlay` for `events` (`overlays.py:844`);
        - `EpochOverlay(epochs)` (`overlays.py:2212`).
      - **regions:** `show_regions=True` when the environment has regions.
      - **scope:** the default `time_range` is the first 60 s of tracking, so the first `.show()` opens fast. A recording without positions (Peyrache) raises `InputError`, saying the session has no positions and pointing to `headings` and `spike_times`.
    - **What the user does next:** `Animation.show(backend=...)` (`scene.py:657`) to open it, and `Animation.export(path, *, time_range, fps, ...)` (`scene.py:505`) for a movie.
    - **Not a workaround:**
      - `Animation` refuses scenes without a field stream (`scene.py:345`, "fields must contain at least one stream"). If a single-sample static field is refused or animates wrongly, fix that in `neurospatial.animation` with its own test and a CHANGELOG line. Don't work around it here.
      - The same applies to any overlay that can't take a `Recording`'s arrays as they are.
    - **Docs:** the user guide shows `rec.animation().show(backend="fastplotlib")` and a 10 s export. The 3-D bat page shows the room in fastplotlib or napari.

6c. **Data over the session's video** (`_video.py`, `_overview.py`).
    - **`VideoSource`** is a frozen dataclass with these fields:
      - `url` or `asset_id`, `sha256`, `size_bytes`, from the registry;
      - `frame_times`, shape `(n_frames,)`, and `frame_times_basis`, a `Conversion`;
      - `calibration`, a `VideoCalibration` (`neurospatial.ops.transforms`) or `None`;
      - `calibration_basis`, a `Conversion` whose evidence holds the fit residual in px and cm;
      - `fetch(*, cache=None) -> Path`, which reuses task 1's fetch and checksum code.
    - **The scene:** `rec.animation(video=True)` builds task 6b's scene and adds a `VideoOverlay(path, calibration=..., times=frame_times, z_order="below")` (`src/neurospatial/animation/overlays.py:2601`). The occupancy field is drawn transparent, so the frames show through. Pose keypoints, where `rec` has them, are drawn with the existing pose or position overlays.
    - **Refusals:**
      - With no video for the dataset: `InputError` saying so.
      - With no calibration: `InputError` saying which calibration sources were tried. `VideoOverlay`'s fallback that stretches the frame over `env.dimension_ranges` is never used.
    - **Frame times:** read the video `ImageSeries` `timestamps`. If the frame count equals the position sample count and the two clocks differ by a constant (000568: 22.55–23.12 s, survey gap G-054), use the position times. Record that as an inferred `Conversion` with the offset as evidence, and add the offset to `watch_outs`. Otherwise use the video's own times and say so.
    - **Calibration, in order:**
      1. A pixel-to-cm factor or transform stated in the file.
      2. A least-squares affine between pixel tracking and the converted positions, where the file stores both. Report the residual. Refuse a fit whose residual exceeds one bin.
      3. Landmarks (track ends, arena corners) placed by hand on one frame, stored in `registry/<dataset>.json` with who placed them. Build it with `calibrate_video(env, path, landmarks_px=..., landmarks_env=...)` (`src/neurospatial/animation/calibration.py:32`).
    - **Dataset:** add `valero_2022` (DANDI 000568) as the sixth dataset. Its default session is the survey's (`sub-fCamk2_ses-fCamk2-201014-sess3`); move the logic from `datasets/load_valero_000568.py`. It is the real-data case for this task because its video is small. Before writing the calibration, read which tracking the file stores in pixels. If none is stored, use landmarks. 001845's videos are too large for the first group and stay with the showcase datasets.
    - **Gaps in the animation module:** fix them there, with their own tests, as in 6b. For example, `VideoOverlay` may not accept an NWB external-file path relative to the NWB file.
    - **Docs:** the Valero dataset page shows the position over 20 s of video, and how to pass the user's own decoded posterior into the same scene.

7. **Registry and catalog.**
   - **Registry contents.** `registry/<dataset>.json` holds:
     - dataset metadata: citation, species, brain regions, apparatus, license as the archive states it, the dataset's stated position span and sampling rate;
     - one entry per session: `session`, `archive`, `url` or DANDI `asset_id`, `sha256`, `size_bytes`, `verified`, `default`, `n_units` if known.
   - **Generation.** Write `scripts/build_dataset_registry.py`. For DANDI it lists a pinned dandiset version through the DANDI REST API: asset path, id, size, and the `dandi:sha2-256` digest. Pinned versions are immutable, so the digests are stable. OSF and Zenodo entries are written by hand, with a SHA-256 computed from one download.
   - **`catalog()`** concatenates the registries. Test that every registry entry has a SHA-256, a size and a license field.
8. **The first five datasets.** Each module has `load_session(entry, path_or_stream) -> Recording` and a `WATCH_OUTS: dict[str, tuple[str, ...]]` keyed by verified session. Move the survey loader's logic in and keep every conversion, recorded as a `Conversion` with its basis.

   | Dataset id | Default session (source) | Other sessions | Verified sessions |
   | --- | --- | --- | --- |
   | `comrie_2026` | J16 `02_r1`, the Zenodo 23020757 `.npz` | none public | it |
   | `grosmark_2016` | `Achilles_10252013`, the OSF `hu5ma` NWB used by NeMoS | the DANDI 000044 sessions (appendix table, 8 sessions, 5–9 GB each), streamed or fetched per session | the default, cross-checked against DANDI 000044's copy of the same session |
   | `peyrache_2015` | `Mouse32-140822`, OSF `jb2gd` (36.6 MB) | the DANDI 000056 sessions (40 files) | the default |
   | `sargolini_2006` | `sub-10073_ses-17010302` (DANDI 000582; the pynapple tutorial's session) | the other 117 DANDI 000582 sessions | the default, and `sub-10884_ses-03080402` (the survey's) |
   | the bat (Zenodo 15738988) | `Dataset_3_14543_240419` | none public | it |

   - **Read the OSF files' structure before writing code.** The OSF NWB files are not the DANDI files: the NeMoS Achilles file has `units`, `position`, `theta_phase` and `trials`. When an `io.nwb` reader cannot read a field correctly, fix the reader in `src/neurospatial/io/nwb/`, with a test on a synthetic NWB built with pynwb. Don't work around it in the loader. List each reader fix in the CHANGELOG.
   - **Declared environments** follow survey phase 5:
     - J16: the authors' track graph, every edge kept;
     - Grosmark: the 1.6 m linear track, with the stated width if the paper states it, else the measured placement tolerance labelled as such;
     - Sargolini: the 1 m square;
     - the bat: the room from `r_lim`;
     - Peyrache: none (no positions).
   - **Default and verified sessions.** Sargolini's default is a session the survey did not check. Check it by hand now, as the survey did for `sub-10884`, before marking it verified.
   - **Inferred conversions** use `basis="inferred"` with their evidence: J16's cm from the `head_speed` match (r ≥ 0.8); Mouse32's heading frame, assumed.
   - **J16 headings.** `head_orientation` is not used as `headings` (+13° offset). Set `headings=None`, keep it in `params["head_orientation"]`, and describe it in `watch_outs`.
9. **Worked results.** For each dataset, read the paper's text (local Zotero first; see the `local-zotero-library` memory) and propose one result:
   - **level A,** a number reproduced with the stated parameters; or
   - **level B,** a known property.

   Write it as a test in `tests/datasets/test_worked_results.py`, marked `realdata`.
   - Register `realdata` in `pyproject.toml`'s markers (lines 164–173), and add `and not realdata` to the default `-m` (line 140).
   - A dataset whose result is not possible yet ships with its docs page saying why.
   - Candidate results for the maintainer to approve at PR review, each to be confirmed in the paper text:
     - Grosmark: CA1 fields tile the linear track and differ by running direction;
     - Peyrache: the share and tuning width of ADn head-direction cells;
     - Sargolini: grid scores of the session's grid cells;
     - J16: the property the paper's Figure 4 shows, since the file is named `figure04_inputs`;
     - the bat: compact 3-D place fields.
   - Cite only what was read.
10. **Survey harness on the public loaders.**
    - **Adapters.** The survey keys for these datasets (`j16_23020757`, `hc11_achilles`, `grosmark_000044_linear`, `peyrache_mouse32`, `sargolini_000582`, `yartsev_bat_demo`) become thin adapters. Each calls `neurospatial.datasets.load(...)` and converts with a new `datasets/_record.py:from_public(rec) -> Recording`, keeping its `EXPECTATIONS` and `HAZARDS`.
    - **Delete the copies.** Delete the moved loader code from these survey files.
    - **`hc11_achilles`** stays a survey loader. Its source is CRCNS hc-11, which this plan does not move. It still serves as the cross-check.
    - **Re-run.** Re-run the driver on these keys and confirm every operation's summary matches the pre-move results within float tolerance, or explain each difference.
11. **Replace `data/load_bandit_data.py`.**
    - **Example.** `examples/04_real_data_analysis.py` loads J16 with `neurospatial.datasets.load("comrie_2026")`. Its analysis cells keep working and their outputs don't change: compare the notebook outputs before and after.
    - **Delete the old loader and its uses:** `data/load_bandit_data.py`, `tests/test_example_data.py` (its checks move to `tests/datasets/test_comrie_2026.py`), and the CLI mentions in `README.md:254`, `examples/README.md:59` and `data/README.md`.
    - **Update the callers:** `scripts/profile_napari_animation.py`, and the notebook mirror via `docs/sync_notebooks.py`.
12. **Docs.**
    - `docs/user-guide/datasets.md`: what the loaders are; `catalog()`; `load`; reading the summary and the watch-outs; `plot_overview`; moving to pynapple with `to_pynapple` and to NWB with `write_environment`; the cache and `$NEUROSPATIAL_DATA_DIR`; CRCNS credentials (`CRCNS_USERNAME`, `CRCNS_PASSWORD`); meta-analysis iteration over `catalog()`.
    - One page per dataset at `docs/datasets/<dataset>.md`: what was recorded, units and conversions with their basis, the watch-outs, the worked result and its level, and how to cite.
    - Add the pages to the nav (`mkdocs.yml:155-217`). Fences that download are marked `<!-- docs-test: skip needs a download -->`.
    - Add a CHANGELOG entry under "Unreleased", with the reader fixes. Mention the module in `README.md`.
    - Regenerate `docs/llms.txt` (`uv run python scripts/generate_llms_txt.py`) and the API snapshot (`NEUROSPATIAL_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api_snapshot.py`), and review the diff.

## Deliberately not in this plan

- **Datasets beyond these five.** The other showcase datasets (Grieves lattice, labyrinth, Shipley, 000447, Alexander, Kanter, pose) and the rest of the catalogue are planned next, on this module.
- **CRCNS-sourced loaders** (hc-3, hc-11, hc-28, …). The fetch code moves with the credential fix, but no CRCNS dataset ships here.
- **Safe-pickle readers.** They stay in the survey's Gardner and Rosenberg loaders until those datasets move.
- **CI on real data:** deferred by the maintainer. `realdata` tests run locally.
- **Planted-truth sessions** from the loaders (DESIGN.md, method-testing use): a later step.
- **Survey phase 6's cross-cutting work** (simulation ladder, messiness inventory, UX and figure checks). It runs against these loaders afterwards.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/datasets/test_recording.py::test_validation_refuses_bad_shapes` | non-increasing `times`, mismatched `positions`, duplicate unit ids and an inferred `Conversion` without evidence each raise `InputError`, naming the field |
| `test_recording.py::test_summary_keys_read_back` | every `summary()` key is a property of `rec` with that value |
| `test_recording.py::test_arrays_read_only` | `rec.positions.flags.writeable` is False |
| `tests/datasets/test_checks.py::test_each_check_fails_alone` | parametrized: for each check, a synthetic recording that fails only that check raises `DatasetCheckError`, naming it, with the numbers |
| `test_checks.py::test_untracked_sleep_passes` | spikes in an untracked sleep epoch don't fail the clock check |
| `test_checks.py::test_origin_zeros_fail` | 1% of samples at exactly (0, 0) fails, and 0.05% passes |
| `tests/datasets/test_summary.py::test_summary_text` | a synthetic recording prints the six sections in order, "Watch out" last; an unverified one prints "Not yet checked by hand" |
| `tests/datasets/test_overview.py::test_occupancy_panel_is_the_data` | the occupancy panel's array equals `env.occupancy(...)` and its NaNs are masked; a dataset without headings shows "not in this dataset" |
| `tests/datasets/test_overview.py::test_animation_scene_holds_the_data` | `rec.animation()` on a synthetic recording has the occupancy field and position, raster, event and epoch overlays built from `rec`'s arrays; a heading-only recording raises `InputError` naming what is missing |
| `test_overview.py::test_animation_export_frames` | a 2 s widget-backend export of a synthetic recording at 10 fps has 20 frames, headless |
| `test_overview.py::test_animation_show_fastplotlib` (`gui`) | `.show(backend="fastplotlib")` opens on 2-D and 3-D synthetic recordings |
| `tests/datasets/test_video.py::test_overlay_lands_on_tracked_point` | on a synthetic video that draws a bright dot at known cm positions through a known calibration, the exported frame's brightest pixel is within 1 px of the `PositionOverlay` marker, at 3 times |
| `test_video.py::test_frame_times_follow_position_clock` | a synthetic recording whose video timestamps are offset by a constant, with equal frame counts, uses the position times and records an inferred `Conversion` and a watch-out with the offset |
| `test_video.py::test_refuses_without_calibration` | `rec.animation(video=True)` with no calibration source raises `InputError` naming the sources tried; with no video it raises `InputError` saying so |
| `tests/datasets/test_valero_2022.py::test_video_scene` (`realdata`) | the default session's video fetches with its checksum, and a 2 s export over it has the expected frame count; a frame sheet is written for the person who verifies it |
| `tests/datasets/test_catalog.py::test_registry_complete` | every registry session has a SHA-256, a size and a license field; exactly one `default` per dataset |
| `tests/datasets/test_fetch.py` (moved) plus `::test_crcns_http_client_error_hides_credentials` | the cache, checksums and refusals behave as before; a `RemoteDisconnected` traceback with `capture_locals=True` contains neither secret |
| `tests/nwb/...::test_<reader>_<fix>` | one per `io.nwb` reader fix, on a synthetic pynwb file |
| `tests/datasets/test_<dataset>.py::test_loads_default_session` (`realdata`) | `load("<dataset>")` passes the checks and matches the survey's `EXPECTATIONS` for that session (unit count, rate, extents) |
| `tests/datasets/test_grosmark_2016.py::test_osf_matches_dandi` (`realdata`) | Achilles 10252013 from OSF and from DANDI 000044 agree on unit count, spike counts per unit, and positions after resampling, within stated tolerances |
| `tests/datasets/test_worked_results.py` (`realdata`) | each approved result reproduces at its stated level |
| `tests/test_lazy_imports.py`, `tests/test_public_api_snapshot.py`, `tests/test_llms_txt.py` | the module imports lazily without the extra, and the snapshot and llms.txt are current |
| `datasets/tests` | the survey harness passes on the adapters |
| Manual | the driver on the six survey keys matches pre-move results; the example notebook outputs are unchanged; mypy is clean (`uv run mypy src/neurospatial`, also `--platform win32`) |

`realdata` tests download data and are deselected by default. Run them with `uv run --extra datasets pytest -m realdata tests/datasets -n 0`.

## Fixtures

- **Synthetic recordings** (`tests/datasets/conftest.py`): a 2-D square session built with `neurospatial.simulation` (seeded), with one track graph variant and one heading-only variant. Every non-`realdata` test uses these.
- **Synthetic NWB files** built with pynwb in `tmp_path`, to test the reader fixes and the NWB loading path without a network.
- **Real data:** fetched by `realdata` tests into a cache under `$NEUROSPATIAL_DATA_DIR`. No real data is committed (CI is deferred).

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- **Scope.** Every task is implemented as specified, and nothing in "Deliberately not in this plan" was added.
- **Tests:**
  - the validation slice passes, and `realdata` tests are marked;
  - tests aren't trivial: they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured);
  - shared setup is in fixtures, not copy-pasted across tests (`testing-anti-patterns` covers the failure modes in detail).
- **Naming.** Docstrings, test names and module names don't reference this plan.
- **Evidence for what students see:**
  - every `Conversion` with `basis="inferred"` names its evidence;
  - every watch-out entry has a number from the data or a cited source;
  - every worked result cites text that was read.
- **One implementation.** Each loader calls the `io.nwb` readers instead of re-reading NWB itself, and the survey copies of the moved code are deleted.
- **Credentials.** No credential can reach a traceback, log or summary.
