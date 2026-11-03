# Appendix — Dataset access

[← back to PLAN.md](PLAN.md)

Facts marked **verified** were checked on 2026-11-02 while planning. Everything
else comes from the catalog,
`docs/research/reports/Open spatial neuroscience datasets.md` (cited by
dataset name), and must be checked before the loader's expectations are
written. Keys are the proposed `DATASET["key"]` values.

## Phase 1

### `gardner_2022_grid` — Gardner et al. 2022, toroidal grid modules

- Figshare article 16764508, CC0 (**verified**, Figshare API).
- One zip, `Toroidal_topology_grid_cell_data.zip`, 319,902,753 bytes,
  `https://ndownloader.figshare.com/files/35078602` (**verified**). Members
  (**verified** by byte-range listing, compress type 8 = Deflate, so plain
  `zipfile` reads them):

  | Member | MB |
  | --- | --- |
  | `rat_q_grid_modules_1_2.npz` | 103.3 |
  | `rat_q_sessions.txt` | <0.1 |
  | `rat_r_day1_grid_modules_1_2_3.npz` | 168.9 |
  | `rat_r_day1_sessions.txt` | <0.1 |
  | `rat_r_day2_grid_modules_1_2_3.npz` | 150.9 |
  | `rat_r_day2_sessions.txt` | <0.1 |
  | `rat_s_grid_modules_1.npz` | 69.0 |
  | `rat_s_sessions.txt` | <0.1 |

- Fetch members with `_fetch.fetch_zip_member` (`datasets/_fetch.py:90`).
  Start with `rat_r_day1` (three modules) for the grid check; `rat_s` is the
  smallest for the smoke test.
- Catalog: arrays `spikes_mod<N>`, `t`, `x`, `y`; units and sampling rate
  unverified; `.npz` needs `allow_pickle=True` (see
  [designs.md#safe-pickles](designs.md#safe-pickles)). The `_sessions.txt`
  files presumably give the open-field / sleep windows: read them.

### `vollan_lt` and `vollan_of` — Vollan et al. 2024/25, left-right theta sweeps

- EBRAINS dataset DOI 10.25493/R5FR-EDG, CC BY-SA 4.0, 19 rats, Neuropixels
  MEC and/or hippocampus (**verified**, DataCite).
- Public bucket, no login: list with
  `https://data-proxy.ebrains.eu/api/v1/buckets/d-4080b78d-edc5-4ae4-8144-7f6de79930ea?limit=200`;
  download with `<bucket>/<name>?redirect=true`, which redirects to
  `rgw.cscs.ch` and serves byte ranges (HTTP 206 **verified**; `HEAD` is
  refused with 405, so do not use it to probe).
- Files (**verified**): `sharing_v4/navigation/lt/` 7 linear-track sessions,
  110.6–494.6 MB (smallest `27764_1.mat`); `of/` 32 open-field sessions,
  101.3–851.8 MB (smallest `25954_1.mat`); `mmaze/`, `ww/` (wagon wheel),
  `of_novel/`; `sleep/` 9 files, 49–211 MB; `documentation/mat_files.md`.
- Layout (**verified** from `mat_files.md`): one struct `Dsession` with `t`
  (s since acquisition start), `x`, `y` (m, relative to arena centre), `z` (m,
  above floor), `hd` ("2D head direction (azimuth)", unit and zero direction
  not stated), `speed` (m/s), `dt` 0.01 s, `units.<area>(i).spikeTimes`,
  `acorrClu` / `acorrCluId` (grid-module membership), `unitAcorrClus` with
  `medianGridScore` per module. **All time series are speed-filtered: samples
  below 5 cm/s are discarded**, so time has gaps; several sessions of one type
  may be concatenated. Both are data quirks for the loader to record.
- Which open-field file has the most grid units: unknown. Read `unitAcorrClus`
  of a few small `of/` files before choosing (overview open question 2).

### `grosmark_000044_circular` and `grosmark_000044_linear` — Grosmark & Buzsáki 2016

- DANDI 000044 version `0.261005.2101`, CC-BY-4.0, 8 assets of 5.2–9.2 GB
  (**verified**). Read by byte range
  ([designs.md#streamed-nwb](designs.md#streamed-nwb)).
- Track per file (**verified** from `processing/behavior` names):

  | Asset id | Session | Track | Units |
  | --- | --- | --- | --- |
  | `080a847a-5211-4101-bc05-fd45b7e77dbd` | Achilles 11012013 | circular | 104 |
  | `d6b8092b-94e2-4e37-8660-86f9e856aec8` | Cicero 09102014 | circular | 105 |
  | `402f78e1-9e7d-486c-8822-93d538ed6ccc` | Gatsby 08282013 | circular | 51 |
  | `c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d` | Achilles 10252013 | 1.6 m linear | 137 |
  | `185b8a36-d671-4688-ba05-9e89a902c486` | Buddy 06272013 | 1.6 m linear | — |
  | `97252767-5e90-45cf-a1b8-8cbff2f2a3a3` | Cicero 09012014 | 1.6 m linear | 73 |
  | `93569d6c-781f-4422-938e-e935a62863de` | Gatsby 08022013 | 1.6 m linear | — |
  | `d9a1e010-af4a-4fba-812d-66e2d1a71e35` | Cicero 09172014 | 2 m linear | — |

  "—": not checked. Use Achilles 11012013 (circular) and Achilles 10252013 (linear; the hc-11
  session also on OSF, which phase 4 cross-checks).
- Each file has `intervals/epochs` with `PREEpoch`, `MazeEpoch`, `POSTEpoch`
  (**verified** on three files; e.g. Buddy: maze 10,717.9–13,046.0 s), and
  `processing/behavior` with `<track>Position`, `<track>LinearizedPosition`
  and `states`.
- Catalog: x, y in m at 39.06 Hz, linearized 0–1.6 m, 93 % NaN outside the
  maze epoch. Ripple events: unverified; walk the file.
- SHA-256 to pin: from `GET /api/assets/<id>/` → `digest["dandi:sha2-256"]`
  (Buddy's is `a88825bef47991f41f63ed8ebf5a7a6b392ce235e9e0ea22c22088c11fa3b20e`,
  **verified**).

### `jadhav_000447` — Shin & Jadhav W-track

- DANDI 000447 version `0.250724.2224`, CC-BY-4.0, 5 assets (**verified**);
  smallest `sub-JDS-NFN-AM2/sub-JDS-NFN-AM2_behavior+ecephys.nwb`, asset
  `b048bd0a-48a4-4d19-ad96-fd879666bca5`, 5.05 GB. Read by byte range.
- Catalog: position columns (x, y, velocity) under unit
  `centimeters; centimeters/second` at about 32.9 Hz; W-track
  novel–familiar–novel, 104 trials with wells; 57 units. Head direction not
  listed. The velocity column and the compound unit string are the quirks
  `io.nwb_readers` should meet.

## Phase 2

All from the catalog; access and layout unverified. Pick the smallest file
that has units and position; walk the whole file before calling anything
missing.

| Key | Dataset | Why it is in the batch |
| --- | --- | --- |
| `rsc_000987` | DANDI 000987 RSC | Head direction plus 2-D in square and circular arenas; blank position units |
| `ca1ca3_001880` | DANDI 001880, CC0 | Place–reward task, open box + maze, ripples, 5 % NaN position |
| `gonzalez_001695` | DANDI 001695 | 430 units, `conversion` 0.01 (cm stored as m-scaled values) |
| `neurolab_001754` | DANDI 001754 Neurolab | Pixel coordinates 0–255; Escher staircase 3-D track; one file sampled at spike times |
| `valero_000568` | DANDI 000568, "no-raw-data" files 6–8 MB | 110 cm linear track; ripples |
| `aeryjones_000165` | DANDI 000165 | Linear track and home cage in one session |
| `dyads_001845` | DANDI 001845 rat dyads, CC0 | Pose keypoints with 28–37 % NaN; two animals |
| `senzai_000003` | DANDI 000003 | Position labelled "meters" with pixel-like values 117–500 |
| `kanter_2025` | EBRAINS GTVB-FQR, NWB, CC BY-SA | Novel objects (object positions unverified), lap maze, LEC/MEC/CA1 |

## Phase 3

All from the catalog; access and layout unverified.

| Key | Dataset | Format | Why |
| --- | --- | --- | --- |
| `zhang_2022_ebc` | Figshare 21647423, CC BY | MATLAB `sdata` | Egocentric boundary cells; whether raw positions and spikes are included is unverified — open it first |
| `grieves_2020_lattice` | UCL RDR, CC0, 11.2 GB zip | Deflate64 zip | Second 3-D dataset (lattice); needs `inflate64` |
| `watermaze_10248158` | Figshare 10248158, CC BY | EthoVision JSON | Circular arena behavior only |
| `olafsdottir_5566548` | Zenodo 5566548, CC BY | Axona `.pos` / `.cut` | Z-track, rest, 1 m square; Axona format |
| `takigawa_7114733` | Zenodo 7114733, CC0 | MATLAB | Two-context linear tracks |
| `hafting_2008_nird` | NIRD DataCite 10.11582/2017.00020 | MATLAB | Moser-lab grid recordings; downloads untested |
| `j16_23020757` | Zenodo 23020757, CC BY | `.npz` without pickles | Track graph included; 203 units |
| `agarwal_24757638` | Figshare 24757638, CC BY | MATLAB | 3-arm maze with wheel delay |
| `shipley_30902972` | UCL RDR 30902972, CC BY | NPY + MATLAB | 8-arm radial maze |

## Phase 4

CRCNS sets need an account; credentials are `CRCNS_USERNAME` and
`CRCNS_PASSWORD` in `~/.zshrc`, used by name and never printed. Access
pattern, file lists and formats unverified.

| Key | Set | Why |
| --- | --- | --- |
| `hc11_achilles` | hc-11, Achilles_10252013 | Same session as `grosmark_000044_linear`: cross-check the two formats |
| `hc3_<session>` | hc-3, two or three sessions | Two LEDs (head direction), linear track and open field, `.whl` |
| `hc2_<session>` | hc-2 | Elevated square open field with head direction |
| `hc13_<session>` | hc-13 | `rawpos` stores 0 for untracked frames: the origin-share hazard in a real file |
| `hc28_<session>` | hc-28 (or hc-6) | W-track, `pos` + `cmperpix` |
| `hc31_<session>` | hc-31 | 48 m linear track: extreme length |
| `hc18_<session>` | hc-18 | Treadmill on a model train |
