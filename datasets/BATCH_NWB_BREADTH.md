# NWB breadth batch: findings

Nine small NWB recordings from DANDI and EBRAINS, chosen to meet the ways
real files store position, time and units. All nine loaded; one planned
operation column (`rates.object_vector`) was not added because Kanter's files
have no object positions. Gaps G-050 to G-062 are new; G-005, G-010, G-012,
G-014, G-015, G-017, G-032 and G-038 got more evidence. Tables:
`results/matrix.md`, `results/warnings.md`, `results/<key>.json`. This note
does not change `FINDINGS.md`; the last section lists what it should change
there.

## What each recording exercised

| Key | File | What it tested | Outcome |
| --- | --- | --- | --- |
| `rsc_000987` | DANDI 000987, 20.2 MB | Position as three one-axis series with a blank unit; Euler-angle head orientation | Values decide: y is height, x/z metres; `ry` is the yaw. Readers refuse the per-axis layout (G-050). Heading frame not verified (G-051) |
| `ca1ca3_001880` | DANDI 001880, 65.7 MB | Position plus a velocity series in one container; 5.3 % NaN; ripples on a cheeseboard | Loads with `read_position(position_name=)` (G-012). No replay score for an open maze (G-060) |
| `gonzalez_001695` | DANDI 001695, 61.3 MB | `conversion=0.01` on a "centimeters" series | The stored values are cm; the reader, following the spec, returns a 1.1 cm track (G-052) |
| `neurolab_001754` | DANDI 001754, 3.2 MB | Pixels 0-255; position sampled at spike times; 3-D track seen from above | Pixels kept (no calibration). The lost-LED code (0, 255) is set to NaN; occupancy from spike-time samples is flagged by the masking warning (G-053) |
| `valero_000568` | DANDI 000568, 5.8 MB | 110 cm linear track, ripples | Clean. The video timestamps are 22.6 s off the position timestamps, and spatial information picks the position ones (G-054) |
| `aeryjones_000165` | DANDI 000165, 23.3 MB | Linear track multi-unit activity | Repeated timestamps (refused clearly), spikes cover 270 s of 1,504 s tracked (warned), length unit unconfirmable (kept arbitrary) (G-055) |
| `dyads_001845` | DANDI 001845, 1.4 MB | DeepLabCut pose, NaN-heavy keypoints, behavior only | `read_pose` refuses the file's repeated first timestamps; one processed keypoint is pixels labelled cm (G-056) |
| `senzai_000003` | DANDI 000003, streamed from 4.66 GB | Bare `SpatialSeries` labelled metres holding pixels | Read by byte range in 27 s; positions are pixels (G-062); `read_position` misses them (G-032) |
| `kanter_2025` | EBRAINS 10.25493/GTVB-FQR, 29.6 MB | Novel objects; motion-capture head direction; two probes | No object positions in the data. Units table repeats ids across probes; `read_units` refuses (G-057) |

Kanter's bucket is public (`d-885b4936-9345-43bd-880e-eebc19898ded`); the
data descriptor lists position, head direction, speed and spikes, and says
objects were placed "at a pseudorandom location" without recording where.

001845: the plan's 1.4 MB file was kept. Its pose covers only the first 123 s
of a 1,577 s session; the dandiset's usual files (about 9 MB) cover whole
sessions with 5-12 % NaN, so a later batch wanting long pose should take one
of those.

## Matrix rows

Columns that are `–` on all nine (`env.from_graph`, `behavior.turn_sequence`,
the two imaging columns) are left out.

| dataset | exp. | from_samples | occupancy | rates | info | shifts | place | grid | border | hd | decode | roundtrip | segments | laps | replay | nwb_readers | read_pose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rsc_000987 | 6/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | ok | ok | ok | – | err G-015 | – | err G-050 | – |
| ca1ca3_001880 | 6/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok | ok | err G-015 | err G-060 | err G-012 | – |
| gonzalez_001695 | 5/5 | ok | ok | ok ⚠1 | ok | ok ⚠1 | ok | ok | err G-005 | – | ok | ok | ok | err G-015 | err G-038 | wk G-052 | – |
| neurolab_001754 | 7/7 | ok ⚠1 | ok ⚠1 | ok ⚠1 | ok | ok ⚠1 | ok | ok | err G-005 | – | ok ⚠3 | ok ⚠1 | – | err G-015 ⚠1 | – | wk G-053 | – |
| valero_000568 | 5/5 | ok | ok | ok ⚠1 | ok | ok ⚠1 | ok | ok | err G-005 | – | ok | ok | ok | err G-015 | err G-038 | ok ⚠1 | – ⚠1 |
| aeryjones_000165 | 2/4 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok | – | err G-015 | – | wk G-055 ⚠1 | – ⚠1 |
| dyads_001845 | 4/5 | ok | ok | – | – | – | – | – | – | – | – | – | ok | err G-015 | – | err G-012 ⚠2 | err G-056 ⚠2 |
| senzai_000003 | 5/5 | ok ⚠1 | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok ⚠1 | – | err G-015 | – | wk G-032 | – |
| kanter_2025 | 7/7 | ok | ok | ok | ok | ok | ok | ok | err G-005 | ok | ok | ok | – | ok | – | err G-057 | – |

The `⚠` on `io.nwb_readers` and `io.read_pose` are hdmf's deprecation and
schema-conflict notices (third-party). The rows of the earlier datasets have
no `io.read_pose` cell until they are re-run.

## Notable results

- **Labels lie in four of nine files, in four different ways**: a blank unit
  (000987), a conversion factor that contradicts the values (001695), pixels
  labelled metres (000003), pixels labelled cm on one keypoint (001845). In
  each case a magnitude check (span against the apparatus, or the file's own
  speed series) decided; no label was usable alone.
- **neurospatial catches most of what it can see.** Duplicate timestamps
  raise with the first index; spikes that stop before tracking warn ("All 32
  units are silent ... pass spike_window"); irregular, spike-time sampling
  trips the masking warning; a heading in degrees warns on both head-direction
  recordings.
- **Spike times in ms are silent again on 4 of 9** (G-010): when `epochs` are
  passed, the epochs-aware drop count from #265 counts no ms spike (all lie
  outside the epochs), and without `epochs` the clean run's own spike-drop
  warning hides the change. This is a regression of the fix's coverage, not
  of its intent.
- **Missing-as-zero is silent on all four new recordings with NaN** (001880,
  001754, 001845, 000003), and check 1 (origin share) would catch each. A
  file can also code a lost LED as a corner other than the origin (001754's
  (0, 255)), which check 1 does not see.
- **Motion-capture heading frames fail the 0.2 rad rule** on both recordings
  that have them (000987: +0.51 rad, 49 degrees; Kanter: -0.33 rad,
  17 degrees above 20 cm/s), most likely a fixed rigid-body offset. The rule
  needs a decision before it gates anything (G-051).
- **The default rate map can exhaust memory.** A 98,000-bin grid (pixels
  trusted as metres) passed 7 GB for one unit with `diffusion_kde`; `binned`
  took 1 s. The only signal is the environment's memory warning (G-061). The
  driver run that hit it swapped and was killed; the probe is removed from
  the loader.
- **Science sanity**: position decoding beats chance on 001695 (18.5 vs
  39.3 cm), 000568 (32 vs 42 cm), 000003 (38 vs 61 px) and Kanter (21 vs
  57 cm); it is at chance on 000987 (31 vs 29 cm, 37 RSC units) and 001880
  (34 vs 35 cm), and worse than chance on 001754 (103 vs 72 px, position
  only at spikes) and 000165 (17 vs 7, multi-unit channels). 82-88 % of units
  beat every circular shift on 000987, 001880 and 000568.

## Harness limitations found

- `rates.information_vs_shifts` shifts spikes over the whole tracking span,
  but rates use `analysis_epochs`; where the epochs are a small part of the
  tracking (000165, 001754) most shifted trains fall outside them and their
  information is NaN (1 of 32 and 14 of 72 units scored). Its numbers there
  are not meaningful.
- A hazard probe counts the "pixel" units warning (G-058) as a new message
  whenever it rebuilds the environment. The 001754 and 000003 ms-spike rows
  are therefore marked `silent`, not `caught`, against the mechanical rule.
- `io.nwb_readers` compared a reader's "px" with the loader's "pixel" as
  incomparable; fixed in the harness.

## Warning calibration (`studies/warning_calibration.py`, these nine only)

| Dataset | Origin share | Fewest bins | Heading bins | Heading bins, running |
| --- | --- | --- | --- | --- |
| rsc_000987 | 0.000 % | 23 | 36 | 36 |
| ca1ca3_001880 | 0.000 % | 37 | – | – |
| gonzalez_001695 | 0.000 % | **4** | – | – |
| neurolab_001754 | 0.000 % | 49 | – | – |
| valero_000568 | 0.000 % | 8 | – | – |
| aeryjones_000165 | 0.000 % | 7 | – | – |
| dyads_001845 | 0.000 % | 38 | – | – |
| senzai_000003 | 0.000 % | 26 | – | – |
| kanter_2025 | 0.000 % | 41 | 36 | 36 |

No new recording has a linearized coordinate. The shared
`results/studies/warning_calibration.*` files were restored, not overwritten.

## What this changes in FINDINGS.md

1. **Check 2 (fewer than 5 bins along a dimension, accepted) would fire on
   a clean recording.** 001695's linear track recorded in 2-D is 6 cm wide;
   at 2.5 cm bins its narrow axis has 4 bins. The accepted rule needs either a
   lower threshold (the ÷100 hazards give 2-3 bins along *every* axis), or a
   rule on the largest axis, or to fire only when every axis is short.
2. **Check 1 (origin share) gains four recordings of support**: 0 clean
   samples at the origin in all nine, and it would catch the four new silent
   missing-as-zero cases. Mention lost-tracking codes at other corners.
3. **Check 3 (heading coverage)**: 36/36 on both new head-direction
   recordings; no change.
4. **G-010 is not fully fixed**: add the epochs case to the spike-drop check
   (warn when `epochs` are passed and no spike falls inside them).
5. **Proposal 6 (NWB reader coverage)** gets four new layouts to cover:
   per-axis series (G-050), bare `SpatialSeries` outside `Position` (G-062,
   G-032), repeated unit ids across probes (G-057), and pose with repeated
   timestamps (G-056, where the refusal is right).
6. **Performance**: add G-061 (rate-map memory on large grids).

## Not done

- The full driver over all datasets was not run (only these nine), and the
  information-bias and grid-score studies were not re-run.
- Bytes read by the streamed loader were not counted; its wall time was
  27 s.
- No 2-D replay was attempted on 001880 (G-060).
