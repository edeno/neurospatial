# Archive batch: findings

Eight recordings from Zenodo, Figshare, the UCL Research Data Repository and
NIRD, none of them NWB, added to the survey. This note says what each one
exercised, what it found and what it changes in [FINDINGS.md](FINDINGS.md).
The evidence is in [GAPS.md](GAPS.md) (G-070 to G-082, and "More evidence
(archive batch)" lines on earlier entries) and in `results/`.

## What each dataset exercised

| Key | Format work | Exercised | Status |
| --- | --- | --- | --- |
| `grieves_2020_lattice` | Deflate64 member of an 11.2 GB zip by byte range (`inflate64`, CRC-checked); MATLAB v7.3 with `h5py` | Second 3-D recording: rates, shifts and decoding in 3-D; the information-bias study | loaded; 24 units, 3642 s |
| `watermaze_10248158` | EthoVision/Rtrack JSON | Behavior only in a circular pool; no length unit (G-077) | loaded; 30 trials of one mouse |
| `olafsdottir_5566548` | Axona `.pos`, tetrode and Tint `.cut` files parsed with NumPy, 34 members of a 9.9 GB zip | Z-track, hippocampus and deep MEC tetrodes; pixel calibration from the header | loaded; 57 units, 2276 s |
| `takigawa_7114733` | MATLAB v5 structs | Two novel linear tracks in one session, each converted with its own calibration (G-076) | loaded; 109 units, 12661 s |
| `hafting_2008_nird` | NIRD archive: resource URLs from the dataset page's table of contents, S3 redirect, members by byte range | Moser-lab grid cells on a 320 cm track; one-edge track graph | loaded; 3 units, 1200 s |
| `j16_23020757` | `.npz` read with `allow_pickle=False` | The authors' track graph for `Environment.from_graph`; Unix-epoch clock | loaded; 203 units, 1419 s |
| `agarwal_24757638` | MATLAB v5, 1250 Hz tracking kept at 50 Hz | 3-arm maze with CA1 and mPFC units; a sentinel for untracked samples | loaded; 439 units, 3795 s |
| `shipley_30902972` | Kilosort `.npy` (no objects) + MATLAB v5, members of a 4.4 GB zip | Radial maze and sleep; a position clock rebuilt from the file's spike-to-sample map (G-075) | loaded; 71 units, 4862 s |
| `zhang_2022_ebc` | MATLAB `table` files read with `mat-io` | — | dropped: spike times and the authors' maps, but no per-sample positions or head direction (G-070) |

Every loader fetches by pinned SHA-256, range-checks its positions against
the apparatus (`checked_span`; per track on Takigawa; the pool's diameter on
the water maze) and records each conversion. The Axona format
follows Axona's "dacqUSB Data File Formats" (archived copy of
space-memory-navigation.org/DacqUSBFileFormats.pdf); the `.cut` layout is not
in that document and is described from the files.

## The matrix rows

| dataset | exp. | env | occ | rates | SI | shifts | place | grid | border | hd | ego | decode | nwb | graph | segments | laps | turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agarwal_24757638 | 5/5 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | – | ok | err G-015 | – |
| grieves_2020_lattice | 5/5 | ok | ok | ok | ok | ok | ok | err G-024 | err G-024 | – | – | ok | ok | – | ok | err G-015 | – |
| hafting_2008_nird | 5/5 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | ok | ok | err G-016 | err G-016 |
| j16_23020757 | 5/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | ok | ok | err G-016 | err G-016 |
| olafsdottir_5566548 | 6/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | – | ok | err G-015 | – |
| shipley_30902972 | 6/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | – | ok | err G-015 | – |
| takigawa_7114733 | 4/5 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | – | ok | ok | – | ok | err G-015 | – |
| watermaze_10248158 | 6/6 | ok | ok | – | – | – | – | – | – | – | – | – | – | – | – | err G-015 | – |

Replay, NWB readers and imaging are `–` on all eight. No neurospatial
warning fired on any clean recording. The two failed expectations (Takigawa's
combined track span, J16's heading) were wrong as registered (G-071).

## Where the numbers come from

None of `results/studies/*` is committed from this batch; the studies are
regenerated once for every batch. The numbers below come from:

| Numbers | Source |
| --- | --- |
| Matrix rows, decoding, shuffle test, hazards | `run.py --datasets <the eight keys>` at this branch's head (`results/<key>.json`) |
| Egocentric boundary maps on earlier recordings | `studies/egocentric_boundary_survey.py gardner_2022_grid vollan_lt vollan_of giocomo_001701` after merging main (it reproduces the earlier hand-run exactly) |
| Warning calibration | `studies/warning_calibration.py <the eight keys>` (Takigawa, Olafsdottir and the water maze rerun after their loader fixes) |
| J16 heading coverage | a one-off call of `warning_calibration.heading_coverage` on `params["head_orientation"]`, not a script |
| Information bias | `studies/information_bias_null.py yartsev_bat_demo grieves_2020_lattice` (Grieves is now in its `DATASETS`), run before this branch merged main |
| Grieves mm-as-cm probe (G-079) | a one-off manual run, not a script |
| Z-track decoding diagnostics (errors by speed, spike shifts) | one-off manual runs, not scripts |

## Notable results

- **Decoding** (median error / chance, cm, both geodesic, chance shuffled
  within each connected part, `dt` 0.25 s): Agarwal 5.7 / 192, J16
  5.7 / 113, Takigawa 13.7 / 68.3 (a global shuffle would put 51 % of pairs
  on the other, disconnected track), Shipley 51.9 / 93.9, Olafsdottir
  115 / 142, Grieves 82.7 / 76.8 (3-D, at chance as on the bat, G-023),
  Hafting 157 / 130 (3 units). On the Z-track a quarter of time bins are
  within 12 cm; bins with the rat running faster than 15 cm/s have a median
  of 55 cm. Why the rest are far off was not resolved (no clock offset
  found: spatial information is flat for spike shifts of 0.05–0.4 s).
- **Shuffle test** (`rates.information_vs_shifts`, 4 shifts): the share of
  units beating every shift is 75 % (Grieves) to 100 % (Hafting); 86 % on
  J16, 90 % on Agarwal, 94 % on Takigawa.
- **Hazards**: every spike-times-in-ms probe is caught (the drop warning or
  the "all units silent" warning). Length off by 100 is caught at the
  environment but `segment_by_velocity` goes to 0 segments silently (G-014).
  New silent hazards: pixels taken for cm (G-080) and a position clock
  0.17 % slow (G-081). Grieves' mm read as cm warns only through the
  large-grid warning, whose suggested bin size is a fixed 2.0 (G-079); in a
  first driver run its rate maps ran over an hour at 18 GB, so it is
  documented in the loader and was probed once by hand (those numbers come
  from that one-off run, not from a committed script).
- **Harness fix**: the decoding chance level used straight-line distance
  while `error_against` is geodesic by default, and now shuffles within each
  connected part (G-078). Every recording's chance number changes when it
  is rerun, not only the mazes': geodesic distances are quantized to bin
  centres and 8-connected paths run up to about 8 % longer than the
  straight line. The earlier batches' result files still hold the old
  numbers.

## The egocentric boundary operation

`rates.egocentric_boundary` runs `compute_boundary_vector_rates(...,
direction_frame="egocentric", boundary_source="analysis")` on every recording
with 2-D positions and head direction. No new recording has a usable head
direction: J16's is 13° off the direction of movement, beyond the 0.2 rad
frame check (G-072); Agarwal's is not a heading (G-073); Grieves' azimuth is
in the camera's frame, not the 3-D frame (G-074); the others have one LED. On
the earlier recordings, from `studies/egocentric_boundary_survey.py` (which
leaves the driver's results files alone; run here on these four keys):

| Dataset | Units | Median preferred distance (cm) | Peak in the last ring (45–50 cm) | Peak on the left | Median distance information |
| --- | --- | --- | --- | --- | --- |
| gardner_2022_grid | 483 | 12.5 | 10 % | 43 % | 0.023 |
| vollan_of | 644 | 42.5 | 48 % | 45 % | 0.021 |
| vollan_lt | 389 | 42.5 | 44 % | 70 % | 0.054 |
| giocomo_001701 | 320 | 42.5 | 33 % | 48 % | 0.029 |

The function's docstring says untuned simulated cells peak in the last ring
16–31 % of the time; without an egocentric boundary cell test (G-082) these
maps cannot be screened. On the Vollan files the function warns that the
times look boolean-masked, which is correct (G-040). The validation test
(`tests/test_operations_egocentric.py`) recovers a simulated cell with a wall
20 cm to its left within one ring and one 30° sector, and moves it to the
right when the frame is mirrored.

## Warning calibration (new recordings)

| Dataset | Origin share | Fewest bins | Heading bins |
| --- | --- | --- | --- |
| watermaze_10248158 | 0.000 % | 39 | – |
| hafting_2008_nird | 0.000 % | 11 | – |
| takigawa_7114733 | 0.000 % | 57 | – |
| olafsdottir_5566548 | 0.000 % | 35 | – |
| shipley_30902972 | 0.000 % | 60 | – |
| grieves_2020_lattice | 0.000 % | 38 | – |
| agarwal_24757638 | 0.000 % | 27 | – |
| j16_23020757 | 0.000 % (linearized 0.000 %) | 49 | – (its `head_orientation`, not used: 36/36, running 36/36) |

Check 1 fires on no clean recording, and would have caught all 18 silent
missing-as-zero probes in this batch (four recordings, 0.20–85 % of samples
untracked). Check 2's clean minimum is 11 bins (across Hafting's 36 cm-wide
track), above the threshold of 5.

## Information-bias study with Grieves

Grieves' lattice is sparse 3-D occupancy like the bat's, though less so
(8182 visited bins, 24 % under 0.1 s, median 0.23 s, against the bat's 70 %
and 0.07 s).

| | Grieves | Bat (rerun, unchanged) |
| --- | --- | --- |
| Poisson: information / bias at 0.5, 2, 8 Hz | 0.94, 1.00, 1.00 | NaN, NaN, 0.92 |
| Bursty: information / bias at 0.5, 2, 8 Hz | 3.02, 4.11, 4.65 | NaN, NaN, 2.24 |
| Real units with a defined bias | 17 of 24 | 39 |
| Median information / bias / shift-null mean (bits/spike) | 0.579 / 0.169 / 0.392 | 0.486 / 0.370 / 0.672 |
| Median shift null / bias | 2.51 | 2.19 |
| Pass the bias rule but not the shifts' 95th percentile | 6 % | 44 % |

The second 3-D dataset supports the conclusion in proposal 4: the bias
estimate is right for an untuned Poisson cell on sparse occupancy, and the
gap to the shift null is temporal structure. Grieves' cells carry enough
information that the difference changes only one verdict.

## What this changes in FINDINGS.md

- **Proposal 4 (information bias):** a second sparse 3-D recording agrees;
  the recommendation (document the bias as a Poisson null, no sparse
  occupancy warning) stands.
- **Warning checks 1 and 2:** 0 false alarms on eight more clean recordings;
  check 1 catches every missing-as-zero probe. Neither check sees a pixel
  calibration missed (G-080): it inflates the extent instead of shrinking it.
- **Head-direction frame check:** the 0.2 rad limit rejects J16's heading,
  which is offset by a constant 13° but otherwise a heading (G-072). Whether
  the check should separate frame errors (about 90° or 180°) from small
  offsets is a question for the maintainer.
- **New candidate proposals:** a shuffle verdict for boundary-vector and
  egocentric boundary cells plus `min_speed=` on the boundary-vector entry
  points (G-082); a computed example in the large-grid warning (G-079); show
  the metric with decoding errors (G-078).
- **Loader code in neurospatial:** the Axona reader is about 80 lines and the
  Deflate64 path about 50; neither changes the recommendation that format
  readers stay in loaders.

## Not done

- The full driver was not rerun on the earlier datasets, so their result
  files have no `rates.egocentric_boundary` cell yet and their decoding
  chance levels still use straight-line distance with a global shuffle.
- Zhang 2022's All_TDcells.mat was not opened (the other two tables had no
  tracking).
- The cause of the poor Z-track decoding was not found.
