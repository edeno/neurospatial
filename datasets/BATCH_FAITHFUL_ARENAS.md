# Faithful environments: open arenas and 3-D recordings

The open-arena and 3-D loaders now declare the apparatus their source states
in `geometry["environment"]`, and the driver builds every operation on it;
the samples only place it. Where no source read here states the geometry,
the environment stays built from the samples (`inferred`) and the loader
says why. Each recording was run twice: with the declared environment
(`results/`) and with `--inferred-env` (`results/inferred_env/`, every
environment from the samples, as before this batch). The fidelity study
(`studies/environment_fidelity.py`) now measures the declared environment
and, beside it, the samples-built one against the same outline, bars or room
([results/studies/environment_fidelity.md](results/studies/environment_fidelity.md),
section "Declared against samples-built environments"). New gaps: G-130 to
G-134 in [GAPS.md](GAPS.md); added evidence on G-005, G-015, G-025, G-082,
G-111 and G-114 (resolved in the harness).

## How a stated arena is placed

`_geometry.declared_rectangle` and `declared_circle` take the stated size
from the source and fit only the placement to the tracked samples: centre
and rotation from the samples' minimum rotated rectangle (after trimming
0.1 % of each tail per axis), or the centre of a least-squares circle fit to
their convex hull, unless the source states the centre. The environment is
`Environment.from_polygon` on the placed shape at the loader's `bin_size`,
so its outline is a declared wall. The residual is the samples' own fitted
size over the stated one, minus 1; it is negative when the tracked point
stays off the walls. A residual beyond 15 % is refused
(`_geometry.MAX_SIZE_RESIDUAL`): the stated size, the unit or the session's
apparatus is then wrong. Each loader writes the placement and residuals as
its last `conversions` line.

## Per recording

| Recording | Environment | Source (read) | Placement | Residual (samples' fit vs stated) | Samples outside |
| --- | --- | --- | --- | --- | --- |
| sargolini_000582 | declared: 100 x 100 cm square | NWB `session_description`: "a 1 x 1 meter enclosure" | fitted: centre (0.0, 0.1), rotation 90 deg | 99.5 x 97.1 cm: -0.5 %, -2.9 % | 0.02 % |
| miniscope_002030 | declared: 96 cm circle | DANDI 002030 description: "circular open-field arena (96 cm in diameter)" | fitted centre (57.6, 56.5) | 94.6 cm: -1.5 % | 0.73 % |
| giocomo_000405 | declared: 150 x 130 cm rectangle | Gonzalez & Giocomo 2024, eLife 13:RP85646, Methods "Open-Field" (open-access PDF): "The dimensions of field are 1.3m x 1.5m" | fitted: centre (-2.1, 64.5); the 1.5 m side matched to the samples' longer side | 138 x 120 cm: -8.2 %, -8.0 % | 0 % |
| gardner_2022_grid | declared: 150 x 150 cm square | Gardner et al. 2022, Nature 602:123, Methods "Open-field foraging task" (open-access PDF): "square open-field (OF) arena of size 1.5 x 1.5 m", 50 cm walls | fitted: centre (1.0, -0.1) | 146 x 145 cm: -2.9 %, -3.3 % | 0 % |
| vollan_of | declared: 150 cm circle | Vollan et al. 2025 Methods (Zotero PDF): the one exception to the 150 cm box, "a circular arena 150 cm in diameter", for a dark-room session and the lit session after it in the same arena; the file's session type `open_field_light` | stated centre (0, 0) (`mat_files.md`: "relative to arena center"); fitted centre 0.9 cm off | 140 cm: -6.6 % | 0 % |
| kanter_2025 | declared: 100 x 100 cm square | NWB epochs table: `arena_dimensions` [100, 100, 50] cm | fitted: centre (8.7, -24.8) (the stated frame, "Bottom left of the arena", does not hold) | 97.4 x 96.6 cm: -2.6 %, -3.4 % | 0 % |
| watermaze_10248158 | declared: circle of radius 95 units | `Experiment.json` `arena.bounds` (centre and radius) | stated centre (133.7, 103.5); fitted centre 1.9 units off | 187 units: -1.6 % | 0.34 % |
| hc2_ec013_527 | declared: 180 cm square, 180.8 px at the inferred scale | hc-2 description: ec013.527 180 cm square (Mizuseki et al. 2012 Methods list the lab's 180 x 180 cm open field) | fitted: centre (170, 117) px | 177 x 175 px: -2.1 %, -3.0 % (the longer side is the scale's source, so only the shorter side is a check) | 0.01 % |
| grieves_2020_lattice | declared: the 147 bars as a graph, 343 crossings, 882 segments, all kept | Grieves et al. 2020 Supplementary Methods (open access): 6 x 6 x 6 cubic lattice, 0.97 m a side, tubes 150 mm long and 10 mm across; the file's `lattice_3d` bars and `lattice_type` "diagonal" | the file's bars, in the positions' frame (not fitted) | head a median 6.0 cm from the nearest bar (99th percentile 15.4); `track_width` 30.8 cm = 2 x that 99th percentile | 1.0 % |
| yartsev_bat_demo | declared: the room, every 0.2 m voxel inside `r_lim` (5.8 x 5.2 x 2.3 m) | `NP_Bat_Demo.m` (`r_lim`, "boundaries of the room"); Forli et al. 2025 Methods (Zotero full text): "an indoor flight room (n = 4 bats, 5.6 m x 5.2 m x 2.5 m)" | the file's frame (not fitted) | samples span 5.69 x 6.72 x 1.88 m; outside `r_lim` 3.66 %, outside the paper's size centred on `r_lim` 3.77 % | 3.66 % |
| rsc_000987 | inferred | NWB: "square arena"; Mehrotra, Levenstein et al. 2023 bioRxiv Methods (open-access PDF): square of "removable walls (width: 80 cm, height: 50 cm)" | – | 52.6 x 52.1 cm against 80 cm: -34 %, refused (G-130) | – |
| ca1ca3_001880 | inferred | no size in the file or the DANDI description; the paper (Neuron 2026) could not be read | – | – | – |
| hafting_2008_nird | inferred | archive readme (read): "a linear track (mostly 320 cm)", no width | – | `Environment.bin_at` refuses a track without `track_width`, and no source gives one | – |
| senzai_000003 | inferred | pixels with no stated scale; the open field and the theta maze share one image with no epochs between them | – | – | – |

Vollan's open field needed the most evidence. The paper states a 150 x 150
cm box for its open-field rats and two exceptions: a 1 m square with a
transparent floor, ruled out by the samples' 1.4 m span, and "a circular
arena 150 cm in diameter" for one rat's first session in a dark room,
decoded with "tuning curves from a succeeding recording session with room
lights on (same arena)" (Extended Data Fig. 9k). The file's one session is
of type `open_field_light`. Of the 18 other open-field files downloaded for
this survey, 16 are `open_field`, whose samples fill a square (convex hull
0.98-1.00 of its minimum rectangle, one at 0.92), and 2 are
`open_field_darkness`, whose samples are round (0.79). This session's
samples are round too (0.81 of their minimum rectangle, 0.94 of their
minimum circle), so it is taken to be the lit session in the circular
arena. The paper names no rat or file: the identification is an inference
from the session type and the samples' shape (G-114).

The bat's room is `r_lim`, not the paper's size, because `r_lim` is stated
in the positions' frame and the paper gives only a size; centred on `r_lim`,
the paper's 5.6 m would cut the samples at the x walls (they reach x =
+-2.85 m). The lattice's track width is not a stated value: the tubes are
10 mm across, but the tracked point is the head, held a median 6 cm from
the bar, so a 10 mm track would place almost no sample. Twice the 99th
percentile of the head's distance from a bar (30.8 cm) places 99 %.

## What changed between declared and inferred

Each cell: declared / inferred (`results/` / `results/inferred_env/`).

| Recording | Bins | Visited bins | Spatial information (median bits/spike) | Border score (median) | Egocentric boundary: peak in last ring, preferred distance (median) | Decoding error (median / chance) | Laps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sargolini_000582 | 1601 / 1399 | 87 % / 100 % | 0.95 / 1.01 | -0.43 / refused (G-005) | – (one LED) | 64.3 / 56.2 cm; 66.0 / 56.6 cm | refused / 1 |
| miniscope_002030 | 1153 / 455 | 36 % / 100 % | – (imaging) | – | – | – | refused / 1 |
| giocomo_000405 | 3121 / 2076 | 67 % / 100 % | – (binned rates) | – | – | – | refused / 0 |
| gardner_2022_grid | 3601 / 3463 | 97 % / 100 % | 0.479 / 0.479 | -0.23 / refused | 13 %, 12.5 cm / 10 %, 12.5 cm | 7.07 / 78.1 cm; 7.07 / 78.2 cm | refused / refused |
| vollan_of | 2828 / 2041 | 72 % / 100 % | 0.38 / 0.44 | -1 / refused (G-131) | 5 %, 7.5 cm / 48 %, 42.5 cm | 8.54 / 64.9 cm; 8.54 / 66.0 cm | refused / 0 |
| kanter_2025 | 1601 / 1049 | 65 % / 100 % | 0.151 / 0.161 | 0.08 / refused | 20 %, 37.5 cm / 38 %, 42.5 cm | 21.2 / 60.0 cm; 21.2 / 60.6 cm | refused / 1 |
| watermaze_10248158 | 1124 / 1144 | 99 % / 100 % | – (behaviour only) | – | – | – | refused / refused |
| hc2_ec013_527 | 5329 / 3210 | 60 % / 100 % | 1.44 / 1.66 | -1 / refused (G-131) | refused (pixels, G-133) | 74.1 / 105 px; 83.4 / 110 px | refused / 1 |
| grieves_2020_lattice | 4410 / 8193 | 57 % / 100 % | 0.86 / 0.78 | refused (3-D) | – | 97.0 / 97.0 cm; 82.7 / 76.8 cm | 1 (G-134) / refused |
| yartsev_bat_demo | 9048 / 1352 | 13 % / 89 % | 3.50 / 3.47 | refused (3-D) | – | 2.12 / 2.55 m; 2.18 / 2.64 m | refused / refused |

**Occupancy and rate maps.** A declared arena keeps the bins the animal never
visited, as NaN: 3 % of Gardner's (143 min, walls reached), 13 % of
000582's, 28 % of Vollan's, 35 % of Kanter's, 40 % of hc-2's, 87 % of the
bat's room in flight. Occupancy totals change by under 1 % (a few samples
outside the outline are dropped: 0.2 s of 600 s on 000582). The maps no
longer stop at the edge of the visited bins, so a map now shows where the
animal did not go.

**Spatial information** falls slightly where the declared arena adds
unvisited interior bins next to visited ones (000582 1.01 -> 0.95, Vollan
0.44 -> 0.38, hc-2 1.66 -> 1.44 bits/spike) and is unchanged where the
samples already filled the box (Gardner). The share of units beating every
circular shift moves by under 6 points either way. On the lattice the bar
graph raises it (0.78 -> 0.86). Place-screen counts change by at most 13
units of 644 (Vollan 250 -> 237).

**Grid scores** of grid cells are unchanged: Gardner median 0.952 against
0.949 (all 470 finite both ways, correlation 1.0 across units), Vollan 0.989
correlation. Units that are not grid cells lose a score more often: Kanter
235 -> 144 finite (the lost ones scored a median -0.37 and at most 0.38 on
the samples-built environment), Vollan 611 -> 593 (at most -0.02), 000582 6
-> 5 (the lost unit scored 0.006). Why the autocorrelogram leaves these
scores NaN on the declared grid was not traced.

**Border scores** now run on every declared 2-D arena (refused before,
G-005). They are sensible where the tracked point reaches the walls (Gardner
median -0.23, Kanter 0.08, 000582 -0.43) and -1 for most cells where it
does not: 407 of 619 on Vollan's open field and 27 of 36 on hc-2, where
only 1.4 % and 4.9 % of the wall is next to a visited bin (G-131). On the
samples-built environment with `boundary_source="analysis"` the edges of
unvisited holes would be walls instead, and 86-100 % of the units score
above 0.5 there (Vollan, Kanter, hc-2).

**Egocentric boundary maps** take the declared outline as the walls. On
Vollan's open field the share of units peaking in the outermost ring falls
from 48 % to 5 % and the median preferred distance from 42.5 to 7.5 cm; on
Kanter from 38 % to 20 %. The samples-built environment's 48 % came from the
edges of unvisited holes standing in for walls (G-082). Gardner, whose
samples leave no holes, barely moves (10 % -> 13 %).

**Decoding** is unchanged in the arenas the animal covered (Gardner, Vollan,
Kanter: identical median errors; chance within 1 cm) and better on hc-2
(83.4 -> 74.1 px; chance 110 -> 105 px). On the lattice the bar graph's
geodesic errors are larger than the voxels' and equal to chance (97.0 cm
against 97.0; the 10th-90th percentiles of decoded and chance errors are
45-162 cm both); the 24 CA1 units decode at chance on both (voxels: 82.7
against 76.8 cm).

**Laps.** `detect_laps` now refuses every declared open arena ("no loop"),
where the samples-built environments gave 0-1 spurious laps (G-015); the
holes were its loops. On the lattice's bar graph it accepts the cube faces
as loops and reports one lap of unknown direction (G-134).

**Hazards.** A declared environment changes what catches a unit error. Metres
labelled cm used to be caught by `from_samples` (bin size at least the data
extent) and by the rate maps (bandwidth spans the environment); with the
stated box declared, the positions shrink into its centre, occupancy piles
into 1-4 bins and nothing warns (G-132). Where the shrunk positions leave
the outline (002030's circle, the bat's room, the lattice), occupancy raises
and says to check the coordinates. Untracked frames written as (0, 0) or
(-1, -1) fall outside the declared hc-2 square and the water-maze pool and
are dropped, so those hazards no longer move the results. Verdicts are in
`verdicts.json`; `warnings.md` has no unreviewed row.

## Fidelity of the declared environments

Every declared environment matches its source in the fidelity study: the
stated outline is fully covered (100 % against 37-98 % for the samples-built
environments), its outermost bins sit within 0.4-0.8 cm of the wall (against
0.3-6.3 cm, median 1.9), and no interior holes remain (000582 had 72, Vollan 131, hc-2
216). On the lattice, the path through the declared bins is a median 0.95 x
the distance along the bars (voxels 0.77); for pairs on parallel bars of one
face it is 30.4 cm against 31.1 cm along the bars (voxels 18.2 cm). The
bat's room has no voxel outside `r_lim` (4.4 % before) and both feeders in
the environment. The study's "far links" and "sparse bins" are now large for
declared environments (for example 53 % and 68 % on 002030): they count
declared bins the animal never visited, which is the point of declaring
them, not air or a wall.

## What would make the declared path the easy one in neurospatial

- A border score that works where the tracked point cannot reach the wall:
  an inward lookup like BNT's, or a stated offset of the tracking point, and
  a warning when `observed_wall_fraction` is small (G-131).
- A units check that does not depend on the environment being built from the
  same samples: warn when samples cover a tiny share of a declared
  environment (G-132).
- `from_graph` for climbing structures (G-111), with `detect_laps` refusing
  3-D graphs (G-134).
- A documented way to declare a 3-D room or box (here `from_samples` with
  `dimension_ranges` and `infer_active_bins=False`, which still needs a
  positions array).

## What could not be verified

The size of 000987's square (the published methods were not available; the
preprint's 80 cm walls do not fit the samples), 001880's cheeseboard, any
scale for 000003's pixels, the width of Hafting's track, how far a rat's
head reaches from a lattice bar, and which Vollan rat and file the paper's
circular arena belongs to (inferred from the session type). Whether the
hc-2 field is walled or elevated was not checked; border scores treat a wall
and a drop edge alike.
