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
G-135 in [GAPS.md](GAPS.md); added evidence on G-005, G-015, G-025, G-082,
G-111 and G-114 (resolved in the harness).

## How a stated arena is placed

`_geometry.declared_rectangle` and `declared_circle` take the stated size
from the source and fit only the placement to the tracked samples: centre
and rotation from the samples' minimum rotated rectangle, or the centre of
their minimum bounding circle (both after trimming 0.1 % of each tail per
axis), unless the source states the centre. The environment is
`Environment.from_polygon` on the placed shape at the loader's `bin_size`,
so its outline is a declared wall (a drop edge for hc-2's elevated
platform). The residual is the samples' own fitted size over the stated
one, minus 1; it is negative when the tracked point stays off the walls.
The loader refuses the stated size when the samples fall more than 15 %
short of it (`_geometry.MAX_SIZE_RESIDUAL`; animals often keep off walls)
or reach past it by more than one bin over the side (half a bin at each
wall, which the grid cannot resolve), and a stated centre more than 15 % of
the radius from the samples' centre (`MAX_CENTRE_OFFSET`, a frame or unit
error). Each loader writes the placement and residuals as its last
`conversions` line and keeps them in `geometry["environment_fit"]`.

## Per recording

| Recording | Environment | Source (read) | Placement | Residual (samples' fit vs stated) | Samples outside |
| --- | --- | --- | --- | --- | --- |
| sargolini_000582 | declared: 100 x 100 cm square | NWB `session_description`: "a 1 x 1 meter enclosure" | fitted: centre (0.3, -0.1), rotation 0 deg | 99.9 x 97.4 cm: -0.1 %, -2.6 % | 0.02 % |
| miniscope_002030 | declared: 96 cm circle | DANDI 002030 description: "circular open-field arena (96 cm in diameter)" | fitted centre (57.9, 56.2) | 96.0 cm: +0.0 % | 0.03 % |
| giocomo_000405 | declared: 150 x 130 cm rectangle | Gonzalez & Giocomo 2024, eLife 13:RP85646, Methods "Open-Field" (open-access PDF): "The dimensions of field are 1.3m x 1.5m" | fitted: centre (-1.2, 64.2); the 1.5 m side matched to the samples' longer side | 140 x 122 cm: -6.9 %, -6.3 % | 0 % |
| gardner_2022_grid | declared: 150 x 150 cm square | Gardner et al. 2022, Nature 602:123, Methods "Open-field foraging task" (open-access PDF): "square open-field (OF) arena of size 1.5 x 1.5 m", 50 cm walls | fitted: centre (0.7, 0.2) | 147 x 146 cm: -1.9 %, -2.4 % | 0 % |
| vollan_of | declared: 150 cm circle | Vollan et al. 2025 Methods (Zotero PDF): the one exception to the 150 cm box, "a circular arena 150 cm in diameter", for a dark-room session and the lit session after it in the same arena; the file's session type `open_field_light` | stated centre (0, 0) (`mat_files.md`: "relative to arena center"); fitted centre 1.66 cm off | 144 cm: -4.0 % | 0 % |
| kanter_2025 | declared: 100 x 100 cm square | NWB epochs table: `arena_dimensions` [100, 100, 50] cm | fitted: centre (8.7, -24.6) (the stated frame, "Bottom left of the arena", does not hold) | 97.9 x 97.7 cm: -2.1 %, -2.3 % | 0 % |
| watermaze_10248158 | declared: circle of radius 95 units | `Experiment.json` `arena.bounds` (centre and radius) | stated centre (133.7, 103.5); fitted centre 0.9 units off | 190 units: +0.2 % | 0.34 % |
| hc2_ec013_527 | declared: 180 cm square with a drop edge, 180.8 px at the inferred scale | `crcns-hc2-data-description.pdf` (read): "an elevated square platform (180 cm x 180 cm, or 120 cm x 120 cm)", ec013.527 180 cm in its session table | fitted: centre (169, 116) px | 181 x 179 px: +0.0 %, -0.8 % (the longer side is the scale's source, so only the shorter side is a check) | 0.01 % |
| grieves_2020_lattice | declared: the 147 bars as a graph, 343 crossings, 882 segments, all kept | Grieves et al. 2020 Supplementary Methods (open access): 6 x 6 x 6 cubic lattice, 0.97 m a side, tubes 150 mm long and 10 mm across; the file's `lattice_3d` bars and `lattice_type` "diagonal" | the file's bars, in the positions' frame (not fitted) | head a median 6.0 cm from the nearest bar (99th percentile 15.4); `track_width` 30.8 cm = 2 x that 99th percentile | 1.0 % |
| yartsev_bat_demo | declared: the room, every voxel of a grid tiling `r_lim` (5.8 x 5.2 x 2.3 m) exactly, 29 x 26 x 11 cells of 0.2 x 0.2 x 0.209 m | `NP_Bat_Demo.m` (`r_lim`, "boundaries of the room"); Forli et al. 2025 Methods (Zotero full text): "an indoor flight room (n = 4 bats, 5.6 m x 5.2 m x 2.5 m)" | the file's frame (not fitted) | samples span 5.69 x 6.72 x 1.88 m; outside `r_lim` 3.66 %, outside the paper's size centred on `r_lim` 3.77 % | 3.66 % |
| rsc_000987 | inferred | NWB: "square arena"; Mehrotra, Levenstein et al. 2023 bioRxiv Methods (open-access PDF): square of "removable walls (width: 80 cm, height: 50 cm)" | – | 52.6 x 52.1 cm against 80 cm: -34 %, refused (G-130) | – |
| ca1ca3_001880 | inferred | no size in the file or the DANDI description; the paper (Neuron 2026) could not be read | – | – | – |
| hafting_2008_nird | declared as a track (BATCH_FAITHFUL_TRACKS.md): a 320 cm line in the file's frame | archive readme (read): "a linear track (mostly 320 cm)", no width | the file's coordinates, which the authors scaled to the stated 320 cm (so the length checks nothing) | not verified (see the tracks page) | – |
| senzai_000003 | inferred | pixels with no stated scale; the open field and the theta maze share one image with no epochs between them | – | – | – |

Each loader also records what its positions track
(`geometry["tracked_point"]`), from its own sources or "not stated": one
LED (000582), the head (Gardner; Kanter's and Vollan's OptiTrack; the
lattice's head positions), the midpoint of two head LEDs (hc-2), the DLC
body centre (002030), and "not stated" for 000405, the water maze's
EthoVision paths and the bat's room tracking. A head marker keeps off walls
a body point would reach, which the fidelity verdicts below meet.

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
| miniscope_002030 | 1153 / 455 | 37 % / 100 % | – (imaging) | – | – | – | refused / 1 |
| giocomo_000405 | 3121 / 2076 | 67 % / 100 % | – (binned rates) | – | – | – | refused / 0 |
| gardner_2022_grid | 3601 / 3463 | 97 % / 100 % | 0.479 / 0.479 | -0.23 / refused | 13 %, 12.5 cm / 10 %, 12.5 cm | 7.07 / 78.1 cm; 7.07 / 78.2 cm | refused / refused |
| vollan_of | 2828 / 2041 | 72 % / 100 % | 0.38 / 0.44 | -1 / refused (G-131) | 5 %, 7.5 cm / 48 %, 42.5 cm | 8.54 / 64.9 cm; 8.54 / 66.0 cm | refused / 0 |
| kanter_2025 | 1601 / 1049 | 65 % / 100 % | 0.151 / 0.161 | 0.08 / refused | 20 %, 37.5 cm / 38 %, 42.5 cm | 21.2 / 60.0 cm; 21.2 / 60.6 cm | refused / 1 |
| watermaze_10248158 | 1124 / 1144 | 99 % / 100 % | – (behaviour only) | – | – | – | refused / refused |
| hc2_ec013_527 | 5329 / 3210 | 60 % / 100 % | 1.44 / 1.66 | -1 / refused (G-131) | refused (pixels, G-133) | 74.1 / 105 px; 83.4 / 110 px | refused / 1 |
| grieves_2020_lattice | 4410 / 8193 | 57 % / 100 % | 0.86 / 0.78 | refused (3-D) | – | 97.0 / 97.0 cm; 82.7 / 76.8 cm | 1 (G-134) / refused |
| yartsev_bat_demo | 8294 / 1352 | 14 % / 89 % | 3.52 / 3.47 | refused (3-D) | – | 2.36 / 2.56 m; 2.18 / 2.64 m | refused / refused |

**Occupancy and rate maps.** A declared arena keeps the bins the animal never
visited, as NaN: 3 % of Gardner's (143 min, walls reached), 13 % of
000582's, 28 % of Vollan's, 35 % of Kanter's, 40 % of hc-2's, 86 % of the
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
-> 5 (the lost unit scored 0.006). On Kanter the drop is not the grid's
alignment (the samples-built environment on the declared grid keeps 262)
and disappears with unsmoothed rates (361 against 364): it comes with the
default smoothed maps on an environment that holds unvisited bins. Why the
score is then NaN was not traced (G-135).

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
against 76.8 cm). In the bat's room the declared grid decodes worse than the
visited voxels (2.36 m against 2.18 m; chance 2.56 and 2.64 m): the
uniform prior now spreads over the 86 % of the room the bat never entered
in flight.

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

The declared arenas, the lattice and the room are judged by the same
function as the declared tracks (`_geometry.judge_declared`; the arenas'
own `placement_reasons`, with its 2 % limit on samples outside, was folded
into it). A declared outline covers itself by construction, so the
evidence is the samples against it: samples reaching more than one bin
past a wall, the samples' extent more than 15 % short of the outline, a
stated size more than 15 % from the samples' fitted size, and a stated
centre more than 15 % of the radius off. The share of a side with no
sample within 3 bins (a circle is one side), and of the lattice's segments
with no sample within a bin, is reported, not judged: head samples fall
short of walls and bars because the tracked LEDs sit on a headstage above
the animal (the maintainer's rule, G-157). Samples outside the outline or
room are a tracking measure, reported by speed, not judged. A verdict of
"matches" needs a stated fact (a size, or the authors' bars). A room is
judged as an outline is, per face (the 95th percentile of the samples
nearest a face more than one bin past it is a mismatch); the lattice's
placement is not judged (its bars are tilted, so no bar is vertical, and a
head held above a bar biases any offset), so its match rests on its stated
side. Nine of the ten match and the bat's room is a mismatch. Shares are
over the judged samples (`analysis_epochs`):

| Recording | Verdict | Stated fact (fitted vs stated) | Samples outside | Worst side: no sample within 3 bins (reported) | Outermost samples inside the wall (95th percentile, worst side) |
| --- | --- | --- | --- | --- | --- |
| sargolini_000582 | matches | -0.1 %, -2.6 % | 0 % | 0 % | 0.8 cm |
| miniscope_002030 | matches | diameter +0.0 % | 0.05 % | 0 % | 0.1 cm |
| giocomo_000405 | matches | -6.9 %, -6.3 % | 0 % | 71 % | 7.2 cm |
| gardner_2022_grid | matches | -1.9 %, -2.4 % | 0 % | 0 % | 3.7 cm |
| vollan_of | matches | diameter -4.0 %, centre 2.2 % of the radius | 0 % | 31 % | 3.3 cm |
| kanter_2025 | matches | -2.1 %, -2.3 % | 0 % | 0 % | 1.9 cm |
| watermaze_10248158 | matches | diameter +0.2 %, centre 0.9 % of the radius | 0.34 % | 0 % | 0.2 units |
| hc2_ec013_527 | matches | +0.0 %, -0.8 % | 0.00 % | 48 % | 3.7 px |
| grieves_2020_lattice | matches | side +0.0 % of 97 cm (the file's bars); placement not judged | 1.0 % (placement tolerance) | every bar segment is shorter than a judged stretch; the worst has no head sample within 4 cm (reported) | – |
| yartsev_bat_demo | mismatch | `r_lim` against the paper's room: +3.6 %, +0.0 %, -8.0 %; but 17 of the 57 flight samples nearest the y-low face lie 1.4 m past it (95th percentile 1.41 m, limit 0.2 m) | 0.11 % of flight samples (3.66 % of all) | – | – |

The residuals moved by up to 2 points from the faithful-arenas round: the
stated shapes are now placed after a radial trim of the samples (the
farthest 0.4 % from their median) instead of a per-axis one, which kept
glitches off a diagonal. Before the support rule the four (000405,
Vollan's circle, hc-2's platform and the lattice) were mismatches on
support alone: the tracked head keeps 3-7 cm off the walls of the first
three and a median 6.0 cm off the lattice's bars, while every stated size
fits within 15 %. The checks that remain (15 % and one bin past a wall or
face) were not tuned to these recordings, and how sensitive they are is
not measured here (G-157). In the faithful-arenas round the bat's room was
a mismatch on its 3.7 % of samples outside `r_lim`, then a match on its
stated sides once that share became a room measure (G-025); judged per
face on the flight samples, it is a mismatch again, on the flights that
reach the perch beyond the y-low wall.

The lattice's `track_width` is a placement tolerance chosen from the
samples, 2 x the head's 99th percentile distance from a bar (30.8 cm), not
a stated width: the stated tubes are 10 mm across, and 99.9 % of head
samples lie beyond 5 mm of a bar. The samples-built environments, measured
against the same outlines, cover 37-98 % of them (75 % of Vollan's
circle, 67 % of 000405's rectangle) with their outermost bins 0.3-6.3 cm
inside the wall; the declared environments have no interior holes (the
samples-built ones had 72 on 000582, 131 on Vollan, 216 on hc-2). On the
lattice, the path through the declared bins is a median 0.95 x the
distance along the bars (voxels 0.77); for pairs on parallel bars of one
face it is 30.4 cm against 31.1 cm along the bars (voxels 18.2 cm). The
bat's room has no voxel outside `r_lim` (4.4 % of the samples-built voxels)
and both feeders in the environment. The study's "far links" and "sparse
bins" are large for declared environments (for example 53 % and 68 % on
002030): they count declared bins the animal never visited, which is the
point of declaring them, not air or a wall.

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
