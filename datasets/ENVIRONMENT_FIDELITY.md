# Environment fidelity

Does the environment the survey builds for each recording have the shape and
connectivity of the apparatus, not only its scale? The script is
[`studies/environment_fidelity.py`](studies/environment_fidelity.py); its
docstring defines every measure, and the full numbers are in
[`results/studies/environment_fidelity.md`](results/studies/environment_fidelity.md)
and `.json`. Figures (bins, links, samples, the reference track or bars, the
stated arena or room; three projections and height slices for 3-D) are
written to `$NEUROSPATIAL_DATA_DIR/figures/environment_fidelity/` and are not
tracked.

## Method

Each environment is the driver's: `Environment.from_samples` on every tracked
sample at the loader's `bin_size` (`operations._env`). Against the samples it
is measured by its connected parts, its holes (unvisited cells that visited
bins enclose), the share of links between neighbouring bins whose midpoint
(where a path crosses from one bin to the next) has no sample within one bin
("far" links: a wall, a gap or air), links that pass only through a corner,
and the path length through the bins over the straight line for random
sample pairs (seeded). Against the apparatus, where a source read for this
study states it, it is measured by the path through the bins over the
distance along the track (the authors' graph or centreline, the file's
linearized position, a straight track's axis, or the lattice's bars; computed
with NumPy and SciPy, not neurospatial), the share of samples and bins
within a track width, and, for a stated square or circle, the share of the
shape the bins cover and the gap from the wall to the nearest bin. Derived
views answer specific questions: the figure-8's camera positions, each
000447 W-track alone, 001754's staircase session alone, and the 1-D
environment from each file's linearized position.

A verdict is `matches`, `mismatch` (reasons listed) or `not verified` when no
source read here states the geometry. Sources read: the NWB files' own
descriptions (000582 "1 x 1 meter enclosure", 000568 "110 cm long, 6.35 cm
wide", Kanter's epochs table 100 x 100 x 50 cm, 000987 "square arena"), the
DANDI descriptions (002030 96 cm circle; 000044 1.6 m linear and 1 m circular
platforms; 000447 novel and familiar W-tracks; 001754 Escher staircase;
001845 paired W-mazes), the water maze's JSON arena, the Zenodo README of
5566548 (Z-track), the Figshare description of 24757638 (3-arm maze), the
bat demo's `NP_Bat_Demo.m` (`r_lim` is "boundaries of the room", feeder
coordinates `r_fd`), the files' own geometry (J16's graph, Takigawa's
centrelines, the lattice's bars), and these papers: Grosmark & Buzsaki 2016
(Zotero PDF), Tirole et al. 2022 (Zotero PDF), Vollan et al. 2025 (Zotero
PDF), Shin et al. 2019 (Zotero PDF), Aery Jones et al. 2026 (Zotero PDF) and
Grieves et al. 2020 with its Supplementary Methods (open access, downloaded).
Forli et al. 2025 was read from Zotero's full-text cache (its PDF is not on
disk). Not read, so not verified: Knierim et al. 2000 (no PDF), Gardner et al. 2022, Hafting et al.
2008's archive readme, Shipley et al. 2025, and any source for 000405,
000165's length, 001695, 001880 and 000003.

## Per recording

Components: connected parts (in brackets, parts with 1 % of the bins or
more). Far links: links whose midpoint has no sample within one bin.
Geodesic / straight: median over random sample pairs. On-track: samples
within half the track width of the reference / bins within half the width
plus half a bin (`*` an assumed 10 cm width, not judged; none for the
lattice, where every point is within 11.4 cm of a bar). Env / track: median path through the bins over the distance
along the reference. Covered: share of the stated square or circle the bins
cover (with holes filled by `Environment.open_field`). Rows ending in
`_camera`, `_novel`, `_familiar`, `_escher` or `_linearized` are the derived
views.

| Recording | Dims | Bins | Components | Holes (bins) | Far links | Corner links | Geodesic / straight | On-track | Env / track | Covered | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sargolini_000582 | 2-D | 1399 | 1 (1) | 72 (136) | 0.10% | 12 | 1.06 | – | – | 86% (94% filled) | mismatch: bins cover 86% of the stated square (94% with holes filled) |
| figure8_000552 | 1-D | 112 | 1 (1) | – | 0.00% | 0 | 1.00 | – | – | – | mismatch: loop ends 222 apart along the bins, 3 in the camera |
| figure8_000552_camera | 2-D | 667 | 3 (1) | 7 (806) | 0.00% | 6 | 1.21 | – | – | – | matches |
| rosenberg_labyrinth | 2-D | 2780 | 1 (1) | 41 (185) | 0.05% | 8 | 1.08 | 95% / 99% | 0.66 | – | mismatch: geodesic 0.66 x the distance along the track |
| yartsev_bat_demo | 3-D | 1352 | 1 (1) | – | 0.73% | 211 | 1.10 | – | – | – | mismatch: 4.4% of bins outside the room bounds |
| miniscope_002030 | 2-D | 455 | 1 (1) | 19 (722) | 0.09% | 8 | 1.21 | – | – | 37% (99% filled) | mismatch: bins cover 37% of the stated circle (99% with holes filled) |
| giocomo_000405 | 2-D | 2076 | 1 (1) | 105 (446) | 0.03% | 22 | 1.06 | – | – | – | not verified: no source read here states the geometry |
| giocomo_001701 | 2-D | 266 | 1 (1) | 0 (0) | 0.00% | 0 | 1.12 | – | – | – | matches |
| gardner_2022_grid | 2-D | 3463 | 1 (1) | 0 (0) | 0.00% | 0 | 1.06 | – | – | 96% (96% filled) | not verified: no source read here states the geometry |
| vollan_lt | 2-D | 556 | 1 (1) | 11 (34) | 0.06% | 3 | 1.04 | 57% / 58%* | 1.04 | – | matches |
| vollan_of | 2-D | 2041 | 1 (1) | 131 (297) | 0.12% | 17 | 1.07 | – | – | 57% (65% filled) | mismatch: bins cover 57% of the stated square (65% with holes filled) |
| grosmark_000044_circular | 2-D | 1547 | 2 (1) | 17 (609) | 0.02% | 10 | 1.09 | 69% / 43%* | 0.93 | – | matches |
| grosmark_000044_circular_linearized | 1-D | 117 | 1 (1) | – | 0.00% | 0 | 1.00 | – | – | – | mismatch: loop ends 290 apart along the bins, 27 in the camera |
| grosmark_000044_linear | 2-D | 1096 | 21 (3) | 13 (49) | 0.06% | 11 | 1.04 | 13% / 27%* | 1.19 | – | mismatch: 3 parts with 1% of the bins or more, apparatus has 1 |
| grosmark_000044_linear_linearized | 1-D | 65 | 1 (1) | – | 0.00% | 0 | 1.00 | – | – | – | matches |
| jadhav_000447 | 2-D | 1465 | 1 (1) | 29 (240) | 0.02% | 7 | 1.07 | – | – | – | mismatch: 1 part with 1% of the bins or more, apparatus has 2 |
| jadhav_000447_novel | 2-D | 1172 | 1 (1) | 20 (81) | 0.00% | 5 | 1.19 | – | – | – | matches |
| jadhav_000447_familiar | 2-D | 788 | 1 (1) | 21 (140) | 0.00% | 8 | 2.53 | – | – | – | matches |
| grieves_2020_lattice | 3-D | 8193 | 1 (1) | – | 0.66% | 921 | 1.14 | – | 0.77 | – | mismatch: geodesic 0.77 x the distance along the bars |
| watermaze_10248158 | 2-D | 1144 | 1 (1) | 8 (9) | 0.02% | 1 | 1.05 | – | – | 99% (99% filled) | matches |
| olafsdottir_5566548 | 2-D | 556 | 7 (2) | 3 (3) | 0.00% | 13 | 1.05 | – | – | – | mismatch: 2 parts with 1% of the bins or more, apparatus has 1 |
| takigawa_7114733 | 2-D | 952 | 13 (2) | 8 (9) | 0.13% | 15 | 1.09 | 59% / 37%* | 0.87 | – | mismatch: geodesic 0.87 x the distance along the track |
| hafting_2008_nird | 2-D | 507 | 1 (1) | 1 (2) | 0.00% | 1 | 1.01 | 64% / 47%* | 1.01 | – | not verified: no source read here states the geometry |
| j16_23020757 | 2-D | 545 | 1 (1) | 16 (48) | 0.13% | 5 | 1.14 | 69% / 63%* | 1.01 | – | matches |
| j16_23020757_linearized | 1-D | 129 | 9 (9) | – | 0.00% | 0 | 1.01 | – | – | – | mismatch: 9 parts with 1% of the bins or more, apparatus has 1 |
| agarwal_24757638 | 2-D | 449 | 1 (1) | 3 (3) | 0.00% | 1 | 1.10 | – | – | – | matches |
| shipley_30902972 | 2-D | 697 | 1 (1) | 5 (10) | 0.00% | 6 | 1.08 | – | – | – | not verified: no source read here states the geometry |
| rsc_000987 | 2-D | 453 | 1 (1) | 0 (0) | 0.00% | 0 | 1.05 | – | – | – | matches |
| ca1ca3_001880 | 2-D | 1080 | 1 (1) | 0 (0) | 0.00% | 0 | 1.06 | – | – | – | not verified: no source read here states the geometry |
| gonzalez_001695 | 2-D | 95 | 1 (1) | 0 (0) | 0.00% | 0 | 1.01 | – | – | – | not verified: no source read here states the geometry |
| neurolab_001754 | 2-D | 2174 | 9 (1) | 117 (217) | 0.03% | 90 | 1.07 | – | – | – | not verified: no source read here states the geometry |
| neurolab_001754_escher | 2-D | 1233 | 23 (2) | 74 (228) | 0.06% | 76 | 1.10 | – | – | – | not verified: no source read here states the geometry |
| valero_000568 | 2-D | 190 | 1 (1) | 0 (0) | 0.00% | 1 | 1.02 | 88% / 78% | 1.02 | – | mismatch: 78% of bins within the stated track width |
| aeryjones_000165 | 2-D | 163 | 1 (1) | 0 (0) | 0.00% | 0 | 1.03 | – | – | – | matches |
| dyads_001845 | 2-D | 245 | 4 (3) | 6 (32) | 0.00% | 9 | 1.04 | – | – | – | mismatch: 3 parts with 1% of the bins or more, apparatus has 1 |
| senzai_000003 | 2-D | 840 | 1 (1) | 17 (33) | 0.00% | 0 | 1.04 | – | – | – | not verified: no source read here states the geometry |
| kanter_2025 | 2-D | 1049 | 1 (1) | 49 (460) | 0.18% | 9 | 1.07 | – | – | 65% (94% filled) | mismatch: bins cover 65% of the stated square (94% with holes filled) |

## What the numbers say

**Open arenas.** Where the animal reached the walls, the outline is right:
the outermost bins come within a median 1.6 cm (0.6 bin) of the stated wall
in 000582, Kanter and Gardner, and the samples' convex hull has the stated
shape (hull over its minimum rectangle and over its minimum circle: squares
0.98-0.99 and 0.64-0.66; 002030's circle and the pool 0.79-0.80 and
0.93-0.97). The interior is not: short sessions leave holes (000582 72,
Kanter 49, 000405 105, Vollan's open field 131; 002030's 96 cm circle is 37 %
covered because the mouse ran along the wall), and `Environment.open_field`
(`fill_holes=True`) restores 94-99 % coverage (G-015). Walls are never
represented: every arena is a `MaskArena` of visited bins with no outline, so
border scores refuse it (G-005) and egocentric boundary maps take the edge of
the visited bins as the wall (G-082). Vollan's open-field session is round
(hull 0.81 of its rectangle, 0.94 of its circle) where the paper states a
150 cm square box for that task (G-114). 001701's X maze has its four arms
and centre segment in one part; the paper's "130-cm X-maze" does not say what
the 130 cm spans (the samples span 112 cm tip to tip), so its size is not
compared.

**Tracks and mazes.** No track or maze has its arms linked through the gaps
between them: the far-link share is 0-0.13 %, corner-only links number 0-15,
and the path lengths keep the detours of a maze (each W-track alone: 95th
percentile 4.3-4.9 x the straight line; the 3-arm maze 8.5). Straight tracks follow their axis (000568,
Vollan's linear track and Hafting: path through the bins 1.01-1.04 x the
distance along the track axis; 001695 and 000165 1.01-1.03 x the straight
line). On 000568, the one track with a stated width, 88 % of samples and 78 %
of bins are on the 6.35 cm track: the head LED leans past its edges (the
loader notes x spans 4-21 cm), so the bins are wider than the track and the
verdict is a mismatch. The W-tracks, radial arms, 3-arm maze and J16's graph
come out as one connected track each with maze-like detours (J16: 1.01 x the
distance along the authors' graph). The failures are of three kinds:

- Walls thinner than a bin are invisible. The labyrinth's environment joins
  neighbouring corridors: the path through its bins is 0.66 x the corridor
  distance (5th percentile 0.28), while no link is "far" (G-110). The
  graph environment the driver also builds is right.
- One environment for several apparatus. 000447's novel and familiar
  W-tracks, rotated 90 degrees in one camera frame, merge into one 110 x 105
  cm block (path / straight 1.07); each alone is a W (1.19 and 2.53, 95th
  percentile up to 4.9). 001754's three apparatus, 000003's open field and
  maze, and 000044's pre-track excursion are merged the same way (G-113).
- Samples that are not on the track. 000044's linear track: 84 % of samples
  are at the two reward ends or before the run, so only 27 % of the bins lie
  on a 10 cm wide stem (G-039); the Z-track's tracking jumps add a sparse
  second diagonal and a second part (G-117); 001845's 123 s of head
  positions leave its W in three parts (G-116). Takigawa's two tracks are two
  parts (511 and 411 bins) as they should be, with wide sample clouds that
  cut the bends (0.87 x the centreline distance). Their centrelines are 2 m
  long, but that is not a check of the scale: the authors derived
  `cm_per_pixel` from the 2 m length.

**1-D inputs.** The figure-8's linearized position and 000044's circular
coordinate are covered without gaps (bins over range 1.01), but each loop's
two ends are 222 and 290 cm apart along the bins while they sit 3 and 27 cm
apart in the camera (G-021). 000044's circular coordinate does not quite
close (27 cm). J16's linearized position, laid out with 15 cm gaps between
corridors, gives a 1-D environment in 9 parts with no junction (G-115). The
figure-8's camera positions give the right 2-D shape: one main part with its
two loops as holes (806 bins in 7 holes, the largest 445).

## 3-D deep dives

**Grieves 2020 lattice (rat 926, tilted lattice).** The file holds the 147
bars (97 cm, crossings every 16.2 cm), matching the Supplementary Methods'
6 x 6 x 6 lattice 0.97 m a side; `config3.lattice_type` "diagonal" says this
is the tilted lattice. Head positions are a median 6.0 cm from the nearest
bar (95th percentile 10.4, 99th 15.4), and 17 % lie farther from every bar
than the centre of a cube face (8.1 cm): the head moves through the space
between bars, so the 8,193 voxels (one part) are supported by samples (0.66 %
far links; 31 % of voxels have no sample within 2 cm of their centre, which
is sparse 3-D sampling, not air). The voxel graph is not the climbing
structure: for random pairs the path through the voxels is a median 0.77 x
the distance along the bars (5th percentile 0.62), and for 5,478 pairs of
samples at the middle of parallel bars that bound one cube face, 16.6 cm
apart (about two spacings along the bars), it is 18.2 cm against 31.1 cm
along the bars (below 0.75 x for 83 %). A bar graph built from the file (343
crossings, 882 bar segments) with `Environment.from_graph` and 3-D
node positions gives 4,410 bins in one part and, with `track_width` = 2 x
the 99th percentile offset (30.8 cm), places 99.0 % of head positions; 57 %
of its bins get occupancy. It only works if `edge_order` lists every
segment: the depth-first order the driver uses keeps 342 of 882, a spanning
tree, silently, and still counts positions on the dropped segments as on the
track, putting them in the wrong bins (G-112). Verdict: the voxel environment represents the head's
space, not the bars; which one an analysis wants depends on whether place
is the head or the body on the lattice, and neurospatial offers only the
first directly (G-111). The movement direction itself is only weakly
aligned with the bars at this tracking point (22 % of 0.2 s steps within 20
degrees of a bar axis against 18 % for random directions), so distances
along the bars are an upper bound on the head's travel, and voxel
distances a lower one.

**Bat demo (Forli et al. 2025).** The 0.2 m voxels (1,352, one part) fill
z 0.12-2.3 m and reach the x walls and the ceiling, but
4.4 % of them lie outside `r_lim`, a corridor out of the room to y = -4.2 m
(G-025). 97.6 % of samples are rest, mostly at one perch on the -x wall;
one of the two feeders in `NP_Bat_Demo.m` is in a voxel and the other 0.17 m
from the nearest one. The room's walls are not part of the environment. Air
links are expected for a flying animal; 0.73 % of links have no sample
within a bin of their midpoint, and 34 % of voxels none within half a bin of
their centre: flights follow stereotyped paths through a mostly empty room.
Forli et al. 2025 (Methods, read from Zotero's full-text cache) give the
indoor flight room as 5.6 x 5.2 x 2.5 m with two feeders on the wall at one
end; `r_lim` is 5.8 x 5.2 x 2.3 m and both `r_fd` feeders are at x = 2.8 m,
the +x wall. The voxels outside `r_lim` (to y = -4.2 m) are outside the
paper's room too.

## Declared tracks and mazes

Twenty-two track and maze loaders now try to declare their apparatus as a
track graph (`geometry["environment"]`, [BATCH_FAITHFUL_TRACKS.md](BATCH_FAITHFUL_TRACKS.md));
19 do, and three fitted layouts are refused on misfit evidence and stay
inferred (000447's novel W-track and the Z-track, whose fitted lengths are
more than 15 % off the stated ones, and 001845, whose W-maze base is
barely sampled; G-153). The study measures each declared environment
beside the samples-built one ("Declared environments" in the results). The
table above is from the first run; in the rerun's results file
figure8_000552 is the camera positions (the old 1-D row is
`figure8_000552_linearized`) and jadhav_000447 is the novel W-track (the
familiar one is jadhav_000447_familiar).

A declared track is judged by one function for any declared environment
(`_geometry.judge_declared`), on misfit evidence: a straight stretch of
track (at least 20 bins, an arm's length, not crossing a corner) whose
samples lie a median of more than one bin to one side, a segment with no
sample within a bin over more than a quarter of its length, or a stated
length more than 15 % off. The share of samples beyond the physical width
is not misfit: the tracked point is usually an LED on a headstage above an
unwalled track, which swings past the edge (and a 2-D camera projects a
raised head outward); it is reported as a tracking measure, split by speed
(G-152). Each track is built with a placement tolerance measured from the
samples, so about 1 % of samples are outside it by construction. Its bins
lie on its own graph, so the share of bins on the track and the path along
it are evidence only against a reference that is not the geometry the track
was declared from. In short:

- 1 matches: the 000044 circle, placed by a circle fit and checked against
  the file's linearized centreline (path along the bins 0.96 x the distance
  along it; its worst stretch 2.38 cm to one side, within the 2.5 cm bin).
- 2 are mismatches on misfit: Takigawa's authors' centreline lies 4.12 cm
  to one side of its samples on one stretch (the limit is one 4 cm bin), and
  000044's linear 15 cm reward-area segments have no sample within a bin of
  their centreline over 44 % of their length (the samples spread across the
  reward areas).
- 16 are not verified: no misfit, but no independent reference checks the
  layout (the figure-8, the X maze, Agarwal's and Shipley's mazes, 000447's
  familiar W-track, hc-28's W-track, 001695, hc-31, and the straight tracks
  of Vollan, 000165, 000568, hc-3 and Hafting, the labyrinth and J16, whose
  only reference is the axis or graph they were declared from), and hc-18's
  circuit, whose shape no source states.
- Beyond the physical width, as a tracking measure: J16 63.8 %, hc-28
  37.3 %, hc-3 26.3 %, 000447's familiar W-track 13.0 %, 000568 12.4 %, the
  labyrinth 4.6 % (one cell, chosen; 17.9 % at the paper's 0.75 cell),
  Shipley 0.3 % of all samples. On J16 the graph is placed right (median
  signed offset -1.3 to +1.8 cm per corridor) and the share falls from 87 %
  stopped to 35 % above 30 cm/s (G-152).
- Against their own geometry the declared environments measure only their
  binning: the labyrinth's path along its bins is 0.95 x the distance along
  the graph it was built from, Takigawa's 0.97. These are not comparisons
  with the samples' grids (0.66 and 0.87 against the same references), which
  show what the grid loses; they show that the graph's bins follow the
  graph.

## What should change in neurospatial

- Climbing structures: a constructor (or a documented recipe) for an
  environment from 3-D bar segments, with head positions placed on the bars
  within a stated reach (G-111); and `from_graph` should refuse or warn when
  `edge_order` leaves graph edges out (G-112).
- Walls the samples cannot show: say in the `from_samples` docs that walls
  thinner than a bin are invisible to it, and point mazes to `from_graph` or
  to declared walls (G-110).
- Declared arenas: open arenas are always "visited bins"; a stated shape
  (`from_polygon`, `arena=`) is the only way to get walls, and the open-field
  holes need `open_field` or `fill_holes=True`; both should be the
  documented first step for border and boundary analyses (G-005, G-015).
- Track-only and per-apparatus environments are harness changes: build one
  environment per apparatus from the samples of its epochs, and use the
  graph environment where the loader gives one (G-113, G-115). The track
  loaders now do (above); what they need from neurospatial is a placement
  tolerance separate from the track's physical width (G-152; J16 is the
  case), laps that bridge a
  sample outside the track (G-151), a warning when occupancy drops samples
  outside the environment (G-155), and docstrings that match the refusal of
  a widthless track (G-150).

## What could not be checked

The Escher staircase's layout (no dimensions in the dandiset, Knierim et al.
2000 not available; from one overhead camera at spike times its parts cannot
be separated, G-053), Gardner's stated size (source not read), and 000405,
001880 and 000003 (no source read). Since this study was first run,
Hafting's archive readme ("a linear track (mostly 320 cm)"), Shipley et al.
2026 (80 x 10 cm arms on a 24.7 cm platform), Aery Jones et al. 2021 (a 36
cm track) and Gonzalez et al. 2026 (a linear track with a well at either
end; its size is in a supplementary table not read) were read for the track
loaders (BATCH_FAITHFUL_TRACKS.md).
