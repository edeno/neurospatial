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

Each row's environment is the one the driver builds without a declared
apparatus: `Environment.from_samples` on every tracked sample at the
loader's `bin_size` (`operations._samples_env`, the `--inferred-env` run).
Where the loader declares the apparatus, that environment is measured the
same way and judged in its own section below. Against the samples it
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
disk). Read later, for the declared environments: Gardner et al. 2022 (1.5 x
1.5 m), Gonzalez & Giocomo 2024 (000405, 1.3 x 1.5 m), Mehrotra, Levenstein
et al. 2023 (000987, 80 cm walls), the hc-2 data description (180 cm
platform), Hafting et al. 2008's archive readme, and the track papers listed
in [BATCH_FAITHFUL_TRACKS.md](BATCH_FAITHFUL_TRACKS.md). Not read, so not
verified: Knierim et al. 2000 (no PDF) and any source for 001880 and 000003.

## Per recording

The table is the samples-built rows of the results file
(`results/studies/environment_fidelity.md`), as rerun. Components: connected
parts (in brackets, parts with 1 % of the bins or more). Sparse / thin bins:
share of bins with no sample within half a bin of the centre, and holding
fewer than 5 samples. Unsupported / far links:
share of neighbour links with no sample within half a bin of the link's
midpoint (far: within one bin). Geodesic / straight: median and 95th
percentile over random sample pairs. On-track: samples within half the
track width of the reference / bins within half the width plus half a bin
("assumed width": an assumed 10 cm, not judged). Env / track: median and
5th percentile of the path through the bins over the distance along the
reference. Outline: share of the stated square or circle the bins cover,
bin centres outside it, the gap from the wall to the nearest bin, and the
samples' hull over its minimum rectangle and circle. Rows ending in
`_camera`, `_novel`, `_familiar`, `_escher` or `_linearized` are the derived
views. A verdict of "matches" rests only on a stated fact compared with the
samples (a stated length, diameter, side, loop count or room); a reference
built from the samples (the file's linearized centreline, the samples'
principal axis) is reported but does not judge.

| Dataset | Kind | Bins | Dims | Components (1 % of bins or more) | Holes (bins) | Sparse / thin bins | Unsupported / far links | Corner links | Geodesic / straight (median, p95) | On-track samples / bins | Env / track (median, p5) | Outline: covered / centres outside / wall gap / hull over rectangle, circle | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sargolini_000582 | open | 1399 | 2 | 1 (1) | 72 (136) | 6% / 14% | 7.6% / 0.10% | 12 | 1.06, 1.12 | – | – | 91% / 0% / 1.3 / 0.99, 0.64 | matches |
| figure8_000552 | track | 667 | 2 | 3 (1) | 7 (806) | 6% / 13% | 3.4% / 0.00% | 6 | 1.21, 1.68 | – | – | – | matches |
| figure8_000552_linearized | track | 112 | 1 | 1 (1) | – | 0% / 0% | 0.0% / 0.00% | 0 | 1.00, 1.04 | – | – | – | mismatch: loop ends 222 apart along the bins, 3 in the camera |
| rosenberg_labyrinth | track | 2780 | 2 | 1 (1) | 41 (185) | 4% / 10% | 2.4% / 0.05% | 8 | 1.08, 1.76 | 95% / 99% | 0.66, 0.28 | – | mismatch: 795 links join places far apart along the track (through a wall or gap); geodesic 0.66 x the distance along the track |
| yartsev_bat_demo | 3d | 1352 | 3 | 1 (1) | – | 34% / 27% | 29.6% / 0.73% | 211 | 1.10, 1.18 | – | – | – | mismatch: 4.4% of bins outside the room bounds |
| miniscope_002030 | open | 455 | 2 | 1 (1) | 19 (722) | 13% / 31% | 8.0% / 0.09% | 8 | 1.21, 1.52 | – | – | 37% / 0% / 0.3 / 0.79, 0.97 | mismatch: bins cover 37% of the stated circle (99% with holes filled) |
| giocomo_000405 | open | 2076 | 2 | 1 (1) | 105 (446) | 5% / 20% | 6.1% / 0.03% | 22 | 1.06, 1.13 | – | – | 67% / 0% / 6.3 / 0.96, 0.67 | mismatch: bins cover 67% of the stated rectangle (80% with holes filled) |
| giocomo_001701 | track | 266 | 2 | 1 (1) | 0 (0) | 8% / 8% | 0.4% / 0.00% | 0 | 1.12, 1.71 | – | – | – | not verified: no stated fact independent of the samples checks it |
| gardner_2022_grid | open | 3463 | 2 | 1 (1) | 0 (0) | 0% / 0% | 0.1% / 0.00% | 0 | 1.06, 1.10 | – | – | 96% / 0% / 1.6 / 0.99, 0.65 | matches |
| vollan_lt | track | 556 | 2 | 1 (1) | 11 (34) | 7% / 7% | 3.5% / 0.06% | 3 | 1.04, 1.13 | 57% / 58% (assumed width) | 1.04, 1.00 | – | mismatch: 2 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| vollan_of | open | 2041 | 2 | 1 (1) | 131 (297) | 9% / 8% | 8.4% / 0.12% | 17 | 1.07, 1.13 | – | – | 75% / 0% / 4.3 / 0.81, 0.94 | mismatch: bins cover 75% of the stated circle (83% with holes filled) |
| grosmark_000044_circular | track | 2318 | 2 | 2 (1) | 38 (1013) | 3% / 17% | 2.3% / 0.04% | 14 | 1.09, 1.39 | 69% / 43% (assumed width) | 0.94, 0.83 | – | matches |
| grosmark_000044_circular_linearized | track | 146 | 1 | 1 (1) | – | 0% / 0% | 0.0% / 0.00% | 0 | 1.00, 1.03 | – | – | – | mismatch: loop ends 290 apart along the bins, 26 in the camera |
| grosmark_000044_linear | track | 1530 | 2 | 35 (2) | 23 (98) | 5% / 28% | 3.5% / 0.00% | 21 | 1.04, 1.13 | 13% / 28% (assumed width) | 1.19, 1.06 | – | mismatch: 2 parts with 1% of the bins or more, apparatus has 1; 2 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| grosmark_000044_linear_linearized | track | 81 | 1 | 1 (1) | – | 0% / 0% | 0.0% / 0.00% | 0 | 1.00, 1.04 | – | – | – | not verified: no stated fact independent of the samples checks it |
| jadhav_000447 | track | 1172 | 2 | 1 (1) | 20 (81) | 3% / 19% | 2.2% / 0.00% | 5 | 1.19, 4.35 | – | – | – | mismatch: 2 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| jadhav_000447_familiar | track | 788 | 2 | 1 (1) | 21 (140) | 5% / 33% | 3.3% / 0.00% | 8 | 2.53, 4.87 | – | – | – | mismatch: 8 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| hc11_achilles | None | 1096 | 2 | 21 (3) | 13 (49) | 5% / 26% | 3.8% / 0.06% | 11 | 1.04, 1.12 | 13% / 27% (assumed width) | 1.19, 1.06 | – | not verified: no source read here states the geometry |
| hc11_achilles_linearized | None | 65 | 1 | 1 (1) | – | 0% / 0% | 0.0% / 0.00% | 0 | 1.00, 1.05 | – | – | – | not verified: no source read here states the geometry |
| hc3_ec013_669 | track | 540 | 2 | 1 (1) | 6 (12) | 6% / 20% | 3.6% / 0.06% | 3 | 1.05, 1.11 | – | – | – | not verified: no stated fact independent of the samples checks it |
| hc2_ec013_527 | open | 3210 | 2 | 1 (1) | 216 (1319) | 9% / 33% | 8.9% / 0.08% | 78 | 1.11, 1.23 | – | – | 62% / 0% / 3.8 / 0.96, 0.66 | mismatch: bins cover 62% of the stated square (87% with holes filled) |
| hc28_hpa | track | 712 | 2 | 1 (1) | 10 (16) | 3% / 16% | 3.1% / 0.04% | 0 | 1.11, 3.90 | – | – | – | not verified: no stated fact independent of the samples checks it |
| hc31_anm190422 | track | 1928 | 1 | 1 (1) | – | 0% / 5% | 0.1% / 0.00% | 0 | 1.00, 1.00 | – | – | – | not verified: no stated fact independent of the samples checks it |
| hc18_train261 | track | 2723 | 2 | 27 (2) | 39 (10750) | 8% / 28% | 2.9% / 0.01% | 70 | 1.06, 1.16 | – | – | – | not verified: no source read here states the geometry |
| grieves_2020_lattice | 3d | 8193 | 3 | 1 (1) | – | 31% / 21% | 23.4% / 0.66% | 921 | 1.14, 1.30 | – | 0.77, 0.62 | – | mismatch: 1757 links join places far apart along the track (through a wall or gap); geodesic 0.77 x the distance along the bars |
| watermaze_10248158 | open | 1144 | 2 | 1 (1) | 8 (9) | 2% / 11% | 1.5% / 0.02% | 1 | 1.05, 1.12 | – | – | 98% / 0% / 0.6 / 0.79, 0.93 | matches |
| olafsdottir_5566548 | track | 556 | 2 | 7 (2) | 3 (3) | 8% / 27% | 4.3% / 0.00% | 13 | 1.05, 2.74 | – | – | – | mismatch: 2 parts with 1% of the bins or more, apparatus has 1 |
| takigawa_7114733 | track | 952 | 2 | 13 (2) | 8 (9) | 4% / 20% | 3.2% / 0.13% | 15 | 1.09, 1.23 | 59% / 37% (assumed width) | 0.87, 0.75 | – | mismatch: 2 links join places far apart along the track (through a wall or gap); geodesic 0.87 x the distance along the track |
| hafting_2008_nird | track | 507 | 2 | 1 (1) | 1 (2) | 4% / 8% | 2.6% / 0.00% | 1 | 1.01, 1.08 | 64% / 47% (assumed width) | 1.01, 0.99 | – | not verified: no stated fact independent of the samples checks it |
| j16_23020757 | track | 545 | 2 | 1 (1) | 16 (48) | 11% / 2% | 3.5% / 0.13% | 5 | 1.14, 2.22 | 36% / 50% | 1.01, 0.96 | – | mismatch: 2 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| j16_23020757_linearized | track | 129 | 1 | 9 (9) | – | 0% / 1% | 0.0% / 0.00% | 0 | 1.01, 1.10 | – | – | – | mismatch: 9 parts with 1% of the bins or more, apparatus has 1 |
| agarwal_24757638 | track | 449 | 2 | 1 (1) | 3 (3) | 5% / 9% | 0.9% / 0.00% | 1 | 1.10, 8.46 | – | – | – | not verified: no stated fact independent of the samples checks it |
| shipley_30902972 | track | 697 | 2 | 1 (1) | 5 (10) | 8% / 19% | 2.0% / 0.00% | 6 | 1.08, 1.58 | – | – | – | not verified: no stated fact independent of the samples checks it |
| rsc_000987 | open | 453 | 2 | 1 (1) | 0 (0) | 2% / 0% | 1.1% / 0.00% | 0 | 1.05, 1.13 | – | – | 44% / 0% / 13.9 / 0.98, 0.64 | mismatch: bins cover 44% of the stated square (44% with holes filled) |
| ca1ca3_001880 | open | 1080 | 2 | 1 (1) | 0 (0) | 1% / 2% | 0.7% / 0.00% | 0 | 1.06, 1.12 | – | – | – | not verified: no source read here states the geometry |
| gonzalez_001695 | track | 95 | 2 | 1 (1) | 0 (0) | 1% / 1% | 0.4% / 0.00% | 0 | 1.01, 1.07 | – | – | – | not verified: no stated fact independent of the samples checks it |
| neurolab_001754 | track | 2174 | 2 | 9 (1) | 117 (217) | 5% / 49% | 9.6% / 0.03% | 90 | 1.07, 1.23 | – | – | – | not verified: no source read here states the geometry |
| neurolab_001754_escher | track | 1233 | 2 | 23 (2) | 74 (228) | 10% / 58% | 13.2% / 0.06% | 76 | 1.10, 1.43 | – | – | – | not verified: no source read here states the geometry |
| valero_000568 | track | 190 | 2 | 1 (1) | 0 (0) | 4% / 9% | 1.5% / 0.00% | 1 | 1.02, 1.09 | 88% / 78% | 1.02, 0.98 | – | matches |
| aeryjones_000165 | track | 163 | 2 | 1 (1) | 0 (0) | 7% / 18% | 1.5% / 0.00% | 0 | 1.03, 1.07 | 100% / 100% (assumed width) | 1.03, 0.99 | – | matches |
| dyads_001845 | track | 245 | 2 | 4 (3) | 6 (32) | 9% / 84% | 14.0% / 0.00% | 9 | 1.04, 1.19 | – | – | – | mismatch: 3 parts with 1% of the bins or more, apparatus has 1; 3 loop-sized holes (1% of the bins or more), apparatus has 0 loops |
| senzai_000003 | open | 840 | 2 | 1 (1) | 17 (33) | 3% / 21% | 3.6% / 0.00% | 0 | 1.04, 1.15 | – | – | – | not verified: no source read here states the geometry |
| kanter_2025 | open | 1049 | 2 | 1 (1) | 49 (460) | 7% / 8% | 6.6% / 0.18% | 9 | 1.07, 1.19 | – | – | 65% / 0% / 1.6 / 0.98, 0.65 | mismatch: bins cover 65% of the stated square (94% with holes filled) |

## What the numbers say

**Open arenas.** These are the samples-built environments, measured
against the outline each loader now declares (below). Where the animal
reached the walls, the outline is right: the outermost bins come within a
median 1.2-1.9 cm (0.5-0.8 bin) of the stated wall in 000582, Kanter and
Gardner, and the samples' convex hull has the stated
shape (hull over its minimum rectangle and over its minimum circle: squares
0.98-0.99 and 0.64-0.66; 002030's circle and the pool 0.79-0.80 and
0.93-0.97). The interior is not: short sessions leave holes (000582 72,
Kanter 49, 000405 105, Vollan's open field 131; 002030's 96 cm circle is 37 %
covered because the mouse ran along the wall), and `Environment.open_field`
(`fill_holes=True`) restores 94-99 % coverage on the arenas the animal
crossed (000582, Kanter, 002030) but only 80 % on 000405 and 83 % on
Vollan's circle, whose tracked head stays 4-6 cm off the wall (wall gap 6.3
and 4.3 cm), and 44 % of 000987's stated 80 cm square, whose samples span
52 cm (G-130) (G-015). Walls are never
represented: every arena is a `MaskArena` of visited bins with no outline, so
border scores refuse it (G-005) and egocentric boundary maps take the edge of
the visited bins as the wall (G-082). Vollan's open-field session is round
(hull 0.81 of its rectangle, 0.94 of its circle) and is now compared with the
paper's one circular arena, 150 cm across (G-114). 001701's X maze has its four arms
and centre segment in one part; the paper's "130-cm X-maze" does not say what
the 130 cm spans (the samples span 112 cm tip to tip), so its size is not
compared.

**Tracks and mazes.** The far-link share (0-0.13 %) cannot show arms linked
through a gap: when every bin holds a sample, a link's midpoint is at most
about 1.4 bins from one, so two 10 cm corridors behind a 0.5-3 cm wall merge
with no far link (`test_jump_links_find_corridors_joined_through_a_thin_wall`).
The rerun counts instead the links whose two bins the stated layout puts far
apart along the track (`jump_links`: more than 3 x the link plus both bins'
offsets), where a layout is known: the labyrinth has 795 (8.1 % of its
links, G-110), Takigawa's two tracks 2, the lattice's voxels 1,757 (links
across a cube face, between bars that meet only at a crossing), and the
eight other recordings
with a reference none; the W-tracks, radial, 3-arm, X and Z mazes have no
independent layout, so their links are not checked. At an open hub, where
arms' corridors overlap, `jump_links` flags links that join no corridors
through a wall (G-156). The rerun also judges
loops: holes of 1 % of the bins or more are loops in a grid from samples,
and a tree or straight track should have none, but J16, Vollan's linear
track, both 000447 W-tracks (2 and 8), 001845 (3) and 000044's linear track
have some; the circle has its 1 and the figure-8 its 2. Corner-only links
number 0-15, and the path lengths keep the detours of a maze (each W-track alone: 95th
percentile 4.3-4.9 x the straight line; the 3-arm maze 8.5). Straight tracks follow their axis (000568,
Vollan's linear track and Hafting: path through the bins 1.01-1.04 x the
distance along the track axis; 001695 and 000165 1.01-1.03 x the straight
line). On 000568, the one track with a stated width, 88 % of samples and 78 %
of bins are on the 6.35 cm track: the head LED leans past its edges (the
loader notes x spans 4-21 cm), so the bins are wider than the track; that
is a tracking measure, not judged, and 000568 matches on its stated length
(the samples span +5.1 % of the stated 110 cm). Links and the path along a
reference built from the samples (Vollan's and 000568's principal axis,
000044's linearized centreline, Hafting's line, drawn in a frame the
authors scaled to the stated 320 cm) are reported but not judged, so 001701,
Agarwal's and Shipley's mazes, 001695, hc-3, hc-28, hc-31, Hafting and
000044's 1-D linear row are "not verified" from the samples alone. The W-tracks, radial arms, 3-arm maze and J16's graph
come out as one connected track each with maze-like detours (J16: 1.01 x the
distance along the authors' graph). The failures are of three kinds:

- Walls thinner than a bin are invisible to the samples. The labyrinth's
  environment joins neighbouring corridors: the path through its bins is
  0.66 x the corridor distance (5th percentile 0.28), and 795 of its links
  join bins far apart along the corridors, while no link is "far" from the
  samples (G-110). The graph environment the driver also builds is right.
- One environment for several apparatus. 000447's novel and familiar
  W-tracks, rotated 90 degrees in one camera frame, merge into one 110 x 105
  cm block (path / straight 1.07); each alone is a W (1.19 and 2.53, 95th
  percentile up to 4.9). 001754's three apparatus, 000003's open field and
  maze, and 000044's pre-track excursion are merged the same way (G-113).
- Samples that are not on the track. 000044's linear track: 84 % of samples
  are at the two reward ends or before the run, so only 28 % of the bins lie
  on a 10 cm wide stem (G-039); the Z-track's tracking jumps add a sparse
  second diagonal and a second part (G-117); 001845's 123 s of head
  positions leave its W in three parts (G-116). Takigawa's two tracks are two
  parts (511 and 411 bins) as they should be, with wide sample clouds that
  cut the bends (0.87 x the centreline distance). Their centrelines are 2 m
  long, but that is not a check of the scale: the authors derived
  `cm_per_pixel` from the 2 m length.

**1-D inputs.** The figure-8's linearized position and 000044's circular
coordinate are covered without gaps (bins over range 1.01), but each loop's
two ends are 222 and 290 cm apart along the bins while they sit 3 and 26 cm
apart in the camera (G-021). 000044's circular coordinate does not quite
close (26 cm). J16's linearized position, laid out with 15 cm gaps between
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
the 99th percentile offset (30.8 cm), places 99.0 % of head positions (by
construction, at that width); 57 %
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
20 do, and two fitted layouts are refused on misfit evidence and stay
inferred (000447's novel W-track and the Z-track, whose fitted lengths are
more than 15 % off the stated ones; G-153). 001845's W-maze was refused for
its thinly sampled base while support was judged per segment; judged per
stretch it is kept. The study measures each declared environment
beside the samples-built one ("Declared environments" in the results). In
the table above figure8_000552 is the camera positions (the 1-D row is
`figure8_000552_linearized`) and jadhav_000447 is the novel W-track (the
familiar one is jadhav_000447_familiar).

A declared track is judged by one function for any declared environment
(`_geometry.judge_declared`), on misfit evidence: a straight stretch of
track (at least 20 bins, an arm's length, not crossing a corner or a
junction) whose samples lie a median of more than one bin to one side, a
stretch with no sample within a bin over more than a quarter of its length
(points half a bin apart; the part within 3 bins of a free end is
reported, not judged, as for walls), no samples near any stretch at all
(a NaN bias is a mismatch, not a pass), samples that
reach past the track's free ends by more than 15 % of its length (summed
over the ends: a track drawn too short), or a stated length more than 15 %
off (`MAX_SIZE_RESIDUAL`, one tolerance for every stated size). A stretch
that cannot grow to 20 bins (a reward arm between a junction and its end)
has its bias and support reported, not judged; support is judged per
stretch, not per segment, so it does not depend on how the graph is cut
into segments. Every declared track and the lattice is also judged as a
whole (`_tracks.frame_offset`): each straight piece of any length with 10
samples near it (assigned as for the stretch bias: nearest segment, within
3 bins) gives the median of its samples' offsets perpendicular to it, and
one translation `s` of the samples is fitted to those medians by least
squares, each piece weighted once (`(sum P_k) s = sum m_k`, `P_k` the
projection across piece k). One step recovered only half to two thirds of
a planted shift on short corridors (samples moved off one are taken by its
neighbours), so the fit is repeated on the samples less the shift found
until a step is below 0.001 bin. No single piece decides: a direction is
read only where, with any one piece left out, the others still add up to
one piece lying across it, and the judged norm is the smallest over the
fits with one piece left out. The judged norm is read over both axes in
2-D and the horizontal ones on the lattice, where a head above the bars
raises the vertical, which is reported. It is read by two estimators, the
single least-squares step and the refitted shift, and neither is right
everywhere: on short corridors and bars the single step reads only half
to two thirds of a planted shift, and on a lateral scatter wider than the
window the refitted window follows the heavier side (G-160). Both more
than one bin is a mismatch; both under it passes this check; when they
straddle it the verdict is "not verified: the frame offset depends on the
estimator", unless another check gives a mismatch. On the test cases a
single step exceeds one bin only for a shift of about 2 bins on a grid of
15 cm corridors and about 3 bins on a tilted lattice of 16 cm bars, so a
layout moved by 1 to 2-3 bins there is not verified, not a mismatch; at
the real recordings' scatter the single step stays under a bin, so this
check gives at most "not verified" on a misplaced layout. The offset is
resolved only below half the smallest spacing between two distinct
parallel pieces (corridors or bars side by side over at least a bin, not
joined along the graph within a bin; from the declared geometry,
`_tracks.piece_spacing`): samples moved farther
are nearer the neighbouring piece and read as its offset, the other way,
so a shift of one spacing reads as none. When the 3-bin reading window is
wider than half that spacing the check passes no verdict ("not verified:
the frame offset cannot be read beyond half the corridor spacing",
G-160), and a refit that has not settled after 50 steps is not verified
either. A shift found by
searching for the one that brings the samples closest to the track (the
median distance) is not used: it moves the track onto the densest band of
samples (+10 cm across 000044's linear track) and has several near-equal
minima on the lattice's repeating bars. A track whose pieces all run
one way (a straight track) cannot have its along-track offset read this
way; the reach past its free ends covers that direction. An outline
is also judged by how far its samples fall short of its side (at most
15 %; they may fall short of the walls but not reach past them). The
share of samples beyond the physical width is not misfit: the tracked
point is usually an LED on a headstage above an unwalled track, which
swings past the edge (and a 2-D camera projects a raised head outward); it
is reported as a tracking measure, split by speed (G-152). Each track is
built with a placement tolerance measured from the samples (twice their
99th percentile distance), so it places 99 % of them by construction; that
share is not evidence of fit. Its bins lie on its own graph, so the share
of bins on the track and the path along it are evidence only against a
reference that is neither built from the samples nor the geometry the
track was declared from.

"Matches" needs a stated fact independent of the samples: a stated
length, diameter or arm length compared with the samples' own, or the
authors' geometry (graph, centreline, coordinate) used as stated. Without
one the verdict is "not verified" and the measurements stand. The limits
are in bins, so in physical units they vary with each recording's bin
(0.5 cell on the labyrinth, 2-4 cm, 5 px on hc-18), and the 20-bin stretch
was set after seeing the data (it was 10 bins until the 000044 circle was
refused on one short side under its reward area). How sensitive the judge
is to misfits of known size is not measured here: that is the survey's
simulation ladder (phase 6, rungs 2-3, planted misfits). Of the 20
declared tracks:

- 10 match, and the frame offset passes on each. On a stated size against the samples' own: Vollan's linear
  track +5.5 %, 000568 +5.1 %, 000165 -1.6 %, 000447's familiar W-track (base -1.2 %, arms +6.5 %),
  hc-28's W-track (base -10.5 %, arms -11.1 %, within the 15 %), 001845's
  W-maze (base +14 %, arms +7 %: close to the limit) and Shipley's eight
  arms (-5.9 to +3.4 %). On the authors' geometry or the file's coordinate
  as stated: J16's graph, hc-31's track coordinate (length +0.3 %),
  and Takigawa's centrelines. Their frame offsets are
  0.42 bin or less by either estimator (refitted: hc-28 0.90 cm,
  Takigawa 1.59 cm of a 4 cm bin); on the single straight tracks (Vollan's, 000568, 000165, hc-31)
  one piece reads nothing, and the stated length, the stretch bias and the
  reach past the ends judge them.
- The 000044 circle is a mismatch: its worst stretch's samples lie 2.50 cm
  to one side of it, against a one-bin limit of 2.0 cm. Its diameter is
  -0.2 % of the stated 1 m and its frame offset passes (1.21 cm refitted,
  0.85 cm in one step). The verdict flipped from matches with the public
  loader's 2 cm bin, not with the data: the limit is one bin, and a judged
  stretch is 20 bins (40 cm, not 50 cm); against the 2.5 cm bin before,
  its worst stretch read 2.38 cm.
- 000044's linear track is not verified: its frame offset depends on the
  estimator. Refitted, it is 3.81 cm across the track (3.63 cm with any
  one of its 3 pieces left out, 1.81 bins of 2 cm); in one step 1.00 cm
  (0.90 cm judged, 0.45 bin). Its samples scatter from -12 to +25 cm across
  it (5th-95th percentile), far wider than the 6 cm (3-bin) window the
  offset is read in, and
  the refitted window follows the heavier side (G-160). It matched on the
  file's linearized coordinate before.
- The labyrinth is not verified: its corridors lie 1 cell apart, and the
  3-bin (1.5-cell) window the frame offset is read in is wider than half
  of that, so a shift past half a cell would read as the next corridor's
  (G-160). Its frame offset reads 0.105 cell (0.21 bin) over 127 pieces,
  under the limit, and it otherwise matches on the authors' graph; it
  matched before the spacing rule. It was a mismatch in the tracks round: its graph ended at the
  centre of each dead-end cell, and the samples reached 15.6 % of its
  length past its ends; the loader now moves each of the 64 dead ends half
  a cell on, to the end wall `MazeWall` draws, and the samples reach 3.0 %
  past the free ends.
- Two verdicts changed with the stretch rule, both from mismatch to
  matches. 000044's linear 15 cm reward-area segments, with no sample
  within a bin over 44 % of their length, are shorter than a judged
  stretch, so their support is reported. Takigawa's stretch 4.12 cm to one
  side of its samples (limit one 4 cm bin) is shorter than 20 bins (80 cm);
  no stretch of its centrelines reaches 20 bins, and none of the
  labyrinth's corridors does (worst 0.49 cell to one side, reported), so on
  these two neither a stretch's bias nor its support is judged. The frame
  offset now judges both as a whole (above): Takigawa passes it
  (+0.94, -2.07) cm over 8 pieces; on the labyrinth it reads (-0.11,
  +0.03) cell over 127 but cannot be read beyond half its corridor
  spacing, so it is not verified.
- 9 are not verified: the labyrinth and 000044's linear track (above), and 7 with no stated size compared and no independent
  reference (the figure-8, the X maze, Agarwal's maze, 001695, hc-3, whose
  pixel scale comes from the same 250 cm), hc-18's circuit, whose shape no
  source states, and Hafting's line, drawn in the file's frame: the frame
  spans exactly the stated 320 cm because the authors scaled it to that
  length, and the loader reads the unit from that span, so the length
  checks nothing. The tracks round called it a match on the authors'
  graph.
- Beyond the physical width, as a tracking measure: J16 63.8 %, hc-28
  37.3 %, hc-3 26.3 %, 000447's familiar W-track 13.0 %, 000568 12.4 %
  (positions as stored), 001845 6.1 %, the labyrinth 2.0 % (one cell,
  chosen; 12.8 % at the paper's 0.75 cell; 4.6 % and 17.9 % before its dead
  ends reached their walls), Shipley 0.7 % of the judged samples. On J16
  the graph is placed right (median signed offset -1.3 to +1.8 cm per
  corridor) and the share falls from 80 % below 10 cm/s (the preprint's
  running speed) to 35 % above 30 cm/s (G-152).
- Against their own geometry the declared environments measure only their
  binning: the labyrinth's path along its bins is 0.96 x the distance along
  the graph it was built from (0.95 before the dead ends were extended),
  Takigawa's 0.97. These are not comparisons
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
be separated, G-053), and 001880 and 000003 (no source read). Since this
study was first run, Gardner et al. 2022 (a 1.5 x 1.5 m box) and Gonzalez &
Giocomo 2024 (000405, 1.3 x 1.5 m) were read for the arena loaders
(BATCH_FAITHFUL_ARENAS.md), and Hafting's archive readme ("a linear track
(mostly 320 cm)"), Shipley et al. 2026 (80 x 10 cm arms on a 24.7 cm
platform), Aery Jones et al. 2021 (a 36 cm track) and Gonzalez et al. 2026
(a linear track with a well at either end; its size is in a supplementary
table not read) for the track loaders (BATCH_FAITHFUL_TRACKS.md).

## Declared environments (open arenas and 3-D)

The open-arena and 3-D loaders declare the stated apparatus
(`geometry["environment"]`), and the study measures that environment, with
the samples-built one beside it against the same outline, bars or room
(table "Declared against samples-built environments" in
[`results/studies/environment_fidelity.md`](results/studies/environment_fidelity.md)).
They are judged by the same function as the tracks
(`_geometry.judge_declared`). A declared outline covers itself by
construction, so it is judged on misfit evidence: samples reaching more than
one bin past a wall (the 95th percentile of their signed distance to each
edge), the samples' extent across the outline more than 15 % short of it, a
stated size more than 15 % from the samples' own fitted size, and a stated
centre more than 15 % of the radius from the samples' centre. The share of
a side (a run of edges turning less than 10 degrees; a circle is one side)
with no sample within 3 bins is reported, not judged, and so is the share
of the lattice's bar segments with no sample within a bin: the tracked LEDs
sit on a headstage above the animal, so head samples fall short of walls
and bars (the maintainer's rule, G-157). The share of samples outside the
outline or room is a tracking measure, reported split by speed, not judged
(the arenas' earlier limit of 2 % is gone). A room is judged as an outline
is, per face: each judged sample is assigned to the face it lies farthest
past (inside, the nearest face), and the 95th percentile of a face's
samples' distance past it more than one bin is a mismatch; and on its
stated sizes. In 3-D a side of a bar is not defined, so no lattice
stretch has a bias; the lattice is judged as a whole as the tracks are
(`_tracks.frame_offset`): each bar segment's median head offset
perpendicular to it, one translation fitted to all of them, its horizontal
part judged at one bin by both estimators, as on the tracks. Its bars are
tilted about 35 degrees, in three directions, so every direction is read;
the vertical part, raised by a head held above the bars, is reported.

Eight of the ten declared arenas and 3-D environments match; the bat's
room is a mismatch and the lattice is not verified:

- On the stated size against the samples' own: 000582 (sides x -2 % and
  y +0 %, of the public loader's square at the samples' midrange), 002030 (diameter +0.0 %), 000405 (-6.9 %, -6.3 %), Gardner
  (-1.9 %, -2.4 %), Vollan's 150 cm circle (diameter -4.0 %, centre 2.2 %
  of the radius off; the samples' extent -14.9 % of the outline, just
  inside the 15 %), Kanter (-2.1 %, -2.3 %), hc-2's platform (+0.0 %,
  -0.8 %) and the water maze (+0.2 %, centre 0.9 % of the radius from the
  stated one). The stated shapes are placed after a radial trim of the
  samples (`_geometry._tracked`), which moved these residuals by up to 2
  points. 0-0.34 % of their judged samples lie outside the outline.
- The lattice's side matches (+0.0 % of the stated 97 cm, the authors'
  bars), but its frame offset depends on the estimator, so it is not
  verified (G-159). Refitted, its head samples sit (-8.2, +4.6) cm
  horizontally off the bars, 9.4 cm or 2.35 bins, and +2.2 cm vertically
  (reported), over 738 bar segments; in one step (-1.4, +0.6) cm, 0.38 bin,
  over 792. On a synthetic tilted lattice moved 1.5 bins, a single step
  reads under a bin too (0.78), so a single step cannot tell this lattice
  from one moved by a few cm. Its parallel bars also lie 16 cm apart,
  closer than twice the 12 cm window the offset is read in, so a shift
  past 8 cm would read as the next bar's and the check passes no verdict
  there in any case (G-160). Until this round its placement was not
  judged and it matched on its side alone.
- The bat's room is a mismatch on its faces: `r_lim`'s sizes match the
  paper's room (x +3.6 %, y +0.0 %, z -8.0 %), but of the 57 flight samples
  nearest the y-low face, 17 lie 1.4 m past it (the perch beyond `r_lim`,
  G-025), so that face's 95th percentile is 1.41 m past the wall against a
  0.2 m limit. 0.11 % of the flight samples lie outside `r_lim` (3.66 % of
  all samples). Until this round a room was judged on its sizes only, and
  a room with every sample 10 m outside had no misfit.
- Support, reported: 71 % of one side of 000405 has no sample within
  7.5 cm, 31 % of Vollan's circle, 48 % of one side of hc-2's platform; on
  the lattice no bar stretch reaches 20 bins (the bars cross every 16 cm),
  so its support is reported per segment (the head is a median 6.0 cm
  from a bar). Before the support rule 000405, Vollan's circle, hc-2 and
  the lattice were mismatches on support alone.

The lattice is built with a placement tolerance chosen from the samples,
2 x the head's 99th percentile distance from a bar (30.8 cm), not a stated
width; the stated tubes are 10 mm across, beyond half of which lie 99.9 %
of head samples. 000987 (stated 80 cm walls, samples 52 cm, G-130), 001880
and 000003 stay built from the samples. Sources, residuals and what changed
in the analyses: [BATCH_FAITHFUL_ARENAS.md](BATCH_FAITHFUL_ARENAS.md).
