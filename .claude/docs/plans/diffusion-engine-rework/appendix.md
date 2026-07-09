# Appendix — external references

[← back to PLAN.md](PLAN.md)

## NLD junction contraction

Reference implementation for [D2](designs.md#d2-graph-junction-contraction) (along-track
distance between bin-centers through junction nodes):

- Repo: `~/Documents/GitHub/non_local_detector`, branch **`sorted-spikes-diffusion`**.
- `src/non_local_detector/likelihoods/diffusion.py::_neighbor_centers` — Dijkstra from a
  start bin-center over the substrate graph (bin-center, bin-edge, junction nodes), treating
  every other bin-center as a sink (recorded, not expanded through); returns
  `(center_node, path_distance)` pairs. Adapt to neurospatial's `GraphLayout` substrate.
- Same file: `build_laplacian` (finite-difference `w=1/d²`), `diffuse` (heat kernel in the
  eigenbasis — PR2 analogue), `to_density` (post-hoc density normalization). These informed
  the finite-volume operator (spec §3.A) and are the PR2 performance reference (cached
  eigenbasis + bandwidth-aware truncation), **not** used in PR1.

## Finite-volume / TPFA background

- Two-point flux approximation (TPFA) cell-centered finite-volume: flux weight `A/d` (shared
  face measure ÷ center distance), cell-volume mass `M`; continuum limit `−∇²` on
  K-orthogonal meshes (center-to-center line ⟂ shared face). All neurospatial lattices
  (Cartesian, hex, polar sectors, 1D track) are K-orthogonal; triangle-centroid meshes are
  only approximately so (spec §3.E, E2 + skew guard).
- Circumcentric-Voronoi finite volume (the exact-on-any-Delaunay alternative, E1) is held in
  reserve; see [design-correctness.md](design-correctness.md) §3.E.
