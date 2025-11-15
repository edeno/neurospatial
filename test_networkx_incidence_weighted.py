#!/usr/bin/env python
"""Test NetworkX incidence_matrix with NON-UNIFORM edge weights"""

import networkx as nx
import numpy as np

from neurospatial import Environment
from neurospatial.differential import compute_differential_operator

# Create environment with non-uniform spacing (varying edge weights)
data_nonuniform = np.array([[0.0], [1.0], [3.0], [6.0]])  # Spacing: 1, 2, 3
env = Environment.from_samples(
    data_nonuniform, bin_size=3.5
)  # Large enough to connect all

print("=" * 60)
print("NetworkX incidence vs neurospatial differential (NON-UNIFORM WEIGHTS)")
print("=" * 60)

# Get edge weights
edge_weights = [
    env.connectivity.edges[u, v]["distance"] for u, v in env.connectivity.edges()
]
print(f"\nEdge weights (distances): {edge_weights}")
print(f"sqrt(Edge weights): {np.sqrt(edge_weights)}")

# NetworkX incidence matrix (oriented, with weight)
M_nx = nx.incidence_matrix(env.connectivity, oriented=True, weight="distance")
M_nx_dense = M_nx.toarray()

# neurospatial differential operator
D_neurospatial = compute_differential_operator(env)
D_dense = D_neurospatial.toarray()

print("\nNetworkX incidence matrix (uses ±weight directly):")
print(M_nx_dense)

print("\nneurospatial differential operator (uses ±sqrt(weight)):")
print(D_dense)

print("\n--- Difference Visualization ---")
diff = np.abs(M_nx_dense - D_dense)
print("Absolute difference:")
print(diff)
print(f"Max difference: {np.max(diff):.6f}")

# Verify Laplacian construction
L_from_nx = M_nx @ M_nx.T
L_from_neurospatial = D_neurospatial @ D_neurospatial.T
L_networkx_reference = nx.laplacian_matrix(env.connectivity, weight="distance")

print("\n--- Laplacian Verification ---")
print("L from NetworkX incidence (M @ M.T):")
print(L_from_nx.toarray())

print("\nL from neurospatial (D @ D.T):")
print(L_from_neurospatial.toarray())

print("\nL from NetworkX reference (nx.laplacian_matrix):")
print(L_networkx_reference.toarray())

print("\n--- Which Laplacian is Correct? ---")
nx_incidence_matches = np.allclose(L_from_nx.toarray(), L_networkx_reference.toarray())
neurospatial_matches = np.allclose(
    L_from_neurospatial.toarray(), L_networkx_reference.toarray()
)

print(f"NetworkX: M @ M.T == nx.laplacian_matrix: {nx_incidence_matches}")
print(f"neurospatial: D @ D.T == nx.laplacian_matrix: {neurospatial_matches}")

if neurospatial_matches and not nx_incidence_matches:
    print("\n⚠️ IMPORTANT: Only neurospatial's sqrt(weight) gives correct Laplacian!")
    print("NetworkX incidence_matrix with weights does NOT satisfy L = M @ M.T")
elif nx_incidence_matches and not neurospatial_matches:
    print("\n⚠️ UNEXPECTED: NetworkX incidence gives correct Laplacian?")
elif both_match := (nx_incidence_matches and neurospatial_matches):
    print("\n✅ Both match (likely uniform weights)")

print(f"\n{'=' * 60}")
print("CONCLUSION")
print("=" * 60)
print("✅ NetworkX provides: nx.incidence_matrix(G, oriented=True, weight='distance')")
print("❌ It uses ±weight directly, NOT ±sqrt(weight)")
print("❌ Therefore: M @ M.T ≠ Laplacian for non-uniform weights")
print("✅ neurospatial uses ±sqrt(weight) to ensure D @ D.T = Laplacian")
print("\nVerdict: NetworkX incidence_matrix is NOT a suitable replacement")
