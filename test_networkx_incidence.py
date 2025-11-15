#!/usr/bin/env python
"""Quick test: Does NetworkX incidence_matrix match neurospatial differential operator?"""

import networkx as nx
import numpy as np

from neurospatial import Environment
from neurospatial.differential import compute_differential_operator

# Create simple test environment
data_1d = np.array([[0.0], [1.0], [2.0], [3.0]])
env = Environment.from_samples(data_1d, bin_size=1.0)

print("=" * 60)
print("NetworkX incidence_matrix vs neurospatial differential operator")
print("=" * 60)

# NetworkX incidence matrix (oriented)
M_nx = nx.incidence_matrix(env.connectivity, oriented=True, weight="distance")
M_nx_dense = M_nx.toarray()

# neurospatial differential operator
D_neurospatial = compute_differential_operator(env)
D_dense = D_neurospatial.toarray()

print("\nNetworkX incidence matrix (oriented=True, weight='distance'):")
print(f"Shape: {M_nx_dense.shape}")
print(M_nx_dense)

print("\nneurospatial differential operator:")
print(f"Shape: {D_dense.shape}")
print(D_dense)

print("\n--- Comparison ---")
print(f"Shapes match: {M_nx_dense.shape == D_dense.shape}")
print(f"Matrices identical: {np.allclose(M_nx_dense, D_dense)}")
print(f"Max absolute difference: {np.max(np.abs(M_nx_dense - D_dense)):.6f}")

# Check the key difference: sqrt(weight) vs weight
print("\n--- Key Difference ---")
edge_weights = [
    env.connectivity.edges[u, v]["distance"] for u, v in env.connectivity.edges()
]
print(f"Edge weights: {edge_weights}")
print(f"sqrt(Edge weights): {np.sqrt(edge_weights)}")

print("\nNetworkX uses: ±weight directly")
print("neurospatial uses: ±sqrt(weight)")

# Verify Laplacian construction
L_from_nx = M_nx @ M_nx.T
L_from_neurospatial = D_neurospatial @ D_neurospatial.T
L_networkx_reference = nx.laplacian_matrix(env.connectivity, weight="distance")

print("\n--- Laplacian Verification ---")
print("L from NetworkX incidence: M @ M.T")
print(L_from_nx.toarray())

print("\nL from neurospatial: D @ D.T")
print(L_from_neurospatial.toarray())

print("\nL from NetworkX Laplacian (reference):")
print(L_networkx_reference.toarray())

print("\n--- Which is correct? ---")
print(
    f"M @ M.T matches NetworkX Laplacian: {np.allclose(L_from_nx.toarray(), L_networkx_reference.toarray())}"
)
print(
    f"D @ D.T matches NetworkX Laplacian: {np.allclose(L_from_neurospatial.toarray(), L_networkx_reference.toarray())}"
)

print(f"\n{'=' * 60}")
print("ANSWER: NetworkX provides incidence matrix, but...")
print("=" * 60)
print("✅ NetworkX has nx.incidence_matrix(G, oriented=True, weight='distance')")
print("❌ It uses ±weight, not ±sqrt(weight)")
print("✅ neurospatial uses ±sqrt(weight) to ensure L = D @ D.T is correct")
print("\nVerdict: NetworkX incidence_matrix is NOT a drop-in replacement")
print("Reason: Different weight handling (sqrt vs direct)")
