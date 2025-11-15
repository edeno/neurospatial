#!/usr/bin/env python
"""Investigation: Can scipy.sparse.csgraph.laplacian replace custom differential operator?

This script compares the current neurospatial implementation (D @ D.T construction)
with scipy.sparse.csgraph.laplacian to determine if they produce identical results.

Key Questions:
1. Does scipy's Laplacian match the D @ D.T construction?
2. Are eigenvalue properties preserved?
3. Do gradient/divergence operators work correctly with scipy's Laplacian?
4. Are there sign or normalization differences?

Author: Investigation for Task 2.4
Date: 2025-11-15
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import laplacian as scipy_laplacian

# Import neurospatial
from neurospatial import Environment
from neurospatial.differential import (
    compute_differential_operator,
    divergence,
    gradient,
)


def create_test_environments():
    """Create multiple test environments for comparison."""

    # Test 1: Simple 1D chain (4 nodes)
    data_1d = np.array([[0.0], [1.0], [2.0], [3.0]])
    env_1d = Environment.from_samples(data_1d, bin_size=1.0)

    # Test 2: Small 2D grid (3×3 = 9 nodes, 4-connected)
    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    xx, yy = np.meshgrid(x, y)
    data_2d_grid = np.column_stack([xx.ravel(), yy.ravel()])
    env_2d_grid = Environment.from_samples(data_2d_grid, bin_size=1.0)

    # Test 3: 2D grid with diagonal connections (8-connected)
    env_2d_diag = Environment.from_samples(
        data_2d_grid, bin_size=1.0, connect_diagonal_neighbors=True
    )

    # Test 4: Irregular graph (plus maze)
    # Create a simple plus-shaped maze
    data_plus = np.array(
        [
            [0.0, 1.0],  # center
            [0.0, 0.0],  # down
            [0.0, 2.0],  # up
            [-1.0, 1.0],  # left
            [1.0, 1.0],  # right
        ]
    )
    env_plus = Environment.from_samples(data_plus, bin_size=0.5)

    return {
        "1D chain": env_1d,
        "2D grid (4-conn)": env_2d_grid,
        "2D grid (8-conn)": env_2d_diag,
        "Plus maze": env_plus,
    }


def compare_laplacians(env, name):
    """Compare neurospatial differential operator Laplacian with scipy Laplacian.

    Returns
    -------
    dict
        Comparison results including max difference, eigenvalue comparison, etc.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")
    print(f"Bins: {env.n_bins}, Edges: {len(env.connectivity.edges)}")

    # =========================================================================
    # Method 1: neurospatial custom implementation (D @ D.T)
    # =========================================================================
    diff_op = compute_differential_operator(env)
    laplacian_custom = (diff_op @ diff_op.T).toarray()

    print("\nCustom Laplacian (D @ D.T):")
    print(f"  Shape: {laplacian_custom.shape}")
    print(f"  Sparse: {type(diff_op)}")

    # =========================================================================
    # Method 2: scipy.sparse.csgraph.laplacian (unnormalized)
    # =========================================================================
    # Convert NetworkX graph to scipy sparse adjacency matrix
    adj_matrix = nx.to_scipy_sparse_array(
        env.connectivity, weight="distance", format="csr"
    )

    # Compute scipy Laplacian (unnormalized)
    laplacian_scipy = scipy_laplacian(adj_matrix, normed=False, copy=True, form="array")

    # Handle sparse vs dense output
    if sparse.issparse(laplacian_scipy):
        laplacian_scipy = laplacian_scipy.toarray()

    print("\nscipy Laplacian (degree - adjacency):")
    print(f"  Shape: {laplacian_scipy.shape}")
    print(f"  Input adjacency shape: {adj_matrix.shape}")

    # =========================================================================
    # Method 3: NetworkX Laplacian (reference)
    # =========================================================================
    laplacian_nx = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()

    print("\nNetworkX Laplacian (reference):")
    print(f"  Shape: {laplacian_nx.shape}")

    # =========================================================================
    # Comparison: Custom vs scipy
    # =========================================================================
    diff_custom_scipy = np.abs(laplacian_custom - laplacian_scipy)
    max_diff_custom_scipy = np.max(diff_custom_scipy)

    print("\n--- Comparison: Custom vs scipy ---")
    print(f"Max absolute difference: {max_diff_custom_scipy:.2e}")
    print(
        f"Matrices identical (tol=1e-10): {np.allclose(laplacian_custom, laplacian_scipy, atol=1e-10)}"
    )

    # =========================================================================
    # Comparison: Custom vs NetworkX (sanity check)
    # =========================================================================
    diff_custom_nx = np.abs(laplacian_custom - laplacian_nx)
    max_diff_custom_nx = np.max(diff_custom_nx)

    print("\n--- Comparison: Custom vs NetworkX (sanity check) ---")
    print(f"Max absolute difference: {max_diff_custom_nx:.2e}")
    print(
        f"Matrices identical (tol=1e-10): {np.allclose(laplacian_custom, laplacian_nx, atol=1e-10)}"
    )

    # =========================================================================
    # Comparison: scipy vs NetworkX
    # =========================================================================
    diff_scipy_nx = np.abs(laplacian_scipy - laplacian_nx)
    max_diff_scipy_nx = np.max(diff_scipy_nx)

    print("\n--- Comparison: scipy vs NetworkX ---")
    print(f"Max absolute difference: {max_diff_scipy_nx:.2e}")
    print(
        f"Matrices identical (tol=1e-10): {np.allclose(laplacian_scipy, laplacian_nx, atol=1e-10)}"
    )

    # =========================================================================
    # Eigenvalue analysis
    # =========================================================================
    print("\n--- Eigenvalue Analysis ---")

    # Compute eigenvalues for custom Laplacian
    eigvals_custom = np.linalg.eigvalsh(laplacian_custom)
    eigvals_custom_sorted = np.sort(eigvals_custom)

    # Compute eigenvalues for scipy Laplacian
    eigvals_scipy = np.linalg.eigvalsh(laplacian_scipy)
    eigvals_scipy_sorted = np.sort(eigvals_scipy)

    print(f"Custom Laplacian eigenvalues (smallest 5): {eigvals_custom_sorted[:5]}")
    print(f"scipy Laplacian eigenvalues (smallest 5):  {eigvals_scipy_sorted[:5]}")

    # Check if smallest eigenvalue is near zero (connected graph)
    print("\nSmallest eigenvalue (should be ≈0 for connected graph):")
    print(f"  Custom: {eigvals_custom_sorted[0]:.2e}")
    print(f"  scipy:  {eigvals_scipy_sorted[0]:.2e}")

    # Check if eigenvalues match
    eigval_diff = np.abs(eigvals_custom_sorted - eigvals_scipy_sorted)
    max_eigval_diff = np.max(eigval_diff)
    print(f"\nMax eigenvalue difference: {max_eigval_diff:.2e}")
    print(
        f"Eigenvalues match (tol=1e-10): {np.allclose(eigvals_custom_sorted, eigvals_scipy_sorted, atol=1e-10)}"
    )

    # =========================================================================
    # Gradient/Divergence test
    # =========================================================================
    print("\n--- Gradient/Divergence Consistency Check ---")

    # Create a test scalar field (random values)
    np.random.seed(42)
    test_field = np.random.randn(env.n_bins)

    # Compute Laplacian applied to field using custom implementation
    laplacian_field_custom = laplacian_custom @ test_field

    # Compute Laplacian applied to field using scipy
    laplacian_field_scipy = laplacian_scipy @ test_field

    # Compute Laplacian via gradient → divergence
    grad_field = gradient(env, test_field)
    laplacian_field_grad_div = divergence(env, grad_field)

    print(
        f"Laplacian @ field via custom:    L2 norm = {np.linalg.norm(laplacian_field_custom):.6f}"
    )
    print(
        f"Laplacian @ field via scipy:     L2 norm = {np.linalg.norm(laplacian_field_scipy):.6f}"
    )
    print(
        f"Laplacian @ field via grad→div:  L2 norm = {np.linalg.norm(laplacian_field_grad_div):.6f}"
    )

    diff_custom_grad_div = np.abs(laplacian_field_custom - laplacian_field_grad_div)
    print(f"\nMax diff (custom vs grad→div): {np.max(diff_custom_grad_div):.2e}")
    print(
        f"grad→div matches custom (tol=1e-10): {np.allclose(laplacian_field_custom, laplacian_field_grad_div, atol=1e-10)}"
    )

    diff_scipy_grad_div = np.abs(laplacian_field_scipy - laplacian_field_grad_div)
    print(f"\nMax diff (scipy vs grad→div): {np.max(diff_scipy_grad_div):.2e}")
    print(
        f"grad→div matches scipy (tol=1e-10): {np.allclose(laplacian_field_scipy, laplacian_field_grad_div, atol=1e-10)}"
    )

    # =========================================================================
    # Return results
    # =========================================================================
    return {
        "name": name,
        "n_bins": env.n_bins,
        "n_edges": len(env.connectivity.edges),
        "max_diff_custom_scipy": max_diff_custom_scipy,
        "max_diff_custom_nx": max_diff_custom_nx,
        "max_diff_scipy_nx": max_diff_scipy_nx,
        "matrices_match": np.allclose(laplacian_custom, laplacian_scipy, atol=1e-10),
        "eigvals_match": np.allclose(
            eigvals_custom_sorted, eigvals_scipy_sorted, atol=1e-10
        ),
        "grad_div_matches_custom": np.allclose(
            laplacian_field_custom, laplacian_field_grad_div, atol=1e-10
        ),
        "grad_div_matches_scipy": np.allclose(
            laplacian_field_scipy, laplacian_field_grad_div, atol=1e-10
        ),
        "smallest_eigval_custom": eigvals_custom_sorted[0],
        "smallest_eigval_scipy": eigvals_scipy_sorted[0],
    }


def test_normalized_laplacian(env, name):
    """Test scipy's normalized Laplacian option.

    The normalized Laplacian is L_norm = I - D^(-1/2) A D^(-1/2),
    which is useful for spectral clustering and graph partitioning.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Normalized Laplacian: {name}")
    print(f"{'=' * 60}")

    # Get adjacency matrix
    adj_matrix = nx.to_scipy_sparse_array(
        env.connectivity, weight="distance", format="csr"
    )

    # Compute normalized Laplacian
    laplacian_scipy_norm = scipy_laplacian(
        adj_matrix, normed=True, copy=True, form="array"
    )
    if sparse.issparse(laplacian_scipy_norm):
        laplacian_scipy_norm = laplacian_scipy_norm.toarray()

    # Compare with NetworkX normalized Laplacian
    laplacian_nx_norm = nx.normalized_laplacian_matrix(
        env.connectivity, weight="distance"
    ).toarray()

    diff_norm = np.abs(laplacian_scipy_norm - laplacian_nx_norm)
    max_diff_norm = np.max(diff_norm)

    print(f"scipy normalized Laplacian shape: {laplacian_scipy_norm.shape}")
    print(f"NetworkX normalized Laplacian shape: {laplacian_nx_norm.shape}")
    print(f"Max absolute difference: {max_diff_norm:.2e}")
    print(
        f"Matrices match (tol=1e-10): {np.allclose(laplacian_scipy_norm, laplacian_nx_norm, atol=1e-10)}"
    )

    # Eigenvalue analysis
    eigvals_scipy_norm = np.linalg.eigvalsh(laplacian_scipy_norm)
    eigvals_scipy_norm_sorted = np.sort(eigvals_scipy_norm)

    print(
        f"\nNormalized Laplacian eigenvalues (smallest 5): {eigvals_scipy_norm_sorted[:5]}"
    )
    print(
        "Note: Normalized Laplacian eigenvalues should be in [0, 2] for undirected graphs"
    )
    print(
        f"Smallest: {eigvals_scipy_norm_sorted[0]:.6f}, Largest: {eigvals_scipy_norm_sorted[-1]:.6f}"
    )

    return {
        "normalized_matches_nx": np.allclose(
            laplacian_scipy_norm, laplacian_nx_norm, atol=1e-10
        ),
        "eigvals_in_range": np.all(
            (eigvals_scipy_norm_sorted >= -1e-10)
            & (eigvals_scipy_norm_sorted <= 2.0 + 1e-10)
        ),
    }


def run_investigation():
    """Run full investigation and print summary."""

    print("=" * 80)
    print("scipy.sparse.csgraph.laplacian Investigation")
    print("Task 2.4: Can scipy replace custom differential operator Laplacian?")
    print("=" * 80)

    # Create test environments
    test_envs = create_test_environments()

    # Run comparisons
    results = []
    for name, env in test_envs.items():
        result = compare_laplacians(env, name)
        results.append(result)

        # Test normalized Laplacian
        norm_result = test_normalized_laplacian(env, name)
        result.update(norm_result)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_pass = True
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Nodes: {result['n_bins']}, Edges: {result['n_edges']}")
        print(
            f"  Custom vs scipy match: {result['matrices_match']} (max diff: {result['max_diff_custom_scipy']:.2e})"
        )
        print(f"  Eigenvalues match: {result['eigvals_match']}")
        print(f"  grad→div consistent with custom: {result['grad_div_matches_custom']}")
        print(f"  grad→div consistent with scipy: {result['grad_div_matches_scipy']}")
        print(
            f"  Normalized Laplacian matches NetworkX: {result['normalized_matches_nx']}"
        )

        # Check if all tests passed
        if not (
            result["matrices_match"]
            and result["eigvals_match"]
            and result["grad_div_matches_custom"]
            and result["grad_div_matches_scipy"]
            and result["normalized_matches_nx"]
        ):
            all_pass = False
            print("  ⚠️ SOME TESTS FAILED")
        else:
            print("  ✅ ALL TESTS PASSED")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if all_pass:
        print(
            "✅ scipy.sparse.csgraph.laplacian produces IDENTICAL results to custom implementation"
        )
        print("✅ Eigenvalue properties PRESERVED")
        print("✅ Gradient/divergence operators WORK CORRECTLY")
        print("✅ Normalized Laplacian option AVAILABLE and matches NetworkX")
        print("\nCONCLUSION: scipy.sparse.csgraph.laplacian is a SUITABLE replacement")
        print("BENEFITS:")
        print("  - Standard library implementation (more reliable)")
        print("  - Normalized Laplacian option (useful for spectral clustering)")
        print("  - Potentially faster (C implementation)")
        print("  - Well-tested and maintained")
        print("\nHOWEVER:")
        print(
            "  - Current implementation requires differential operator D for gradient/divergence"
        )
        print("  - scipy only provides Laplacian L = D @ D.T, not D itself")
        print(
            "  - To replace compute_differential_operator(), need to keep D construction OR"
        )
        print("    reimplement gradient/divergence using Laplacian directly")
        print("\nRECOMMENDATION:")
        print(
            "  - Keep current differential operator implementation for gradient/divergence"
        )
        print(
            "  - Optionally add scipy Laplacian as alternative for Laplacian-only operations"
        )
        print(
            "  - Document that L = D @ D.T = scipy.sparse.csgraph.laplacian(adjacency)"
        )
    else:
        print("❌ scipy.sparse.csgraph.laplacian DIFFERS from custom implementation")
        print("⚠️ Further investigation needed before replacement")

    return results


if __name__ == "__main__":
    results = run_investigation()
