"""Experiment (a): is neurospatial's diffusion_kde bandwidth grid-size-independent?

Hypothesis: neurospatial's `env.compute_kernel(bandwidth, mode="density")` folds sigma
into the graph edge weight (w = exp(-d^2/2 sigma^2)) AND folds the mass matrix into the
Laplacian (L_vol = M^-1 L) before exponentiating. Both couple the *effective* smoothing
width to the bin spacing h, so the same `bandwidth` produces different physical smoothing
at different bin sizes.

Prediction (grid-DEPENDENT): the measured Gaussian std of a smoothed point source varies
with bin size h.

Contrast: an NLD-style kernel -- plain finite-difference Laplacian with w = 1/d^2 on the
face-adjacent graph, exp(-t L), t = sigma^2/2, density-normalized AFTER -- should give a
measured std == bandwidth, independent of h.

Method: smooth a unit point source at the domain center; measure the physical std of the
resulting (probability-normalized) kernel column. Probability-normalizing a single column
is a per-column scalar, so it does not change the column's shape/std -- it just makes the
second moment well-defined across the density/transition kernels being compared.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from neurospatial import Environment


def measure_std_along_axis0(kernel, centers, bin_sizes, center_bin):
    """Physical std of the smoothed point source along axis 0 (the x axis).

    kernel[:, center_bin] is the smoothed unit mass at `center_bin`. Normalize that
    column to a probability distribution (per-column scalar; does not change its shape),
    then return sqrt(sum_i p_i (x_i - x_c)^2). For an isotropic Gaussian of std sigma the
    x-marginal std is sigma, so a grid-independent kernel returns `bandwidth` at every h.
    """
    col = np.asarray(kernel[:, center_bin], dtype=float)
    col = np.clip(col, 0.0, None)
    p = col / col.sum()
    x = centers[:, 0]
    xc = centers[center_bin, 0]
    mean = np.sum(p * x)
    var = np.sum(p * (x - xc) ** 2)
    return float(np.sqrt(var)), float(mean - xc)  # std, and mean-offset (should be ~0)


def nld_style_kernel(env, bandwidth, *, face_adjacent_only=True):
    """Plain finite-difference heat kernel: w = 1/d^2, exp(-t L), t = sigma^2/2.

    No mass matrix folded into L (unlike neurospatial's density mode); density
    normalization, if wanted, is applied to the *output*, not the operator. This mirrors
    non_local_detector.likelihoods.diffusion.build_laplacian + diffuse.
    """
    g = env.connectivity
    centers = env.bin_centers
    n = env.n_bins
    rows, cols, vals = [], [], []
    for u, v, data in g.edges(data=True):
        if u >= n or v >= n:
            continue
        offset = centers[u] - centers[v]
        if face_adjacent_only and np.count_nonzero(np.abs(offset) > 1e-9) != 1:
            continue  # drop diagonal / Moore edges (oversmooth with 1/d^2 weighting)
        d = float(data["distance"])
        w = 1.0 / d**2
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]
    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(W.sum(axis=1)).ravel()
    L = scipy.sparse.diags(degree) - W
    t = bandwidth**2 / 2.0
    K = scipy.sparse.linalg.expm(-t * L)
    if hasattr(K, "toarray"):
        K = K.toarray()
    return np.clip(K, 0.0, None)


def build_1d_env(bin_size, domain=(0.0, 100.0), seed=0):
    """A dense-coverage 1-D regular grid over `domain` at spacing `bin_size`."""
    rng = np.random.default_rng(seed)
    # Dense uniform samples so every bin in [domain] is marked active.
    n = int((domain[1] - domain[0]) / bin_size) * 200
    pos = rng.uniform(domain[0], domain[1], size=(n, 1))
    return Environment.from_samples(pos, bin_size=bin_size)


def build_2d_env(bin_size, domain=(0.0, 60.0), seed=0):
    rng = np.random.default_rng(seed)
    n = int(((domain[1] - domain[0]) / bin_size) ** 2) * 60
    pos = rng.uniform(domain[0], domain[1], size=(n, 2))
    return Environment.from_samples(pos, bin_size=bin_size)


def center_bin(env, target):
    return int(np.argmin(np.linalg.norm(env.bin_centers - np.asarray(target), axis=1)))


def run(dim, bandwidths, bin_sizes):
    print(
        f"\n{'=' * 78}\n{dim}-D point-source smoothing: measured physical std vs bin size"
    )
    print("(grid-INDEPENDENT kernel => 'measured std' == bandwidth at every bin size)")
    print("=" * 78)
    for bw in bandwidths:
        print(f"\nbandwidth (requested sigma) = {bw}")
        header = (
            f"  {'bin_size':>8} {'n_bins':>7} {'edge_d':>7} "
            f"{'ns_density':>11} {'ns_transit':>11} {'nld_1/d^2':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for h in bin_sizes:
            if dim == 1:
                env = build_1d_env(h)
                target = [50.0]
            else:
                env = build_2d_env(h)
                target = [30.0, 30.0]
            cb = center_bin(env, target)
            centers = env.bin_centers
            bin_sizes_arr = env.bin_sizes
            # median edge distance (characteristic spacing)
            ed = np.median(
                [d["distance"] for _, _, d in env.connectivity.edges(data=True)]
            )
            k_dens = env.compute_kernel(bw, mode="density", cache=False)
            k_tran = env.compute_kernel(bw, mode="transition", cache=False)
            k_nld = nld_style_kernel(env, bw, face_adjacent_only=(dim > 1))
            s_dens, _ = measure_std_along_axis0(k_dens, centers, bin_sizes_arr, cb)
            s_tran, _ = measure_std_along_axis0(k_tran, centers, bin_sizes_arr, cb)
            s_nld, _ = measure_std_along_axis0(k_nld, centers, bin_sizes_arr, cb)
            print(
                f"  {h:>8.2f} {env.n_bins:>7d} {ed:>7.2f} "
                f"{s_dens:>11.3f} {s_tran:>11.3f} {s_nld:>10.3f}"
            )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore"
    )  # silence large-kernel GB warnings for readability
    run(dim=1, bandwidths=[5.0, 10.0], bin_sizes=[0.5, 1.0, 2.0, 4.0])
    run(dim=2, bandwidths=[5.0], bin_sizes=[1.0, 2.0, 3.0, 5.0])
