"""Spatial field operations for Environment class.

This module provides methods for processing spatial fields, including
kernel computation, smoothing, and interpolation.

Key Features
------------
- Diffusion kernel computation with caching
- Graph-based field smoothing
- Field interpolation at arbitrary points (nearest/linear modes)

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._protocols import EnvironmentProtocol, SelfEnv
from neurospatial.environment.decorators import check_fitted, versioned_cached_property

if TYPE_CHECKING:
    pass
    from neurospatial import Environment


class EnvironmentFields:
    """Spatial field operations mixin.

    Provides methods for processing fields over spatial environments.
    """

    @check_fitted
    def compute_kernel(
        self: SelfEnv,
        bandwidth: float,
        *,
        mode: Literal["transition", "density", "average"] = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """Compute the finite-volume diffusion kernel for smoothing operations.

        Wrapper around :func:`neurospatial.ops.diffusion.diffusion_kernel` that
        resolves this environment's layout geometry and cell volumes, so
        ``bandwidth`` is the true physical standard deviation (σ) of the
        smoothing on any supported layout, independent of bin size.

        Parameters
        ----------
        bandwidth : float
            Smoothing bandwidth in physical units (the standard deviation σ of
            the diffusion), must be > 0. This is the true physical σ: a
            point source smoothed by this kernel has physical standard
            deviation ``bandwidth`` regardless of bin size.
        mode : {'transition', 'density', 'average'}, default='density'
            Kernel orientation, distinguished by the input field type:

            - 'transition': ``Hᵀ`` — column-stochastic (``sum_i K[i, j] = 1``).
              ``K @ field`` conserves ``sum(field)``; use for **extensive**
              quantities (occupancy, spike counts, discrete probability *mass*).
            - 'density': ``H·M⁻¹`` — M-weighted columns integrate to 1
              (``sum_i M_i K[i, j] = 1``). Takes an **extensive** input (counts)
              and returns a **density** (KDE). Do NOT apply to an already
              intensive field (a rate map): on non-uniform bin volumes that
              divides by cell volume twice.
            - 'average': ``H`` — row-stochastic (``sum_j K[i, j] = 1``).
              ``K @ field`` averages an **intensive** field (a rate map or
              probability *density*), volume-unbiased on non-uniform ``M``. Not
              for discrete probability *mass* (use 'transition').
        cache : bool, default=True
            If True, cache the computed kernel for reuse. Subsequent calls
            with the same (bandwidth, mode) will return the cached result.

        Returns
        -------
        kernel : NDArray[np.float64], shape (n_bins, n_bins)
            Diffusion kernel matrix.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If bandwidth is not positive, or ``mode`` is not one of
            ``{'transition', 'density', 'average'}``.

        See Also
        --------
        neurospatial.ops.diffusion.diffusion_kernel :
            Lower-level function operating directly on an environment.
        neurospatial.ops.smoothing.compute_diffusion_kernels :
            Graph-level primitive with explicit face measures.

        Notes
        -----
        The kernel is the finite-volume heat operator ``H = exp(-t L)`` with
        ``t = σ² / 2`` and ``L = M⁻¹ (D − W)``, ``W[i, j] = A[i, j] / d[i, j]``
        (``A`` the shared-face measure, ``d`` the center distance, ``M`` the
        per-bin cell volumes). On any K-orthogonal layout ``L`` has the
        continuum limit ``−∇²``, so the smoothing width equals ``σ``.

        **Memory cost is O(n²).** The diffusion heat kernel ``exp(-t L)`` of a
        connected graph is *dense by construction* (every entry > 0), so the
        returned matrix occupies ``n_bins**2 * 8`` bytes of float64 memory —
        for example ``20000**2 * 8 / 1e9 ≈ 3.2 GB`` at 20,000 bins. This peak
        cannot be avoided while using the dense diffusion kernel. There is no
        hard limit: a ``UserWarning`` estimating the GB is issued above 3,000
        bins and the kernel is then built regardless of size. The matrix
        exponential is also O(n³) in time, so large environments are slow as
        well as memory-hungry.

        For large environments, reduce the number of bins by increasing
        ``bin_size`` when constructing the environment. (Every
        ``smoothing_method`` in the higher-level encoding functions -- including
        ``"binned"`` -- builds this dense kernel, so switching method is not a
        memory mitigation.)

        The physical-σ guarantee assumes uniform bin spacing per axis (the
        standard grid, hex, polar-sector, graph, and mesh layouts). A custom
        *nonuniform* Cartesian ``grid_edges`` inherits a uniform-cell
        approximation for both the face measure and the cell volume, so it is
        outside this guarantee (a tracked follow-up).

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> # Compute kernel for smoothing
        >>> kernel = env.compute_kernel(bandwidth=5.0, mode="density")  # doctest: +SKIP
        >>> # Apply to field
        >>> smoothed_field = kernel @ field  # doctest: +SKIP

        """
        from neurospatial.ops.diffusion import diffusion_kernel

        # Validate mode.
        valid_modes = {"transition", "density", "average"}
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes} (got '{mode}'). "
                "Use 'transition' for mass-conserving smoothing of extensive "
                "quantities, 'density' for count-to-density (KDE), or 'average' "
                "for volume-unbiased averaging of an intensive field (rate map)."
            )

        # Initialize cache if it doesn't exist
        # (for backward compatibility with environments deserialized from older versions)
        if not hasattr(self, "_kernel_cache"):
            self._kernel_cache = {}

        # Check cache first if enabled
        cache_key = (bandwidth, mode)
        if cache and cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # Compute kernel. diffusion_kernel reads env.bin_sizes internally as the
        # canonical cell-volume mass matrix M (used by every mode).
        from typing import cast

        kernel = diffusion_kernel(
            cast("EnvironmentProtocol", self), bandwidth, mode=mode
        )

        # Store in cache if enabled
        if cache:
            self._kernel_cache[cache_key] = kernel

        return kernel

    def smooth(
        self: SelfEnv,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: Literal["transition", "density", "average"] = "density",
    ) -> NDArray[np.float64]:
        """Apply diffusion kernel smoothing to a field.

        This method smooths bin-valued fields using diffusion kernels computed
        via the graph Laplacian. It works uniformly across all layout types
        (grids, graphs, meshes) and respects the connectivity structure.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values per bin to smooth. Must be a 1-D array with length
            equal to n_bins.
        bandwidth : float
            Smoothing bandwidth in physical units (σ). Controls the scale
            of spatial smoothing. Must be positive.
        mode : {'transition', 'density', 'average'}, default='density'
            Smoothing mode, distinguished by the **input type**. The default
            stays ``'density'`` for backward compatibility; ``'average'`` is the
            correct choice for an intensive rate map.

            - 'transition': Mass-conserving smoothing of an **extensive**
              quantity (a total per bin). Total sum is preserved:
              ``smoothed.sum() == field.sum()``. Use for occupancy, spike
              counts, and discrete probability *mass* (a posterior summing to 1
              stays summing to 1).
            - 'density': Count → density (KDE). Takes an **extensive** input
              (counts / occupancy) and returns a density whose integral under
              bin volumes equals the input total: ``sum(smoothed * bin_sizes)
              == sum(field)`` (each kernel column integrates to 1 under bin
              volumes). Do **not** apply to an already **intensive** field (a
              rate map or probability *density*): on non-uniform bin volumes
              that divides by cell volume twice.
            - 'average': Volume-unbiased averaging of an **intensive** field (a
              rate map or probability *density*). Row-stochastic ``H``, so a
              constant field is preserved and there is no cell-volume bias on
              non-uniform ``M``. Not for discrete probability *mass* (use
              'transition').

        Returns
        -------
        smoothed : NDArray[np.float64], shape (n_bins,)
            Smoothed field values.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If field has wrong shape, wrong dimensionality, bandwidth is not
            positive, or mode is invalid.

        See Also
        --------
        compute_kernel : Compute the smoothing kernel explicitly.
        occupancy : Compute occupancy with optional smoothing.

        Notes
        -----
        The smoothing operation is:

        .. math::
            \\text{smoothed} = K \\cdot \\text{field}

        where :math:`K` is the diffusion kernel computed via matrix exponential
        of the graph Laplacian.

        For mode='transition', mass is conserved:

        .. math::
            \\sum_i \\text{smoothed}_i = \\sum_i \\text{field}_i

        For mode='density', the kernel maps an extensive input (counts) to a
        density whose integral under bin volumes equals the input total
        (``sum(smoothed * bin_sizes) == sum(field)``); it must not be applied to
        an already-intensive field (see the ``mode`` parameter).

        For mode='average', the kernel is row-stochastic, so a constant field is
        preserved and an intensive field (rate map) is averaged without the
        cell-volume bias that ``density`` would introduce on non-uniform ``M``.

        ``bandwidth`` is the true physical standard deviation (σ) of the
        smoothing on any supported layout, independent of bin size.

        **Implementation.** Smoothing is applied matrix-free via
        :meth:`diffuse` (a cached truncated eigenbasis), so it never materializes
        the dense ``(n_bins, n_bins)`` kernel and scales to large/fine grids. The
        eigenbasis is cached on the environment and reused across bandwidths,
        modes, and calls.

        **Approximation contract.** ``smooth`` is a pure linear operator (signed
        fields are preserved) that reproduces the dense kernel to within a
        near-lossless truncation tolerance: the deviation is ≤ ``tol`` (default
        ``1e-6``) in the **M-weighted norm** relative to the dense output (the raw
        per-bin error carries a volume-conditioning factor ``κ(M)``, worst on
        polar ``r→0`` / skewed mesh). On a **non-negative** input this can leave
        tolerance-level negatives instead of the dense kernel's exact 0-floor —
        clip the result yourself if you need a strict floor. Mass conservation
        (``mode='transition'``) and constant-field preservation
        (``mode='average'``) hold **exactly** under truncation (the per-component
        null mode is always retained).

        Edge preservation: Smoothing respects graph connectivity. Mass does
        not leak between disconnected components (even under truncation, because
        the eigenbasis modes are component-local).

        Examples
        --------
        >>> # Smooth spike counts (mass-conserving)
        >>> smoothed_counts = env.smooth(
        ...     spike_counts, bandwidth=5.0, mode="transition"
        ... )  # doctest: +SKIP
        >>> # Total spikes preserved
        >>> assert np.isclose(
        ...     smoothed_counts.sum(), spike_counts.sum()
        ... )  # doctest: +SKIP

        >>> # Turn spike counts into a smoothed density (KDE)
        >>> spike_density = env.smooth(
        ...     spike_counts, bandwidth=3.0, mode="density"
        ... )  # doctest: +SKIP

        >>> # Smooth a discrete probability distribution (mass, sums to 1)
        >>> smoothed_prob = env.smooth(
        ...     posterior, bandwidth=2.0, mode="transition"
        ... )  # doctest: +SKIP

        """
        # Input validation
        field = np.asarray(field, dtype=np.float64)

        # Check field dimensionality
        if field.ndim != 1:
            raise ValueError(
                f"Field must be 1-D array (got {field.ndim}-D array). "
                f"Expected shape (n_bins,) = ({self.n_bins},), got shape {field.shape}."
            )

        # Check field shape matches n_bins
        if field.shape[0] != self.n_bins:
            raise ValueError(
                f"Field shape {field.shape} must match n_bins={self.n_bins}. "
                f"Expected shape (n_bins,) = ({self.n_bins},), got ({field.shape[0]},)."
            )

        # Check for NaN/Inf values
        if np.any(np.isnan(field)):
            raise ValueError(
                "Field contains NaN values. "
                f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
                "NaN values are not supported in smoothing operations."
            )

        if np.any(np.isinf(field)):
            raise ValueError(
                "Field contains infinite values. "
                f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
                "Infinite values are not supported in smoothing operations."
            )

        # Validate bandwidth
        if bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be positive (got {bandwidth}). "
                "Bandwidth controls the spatial scale of smoothing."
            )

        # Validate mode
        valid_modes = {"transition", "density", "average"}
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes} (got '{mode}'). "
                "Use 'transition' for mass-conserving smoothing of extensive "
                "data, 'density' for count-to-density, or 'average' for "
                "volume-unbiased averaging of an intensive field."
            )

        # Apply the diffusion smoothing via the matrix-free apply-path
        # (env.diffuse). This is a PURE LINEAR operator -- no clip / renorm --
        # so smoothing a SIGNED field is preserved. It reproduces the dense
        # kernel to within the truncation tolerance (see the "Approximation"
        # note below); positivity, where needed, is the caller's job.
        return self.diffuse(field, bandwidth, mode=mode)

    @check_fitted
    def diffuse(
        self: SelfEnv,
        fields: Any,
        bandwidth: float,
        *,
        mode: Literal["transition", "density", "average"] = "density",
        backend: Literal["numpy", "jax"] = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free finite-volume diffusion smoothing (no ``(n, n)`` kernel).

        Applies the finite-volume heat operator ``H = exp(-t L)``, ``t = σ²/2``,
        ``L = M⁻¹(D − W)``, to one or more fields **without ever materializing
        the dense ``(n_bins, n_bins)`` kernel**. It uses a cached, per-component,
        bandwidth-aware **truncated symmetric eigenbasis** of
        ``S = M^{-1/2}(D − W) M^{-1/2}``, so both time and memory scale with
        ``n_bins × rank`` (``rank ~ measure(domain)/σ^d``), not ``n_bins²``. The
        eigenbasis depends only on geometry, so it is built once and reused across
        every bandwidth, mode, and neuron (auto-invalidated when the environment
        is mutated).

        ``diffuse`` is a **pure linear operator**: it applies **no output clip
        and no renormalization**. The always-retained per-component null mode
        makes mass conservation exact under truncation (``mode="transition"``
        preserves ``sum``; a constant field is preserved by ``mode="average"``),
        so smoothing a **signed** field is preserved. Where the result must be
        non-negative (a density, a rate), the **caller** clips it (as the KDE
        encoders do); ``diffuse`` itself never does.

        Parameters
        ----------
        fields : NDArray[np.float64], shape (n_bins,) or (n_bins, n_fields)
            Field(s) to smooth, indexed by bin. A 1-D field is smoothed and
            returned 1-D; a 2-D batch smooths every column in one pass. NaN/Inf
            are not validated (``diffuse`` is a raw linear operator: NaN in →
            NaN out); callers needing validation use :meth:`smooth`.
        bandwidth : float
            Physical smoothing standard deviation σ (> 0), independent of bin
            size on any supported layout.
        mode : {'transition', 'density', 'average'}, default='density'
            Kernel orientation, matching :meth:`compute_kernel` / :meth:`smooth`
            (``'transition'`` = ``Hᵀ`` mass-conserving extensive smoothing;
            ``'density'`` = ``H·M⁻¹`` count→density; ``'average'`` = ``H``
            row-stochastic intensive averaging).
        backend : {'numpy', 'jax'}, default='numpy'
            Where the apply runs. ``'jax'`` casts the cached NumPy eigenbasis to
            ``jax.numpy`` and runs the apply on device, so ``jit`` / ``grad`` /
            GPU work through the smoothing; the returned array is a ``jax.Array``.
            The eigenbasis **build** always uses NumPy/SciPy.

        Returns
        -------
        smoothed : NDArray[np.float64], shape matching ``fields``
            The diffused field(s). ``numpy.ndarray`` for ``backend='numpy'``,
            ``jax.Array`` for ``backend='jax'``.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If ``bandwidth`` is not positive, ``mode`` is invalid, ``backend`` is
            invalid, or ``fields`` is not 1-D/2-D with leading dim ``n_bins``.

        See Also
        --------
        smooth : 1-D smoothing with input validation (delegates here).
        compute_kernel : Build the DENSE kernel matrix (kept for callers that
            genuinely need the ``(n, n)`` matrix; uses the dense ``expm`` path).

        Notes
        -----
        **Approximation contract.** ``diffuse`` reproduces the dense operator to
        within a near-lossless truncation tolerance: dropped modes contribute at
        most ``tol`` (default ``1e-6``) in the **M-weighted norm** (per-bin error
        carries a volume-conditioning factor ``κ(M) = sqrt(max vol / min vol)``,
        worst on polar ``r→0`` / skewed mesh). On a **non-negative** input the
        linear apply can therefore leave tolerance-level negatives (bounded
        relative to the dense output in the M-norm) instead of the dense kernel's
        exact 0-floor — clip the result yourself if you need a strict floor.

        A light bandwidth on a large grid needs most modes; there the eigenbasis
        cannot beat the dense kernel, so ``diffuse`` builds a **transient** dense
        basis (applied then dropped, never cached) with the same large-matrix
        ``UserWarning`` as :meth:`compute_kernel`.
        """
        from neurospatial.ops.diffusion import diffusion_apply

        if mode not in ("transition", "density", "average"):
            raise ValueError(
                f"mode must be one of {{'transition', 'density', 'average'}} "
                f"(got '{mode}')."
            )
        if backend not in ("numpy", "jax"):
            raise ValueError(f"backend must be 'numpy' or 'jax' (got '{backend}').")
        if not bandwidth > 0:
            raise ValueError(
                f"bandwidth must be positive (got {bandwidth}). "
                "Bandwidth controls the spatial scale of smoothing."
            )

        # Coerce to 2-D (n_bins, n_fields); a 1-D field is squeezed back on
        # return. np.ndim / np.shape avoid forcing a JAX array to NumPy.
        if backend == "numpy":
            fields = np.asarray(fields, dtype=np.float64)
        ndim = int(np.ndim(fields))
        if ndim not in (1, 2):
            raise ValueError(
                f"fields must be 1-D (n_bins,) or 2-D (n_bins, n_fields), got "
                f"{ndim}-D array."
            )
        shape = np.shape(fields)
        if shape[0] != self.n_bins:
            raise ValueError(
                f"fields leading dimension {shape[0]} must match n_bins={self.n_bins}."
            )
        was_1d = ndim == 1
        fields_2d = fields.reshape(-1, 1) if was_1d else fields

        W, volumes, n_components, labels = self._diffusion_geometry
        holder = self._diffusion_eigenbasis
        result = diffusion_apply(
            holder,
            W,
            volumes,
            n_components,
            labels,
            fields_2d,
            float(bandwidth),
            mode,
            backend=backend,
        )
        # numpy backend -> ndarray; jax backend -> jax.Array (documented above).
        # The NDArray annotation is the numpy-path type; cast covers both.
        return cast("NDArray[np.float64]", result[:, 0] if was_1d else result)

    @versioned_cached_property
    def _diffusion_geometry(
        self,
    ) -> tuple[Any, NDArray[np.float64], int, NDArray[np.int_]]:
        """Cached finite-volume geometry ``(W, volumes, n_components, labels)``.

        Geometry-only, so it is built once and dropped wholesale on any
        ``_state_version`` bump (like every ``versioned_cached_property``). Shared
        by :meth:`diffuse`'s eigenbasis build and the W-component support gates
        the smoothing consumers use.
        """
        from neurospatial.ops.diffusion import (
            _assemble_W,
            _components_from_W,
            _finite_volume_geometry,
        )

        graph, volumes = _finite_volume_geometry(cast("EnvironmentProtocol", self))
        volumes = np.asarray(volumes, dtype=np.float64)
        W = _assemble_W(graph, int(volumes.shape[0]))
        n_components, labels = _components_from_W(W)
        return W, volumes, n_components, labels

    @versioned_cached_property
    def _diffusion_eigenbasis(self) -> dict[str, Any]:
        """Growable single-entry truncated-eigenbasis cache for :meth:`diffuse`.

        Returns a small mutable holder (the max-rank truncated basis, grown by
        replace, plus a per-``(sigma, tol)`` resolved-rank memo) that
        :func:`neurospatial.ops.diffusion.diffusion_apply` mutates in place. The
        ``versioned_cached_property`` owns it, so it is dropped wholesale on any
        ``_state_version`` bump / ``clear_cache``, and it holds only a single
        basis strictly below ``dense_fraction·n`` — never a persistent ``(n, n)``.
        """
        return {}

    def interpolate(
        self: SelfEnv,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
        *,
        mode: Literal["nearest", "linear"] = "nearest",
    ) -> NDArray[np.float64]:
        """Interpolate field values at arbitrary points.

        Evaluates bin-valued fields at continuous query points using either
        nearest-neighbor or linear interpolation. Nearest mode works on all
        layout types; linear mode requires regular grid layouts.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values per bin. Must be a 1-D array with length equal to n_bins.
            Must not contain NaN or Inf values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points in environment coordinates. Must be a 2-D array where
            each row is a point with dimensionality matching the environment.
        mode : {'nearest', 'linear'}, default='nearest'
            Interpolation mode:

            - 'nearest': Use value of nearest bin center (all layouts).
              Points outside environment bounds return NaN.
            - 'linear': Bilinear (2D) or trilinear (3D) interpolation for
              regular grids. Only supported for RegularGridLayout.
              Points outside grid bounds return NaN.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated field values. Points outside environment → NaN.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If field has wrong shape, wrong dimensionality, contains NaN/Inf,
            points have wrong dimensionality, mode is invalid, or dimensions
            don't match.
        NotImplementedError
            If mode='linear' is requested for non-grid layout.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        occupancy : Compute occupancy with optional smoothing.

        Notes
        -----
        **Nearest-neighbor mode**: Uses KDTree to find closest bin center.
        Deterministic and works on all layout types. Points farther than a
        reasonable threshold from any bin center are marked as outside (NaN).

        **Linear mode**: Uses scipy.interpolate.RegularGridInterpolator for
        smooth interpolation on rectangular grids. For linear functions
        f(x,y) = ax + by + c, interpolation is exact up to numerical precision.

        **Outside handling**: Points outside the environment bounds return NaN
        in both modes. This prevents extrapolation errors.

        Examples
        --------
        >>> # Nearest-neighbor interpolation (all layouts)
        >>> field = np.random.rand(env.n_bins)  # doctest: +SKIP
        >>> query_points = np.array([[5.0, 5.0], [7.5, 3.2]])  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     field, query_points, mode="nearest"
        ... )  # doctest: +SKIP

        >>> # Linear interpolation (grids only)
        >>> # For plane f(x,y) = 2x + 3y, interpolation is exact
        >>> plane_field = (
        ...     2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
        ... )  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     plane_field, query_points, mode="linear"
        ... )  # doctest: +SKIP

        >>> # Evaluate rate map at trajectory positions
        >>> rates_at_trajectory = env.interpolate(
        ...     rate_map, positions, mode="linear"
        ... )  # doctest: +SKIP

        """
        # Input validation - field
        field = np.asarray(field, dtype=np.float64)

        # Check field dimensionality
        if field.ndim != 1:
            raise ValueError(
                f"Field must be 1-D array (got {field.ndim}-D array). "
                f"Expected shape (n_bins,) = ({self.n_bins},), got shape {field.shape}."
            )

        # Check field shape matches n_bins
        if field.shape[0] != self.n_bins:
            raise ValueError(
                f"Field shape {field.shape} must match n_bins={self.n_bins}. "
                f"Expected shape (n_bins,) = ({self.n_bins},), got ({field.shape[0]},)."
            )

        # Check for NaN/Inf values in field
        if np.any(np.isnan(field)):
            raise ValueError(
                "Field contains NaN values. "
                f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
                "NaN values are not supported in interpolation operations."
            )

        if np.any(np.isinf(field)):
            raise ValueError(
                "Field contains infinite values. "
                f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
                "Infinite values are not supported in interpolation operations."
            )

        # Input validation - points
        points = np.asarray(points, dtype=np.float64)

        # Check points dimensionality
        if points.ndim != 2:
            raise ValueError(
                f"Points must be 2-D array (got {points.ndim}-D array). "
                f"Expected shape (n_points, n_dims), got shape {points.shape}."
            )

        # Check points dimension matches environment
        n_dims = self.bin_centers.shape[1]
        if points.shape[1] != n_dims:
            raise ValueError(
                f"Points dimension {points.shape[1]} must match environment "
                f"dimension {n_dims}. Expected shape (n_points, {n_dims}), "
                f"got shape {points.shape}."
            )

        # Check for NaN/Inf values in points
        if np.any(~np.isfinite(points)):
            n_invalid = np.sum(~np.isfinite(points))
            raise ValueError(
                f"Points array contains {n_invalid} non-finite value(s) (NaN or Inf). "
                f"All point coordinates must be finite. Check your input data for "
                f"missing values or infinities."
            )

        # Validate mode
        valid_modes = {"nearest", "linear"}
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes} (got '{mode}'). "
                "Use 'nearest' for nearest-neighbor interpolation or 'linear' "
                "for bilinear/trilinear interpolation (grids only)."
            )

        # Handle empty points array
        if points.shape[0] == 0:
            return np.array([], dtype=np.float64)

        # Dispatch based on mode
        if mode == "nearest":
            return self._interpolate_nearest(field, points)
        else:  # mode == "linear"
            return self._interpolate_linear(field, points)

    def _interpolate_nearest(
        self: SelfEnv,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Nearest-neighbor interpolation using KDTree.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated values (NaN for points outside).

        """
        from typing import cast

        from neurospatial.ops.binning import map_points_to_bins

        # Map points to bins (-1 for outside points)
        # With return_dist=False, we get just the indices (not a tuple)
        bin_indices = cast(
            "NDArray[np.int64]",
            map_points_to_bins(
                points,
                cast("Environment", self),
                tie_break="lowest_index",
                return_dist=False,
            ),
        )

        # Initialize result with NaN
        result = np.full(points.shape[0], np.nan, dtype=np.float64)

        # Fill in values for points inside environment
        inside_mask = bin_indices >= 0
        result[inside_mask] = field[bin_indices[inside_mask]]

        return result

    def _interpolate_linear(
        self: SelfEnv,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Linear interpolation using scipy RegularGridInterpolator.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated values (NaN for points outside).

        Raises
        ------
        NotImplementedError
            If layout is not RegularGridLayout.

        Notes
        -----
        On a holed/masked grid (where ``n_bins < prod(grid_shape)``), inactive
        cells are filled with NaN before interpolation. Query points over an
        inactive cell therefore return NaN, consistent with the out-of-bounds
        ``fill_value=np.nan`` behavior for points outside the grid entirely.

        """
        # Check layout type - must be RegularGridLayout, not masked/polygon layouts
        # Use _layout_type_tag to avoid mypy Protocol isinstance issues
        if self.layout._layout_type_tag != "RegularGrid":
            raise NotImplementedError(
                f"Linear interpolation (mode='linear') is only supported for "
                f"RegularGridLayout. Current layout type: {self.layout._layout_type_tag}. "
                f"Use mode='nearest' for non-grid layouts, or create a regular grid "
                f"environment with Environment.from_samples()."
            )

        # Import scipy
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError as e:
            raise ImportError(
                "Linear interpolation requires scipy. Install with: pip install scipy"
            ) from e

        # Get grid properties (we know layout has these from the check above)
        # Cast to Any to work around mypy Protocol limitation
        from typing import cast

        layout_any = cast("Any", self.layout)
        grid_shape: tuple[int, ...] = layout_any.grid_shape
        grid_edges: tuple[NDArray[np.float64], ...] = layout_any.grid_edges
        n_dims = len(grid_shape)

        # Scatter active-bin values into a full-grid array before reshaping.
        # `field` is indexed by active-bin id (length n_bins); `grid_shape`
        # is the FULL grid. On a holed/masked grid n_bins < prod(grid_shape),
        # so a direct reshape would raise. Inactive cells are filled with NaN,
        # which the RegularGridInterpolator (fill_value=np.nan) already treats
        # as "no data" — query points over a hole interpolate to NaN.
        n_full = int(np.prod(grid_shape))
        active_mask = getattr(self.layout, "active_mask", None)
        if active_mask is None or field.shape[0] == n_full:
            # Fully-active grid: field already covers every cell.
            field_grid = field.reshape(grid_shape)
        else:
            full_field = np.full(n_full, np.nan, dtype=np.float64)
            full_field[np.flatnonzero(active_mask.ravel())] = field
            field_grid = full_field.reshape(grid_shape)

        # Create grid points for each dimension (bin centers)
        grid_points: list[NDArray[np.float64]] = []
        for dim in range(n_dims):
            edges = grid_edges[dim]
            # Bin centers are midpoints between edges
            centers = (edges[:-1] + edges[1:]) / 2
            grid_points.append(centers)

        # Create interpolator
        # bounds_error=False, fill_value=np.nan → outside points return NaN
        interpolator = RegularGridInterpolator(
            grid_points,
            field_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Evaluate at query points
        result: NDArray[np.float64] = interpolator(points)

        return result
