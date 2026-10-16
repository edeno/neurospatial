"""Static contracts for Cartesian and polar finite-volume consumers and caches.

Run with ``uv run mypy tests/typing/finite_volume_consumers.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.binning import DiffusionGeometry
    from neurospatial.environment import (
        CartesianSystem,
        EgocentricPolarSystem,
    )
    from neurospatial.ops import MRFBasis
    from neurospatial.ops._protocols import (
        DiffusionGeometryConsumer,
        DivergenceConsumer,
        GradientConsumer,
        WaveletGeometryConsumer,
    )
    from neurospatial.ops._types import DiffusionEigenbasisCache, MRFEigenbasisCache
    from neurospatial.ops.basis import _diffusion_generator
    from neurospatial.ops.calculus import divergence, gradient
    from neurospatial.ops.diffusion import diffusion_kernel

    def check_consumers(
        env: Environment[CartesianSystem] | Environment[EgocentricPolarSystem],
    ) -> None:
        """Both coordinate meanings satisfy the narrow contracts without casts."""
        geometry: DiffusionGeometryConsumer = env
        scalar: GradientConsumer = env
        flux: DivergenceConsumer = env
        wavelets: WaveletGeometryConsumer = env
        diffusion_kernel(geometry, 1.0)
        divergence(flux, gradient(scalar, np.ones(env.n_bins)))
        _diffusion_generator(wavelets)
        assert_type(env.get_diffusion_geometry(), DiffusionGeometry)
        assert_type(env.get_mrf_basis(np.ones(env.n_bins)), MRFBasis)
        assert_type(env.compute_gaussian_kernel(1.0), NDArray[np.float64])
        assert_type(env.diffusion_accuracy(1.0), float)
        assert_type(env._diffusion_eigenbasis, DiffusionEigenbasisCache)
        assert_type(env._mrf_eigenbasis, MRFEigenbasisCache)
        assert_type(env._diffusion_eigenbasis["basis"][0], NDArray[np.float64])
        assert_type(env._mrf_eigenbasis["G"], int)
