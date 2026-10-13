"""Static contracts for Cartesian and polar finite-volume consumers and caches.

Run with ``uv run mypy tests/typing/finite_volume_consumers.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.environment.polar import EgocentricPolarEnvironment
    from neurospatial.ops._protocols import (
        DiffusionAccuracyConsumer,
        DiffusionGeometryConsumer,
        DivergenceConsumer,
        FaceGeometryConsumer,
        GradientConsumer,
        WaveletGeometryConsumer,
    )
    from neurospatial.ops._types import DiffusionEigenbasisCache, MRFEigenbasisCache
    from neurospatial.ops.basis import _diffusion_generator
    from neurospatial.ops.calculus import (
        compute_differential_operator,
        divergence,
        gradient,
    )
    from neurospatial.ops.diffusion import diffusion_apply_rtol, diffusion_kernel

    def check_consumers(env: Environment | EgocentricPolarEnvironment) -> None:
        """Both concrete environments satisfy the narrow contracts without casts."""
        geometry: DiffusionGeometryConsumer = env
        accuracy: DiffusionAccuracyConsumer = env
        faces: FaceGeometryConsumer = env
        scalar: GradientConsumer = env
        flux: DivergenceConsumer = env
        wavelets: WaveletGeometryConsumer = env
        diffusion_kernel(geometry, 1.0)
        diffusion_apply_rtol(accuracy, 1.0)
        divergence(flux, gradient(scalar, np.ones(env.n_bins)))
        compute_differential_operator(faces)
        _diffusion_generator(wavelets)
        assert_type(env._diffusion_eigenbasis, DiffusionEigenbasisCache)
        assert_type(env._mrf_eigenbasis, MRFEigenbasisCache)
        assert_type(env._diffusion_eigenbasis["basis"][0], NDArray[np.float64])
        assert_type(env._mrf_eigenbasis["G"], int)
