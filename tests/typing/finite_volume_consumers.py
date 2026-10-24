"""Static contracts for Cartesian and polar finite-volume consumers and caches.

Run with ``uv run mypy tests/typing/finite_volume_consumers.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.binning import (
        Binning,
        DiffusionGeometry,
        FaceGeometry,
        KnownFaces,
        MissingFaces,
    )
    from neurospatial.environment import (
        CartesianSystem,
        EgocentricPolarSystem,
    )
    from neurospatial.geometry import MissingCellGeometryError, NativeFaces
    from neurospatial.ops import MRFBasis
    from neurospatial.ops._protocols import (
        DiffusionGeometryConsumer,
        DivergenceConsumer,
        GradientConsumer,
        WaveletBasisConsumer,
    )
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
        wavelets: WaveletBasisConsumer = env
        diffusion_kernel(geometry, 1.0)
        divergence(flux, gradient(scalar, np.ones(env.n_bins)))
        _diffusion_generator(
            wavelets.bin_centers,
            wavelets.bin_center_tree(),
            wavelets.get_diffusion_geometry(),
        )
        assert_type(env.get_diffusion_geometry(), DiffusionGeometry)
        assert_type(env.get_mrf_basis(np.ones(env.n_bins)), MRFBasis)
        assert_type(env.compute_gaussian_kernel(bandwidth=1.0), NDArray[np.float64])
        assert_type(env.diffusion_accuracy(bandwidth=1.0), float)

    def check_faces(binning: Binning) -> None:
        """A binning's faces are one of the three built-in face records."""
        faces = binning.faces
        assert_type(binning.build_faces(), FaceGeometry)
        if isinstance(faces, MissingFaces):
            assert_type(faces.missing_error("smoothing"), MissingCellGeometryError)
        elif isinstance(faces, KnownFaces):
            assert_type(faces.faces, FaceGeometry)
        else:
            assert_type(faces, NativeFaces)
