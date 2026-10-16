"""Static contracts: public ops functions accept an Environment at their boundary.

Each public ops function that takes an environment reads a few members through
the protocols in ``neurospatial.ops._protocols`` and computes from those
components. Both coordinate kinds satisfy the protocols that read graph, bin
and lookup members. Protocols that read Cartesian positions declare
``coordinate_system: CartesianSystem``, so a polar environment is a type error
there. Internal helpers take the components themselves (``Binning``,
``Arena``, arrays). Run with ``uv run mypy tests/typing/ops_consumers.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.binning import Binning
    from neurospatial.environment import CartesianSystem, EgocentricPolarSystem
    from neurospatial.geometry._arena import Arena
    from neurospatial.ops import (
        compute_egocentric_distance,
        geodesic_rbf_basis,
        graph_convolve,
        heat_kernel_wavelet_basis,
        map_probabilities,
        plot_basis_functions,
        regions_to_mask,
        resample_field,
        select_basis_centers,
        spatial_basis,
    )
    from neurospatial.ops._boundaries import (
        analysis_interior,
        boundary_geometry,
        valid_observers,
    )
    from neurospatial.ops._protocols import (
        BasisCentersConsumer,
        BinCentersConsumer,
        BinningConsumer,
        BinsConsumer,
        BinSpacingConsumer,
        CartesianBinsConsumer,
        FieldPlotConsumer,
        GraphConsumer,
        GraphLookupConsumer,
        RegionMaskConsumer,
        ResampleTargetConsumer,
        SpatialBasisConsumer,
        WaveletBasisConsumer,
    )
    from neurospatial.ops.basis import _diffusion_generator
    from neurospatial.ops.binning import _pull_back_nearest
    from neurospatial.ops.transforms import AffineND
    from neurospatial.ops.visibility import (
        ViewshedResult,
        _line_of_sight_clear,
        compute_viewshed,
    )

    def _either_coordinate_system(
        env: Environment[CartesianSystem] | Environment[EgocentricPolarSystem],
        values: NDArray[np.float64],
    ) -> None:
        _centers: BinCentersConsumer = env
        _bins: BinsConsumer = env
        _graph: GraphConsumer = env
        _binning: BinningConsumer = env
        _lookup: GraphLookupConsumer = env
        _basis: BasisCentersConsumer = env
        _wavelets: WaveletBasisConsumer = env
        _plot: FieldPlotConsumer = env
        _spacing: BinSpacingConsumer = env
        _regions: RegionMaskConsumer = env
        graph_convolve(env, values, lambda distance: distance)
        regions_to_mask(env, "goal")
        select_basis_centers(env, 3, method="farthest_point")
        geodesic_rbf_basis(env, n_centers=3, center_method="farthest_point")
        heat_kernel_wavelet_basis(env, center_method="farthest_point")
        plot_basis_functions(env, np.ones((env.n_bins, 2)))
        compute_egocentric_distance(values, values, values, metric="geodesic", env=env)
        assert_type(compute_viewshed(env, values, 0.0), ViewshedResult)

    def _cartesian_only(
        cartesian: Environment[CartesianSystem],
        polar: Environment[EgocentricPolarSystem],
        values: NDArray[np.float64],
        transform: AffineND,
    ) -> None:
        _bins: CartesianBinsConsumer = cartesian
        _target: ResampleTargetConsumer = cartesian
        _spatial: SpatialBasisConsumer = cartesian
        resample_field(values, cartesian, cartesian)
        map_probabilities(cartesian, cartesian, values)
        spatial_basis(cartesian)
        assert_type(cartesian.apply_transform(transform), Environment[CartesianSystem])

        _polar_bins: CartesianBinsConsumer = polar  # type: ignore[assignment]
        _polar_target: ResampleTargetConsumer = polar  # type: ignore[assignment]
        _polar_spatial: SpatialBasisConsumer = polar  # type: ignore[assignment]
        resample_field(values, polar, cartesian)  # type: ignore[arg-type]
        resample_field(values, cartesian, polar)  # type: ignore[arg-type]
        spatial_basis(polar)  # type: ignore[arg-type]

    def _internals_take_components(
        binning: Binning,
        arena: Arena,
        centers: NDArray[np.float64],
        values: NDArray[np.float64],
        env: Environment[CartesianSystem],
    ) -> None:
        boundary_geometry(binning, arena, "analysis")
        analysis_interior(binning, arena)(centers)
        valid_observers(binning, arena, centers, "physical")
        _line_of_sight_clear(binning, arena, centers[0], centers[1])
        _pull_back_nearest(values, centers, lambda points: points[:, 0].astype(int))
        _diffusion_generator(centers, env.get_diffusion_geometry())
        boundary_geometry(env, "analysis")  # type: ignore[call-arg, arg-type]
        _line_of_sight_clear(env, centers[0], centers[1])  # type: ignore[call-arg, arg-type]
