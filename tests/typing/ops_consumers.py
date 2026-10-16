"""Static contracts: Environment satisfies the narrow protocols ops functions read.

Polar environments satisfy the graph and bin protocols and fail the ones that
read Cartesian positions (``coordinate_system: CartesianSystem``), so ops
functions that read Cartesian centers keep refusing them statically. Protocols
naming only a Cartesian-restricted method, such as ``bin_at``, still accept a
polar environment statically; those calls refuse at runtime. Run with
``uv run mypy tests/typing/ops_consumers.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.environment import CartesianSystem, EgocentricPolarSystem
    from neurospatial.ops._protocols import (
        AffineTransformConsumer,
        BasisCentersConsumer,
        BinCentersConsumer,
        BinLookupConsumer,
        BinsConsumer,
        BinSpacingConsumer,
        CartesianBinsConsumer,
        CartesianNearestBinConsumer,
        FieldPlotConsumer,
        GraphConsumer,
        RegionMaskConsumer,
        ResampleSourceConsumer,
        ResampleTargetConsumer,
        SpatialBasisConsumer,
        ViewshedConsumer,
        WaveletBasisConsumer,
    )
    from neurospatial.ops.transforms import AffineND, apply_transform_to_environment

    def _either_coordinate_system(
        env: Environment[CartesianSystem] | Environment[EgocentricPolarSystem],
    ) -> None:
        _centers: BinCentersConsumer = env
        _bins: BinsConsumer = env
        _graph: GraphConsumer = env
        _basis: BasisCentersConsumer = env
        _wavelets: WaveletBasisConsumer = env
        _plot: FieldPlotConsumer = env
        _spacing: BinSpacingConsumer = env
        _regions: RegionMaskConsumer = env

    def _cartesian_only(
        cartesian: Environment[CartesianSystem],
        polar: Environment[EgocentricPolarSystem],
        transform: AffineND,
    ) -> None:
        _bins: CartesianBinsConsumer = cartesian
        _nearest: CartesianNearestBinConsumer = cartesian
        _source: ResampleSourceConsumer = cartesian
        _target: ResampleTargetConsumer = cartesian
        _spatial: SpatialBasisConsumer = cartesian
        _viewshed: ViewshedConsumer = cartesian
        _lookup: BinLookupConsumer = cartesian
        _affine: AffineTransformConsumer[Environment[CartesianSystem]] = cartesian
        assert_type(
            apply_transform_to_environment(cartesian, transform),
            Environment[CartesianSystem],
        )

        _polar_bins: CartesianBinsConsumer = polar  # type: ignore[assignment]
        _polar_nearest: CartesianNearestBinConsumer = polar  # type: ignore[assignment]
        _polar_source: ResampleSourceConsumer = polar  # type: ignore[assignment]
        _polar_target: ResampleTargetConsumer = polar  # type: ignore[assignment]
        _polar_spatial: SpatialBasisConsumer = polar  # type: ignore[assignment]
