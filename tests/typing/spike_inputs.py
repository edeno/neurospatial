"""Static contract for the spike inputs every population function accepts.

Run with ``uv run mypy tests/typing/spike_inputs.py``. At runtime each
population encoder, significance test, peri-event histogram and decoder accepts
a list of per-unit arrays, a 1-D or NaN-padded 2-D array, a ``SpikeTrains``
container and a pynapple ``TsGroup``-like group. These calls fail type-checking
when a signature rejects one of those documented inputs.
"""

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from neurospatial import Environment, compute_spatial_rate, compute_spatial_rates
    from neurospatial._typing import SpikeGroupLike
    from neurospatial.decoding import (
        BayesianDecoder,
        decode_session,
        decode_session_summary,
    )
    from neurospatial.encoding import (
        SpikeTrains,
        compute_directional_rates,
        compute_egocentric_rates,
        compute_object_vector_rates,
        compute_view_rates,
        egocentric_object_vector_cell_significance,
        head_direction_cell_significance,
        object_vector_cell_significance,
        place_cell_significance,
        spatial_view_cell_significance,
    )
    from neurospatial.events import population_peri_event_histogram

    Spikes: TypeAlias = (
        list[NDArray[np.float64]]
        | tuple[NDArray[np.float64], ...]
        | NDArray[np.float64]
        | SpikeTrains
        | SpikeGroupLike
    )

    def population_calls(
        env: Environment,
        spikes: Spikes,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        headings: NDArray[np.float64],
        objects: NDArray[np.float64],
    ) -> None:
        compute_spatial_rates(env, spikes, times, positions)
        compute_directional_rates(spikes, times, headings)
        compute_view_rates(env, spikes, times, positions, headings)
        compute_egocentric_rates(env, spikes, times, positions, headings, objects)
        compute_object_vector_rates(env, spikes, times, positions, objects)
        place_cell_significance(env, spikes, times, positions)
        head_direction_cell_significance(spikes, times, headings)
        spatial_view_cell_significance(env, spikes, times, positions, headings)
        object_vector_cell_significance(env, spikes, times, positions, objects)
        egocentric_object_vector_cell_significance(
            env, spikes, times, positions, headings, objects
        )
        population_peri_event_histogram(spikes, times[:2], (-1.0, 1.0))
        decode_session(env, spikes, times, positions)
        decode_session_summary(env, spikes, times, positions)
        decoder = BayesianDecoder(env).fit(spikes, times, positions)
        decoder.predict(spikes, times)
        decoder.predict_summary(spikes, times)
        decoder.score(spikes, times, positions)

    def single_unit_from_container(
        env: Environment,
        container: SpikeTrains,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
    ) -> None:
        compute_spatial_rate(env, container.sel(unit_id=7), times, positions)
        compute_spatial_rate(env, container[0], times, positions)
