"""The JAX backend computes in float64 whichever call runs first in a process.

JAX's float64 switch is process-global, so each check runs in a fresh
interpreter where nothing has turned it on yet.
"""

import subprocess
import sys
import textwrap

import pytest

from neurospatial.encoding._backend import is_jax_available

pytestmark = pytest.mark.skipif(
    not is_jax_available(), reason="the JAX backend is unavailable here"
)

_SETUP = """
import numpy as np
from shapely.geometry import box
from neurospatial import Environment
env = Environment.from_polygon(box(0, 0, 50, 50), bin_size=5.0, units="cm")
rng = np.random.default_rng(0)
times = np.arange(3000) / 30.0
positions = rng.uniform(0, 50, (3000, 2))
spikes = [np.sort(rng.uniform(0, 100, 300))]
"""


def _run(body: str) -> str:
    script = textwrap.dedent(_SETUP) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


_FAMILY_CALLS = {
    "spatial": "compute_spatial_rates(env, spikes, times, positions, {kw})",
    "directional": "compute_directional_rates(spikes, times, headings, {kw})",
    "view": "compute_view_rates(env, spikes, times, positions, headings, {kw})",
    "object_vector": (
        "compute_object_vector_rates(env, spikes, times, positions, objects, {kw})"
    ),
}


@pytest.mark.parametrize("family", list(_FAMILY_CALLS))
@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
def test_first_jax_call_returns_float64(family, method):
    keywords = 'backend="jax"' + (
        "" if family == "directional" else f', method="{method}"'
    )
    if family == "directional" and method != "diffusion_kde":
        pytest.skip("directional rates have one smoother")
    call = _FAMILY_CALLS[family].format(kw=keywords)
    out = _run(
        f"""
        import warnings
        from neurospatial import encoding
        headings = rng.uniform(-np.pi, np.pi, len(times))
        objects = np.array([[25.0, 25.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rates = encoding.{call}
        truncated = [w for w in caught if "truncated to dtype float32" in str(w.message)]
        print(rates.firing_rates.dtype, rates.occupancy.dtype, len(truncated))
        """
    )
    assert out == "float64 float64 0"


def test_tracing_with_float64_off_says_how_to_fix():
    out = _run(
        """
        import jax
        from neurospatial.encoding._smoothing import smooth_rate_maps_batch
        counts = np.ones((1, env.n_bins))
        occupancy = np.ones(env.n_bins)
        compiled = jax.jit(
            lambda c: smooth_rate_maps_batch(
                env, c, occupancy, method="binned", bandwidth=5.0, backend="jax"
            )
        )
        try:
            compiled(counts)
        except RuntimeError as error:
            print("jax_enable_x64" in str(error) and "Fix:" in str(error))
        """
    )
    assert out == "True"
