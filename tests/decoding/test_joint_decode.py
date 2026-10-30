"""Tests for decode_jointly: one posterior over several sets of rate maps."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import poisson

from neurospatial import Environment
from neurospatial.decoding import JointDecodingResult, decode_jointly, decode_position

DT = 0.1


def _track(length: float, bin_size: float = 10.0, units: str = "cm") -> Environment:
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [length, 0.0]], bin_size=bin_size, units=units
    )


def _place_maps(env: Environment, centers: np.ndarray) -> np.ndarray:
    x = env.bin_centers[:, 0]
    return 1.0 + 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)


@pytest.fixture
def two_directions():
    """One track with an outbound and an inbound set of maps for 8 cells."""
    env = _track(100.0)
    outbound = _place_maps(env, np.linspace(5.0, 95.0, 8))
    inbound = outbound[[0, 5, 2, 7, 4, 1, 6, 3]]
    return env, {"outbound": outbound, "inbound": inbound}


def _brute_force(envs, models, counts, weights):
    """P(k, x | n) from Poisson pmfs, with no shortcut or log-space trick."""
    joint = {}
    for name, rates in models.items():
        expected = rates * DT  # (n_neurons, n_bins)
        likelihood = np.prod(
            poisson.pmf(counts[:, :, None], expected[None, :, :]), axis=1
        )  # (n_time, n_bins)
        joint[name] = likelihood * weights[name]
    total = sum(value.sum(axis=1) for value in joint.values())
    return {name: value / total[:, None] for name, value in joint.items()}


class TestPosterior:
    def test_matches_brute_force_poisson_posterior(self, two_directions):
        env, models = two_directions
        counts = np.random.default_rng(1).poisson(1.0, size=(6, 8))
        result = decode_jointly(env, counts, models, dt=DT)

        expected = _brute_force(
            dict.fromkeys(models, env),
            models,
            counts,
            dict.fromkeys(models, env.bin_sizes),
        )
        for column, name in enumerate(result.template_names):
            assert_allclose(result.joint(name), expected[name], rtol=1e-10)
            assert_allclose(
                result.template_probability[:, column],
                expected[name].sum(axis=1),
                rtol=1e-10,
            )

    def test_joint_rows_sum_to_one_over_all_templates(self, two_directions):
        env, models = two_directions
        counts = np.random.default_rng(2).poisson(1.0, size=(10, 8))
        result = decode_jointly(env, counts, models, dt=DT)
        total = sum(result.joint(name).sum(axis=1) for name in result.template_names)
        assert_allclose(total, 1.0, atol=1e-12)

    def test_conditional_is_the_single_template_decode(self, two_directions):
        env, models = two_directions
        counts = np.random.default_rng(3).poisson(1.0, size=(10, 8))
        times = np.arange(10) * DT
        result = decode_jointly(env, counts, models, dt=DT, times=times)
        for name, rates in models.items():
            alone = decode_position(env, counts, rates, dt=DT)
            assert_allclose(
                result.conditional[name].posterior, alone.posterior, atol=1e-12
            )
            assert_allclose(result.conditional[name].times, times)

    def test_conditional_stays_accurate_for_an_improbable_template(
        self, two_directions
    ):
        """A template with probability ~1e-100 keeps an exact conditional."""
        env, models = two_directions
        # Cells 2-4 fire together. Their outbound fields are adjacent; the
        # inbound field of cell 3 is at the far end of the track.
        counts = np.zeros((1, 8), dtype=np.int64)
        counts[0, [2, 3, 4]] = [25, 40, 25]
        result = decode_jointly(env, counts, models, dt=DT)

        assert result.template_probability[0, 1] < 1e-30
        alone = decode_position(env, counts, models["inbound"], dt=DT)
        assert_allclose(
            result.conditional["inbound"].posterior, alone.posterior, rtol=1e-9
        )


class TestUnequalBins:
    def test_matches_brute_force_with_unequal_bins(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(30.0, 0.0))
        graph.add_node(2, pos=(30.0, 25.0))
        graph.add_edge(0, 1, distance=30.0)
        graph.add_edge(1, 2, distance=25.0)
        env = Environment.from_graph(
            graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=0.0,
            bin_size=7.0,
            units="cm",
        )
        assert len(np.unique(np.round(env.bin_sizes, 6))) > 1
        rng = np.random.default_rng(3)
        models = {
            "a": rng.uniform(1.0, 20.0, (5, env.n_bins)),
            "b": rng.uniform(1.0, 20.0, (5, env.n_bins)),
        }
        counts = rng.poisson(1.0, size=(4, 5))
        result = decode_jointly(env, counts, models, dt=DT)
        expected = _brute_force(
            dict.fromkeys(models, env),
            models,
            counts,
            dict.fromkeys(models, env.bin_sizes),
        )
        for name in models:
            assert_allclose(result.joint(name), expected[name], rtol=1e-10)


class TestTemplatePrior:
    def test_one_environment_gives_each_template_equal_prior(self):
        env = _track(100.0)
        flat = np.full((4, env.n_bins), 5.0)
        counts = np.random.default_rng(4).poisson(0.5, size=(5, 4))
        result = decode_jointly(env, counts, {"a": flat, "b": flat}, dt=DT)
        assert_allclose(result.template_probability, 0.5, atol=1e-12)

    def test_separate_tracks_get_prior_proportional_to_length(self):
        """Uniform over space across tracks, as in Tirole et al. (2022)."""
        long, short = _track(100.0), _track(50.0)
        counts = np.random.default_rng(5).poisson(0.5, size=(5, 4))
        result = decode_jointly(
            {"long": long, "short": short},
            counts,
            {
                "long": np.full((4, long.n_bins), 5.0),
                "short": np.full((4, short.n_bins), 5.0),
            },
            dt=DT,
        )
        assert_allclose(result.template_probability[:, 0], 2 / 3, atol=1e-12)
        assert_allclose(result.template_probability[:, 1], 1 / 3, atol=1e-12)

    def test_explicit_weights_share_one_scale(self):
        env = _track(100.0)
        flat = np.full((4, env.n_bins), 5.0)
        counts = np.random.default_rng(6).poisson(0.5, size=(5, 4))
        result = decode_jointly(
            env,
            counts,
            {"a": flat, "b": flat},
            dt=DT,
            prior={"a": np.full(env.n_bins, 3.0), "b": np.ones(env.n_bins)},
        )
        assert_allclose(result.template_probability[:, 0], 0.75, atol=1e-12)


class TestCoverage:
    """A template's prior does not shrink with its training coverage."""

    def test_excluded_bins_keep_the_template_prior(self, two_directions):
        env, models = two_directions
        flat = np.full_like(models["outbound"], 5.0)
        partial = flat.copy()
        partial[:, :4] = np.nan  # inbound training never reached these bins
        with pytest.warns(UserWarning, match="zero posterior probability"):
            result = decode_jointly(
                env,
                np.zeros((3, 8), dtype=np.int64),
                {"outbound": flat, "inbound": partial},
                dt=DT,
            )
        assert_allclose(result.template_probability, 0.5, atol=1e-12)
        assert np.all(result.joint("inbound")[:, :4] == 0.0)

    def test_partial_spatial_support_keeps_the_template_prior(self):
        from neurospatial.encoding import compute_spatial_rates

        env = _track(100.0).with_track_width(10.0)
        times = np.arange(0.0, 200.0, 0.02)
        rng = np.random.default_rng(10)
        spike_times = [np.sort(rng.uniform(0, 200.0, 300)) for _ in range(3)]
        full = compute_spatial_rates(
            env, spike_times, times, np.c_[50 + 49 * np.sin(times / 3), 0 * times]
        )
        half = compute_spatial_rates(
            env, spike_times, times, np.c_[25 + 24 * np.sin(times / 3), 0 * times]
        )
        assert half.spatial_support.sum() < full.spatial_support.sum()
        # Almost no time per bin: the likelihood is ~1 and the prior shows.
        result = decode_jointly(
            env, np.zeros((2, 3), dtype=np.int64), {"full": full, "half": half}, dt=1e-9
        )
        assert_allclose(result.template_probability, 0.5, atol=1e-6)


class TestTimeVaryingPrior:
    def test_weights_per_time_bin_share_one_scale(self):
        env = _track(100.0)
        flat = np.full((4, env.n_bins), 5.0)
        weights_b = np.ones((3, env.n_bins))
        weights_b[1] *= 3.0
        result = decode_jointly(
            env,
            np.zeros((3, 4), dtype=np.int64),
            {"a": flat, "b": flat},
            dt=DT,
            prior={"a": np.ones((3, env.n_bins)), "b": weights_b},
        )
        assert_allclose(result.template_probability[:, 1], [0.5, 0.75, 0.5], atol=1e-12)


class TestRecovery:
    def test_spikes_from_one_template_are_assigned_to_it(self, two_directions):
        env, models = two_directions
        # Half the cells have the same field in both templates, so each time
        # bin needs enough spikes (~19 here) to tell the templates apart.
        positions = np.random.default_rng(7).integers(0, env.n_bins, size=200)
        spike_rng = np.random.default_rng(17)
        for source in ("outbound", "inbound"):
            counts = spike_rng.poisson(models[source][:, positions].T * 0.5)
            result = decode_jointly(env, counts, models, dt=0.5)
            column = result.template_names.index(source)
            assert result.template_probability[:, column].mean() > 0.95, source

    def test_bins_excluded_from_one_template_only(self, two_directions):
        env, models = two_directions
        inbound = models["inbound"].copy()
        inbound[:, 2] = np.nan
        counts = np.random.default_rng(8).poisson(1.0, size=(5, 8))
        with pytest.warns(UserWarning, match="zero posterior probability"):
            result = decode_jointly(
                env, counts, {"outbound": models["outbound"], "inbound": inbound}, dt=DT
            )
        assert np.all(result.joint("inbound")[:, 2] == 0.0)
        assert np.all(result.joint("outbound")[:, 2] > 0.0)


class TestRefusals:
    def test_needs_two_templates(self, two_directions):
        env, models = two_directions
        with pytest.raises(ValueError, match="at least two templates"):
            decode_jointly(env, np.zeros((2, 8)), {"a": models["outbound"]}, dt=DT)

    @pytest.mark.parametrize("argument", ["env", "prior"])
    def test_mapping_keys_must_match(self, two_directions, argument):
        env, models = two_directions
        bad = {"outbound": env, "sideways": env}
        kwargs = {"prior": None}
        if argument == "env":
            target = bad
        else:
            target = env
            kwargs["prior"] = {"outbound": env.bin_sizes, "sideways": env.bin_sizes}
        with pytest.raises(ValueError, match=f"{argument} has keys"):
            decode_jointly(target, np.zeros((2, 8)), models, dt=DT, **kwargs)

    def test_default_prior_refuses_different_units(self):
        cm, m = _track(100.0), _track(1.0, bin_size=0.1, units="m")
        models = {"cm": np.ones((2, cm.n_bins)), "m": np.ones((2, m.n_bins))}
        with pytest.raises(ValueError, match="not measured alike"):
            decode_jointly({"cm": cm, "m": m}, np.zeros((2, 2)), models, dt=DT)

    def test_default_prior_refuses_length_against_area(self):
        track = _track(100.0)
        arena = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=10.0
        ).with_metadata(units="cm")
        models = {
            "track": np.ones((2, track.n_bins)),
            "arena": np.ones((2, arena.n_bins)),
        }
        envs = {"track": track, "arena": arena}
        with pytest.raises(ValueError, match="not measured alike"):
            decode_jointly(envs, np.zeros((2, 2)), models, dt=DT)

        # Explicit weights on one scale are allowed.
        result = decode_jointly(
            envs,
            np.zeros((2, 2)),
            models,
            dt=DT,
            prior={"track": np.ones(track.n_bins), "arena": np.ones(arena.n_bins)},
        )
        expected = track.n_bins / (track.n_bins + arena.n_bins)
        assert_allclose(result.template_probability[:, 0], expected)


class TestMoreRefusals:
    def test_separate_environments_need_units_for_the_default_prior(self):
        a = Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0
        )
        b = Environment.linear_track(endpoints=[[0.0, 0.0], [1.0, 0.0]], bin_size=0.1)
        models = {"a": np.ones((2, a.n_bins)), "b": np.ones((2, b.n_bins))}
        with pytest.raises(ValueError, match="do not all have units"):
            decode_jointly({"a": a, "b": b}, np.zeros((2, 2)), models, dt=DT)

    def test_prior_must_be_a_mapping(self, two_directions):
        env, models = two_directions
        with pytest.raises(ValueError, match="prior must be a mapping"):
            decode_jointly(env, np.zeros((2, 8)), models, dt=DT, prior=env.bin_sizes)

    def test_template_names_must_be_strings(self, two_directions):
        env, models = two_directions
        with pytest.raises(ValueError, match="must be strings"):
            decode_jointly(
                env,
                np.zeros((2, 8)),
                {0: models["outbound"], 1: models["inbound"]},
                dt=DT,
            )


class TestResult:
    def test_pickle_and_deepcopy_keep_read_only_state(self, two_directions):
        import copy
        import pickle

        env, models = two_directions
        result = decode_jointly(env, np.ones((3, 8), dtype=np.int64), models, dt=DT)
        for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            assert clone.template_names == result.template_names
            assert_allclose(clone.template_probability, result.template_probability)
            assert not clone.template_probability.flags.writeable
            assert_allclose(
                clone.conditional["inbound"].posterior,
                result.conditional["inbound"].posterior,
            )
            with pytest.raises(TypeError):
                clone.conditional["x"] = None  # type: ignore[index]

    def test_arrays_and_mapping_are_read_only(self, two_directions):
        env, models = two_directions
        result = decode_jointly(env, np.ones((3, 8), dtype=np.int64), models, dt=DT)
        with pytest.raises(ValueError, match="read-only"):
            result.template_probability[0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            result.joint("outbound")[0, 0] = 1.0
        with pytest.raises(TypeError):
            result.conditional["outbound"] = result.conditional["inbound"]  # type: ignore[index]

    def test_unknown_template_name(self, two_directions):
        env, models = two_directions
        result = decode_jointly(env, np.ones((3, 8), dtype=np.int64), models, dt=DT)
        with pytest.raises(KeyError, match="no template named"):
            result.joint("sideways")

    def test_summary(self, two_directions):
        env, models = two_directions
        result = decode_jointly(env, np.ones((3, 8), dtype=np.int64), models, dt=DT)
        summary = result.summary()
        assert summary["n_time_bins"] == 3
        assert summary["n_templates"] == 2
        assert_allclose(
            summary["mean_probability[outbound]"]
            + summary["mean_probability[inbound]"],
            1.0,
        )

    def test_constructor_refuses_rows_that_do_not_sum_to_one(self, two_directions):
        env, models = two_directions
        result = decode_jointly(env, np.ones((3, 8), dtype=np.int64), models, dt=DT)
        with pytest.raises(ValueError, match="sum to 1"):
            JointDecodingResult(
                template_names=result.template_names,
                template_probability=np.full((3, 2), 0.6),
                conditional=result.conditional,
            )
