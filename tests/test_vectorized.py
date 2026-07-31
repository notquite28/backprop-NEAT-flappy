import pytest
import jax
import jax.numpy as jnp
import numpy as np

from neat_flappy import vectorized
from neat_flappy.genome import (
    BASE_NODE_IDS,
    ConnectionGene,
    Genome,
    InnovationStore,
)
from neat_flappy.phenotype import compile_genome


def policy_genome(w0: float, w1: float, w2: float):
    """A base-topology genome with output logit w0*obs0 + w1*obs1 + w2.

    Innovations 0, 1, 2 feed inputs 0, 1 and the bias into the output with a
    sum aggregation, so the weights map directly onto the observation terms.
    """
    store = InnovationStore.base()
    genes = {
        innovation: ConnectionGene(innovation, src, 3, 0.0, True)
        for innovation, (src, _dst) in store.connection_endpoints.items()
    }
    genes[0].weight = w0
    genes[1].weight = w1
    genes[2].weight = w2
    genome = Genome(
        0,
        BASE_NODE_IDS,
        genes,
        rms_cache={innovation: 0.0 for innovation in genes},
    )
    return genome, store


def run(genome, store, max_frames, seed=0, sample=False, root_key=None):
    compiled = compile_genome(genome, store)
    weights = jnp.asarray(compiled.weights_for(genome))[None]
    heights = jnp.asarray(vectorized.pipe_heights(seed, max_frames // 40 + 16))
    key = jax.random.PRNGKey(0) if root_key is None else root_key
    return vectorized.run_episodes(
        weights,
        compiled.raw_forward,
        heights,
        max_frames,
        sample,
        key,
        0,
        0,
        jnp.asarray([genome.id], dtype=jnp.int32),
    )


def test_same_seed_is_bitwise_repeatable():
    genome, store = policy_genome(0.0, 0.0, 0.0)
    first = run(genome, store, 120)
    second = run(genome, store, 120)
    np.testing.assert_array_equal(first["returns"], second["returns"])
    np.testing.assert_array_equal(first["scores"], second["scores"])
    np.testing.assert_array_equal(first["frames"], second["frames"])


def test_no_action_policy_dies_on_floor_without_scoring():
    genome, store = policy_genome(0.0, 0.0, 0.0)  # logit 0 -> never jumps
    result = run(genome, store, 300)
    assert int(result["frames"][0]) < 300
    assert int(result["scores"][0]) == 0


def test_survival_reward_is_ten_cents_per_alive_frame():
    # Proportional controller that hovers near the gap center of seed 0's
    # first pipe (height 247, center 347 ~ start y 350) for the whole window.
    genome, store = policy_genome(-80.0, 0.5, 0.0)
    max_frames = 90  # first pipe passes around frame 95, so no pass reward here
    result = run(genome, store, max_frames)
    frames = int(result["frames"][0])
    assert frames == max_frames
    assert int(result["scores"][0]) == 0
    assert float(result["returns"][0]) == pytest.approx(0.1 * frames, abs=1e-4)


def test_passing_a_pipe_adds_five_and_increments_score():
    genome, store = policy_genome(-80.0, 0.5, 0.0)
    max_frames = 300  # survives the full horizon and passes three pipes
    result = run(genome, store, max_frames)
    frames = int(result["frames"][0])
    score = int(result["scores"][0])
    assert frames == max_frames
    assert score == 3
    assert float(result["returns"][0]) == pytest.approx(0.1 * frames + 5.0 * score, abs=1e-3)


def test_trajectory_shapes_and_alive_prefix():
    genome, store = policy_genome(0.0, 0.0, 0.0)
    max_frames = 80
    result = run(genome, store, max_frames)
    assert result["obs"].shape == (max_frames, 1, 2)
    assert result["actions"].shape == (max_frames, 1)
    assert result["alive"].shape == (max_frames, 1)
    alive = np.asarray(result["alive"][:, 0])
    frames = int(alive.sum())
    np.testing.assert_array_equal(alive[:frames], True)
    np.testing.assert_array_equal(alive[frames:], False)


def test_sampling_is_repeatable_for_identical_keys():
    genome, store = policy_genome(0.0, 0.0, 0.0)  # sigmoid(0) = 0.5 -> coin flips
    key = jax.random.PRNGKey(7)
    first = run(genome, store, 60, sample=True, root_key=key)
    second = run(genome, store, 60, sample=True, root_key=key)
    np.testing.assert_array_equal(first["actions"], second["actions"])
