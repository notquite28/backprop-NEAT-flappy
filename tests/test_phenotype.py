import math

import jax.numpy as jnp
import numpy as np
import pytest

from neat_flappy.genome import (
    Activation,
    BASE_NODE_IDS,
    ConnectionGene,
    Genome,
    InnovationStore,
    NodeKind,
    OUTPUT_NODE_ID,
    initial_population,
)
from neat_flappy.phenotype import clear_compile_cache, compile_cache_size, compile_genome


def activation_genome(activation):
    store = InnovationStore.base()
    hidden = store.add_node(NodeKind.HIDDEN, activation)
    first = store.connection(0, hidden)
    second = store.connection(1, hidden)
    out = store.connection(hidden, OUTPUT_NODE_ID)
    genome = Genome(1, frozenset((*BASE_NODE_IDS, hidden)), {
        first: ConnectionGene(first, 0, hidden, 1.0),
        second: ConnectionGene(second, 1, hidden, 1.0),
        out: ConnectionGene(out, hidden, OUTPUT_NODE_ID, 1.0),
    })
    return genome, store


@pytest.mark.parametrize(("activation", "expected"), [
    (Activation.SIGMOID, 1 / (1 + math.exp(-5))),
    (Activation.TANH, math.tanh(5)),
    (Activation.RELU, 5),
    (Activation.GAUSSIAN, math.exp(-25)),
    (Activation.SIN, math.sin(5)),
    (Activation.ABS, 5),
    (Activation.MULTIPLY, 6),
    (Activation.SQUARE, 25),
    (Activation.ADD, 5),
])
def test_all_activation_and_aggregation_operators(activation, expected):
    genome, store = activation_genome(activation)
    compiled = compile_genome(genome, store)
    actual = compiled.forward(compiled.weights_for(genome), jnp.asarray([2.0, 3.0]))
    assert float(actual) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(("activation", "expected"), [
    (Activation.ADD, 0.0),
    (Activation.MULTIPLY, 1.0),
])
def test_zero_input_aggregation_identity(activation, expected):
    store = InnovationStore.base()
    hidden = store.add_node(NodeKind.HIDDEN, activation)
    out = store.connection(hidden, OUTPUT_NODE_ID)
    genome = Genome(1, frozenset((*BASE_NODE_IDS, hidden)), {
        out: ConnectionGene(out, hidden, OUTPUT_NODE_ID, 1.0),
    })
    compiled = compile_genome(genome, store)
    assert float(compiled.forward(compiled.weights_for(genome), jnp.zeros(2))) == pytest.approx(expected)


def test_hand_calculated_dag_and_weight_only_cache_reuse():
    clear_compile_cache()
    genome, store = activation_genome(Activation.SQUARE)
    first = compile_genome(genome, store)
    assert float(first.forward(first.weights_for(genome), jnp.asarray([2.0, 3.0]))) == pytest.approx(25)
    next(iter(genome.connections.values())).weight = 2.0
    second = compile_genome(genome, store)
    assert second is first
    assert compile_cache_size() == 1


def test_policy_gradient_is_finite_and_improves_selected_actions():
    store = InnovationStore.base()
    genome = initial_population(store, __import__("random").Random(0), 1)[0]
    for gene in genome.connections.values():
        gene.weight = 0.0
    compiled = compile_genome(genome, store)
    weights = compiled.weights_for(genome)
    observations = jnp.asarray([[1.0, 0.0], [-1.0, 0.0]])
    actions = jnp.asarray([True, False])
    advantages = jnp.asarray([1.0, 1.0])
    mask = jnp.asarray([True, True])
    loss, gradient = compiled.value_and_grad(weights, observations, actions, advantages, mask)
    assert np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(gradient)))
    updated = weights - 0.1 * np.asarray(gradient)
    before = np.asarray([compiled.forward(weights, row) for row in observations])
    after = np.asarray([compiled.forward(updated, row) for row in observations])
    assert after[0] > before[0]
    assert after[1] < before[1]


def test_jitted_vmap_paths_match_scalar_paths_and_sampling_is_repeatable():
    store = InnovationStore.base()
    genome = initial_population(store, __import__("random").Random(5), 1)[0]
    compiled = compile_genome(genome, store)
    weights = jnp.asarray(compiled.weights_for(genome))
    batched_weights = jnp.stack((weights, weights))
    observations = jnp.asarray([[0.1, 0.2], [0.4, 0.5]])
    expected = jnp.stack(
        (compiled.forward(weights, observations[0]), compiled.forward(weights, observations[1]))
    )
    np.testing.assert_allclose(
        compiled.batch_forward(batched_weights, observations), expected, rtol=1e-6
    )

    trajectories = jnp.stack((observations, observations))
    actions = jnp.asarray([[True, False], [False, True]])
    advantages = jnp.ones((2, 2), dtype=jnp.float32)
    mask = jnp.ones((2, 2), dtype=jnp.bool_)
    losses, gradients = compiled.batch_value_and_grad(
        batched_weights, trajectories, actions, advantages, mask
    )
    for row in range(2):
        scalar_loss, scalar_gradient = compiled.value_and_grad(
            weights, trajectories[row], actions[row], advantages[row], mask[row]
        )
        np.testing.assert_allclose(losses[row], scalar_loss, rtol=1e-6)
        np.testing.assert_allclose(gradients[row], scalar_gradient, rtol=1e-6, atol=1e-7)

    key = __import__("jax").random.PRNGKey(11)
    ids = jnp.asarray([10, 20], dtype=jnp.int32)
    first = compiled.batch_sample(batched_weights, observations, key, 2, 3, ids, 4)
    second = compiled.batch_sample(batched_weights, observations, key, 2, 3, ids, 4)
    np.testing.assert_array_equal(first, second)
