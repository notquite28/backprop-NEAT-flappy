"""Compile validated acyclic genomes into fixed-topology JAX programs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .genome import (
    Activation,
    BIAS_NODE_ID,
    Genome,
    InnovationStore,
    INPUT_NODE_IDS,
    NodeKind,
    OUTPUT_NODE_ID,
    validate_genome,
)

ACTIVATION_CODE = {activation: index for index, activation in enumerate(Activation)}


@dataclass(frozen=True)
class CompiledPhenotype:
    signature: tuple
    innovations: tuple[int, ...]
    forward: Callable[[jax.Array, jax.Array], jax.Array]
    batch_forward: Callable[[jax.Array, jax.Array], jax.Array]
    batch_sample: Callable
    policy_loss: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    value_and_grad: Callable
    batch_value_and_grad: Callable

    def weights_for(self, genome: Genome) -> np.ndarray:
        return np.asarray([genome.connections[i].weight for i in self.innovations], np.float32)

    def assign_weights(self, genome: Genome, weights: np.ndarray) -> None:
        for innovation, weight in zip(self.innovations, weights, strict=True):
            genome.connections[innovation].weight = float(weight)


_CACHE: dict[tuple, CompiledPhenotype] = {}


def clear_compile_cache() -> None:
    _CACHE.clear()


def compile_cache_size() -> int:
    return len(_CACHE)


def _reachable_graph(genome: Genome) -> tuple[set[int], list]:
    enabled = [gene for gene in genome.connections.values() if gene.enabled]
    incoming: dict[int, list] = {}
    for gene in enabled:
        incoming.setdefault(gene.dst, []).append(gene)
    nodes = {OUTPUT_NODE_ID}
    stack = [OUTPUT_NODE_ID]
    while stack:
        dst = stack.pop()
        for gene in incoming.get(dst, ()):
            if gene.src not in nodes:
                nodes.add(gene.src)
                stack.append(gene.src)
    edges = [gene for gene in enabled if gene.src in nodes and gene.dst in nodes]
    return nodes, edges


def structural_signature(genome: Genome, store: InnovationStore) -> tuple:
    validate_genome(genome, store)
    nodes, edges = _reachable_graph(genome)
    node_part = tuple(
        (node_id, -1 if store.nodes[node_id].activation is None else ACTIVATION_CODE[store.nodes[node_id].activation])
        for node_id in sorted(nodes)
    )
    edge_part = tuple(
        (gene.innovation, gene.src, gene.dst)
        for gene in sorted(edges, key=lambda value: value.innovation)
    )
    return node_part, edge_part


def compile_genome(genome: Genome, store: InnovationStore) -> CompiledPhenotype:
    signature = structural_signature(genome, store)
    cached = _CACHE.get(signature)
    if cached is not None:
        return cached
    nodes, edges = _reachable_graph(genome)
    ordered_nodes = sorted(nodes)
    slot = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    innovations = tuple(gene.innovation for gene in sorted(edges, key=lambda value: value.innovation))
    weight_slot = {innovation: index for index, innovation in enumerate(innovations)}
    incoming: dict[int, list] = {}
    outgoing: dict[int, list[int]] = {}
    indegree = {node_id: 0 for node_id in nodes}
    for gene in edges:
        incoming.setdefault(gene.dst, []).append(gene)
        outgoing.setdefault(gene.src, []).append(gene.dst)
        indegree[gene.dst] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    topological: list[int] = []
    while queue:
        node = queue.pop(0)
        topological.append(node)
        for dst in sorted(outgoing.get(node, ())):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
                queue.sort()
    if len(topological) != len(nodes):
        raise ValueError("cannot compile a cyclic genome")

    schedule = []
    for node_id in topological:
        node = store.nodes[node_id]
        if node.kind in (NodeKind.INPUT, NodeKind.BIAS):
            continue
        genes = sorted(incoming.get(node_id, ()), key=lambda value: value.innovation)
        sources = tuple(slot[gene.src] for gene in genes)
        weights = tuple(weight_slot[gene.innovation] for gene in genes)
        activation = node.activation
        code = -1 if activation is None else ACTIVATION_CODE[activation]
        schedule.append((slot[node_id], sources, weights, code))

    def raw_forward(weights: jax.Array, observation: jax.Array) -> jax.Array:
        values = jnp.zeros((len(ordered_nodes),), dtype=jnp.float32)
        for observation_slot, input_id in enumerate(INPUT_NODE_IDS):
            if input_id in slot:
                values = values.at[slot[input_id]].set(observation[observation_slot])
        if BIAS_NODE_ID in slot:
            values = values.at[slot[BIAS_NODE_ID]].set(1.0)

        def sigmoid(value: jax.Array) -> jax.Array:
            return jax.nn.sigmoid(value)
        operations = (
            sigmoid,
            jnp.tanh,
            lambda value: jnp.maximum(value, 0.0),
            lambda value: jnp.exp(-(value * value)),
            jnp.sin,
            jnp.abs,
            lambda value: value,
            lambda value: value * value,
            lambda value: value,
        )
        multiply_code = ACTIVATION_CODE[Activation.MULTIPLY]
        for destination, sources, weight_indexes, code in schedule:
            if sources:
                terms = values[jnp.asarray(sources)] * weights[jnp.asarray(weight_indexes)]
                aggregate = jnp.prod(terms) if code == multiply_code else jnp.sum(terms)
            else:
                aggregate = jnp.asarray(1.0 if code == multiply_code else 0.0, jnp.float32)
            result = aggregate if code < 0 else jax.lax.switch(code, operations, aggregate)
            values = values.at[destination].set(result)
        return values[slot[OUTPUT_NODE_ID]]

    def raw_policy_loss(
        weights: jax.Array,
        observations: jax.Array,
        actions: jax.Array,
        advantages: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        logits = jax.vmap(raw_forward, in_axes=(None, 0))(weights, observations)
        actions_f = actions.astype(jnp.float32)
        log_prob = -actions_f * jax.nn.softplus(-logits) - (1.0 - actions_f) * jax.nn.softplus(logits)
        probability = jax.nn.sigmoid(logits)
        entropy = -probability * jax.nn.log_sigmoid(logits) - (1.0 - probability) * jax.nn.log_sigmoid(-logits)
        valid = mask.astype(jnp.float32)
        count = jnp.sum(valid)
        policy = -jnp.sum(valid * jax.lax.stop_gradient(advantages) * log_prob) / count
        return policy - 0.01 * jnp.sum(valid * entropy) / count

    def raw_batch_sample(
        weights: jax.Array,
        observations: jax.Array,
        root_key: jax.Array,
        generation: jax.Array,
        cycle: jax.Array,
        genome_ids: jax.Array,
        frame: jax.Array,
    ) -> jax.Array:
        logits = jax.vmap(raw_forward, in_axes=(0, 0))(weights, observations)
        generation_key = jax.random.fold_in(root_key, generation)
        cycle_key = jax.random.fold_in(generation_key, cycle)

        def sample(genome_id: jax.Array, logit: jax.Array) -> jax.Array:
            genome_key = jax.random.fold_in(cycle_key, genome_id)
            frame_key = jax.random.fold_in(genome_key, frame)
            return jax.random.bernoulli(frame_key, jax.nn.sigmoid(logit))

        return jax.vmap(sample)(genome_ids, logits)


    forward = jax.jit(raw_forward)
    batched_forward = jax.jit(jax.vmap(raw_forward, in_axes=(0, 0)))
    batch_sample = jax.jit(raw_batch_sample)
    policy_loss = jax.jit(raw_policy_loss)
    value_and_grad = jax.jit(jax.value_and_grad(raw_policy_loss))
    batch_value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(raw_policy_loss), in_axes=(0, 0, 0, 0, 0))
    )
    compiled = CompiledPhenotype(
        signature,
        innovations,
        forward,
        batched_forward,
        batch_sample,
        policy_loss,
        value_and_grad,
        batch_value_and_grad,
    )
    _CACHE[signature] = compiled
    return compiled


def batch_forward(
    genomes: Sequence[Genome],
    stores: Sequence[InnovationStore] | InnovationStore,
    observations: np.ndarray,
) -> np.ndarray:
    """Group matching structures and vmap their shared compiled function."""
    if len(genomes) != len(observations):
        raise ValueError("genomes and observations must have equal length")
    store_list = [stores] * len(genomes) if isinstance(stores, InnovationStore) else list(stores)
    result = np.empty(len(genomes), dtype=np.float32)
    groups: dict[tuple, list[int]] = {}
    compiled_by_key: dict[tuple, CompiledPhenotype] = {}
    for index, (genome, store) in enumerate(zip(genomes, store_list, strict=True)):
        compiled = compile_genome(genome, store)
        groups.setdefault(compiled.signature, []).append(index)
        compiled_by_key[compiled.signature] = compiled
    for key, indexes in groups.items():
        compiled = compiled_by_key[key]
        weights = jnp.asarray(np.stack([compiled.weights_for(genomes[i]) for i in indexes]))
        obs = jnp.asarray(observations[indexes])
        values = compiled.batch_forward(weights, obs)
        result[indexes] = np.asarray(values)
    return result


def batch_sample_actions(
    genomes: Sequence[Genome],
    store: InnovationStore,
    observations: np.ndarray,
    root_key: jax.Array,
    generation: int,
    cycle: int,
    frame: int,
) -> np.ndarray:
    """Group matching structures and sample all live policies on device."""
    if len(genomes) != len(observations):
        raise ValueError("genomes and observations must have equal length")
    result = np.empty(len(genomes), dtype=np.bool_)
    groups: dict[tuple, list[int]] = {}
    compiled_by_key: dict[tuple, CompiledPhenotype] = {}
    for index, genome in enumerate(genomes):
        compiled = compile_genome(genome, store)
        groups.setdefault(compiled.signature, []).append(index)
        compiled_by_key[compiled.signature] = compiled
    for key, indexes in groups.items():
        compiled = compiled_by_key[key]
        weights = jnp.asarray(np.stack([compiled.weights_for(genomes[i]) for i in indexes]))
        obs = jnp.asarray(observations[indexes])
        ids = jnp.asarray([genomes[i].id for i in indexes], dtype=jnp.int32)
        sampled = compiled.batch_sample(
            weights, obs, root_key, generation, cycle, ids, frame
        )
        result[indexes] = np.asarray(sampled)
    return result
