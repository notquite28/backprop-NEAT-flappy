"""Hardmaru-style genome records with a strict feed-forward invariant."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np


class NodeKind(str, Enum):
    INPUT = "input"
    BIAS = "bias"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Activation(str, Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    GAUSSIAN = "gaussian"
    SIN = "sin"
    ABS = "abs"
    MULTIPLY = "multiply"
    SQUARE = "square"
    ADD = "add"


ACTIVATION_PALETTE = tuple(Activation)

INPUT_COUNT = 2
OUTPUT_COUNT = 1
INPUT_NODE_IDS = tuple(range(INPUT_COUNT))
BIAS_NODE_ID = 2
OUTPUT_NODE_ID = 3
BASE_NODE_IDS = frozenset((*INPUT_NODE_IDS, BIAS_NODE_ID, OUTPUT_NODE_ID))


@dataclass(frozen=True)
class NodeGene:
    id: int
    kind: NodeKind
    activation: Activation | None


@dataclass
class ConnectionGene:
    innovation: int
    src: int
    dst: int
    weight: float
    enabled: bool = True

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(self.innovation, self.src, self.dst, self.weight, self.enabled)


@dataclass
class Genome:
    id: int
    node_ids: frozenset[int]
    connections: dict[int, ConnectionGene]
    fitness: float | None = None
    score: float = 0.0
    frames: float = 0.0
    rms_cache: dict[int, float] = field(default_factory=dict)

    def copy(self, new_id: int | None = None) -> "Genome":
        return Genome(
            self.id if new_id is None else new_id,
            frozenset(self.node_ids),
            {i: gene.copy() for i, gene in self.connections.items()},
            self.fitness,
            self.score,
            self.frames,
            dict(self.rms_cache),
        )


class InnovationStore:
    """Append-only process-owned node and connection innovation records."""

    def __init__(self) -> None:
        self.nodes: dict[int, NodeGene] = {}
        self.connection_endpoints: dict[int, tuple[int, int]] = {}
        self.endpoint_to_innovation: dict[tuple[int, int], int] = {}

    @classmethod
    def base(cls) -> "InnovationStore":
        store = cls()
        for node_id in INPUT_NODE_IDS:
            store.add_node(NodeKind.INPUT, None, expected_id=node_id)
        store.add_node(NodeKind.BIAS, None, expected_id=BIAS_NODE_ID)
        store.add_node(NodeKind.OUTPUT, None, expected_id=OUTPUT_NODE_ID)
        for src in (*INPUT_NODE_IDS, BIAS_NODE_ID):
            store.connection(src, OUTPUT_NODE_ID)
        return store

    def add_node(
        self,
        kind: NodeKind,
        activation: Activation | None,
        expected_id: int | None = None,
    ) -> int:
        node_id = max(self.nodes, default=-1) + 1 if expected_id is None else expected_id
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        if (kind is NodeKind.HIDDEN) != (activation is not None):
            raise ValueError("only hidden nodes require an activation")
        self.nodes[node_id] = NodeGene(node_id, kind, activation)
        return node_id

    def connection(self, src: int, dst: int) -> int:
        endpoint = (src, dst)
        known = self.endpoint_to_innovation.get(endpoint)
        if known is not None:
            return known
        innovation = max(self.connection_endpoints, default=-1) + 1
        self.connection_endpoints[innovation] = endpoint
        self.endpoint_to_innovation[endpoint] = innovation
        return innovation

    def import_connection(self, innovation: int, src: int, dst: int) -> None:
        if innovation in self.connection_endpoints:
            raise ValueError("duplicate connection innovation")
        if (src, dst) in self.endpoint_to_innovation:
            raise ValueError("duplicate connection endpoint")
        self.connection_endpoints[innovation] = (src, dst)
        self.endpoint_to_innovation[(src, dst)] = innovation


def initial_population(
    store: InnovationStore,
    rng: random.Random,
    size: int = 100,
    start_id: int = 0,
) -> list[Genome]:
    if size <= 0:
        raise ValueError("population size must be positive")
    population = []
    for genome_id in range(start_id, start_id + size):
        genes: dict[int, ConnectionGene] = {}
        for innovation in range(INPUT_COUNT + 1):
            src, dst = store.connection_endpoints[innovation]
            weight = rng.gauss(0.0, 0.25) + rng.gauss(0.0, 0.005)
            genes[innovation] = ConnectionGene(innovation, src, dst, weight, True)
        population.append(
            Genome(
                genome_id,
                BASE_NODE_IDS,
                genes,
                rms_cache={innovation: 0.0 for innovation in genes},
            )
        )
    return population


def _reachable(genome: Genome, start: int, target: int) -> bool:
    outgoing: dict[int, list[int]] = {}
    for gene in genome.connections.values():
        if gene.enabled:
            outgoing.setdefault(gene.src, []).append(gene.dst)
    stack = [start]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(outgoing.get(node, ()))
    return False


def validate_genome(genome: Genome, store: InnovationStore) -> None:
    if not BASE_NODE_IDS.issubset(genome.node_ids):
        raise ValueError("genome is missing a base node")
    for node_id in genome.node_ids:
        node = store.nodes.get(node_id)
        if node is None:
            raise ValueError(f"unknown node {node_id}")
        if node.kind is NodeKind.HIDDEN and node.activation is None:
            raise ValueError("hidden node activation cannot be null")
        if node.kind is not NodeKind.HIDDEN and node.activation is not None:
            raise ValueError("base node activation must be null")
    for innovation, gene in genome.connections.items():
        if innovation != gene.innovation or innovation not in store.connection_endpoints:
            raise ValueError("invalid connection innovation")
        if store.connection_endpoints[innovation] != (gene.src, gene.dst):
            raise ValueError("innovation endpoint mismatch")
        if gene.src not in genome.node_ids or gene.dst not in genome.node_ids:
            raise ValueError("connection endpoint is not owned by genome")
        src_kind = store.nodes[gene.src].kind
        dst_kind = store.nodes[gene.dst].kind
        if src_kind is NodeKind.OUTPUT:
            raise ValueError("connection cannot leave output")
        if dst_kind in (NodeKind.INPUT, NodeKind.BIAS):
            raise ValueError("connection cannot target input or bias")
        if not math.isfinite(gene.weight):
            raise ValueError("connection weight must be finite")
    enabled = [gene for gene in genome.connections.values() if gene.enabled]
    indegree = {node_id: 0 for node_id in genome.node_ids}
    outgoing: dict[int, list[int]] = {}
    for gene in enabled:
        indegree[gene.dst] += 1
        outgoing.setdefault(gene.src, []).append(gene.dst)
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for dst in outgoing.get(node, ()):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
                queue.sort()
    if visited != len(indegree):
        raise ValueError("enabled connections contain a cycle")


def add_node(genome: Genome, store: InnovationStore, rng: random.Random) -> bool:
    connections = list(genome.connections.values())
    if not connections:
        return False
    old = rng.choice(connections)
    if not old.enabled:
        return False
    old.enabled = False
    activation = rng.choice(ACTIVATION_PALETTE)
    node_id = store.add_node(NodeKind.HIDDEN, activation)
    genome.node_ids = frozenset((*genome.node_ids, node_id))
    first = store.connection(old.src, node_id)
    second = store.connection(node_id, old.dst)
    genome.connections[first] = ConnectionGene(first, old.src, node_id, 1.0, True)
    genome.connections[second] = ConnectionGene(second, node_id, old.dst, old.weight, True)
    genome.rms_cache[first] = 0.0
    genome.rms_cache[second] = 0.0
    validate_genome(genome, store)
    return True


def add_connection(genome: Genome, store: InnovationStore, rng: random.Random) -> bool:
    sources = sorted(
        node_id
        for node_id in genome.node_ids
        if store.nodes[node_id].kind is not NodeKind.OUTPUT
    )
    targets = sorted(
        node_id
        for node_id in genome.node_ids
        if store.nodes[node_id].kind in (NodeKind.HIDDEN, NodeKind.OUTPUT)
    )
    if not sources or not targets:
        return False
    src = rng.choice(sources)
    dst = rng.choice(targets)
    if src == dst or _reachable(genome, dst, src):
        return False
    known = store.endpoint_to_innovation.get((src, dst))
    if known is not None and known in genome.connections:
        if genome.connections[known].enabled:
            return False
        innovation = known
    else:
        innovation = store.connection(src, dst)
    weight = rng.gauss(0.0, 0.25)
    if innovation in genome.connections:
        gene = genome.connections[innovation]
        gene.enabled = True
        gene.weight = weight
    else:
        genome.connections[innovation] = ConnectionGene(innovation, src, dst, weight, True)
    genome.rms_cache[innovation] = 0.0
    validate_genome(genome, store)
    return True


def crossover(
    left: Genome,
    right: Genome,
    child_id: int,
    store: InnovationStore,
    rng: random.Random,
) -> Genome:
    inherited: dict[int, ConnectionGene] = {}
    node_ids = set(BASE_NODE_IDS)
    for innovation in sorted(set(left.connections) | set(right.connections)):
        l_gene = left.connections.get(innovation)
        r_gene = right.connections.get(innovation)
        if l_gene is not None and r_gene is not None:
            selected = rng.choice((l_gene, r_gene)).copy()
            selected.enabled = not (not l_gene.enabled and not r_gene.enabled)
        else:
            selected = (l_gene or r_gene).copy()  # type: ignore[union-attr]
        node_ids.update((selected.src, selected.dst))
        inherited[innovation] = selected
    child = Genome(child_id, frozenset(node_ids), {}, rms_cache={})
    for innovation, gene in inherited.items():
        if gene.enabled and _reachable(child, gene.dst, gene.src):
            gene.enabled = False
        child.connections[innovation] = gene
        child.rms_cache[innovation] = 0.0
    validate_genome(child, store)
    return child


def mutate_offspring(genome: Genome, store: InnovationStore, rng: random.Random) -> None:
    if rng.random() < 0.2:
        add_node(genome, store, rng)
    if rng.random() < 0.5:
        add_connection(genome, store, rng)
    for gene in genome.connections.values():
        if rng.random() < 0.9:
            gene.weight += rng.gauss(0.0, 0.005)
    genome.rms_cache = {innovation: 0.0 for innovation in genome.connections}
    validate_genome(genome, store)


def genome_record(genome: Genome, store: InnovationStore, generation: int) -> dict[str, Any]:
    validate_genome(genome, store)
    return {
        "version": 1,
        "generation": generation,
        "fitness": float(genome.fitness if genome.fitness is not None else 0.0),
        "score": float(genome.score),
        "input_count": INPUT_COUNT,
        "output_count": OUTPUT_COUNT,
        "nodes": [
            {
                "id": node_id,
                "kind": store.nodes[node_id].kind.value,
                "activation": (
                    store.nodes[node_id].activation.value
                    if store.nodes[node_id].activation is not None
                    else None
                ),
            }
            for node_id in sorted(genome.node_ids)
        ],
        "connections": [
            {
                "innovation": gene.innovation,
                "src": gene.src,
                "dst": gene.dst,
                "weight": float(gene.weight),
                "enabled": gene.enabled,
            }
            for gene in sorted(genome.connections.values(), key=lambda value: value.innovation)
        ],
    }


def save_genome(path: Path, genome: Genome, store: InnovationStore, generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genome_record(genome, store, generation), indent=2) + "\n")


def load_genome(path: Path) -> tuple[Genome, InnovationStore, int]:
    data = json.loads(path.read_text())
    expected = {
        "version", "generation", "fitness", "score", "input_count", "output_count",
        "nodes", "connections",
    }
    if set(data) != expected or data["version"] != 1:
        raise ValueError("unsupported or malformed genome schema")
    if data["input_count"] != INPUT_COUNT or data["output_count"] != OUTPUT_COUNT:
        raise ValueError("genome must have two inputs and one output")
    store = InnovationStore()
    node_ids: set[int] = set()
    for record in sorted(data["nodes"], key=lambda value: value["id"]):
        if set(record) != {"id", "kind", "activation"}:
            raise ValueError("malformed node record")
        kind = NodeKind(record["kind"])
        activation = Activation(record["activation"]) if record["activation"] is not None else None
        node_id = store.add_node(kind, activation, expected_id=record["id"])
        node_ids.add(node_id)
    genes: dict[int, ConnectionGene] = {}
    for record in sorted(data["connections"], key=lambda value: value["innovation"]):
        if set(record) != {"innovation", "src", "dst", "weight", "enabled"}:
            raise ValueError("malformed connection record")
        innovation = int(record["innovation"])
        store.import_connection(innovation, int(record["src"]), int(record["dst"]))
        genes[innovation] = ConnectionGene(
            innovation, int(record["src"]), int(record["dst"]),
            float(record["weight"]), bool(record["enabled"]),
        )
    genome = Genome(-1, frozenset(node_ids), genes, float(data["fitness"]), float(data["score"]))
    genome.rms_cache = {innovation: 0.0 for innovation in genes}
    validate_genome(genome, store)
    return genome, store, int(data["generation"])
