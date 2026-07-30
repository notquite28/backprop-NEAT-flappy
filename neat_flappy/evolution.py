"""Hardmaru-style clustering, archives, extinction, and reproduction."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np

from .genome import Genome, InnovationStore, crossover, mutate_offspring


@dataclass(frozen=True)
class Clustering:
    medoids: tuple[int, ...]
    assignments: dict[int, int]
    clusters: dict[int, tuple[Genome, ...]]


def compatibility_distance(left: Genome, right: Genome) -> float:
    left_keys = set(left.connections)
    right_keys = set(right.connections)
    matching = left_keys & right_keys
    left_max = max(left_keys, default=-1)
    right_max = max(right_keys, default=-1)
    disjoint = 0
    excess = 0
    for innovation in left_keys - right_keys:
        if innovation > right_max:
            excess += 1
        else:
            disjoint += 1
    for innovation in right_keys - left_keys:
        if innovation > left_max:
            excess += 1
        else:
            disjoint += 1
    differences = [
        abs(left.connections[i].weight - right.connections[i].weight)
        for i in matching
        if left.connections[i].enabled and right.connections[i].enabled
    ]
    weight_difference = float(np.mean(differences)) if differences else 0.0
    normalizer = max(len(left.node_ids), len(right.node_ids), 1)
    return 10.0 * excess / normalizer + 10.0 * disjoint / normalizer + 0.1 * weight_difference


def _assign(population: Sequence[Genome], medoid_ids: Sequence[int]) -> Clustering:
    by_id = {genome.id: genome for genome in population}
    assignments: dict[int, int] = {}
    grouped: dict[int, list[Genome]] = {medoid: [] for medoid in medoid_ids}
    for genome in sorted(population, key=lambda value: value.id):
        medoid = min(
            medoid_ids,
            key=lambda candidate: (compatibility_distance(genome, by_id[candidate]), candidate),
        )
        assignments[genome.id] = medoid
        grouped[medoid].append(genome)
    return Clustering(
        tuple(sorted(medoid_ids)),
        assignments,
        {medoid: tuple(grouped[medoid]) for medoid in sorted(grouped)},
    )


def _cost(population: Sequence[Genome], medoid_ids: Sequence[int]) -> float:
    by_id = {genome.id: genome for genome in population}
    return sum(
        min(compatibility_distance(genome, by_id[medoid]) for medoid in medoid_ids)
        for genome in population
    )


def pam_cluster(
    population: Sequence[Genome],
    cluster_count: int,
    rng: random.Random,
    initial_medoids: Sequence[int] | None = None,
) -> Clustering:
    if cluster_count <= 0 or cluster_count > len(population):
        raise ValueError("cluster count must be positive and no larger than population")
    ids = sorted(genome.id for genome in population)
    medoids = sorted(initial_medoids if initial_medoids is not None else rng.sample(ids, cluster_count))
    if len(set(medoids)) != cluster_count or not set(medoids).issubset(ids):
        raise ValueError("medoids must be distinct population IDs")
    current_cost = _cost(population, medoids)
    for _ in range(100):
        accepted = False
        nonmedoids = [identifier for identifier in ids if identifier not in medoids]
        for outgoing in sorted(medoids):
            for incoming in nonmedoids:
                candidate = sorted(incoming if value == outgoing else value for value in medoids)
                candidate_cost = _cost(population, candidate)
                if candidate_cost < current_cost:
                    medoids = candidate
                    current_cost = candidate_cost
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            break
    return _assign(population, medoids)


def cluster_best(clustering: Clustering) -> list[Genome]:
    return [
        min(
            members,
            key=lambda genome: (-(genome.fitness if genome.fitness is not None else -float("inf")), genome.id),
        )
        for _, members in sorted(clustering.clusters.items())
    ]


def cluster_scores(clustering: Clustering) -> dict[int, float]:
    """Return each cluster's best fitness, as Hardmaru's extinction rule does."""
    return {
        medoid: max(float(member.fitness) for member in members)
        for medoid, members in clustering.clusters.items()
    }


def _roulette(members: Sequence[Genome], rng: random.Random) -> Genome:
    values = [genome.fitness for genome in members]
    if any(value is None for value in values):
        raise ValueError("all parents must have fitness")
    maximum = max(float(value) for value in values)
    weights = [1.0 / (maximum - float(value) + 0.01) for value in values]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for genome, weight in zip(members, weights, strict=True):
        cumulative += weight
        if threshold <= cumulative:
            return genome
    return members[-1]


def reproduce(
    clustering: Clustering,
    store: InnovationStore,
    rng: random.Random,
    population_size: int,
    cluster_count: int,
    first_child_id: int,
) -> tuple[list[Genome], bool]:
    if population_size <= 0 or cluster_count <= 0 or population_size % cluster_count:
        raise ValueError("population_size must be divisible by cluster_count")
    if len(clustering.clusters) != cluster_count:
        raise ValueError("clustering count does not match configuration")
    scores = cluster_scores(clustering)
    extinction = rng.random() < 0.5
    worst = min(scores, key=lambda medoid: (scores[medoid], medoid))
    best = max(scores, key=lambda medoid: (scores[medoid], -medoid))
    per_cluster = population_size // cluster_count
    children: list[Genome] = []
    for allocated in sorted(clustering.clusters):
        source = best if extinction and allocated == worst else allocated
        members = clustering.clusters[source]
        for _ in range(per_cluster):
            left = _roulette(members, rng)
            right = _roulette(members, rng)
            child = crossover(left, right, first_child_id + len(children), store, rng)
            mutate_offspring(child, store, rng)
            children.append(child)
    return children, extinction
