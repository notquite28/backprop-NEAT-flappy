"""Deterministic evolution with interleaved REINFORCE/RMSProp updates."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .evolution import Clustering, cluster_best, pam_cluster, reproduce
from .game import FlappyEpisode
from .genome import (
    BASE_NODE_IDS,
    Genome,
    InnovationStore,
    INPUT_COUNT,
    initial_population,
    save_genome,
)
from .phenotype import batch_forward, batch_sample_actions, compile_genome


CONNECTION_PENALTY_FACTOR = 0.03
HALL_OF_FAME_SIZE = 5


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    generations: int = 50
    minimum_generations: int = 10
    population_size: int = 100
    cluster_count: int = 5
    max_frames: int = 1000
    backprop_cycles: int = 4
    eval_episodes: int = 3
    fitness_threshold: float = 100.0
    checkpoint_dir: Path = Path("checkpoints")

    def validate(self) -> None:
        counts = {
            "generations": self.generations,
            "minimum_generations": self.minimum_generations,
            "population_size": self.population_size,
            "cluster_count": self.cluster_count,
            "max_frames": self.max_frames,
            "backprop_cycles": self.backprop_cycles,
            "eval_episodes": self.eval_episodes,
        }
        for name, value in counts.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.population_size % self.cluster_count:
            raise ValueError("population_size must be divisible by cluster_count")
        if self.cluster_count != 5:
            raise ValueError("Hardmaru mode requires cluster_count=5")
        if self.population_size < self.cluster_count:
            raise ValueError("population_size must be at least cluster_count")
        if not math.isfinite(self.fitness_threshold):
            raise ValueError("fitness_threshold must be finite")


@dataclass(frozen=True)
class UpdateStats:
    accepted: int
    reverted: int
    pre_mean: float
    post_mean: float
    max_abs_weight_delta: float

def complexity_adjusted_fitness(raw_fitness: float, connection_count: int) -> float:
    """Apply Hardmaru's square-root connection-count penalty to a reward."""
    factor = 1.0 + CONNECTION_PENALTY_FACTOR * math.sqrt(connection_count)
    return raw_fitness / factor if raw_fitness >= 0.0 else raw_fitness * factor




def evaluate_candidates(
    candidates: Sequence[Genome],
    store: InnovationStore,
    seeds: Sequence[int],
    max_frames: int,
) -> None:
    totals = np.zeros(len(candidates), np.float64)
    scores = np.zeros(len(candidates), np.float64)
    frames = np.zeros(len(candidates), np.float64)
    for seed in seeds:
        episode = FlappyEpisode(len(candidates), seed, max_frames)
        episode_returns = np.zeros(len(candidates), np.float64)
        episode_scores = np.zeros(len(candidates), np.float64)
        episode_frames = np.zeros(len(candidates), np.float64)
        while not episode.done:
            start = episode.prepare_frame()
            live = [candidates[index] for index in start.bird_ids]
            logits = batch_forward(live, store, start.observations)
            end = episode.step(logits > 0.0)
            for bird_id, reward in zip(end.bird_ids, end.rewards, strict=True):
                episode_returns[bird_id] += float(reward)
                episode_frames[bird_id] += 1
                if reward >= 4.0:
                    episode_scores[bird_id] += 1
        totals += episode_returns
        scores += episode_scores
        frames += episode_frames
    divisor = float(len(seeds))
    for index, genome in enumerate(candidates):
        raw_fitness = float(totals[index] / divisor)
        genome.fitness = complexity_adjusted_fitness(
            raw_fitness, len(genome.connections)
        )
        genome.score = float(scores[index] / divisor)
        genome.frames = float(frames[index] / divisor)


def _discounted_advantages(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    values = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        values[index] = running
    values -= np.mean(values)
    deviation = float(np.std(values))
    if deviation > 1e-8:
        values /= deviation
    return values


@jax.jit
def _rmsprop_update(
    weights: jax.Array, caches: jax.Array, gradients: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    finite = jnp.all(jnp.isfinite(gradients))
    clipped = jnp.clip(gradients, -5.0, 5.0)
    next_caches = 0.999 * caches + 0.001 * clipped**2
    updated = jnp.clip(
        weights
        - 0.01 * clipped / jnp.sqrt(jnp.maximum(next_caches, 1e-8))
        - 0.001 * weights,
        -50.0,
        50.0,
    )
    delta = jnp.max(jnp.abs(updated - weights), initial=0.0)
    return updated, next_caches, finite, delta


def policy_gradient_cycle(
    candidates: Sequence[Genome],
    store: InnovationStore,
    root_key: jax.Array,
    seed: int,
    generation: int,
    cycle: int,
    max_frames: int,
) -> float:
    observations: list[list[np.ndarray]] = [[] for _ in candidates]
    actions: list[list[bool]] = [[] for _ in candidates]
    rewards: list[list[float]] = [[] for _ in candidates]
    episode = FlappyEpisode(len(candidates), seed, max_frames)
    while not episode.done:
        start = episode.prepare_frame()
        live = [candidates[index] for index in start.bird_ids]
        sampled = batch_sample_actions(
            live,
            store,
            start.observations,
            root_key,
            generation,
            cycle,
            episode.frame,
        )
        for row, candidate_index in enumerate(start.bird_ids):
            observations[candidate_index].append(start.observations[row].copy())
            actions[candidate_index].append(bool(sampled[row]))
        end = episode.step(sampled)
        for bird_id, reward in zip(end.bird_ids, end.rewards, strict=True):
            rewards[bird_id].append(float(reward))

    largest_delta = 0.0
    groups: dict[tuple, list[int]] = {}
    compiled_by_key = {}
    for index, genome in enumerate(candidates):
        compiled = compile_genome(genome, store)
        groups.setdefault(compiled.signature, []).append(index)
        compiled_by_key[compiled.signature] = compiled
    for signature, indexes in groups.items():
        compiled = compiled_by_key[signature]
        batch_size = len(indexes)
        observation_batch = np.zeros(
            (batch_size, max_frames, INPUT_COUNT), dtype=np.float32
        )
        action_batch = np.zeros((batch_size, max_frames), dtype=np.bool_)
        advantage_batch = np.zeros((batch_size, max_frames), dtype=np.float32)
        mask_batch = np.zeros((batch_size, max_frames), dtype=np.bool_)
        for row, index in enumerate(indexes):
            length = len(actions[index])
            observation_batch[row, :length] = observations[index]
            action_batch[row, :length] = actions[index]
            advantage_batch[row, :length] = _discounted_advantages(rewards[index])
            mask_batch[row, :length] = True
        weights = np.stack([compiled.weights_for(candidates[index]) for index in indexes])
        caches = np.asarray(
            [
                [candidates[index].rms_cache.get(i, 0.0) for i in compiled.innovations]
                for index in indexes
            ],
            dtype=np.float32,
        )
        _, gradients = compiled.batch_value_and_grad(
            jnp.asarray(weights),
            jnp.asarray(observation_batch),
            jnp.asarray(action_batch),
            jnp.asarray(advantage_batch),
            jnp.asarray(mask_batch),
        )
        updated, next_caches, finite, delta = _rmsprop_update(
            jnp.asarray(weights), jnp.asarray(caches), gradients
        )
        if not bool(finite):
            raise FloatingPointError("policy gradient is not finite")
        updated_np = np.asarray(updated)
        caches_np = np.asarray(next_caches)
        largest_delta = max(largest_delta, float(delta))
        for row, index in enumerate(indexes):
            genome = candidates[index]
            compiled.assign_weights(genome, updated_np[row])
            for innovation, value in zip(compiled.innovations, caches_np[row], strict=True):
                genome.rms_cache[innovation] = float(value)
    return largest_delta


def paired_backprop(
    candidates: Sequence[Genome],
    store: InnovationStore,
    config: TrainingConfig,
    generation: int,
    root_key: jax.Array,
) -> UpdateStats:
    seeds = [
        config.seed + 9_000 + index
        for index in range(config.eval_episodes)
    ]
    evaluate_candidates(candidates, store, seeds, config.max_frames)
    pre_fitness = [float(genome.fitness) for genome in candidates]
    pre_scores = [genome.score for genome in candidates]
    pre_frames = [genome.frames for genome in candidates]
    pre_weights = [{i: gene.weight for i, gene in genome.connections.items()} for genome in candidates]
    pre_caches = [dict(genome.rms_cache) for genome in candidates]
    max_delta = 0.0
    for cycle in range(config.backprop_cycles):
        episode_seed = config.seed + generation * 10_000 + cycle
        max_delta = max(
            max_delta,
            policy_gradient_cycle(
                candidates, store, root_key, episode_seed, generation, cycle, config.max_frames
            ),
        )
    evaluate_candidates(candidates, store, seeds, config.max_frames)
    post_fitness = [float(genome.fitness) for genome in candidates]
    accepted = 0
    reverted = 0
    for index, genome in enumerate(candidates):
        if post_fitness[index] < pre_fitness[index]:
            for innovation, weight in pre_weights[index].items():
                genome.connections[innovation].weight = weight
            genome.rms_cache = pre_caches[index]
            genome.fitness = pre_fitness[index]
            genome.score = pre_scores[index]
            genome.frames = pre_frames[index]
            reverted += 1
        else:
            accepted += 1
    return UpdateStats(
        accepted,
        reverted,
        float(np.mean(pre_fitness)),
        float(np.mean([genome.fitness for genome in candidates])),
        max_delta,
    )


def _rank_key(genome: Genome) -> tuple[float, float, int]:
    return (float(genome.fitness), genome.score, -genome.id)

def _representative(population: Sequence[Genome]) -> Genome:
    mean_fitness = float(np.mean([genome.fitness for genome in population]))
    return min(
        population,
        key=lambda genome: (
            abs(float(genome.fitness) - mean_fitness),
            -genome.score,
            genome.id,
        ),
    )

def _topology_key(genome: Genome) -> tuple[int, int, int, float, float, int]:
    return (
        len(genome.node_ids),
        len(genome.connections),
        sum(gene.enabled for gene in genome.connections.values()),
        float(genome.fitness),
        genome.score,
        -genome.id,
    )


def _generation_records(population: Sequence[Genome]) -> tuple[Genome, Genome, Genome]:
    if not population:
        raise ValueError("generation population cannot be empty")
    return (
        max(population, key=_rank_key),
        _representative(population),
        max(population, key=_topology_key),
    )

def _select_elites(
    population: Sequence[Genome], clustering: Clustering
) -> tuple[list[Genome], list[Genome]]:
    ranked = sorted(
        population,
        key=lambda genome: (-float(genome.fitness), -genome.score, genome.id),
    )
    return ranked[:HALL_OF_FAME_SIZE], cluster_best(clustering)


def _inject_elites(
    offspring: Sequence[Genome],
    hall_of_fame: Sequence[Genome],
    species_elites: Sequence[Genome],
    rng: random.Random,
    population_size: int,
    first_elite_id: int,
) -> tuple[list[Genome], list[Genome], int]:
    unique_sources: list[Genome] = []
    seen: set[int] = set()
    for genome in (*hall_of_fame, *species_elites):
        identity = id(genome)
        if identity not in seen:
            seen.add(identity)
            unique_sources.append(genome)
    unique_sources = unique_sources[:population_size]
    elite_copies = [
        genome.copy(first_elite_id + index)
        for index, genome in enumerate(unique_sources)
    ]
    keep_count = population_size - len(elite_copies)
    if keep_count > len(offspring):
        raise ValueError("not enough offspring to fill the population")
    keep_indexes = sorted(rng.sample(range(len(offspring)), keep_count))
    kept_offspring = [offspring[index] for index in keep_indexes]
    return (
        elite_copies + kept_offspring,
        kept_offspring,
        first_elite_id + len(elite_copies),
    )


def _save_elite_snapshots(
    checkpoint_dir: Path,
    generation: int,
    generation_elite: Genome,
    hall_of_fame: Sequence[Genome],
    species_elites: Sequence[Genome],
    store: InnovationStore,
) -> None:
    save_genome(
        checkpoint_dir / "elites" / f"generation_{generation:03d}.json",
        generation_elite,
        store,
        generation,
    )
    for rank, genome in enumerate(hall_of_fame):
        save_genome(
            checkpoint_dir
            / "hall_of_fame"
            / f"generation_{generation:03d}_rank_{rank:02d}.json",
            genome,
            store,
            generation,
        )
    for species, genome in enumerate(species_elites):
        save_genome(
            checkpoint_dir
            / "species_elites"
            / f"generation_{generation:03d}_species_{species:02d}.json",
            genome,
            store,
            generation,
        )






def train(config: TrainingConfig) -> Genome:
    config.validate()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    root_key = jax.random.PRNGKey(config.seed)
    store = InnovationStore.base()
    next_id = config.population_size
    mutable = initial_population(store, rng, config.population_size)
    save_genome(
        config.checkpoint_dir / "starting_genome.json", mutable[0], store, 0
    )
    initial_seeds = [config.seed + 9_000 + index for index in range(config.eval_episodes)]
    evaluate_candidates(mutable, store, initial_seeds, config.max_frames)
    clustering = pam_cluster(mutable, config.cluster_count, rng)
    hall, champions = _select_elites(mutable, clustering)
    generation_best, representative, topology_sample = _generation_records(mutable)
    global_best = generation_best.copy()
    save_genome(config.checkpoint_dir / "best_genome.json", global_best, store, 0)
    save_genome(config.checkpoint_dir / "generation_000.json", generation_best, store, 0)
    save_genome(
        config.checkpoint_dir / "representative_000.json", representative, store, 0
    )
    save_genome(
        config.checkpoint_dir / "topology_000.json", topology_sample, store, 0
    )
    _save_elite_snapshots(
        config.checkpoint_dir, 0, generation_best, hall, champions, store
    )
    sizes = [len(value) for _, value in sorted(clustering.clusters.items())]
    print(
        f"generation=0 best={global_best.fitness:.6f} mean={np.mean([g.fitness for g in mutable]):.6f} "
        f"score={global_best.score:.3f} frames={global_best.frames:.1f} nodes={len(global_best.node_ids)} "
        f"connections={len(global_best.connections)} accepted=0 reverted=0 clusters={sizes}"
    )
    effective_minimum = min(config.minimum_generations, config.generations)
    if effective_minimum <= 1 and float(global_best.fitness) >= config.fitness_threshold:
        return global_best

    for generation in range(1, config.generations):
        children, extinction = reproduce(
            clustering, store, rng, config.population_size, config.cluster_count, next_id
        )
        next_id += len(children)
        candidates, offspring, next_id = _inject_elites(
            children,
            hall,
            champions,
            rng,
            config.population_size,
            next_id,
        )
        if (
            len(candidates) != config.population_size
            or len({id(value) for value in candidates}) != config.population_size
        ):
            raise RuntimeError("fixed population elite invariant failed")
        updates = paired_backprop(candidates, store, config, generation, root_key)
        clustering = pam_cluster(candidates, config.cluster_count, rng)
        candidate_best = max(candidates, key=_rank_key)
        generation_best, representative, topology_sample = _generation_records(offspring)
        if _rank_key(candidate_best) > _rank_key(global_best):
            global_best = candidate_best.copy()
            save_genome(config.checkpoint_dir / "best_genome.json", global_best, store, generation)
        save_genome(
            config.checkpoint_dir / f"generation_{generation:03d}.json",
            generation_best,
            store,
            generation,
        )
        save_genome(
            config.checkpoint_dir / f"representative_{generation:03d}.json",
            representative,
            store,
            generation,
        )
        save_genome(
            config.checkpoint_dir / f"topology_{generation:03d}.json",
            topology_sample,
            store,
            generation,
        )
        hall, champions = _select_elites(candidates, clustering)
        _save_elite_snapshots(
            config.checkpoint_dir,
            generation,
            candidate_best,
            hall,
            champions,
            store,
        )
        sizes = [len(value) for _, value in sorted(clustering.clusters.items())]
        hidden_count = sum(len(genome.node_ids) > len(BASE_NODE_IDS) for genome in offspring)
        print(
            f"generation={generation} best={candidate_best.fitness:.6f} "
            f"mean={np.mean([g.fitness for g in candidates]):.6f} score={candidate_best.score:.3f} "
            f"frames={candidate_best.frames:.1f} nodes={len(candidate_best.node_ids)} "
            f"connections={len(candidate_best.connections)} accepted={updates.accepted} "
            f"reverted={updates.reverted} pre_mean={updates.pre_mean:.6f} "
            f"post_mean={updates.post_mean:.6f} max_abs_weight_delta={updates.max_abs_weight_delta:.9f} "
            f"extinction={int(extinction)} mutable={len(offspring)} "
            f"elites={len(candidates) - len(offspring)} hidden={hidden_count} "
            f"max_nodes={len(topology_sample.node_ids)} max_connections={len(topology_sample.connections)} "
            f"clusters={sizes}"
        )
        if (
            generation + 1 >= effective_minimum
            and float(global_best.fitness) >= config.fitness_threshold
        ):
            break
    return global_best
