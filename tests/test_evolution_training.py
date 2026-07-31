import random

import jax
import numpy as np
import pytest

from neat_flappy.evolution import compatibility_distance, pam_cluster, reproduce
from neat_flappy.genome import (
    BASE_NODE_IDS,
    ConnectionGene,
    Genome,
    InnovationStore,
    add_node,
    initial_population,
    load_genome,
)
from neat_flappy.training import (
    TrainingConfig,
    complexity_adjusted_fitness,
    paired_backprop,
    policy_gradient_cycle,
    _generation_records,
    _inject_elites,
    _save_elite_snapshots,
    _select_elites,
)


def population_with_fitness(size=10):
    store = InnovationStore.base()
    population = initial_population(store, random.Random(4), size)
    for index, genome in enumerate(population):
        genome.fitness = float(index - 5)
    return population, store


def test_distance_and_pam_are_deterministic():
    population, _ = population_with_fitness()
    assert compatibility_distance(population[0], population[0]) == 0
    first = pam_cluster(population, 5, random.Random(7), initial_medoids=[0, 2, 4, 6, 8])
    second = pam_cluster(population, 5, random.Random(99), initial_medoids=[0, 2, 4, 6, 8])
    assert first.medoids == second.medoids
    assert first.assignments == second.assignments
    assert all(first.assignments[medoid] == medoid for medoid in first.medoids)


def test_reproduction_count_and_extinction_are_seeded():
    population, store = population_with_fitness()
    clustering = pam_cluster(population, 5, random.Random(2), initial_medoids=[0, 2, 4, 6, 8])
    children_a, extinction_a = reproduce(clustering, store, random.Random(12), 10, 5, 100)

    population_b, store_b = population_with_fitness()
    clustering_b = pam_cluster(population_b, 5, random.Random(2), initial_medoids=[0, 2, 4, 6, 8])
    children_b, extinction_b = reproduce(clustering_b, store_b, random.Random(12), 10, 5, 100)
    assert len(children_a) == 10
    assert extinction_a == extinction_b
    assert [sorted(g.connections) for g in children_a] == [sorted(g.connections) for g in children_b]


def test_hardmaru_connection_penalty_favors_simpler_genomes():
    simple = complexity_adjusted_fitness(100.0, 3)
    complex_ = complexity_adjusted_fitness(100.0, 12)
    assert simple == pytest.approx(100.0 / (1.0 + 0.03 * 3**0.5))
    assert simple > complex_
    assert complexity_adjusted_fitness(-1.0, 12) < complexity_adjusted_fitness(-1.0, 3)

def test_generation_records_use_only_supplied_offspring():
    offspring, store = population_with_fitness(3)
    archive = initial_population(store, random.Random(8), 1, start_id=99)[0]
    archive.fitness = 1000.0
    assert add_node(offspring[0], store, random.Random(2))

    generation_best, representative, topology = _generation_records(offspring)

    assert generation_best in offspring
    assert representative in offspring
    assert topology is offspring[0]
    assert archive not in (generation_best, representative, topology)

def test_hall_and_species_elites_inject_without_growing_population(tmp_path):
    population, store = population_with_fitness()
    clustering = pam_cluster(
        population, 5, random.Random(2), initial_medoids=[0, 2, 4, 6, 8]
    )
    hall, species_elites = _select_elites(population, clustering)
    offspring = initial_population(store, random.Random(8), 10, start_id=100)

    candidates, kept, next_id = _inject_elites(
        offspring, hall, species_elites, random.Random(11), 10, 200
    )

    assert [genome.fitness for genome in hall] == [4.0, 3.0, 2.0, 1.0, 0.0]
    assert len(species_elites) == 5
    assert len(candidates) == 10
    assert len({genome.id for genome in candidates}) == 10
    assert len(kept) + (next_id - 200) == 10
    elite_count = next_id - 200
    assert max(float(genome.fitness) for genome in candidates[:elite_count]) == 4.0

    _save_elite_snapshots(
        tmp_path, 3, candidates[0], hall, species_elites, store
    )
    assert len(list((tmp_path / "hall_of_fame").glob("*.json"))) == 5
    assert len(list((tmp_path / "species_elites").glob("*.json"))) == 5
    saved, _, generation = load_genome(
        tmp_path / "elites" / "generation_003.json"
    )
    assert generation == 3
    assert saved.fitness == candidates[0].fitness


def test_population_divisibility_validation():
    with pytest.raises(ValueError, match="divisible"):
        TrainingConfig(population_size=11).validate()


def test_harmful_gradient_update_rolls_back(monkeypatch, tmp_path):
    candidates, store = population_with_fitness(1)
    original = candidates[0].connections[0].weight
    evaluation_seeds = []

    def fake_evaluate(items, _store, seeds, _frames):
        evaluation_seeds.append(tuple(seeds))
        for genome in items:
            genome.fitness = -abs(genome.connections[0].weight)
            genome.score = 0
            genome.frames = 1

    def harmful(items, _store, _root, _seed, _generation, _cycle, _frames):
        for genome in items:
            genome.connections[0].weight += 100
            genome.rms_cache[0] = 10
        return 100.0

    monkeypatch.setattr("neat_flappy.training.evaluate_candidates", fake_evaluate)
    monkeypatch.setattr("neat_flappy.training.policy_gradient_cycle", harmful)
    config = TrainingConfig(
        population_size=5, cluster_count=5, generations=1, max_frames=1,
        backprop_cycles=1, eval_episodes=1, checkpoint_dir=tmp_path,
    )
    stats = paired_backprop(candidates, store, config, 1, __import__("jax").random.PRNGKey(0))
    assert stats.reverted == 1 and stats.accepted == 0
    assert candidates[0].connections[0].weight == pytest.approx(original)
    assert candidates[0].fitness == pytest.approx(-abs(original))
    assert evaluation_seeds == [(9_000,), (9_000,)]



def base_genome(genome_id, w0, w1, w2):
    """A base-topology genome: output logit w0*obs0 + w1*obs1 + w2."""
    store = InnovationStore.base()
    genes = {
        innovation: ConnectionGene(innovation, src, 3, 0.0, True)
        for innovation, (src, _dst) in store.connection_endpoints.items()
    }
    genes[0].weight = w0
    genes[1].weight = w1
    genes[2].weight = w2
    genome = Genome(
        genome_id,
        BASE_NODE_IDS,
        genes,
        rms_cache={innovation: 0.0 for innovation in genes},
    )
    return genome, store


def test_policy_gradient_gathers_trajectory_by_lane_not_population_index():
    # A genome's REINFORCE update depends only on its own weights and its
    # per-lane sampling key (folded from genome id), so it must be identical
    # whether the genome runs alone or at a non-contiguous position in a
    # mixed-topology population. The runner output column is the LANE position
    # in the padded group, not the population index; reading it by population
    # index corrupts every genome whose group position differs from its index.
    #
    # Three base-topology genomes with distinct weights and ids (so distinct
    # trajectories) are interleaved with two hidden-node genomes. The base
    # signature group is then [0, 2, 4], not [0, 1, 2]: under the old bug the
    # genome at population index 2 reads lane 2 (the index-4 genome's real
    # trajectory) instead of its own lane 1.
    weights = [(0.3, -0.4, 0.1), (1.5, 0.0, -0.5), (-2.0, 3.0, 0.7)]
    store = InnovationStore.base()

    solo_weights = []
    for genome_id, (w0, w1, w2) in enumerate(weights):
        solo, _ = base_genome(genome_id, w0, w1, w2)
        policy_gradient_cycle([solo], store, jax.random.PRNGKey(1), 5, 0, 0, 120)
        solo_weights.append(
            [solo.connections[i].weight for i in sorted(solo.connections)]
        )

    bases = [base_genome(genome_id, *weights[genome_id])[0] for genome_id in range(3)]
    hidden_a, _ = base_genome(10, 0.9, 0.9, 0.9)
    hidden_b, _ = base_genome(11, -0.9, 0.4, 0.2)
    add_node(hidden_a, store, random.Random(3))  # distinct structural signature
    add_node(hidden_b, store, random.Random(4))
    mixed = [bases[0], hidden_a, bases[1], hidden_b, bases[2]]
    policy_gradient_cycle(mixed, store, jax.random.PRNGKey(1), 5, 0, 0, 120)

    for genome_id, genome in enumerate(bases):
        mixed_weights = [genome.connections[i].weight for i in sorted(genome.connections)]
        np.testing.assert_allclose(
            mixed_weights, solo_weights[genome_id], rtol=1e-6, atol=1e-7
        )