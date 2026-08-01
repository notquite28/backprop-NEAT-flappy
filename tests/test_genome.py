import json
import random

import pytest

from neat_flappy.genome import (
    Activation,
    BASE_NODE_IDS,
    INPUT_COUNT,
    ConnectionGene,
    Genome,
    InnovationStore,
    NodeKind,
    OUTPUT_NODE_ID,
    add_connection,
    add_node,
    crossover,
    initial_population,
    load_genome,
    save_genome,
    validate_genome,
)


def test_initial_topology_is_every_input_and_bias_to_linear_output():
    store = InnovationStore.base()
    genome = initial_population(store, random.Random(0), 1)[0]
    assert genome.node_ids == BASE_NODE_IDS
    assert [
        store.nodes[node_id].kind for node_id in range(OUTPUT_NODE_ID + 1)
    ] == [
        *[NodeKind.INPUT] * INPUT_COUNT,
        NodeKind.BIAS,
        NodeKind.OUTPUT,
    ]
    assert [
        (gene.innovation, gene.src, gene.dst)
        for gene in genome.connections.values()
    ] == [(src, src, OUTPUT_NODE_ID) for src in range(INPUT_COUNT + 1)]


def test_add_node_attempt_stops_when_sampled_gene_is_disabled():
    class FirstChoice:
        @staticmethod
        def choice(values):
            return values[0]

    store = InnovationStore.base()
    genome = initial_population(store, random.Random(0), 1)[0]
    genome.connections[0].enabled = False
    before_nodes = genome.node_ids
    before_connections = set(genome.connections)
    assert not add_node(genome, store, FirstChoice())
    assert genome.node_ids == before_nodes
    assert set(genome.connections) == before_connections


def test_add_node_exact_split_and_fresh_ids():
    store = InnovationStore.base()
    genomes = initial_population(store, random.Random(1), 2)
    old_weight = genomes[0].connections[0].weight
    assert add_node(genomes[0], store, random.Random(2))
    new_id = max(genomes[0].node_ids)
    assert genomes[0].connections[0].enabled is False
    split = [gene for gene in genomes[0].connections.values() if new_id in (gene.src, gene.dst)]
    assert sorted(gene.weight for gene in split) == pytest.approx(sorted([1.0, old_weight]))
    assert add_node(genomes[1], store, random.Random(2))
    assert max(genomes[1].node_ids) != new_id


def test_add_connection_never_closes_cycle():
    store = InnovationStore.base()
    genome = initial_population(store, random.Random(2), 1)[0]
    add_node(genome, store, random.Random(3))
    for seed in range(30):
        add_connection(genome, store, random.Random(seed))
        validate_genome(genome, store)


def test_crossover_disables_cycle_in_innovation_order():
    store = InnovationStore.base()
    five = store.add_node(NodeKind.HIDDEN, Activation.ADD)
    six = store.add_node(NodeKind.HIDDEN, Activation.ADD)
    a = store.connection(0, five)
    b = store.connection(five, six)
    c = store.connection(six, OUTPUT_NODE_ID)
    d = store.connection(0, six)
    e = store.connection(six, five)
    f = store.connection(five, OUTPUT_NODE_ID)
    nodes = frozenset((*BASE_NODE_IDS, five, six))
    left = Genome(1, nodes, {
        a: ConnectionGene(a, 0, five, 1), b: ConnectionGene(b, five, six, 1),
        c: ConnectionGene(c, six, OUTPUT_NODE_ID, 1),
    })
    right = Genome(2, nodes, {
        d: ConnectionGene(d, 0, six, 1), e: ConnectionGene(e, six, five, 1),
        f: ConnectionGene(f, five, OUTPUT_NODE_ID, 1),
    })
    child = crossover(left, right, 3, store, random.Random(0))
    validate_genome(child, store)
    assert not (child.connections[b].enabled and child.connections[e].enabled)


def test_stable_sparse_ids_json_round_trip_and_null_validation(tmp_path):
    store = InnovationStore.base()
    store.add_node(NodeKind.HIDDEN, Activation.TANH, expected_id=9)
    innovation = 12
    store.import_connection(innovation, 0, 9)
    store.import_connection(13, 9, OUTPUT_NODE_ID)
    genome = Genome(7, frozenset((*BASE_NODE_IDS, 9)), {
        innovation: ConnectionGene(innovation, 0, 9, 0.5),
        13: ConnectionGene(13, 9, OUTPUT_NODE_ID, -0.25),
    }, 4.5, 2.0)
    path = tmp_path / "genome.json"
    save_genome(path, genome, store, 3)
    loaded, loaded_store, generation = load_genome(path)
    assert generation == 3
    assert loaded.node_ids == genome.node_ids
    assert loaded_store.connection_endpoints[12] == (0, 9)

    record = json.loads(path.read_text())
    next(node for node in record["nodes"] if node["id"] == 9)["activation"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hidden"):
        load_genome(path)
