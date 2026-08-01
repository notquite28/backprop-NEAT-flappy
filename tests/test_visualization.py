import random

from neat_flappy.genome import (
    INPUT_COUNT,
    InnovationStore,
    add_node,
    initial_population,
    save_genome,
)
from neat_flappy.visualization import render_comparison


def test_render_comparison_contains_start_and_champion_graphs(tmp_path):
    store = InnovationStore.base()
    champion = initial_population(store, random.Random(9), 1)[0]
    champion.fitness = 12.5
    champion.score = 2.0
    genome_path = tmp_path / "best.json"
    output_path = tmp_path / "graphs.svg"
    save_genome(genome_path, champion, store, generation=4)

    assert render_comparison(genome_path, output_path, seed=3) == output_path
    svg = output_path.read_text()
    assert "Starting genome · seed 3 · genome 0" in svg
    assert "Saved genome · generation 4 · fitness 12.500 · score 2.000" in svg
    base_nodes = INPUT_COUNT + 2
    base_connections = INPUT_COUNT + 1
    assert svg.count(
        f"{base_nodes} nodes · {base_connections} enabled "
        f"/ {base_connections} total connections"
    ) == 2
    assert svg.count("gap Δy") == 2
    assert svg.count("velocity") == 2
    assert svg.count("i0 ·") == 2
    assert "#2E7D32" in svg


def test_render_comparison_shows_hidden_activation_and_disabled_split(tmp_path):
    store = InnovationStore.base()
    champion = initial_population(store, random.Random(1), 1)[0]
    best = champion.copy()
    best.fitness = 20.0
    best.score = 3.0
    assert add_node(champion, store, random.Random(2))
    genome_path = tmp_path / "topology.json"
    best_path = tmp_path / "best.json"
    output_path = tmp_path / "topology.svg"
    save_genome(genome_path, champion, store, generation=2)

    save_genome(best_path, best, store, generation=0)
    render_comparison(genome_path, output_path, best_path=best_path)
    svg = output_path.read_text()
    assert 'width="2100"' in svg
    assert "Global best · generation 0 · fitness 20.000 · score 3.000" in svg
    hidden_id = max(champion.node_ids)
    activation = store.nodes[hidden_id].activation.value
    assert f"#{hidden_id}" in svg
    assert activation in svg
    assert (
        f"{INPUT_COUNT + 3} nodes · {INPUT_COUNT + 2} enabled "
        f"/ {INPUT_COUNT + 3} total connections"
    ) in svg
    assert 'stroke-dasharray="7 5"' in svg
    assert "arrow-disabled" in svg
