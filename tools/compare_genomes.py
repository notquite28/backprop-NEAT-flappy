"""Compare two saved genomes on identical long-horizon pipe sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from neat_flappy.game import FlappyEpisode
from neat_flappy.genome import Genome, InnovationStore, load_genome
from neat_flappy.phenotype import batch_forward
from neat_flappy.training import complexity_adjusted_fitness


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def compare(
    genome_paths: tuple[Path, Path],
    labels: tuple[str, str],
    seeds: list[int],
    max_score: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    loaded = [load_genome(path) for path in genome_paths]
    genomes: list[Genome] = [record[0] for record in loaded]
    stores: list[InnovationStore] = [record[1] for record in loaded]
    generations = [record[2] for record in loaded]
    results: list[dict[str, Any]] = []

    for seed in seeds:
        episode = FlappyEpisode(
            len(genomes), seed, max_frames=max_frames, max_score=max_score
        )
        returns = np.zeros(len(genomes), dtype=np.float64)
        scores = np.zeros(len(genomes), dtype=np.int64)
        frames = np.zeros(len(genomes), dtype=np.int64)
        termination = ["death"] * len(genomes)

        while not episode.done:
            start = episode.prepare_frame()
            live_genomes = [genomes[index] for index in start.bird_ids]
            live_stores = [stores[index] for index in start.bird_ids]
            logits = batch_forward(live_genomes, live_stores, start.observations)
            end = episode.step(logits > 0.0)
            for row, genome_index in enumerate(end.bird_ids):
                reward = float(end.rewards[row])
                returns[genome_index] += reward
                frames[genome_index] += 1
                if reward >= 4.0:
                    scores[genome_index] += 1
                if end.terminal[row]:
                    if scores[genome_index] >= max_score:
                        termination[genome_index] = "score_cap"
                    elif episode.frame >= max_frames:
                        termination[genome_index] = "frame_cap"

        for index, genome in enumerate(genomes):
            results.append(
                {
                    "label": labels[index],
                    "seed": seed,
                    "checkpoint_generation": generations[index],
                    "saved_fitness": float(genome.fitness),
                    "score": int(scores[index]),
                    "frames": int(frames[index]),
                    "return": float(returns[index]),
                    "replay_fitness": complexity_adjusted_fitness(
                        float(returns[index]), len(genome.connections)
                    ),
                    "termination": termination[index],
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-label", default="generation-9")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-score", type=positive, default=1000)
    parser.add_argument("--max-frames", type=positive, default=100_000)
    arguments = parser.parse_args()
    results = compare(
        (arguments.best, arguments.candidate),
        ("global-best", arguments.candidate_label),
        arguments.seeds,
        arguments.max_score,
        arguments.max_frames,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
