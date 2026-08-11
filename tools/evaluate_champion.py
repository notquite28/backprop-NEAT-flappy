"""Evaluate one saved genome on an inclusive range of held-out JAX schedules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neat_flappy.genome import load_genome
from neat_flappy.phenotype import compile_genome
from neat_flappy.schedule import pipe_count_for, pipe_schedule
from neat_flappy.vectorized import get_runner


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _summarize(
    pipes_cleared: Sequence[int], survived_cap: Sequence[bool]
) -> dict[str, float | int]:
    pipes = np.asarray(pipes_cleared, dtype=np.int64)
    return {
        "mean_pipes": float(np.mean(pipes)),
        "median_pipes": float(np.median(pipes)),
        "p10_pipes": float(np.percentile(pipes, 10, method="linear")),
        "p90_pipes": float(np.percentile(pipes, 90, method="linear")),
        "minimum_pipes": int(np.min(pipes)),
        "cap_survival_rate": float(np.mean(np.asarray(survived_cap, dtype=np.bool_))),
    }


def evaluate_champion(
    genome_path: Path, seed_start: int, seed_stop: int, max_frames: int
) -> dict[str, Any]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if seed_stop < seed_start:
        raise ValueError("seed_stop must be greater than or equal to seed_start")

    genome, store, generation = load_genome(genome_path)
    compiled = compile_genome(genome, store)
    weights = jnp.asarray(compiled.weights_for(genome))[None]
    runner = get_runner(compiled.signature, compiled.raw_forward, max_frames, sample=False)
    genome_ids = jnp.asarray([genome.id], dtype=jnp.int32)
    root_key = jax.random.PRNGKey(0)

    results: list[dict[str, Any]] = []
    pipes_cleared: list[int] = []
    survived_cap: list[bool] = []
    for seed in range(seed_start, seed_stop + 1):
        heights, gaps = pipe_schedule(seed, pipe_count_for(max_frames))
        episode = runner(
            weights,
            jnp.asarray(heights),
            jnp.asarray(gaps),
            root_key,
            0,
            0,
            genome_ids,
        )
        pipes = int(episode["scores"][0])
        frames = int(episode["frames"][0])
        survived = bool(episode["survived"][0])
        pipes_cleared.append(pipes)
        survived_cap.append(survived)
        results.append(
            {
                "seed": seed,
                "pipes_cleared": pipes,
                "death_frame": None if survived else frames,
                "survived_cap": survived,
            }
        )

    return {
        "checkpoint": str(genome_path),
        "checkpoint_generation": generation,
        "engine": "jax_aabb",
        "seed_start": seed_start,
        "seed_stop": seed_stop,
        "seed_count": seed_stop - seed_start + 1,
        "max_frames": max_frames,
        "results": results,
        "summary": _summarize(pipes_cleared, survived_cap),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--max-frames", type=positive, required=True)
    arguments = parser.parse_args()
    if arguments.seed_stop < arguments.seed_start:
        parser.error("seed_stop must be greater than or equal to seed_start")
    report = evaluate_champion(
        arguments.genome,
        arguments.seed_start,
        arguments.seed_stop,
        arguments.max_frames,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
