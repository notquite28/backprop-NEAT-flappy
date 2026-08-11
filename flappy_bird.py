"""Command-line entry point for JAX NEAT Flappy Bird."""
from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, replay, or visualize a JAX NEAT Flappy Bird policy"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train", help="train without a Pygame window")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--generations", type=positive_int, default=50)
    train.add_argument("--minimum-generations", type=positive_int, default=10)
    train.add_argument("--population-size", type=positive_int, default=100)
    train.add_argument("--cluster-count", type=positive_int, default=5)
    train.add_argument("--max-frames", type=positive_int, default=1000)
    train.add_argument("--backprop-cycles", type=positive_int, default=4)
    train.add_argument("--eval-episodes", type=positive_int, default=3)
    train.add_argument("--fitness-threshold", type=float, default=100.0)
    train.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    train.add_argument(
        "--eval-seeds",
        type=int,
        nargs="*",
        default=None,
        help="fixed selection/rollback layouts; overrides derived seed+9000+i",
    )
    train.add_argument(
        "--pg-seed",
        type=int,
        default=None,
        help="pin policy-gradient rollouts to one layout",
    )

    replay = subcommands.add_parser("replay", help="replay a saved genome")
    replay.add_argument("--genome", type=Path, required=True)
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--max-frames", type=positive_int, default=1000)

    visualize = subcommands.add_parser(
        "visualize", help="compare starting, global-best, and saved graphs"
    )
    visualize.add_argument("--genome", type=Path, required=True)
    visualize.add_argument("--best-genome", type=Path)
    visualize.add_argument("--output", type=Path, default=Path("checkpoints/genomes.svg"))
    visualize.add_argument("--seed", type=int, default=0)
    return parser


def run_train(arguments: argparse.Namespace) -> int:
    if arguments.population_size % arguments.cluster_count:
        raise SystemExit("population-size must be divisible by cluster-count")
    if arguments.cluster_count != 5:
        raise SystemExit("Hardmaru mode requires --cluster-count 5")
    from neat_flappy.training import TrainingConfig, train

    config = TrainingConfig(
        seed=arguments.seed,
        generations=arguments.generations,
        minimum_generations=arguments.minimum_generations,
        population_size=arguments.population_size,
        cluster_count=arguments.cluster_count,
        max_frames=arguments.max_frames,
        backprop_cycles=arguments.backprop_cycles,
        eval_episodes=arguments.eval_episodes,
        fitness_threshold=arguments.fitness_threshold,
        eval_seeds=tuple(arguments.eval_seeds) if arguments.eval_seeds else None,
        pg_seed=arguments.pg_seed,
        checkpoint_dir=arguments.checkpoint_dir,
    )
    train(config)
    return 0


def run_replay(arguments: argparse.Namespace) -> int:
    import jax.numpy as jnp

    from neat_flappy.game import FlappyEpisode, Renderer
    from neat_flappy.genome import load_genome
    from neat_flappy.phenotype import compile_genome

    genome, store, generation = load_genome(arguments.genome)
    compiled = compile_genome(genome, store)
    weights = jnp.asarray(compiled.weights_for(genome))
    episode = FlappyEpisode(1, arguments.seed, arguments.max_frames)
    renderer = Renderer(generation, float(genome.fitness))
    try:
        while not episode.done and renderer.poll_open():
            start = episode.prepare_frame()
            logit = compiled.forward(weights, jnp.asarray(start.observations[0]))
            episode.step([bool(logit > 0.0)])
            renderer.draw(episode)
    finally:
        score = episode.score
        renderer.close()
    print(f"terminal_score={score}")
    return 0


def run_visualize(arguments: argparse.Namespace) -> int:
    from neat_flappy.visualization import render_comparison

    output = render_comparison(
        arguments.genome,
        arguments.output,
        arguments.seed,
        arguments.best_genome,
    )
    print(f"graph={output}")
    return 0


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        return run_train(arguments)
    if arguments.command == "replay":
        return run_replay(arguments)
    return run_visualize(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
