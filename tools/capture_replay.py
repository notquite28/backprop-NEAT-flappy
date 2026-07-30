"""Capture a saved genome replay as numbered PNG frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import pygame

from neat_flappy.game import FlappyEpisode, WIN_HEIGHT, WIN_WIDTH
from neat_flappy.genome import load_genome
from neat_flappy.training import complexity_adjusted_fitness
from neat_flappy.phenotype import compile_genome


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def draw_frame(
    surface: pygame.Surface,
    episode: FlappyEpisode,
    font: pygame.font.Font,
    generation: int,
    saved_fitness: float,
    replay_fitness: float,
    caption: str,
) -> None:
    surface.blit(episode.assets.background, (0, 0))
    for pipe in episode.pipes:
        pipe.draw(surface)
    episode.base.draw(surface)
    for bird in episode.birds.values():
        bird.draw(surface)
    labels = (
        (caption, 12, 10),
        (f"Score: {episode.score}", 12, 50),
        (f"Checkpoint generation: {generation}", 12, 90),
        (f"Saved fitness: {saved_fitness:.2f}", 12, 130),
        (f"Replay fitness: {replay_fitness:.2f}", 12, 170),
    )
    for text, x, y in labels:
        shadow = font.render(text, True, (0, 0, 0))
        label = font.render(text, True, (255, 255, 255))
        surface.blit(shadow, (x + 2, y + 2))
        surface.blit(label, (x, y))


def capture(
    genome_path: Path,
    output_dir: Path,
    seed: int,
    max_frames: int,
    stride: int,
    caption: str,
) -> int:
    pygame.font.init()
    genome, store, generation = load_genome(genome_path)
    compiled = compile_genome(genome, store)
    weights = jnp.asarray(compiled.weights_for(genome))
    episode = FlappyEpisode(1, seed, max_frames)
    surface = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    font = pygame.font.SysFont("comicsans", 34)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in output_dir.glob("frame_*.png"):
        stale_frame.unlink()
    captured = 0
    replay_return = 0.0
    while not episode.done:
        start = episode.prepare_frame()
        logit = compiled.forward(weights, jnp.asarray(start.observations[0]))
        end = episode.step([bool(logit > 0.0)])
        replay_return += float(end.rewards[0])
        if episode.frame % stride == 0 or episode.done:
            replay_fitness = complexity_adjusted_fitness(
                replay_return, len(genome.connections)
            )
            draw_frame(
                surface,
                episode,
                font,
                generation,
                float(genome.fitness),
                replay_fitness,
                caption,
            )
            pygame.image.save(surface, output_dir / f"frame_{captured:04d}.png")
            captured += 1
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=positive, default=300)
    parser.add_argument("--stride", type=positive, default=2)
    parser.add_argument("--caption", default="Champion replay")
    arguments = parser.parse_args()
    count = capture(
        arguments.genome,
        arguments.output_dir,
        arguments.seed,
        arguments.max_frames,
        arguments.stride,
        arguments.caption,
    )
    print(f"frames={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
