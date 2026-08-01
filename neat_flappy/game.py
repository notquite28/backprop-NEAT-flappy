"""Original Flappy Bird mechanics and a deterministic episode state machine."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pygame

from .schedule import OBSERVATION_COUNT, pipe_count_for, pipe_schedule

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
ASSET_DIR = Path(__file__).parent.parent / "imgs"


@dataclass(frozen=True)
class Assets:
    pipe: pygame.Surface
    pipe_top: pygame.Surface
    background: pygame.Surface
    birds: tuple[pygame.Surface, pygame.Surface, pygame.Surface]
    base: pygame.Surface


@lru_cache(maxsize=1)
def load_assets() -> Assets:
    """Load image surfaces without creating a window."""
    pipe = pygame.transform.scale2x(pygame.image.load(ASSET_DIR / "pipe.png"))
    background = pygame.transform.scale(
        pygame.image.load(ASSET_DIR / "bg.png"), (600, 900)
    )
    birds = tuple(
        pygame.transform.scale2x(pygame.image.load(ASSET_DIR / f"bird{i}.png"))
        for i in range(1, 4)
    )
    base = pygame.transform.scale2x(pygame.image.load(ASSET_DIR / "base.png"))
    return Assets(pipe, pygame.transform.flip(pipe, False, True), background, birds, base)  # type: ignore[arg-type]


class Bird:
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x: int, y: float, assets: Assets | None = None) -> None:
        self.assets = assets or load_assets()
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0.0
        self.last_displacement = 0.0
        self.height = self.y
        self.img_count = 0
        self.img = self.assets.birds[0]

    def jump(self) -> None:
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self) -> None:
        self.tick_count += 1
        displacement = self.vel * self.tick_count + 0.5 * 3 * self.tick_count**2
        if displacement >= 16:
            displacement = 16
        if displacement < 0:
            displacement -= 2
        self.last_displacement = displacement
        self.y += displacement
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        elif self.tilt > -90:
            self.tilt -= self.ROT_VEL
        self._advance_animation()

    def _advance_animation(self) -> None:
        self.img_count += 1
        t = self.ANIMATION_TIME
        if self.img_count <= t:
            self.img = self.assets.birds[0]
        elif self.img_count <= t * 2:
            self.img = self.assets.birds[1]
        elif self.img_count <= t * 3:
            self.img = self.assets.birds[2]
        elif self.img_count <= t * 4:
            self.img = self.assets.birds[1]
        elif self.img_count == t * 4 + 1:
            self.img = self.assets.birds[0]
            self.img_count = 0
        if self.tilt <= -80:
            self.img = self.assets.birds[1]
            self.img_count = t * 2

    def draw(self, surface: pygame.Surface) -> None:
        rotated = pygame.transform.rotate(self.img, self.tilt)
        rect = rotated.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        surface.blit(rotated, rect.topleft)

    def get_mask(self) -> pygame.mask.Mask:
        return pygame.mask.from_surface(self.img)


class Pipe:
    VEL = 5

    def __init__(
        self,
        x: int,
        height: float,
        gap: float,
        index: int,
        assets: Assets | None = None,
    ) -> None:
        self.assets = assets or load_assets()
        self.x = x
        self.index = index
        self.height = float(height)
        self.gap = float(gap)
        self.top = self.height - self.assets.pipe_top.get_height()
        self.bottom = self.height + self.gap
        self.passed = False
        self.top_mask = pygame.mask.from_surface(self.assets.pipe_top)
        self.bottom_mask = pygame.mask.from_surface(self.assets.pipe)

    @property
    def width(self) -> int:
        return self.assets.pipe_top.get_width()

    def move(self) -> None:
        self.x -= self.VEL

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.assets.pipe_top, (self.x, self.top))
        surface.blit(self.assets.pipe, (self.x, self.bottom))

    def collide(self, bird: Bird) -> bool:
        bird_mask = bird.get_mask()
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        return bool(
            bird_mask.overlap(self.bottom_mask, bottom_offset)
            or bird_mask.overlap(self.top_mask, top_offset)
        )


class Base:
    VEL = 5

    def __init__(self, y: int, assets: Assets | None = None) -> None:
        self.assets = assets or load_assets()
        self.y = y
        self.width = self.assets.base.get_width()
        self.x1 = 0
        self.x2 = self.width

    def move(self) -> None:
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.assets.base, (self.x1, self.y))
        surface.blit(self.assets.base, (self.x2, self.y))


@dataclass(frozen=True)
class FrameStart:
    bird_ids: tuple[int, ...]
    observations: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True)
class FrameEnd:
    bird_ids: tuple[int, ...]
    rewards: np.ndarray
    terminal: np.ndarray
    score: int
    done: bool


class FlappyEpisode:
    """One deterministic shared-world episode for one or more birds."""

    def __init__(
        self,
        bird_count: int,
        seed: int,
        max_frames: int = 1000,
        max_score: int | None = None,
    ) -> None:
        if bird_count <= 0 or max_frames <= 0:
            raise ValueError("bird_count and max_frames must be positive")
        if max_score is not None and max_score <= 0:
            raise ValueError("max_score must be positive")
        self.assets = load_assets()
        self.max_frames = max_frames
        self.max_score = max_score
        count = pipe_count_for(max_frames if max_score is None else max_frames)
        self.heights, self.gaps = pipe_schedule(seed, count)
        self.birds = {i: Bird(230, 350, self.assets) for i in range(bird_count)}
        self.base = Base(FLOOR, self.assets)
        self.pipes = [Pipe(700, self.heights[0], self.gaps[0], 0, self.assets)]
        self.spawn_index = 1
        self.frame = 0
        self.score = 0
        self._prepared: FrameStart | None = None

    @property
    def done(self) -> bool:
        return (
            not self.birds
            or self.frame >= self.max_frames
            or (self.max_score is not None and self.score >= self.max_score)
        )

    def _pipe_index(self) -> int:
        first_bird = next(iter(self.birds.values()))
        if len(self.pipes) > 1 and first_bird.x > self.pipes[0].x + self.pipes[0].width:
            return 1
        return 0

    def prepare_frame(self) -> FrameStart:
        """Apply survival reward and movement, then expose policy observations."""
        if self.done:
            raise RuntimeError("episode is terminal")
        if self._prepared is not None:
            raise RuntimeError("frame is already prepared")
        ids = tuple(self.birds)
        pipe = self.pipes[self._pipe_index()]
        gap_center = pipe.height + pipe.gap / 2.0
        following = min(pipe.index + 1, len(self.heights) - 1)
        next_center = float(self.heights[following] + self.gaps[following] / 2.0)
        observations = np.empty((len(ids), OBSERVATION_COUNT), dtype=np.float32)
        for row, bird_id in enumerate(ids):
            bird = self.birds[bird_id]
            bird.move()
            observations[row] = (
                (gap_center - bird.y) / WIN_HEIGHT,
                bird.last_displacement / 16.0,
                pipe.gap / WIN_HEIGHT,
                (next_center - bird.y) / WIN_HEIGHT,
                (pipe.x - bird.x) / WIN_WIDTH,
            )
        self._prepared = FrameStart(ids, observations, np.full(len(ids), 0.1, np.float32))
        return self._prepared

    def step(self, actions: Sequence[bool]) -> FrameEnd:
        """Apply one batch of actions and finish the prepared frame."""
        start = self._prepared
        if start is None:
            raise RuntimeError("prepare_frame must be called before step")
        if len(actions) != len(start.bird_ids):
            raise ValueError("one action is required for each live bird")
        rewards = start.rewards.copy()
        for bird_id, action in zip(start.bird_ids, actions, strict=True):
            if action:
                self.birds[bird_id].jump()

        self.base.move()
        for pipe in self.pipes:
            pipe.move()

        dead: set[int] = set()
        for pipe in self.pipes:
            for row, bird_id in enumerate(start.bird_ids):
                if bird_id not in dead and pipe.collide(self.birds[bird_id]):
                    rewards[row] -= 1.0
                    dead.add(bird_id)
        for bird_id in dead:
            del self.birds[bird_id]

        add_pipe = False
        for pipe in self.pipes:
            if not pipe.passed and self.birds:
                bird = next(iter(self.birds.values()))
                if pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
        if add_pipe:
            self.score += 1
            for row, bird_id in enumerate(start.bird_ids):
                if bird_id in self.birds:
                    rewards[row] += 5.0
            index = min(self.spawn_index, len(self.heights) - 1)
            self.pipes.append(
                Pipe(WIN_WIDTH, self.heights[index], self.gaps[index], index, self.assets)
            )
            self.spawn_index += 1

        self.pipes = [pipe for pipe in self.pipes if pipe.x + pipe.width >= 0]
        for bird_id, bird in list(self.birds.items()):
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                del self.birds[bird_id]
                dead.add(bird_id)

        self.frame += 1
        if (
            self.frame >= self.max_frames
            or (self.max_score is not None and self.score >= self.max_score)
        ):
            dead.update(self.birds)
            self.birds.clear()
        terminal = np.asarray([bird_id in dead for bird_id in start.bird_ids], dtype=np.bool_)
        result = FrameEnd(start.bird_ids, rewards, terminal, self.score, self.done)
        self._prepared = None
        return result


def run_controller_episode(
    controller: Callable[[np.ndarray, tuple[int, ...], int], Sequence[bool]],
    bird_count: int,
    seed: int,
    max_frames: int = 1000,
) -> tuple[np.ndarray, int, int]:
    """Run the shared state machine using only a batch-action controller contract."""
    episode = FlappyEpisode(bird_count, seed, max_frames)
    returns = np.zeros(bird_count, dtype=np.float32)
    while not episode.done:
        start = episode.prepare_frame()
        end = episode.step(controller(start.observations, start.bird_ids, episode.frame))
        for bird_id, reward in zip(end.bird_ids, end.rewards, strict=True):
            returns[bird_id] += reward
    return returns, episode.score, episode.frame


class Renderer:
    """Lazy 600x800 Pygame renderer for an existing episode."""

    def __init__(self, generation: int, fitness: float) -> None:
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.font = pygame.font.SysFont("comicsans", 42)
        self.clock = pygame.time.Clock()
        self.generation = generation
        self.fitness = fitness

    def poll_open(self) -> bool:
        return not any(event.type == pygame.QUIT for event in pygame.event.get())

    def draw(self, episode: FlappyEpisode) -> None:
        self.clock.tick(30)
        self.window.blit(episode.assets.background, (0, 0))
        for pipe in episode.pipes:
            pipe.draw(self.window)
        episode.base.draw(self.window)
        for bird in episode.birds.values():
            bird.draw(self.window)
        labels = (
            (f"Score: {episode.score}", WIN_WIDTH - 180, 10),
            (f"Gen: {self.generation}", 10, 10),
            (f"Saved fitness: {self.fitness:.2f}", 10, 55),
        )
        for text, x, y in labels:
            self.window.blit(self.font.render(text, True, (255, 255, 255)), (x, y))
        pygame.display.update()

    def close(self) -> None:
        pygame.quit()
