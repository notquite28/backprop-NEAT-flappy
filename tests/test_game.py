import random

import numpy as np
import pytest

from neat_flappy.game import Bird, FlappyEpisode, FLOOR, Pipe, load_assets


def test_jump_move_and_animation_are_state_updates():
    bird = Bird(230, 350)
    bird.jump()
    assert (bird.vel, bird.tick_count, bird.height) == (-10.5, 0, 350)
    bird.move()
    assert bird.y == pytest.approx(339.0)
    first = bird.img
    for _ in range(5):
        bird.move()
    assert bird.img is not first


def test_seeded_pipe_sequence_is_deterministic():
    left = FlappyEpisode(1, 17, 10)
    right = FlappyEpisode(1, 17, 10)
    assert left.pipes[0].height == right.pipes[0].height
    for _ in range(3):
        l_start = left.prepare_frame()
        r_start = right.prepare_frame()
        np.testing.assert_array_equal(l_start.observations, r_start.observations)
        left.step([False])
        right.step([False])


def test_pass_reward_and_terminal_floor_reward_order():
    episode = FlappyEpisode(1, 3, 10)
    pipe = episode.pipes[0]
    pipe.height = 150
    pipe.top = pipe.height - pipe.assets.pipe_top.get_height()
    pipe.bottom = pipe.height + pipe.GAP
    pipe.x = 234
    episode.birds[0].y = 250
    start = episode.prepare_frame()
    assert start.observations.dtype == np.float32
    assert start.observations.shape == (1, 2)
    np.testing.assert_allclose(
        start.observations[0],
        np.asarray([-1.5 / 800.0, 1.5 / 16.0], dtype=np.float32),
    )
    end = episode.step([False])
    assert end.rewards[0] == pytest.approx(5.1)
    assert end.score == 1

    terminal = FlappyEpisode(1, 4, 10)
    terminal.birds[0].y = FLOOR
    terminal.prepare_frame()
    final = terminal.step([False])
    assert final.rewards[0] == pytest.approx(0.1)
    assert final.terminal[0]
    assert final.done

def test_score_cap_terminates_surviving_birds():
    episode = FlappyEpisode(1, 3, max_frames=100, max_score=1)
    pipe = episode.pipes[0]
    pipe.height = 150
    pipe.top = pipe.height - pipe.assets.pipe_top.get_height()
    pipe.bottom = pipe.height + pipe.GAP
    pipe.x = 234
    episode.birds[0].y = 250

    episode.prepare_frame()
    end = episode.step([False])

    assert end.score == 1
    assert end.terminal[0]
    assert end.done
    assert not episode.birds

    with pytest.raises(ValueError, match="max_score"):
        FlappyEpisode(1, 0, max_score=0)


def test_pixel_mask_collision():
    bird = Bird(230, 350)
    pipe = Pipe(230, random.Random(0), load_assets())
    pipe.bottom = 350
    assert pipe.collide(bird)
