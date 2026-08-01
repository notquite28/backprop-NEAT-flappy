import numpy as np
import pytest

from neat_flappy.game import Bird, FlappyEpisode, FLOOR, Pipe, load_assets
from neat_flappy.schedule import OBSERVATION_COUNT


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
    pipe.gap = 200
    pipe.top = pipe.height - pipe.assets.pipe_top.get_height()
    pipe.bottom = pipe.height + pipe.gap
    pipe.x = 234
    episode.birds[0].y = 250
    start = episode.prepare_frame()
    assert start.observations.dtype == np.float32
    assert start.observations.shape == (1, OBSERVATION_COUNT)
    following = episode.heights[1] + episode.gaps[1] / 2.0
    np.testing.assert_allclose(
        start.observations[0],
        np.asarray(
            [
                -1.5 / 800.0,
                1.5 / 16.0,
                200.0 / 800.0,
                (following - 251.5) / 800.0,
                (234 - 230) / 600.0,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
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
    pipe.gap = 200
    pipe.top = pipe.height - pipe.assets.pipe_top.get_height()
    pipe.bottom = pipe.height + pipe.gap
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
    pipe = Pipe(230, 150, 200, 0, load_assets())
    pipe.bottom = 350
    assert pipe.collide(bird)


def test_gap_size_varies_and_centers_alternate_bands():
    episode = FlappyEpisode(1, 5, 400)
    assert len(set(episode.gaps.tolist())) > 1
    centers = episode.heights + episode.gaps / 2.0
    even = centers[0:20:2]
    odd = centers[1:20:2]
    assert even.max() < odd.min()


def test_observation_exposes_gap_size_and_next_gap_center():
    episode = FlappyEpisode(1, 5, 400)
    pipe = episode.pipes[0]
    observations = episode.prepare_frame().observations[0]
    bird = episode.birds[0]
    following = episode.heights[1] + episode.gaps[1] / 2.0
    assert observations[2] == np.float32(pipe.gap / 800.0)
    assert observations[3] == np.float32((following - bird.y) / 800.0)
