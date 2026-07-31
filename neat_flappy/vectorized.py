"""Pure-JAX vectorized episode engine.

Runs a whole population's episode as one compiled ``lax.scan`` over frames
with ``vmap`` over birds, replacing the Python per-frame dispatch loop used
during training. The pipe schedule is exogenous (pipes spawn deterministically
from the seed and move at constant speed regardless of birds), so every bird's
trajectory is independent given the seed and the population vectorizes.

Collision divergence note: this engine uses axis-aligned bounding-box (AABB)
collision against the bird's full 68x48 image rect, while the pygame engine in
``game.py`` uses pixel-perfect sprite masks and rounds the bird y-position to
an integer pixel. AABB is marginally more conservative; a genome trained here
may replay slightly differently under pygame. Replay, rendering, and the
tools keep the authoritative pygame engine; this module never imports pygame.
"""
from __future__ import annotations

import random
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

BIRD_X = 230.0
BIRD_WIDTH = 68.0
BIRD_HEIGHT = 48.0
PIPE_WIDTH = 104.0
PIPE_SPRITE_HEIGHT = 640.0
PIPE_GAP = 200.0
PIPE_VEL = 5.0
SPAWN_X = 600.0
FIRST_PIPE_X = 700.0
START_Y = 350.0
FLOOR_DEATH_Y = 692.0  # y + BIRD_HEIGHT - 10 >= FLOOR (730)
CEILING_DEATH_Y = -50.0
WIN_HEIGHT = 800.0
MAX_PIPES = 8
JUMP_VEL = -10.5
SURVIVAL_REWARD = 0.1
COLLISION_PENALTY = 1.0
PASS_REWARD = 5.0


def pipe_heights(seed: int, count: int) -> np.ndarray:
    """Precompute the pipe height sequence a pygame episode would draw.

    ``FlappyEpisode`` draws each pipe height as ``randrange(50, 450)`` from
    ``random.Random(seed)`` and uses the RNG for nothing else, so the whole
    schedule is reproducible here. ``count = max_frames // 40 + 16`` is a
    generous upper bound on pipes spawned within ``max_frames`` (a pipe is
    passed roughly every 74 frames); if a run ever needs more, the engine
    clamps to the last precomputed height, which only affects frames beyond
    any realistic horizon.
    """
    rng = random.Random(seed)
    return np.asarray([rng.randrange(50, 450) for _ in range(count)], dtype=np.float32)


def run_episodes(
    weights: jax.Array,
    forward: Callable[[jax.Array, jax.Array], jax.Array],
    heights: jax.Array,
    max_frames: int,
    sample: bool,
    root_key: jax.Array,
    generation: int | jax.Array,
    cycle: int | jax.Array,
    genome_ids: jax.Array,
) -> dict[str, jax.Array]:
    """Simulate one bird per row of ``weights`` through ``max_frames`` frames.

    ``forward(weights_row, obs) -> logit`` is a per-bird compiled network
    (``CompiledPhenotype.raw_forward``), vmapped over the population inside
    the scan. Padded lanes are allowed: callers discard rows beyond their
    real group size.

    Returns a dict with:
      returns: (N,) float32 total reward per bird.
      scores:  (N,) int32 pipes passed per bird.
      frames:  (N,) int32 frames the bird was alive at frame start.
      obs:     (max_frames, N, 2) float32 observations.
      actions: (max_frames, N) bool, False for dead lanes.
      rewards: (max_frames, N) float32 per-frame reward.
      alive:   (max_frames, N) bool, lane alive at that frame's start.
    """
    n = weights.shape[0]
    zeros = jnp.zeros(n, jnp.float32)
    bird = {
        "y": jnp.full(n, START_Y, jnp.float32),
        "vel": zeros,
        "tick": zeros,
        "last_disp": zeros,
        "alive": jnp.ones(n, jnp.bool_),
        "returns": zeros,
        "frames": jnp.zeros(n, jnp.int32),
        "scores": jnp.zeros(n, jnp.int32),
    }
    world = {
        "px": jnp.zeros(MAX_PIPES, jnp.float32).at[0].set(FIRST_PIPE_X),
        "ph": jnp.zeros(MAX_PIPES, jnp.float32).at[0].set(heights[0]),
        "passed": jnp.zeros(MAX_PIPES, jnp.bool_),
        "active": jnp.zeros(MAX_PIPES, jnp.bool_).at[0].set(True),
        "spawn_index": jnp.asarray(1, jnp.int32),
        "score": jnp.asarray(0, jnp.int32),
    }
    cycle_key = jax.random.fold_in(jax.random.fold_in(root_key, generation), cycle)

    def step(carry: tuple[dict, dict], t: jax.Array) -> tuple[tuple[dict, dict], tuple]:
        bird, world = carry
        alive = bird["alive"]

        # 1. Integrate living birds (Bird.move); dead lanes freeze.
        tick = bird["tick"] + 1.0
        disp = bird["vel"] * tick + 1.5 * tick * tick
        disp = jnp.minimum(disp, 16.0)
        disp = jnp.where(disp < 0.0, disp - 2.0, disp)
        y = bird["y"] + jnp.where(alive, disp, 0.0)
        last_disp = jnp.where(alive, disp, bird["last_disp"])
        tick = jnp.where(alive, tick, bird["tick"])

        # 2. Target pipe: smallest px with px + width >= bird x, else smallest px.
        px = world["px"]
        ph = world["ph"]
        active = world["active"]
        qualifies = active & (px + PIPE_WIDTH >= BIRD_X)
        candidate = jnp.where(qualifies, px, jnp.inf)
        target = jnp.argmin(candidate)
        fallback = jnp.argmin(jnp.where(active, px, jnp.inf))
        target = jnp.where(jnp.isinf(candidate[target]), fallback, target)
        gap_center = ph[target] + PIPE_GAP / 2.0

        # 3. Observation.
        obs = jnp.stack(
            ((gap_center - y) / WIN_HEIGHT, last_disp / 16.0), axis=-1
        )

        # 4. Survival reward.
        reward = SURVIVAL_REWARD * alive.astype(jnp.float32)

        # 5. Action: deterministic threshold or per-lane keyed sampling.
        logit = jax.vmap(forward)(weights, obs)
        if sample:
            def lane_key(genome_id: jax.Array) -> jax.Array:
                genome_key = jax.random.fold_in(cycle_key, genome_id)
                return jax.random.fold_in(genome_key, t)

            action = jax.vmap(jax.random.bernoulli)(
                jax.vmap(lane_key)(genome_ids), jax.nn.sigmoid(logit)
            )
        else:
            action = logit > 0.0

        # 6. Jump.
        jump = action & alive
        vel = jnp.where(jump, JUMP_VEL, bird["vel"])
        tick = jnp.where(jump, 0.0, tick)

        # 7. Move pipes.
        px = jnp.where(active, px - PIPE_VEL, px)

        # 8. AABB collision against every active pipe, top and bottom rects.
        px_p = px[:, None]
        ph_p = ph[:, None]
        x_overlap = active[:, None] & (BIRD_X < px_p + PIPE_WIDTH) & (
            BIRD_X + BIRD_WIDTH > px_p
        )
        top_hit = x_overlap & (y < ph_p) & (y + BIRD_HEIGHT > ph_p - PIPE_SPRITE_HEIGHT)
        bottom_hit = x_overlap & (y < ph_p + PIPE_SPRITE_HEIGHT + PIPE_GAP) & (
            y + BIRD_HEIGHT > ph_p + PIPE_GAP
        )
        collided = alive & jnp.any(top_hit | bottom_hit, axis=0)
        reward = reward - collided.astype(jnp.float32)
        alive_post = alive & ~collided

        # 9. Scoring: pipes whose left edge crossed the bird; +5 for survivors.
        newly_passed = active & ~world["passed"] & (px < BIRD_X)
        did_pass = jnp.any(newly_passed)
        passed = world["passed"] | newly_passed
        score = world["score"] + did_pass.astype(jnp.int32)
        reward = reward + PASS_REWARD * (did_pass & alive_post).astype(jnp.float32)

        # 10. Spawn one pipe at SPAWN_X when any pipe was passed this frame.
        free = ~active
        has_free = jnp.any(free)
        slot = jnp.argmin(active.astype(jnp.int32))
        do_spawn = did_pass & has_free
        height = heights[jnp.minimum(world["spawn_index"], heights.shape[0] - 1)]
        px = jnp.where(do_spawn, px.at[slot].set(SPAWN_X), px)
        ph = jnp.where(do_spawn, ph.at[slot].set(height), ph)
        passed = jnp.where(do_spawn, passed.at[slot].set(False), passed)
        active = jnp.where(do_spawn, active.at[slot].set(True), active)
        spawn_index = world["spawn_index"] + do_spawn.astype(jnp.int32)

        # 11. Remove off-screen pipes.
        active = active & (px + PIPE_WIDTH >= 0.0)

        # 12. Floor and ceiling death.
        died = alive_post & ((y >= FLOOR_DEATH_Y) | (y < CEILING_DEATH_Y))
        alive_next = alive_post & ~died

        # 13. Accumulate per-lane statistics.
        bird = {
            "y": y,
            "vel": vel,
            "tick": tick,
            "last_disp": last_disp,
            "alive": alive_next,
            "returns": bird["returns"] + reward,
            "frames": bird["frames"] + alive.astype(jnp.int32),
            "scores": jnp.where(alive_post, score, bird["scores"]),
        }
        world = {
            "px": px,
            "ph": ph,
            "passed": passed,
            "active": active,
            "spawn_index": spawn_index,
            "score": score,
        }
        return (bird, world), (obs, action & alive, reward, alive)

    (bird, _), (obs, actions, rewards, alive) = jax.lax.scan(
        step, (bird, world), jnp.arange(max_frames)
    )
    return {
        "returns": bird["returns"],
        "scores": bird["scores"],
        "frames": bird["frames"],
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "alive": alive,
    }


_RUNNER_CACHE: dict[tuple, Callable] = {}


def get_runner(
    signature: tuple,
    forward: Callable[[jax.Array, jax.Array], jax.Array],
    max_frames: int,
    sample: bool,
) -> Callable:
    """Return a jitted episode runner cached per (topology, horizon, mode).

    The returned callable takes ``(weights, heights, root_key, generation,
    cycle, genome_ids)`` and returns the ``run_episodes`` result dict. One
    XLA program per distinct key replaces the per-frame dispatch storm.
    """
    key = (signature, max_frames, sample)
    runner = _RUNNER_CACHE.get(key)
    if runner is None:
        def run(
            weights: jax.Array,
            heights: jax.Array,
            root_key: jax.Array,
            generation: jax.Array,
            cycle: jax.Array,
            genome_ids: jax.Array,
        ) -> dict[str, jax.Array]:
            return run_episodes(
                weights,
                forward,
                heights,
                max_frames,
                sample,
                root_key,
                generation,
                cycle,
                genome_ids,
            )

        runner = jax.jit(run)
        _RUNNER_CACHE[key] = runner
    return runner


def clear_runner_cache() -> None:
    _RUNNER_CACHE.clear()
