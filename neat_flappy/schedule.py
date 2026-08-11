"""Deterministic pipe schedule shared by the pygame and JAX engines.

Imports neither pygame nor jax so both engines can depend on it and stay on
byte-identical pipe sequences.

Three properties shape the task. Gap size varies per pipe, so the safe flap
threshold scales as error/gap rather than as a fixed offset. Consecutive gaps
alternate between a low and a high band, so the bird must trade off its
position in the current gap against the travel required to reach the next one.
A wide vertical move between two gaps forces the later gap wider. This is an
engineering heuristic that reduces pathological transitions, not an exhaustive
reachability proof in either physics engine.
"""
from __future__ import annotations

import random
import math

import numpy as np

OBSERVATION_COUNT = 5

GAP_MIN = 130
GAP_MAX = 260
HEIGHT_MIN = 50
BOTTOM_LIMIT = 680  # lowest allowed y for the bottom edge of a gap
LOW_BAND_END = 0.35
HIGH_BAND_START = 0.65
JUMP_GAP_FLOOR = 120  # gap width when consecutive gaps share a center
JUMP_GAP_PER_PX = 0.15  # extra gap width per pixel of vertical center move


def pipe_count_for(max_frames: int) -> int:
    """Upper bound on pipes spawned within ``max_frames``."""
    return max_frames // 40 + 16


def _gap_top(rng: random.Random, low: float, high: float, gap: float) -> float:
    """Sample a gap-top y inside ``[low, high]`` of the feasible span for ``gap``."""
    span = BOTTOM_LIMIT - gap - HEIGHT_MIN
    return HEIGHT_MIN + round(rng.uniform(low, high) * span)


def pipe_schedule(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(heights, gaps)`` for the first ``count`` pipes of ``seed``.

    ``heights[i]`` is the y of the gap's top edge and ``gaps[i]`` its height,
    so the gap spans ``[heights[i], heights[i] + gaps[i]]``. Even-indexed
    pipes draw from the low band and odd-indexed pipes from the high band.
    Both are integers so the pygame blit and the JAX AABB test agree.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    heights = np.empty(count, dtype=np.float32)
    gaps = np.empty(count, dtype=np.float32)
    prev_center: float | None = None
    for index in range(count):
        low, high = (
            (0.0, LOW_BAND_END) if index % 2 == 0 else (HIGH_BAND_START, 1.0)
        )
        gap = rng.randrange(GAP_MIN, GAP_MAX + 1)
        top = _gap_top(rng, low, high, gap)
        if prev_center is not None:
            jump = abs(top + gap / 2.0 - prev_center)
            need = JUMP_GAP_FLOOR + JUMP_GAP_PER_PX * jump
            if gap < need:
                gap = min(GAP_MAX, math.ceil(need))
                top = _gap_top(rng, low, high, gap)
                jump = abs(top + gap / 2.0 - prev_center)
                need = JUMP_GAP_FLOOR + JUMP_GAP_PER_PX * jump
                if gap < need:
                    gap = GAP_MAX
                    top = _gap_top(rng, low, high, gap)
        heights[index] = top
        gaps[index] = gap
        prev_center = top + gap / 2.0
    return heights, gaps
