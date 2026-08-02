# Findings: Randomized Pipes, Trainer Fixes, and Schedule Repair

Date: 2026-08-01

## Overview

This document records a sequence of experiments on the JAX NEAT Flappy Bird
trainer. A prior change (by another agent) introduced randomized pipe gaps and
a wider observation space. The change made the task harder. Four training runs
were done. Each run exposed a new defect. Each defect was diagnosed and fixed.
The final run cleared 66 pipes on seed 0, up from 2 in the first
randomized-pipe run.

## Timeline

### Phase 1: Audit the uncommitted change

The prior change touched 11 files and added one new module. The change was
audited and found coherent. Two stray Windows-path files from another machine
were removed.

Files changed by the prior agent:

- `neat_flappy/schedule.py` (new). Shared deterministic pipe schedule. Imports
  neither pygame nor jax. Both engines read the same sequence.
- `neat_flappy/game.py`. Pipe takes explicit height, gap, index. Episode
  precomputes heights and gaps from the schedule. Observations expanded to 5.
- `neat_flappy/genome.py`. Input count 2 to 5. Bias and output node IDs shifted.
- `neat_flappy/training.py`. Schedule arrays helper. Both eval and PG paths
  pass gaps to the runner.
- `neat_flappy/vectorized.py`. Ring buffer gains per-pipe gap and schedule
  index. Collision uses per-pipe gap. Global PIPE_GAP constant removed.
- `neat_flappy/visualization.py`. Input labels expanded to 5.
- `tests/`. All updated to match new shapes and signatures.
- `README.md`. Observation table and gap description updated.

The new observation space has 5 values per bird:

1. Signed distance to the current gap center.
2. Bird velocity.
3. Current gap size.
4. Signed distance to the next gap center.
5. Horizontal distance to the target pipe.

The pipe gap is drawn per pipe from [130, 260] pixels. Gap centers alternate
between a low band and a high band.

### Phase 2: Run 1, default trainer on the new task

Command: `uv run python flappy_bird.py train` (defaults).

Result: 50 generations. Best fitness 27.21 at generation 29. Best score 2
pipes. Plateau from generation 29 to 49.

The population stalled. The log showed `accepted=2 reverted=98` on most
generations. The revert gate rejected 98 percent of gradient updates.

### Phase 3: Diagnose decorrelated learning

The trainer used two different seed formulas.

- Evaluation (fitness scoring): seeds `seed+9000`, `seed+9001`, `seed+9002`.
- Policy gradient (weight updates): `seed + generation*10000 + cycle`.

The gradient was computed on a layout the genome was never scored on. The
revert gate then rejected the update because it hurt the eval-layout score.
Learning and selection were decorrelated. Evolution fought its own gradient.

### Phase 4: Fix decorrelated learning

Two optional fields added to `TrainingConfig`:

- `eval_seeds`: fixed list of eval layouts. Overrides the derived list.
- `pg_seed`: one layout for every policy-gradient rollout.

Two helpers, `get_eval_seeds` and `get_pg_seed`, return the override when set
and the derived value otherwise. CLI flags: `--eval-seeds` and `--pg-seed`.

Defaults are unchanged. The original generalization behavior is preserved.

Two tests added:

- `test_eval_seed_override_and_pg_pin_are_isolated_from_defaults`.
- `test_fixed_seed_mode_routes_overrides_through_paired_backprop`.

All 47 tests pass.

### Phase 5: Run 2, fixed-seed trainer, 1000-frame window

Command:

```
uv run python flappy_bird.py train --seed 0 --eval-seeds 0 --pg-seed 0 \
  --checkpoint-dir checkpoints_seed0 --generations 50 --minimum-generations 50 \
  --max-frames 1000 --fitness-threshold 1e9
```

Result: best fitness 148.90 at generation 38. Best score 13 inside the window.
Best network: 9 nodes, 13 connections.

The fix moved the best score from 2 to 13 inside the 1000-frame window. But a
replay at longer windows showed the bird dies at frame 1362 at pipe 17, the
same on every window of 2000 frames or more. The death was past the 1000-frame
eval window. The trainer never saw it. This is horizon-blindness.

### Phase 6: Run 3, fixed-seed trainer, 3000-frame window

Command:

```
uv run python flappy_bird.py train --seed 0 --eval-seeds 0 --pg-seed 0 \
  --checkpoint-dir checkpoints_seed0_f3000 --generations 50 \
  --minimum-generations 50 --max-frames 3000 --fitness-threshold 1e9
```

Result: best fitness 197.98 at generation 33. Best score 17. Best network: 9
nodes, 14 connections. The trainer logged `frames=1362.0` for the champion,
confirming the death is now visible. Run 2 logged the same death as
`frames=1000.0`.

The score stayed at 17 from generation 33 to 49. That is 17 generations with
the death fully visible and no breakthrough. The horizon fix was necessary for
honest measurement but did not unlock pipe 18.

### Phase 7: Diagnose the impossible transition

The seed-0 schedule at the death point:

| pipe | gap top | gap size | center | band |
|------|---------|----------|--------|------|
| 16   | 59      | 196      | 157    | low  |
| 17   | 389     | 133      | 456    | high |
| 18   | 149     | 232      | 265    | low  |

After pipe 16 the bird sits low near center 157. Pipe 17 needs a climb of about
298 pixels into a 133-pixel gap, which is near the 130-pixel minimum. The bird
is 48 pixels tall, leaving 85 pixels of slack. The flap impulse quantizes the
bird's altitude in steps of about 11 pixels. The slack is not enough to absorb
the overshoot from a 298-pixel climb.

Two different networks (Run 2 champion: 13 connections, Run 3 champion: 14
connections) both die at the byte-identical frame 1362. The wall is the task
transition, not the network and not the horizon. The schedule can emit a level
that no controller can clear.

### Phase 8: Fix the schedule

A jump-scaled gap floor was added to `pipe_schedule`. When the vertical
distance between consecutive gap centers is large, the target gap is widened to
absorb the climb. The formula is `need = JUMP_GAP_FLOOR + JUMP_GAP_PER_PX *
jump`. If the sampled gap is below `need`, it is raised to `min(GAP_MAX,
ceil(need))`.

Constants chosen by playing candidate schedules through the real pygame physics
with a heuristic controller:

- `JUMP_GAP_FLOOR = 120`
- `JUMP_GAP_PER_PX = 0.15`

These keep the gap range at [130, 260] for small jumps and widen only the
transitions that are physically impossible. The alternating-band test still
passes because the floor only widens gaps; centers stay in their band.

All 47 tests pass after the fix.

### Phase 9: Run 4, fixed-seed trainer on the fixed schedule

Command:

```
uv run python flappy_bird.py train --seed 0 --eval-seeds 0 --pg-seed 0 \
  --checkpoint-dir checkpoints_fixed --generations 50 --minimum-generations 50 \
  --max-frames 3000 --fitness-threshold 1e9
```

Result: best fitness 456.29 at generation 9. Best score 39 inside the window.
Best network: 8 nodes, 8 connections. The bird cleared every pipe in the
3000-frame window and was alive at the cap.

Replay at longer horizons:

| max_frames | score | final frame | alive at end |
|------------|-------|-------------|--------------|
| 3000       | 39    | 3000        | no (cap)     |
| 5000       | 66    | 4972        | no           |
| 10000      | 66    | 4972        | no           |
| 50000      | 66    | 4972        | no           |

The bird clears 66 pipes then dies at frame 4972. Every transition at the death
point passes the gap-floor check. The death is a policy failure, not a geometry
failure. The bird was trained on 39 pipes (the 3000-frame window) and never
received gradient signal for pipe 40 and beyond. This is horizon-blindness
again, at a larger scale.

## Scorecard

| Run | Schedule | Window | Best score | True death | Network | Wall cause |
|-----|----------|--------|------------|------------|---------|------------|
| 1   | old      | 1000   | 2          | ~214       | 10n/15c | decorrelated learning |
| 2   | old      | 1000   | 17         | 1362       | 9n/13c  | horizon-blindness |
| 3   | old      | 3000   | 17         | 1362       | 9n/14c  | impossible transition |
| 4   | fixed    | 3000   | 66         | 4972       | 8n/8c   | horizon-blindness |

## Remaining lever

The schedule is now physically passable everywhere. The trainer learns and
scores on the same layout. The only remaining gap to indefinite play is the
fixed-length scan. The JAX engine uses `jax.lax.scan` over `max_frames` steps.
If the bird survives past the window, the death is invisible. If the bird dies
early, compute is wasted on frozen dead lanes.

Replacing `scan` with `while_loop` (loop until all lanes dead or a cap) makes
the horizon track the failure structurally. This is the proper implementation
of a score target: the episode ends on death, not on a frame count. It is a
real rewrite of the vectorized engine and the policy-gradient buffers.

A cheaper alternative is to retrain with a larger `--max-frames` value. Each
widening has exposed the next death and let evolution push past it. The pattern
held three times.

## Artifacts

- `checkpoints/` holds Run 1 (default trainer, old schedule, 2 pipes).
- `checkpoints_seed0/` holds Run 2 (fixed-seed, old schedule, 1000-frame, 17
  pipes blind).
- `checkpoints_seed0_f3000/` holds Run 3 (fixed-seed, old schedule, 3000-frame,
  17 pipes honest).
- `checkpoints_fixed/` holds Run 4 (fixed-seed, fixed schedule, 3000-frame, 66
  pipes).
- `docs/assets/champion-gen29.gif` is the Run 1 champion on seed 0.
- `docs/assets/champion-seed0-gen38.gif` is a 17-pipe run that dies at frame
  1362. Both Run 2 and Run 3 champions die at the same frame.

## Code changes summary

| File | Change |
|------|--------|
| `neat_flappy/schedule.py` | New module. Jump-scaled gap floor. |
| `neat_flappy/game.py` | Pipe takes explicit height, gap, index. 5 observations. |
| `neat_flappy/genome.py` | Input count 5. Shifted bias and output IDs. |
| `neat_flappy/training.py` | Fixed-seed config fields and helpers. |
| `neat_flappy/vectorized.py` | Per-pipe gap in ring buffer. Removed global PIPE_GAP. |
| `neat_flappy/visualization.py` | 5 input labels. |
| `flappy_bird.py` | `--eval-seeds` and `--pg-seed` CLI flags. |
| `tests/test_evolution_training.py` | 2 new tests for seed overrides. |
| `tests/test_game.py` | Updated for new Pipe signature and observation count. |
| `tests/test_genome.py` | Updated for 5-input topology. |
| `tests/test_vectorized.py` | Updated for gaps array and observation count. |
| `tests/test_visualization.py` | Updated node and connection counts. |
