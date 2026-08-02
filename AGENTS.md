# Repository Guidelines

## Project Overview

NEATBird trains acyclic neural networks to play Flappy Bird. Host Python evolves graph topology and manages checkpoints. JAX compiles policies, simulates training episodes, and applies REINFORCE/RMSProp updates. Pygame provides the replay and rendering state machine.

## Architecture & Data Flow

1. `neat_flappy/schedule.py` creates one seeded pipe-height and gap sequence for both engines.
2. `neat_flappy/genome.py` owns genes, innovations, mutations, crossover, validation, and versioned JSON checkpoints.
3. `neat_flappy/phenotype.py` validates reachable enabled graph structure and compiles cached JAX policy functions.
4. `neat_flappy/vectorized.py` batches genomes with the same topology and runs fixed-horizon episodes with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.
5. `neat_flappy/training.py` evaluates candidates, applies policy-gradient updates with rollback, clusters the population, reproduces genomes, injects elites, and writes checkpoints.
6. `neat_flappy/game.py` replays deterministic policies and renders them with Pygame.

Preserve these boundaries:

- Pygame stays out of the JAX training path.
- Both engines use the same seeded schedule, observations, movement, and rewards. Collision differs intentionally: Pygame uses pixel masks; JAX uses AABBs.
- Graphs are feed-forward DAGs. Validate loaded or changed genomes.
- `InnovationStore` is process-owned and append-only. Keep node and innovation IDs stable.
- JIT paths use pure JAX operations and static shapes. Do not add Python or Pygame work inside them.
- Topology signatures exclude numeric weights. Weight-only changes must not force recompilation.
- Seed folding and stable ordering make results deterministic under batching and padding.

## Key Directories

- `neat_flappy/`: Core game, genome, phenotype, evolution, training, schedule, and visualization modules.
- `tests/`: Pytest behavioral, numerical, persistence, and rendering tests.
- `tools/`: Replay capture and long-horizon checkpoint comparison utilities.
- `docs/`: Experiment findings and committed media.
- `imgs/`: Required Flappy Bird image assets.
- `checkpoints*/`: Portable JSON run artifacts. Treat generated SVG and replay frames as replaceable outputs.

## Development Commands

Install the locked development environment:

```sh
uv sync --locked --dev
```

Run the full suite headlessly:

```sh
SDL_VIDEODRIVER=dummy uv run pytest -q
```

Run default headless training:

```sh
SDL_VIDEODRIVER=dummy uv run python flappy_bird.py train
```

Align learning and evaluation on one layout when needed:

```sh
SDL_VIDEODRIVER=dummy uv run python flappy_bird.py train \
  --eval-seeds 0 --pg-seed 0
```

Replay and visualize checkpoints:

```sh
uv run python flappy_bird.py replay \
  --genome checkpoints/best_genome.json --seed 0

uv run python flappy_bird.py visualize \
  --best-genome checkpoints/best_genome.json \
  --genome checkpoints/topology_009.json \
  --output checkpoints/genomes.svg --seed 0
```

The repository configures no lint, format, type-check, coverage, CI, or task-runner command. Do not invent one. Hatchling is the build backend, but the repository defines no build workflow.

## Code Conventions & Common Patterns

- Use Python 3.11+ type annotations, domain-specific names, dataclasses for records/configuration, and enums for closed gene/operator sets.
- Use underscore-prefixed names for internal helpers.
- Pass dependencies explicitly: `Genome`, `InnovationStore`, `TrainingConfig`, `random.Random`, JAX keys, schedules, or controller callables.
- Use `ValueError` for invalid input/configuration, `RuntimeError` for invalid state or broken invariants, `FloatingPointError` for non-finite gradients, and `SystemExit` only at the CLI boundary.
- Preserve `zip(..., strict=True)`, deterministic sorting, stable tie-breaking, and explicit seeded RNG use.
- Keep numeric arrays `float32` where the existing JAX/game contract requires them.
- Group genomes by reachable enabled topology before JAX batching. Keep fixed population widths and frame horizons where they prevent XLA recompilation.
- Update mutable genome weights and RMS caches through the existing compiled innovation order.
- `FlappyEpisode` uses a two-step state transition: `prepare_frame()` creates observations, then `step(actions)` finishes that frame. Do not bypass this protocol.
- There are no async, threading, or multiprocessing patterns. Keep control flow synchronous unless the task explicitly requires a new concurrency model.
- Save portable, versioned JSON. Do not introduce Python pickle checkpoints.

## Important Files

- `flappy_bird.py`: CLI entry point for `train`, `replay`, and `visualize`.
- `pyproject.toml`: Python requirement, dependencies, Hatchling settings, and pytest discovery.
- `uv.lock`: Exact uv dependency resolution, including JAX CUDA 13 packages.
- `neat_flappy/schedule.py`: Shared deterministic layout and five-observation contract.
- `neat_flappy/genome.py`: Load-bearing graph invariants and checkpoint schema.
- `neat_flappy/phenotype.py`: Topology compilation and JAX function cache.
- `neat_flappy/vectorized.py`: Pure-JAX training simulator.
- `neat_flappy/game.py`: Pygame replay/render state machine.
- `neat_flappy/training.py`: End-to-end training lifecycle.
- `neat_flappy/evolution.py`: Compatibility distance, PAM clustering, extinction, and reproduction.
- `README.md`: User workflows and current system behavior.
- `docs/findings.md`: Historical experiment record, not the primary setup guide.

## Runtime/Tooling Preferences

- Use Python `>=3.11` and `uv`; do not use Node, Bun, npm, or pip-based ad hoc environments.
- Use `uv.lock` and `uv sync --locked --dev` for reproducible work.
- Runtime dependencies are JAX with CUDA 13 support, NumPy, and Pygame. Confirm GPU selection with `uv run python -c "import jax; print(jax.devices())"` when GPU behavior matters.
- Training is headless. Use `SDL_VIDEODRIVER=dummy` for tests, training, and replay-frame capture in environments without a display.
- ImageMagick is optional and only encodes captured PNG frames into GIF files.
- `tools/capture_replay.py` deletes existing `frame_*.png` files in its output directory before capture. Never use an input directory that contains frames to preserve.

## Testing & QA

Pytest collects `tests/test_*.py` from the configured `tests` root. Tests are module-level functions named `test_<behavior>`. There is no shared `conftest.py`; use local helpers, built-in `tmp_path`, and `monkeypatch` when needed.

Match existing assertion patterns:

- Exact equality for shapes, state transitions, counts, IDs, and deterministic results.
- `pytest.approx` for scalar floats.
- `np.testing.assert_allclose` for numeric arrays and `assert_array_equal` for exact arrays.
- `pytest.raises(..., match=...)` for validation errors.
- Convert JAX values with `float(...)` or `np.asarray(...)` before NumPy assertions when required.

Tests must defend observable contracts. For JAX changes, compare batched/JIT output with a scalar reference, assert finite loss and gradients, and test update direction. Seed both Python and JAX RNGs. Keep horizons small and scenarios deterministic. Game tests load Pygame surfaces and masks without creating a window; do not add a display dependency. No coverage threshold is configured.