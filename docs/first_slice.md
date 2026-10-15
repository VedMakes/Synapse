# First Python Slice

## What is implemented

The current prototype proves the parts that should be stable before a trainer or Unity client is added:

1. Track geometry is loaded from versioned JSON.
2. The loader checks arclength, sample spacing, nesting, edge placement, and a SHA-256 content hash.
3. `TrackGeometry` owns projection, raycasts, and rectangle-footprint edge checks.
4. `RacingEnv` exposes `reset(seed)` and `step(action)` with two continuous actions.
5. Physics uses a fixed timestep and a deterministic kinematic model.
6. Raw reward signals are returned in `info["signals"]` and can be weighted with the declarative `RewardConfig`.
7. `RunBudget` persists `n` logical runs with at most `k` environment transitions per run.
8. The scripted tool writes the versioned trajectory shape intended for later Unity playback.

## Decisions that remain open

- The 31st and 20th dates still need month and year.
- The exact values of `n` and `k` are not frozen.
- ONNX input/output validation and the model execution backend are not implemented. The current environment can define the observation/action contract, but it does not yet run submitted models.
- Participants may choose architecture, but the tournament runner still needs limits for memory, execution time, and output shape. Without limits, arbitrary ONNX graphs can compromise shared evaluation hardware.
- The ranking and DNF rule must be set by the organizers. Training freedom does not make scoring a participant decision.
- The three training tracks and five held-out tracks should be original geometry inspired by F1 circuits only after data sources and licenses are verified. Current fixtures are synthetic and intentionally not copies of real circuits.

## Contract currently used by the environment

- Observation: 23 `float32` values for the default configuration.
- Action: two values in `[-1, 1]`: steering and throttle/brake.
- Positive longitudinal action is throttle. Negative action is braking.
- Python owns physics, termination, rewards, progress, and lap time.
- Runoff is soft off-track state with reduced grip. Barrier contact terminates.
- The provisional tick rate is 30 Hz.

## Checks

From the project root:

```bash
python3 -m unittest discover -s tests -v
python3 tools/run_scripted.py --steps 120 --output /tmp/racing-run.json
```

The test suite covers track hashes and geometry rejection, deterministic physics and environment stepping, reward signal accounting, local budget persistence/concurrency, and trajectory output shape.
