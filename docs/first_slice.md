# First Python Slice

## What is implemented

The current prototype proves the parts that should be stable before a trainer or Unity client is added:

1. Track geometry is loaded from versioned JSON.
2. The loader checks arclength, sample spacing, nesting, edge placement, and a SHA-256 content hash.
3. `TrackGeometry` owns projection, raycasts, and rectangle-footprint edge checks.
4. `RacingEnv` exposes `reset(seed)` and `step_raw(action)` with two continuous actions; `step(action)` remains a legacy reference-reward adapter.
5. Physics uses a fixed timestep and a deterministic kinematic model.
6. Raw transition signals are returned in `info["signals"]`; the dashboard can
   apply UI-edited `RewardConfig` weights, while a custom participant trainer
   may apply any finite local reward function over copies of the published data.
7. `RunBudget` persists `n` logical runs with at most `k` environment transitions per run.
8. The scripted tool writes the versioned trajectory shape intended for later Unity playback.
9. Rollout diagnostics expose spawn-relative sector coverage, positive off-track
   progress, and boundary excess without changing the policy contract.

## Decisions that remain open

- The 31st and 20th dates still need month and year.
- The exact values of `n` and `k` are not frozen.
- ONNX input/output validation and the model execution backend are not implemented. The current environment can define the observation/action contract, but it does not yet run submitted models.
- Participants may choose architecture, but the tournament runner still needs limits for memory, execution time, and output shape. Without limits, arbitrary ONNX graphs can compromise shared evaluation hardware.
- The ranking and DNF rule must be set by the organizers. Training freedom does not make scoring a participant decision.
- The three training tracks and five held-out tracks should be original geometry inspired by F1 circuits only after data sources and licenses are verified. Current fixtures are synthetic and intentionally not copies of real circuits.

## Contract currently used by the environment

- Observation: `10 + ray_count + lookahead_count` `float32` values; the default is 23. The UI permits 1–16 unique forward rays in `[-90, 90]` degrees, including 0.
- Action: two values in `[-1, 1]`: steering and throttle/brake.
- Positive longitudinal action is throttle. Negative action is braking.
- SYNAPSE owns physics, termination, raw signals, progress, and lap time. Participants own reward calculation during training.
- Runoff is soft off-track state with reduced grip. Barrier contact terminates.
- The provisional tick rate is 30 Hz.

## Checks

From the project root:

```bash
python3 -m unittest discover -s tests -v
OUTPUT="$(mktemp)"
python3 tools/run_scripted.py --steps 120 --output "$OUTPUT"
```

The test suite covers track hashes and geometry rejection, deterministic physics and environment stepping, reward signal accounting, local budget persistence/concurrency, and trajectory output shape.
