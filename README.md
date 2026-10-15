# Racing RL Event

This repository contains the first Python-only vertical slice for the technical-fest racing event.

The current slice provides:

- A deterministic, track-relative racing environment.
- A validated geometry-file format and one synthetic fixture track.
- A kinematic bicycle-style physics model with steering, throttle, braking, grip, runoff, and barriers.
- The participant-facing reward signal DSL.
- A persistent `n` runs by `k` steps-per-run budget.
- A scripted-run tool that writes a trajectory-like JSON record for later replay and Unity integration.
- A versioned state/action contract, serializable observation normalizer, and a batched NumPy environment for headless training.
- A fixed two-layer PyTorch actor-critic, masked GAE, clipped PPO update, safe
  state-dictionary artifact format, and a measured baseline convergence tool.

Python is authoritative. Unity is not part of this slice and will consume recorded state later.

## Explicit working decisions

These are provisional implementation decisions based on the handoff and the latest project answers:

- Three training tracks are the intended first event target. The current code starts with one fixture track so the vertical slice stays testable.
- Five held-out tracks are a planning hypothesis. They are not included in this participant-facing directory.
- Evaluation submissions will use ONNX. The model input/output contract is not implemented yet.
- Participants may choose their model architecture, subject to a future tournament runner contract. The first slice has no trainer or architecture validator.
- The reward interface is declarative signals and weights only. A raw Python reward escape hatch is intentionally deferred because it weakens reproducibility and sandboxing.
- Training uses a configurable maximum of `n` runs, with at most `k` environment steps in each run. The budget state is persistent and tamper-evident by hash, but it is not a cumulative-week budget.
- The provisional simulation rate is 30 Hz.
- The event date is recorded as the 31st and the environment-release target as the 20th. Month/year still need to be recorded before public scheduling.

## Setup

The supported runtime is Python 3.11. The lock file uses the CPU PyTorch wheel
index, so a GPU and CUDA installation are not required for this headless slice.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

The same dependency set is available through the `Dockerfile`.

## Try the first slice

```bash
OUTPUT="$(mktemp)"
.venv/bin/python tools/run_scripted.py --steps 120 --output "$OUTPUT"
```

The command loads `tracks/train/synthetic_oval.json`, drives a deterministic scripted policy, and writes positions, controls, signals, and termination state to the output file.

Run the checks with the project test command:

```bash
make test
```

## Batched engine and contract

`VectorizedRacingEnv` accepts an `(n_envs, 2)` action array and returns a
fixed-size float32 observation for each environment, vector rewards, signals,
events, and termination/truncation flags. The physics transition is batched
with NumPy while track geometry remains the shared Python authority. Use
`evaluation=True` on reset for a fixed start-line standing start with zero
spawn noise.

The field order and bounds are documented in `docs/state_schema.md` and are
also available through `build_observation_schema`. `TrainingContract` can save
the action/observation versions, environment hashes, deterministic evaluation
metadata, and optional running-normalization statistics.

The measured first benchmark is in `reports/e1_vectorized_benchmark.md`. It is
a local baseline only. The measured PPO baseline experiment is in
`reports/e2_baseline_convergence.md`; it covers the current synthetic fixture
only and does not freeze public event settings. ONNX evaluation, held-out-track
storage, ranking, and Unity playback remain separate milestones.

## Trainer boundary

`RolloutCollector` in `racing_event/rollout.py` is the environment-transition
boundary. It accepts a batched policy callable, reserves the persistent
`RunBudget` in individual environment-transition units, uses a partial active
mask for a final batch, and resets only lanes that ended.
`RolloutMetrics` records raw and weighted reward signals, episode outcomes,
speed, off-track rate, action saturation, action smoothness, and per-track
statistics.

The collector can update a `RunningObservationNormalizer` during training. Save
that normalizer with `TrainingContract.save`; freeze it before evaluation so
evaluation does not change the artifact.

`FixedMLPActorCritic` in `racing_event/ppo.py` is the fixed trainer model. It
has two 64-unit tanh layers, a two-dimensional tanh-squashed Gaussian policy,
and a scalar value head. Its initial output bias is `[0.0, 0.5]`, giving the
longitudinal action a small throttle prior at standing starts. The bias is
stored in the artifact metadata. `PPOTrainer.update` computes masked GAE and performs
the configured full-batch clipped PPO epochs. Terminations do not bootstrap;
truncations bootstrap but do not carry GAE across the reset boundary.

Model weights are saved as a state dictionary with
`FixedMLPActorCritic.save_artifact`; loading uses `weights_only=True` and
validates the JSON architecture metadata. The provisional PPO values are
available from `PPOHyperparameters.to_dict()`. They are not public event
settings until the training set and remaining validation experiments are
frozen.

Run the measured baseline experiment with:

```bash
make e2
```

The report is written to `reports/e2_baseline_convergence.md`.

## Deterministic policy evaluation

The engine includes a fixed-start evaluation path that uses the same Python
physics as training. It runs one lane at a time, uses the policy mean action,
keeps the observation normalizer frozen, and streams a trajectory hash instead
of retaining every frame in memory. This keeps evaluation suitable for a
machine with limited RAM and makes repeated runs comparable.

Save an artifact from an E2 run, then evaluate it twice:

```bash
ARTIFACT_DIR="$(mktemp -d)"

.venv/bin/python tools/run_e2_baseline.py \
  --updates 4 --n-envs 4 --rollout-length 64 \
  --artifact-dir "$ARTIFACT_DIR" \
  --output "$ARTIFACT_DIR/training.md"

.venv/bin/python tools/evaluate_policy.py \
  --weights "$ARTIFACT_DIR/policy.pt" \
  --metadata "$ARTIFACT_DIR/policy.json" \
  --contract "$ARTIFACT_DIR/training_contract.json" \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --repeats 2 \
  --output "$ARTIFACT_DIR/evaluation.json"
```

The evaluator reports `finished`, `dnf_crash`, `dnf_timeout`,
`dnf_stuck`, or `dnf_backward`, plus step count, fixed-step lap time when
finished, physical metrics, and the trajectory digest. A repeated evaluation
must produce the same digest; a mismatch is treated as an error.
