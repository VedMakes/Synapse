# SYNAPSE Racing RL Event

This repository contains the first Python-only vertical slice for the SYNAPSE technical-fest event.

The current slice provides:

- A deterministic, track-relative racing environment.
- A validated geometry-file format and three original synthetic development tracks.
- A kinematic bicycle-style physics model with steering, throttle, braking, grip, runoff, and barriers.
- The participant-facing reward signal DSL.
- A persistent `n` runs by `k` steps-per-run budget.
- A scripted-run tool that writes a trajectory-like JSON record for later replay and Unity integration.
- A versioned state/action contract, serializable observation normalizer, and a batched NumPy environment for headless training.
- A fixed two-layer PyTorch actor-critic, masked GAE, clipped PPO update, safe
  state-dictionary artifact format, and a measured baseline convergence tool.
- A validated, streaming replay export and dependency-free read-only browser
  preview, with a documented future Unity adapter contract.

Python is authoritative. Unity has a read-only renderer foundation under
`unity/`; it does not simulate or score cars.

## Explicit working decisions

These are provisional implementation decisions based on the handoff and the latest project answers:

- Three training tracks are the intended first event target. Three synthetic development tracks are present; E2 currently uses the oval only, not a frozen event training set.
- Five held-out tracks are a planning hypothesis. They are not included in this participant-facing directory.
- Evaluation submissions will use ONNX. The model input/output contract is not implemented yet.
- Participant architecture freedom remains subject to the future tournament contract. The local PPO baseline currently uses one fixed MLP; it is not the final architecture submission validator.
- The reward interface is declarative signals and weights only. A raw Python reward escape hatch is intentionally deferred because it weakens reproducibility and sandboxing.
- Training uses a configurable maximum of `n` runs, with at most `k` environment steps in each run. The budget state is persistent and tamper-evident by hash, but it is not a cumulative-week budget.
- The provisional simulation rate is 30 Hz.
- The event date is recorded as the 31st and the environment-release target as the 20th. Month/year still need to be recorded before public scheduling.

## Participant training prototype

The first minimal participant-facing workflow is available as a bounded local
CLI. It is a delivery prototype, not a frozen event launcher or final budget
rule:

First prepare the project environment. This keeps NumPy, PyTorch, and all
dependencies inside `.venv`:

```bash
python3 tools/setup_participant.py
```

The same setup action is available from the dashboard's **Set up / repair
Python** button.

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --steps 2048 --budget-steps 10000 \
  --output-dir participant_run
```

It loads all public training maps, uses the fixed two-layer 64/64 PPO model,
prints progress after each bounded rollout update, persists a local transition
counter, and writes `training_report.md`, `training_metrics.jsonl`, policy
artifacts, and the training contract under `participant_run/`. Run the same
command again with the same output directory to resume the local counter.
The prototype persists policy weights and observation normalization but resets
the optimizer on process restart; this is recorded as an open release decision.

The current onboarding budget example is 10,000 accepted environment
transitions. It is intentionally not the scored event budget. The recommended
fairness direction is a cumulative count of active-lane transitions, with
vector lanes and repeated frames charged individually, but the exact budget
and enforcement model remain open. See `docs/participant_guide.md` and the
research report `reports/Participant delivery options.md`.

For a simple browser dashboard around the same CLI, run:

```bash
.venv/bin/python tools/participant_ui.py
```

Then open **http://127.0.0.1:8766/**. The UI is served by Python's standard
library, binds to loopback, launches the existing CLI, and polls its bounded
progress output. It does not implement a second environment or budget. After a
policy is saved, use **Watch best run for map** to open a deterministic replay
for one public map.

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
speed, off-track rate, action saturation, action smoothness, policy-invisible
sector coverage, positive off-track progress, boundary excess, and per-track
statistics. The diagnostic fields do not enter observations, rewards,
termination, ranking, or replay schema 1.1.

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

The measured local E2 artifact is retained outside Git at
`artifacts/e2-stable/` (`policy.pt`, `policy.json`, and
`training_contract.json`).

## Replay UI boundary (Python simulation only)

Generate a complete smoke-controller replay and bundle an offline preview:

```bash
mkdir -p artifacts/replay
.venv/bin/python tools/export_replay.py --scripted \
  --participant-id scripted-demo --output artifacts/replay/run.json
.venv/bin/python tools/build_replay.py artifacts/replay/run.json \
  --output artifacts/replay/index.html
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory artifacts/replay
```

Open **http://localhost:8765/**, or open `artifacts/replay/index.html` directly.
The HTML contains its own data/styles/scripts and needs no network connection,
frontend dependencies, Node runtime, or running simulator. It has play/pause,
restart, next-tick, scrubber, playback speed, recorded telemetry/events, track
geometry, car pose, and the Python result. Exported files are ignored by Git.

To view the retained learned E2 artifact instead of the scripted controller:

```bash
.venv/bin/python tools/export_replay.py \
  --weights artifacts/e2-stable/policy.pt \
  --metadata artifacts/e2-stable/policy.json \
  --contract artifacts/e2-stable/training_contract.json \
  --participant-id e2-stable --output artifacts/replay/e2-stable.json
.venv/bin/python tools/build_replay.py artifacts/replay/e2-stable.json \
  --output artifacts/replay/index.html
```

The exporter streams frames to an atomic JSON file. The bundler verifies
version, tick ordering, hashes, finite values, and outcome consistency first.
See `docs/replay_contract.md` for schema **1.1**, units, time-zero semantics,
memory limits, checksum details, and the Unity coordinate mapping. The legacy
scripted smoke format 1.0 is intentionally not accepted by the new UI.

The browser interpolates **only the recorded pose**, never physics. Neither
browser nor future Unity may compute contacts, resets, rewards, progress,
finishes, or lap times. Private held-out geometry must never be exported into
a public preview. Checksums are integrity checks, not anti-cheat authentication.

### Unity renderer foundation

The read-only Unity adapter foundation is under `unity/` and targets pinned
Unity `2022.3.22f1` LTS with URP `14.0.8`. It validates replay schema `1.1`,
maps Python coordinates and headings using the same contract, applies recorded
poses directly, and builds visual track geometry from the embedded boundaries.
It contains no vehicle physics, collision, lap timing, reward, progress, or
finish-line logic. `GhostPresentationConfig` keeps simultaneous versus
staggered reveal configurable without changing replay data. See
`docs/unity_renderer_adapter.md` for the handoff and joint acceptance tests.

### Layer checks and container preview

`make test` runs Python simulation/export/read-back checks. If Node is available
it also runs the dependency-free JavaScript playback/DOM tests; otherwise that
test is explicitly **skipped**. Run them directly with:

```bash
node --test tests/replay_frontend.test.cjs
```

No browser binary is installed by the project. The JS checks use a DOM/canvas
test double, not a visual browser test. A real browser/Unity visual parity pass
is still required before the renderer contract is frozen for the event.

To generate and serve a preview using the existing Python container:

```bash
docker build -t racing-event .
mkdir -p artifacts/replay
docker run --rm -v "$PWD/artifacts/replay:/preview" racing-event \
  python tools/export_replay.py --scripted --output /preview/run.json
docker run --rm -v "$PWD/artifacts/replay:/preview" racing-event \
  python tools/build_replay.py /preview/run.json --output /preview/index.html
docker run --rm -p 127.0.0.1:8765:8765 \
  -v "$PWD/artifacts/replay:/preview:ro" racing-event \
  python -m http.server 8765 --bind 0.0.0.0 --directory /preview
```

This is a local preview deployment, not a public upload or tournament service.
Random-respawn findings and the questions deferred for later discussion are in
`research_notes/Track sourcing and racing physics/racing_rl_controls.md`.
