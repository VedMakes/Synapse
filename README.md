# SYNAPSE Racing RL Event

This repository contains the first Python-only vertical slice for the SYNAPSE technical-fest event.

The current slice provides:

- A deterministic, track-relative racing environment.
- A validated geometry-file format and three original synthetic development tracks.
- A kinematic bicycle-style physics model with steering, throttle, braking, grip, runoff, and barriers.
- A participant-owned reward boundary over published raw transition signals.
- A persistent `n` runs by `k` steps-per-run budget.
- A scripted-run and replay-export path for the read-only Pygame renderer.
- A versioned state/action contract, serializable observation normalizer, and a batched NumPy environment for headless training.
- An optional fixed two-layer PyTorch reference baseline, masked GAE, clipped
  PPO update, safe state-dictionary artifact format, and a measured baseline
  convergence tool.
- A validated, streaming replay export with a Pygame renderer and an optional
  dependency-free browser fallback.
- A bounded ONNX ghost-race interface that evaluates multiple model artifacts
  on shared tracks and can emit compatible replay groups.
- A fixed-rate live ONNX heat path that runs fresh model actions during the
  Python simulation and streams read-only state to the browser renderer.
- A participant dashboard with side-by-side live-heat controls, structured
  latency/deadline/failure telemetry, and supervised process resource reporting.

Python is authoritative. Pygame is a read-only renderer that consumes validated
replays; it does not simulate or score cars. Unity is no longer an active
project target.

## Explicit working decisions

These are provisional implementation decisions based on the handoff and the latest project answers:

- Three training tracks are the intended first event target. Three synthetic development tracks are present; E2 currently uses the oval only, not a frozen event training set.
- Five held-out tracks are a planning hypothesis. They are not included in this participant-facing directory.
- Evaluation submissions will use ONNX. A bounded static-vector profile and
  model-only Docker worker are implemented as release candidates; actual Docker
  verification and final event-profile acceptance are pending. See
  `docs/submission_runtime.md`.
- Participants own their model architecture, algorithm, preprocessing, weights,
  and reward function. The local PPO baseline is only an optional starter, not
  the event contract.
- SYNAPSE publishes raw transition signals. Participants may implement reward
  weights or arbitrary local reward code in their own trainer; scoring never
  executes participant reward code.
- Training uses a configurable maximum of `n` runs, with at most `k` environment steps in each run. The budget state is persistent and tamper-evident by hash, but it is not a cumulative-week budget.
- The provisional simulation rate is 30 Hz.
- The event date is recorded as the 31st and the environment-release target as the 20th. Month/year still need to be recorded before public scheduling.

## Participant training launcher

The first minimal participant-facing workflow is a bounded local launcher. It
does not construct a model; it runs the participant trainer you select. The
dashboard owns the static nine-signal reward-weight file and sensor-ray layout.
It is a delivery prototype, not a frozen event launcher or final budget rule:

First prepare the project environment. This keeps the SYNAPSE environment
dependencies inside `.venv`:

```bash
python3.11 tools/setup_participant.py
```

The same setup action is available from the dashboard's **Set up / repair
Python** button.

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python tools/train_participant.py \
  --trainer examples/participant_train.py \
  --reward examples/baseline_reward.json \
  --steps 2048 --budget-steps 10000 \
  --output-dir participant_run
```

The launcher forwards the published environment inputs to the trainer script
you select. A trainer entrypoint must accept the forwarded options
`--track-dir`, `--config`, `--output-dir`, `--steps`, `--budget-steps`,
`--n-envs`, `--rollout-length`, and `--seed`; `--reward` is forwarded only
when supplied. The UI writes `run_config.json`, `reward_config.json`, and
`run_identity.json` into the output directory without mutating the repository
defaults. `examples/participant_train.py` is only a runnable reference
starter; replace it with your own script to choose the model, algorithm,
preprocessing, weights, and reward implementation. Run the same command again
with the same output directory if your trainer supports resuming.

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
library, binds to loopback, launches the participant-owned trainer through the
launcher, and polls its bounded progress output. It does not implement a
second model, environment, or budget. The reward editor and ray controls are
input/configuration UI, not a second simulation authority. The watch panel now
explicitly selects the **latest saved reference artifact** or a **safe NumPy
participant artifact**, plus **Pygame** or the **browser fallback**. It requires
the selected run's `run_config.json` so custom rays cannot silently fall back to
the defaults. Both choices produce browser-playback and JSON-download links.
On a headless host, Pygame is honestly labeled a one-frame check; use the browser
link for playback. Inputs stay locked during a run, and errors appear inline.
The dashboard executes local code: keep it on loopback. Cross-origin mutations
and non-loopback host names are rejected; it is not a public service.

### Participant live heat dashboard

The dashboard also provides a **Grid control / live heat** panel for running two
validated ONNX artifacts side by side. Python remains authoritative for the
observations, model actions, fixed-step physics, deadlines, events, and result;
the browser is only a control surface and telemetry view. The panel exposes:

- two participant IDs and two artifact directories;
- one explicit track, simulation config, output directory, and WebSocket port;
- action budget, live tick limit, and `strict_scored` versus local-demo
  `trusted_live` late-action policy;
- optional organizer isolation, when the evaluator image is available;
- per-entrant status, latency, deadline misses, inference failures, and model
  file size;
- shared live-process RSS/peak RSS, PID count, thread count, and bounded child
  output in the terminal panel.

Start the dashboard from the repository root:

```bash
.venv/bin/python tools/participant_ui.py --port 8766
```

Open **http://127.0.0.1:8766/** and use **03 · Grid control / live heat**.
Choose two artifact directories, then start the heat. The default renderer HTTP
server uses port `8765`; the dashboard's live-heat WebSocket default is `8767`
so the three local services do not collide. Runtime RSS is measured for the
supervised race process tree and is intentionally labeled shared; the dashboard
does not pretend that in-process model memory is separately measurable.

For a direct CLI smoke or a non-dashboard run, use the live runner explicitly:

```bash
.venv/bin/python tools/run_live_onnx.py \
  --entrant small-alpha artifacts/onnx-imitation \
  --entrant small-beta artifacts/onnx-imitation \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --output-dir artifacts/live/small-heat \
  --port 8767 \
  --max-steps 2400 \
  --policy-budget-ms 20 \
  --late-policy strict_scored \
  --telemetry-stdout
```

Use `trusted_live` only for a local demonstration where holding the last action
after one late inference is acceptable. It is not the scored policy. The CLI
prints the actual WebSocket endpoint and emits bounded `LIVE_TELEMETRY` JSON
lines when `--telemetry-stdout` is enabled. The live report is
`artifacts/live/small-heat/live-race.json`; it is a live result, not a replay.

## Setup

The supported runtime is Python 3.11. The lock file uses the CPU PyTorch wheel
index, so a GPU and CUDA installation are not required. Pygame uses software
SDL drawing and is also exercised with dummy SDL drivers in CI.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

The same dependency set is available through the `Dockerfile`.

## Try the first slice

```bash
make render
```

The command exports a schema 1.1 replay from the deterministic scripted policy
and renders one frame through Pygame with dummy SDL drivers. For an interactive
window, run `make render` with `SDL_VIDEODRIVER` unset and launch:

```bash
.venv/bin/python tools/run_pygame.py --replay artifacts/replay/pygame-smoke.json
```

Run the checks with the project test command:

```bash
make test
```

## Batched engine and contract

`VectorizedRacingEnv` accepts an `(n_envs, 2)` action array and returns a
float32 observation whose size is `10 + ray_count + lookahead_count`, raw signals, events, and
termination/truncation flags through `step_raw`. The legacy `step` method also
returns reference weighted rewards for compatibility. The physics transition
is batched with NumPy while track geometry remains the shared Python authority. Use
`evaluation=True` on reset for a fixed start-line standing start with zero
spawn noise.

The field order and bounds are documented in `docs/state_schema.md` and are
also available through `build_observation_schema`. `TrainingContract` can save
the action/observation versions, environment hashes, deterministic evaluation
metadata, and optional running-normalization statistics.

The measured first benchmark is in `reports/e1_vectorized_benchmark.md`. It is
a local baseline only. The measured PPO baseline experiment is in
`reports/e2_baseline_convergence.md`; it covers the current synthetic fixture
only and does not freeze public event settings. Held-out-track storage, final
ranking, and event release freeze remain separate milestones; ONNX inference
and local multi-ghost comparison now have a non-renderer race interface. See
[`docs/submission_runtime.md`](docs/submission_runtime.md). It does not change
the renderer directory or the single-car replay schema.

For live actions rather than replay playback, install
`requirements-live.lock`, serve `renderer/`, and run either the participant
dashboard or `tools/run_live_onnx.py`. The initial profile is 30 Hz
physics/control with a 20 ms candidate action budget and strict late-action DNF.
The renderer is normally served at `http://127.0.0.1:8765/`; connect it to the
printed WebSocket endpoint, for example
`http://127.0.0.1:8765/?live=ws://127.0.0.1:8767/live/v1`. The browser is
presentation-only; Python remains authoritative for observations, actions,
physics, events, and results.

## Trainer boundary

`RolloutCollector` in `racing_event/rollout.py` is a reusable transition
adapter, not the participant's model or algorithm. It accepts a batched policy
callable and an optional participant-owned reward callback, reserves the persistent
`RunBudget` in individual environment-transition units, uses a partial active
mask for a final batch, and resets only lanes that ended. Each call is capped
at 8,192 transitions by default; call `collect()` repeatedly for larger runs.
The local launcher is cooperative: a custom trainer must use the environment
and budget APIs if it needs the same accounting guarantees as the reference
starter.
`RolloutMetrics` records raw signals and optional reference weighted terms, episode outcomes,
speed, off-track rate, action saturation, action smoothness, policy-invisible
sector coverage, positive off-track progress, boundary excess, and per-track
statistics. The diagnostic fields do not enter observations, rewards,
termination, ranking, or replay schema 1.1.

The collector can update a `RunningObservationNormalizer` during training. Save
that normalizer with `TrainingContract.save`; freeze it before evaluation so
evaluation does not change the artifact.

`FixedMLPActorCritic` in `racing_event/ppo.py` is an optional reference model. It
has two 64-unit tanh layers, a two-dimensional tanh-squashed Gaussian policy,
and a scalar value head. Its initial output bias is `[0.0, 0.5]`, giving the
longitudinal action a small throttle prior at standing starts. The bias is
stored in the artifact metadata. `PPOTrainer.update` computes masked GAE and performs
the configured full-batch clipped PPO epochs. Terminations do not bootstrap;
truncations bootstrap but do not carry GAE across the reset boundary.

Reference weights are saved as a state dictionary with
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

Evaluation JSON also includes a bounded `diagnostics` summary: off-track seconds
and longest excursion, positive/off-track/backward progress, maximum boundary
excess, stalled runs, driven sector coverage, first forward sector-crossing
ticks, and per-channel saturation/applied-command deltas. These are physical
audits, not extra policy inputs, rewards, DNF rules, or leaderboard scores.
Evaluation takes one explicit track at a time, never a randomly selected pool.
Replay schema 1.1 and its trajectory digest remain unchanged.

The measured local E2 artifact is retained outside Git at
`artifacts/e2-stable/` (`policy.pt`, `policy.json`, and
`training_contract.json`).

## Declarative participant artifact (first generic evaluator slice)

The repository now includes a Torch-free, organizer-side evaluator for a safe
first generic submission format. A participant artifact is a directory with
exactly:

```text
submission/
  manifest.json
  training_contract.json
  weights.npz
```

`manifest.json` declares a sequential NumPy MLP, preprocessing mode, schema
identity, and file hashes. The loader supports participant-chosen feed-forward
layer widths and `identity`, `relu`, and `tanh` activations, with bounded layer,
parameter, file-size, and inference-batch limits. It does not load participant
Python, pickle, reward modules, or arbitrary operators. The contract normalizer
must be frozen for this artifact path.

Evaluate it against organizer-provided tracks with:

```bash
.venv/bin/python tools/evaluate_participant.py \
  --artifact-dir submission \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --repeats 2 \
  --output artifacts/participant-evaluation.json
```

The same artifact can be exported through the existing replay schema:

```bash
.venv/bin/python tools/export_replay.py \
  --artifact-dir submission \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --participant-id team-a \
  --output artifacts/replay/team-a.json
```

This remains the first generic artifact format, not the final event contract.
For ONNX export, the supported architecture/operator profile, optional pinned
dependencies, and **explicit isolated organizer evaluation**, see
[`docs/submission_runtime.md`](docs/submission_runtime.md). In-process evaluation
is labeled trusted-local preview, not a sandbox. Actual container verification,
outer job supervision, and event rules remain release gates.

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

The exporter streams frames to an atomic JSON file. The renderer verifies
version, tick ordering, hashes, finite values, and outcome consistency before
drawing. See `docs/replay_contract.md` for schema **1.1**, units, time-zero
semantics, memory limits, checksum details, and the Pygame coordinate mapping.
The legacy scripted smoke format 1.0 is intentionally not accepted by the new
renderer.

Pygame and the browser interpolate **only the recorded pose**, never physics.
Neither renderer may compute contacts, resets, rewards, progress, finishes, or
lap times. Private held-out geometry must never be exported into a public
preview. Checksums are integrity checks, not anti-cheat authentication.

### Pygame renderer

`tools/run_pygame.py` is the active local renderer. It loads only validated
schema 1.1 replays, draws the embedded track boundaries and recorded path,
interpolates the recorded pose for smooth playback, and displays the Python
result/events as telemetry. It never imports or steps the racing environment.
Use `--headless` with dummy SDL drivers for CI and one-frame smoke checks. The
HTML viewer remains an optional dependency-free fallback for remote previews;
Unity has been retired from the active product.

The resizable viewer has clickable playback/speed controls, a drag-to-seek
timeline, recorded-event markers and previous/next-event jumps. Arrows seek one
exact tick, Shift + arrows seek one second, Home/End select the endpoints,
Space toggles playback, and `?` shows help. Seeking pauses. The header labels the
recorded final result separately from the currently selected tick's state.
Separate replay-group v1.0 manifests compare 1–4 recordings on the same Python
track/config/standing-start clock, while single-car schema 1.1 stays unchanged.
Use `tools/build_replay_group.py --help` and `tools/run_pygame.py --group ...`;
shorter recordings explicitly hold their final pose rather than extrapolating.
See `docs/participant_guide.md` for controls/limits and exact reference
resume/warm-start/best-selection commands, and
`reports/pygame_review_2026-10-19.md` for screenshot review evidence.

To save a deterministic paused QA image without a desktop:

```bash
.venv/bin/python tools/run_pygame.py \
  --replay artifacts/replay/pygame-smoke.json \
  --start-seconds 1 --screenshot artifacts/replay/viewer.png
```

### Layer checks and container preview

`make test` runs the configured Python simulation/export/read-back checks. The
participant dashboard test invokes the embedded dashboard JavaScript when Node
is available. Run the complete dependency-free JavaScript smoke suite directly
with:

```bash
node --test tests/*.test.cjs
```

The default checks do not install a browser. The JS checks use a DOM/canvas
test double; Pygame uses dummy SDL. An **optional real Chromium smoke** exercises
custom rays/weights, a tiny reference training run, browser replay, and
390/800/1440px layouts, saving screenshots under `artifacts/ui-polish/`:

```bash
.venv/bin/python -m pip install -r requirements-ui-qa.lock
PLAYWRIGHT_BROWSERS_PATH=.venv/qa-browsers \
  .venv/bin/python -m playwright install chromium --only-shell
PLAYWRIGHT_BROWSERS_PATH=.venv/qa-browsers \
  .venv/bin/python tools/check_ui_browser.py
```

This optional check needs Chromium's Linux system libraries and fonts. The
`ui-qa` Docker target installs them reproducibly without changing the default
participant image:

```bash
docker build --target ui-qa -t synapse-ui-qa .
docker run --rm synapse-ui-qa
```

It downloads QA tools at setup/build time, not during `make test`. Real-display
Pygame interaction is still a separate manual check.

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
