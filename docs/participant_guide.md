# SYNAPSE participant guide

This is the first participant-facing training path for the local prototype. It
is ready for local onboarding and dummy training confirmation. The exact event
budget, final map set, submission format, ranking rule, and final artifact
validator are still open in the event decision register.

## Recommended delivery

The event should publish one immutable participant release, not a moving branch.
The release should contain:

- the inspectable Python simulator and an optional reference trainer;
- public training maps and their manifest;
- the observation/action contract;
- raw transition signals and a participant-owned reward/training boundary;
- a local smoke test and deterministic evaluation helper;
- a versioned wheelhouse for the supported Python 3.11 platform; and
- an optional Docker path for participants who want it.

The ordinary leaderboard should later accept a model artifact and manifest. It
should not execute an arbitrary participant checkout as its scoring authority.
Held-out maps and final scoring must stay in an organizer-owned evaluator.

## First local run

From the repository root, prepare the project environment first:

```bash
python3.11 tools/setup_participant.py
```

Then launch the trainer you own:

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python tools/train_participant.py \
  --trainer examples/participant_train.py \
  --reward examples/baseline_reward.json \
  --steps 2048 \
  --budget-steps 10000 \
  --output-dir participant_run
```

The launcher validates workspace paths and forwards the published inputs to
your trainer. The bundled `examples/participant_train.py` delegates to the
optional reference PPO only so the checkout has a runnable smoke test. Replace
that file with your own training program to own the model architecture,
algorithm, preprocessing, checkpoints, and reward function. A trainer that
supports the dashboard should accept the forwarded CLI options
`--track-dir`, `--config`, `--output-dir`, `--steps`, `--budget-steps`,
`--n-envs`, `--rollout-length`, and `--seed`, and print the documented
`update ...` progress line. `--reward` is forwarded only when a reward input is
provided.
The reference starter writes:

```text
participant_run/
  budget.json                 # external persistent transition ledger
  reference_checkpoint.json   # authoritative committed-generation pointer
  reference_generations/      # immutable snapshots, including optimizer/RNG/env
  training_state.json         # latest committed state (compatibility symlink)
  training_metrics.jsonl      # latest metrics (compatibility symlink)
  training_report.md          # human-readable local dashboard
  policy.pt                   # latest reference weights (compatibility symlink)
  policy.json                 # latest reference metadata (compatibility symlink)
  training_contract.json      # latest schema/preprocessing (compatibility symlink)
```

For the bundled reference starter, run the same command again with the same
`--budget-steps` and output directory to resume its committed state and local
counter. The budget is charged per active environment lane, not per vector call.
The final partial batch is supported. Resets and PPO optimizer updates are not
environment transitions. Custom trainers define their own checkpoint lifecycle.

## Optional reference checkpoints and fine-tuning

These are local reference-PPO conveniences, not requirements on participant
architecture, trainer, reward, preprocessing, or checkpoint format. Use
`tools/run_reference_baseline.py` directly for the options below; the participant
launcher/dashboard does not forward these fine-tuning options.

### Exact resume of the same reference run

There is no `--resume` flag. An existing `reference_checkpoint.json` selects one
validated generation under `reference_generations/g-.../`. On the same supported
CPU runtime, the trainer restores policy weights, Adam optimizer state,
observation-normalizer state, collector observations, environment/lane state,
and Torch/NumPy/Python/environment RNG state at a completed-rollout boundary.
Keep the config/ray order, ordered training tracks, reward file, seed, lane
count, PPO hyperparameters, cumulative budget, preprocessing, selection settings,
and source/runtime identity unchanged. A mismatch is rejected before collection.
`--steps` and `--max-updates` are invocation limits and may change; changing PPO
batch boundaries can change the training trajectory even with exact recovery.

Each generation contains `policy.pt`, `policy.json`, `training_contract.json`,
`training_state.json`, `training_metrics.jsonl`, `selection_report.json`,
`local_state.pt` (optimizer/Torch RNG), `collector.json`, and a file-hash manifest.
The local POSIX-filesystem writer flushes and renames the complete generation,
then atomically replaces the pointer containing its manifest hash. Generations
are write-once; their names do not establish which one is current. The root
policy/state/metrics/contract files are compatibility symlinks switched after
the pointer commit. Recovery follows the pointer and can repair lagging mirrors;
it never guesses from root files or selects another generation after corruption.
One writer may own an output directory at a time.

`budget.json` stays outside the snapshots and is **never restored or rolled
back**. If collection spent transitions but an update/evaluation/publication was
interrupted before committing its pointer, the ledger can be ahead of the last
checkpoint. Exact recovery then refuses to continue; those spent transitions
remain spent. A failure after the pointer commit can instead resume that committed
boundary and repair the mirrors. Do not rewind the ledger or pointer to claim a
refund. These local mechanisms do not settle event-wide budget enforcement.

### Explicit non-exact warm start

Legacy root-only artifacts and imitation artifacts are not exact-resume state.
Pass `--warm-start SOURCE_DIRECTORY` into a **new output directory** to load the
reference `policy.pt`, `policy.json`, and `training_contract.json`. If the source
has an authoritative pointer, its committed generation is validated first.
The source observation/action/config and ordered training-track contract must
match the new environment; this loader is for compatible reference-format
artifacts, not arbitrary participant models.

A warm start creates a fresh optimizer, collector/environment, RNG streams, and
local ledger. Source files are hashed for provenance and left untouched. It
preserves the source input mapping: environment-only preprocessing stays that
way, with no newly fitted normalizer; a saved running normalizer retains its
statistics and is **frozen** throughout fine-tuning and evaluation. Fresh reference
training without a warm start still uses its running reference normalizer.
A new local ledger is not permission to reset a future cumulative event budget.

`--warm-start` enables bounded fine-tuning/selection even without `--fine-tune`.
A new `--fine-tune` run requires a warm start. For example, if this checkout has
the retained local `artifacts/pygame-imitation` artifact, this small experiment
uses all three current public development tracks. That retained artifact is a
local fixture, not a promised participant-release download. Choose a new output
path for each new experiment:

```bash
.venv/bin/python tools/run_reference_baseline.py \
  --warm-start artifacts/pygame-imitation --fine-tune \
  --config config/simulation.json --reward examples/baseline_reward.json \
  --track-dir tracks/train --eval-track-dir tracks/train \
  --steps 16 --budget-steps 35 --n-envs 2 --rollout-length 8 --seed 17 \
  --max-updates 3 --learning-rate 0.0001 --update-epochs 1 \
  --eval-every 1 --eval-max-steps 1200 \
  --output-dir artifacts/reference-finetune-demo
```

Continue that new run **exactly**, without reloading the source artifact, for
the remaining 19 transitions:

```bash
.venv/bin/python tools/run_reference_baseline.py \
  --fine-tune \
  --config config/simulation.json --reward examples/baseline_reward.json \
  --track-dir tracks/train --eval-track-dir tracks/train \
  --steps 19 --budget-steps 35 --n-envs 2 --rollout-length 8 --seed 17 \
  --max-updates 3 --learning-rate 0.0001 --update-epochs 1 \
  --eval-every 1 --eval-max-steps 1200 \
  --output-dir artifacts/reference-finetune-demo
```

The first invocation performs one 16-transition update; the second performs a
16-transition update and a partial 3-transition update. Subsequent exact resume
does not need the source directory. These are training-transition counts;
public-track evaluations also take time, separately from the training budget.

Fine-tuning is limited to 65,536 requested transitions and 1–32 updates per
invocation (`--max-updates` defaults to 4). The actual work stops at the first
of the invocation step limit, update limit, or remaining cumulative budget.
`n_envs * rollout_length` must be at most 8,192. PPO permits at most 16 positive
update epochs and a positive learning rate at most 0.01 (default 0.0001).

### Public-development best selection, not event ranking

The source is evaluated at update zero, then every `--eval-every` **global**
updates (1–32, default 1), preserving cadence across process restarts. A final
off-cadence update is saved but not automatically evaluated. Each evaluation
uses one deterministic fixed-start mean-action run per selected public track,
with frozen preprocessing, no retries, and no training reward. Selection is
lexicographic, in this order:

1. More finished tracks.
2. Less total off-track physical time across all evaluated tracks.
3. Less total physical lap time across finished tracks.

Only strict improvement replaces the best; ties and regressions retain it,
including the original source if no update improves. This is a reference-only
public-development heuristic, **not the event ranking, DNF policy, or evidence
of held-out generalization**. Training return is excluded from the selection key.
`--eval-track-dir` must contain 1–16 unique tracks whose geometry hashes belong
to the bundled public `tracks/train` set. `--eval-max-steps` defaults to 2,400
and permits 1–20,000 steps per track; both an evaluation and the invocation's
projected diagnostic work must fit the 100,000-step cap. These bounded diagnostics
make no training `RunBudget` reservations.

`selection_report.json` records the rule, evaluated-generation history, per-track
physical outcomes, and retained best generation. `best/` links to that immutable
generation and contains its weights/metadata/contract. Root policy files and the
dashboard's **latest saved artifact** follow the latest committed update, which
may be worse than `best/`. Exact continuation always follows the current pointer,
not the selected best. Export a single public track directly from the retained
multi-track best without rewriting its contract:

```bash
.venv/bin/python tools/export_replay.py \
  --weights my-reference-run/best/policy.pt \
  --metadata my-reference-run/best/policy.json \
  --contract my-reference-run/best/training_contract.json \
  --track tracks/train/synthetic_oval.json --config config/simulation.json \
  --participant-id reference-best --output artifacts/replay/reference-best.json
```

Create `artifacts/replay` first. The exporter still checks the saved physics,
observation/ray semantics, and frozen preprocessing. Training-track identity is
provenance, not a restriction against evaluating one public or held-out track.
Never export private held-out geometry to participant-facing files.

## Optional local dashboard

The same workflow has a small dependency-free Python web UI:

```bash
.venv/bin/python tools/participant_ui.py
```

Open `http://127.0.0.1:8766/`. The dashboard provides a local night-racing
console for the same `tools/train_participant.py` command. Its run flow is:

1. Choose your trainer script.
2. Set the nine static signal weights in the **Reward function** panel.
3. Set the ray count and ordered angles in **Forward-facing ray sensors**.
4. Set maps, output directory, steps, cumulative budget, parallel environments,
   rollout length, and seed.
5. Click **Start / resume participant training**.
6. Watch the transition counter, update metrics, and bounded process log. Use
   **Stop after current process boundary** to terminate the trainer.

The dashboard is a loopback presentation/controller layer, not a network
service or a second model, reward engine, simulator, or scoring authority. It
launches the existing trainer and writes the selected configuration beside the
run. After training, use the watch panel:

1. Keep **Output directory** pointed at the selected run. Its saved
   `run_config.json` is required, including the actual ray order used to train.
   CLI users can save their effective config beside the run under that name.
2. Choose **Reference · latest saved artifact**, or **Safe NumPy artifact** and
   enter its workspace-relative artifact directory. The UI never substitutes a
   reference model for a failed participant artifact.
3. Choose a public map and either **Pygame** or **Browser · explicit headless /
   remote**. The model's saved contract/config must match; generic artifact
   contracts and weights are never rewritten.
4. Click **Build / watch replay**. Use **Open browser fallback** to play in this
   browser or **Download replay JSON** to keep the validated recording.

“Latest saved” is not a claim of best-checkpoint selection. The UI's last 32
replay links live for the current server session; saved files remain under the
run's `replays/` directory. Inputs are disabled while a process/action is active,
and validation failures appear inline rather than in blocking alert dialogs.
The server rejects cross-origin actions and non-loopback hosts because it can
launch local code. Do not expose it as a public training/upload service.

The dashboard also has a **Set up / repair Python** button. If the UI detects
that NumPy, Pygame, or the reference trainer's PyTorch dependency is missing, it refuses
to launch a broken environment and shows the setup command instead. The first
run should be intentionally small; repeat with the same output directory only
when your trainer supports resuming.

### Replay viewing

Pygame is the active local replay renderer. It consumes only validated replay
JSON and never steps the environment:

```bash
.venv/bin/python tools/run_pygame.py \
  --replay path/to/saved-replay.json
```

For CI or a machine without a display, render one frame with dummy SDL drivers:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv/bin/python tools/run_pygame.py \
  --replay path/to/saved-replay.json --headless
```

Pygame opens on the **server's local desktop**, not on a remote browser's device.
Without a display, the dashboard reports a **one-frame headless check**, not a
playing window; use its browser link for playback. An SDL failure also leaves
the browser/download links usable. Unity is not part of the active workflow.

The Pygame window has clickable play/pause, restart, speed, event navigation and
a draggable timeline. **Left/Right** select one exact tick, **Shift + arrows**
one second, **Home/End** the endpoints, **P/N** the previous/next recorded event,
and **?** keyboard help. Seeking pauses. It displays recorded telemetry at the
completed tick, while only position/heading are interpolated. Final results are
labeled separately, so an eventual crash is not shown as the current state at
tick zero. The window resizes down to 640×480.

Use `--screenshot artifacts/replay/viewer.png --start-seconds 1` to save a
deterministic paused PNG without a display; its parent directory must exist.

### Replay-group comparison

Use the standalone CLI to compare existing compatible recordings. A group is a
separate **v1.0 manifest** referencing ordinary single-car **schema-1.1** replays;
it does not merge, rewrite, or re-simulate them. Given two compatible recordings
with distinct recorded participant IDs, create a new manifest beside or above
them (the parent directory must already exist):

```bash
.venv/bin/python tools/build_replay_group.py \
  --output artifacts/replay/oval-comparison.group.json \
  --replays artifacts/replay/team-a-oval.json artifacts/replay/team-b-oval.json \
  --labels "Team A" "Team B"

.venv/bin/python tools/run_pygame.py \
  --group artifacts/replay/oval-comparison.group.json
```

CLI input paths are relative to the current directory (absolute CLI locations
also work); stored replay references are relative POSIX paths **below the
manifest directory**. The builder refuses an existing output, including a source
recording or symlink. Choose a new name to rebuild. Labels are optional, with
exactly one per replay when supplied; they do not replace recorded participant
IDs. There is no dashboard group picker or browser-group playback in this slice.
`--group` and the single-recording `--replay` option are mutually exclusive.

All recordings must have identical recorded track ID/geometry hash, engine and
config identity, tick rate, car dimensions, and exact standing-start tick-zero
frame. Tick zero must be on track with zero speed, lateral velocity, yaw rate,
steer, and applied controls. Matching uses the recordings' identities, not today's
source hash, so compatible historical recordings can be grouped together. File
SHA-256, payload hashes, and embedded replay validation must all pass. These
integrity checks do not authenticate provenance or decide event eligibility.

| Group limit | Value |
|---|---|
| Ghosts / recordings | 1–4 distinct participant IDs and files |
| Manifest size | 16 KiB |
| Each replay | 16 MiB; existing limits of 20,001 frames and 20,000 track samples |
| Total input | 32 MiB **including the manifest** |
| Display label | 1–48 printable characters |
| Stored file reference | At most 512 characters; no absolute paths, traversal, or symlinks in files/directories/ancestors |

The builder assigns distinct cyan/gold/violet/green palette colors. One shared
clock runs from the common standing start to the longest recording's last tick.
A shorter ghost **holds its final recorded pose, telemetry, events, and result**;
its last speed need not be zero. It never extrapolates, restarts, or interpolates
past its end. Before the end, only position/heading interpolate; telemetry and
events come from completed integer ticks. Results stay in manifest order,
explicitly **unranked**, and the viewer computes no car interactions or scoring.

| Control | Effect |
|---|---|
| **1–4**, **Tab / Shift+Tab** | Select/cycle a ghost's telemetry and event rail without changing time |
| **V** | Hide/show the selected ghost and its path; telemetry remains selectable |
| **T** | Toggle the selected ghost's path preference independently |
| **P / N**, event-marker click | Seek the selected ghost's previous/next recorded event; seeking pauses |
| **Space**, **R** | Play/pause; restart the whole group and play |
| **Left/Right**, **Shift + arrows** | Seek one exact shared tick; one second |
| **Home/End**, draggable timeline | Seek group endpoints; choose a shared tick |
| **[ / ]**, **− / +** | Change playback speed |
| **H / ?**, **Esc / Q** | Toggle help; close the viewer |

For a deterministic paused headless capture, select a present ghost (numbered
from 1), choose an exact shared tick, and optionally seek one of its events:

```bash
.venv/bin/python tools/run_pygame.py \
  --group artifacts/replay/oval-comparison.group.json \
  --select-ghost 2 --start-tick 0 --seek-event next --show-help \
  --size 640x480 --screenshot artifacts/replay/oval-comparison-help.png
```

`--screenshot` implies `--headless`; its parent must exist. `--start-tick` and
`--start-seconds` are mutually exclusive. `--seek-event` accepts `first`, `last`,
`previous`, or `next` after setting the start position, and fails explicitly if
no matching event exists. Use `--headless` alone for a one-frame render without
saving a PNG. These checks exercise dummy SDL, not a physical desktop session.

## Participant-owned training boundary

The SYNAPSE environment publishes `10 + ray_count + lookahead_count` `float32`
track-relative values. The default is 23 values; the UI lets you choose the
forward-ray portion. It accepts two finite actions in `[-1, 1]`:

1. steering;
2. throttle/brake, where positive is throttle and negative is braking.

Participants own the policy/model architecture, inference distribution,
optimizer, training algorithm, preprocessing, checkpoint format, weights, and
custom training logic. SYNAPSE owns the physics transition, observation/action
semantics, termination, public tracks, raw signal names, evaluation starts, and
scoring environment.

`VectorizedRacingEnv.step_raw` returns the next observation, termination flags,
and raw signals. A participant reward callback may use those signals, the
observation, action, and transition outcome to return one finite scalar.
`examples/participant_reward.py` shows the callable shape only; copy and
replace it in your own trainer workspace.
The normal dashboard path uses the static `RewardConfig` weighted-signal
adapter and writes the canonical `reward_config.json` into the run directory.
A custom participant trainer may instead omit `--reward` and compute its own
finite reward callback. The organizer evaluator never executes participant
reward code.

A participant-owned trainer can use the reusable transition adapter without
accepting its reference PPO implementation:

```python
env = VectorizedRacingEnv(tracks, config, reward=None, n_envs=your_batch_size)
collector = RolloutCollector(
    env,
    your_policy,
    budget,
    run_id,
    reward_function=your_reward,
)
batch = collector.collect(your_transition_count).batch
# Update your own model/optimizer/algorithm from `batch`.
```

The callback receives copies of the observation, action, raw signals, and
terminal flags. It cannot replace or mutate the environment transition.
The action passed to the callback is the action after SYNAPSE's `[-1, 1]`
clipping; the rollout batch also retains the raw policy request for saturation
diagnostics. Keep calls to `RolloutCollector.collect()` bounded (8,192
transitions by default) for predictable memory use.

## Budget and fairness draft

The recommended event rule is one cumulative transition budget across the full
training window. A transition is one accepted action applied to one active
environment lane. Vector lanes and repeated simulator transitions count
individually. Resets and optimizer updates do not count. The local budget file
is tamper-evident but is not independent proof of training provenance because a
participant controls the workspace.

The `10,000` value in the example is an onboarding smoke budget only. It is not
the proposed final scored budget. Research recommends measuring candidate
budgets such as 100k, 250k, 500k, and 1M with equal settings and multiple seeds
before freezing one.

The local launcher and dashboard are cooperative development tools. A custom
trainer can ignore the forwarded budget values unless it uses `RunBudget` and
the environment/rollout APIs; independent event-grade enforcement belongs in
the future organizer evaluator/launcher.

The dashboard’s normal participant path edits the nine published signal
weights and writes a canonical `reward_config.json`; it also edits the ordered
forward-ray list. The ray list must contain 1–16 unique finite angles in
`[-90, 90]` degrees and must include `0`. Ray order is part of the observation
contract, so changing it requires a compatible new run/artifact.

### Signal definitions

The UI weights are applied as:

```text
reward = sum(weight[signal] * signal[transition])
```

These are raw transition signals, not equal-scale scores. Some are dense
per-step values and some are one-shot events. The definitions are:

| Signal | Definition | Typical sign |
|---|---|---|
| `progress` | Signed wrapped change in track arclength divided by track length. Positive means forward progress. | Positive |
| `speed` | Dense speed incentive, scaled by `dt`, above the 5 m/s floor and capped against the configured/55 m/s reference. | Positive |
| `curvature_matched_speed` | `-abs(actual_speed - curvature_target_speed)`, scaled by `dt`; zero means a match and more-negative values mean a mismatch. | Positive weight |
| `off_track` | `dt` while the car is off the drivable track; otherwise zero. | Negative |
| `crash` | One-shot `1.0` on a crash transition; otherwise zero. | Negative |
| `stuck` | One-shot `1.0` when the stuck condition occurs; otherwise zero. | Negative |
| `lap_complete` | One-shot `1.0` when a lap completes; otherwise zero. | Positive |
| `time` | Positive `dt` on every transition. | Negative |
| `steer_smoothness` | Steering-command change multiplied by normalized speed. Larger values mean more abrupt steering despite the historical name. | Negative |

Positive weights generally reward `progress`, `speed`, and `lap_complete`.
Negative weights generally penalize `off_track`, `crash`, `stuck`, `time`, and
`steer_smoothness`. `curvature_matched_speed` is already zero or negative, so
a positive weight penalizes mismatch; a negative weight normally rewards the
wrong behavior. Zero disables a signal. All weights must be finite numbers in
`[-100, 100]`.

The values in `examples/baseline_reward.json` are starting points, not event
tuning. Change one weight at a time and compare per-signal contributions with
return, lap completion, speed, crash rate, and off-track rate. Do not assume a
larger coefficient dominates: an event impulse and a dense `dt`-scaled signal
occur at different frequencies.

### Ray layout and dummy diagram

The dashboard’s dummy car diagram is a configuration preview. It shows the car
heading upward, the ordered ray indices, and each entered angle; it does not
perform physics, detect obstacles, or calculate reward. The cyan line is the
straight-ahead center ray.

- Ray count is limited to **1–16**.
- Angles must be finite, unique, and within `[-90, 90]` degrees.
- The list must include `0`.
- `0°` points straight ahead relative to the car heading.
- Positive angles sweep to the car’s left/counterclockwise; negative angles
  sweep to its right/clockwise.
- The default ordered list is `0, -15, 15, -30, 30, -60, 60, -90, 90`.
- The default ray range is 100 m; each observation ray is the nearest boundary
  distance capped at that range and normalized to `[0, 1]`.
- `ray_0`, `ray_1`, and so on follow the configured list order. The UI and
  server never silently sort the list.

Changing the ray list changes observation size and the observation/config
identity. Start a compatible new run and artifact after changing it.

### UI weights versus custom reward code

**Dashboard weights** are a static `RewardConfig` adapter saved as
`reward_config.json`. They are convenient for quick experiments and
reproducible local reference training. They are not an event scoring authority.

**Custom reward code** remains participant-owned. A custom trainer can call
`RolloutCollector` with `reward_function=...`; the callback receives copies of
the observation, clipped action, raw signals, and termination flags and must
return one finite scalar per transition. Passing
`--reward examples/baseline_reward.json` selects the static weighted adapter;
omitting `--reward` is valid for a trainer that computes its own reward.

## Dummy training confirmations

Use a short deterministic run before beginning larger experiments:

```bash
python3.11 tools/setup_participant.py
.venv/bin/python tools/build_training_manifest.py

.venv/bin/python tools/train_participant.py \
  --trainer examples/participant_train.py \
  --reward examples/baseline_reward.json \
  --steps 16 \
  --budget-steps 16 \
  --n-envs 1 \
  --rollout-length 16 \
  --seed 17 \
  --output-dir participant_smoke
```

Confirm that the output contains the local budget/state/metrics artifacts and
that the console prints an `update ... steps ...` progress line. For the UI
confirmation, launch `tools/participant_ui.py`, choose a small budget such as
16, set a visibly different ray layout such as `-30,0,30`, and edit at least
one reward weight. After the run, confirm that its output directory contains:

```text
run_config.json       # selected ray layout and effective simulation config
reward_config.json    # canonical UI signal weights
run_identity.json     # ray count, observation size, and config/reward hashes
training_state.json   # trainer state and budget identity
training_metrics.jsonl
training_contract.json # ordered observation contract and training provenance
```

The identity file should report observation size 17 for the four lookaheads
plus ten self-state fields and three rays. For the bundled reference trainer,
the selected ray layout is required by the saved observation contract and the
selected reward file is recorded in `training_provenance`. Reusing an output
directory with different rays or weights is rejected; use a new output
directory for a new contract. A custom participant trainer still owns its
algorithm and must honor the forwarded `--config` and `--reward` paths itself.
Finish with:

```bash
make test
```

## Generic model artifact preview

For participants who want to separate training from organizer evaluation, the
prototype includes a declarative NumPy MLP artifact path. It accepts model
weights and architecture metadata, but not participant Python or reward code:

```text
submission/
  manifest.json
  training_contract.json
  weights.npz
```

Use `tools/evaluate_participant.py` or `tools/export_replay.py --artifact-dir`
to exercise the artifact. This preview supports feed-forward MLPs only; it is
not the final event submission contract. A separate ONNX static-vector profile,
exporter, CPU backend, and explicit model-only Docker worker are now available
as release candidates; see [submission_runtime.md](submission_runtime.md).
Container verification, outer job supervision, and event approval are still
required before accepting untrusted submissions. The local dashboard/preview
is not a sandbox.

### Physical evaluation diagnostics

Both reference and safe-artifact evaluators return `diagnostics` version `1.0`
beside the outcome. Compare these audits with training return before deciding
that a driver improved:

- `off_track_duration_s`, longest contiguous excursion, positive progress while
  off track, and maximum boundary excess reveal runoff-heavy behavior;
- `backward_progress_m`, stalled-step counts and streaks reveal reversal/stall;
- `sector_coverage_mask/count/fraction` use the engine's 16 spawn-relative
  reached sectors. Reset alone gives no coverage; a stationary transition can
  reach sector zero. Coverage is not proof of forward completion;
- `sector_first_crossing_ticks` has 16 entries for the first **forward** crossing
  of each successive lap-length fraction, including the finish fraction. A
  missing crossing is `null`. Backward recrossings earn no new split, and all
  thresholds crossed during one transition retain the same integer tick;
- saturation counts raw action components with magnitude at least one, while
  mean action deltas use the clipped applied command. Channels are separate;
- elapsed/off-track time is ticks times `dt`, never wall-clock time. The
  off-track-progress fraction is `null` when no positive progress exists.

These values do not change observations, rewards, termination, or ranking and
are not inserted into replay schema 1.1. Evaluate each track separately; the
evaluator rejects a randomly sampled track pool. Repeated deterministic hashes
test repeatability, not unseen-track generalization. The generic multi-track
CLI already accepts repeated `--track` options and reports every track without
inventing a leaderboard formula.

## Maps

The current public set contains five development maps: three original
synthetic fixtures plus Monaco and Spa reference layouts. The
manifest records both the canonical geometry hash and the delivered-file hash.
Before public release, the organizers must still:

- select the final public route families;
- prepare organizer-only development and final holdouts;
- run difficulty-parity and generalization experiments;
- red-team repository history, artifacts, previews, and renderer assets for leaks;
- record provenance, licenses, and contributor rights; and
- freeze the public release ID before the participant training window.

Do not place held-out geometry, seeds, previews, or derived renderer assets in the
participant repository, its history, release assets, or public replays.

## Open event decisions

The following remain discussion items rather than hidden implementation choices:

- cumulative budget enforcement model: local honor system, signed launcher, or
  organizer retraining;
- exact scored transition budget;
- final public/private map counts and identities;
- model artifact format, operator/runtime contract, and submission validator;
- reward-code packaging, checkpoint, and budget-enforcement rules;
- deterministic evaluation starts, retries, DNF scoring, and ranking; and
- supported native platforms and offline wheelhouse contents.
