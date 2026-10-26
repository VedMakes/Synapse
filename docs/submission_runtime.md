# Model-only submission runtime (release candidate)

This is an implemented **engineering profile**, not a declaration that event
rules or the public release are frozen. Participants still own their model,
algorithm, weights, preprocessing, and training reward. The optional fixed PPO
model is only an example. Scoring never imports participant Python or reward
callbacks. Do not submit a trainer, pickle, Torch checkpoint, or Docker image to
the model-only evaluator.

## Install and export locally

Prepare the ordinary environment using the README, then install the optional
CPU inference dependencies into the same project environment:

```bash
.venv/bin/python -m pip install -r requirements-evaluator.lock
make test
```

The ONNX checks skip explicitly if these optional dependencies are absent.
Default checks never download tools, models, or tracks. The dependency lock is
not an offline wheelhouse; offline first-time installation still needs a
platform-matched bundle, which has not yet been rehearsed.

For the optional reference model, export a trusted saved artifact directory:

```bash
.venv/bin/python tools/export_onnx.py \
  --reference --source my-reference-run/best --output submission
```

For the existing declarative NumPy artifact, omit `--reference`:

```bash
.venv/bin/python tools/export_onnx.py --source my-numpy-artifact --output submission
```

The output must be a new directory. Weights are not retrained. Frozen source
normalization or affine preprocessing is embedded in the graph, not applied
again by the evaluator. The reference actor's float64 normalization followed by
float32 conversion is preserved. Source artifacts are unchanged. Participants
may instead export their own supported architecture directly; the converter is
not a requirement or an architecture prescription.

## Build a model-only ONNX submission package

The participant dashboard's **Build ONNX submission** button performs the export
and assembles a new directory containing the validated `submission/`, matching
`config/simulation.json`, bounded run provenance, `package_manifest.json`
hashes, and a README. It deliberately contains no track and no replay. The
live runner supplies the event or local-test track/config separately, and the
ONNX model produces fresh actions during the heat. The same operation is
available without the dashboard:

```bash
.venv/bin/python tools/package_live_submission.py \
  --model-only \
  --source participant_run \
  --output artifacts/onnx-packages/my-model \
  --participant-id team-a
```

The output directory must be new. The package does not copy participant
training code, Torch checkpoints, reward callbacks, tracks, or replay files
into the model submission. Use the package's `submission/` directory as the
entrant path when starting a live runner.

## ONNX profile v1

Exactly three regular, non-symlink files are accepted:

```text
submission/
  manifest.json
  training_contract.json
  model.onnx
```

`manifest.json` has exactly these fields:

| Field | Required value |
|---|---|
| `format` | `synapse_onnx_policy` |
| `format_version` | integer `1` |
| `profile` | `static_vector_cpu_opset17_v1` |
| `environment_code_hash` | lowercase SHA-256 of the named engine sources |
| `model_sha256` | SHA-256 of exact `model.onnx` bytes |
| `training_contract_sha256` | SHA-256 of exact contract bytes |

The canonical `TrainingContract.to_dict()` binds ordered fields, ray angles,
physics/config hash, action semantics, deterministic standing start, and
training-track provenance. Its separate normalizer must be null. Training-track
identity does not prohibit evaluation on unseen tracks. Physics and observation
semantics must still match the evaluator; custom rays require the corresponding
organizer-approved configuration. Hashes are integrity checks, not signatures,
training-budget proof, or authorization to choose event settings.

Runtime boundaries:

- ONNX standard opset **17**, IR ≤10; ONNX **1.17.0**, ONNX Runtime **1.20.1**.
- One input `observations`: float32 `[1,N]`, `1 ≤ N ≤ 256`.
- One output `actions`: float32 `[1,2]`, steering then throttle/brake in `[-1,1]`.
- Final node must be `Tanh` producing `actions`. Input/output must be finite.
- Allowed operators: `Gemm`, `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`,
  `Relu`, `Tanh`, `Sigmoid`, `Clip`, `Identity`, `Cast`.
- Only finite Gemm coefficients, boolean transpose flags, and float32/float64
  Cast attributes are accepted. Other attributes are rejected.
- Static dense/residual DAGs are supported, not arbitrary ONNX programs.
  Tensors are float32/float64, rank ≤2, dimensions 1–1024. Rank-two intermediate
  activations must retain batch size one; weights may be matrices.
- Maximum 16 MiB model, 256 KiB per JSON file, 128 nodes, 256 initializers,
  512 declared intermediate values, 1,000,000 parameters, 1,000,000 elements per
  tensor and 4,000,000 combined parameter/intermediate elements.
- External/segmented/sparse tensors, custom domains/functions, training graphs,
  subgraphs, random operators, strings, dynamic shapes, and separate data files
  are rejected before constructing the inference session.
- CPU provider only, sequential execution, one inter/intra-op thread, graph
  optimization and memory arena/pattern disabled, deterministic-compute setting.
  Repeated local inference is tested; cross-hardware bitwise equality is not
  promised. Organizer hardware/runtime must be fixed for a scored event.

The existing `synapse_numpy_mlp` artifact remains supported for local migration
and tests. It is not silently declared the final event submission format.

## Trusted local preview

Only use in-process loading for your own trusted artifacts:

```bash
.venv/bin/python tools/evaluate_participant.py \
  --artifact-dir submission --config config/simulation.json \
  --track tracks/train/monaco.json --repeats 2 \
  --output artifacts/participant-evaluation.json
```

The report labels this `trusted_local_preview`. `tools/export_replay.py
--artifact-dir submission` also accepts ONNX for local public-map playback.
Neither path is a sandbox. The dashboard's existing artifact picker is labeled
for NumPy; use the CLI for the explicit ONNX workflow.

## Isolated organizer evaluation

Build the organizer-controlled, model-only image from trusted source:

```bash
docker build --target evaluator -t synapse-evaluator:v1 .
SYNAPSE_TEST_DOCKER=1 SYNAPSE_TEST_DOCKER_IMAGE=synapse-evaluator:v1 \
  .venv/bin/python -m unittest tests.test_isolated_policy -v
```

The evaluator target starts from a clean Python image, **not** the participant
runtime stage. It copies only the evaluator lock, public Python modules, and
worker script. No tracks, configs, reports, runs, reward callbacks, source-tree
mount, Docker socket, participant scripts, or private organizer files are
included. Do not customize this image with private inputs. Before release,
inspect its layers and pin/publish the actual base and evaluator image digests;
the development Dockerfile still uses a mutable base tag.

Run the host-side Python evaluator with explicit isolation:

```bash
.venv/bin/python tools/evaluate_participant.py \
  --isolate --worker-image synapse-evaluator:v1 \
  --artifact-dir submission --config config/simulation.json \
  --track tracks/train/monaco.json --repeats 2 \
  --worker-memory-mb 512 --startup-timeout 30 --action-timeout 2 \
  --output artifacts/isolated-evaluation.json
```

For private evaluation, organizer-controlled `--track`, `--config`, and report
paths replace the public examples. Keep reports private: track IDs/hashes and
physical diagnostics belong to the organizer. Never export private geometry to
a replay, dashboard, screenshot, or participant support log.

The controller snapshots only the three bounded model files into an ignored
workspace directory. Directory-FD reads reject symlinks at every level,
nonregular files, extra files, and oversized inputs. It bounds host-side
normalizer allocation before parsing metadata; weights/ONNX native parsing
occurs only inside the worker. The worker validates and acknowledges the exact
snapshot digest. Later source edits cannot alter that worker's input.

## Run several ONNX models as a race

The non-renderer race interface evaluates one to four ONNX artifacts on the
same explicit track set. Each entrant gets a fresh fixed-start environment and
the deterministic mean-action path. Entrants do not interact or collide. This
is the Python-authoritative ghost race used to compare lap outcomes and, when
requested, produce replay-group inputs for the read-only presentation tools.

The default is a trusted local preview. Use it only with artifacts you own:

```bash
.venv/bin/python tools/run_onnx_race.py \
  --entrant team-a submissions/team-a \
  --entrant team-b submissions/team-b \
  --track tracks/train/monaco.json \
  --track tracks/train/spa.json \
  --config config/simulation.json \
  --repeats 2 \
  --output-dir artifacts/onnx-races/oval-flowing \
  --emit-replays
```

The output directory must be new. It contains `race.json`. With
`--emit-replays`, it also contains one schema-1.1 replay per entrant and one
v1.0 replay-group manifest per track:

```text
race-output/
  race.json
  tracks/00-monaco/*.json
  tracks/01-spa/*.json
```

The report binds each result to the artifact, ONNX model, and training
contract hashes. It records all repeated outcomes and fails if a repeated
trajectory hash differs. It intentionally has no leaderboard score or DNF
aggregation because those are event-policy decisions still open in the
handoff. `total_reward` is not a race score.

For organizer-grade evaluation, select the model-only Docker worker explicitly:

```bash
.venv/bin/python tools/run_onnx_race.py \
  --isolate --worker-image synapse-evaluator@sha256:<organizer-digest> \
  --entrant team-a submissions/team-a \
  --entrant team-b submissions/team-b \
  --track private/tracks/heldout-01.json \
  --config private/simulation.json \
  --output-dir private/race-results/heldout-01
```

Do not pass `--emit-replays` for private tracks: replay files embed complete
track geometry. The runner is bounded to four entrants, 32 explicit tracks,
eight repeats, and two million planned transitions. These are operational
limits, not final event scoring rules. Docker availability and the outer
organizer job supervisor remain release gates.

## Live inference and actions

The live heat path is separate from replay comparison. It calls every ONNX
policy during the race, applies the returned actions to the authoritative
Python environment, and publishes post-transition state to a read-only
WebSocket client. The initial live profile uses 30 Hz physics and control,
one action per physics tick, a 20 ms candidate policy budget, and
`strict_scored` late-action handling.

Install the optional live dependencies:

```bash
.venv/bin/python -m pip install -r requirements-live.lock
```

Run a live two-entrant heat:

```bash
.venv/bin/python tools/run_live_onnx.py \
  --entrant team-a submissions/team-a \
  --entrant team-b submissions/team-b \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --output-dir artifacts/live/oval
```

The command prints a loopback endpoint such as
`ws://127.0.0.1:8766/live/v1`. Serve `renderer/` separately and open:

```text
http://127.0.0.1:8765/?live=ws://127.0.0.1:8766/live/v1
```

Each control cycle is: policy observation, concurrent fresh ONNX actions,
finite/bounds/deadline validation, one fixed Python physics step, then one
complete state packet. Policy sessions are loaded and warmed before tick zero.
The browser receives `hello`, latest-only `state`, and reliable `end` packets;
it cannot send actions or alter the race. Slow browser clients can lose
intermediate state packets without changing the simulation.

`strict_scored` marks an entrant `dnf_inference` on a late, missing, invalid,
or failed action. `trusted_live` may hold the previous action for one control
interval and is intended only for local visual demonstrations. The report is
`live-race.json`; it records tick rates, policy deadline, latency percentiles,
deadline misses, inference failures, and per-entrant trajectory hashes. The
live heat remains independent ghosts, not car-to-car interaction.

The launched worker has:

- No network, read-only root filesystem and model mount, non-root UID/GID 65534.
- All capabilities dropped, no privilege escalation, one CPU, 512 MiB default
  memory with matching swap cap, 64 process/thread and open-file limits, no core
  dumps, no shared IPC, and a 16 MiB noexec/nosuid temporary filesystem.
- No automatic image pull. Use only an organizer-built image; preferably pass
  its immutable digest via `--worker-image`.
- JSONL observations/actions only, 16 KiB message cap, strict increasing integer
  sequences, finite/shape/action-bound checks **before** narrowing to float32.
- Default 30-second startup and 2-second per-action deadline. Timeout, EOF,
  oversized/malformed replies, or invalid actions fail closed and clean up the
  worker/snapshot; there is no substitute policy or fabricated DNF result.

The memory option permits 128–4096 MiB and deadlines `(0,300]` seconds. These
are bounded operational defaults, **not approved scoring/latency rules**.
Physics remains in the trusted host process with `reward=None`; no participant
reward executes. The report records the selected image, limits, artifact/config
and engine hashes, and repeated physical outcomes. A failed worker aborts the
evaluation; final penalties/retry/appeal policy is still an organizer decision.

The Docker daemon must see the checkout at the same absolute path as the
controller because the frozen model is bind-mounted. The model copy stays
inside the workspace for host-Docker setups; private track directories are
never mounted. Docker access itself is privileged: do not expose its socket to
participants or run scoring on a personal machine containing secrets. Use a
dedicated patched organizer host and an outer job/service supervisor for total
job lifetime, cancellation, and cleanup after controller/host failure. Containers
are a restriction layer, not a proof against all native-runtime/kernel exploits.

**Verification status:** local real-process JSONL and failure tests pass, but
Docker CLI/socket are unavailable in this workspace session. Container builds,
actual cgroup restrictions, layer inspection, and real isolated inference remain
unverified. The opt-in test above must pass before accepting untrusted models.
