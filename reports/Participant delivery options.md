# Participant delivery options for the college RL racing event

**Research-note basis:** `research_notes/Participant delivery options/packaging.md`, `fairness.md`, and `maps.md`
**Research date in the notes:** 2026-10-17
**Scope:** participant delivery, held-out map preparation, minimal training workflow, environment-step fairness, model-complexity limits, memory-conscious operations, and a staged release plan.

## Executive decision

Use a **hybrid participant delivery model**:

1. Publish a **public, immutable participant release** containing the inspectable Python environment, public training maps, schema, examples, tests, baseline trainer, model exporter/validator, documentation, and a release manifest.
2. Include a **versioned Python wheel/source package and offline wheelhouse** so a participant can install from local files without depending on a package index.
3. Offer **Docker/Dev Container as an optional path**, and use an organizer-owned pinned image for evaluation. Do not require Docker for the default participant path.
4. Accept a **model artifact only** for the default leaderboard—preferably ONNX after its runtime and operator subset are implemented and frozen. The organizer evaluates it with a fixed simulator, fixed CPU runtime, and private held-out tracks.
5. Keep **held-out geometry, final scoring code, evaluator configuration, signing material, and leaderboard orchestration outside the participant release**.
6. If arbitrary participant code is an educational objective, run it as a separate division in an organizer-owned sandbox with explicit resource and network limits. Do not make participant-provided Docker images or arbitrary Python the default scoring boundary.

This gives students a source-level learning experience while making the score depend on an organizer-controlled evaluation, rather than on a participant repository, local replay, executable, or self-reported score. A public repository is therefore the learning interface; the model artifact and private evaluator are the competition interface.

## 1. Evidence from the research notes

### 1.1 Delivery and evaluation are different trust boundaries

The packaging notes distinguish the artifact used to **deliver training** from the system trusted to **score submissions**. A GitHub repository, wheel, executable, or Docker image can make local setup easier, but none should be treated as the scoring authority when a participant can alter or supply it. If the evaluator executes participant-controlled Python, a participant wheel, an executable, or a participant image, that submission must be treated as untrusted.

The strongest boundary identified in the notes is:

> model-only artifact + organizer-owned fixed evaluator + private held-out tracks + no participant code execution during ordinary scoring.

Competition precedents support this separation:

- Kaggle separates public competition data from private rerun-scoring data and treats submissions as versioned artifacts.
- Codabench accepts either result-only submissions or code bundles; when code is needed, the organizer supplies the competition image and controls the worker/container layout.
- AWS DeepRacer on AWS validates uploaded models, queues evaluation jobs, uses private evaluation workflows, and derives public results from server-side state.
- MineRL limits interaction, uses controlled containers and hardware, restricts extra simulator information, and uses held-out evaluation/retraining procedures.

These precedents do not prescribe an identical implementation for this event. They provide evidence for the boundary pattern: **participants may train locally; the organizer chooses the final data, runtime, resources, and score**.

### 1.2 Reproducibility is a release identity, not just a repository URL

A reproducible race needs one immutable release identity covering at least:

- exact Python version and supported OS/architecture;
- environment package version and dependency hashes;
- observation and action schema versions;
- physics, reward, termination, time-step, and evaluation configuration;
- public training-map manifest and private held-out-map identifier;
- seed/start/evaluation procedure;
- evaluator image/package identity;
- model format, ONNX IR/opset policy, names, shapes, and dtypes; and
- submission hash and evaluation result.

The notes also identify several current-project implications:

- `.python-version` and the README identify Python 3.11, while `pyproject.toml` currently permits Python `>=3.10`. A competition release should either test every declared version or narrow the event contract to the tested Python 3.11 target.
- `requirements.lock` includes `-e .`, which is useful for development but is not an ideal offline release primitive. A release should install a versioned wheel or source distribution from a local wheelhouse.
- The current Dockerfile uses the mutable `python:3.11-slim` tag. The evaluator release should record a base-image digest and publish the resulting image digest.
- A Python virtual environment is disposable and generally not portable. Distribute a recipe plus local artifacts or a container image; do not distribute `.venv` as the environment.

### 1.3 Environment-step accounting can be made explicit

Gymnasium defines one `step(action)` as one environment transition and separates `terminated` from `truncated`. MineRL counts environment interactions and explicitly counts skipped frames when frames are stacked or actions are repeated. Stable-Baselines3 documents that PPO rollout size is `n_steps * n_envs` and that `total_timesteps` can be a lower bound, so a library training argument should not be the authoritative competition counter.

The useful unit for this event is therefore:

> One environment transition is one accepted action applied to one active environment lane, producing the next observation, reward, termination/truncation result, and diagnostic record.

This means vectorized calls are charged per active lane, repeated simulator frames are charged individually, resets are not charged, and optimizer updates are not charged. A final partial vector batch must use only the remaining transitions rather than running a full batch and reporting an overrun afterward.

The notes favor a **cumulative budget across the full training window**. A per-run allowance with unlimited restarts does not define a meaningful sample budget because participants can buy additional experience by restarting. A participant-owned hash chain can show accidental changes, but cannot independently prove training provenance when the participant controls the entire workspace. Strong enforcement requires an organizer-controlled launcher/service or organizer retraining.

### 1.4 Generalization needs a route-family split, not merely more files

The map notes support original, procedurally authored 2D geometry as the lowest-risk authoritative source. Procedural-generation research supports varying environment distributions, but a weak generator can produce cosmetic variants of one latent layout. The split must therefore be by complete route family and correlated geometry signature, not by random files or points sampled from the same route.

The notes also identify a direct racing risk: a policy can perform well on one track yet fail on unseen curve shapes. Public tracks should therefore cover different corner families, while private tracks should be difficulty-matched but not near-duplicates.

### 1.5 Reward is not the score, and reward proxies can be exploited

The racing evidence describes wall-grinding and stationary behavior caused by imperfect progress and wall penalties. Reward-hacking research generalizes the same problem: optimizing a proxy can increase the proxy while reducing the intended objective.

The event should expose a constrained training reward surface, but rank using organizer-computed physical outcomes such as completion, fixed-step lap time, DNF reason, crashes, and off-track behavior. The participant’s return and local replay checksum should not be authoritative for ranking.

### 1.6 Model format and architecture affect both fairness and security

The current project’s planned ONNX boundary is preferable to loading untrusted PyTorch pickle/state-dictionary files. PyTorch documentation warns that `torch.load` uses Python pickle by default and warns against loading untrusted data. ONNX is a smaller and more inspectable computation-graph boundary, but it is not automatically safe: external tensor data, custom operators, unsupported opsets, large tensors, and resource exhaustion remain possible.

The fairness notes do not identify a universally fair architecture. They identify a practical v1 choice: fixed observation/action schema, fixed MLP architecture, fixed preprocessing and normalization, fixed inference runtime, and no recurrent state, frame stacking, test-time adaptation, external lookup tables, filesystem, network, subprocess, or dynamic code loading.

## 2. Comparison of delivery options

The ratings below assume a college audience, local/offline training, a Python 3.11 event target, private final tracks, and server-side scoring.

| Option | Participant experience | Reproducibility/offline use | Fairness and secrecy | Organizer burden | Recommended role |
|---|---|---|---|---|---|
| **Full public GitHub cloneable repository** | Familiar, inspectable, easy to fork, and strong for teaching | Medium to high when pinned to an immutable tag/commit; source alone does not package Python/native dependencies | Low secrecy because public maps, code, and reward behavior are visible; acceptable only when the server owns scoring | Low to medium | Canonical learning and local-training interface |
| **Partial public participant release derived from the current repository** | Similar to the full repository, but with clearer public/private boundaries and fewer operational leaks | High when it is a clean, versioned public snapshot plus wheelhouse; must be tested as a release, not just as a commit | Best balance: public API/training maps are visible while final maps, evaluator, and scoring remain private | Medium to high because two synchronized release paths must be maintained | **Recommended public architecture** |
| **Standalone executable/compiled file** | Lowest Python setup friction for a supported target; less inspectable and harder to customize | High after download for one tested OS/CPU target; weak across Windows/macOS/Linux and CPU architectures | Does not provide a security boundary; a binary can still be arbitrary code and PyInstaller bundles can be reverse-engineered | High because each target needs builds, smoke tests, signing/notarization, and dependency notices | Optional launcher, local validator, replay viewer, or demo |
| **Python wheel/source package alone** | Standard for Python users but requires interpreter and packaging knowledge | Good with pinned Python, hashes, and platform-specific wheelhouses; a wheel is not a complete runtime | Good for delivering organizer-controlled code; unsafe as a trust boundary if participant packages are installed by the evaluator | Medium | Stable API component inside the release bundle |
| **Docker image/Dockerfile alone** | Reproducible for Docker-capable participants, but virtualization and disk requirements add friction | High when the image is digest-pinned and available offline; Dockerfile alone is not an offline artifact | Strong for an organizer-owned evaluator image; containers are not an absolute boundary for arbitrary submissions | High for image building, storage, security, and platform support | Optional development path, evaluator image, or separate code division |
| **Hybrid: public source + package/wheelhouse + optional container + model-only submission** | Provides a simple native path and optional advanced paths without forcing Docker | High when every artifact shares one release ID and is tested offline | Strongest practical default: only a validated model crosses into the private evaluator | Medium to high, but each artifact is bounded and versioned | **Recommended event model** |

### 2.1 Full public GitHub repository

**Advantages:** lowest conceptual friction, transparent code, useful issues/examples/history, and high educational value. GitHub release assets can provide the source archive, wheel, hashes, and instructions. Participants can inspect the simulator and learn from the baseline rather than treating the environment as a black box.

**Risks:** all public training maps, reward paths, and implementation details are inspectable. That is acceptable for an open educational event but does not prevent map-specific overfitting. A public repository is also not a sandbox: executing code from a submitted checkout can expose files, use the network, alter scoring, or consume unbounded resources.

**Implementation rule:** tell participants to download a named release or checkout a full commit hash. Do not instruct them to clone a moving `main` branch. Audit the complete history, branches, tags, issues, CI artifacts, release archives, and package/container layers before publication. Deleting a private map in a later commit does not repair a public leak.

### 2.2 Partial commit or partial public release of the current repository

The phrase “partial commit” needs a strict interpretation. A commit in the current repository is **not** a secrecy boundary if held-out files exist elsewhere in that repository’s branches, tags, history, LFS objects, CI artifacts, Docker layers, or release assets. Sparse checkout, `.gitignore`, and a later deletion do not solve that problem.

The safe form is a **curated public participant snapshot or separate public repository** produced from the current codebase after a leakage review. It contains only what participants need to train:

```text
public participant release
  public simulator/API and trainer boundary
  public development/training maps
  examples, tests, documentation, and validator
  model export/validation client
  no held-out geometry, private scorer, answer keys, or secrets

organizer release
  same event release ID
  private held-out maps and starts
  evaluator and ranking policy
  worker/resource configuration
  submission intake, audit, and signing material
```

This is the best architecture for the event because the public code can remain transparent while secrecy is concentrated in the data and evaluation orchestration that actually need protection. The public and private releases must share an API/schema ID and pass a compatibility smoke test before submissions open.

### 2.3 Standalone executable or compiled file

An executable can remove Python installation and dependency-resolution problems, which is useful for a short demo or a non-Python participant. It is a poor canonical RL environment because PyInstaller-style bundles are target-specific, require separate builds for OS/Python/architecture combinations, and can encounter native-library, antivirus, code-signing, macOS notarization, Linux `glibc`, and temporary-extraction failures. Large PyTorch bundles also create significant download and storage cost.

It does not protect source or scoring logic reliably: bundled Python bytecode can be decompiled, and a signed executable proves build identity rather than safe behavior. It should therefore be an optional convenience artifact with `--version`, a build manifest, SHA-256, supported targets, and dependency notices—not the default submission format or anti-cheat mechanism.

### 2.4 Hybrid model

The hybrid model uses each format where it is strongest:

- GitHub/source for inspectability and teaching;
- a wheel plus local wheelhouse for repeatable native installation;
- Docker for maintainers, advanced participants, CI, and the organizer evaluator;
- an optional executable for a launcher or replay viewer; and
- a small model artifact for ordinary leaderboard scoring.

The key operational rule is that these are not independent products. Every artifact must declare the same `event_release` and execute the same public contract tests. Otherwise, “multiple delivery options” become multiple subtly different environments.

## 3. Recommended participant delivery model

### 3.1 Public release contents

Publish a release such as `event-2026.1` with:

```text
participant-release-2026.1/
  README.md
  LICENSE
  THIRD_PARTY_NOTICES/
  RELEASE.json
  SHA256SUMS
  participant-source.tar.gz
  racing_event-<version>-py3-none-any.whl
  requirements-offline-<platform>.txt
  wheelhouse/<platform>/*.whl
  examples/
  tracks/train/*.json
  tools/validate_model.py
  tools/run_local_evaluation.py
  tests/
  Dockerfile                         # optional participant path
  .devcontainer/                     # optional convenience layer
```

`RELEASE.json` should identify the Python version, supported platform targets, package version, schema IDs, public-map manifest hash, commands, model format, model size limit, and evaluator compatibility ID. `SHA256SUMS` should cover every distributed archive and wheel. A campus mirror or USB/offline archive is useful for event-day availability.

The default participant path should be native Python from local artifacts. The release should also provide a matching Docker image digest or image tar for participants who prefer containers. Do not make participants build a wheel just to use the environment; give them the tested wheel. Keep the source tree available for students who want to modify the trainer or simulator.

### 3.2 Default submission boundary

The ordinary leaderboard should accept:

```text
submission.zip
  model.onnx
  manifest.json
```

The manifest should contain:

- `event_release` and schema ID;
- model format and supported ONNX IR/opset;
- exact input/output names, shapes, and dtypes;
- model SHA-256 and byte count;
- producer/runtime version;
- architecture ID;
- optional training seed information, marked informational unless rules use it;
- optional model license/attribution; and
- a participant/team identity supplied by the submission system rather than trusted from the file.

The evaluator should reject malformed or unexpected artifacts before inference: wrong release, invalid graph, unsupported operator, custom operator, external tensor path, unexpected output, dynamic shape unless explicitly allowed, NaN/Infinity, oversized tensors, oversized file, or duplicate/extra files. It should run with no network, no participant filesystem access, fixed CPU/thread settings, fixed memory/wall-time limits, and a fresh worker or process per submission.

If ONNX is not ready by the pilot, use a narrowly validated temporary artifact boundary only for internal testing. Do not describe a future ONNX contract as implemented. The submission format, supported graph subset, inference runtime, and failure behavior must be frozen before the scored round.

### 3.3 Separate code-submission division

If the educational objective requires students to submit source or a callback, make it a separately scored division. Accept a source archive or constrained entrypoint, never a participant Dockerfile/image or Docker socket. Run it in an organizer-owned pinned image with:

- no network;
- no host mounts, privileged mode, host PID/IPC namespaces, or Docker socket;
- non-root user and reduced capabilities;
- read-only application/private-track mounts;
- CPU, memory, disk, process, file-descriptor, wall-time, and output limits;
- explicit working directory and allowed files;
- fresh worker/container/VM per submission;
- archive traversal, malware, secret, and bundled-data checks; and
- a reproducible appeal/re-run process keyed to the exact artifact and evaluator digest.

This division increases organizer cost and should not silently change the default leaderboard’s fairness model.

## 4. Map preparation and held-out-track protection

### 4.1 Authoritative map strategy

Use **original, procedurally authored 2D geometry** as the event’s authoritative source. The current repository is close to this model: `tools/track_builder.py` creates original synthetic loops, and the track schema carries centerline, arclength, heading, curvature, widths, boundaries, runoff, barriers, start line, metadata, and content hash.

Use a pinned and reviewable family of centerline primitives or splines. Expose meaningful, bounded parameters such as lap scale, curvature/radius, straight-section length, corner sequence, width, runoff, barrier policy, and start position. Store generated outputs, not only seeds. A public generator is useful for experimentation, but a private final seed is not secret if the generator, parameters, previews, or seed space make it recoverable.

Real circuits may inspire broad characteristics, but the safest event policy is fictional names, original geometry, and original visual identity. If using licensed geographic input, preserve the source, version/date, query, license, attribution, transformation, and output hash. Do not copy or closely redraw a recognizable circuit, use official names/signage/assets, or assume that educational use removes IP obligations. OpenStreetMap is not blanket clearance for an exact named circuit; ODbL and other rights must be reviewed separately.

### 4.2 Recommended split

Use the following as a launch candidate, subject to the map-count experiments below:

| Split | Initial recommendation | Purpose |
|---|---:|---|
| Public training | **3–5 maps** | Participant development and learning; use five if the budget can support variety, three if the budget is small |
| Organizer-only development holdout | **1–2 maps** | Internal map, reward, and budget selection; do not use it for participant tuning |
| Final private holdout | **2–3 maps** | Scored generalization; keep geometry and exact starts private until the event is complete |

The public set should include distinct route families: for example, a high-speed/sweeper map, a technical/hairpin map, a chicane or linked-turn map, and a mixed map. The fifth public map, if used, should be a distinct family rather than a cosmetic duplicate. At least one final map should come from a family not represented by a near-duplicate public route.

Split by whole route family. Keep smoothed, mirrored, rescaled, seed-neighbour, and other derived variants in the same split. Cluster procedural tracks by geometry signature before splitting. Do not let a long public map dominate step exposure merely because it produces more transitions unless that weighting is an explicit published rule.

### 4.3 Canonical format, hashes, and manifests

Use the existing small, human-reviewable JSON format as the canonical published and hashed artifact. Generate binary caches such as NPZ only as derived local speedups; never let a renderer or trainer silently choose a different resampling, closure, unit, or coordinate transform.

Record two map identities:

1. **Canonical content hash:** SHA-256 of canonical JSON with the embedded content-hash field excluded during calculation.
2. **Delivered-file hash:** SHA-256 of the exact bytes shipped in the release or private evaluator package.

Each map manifest should include:

```text
manifest_version
event_release_id
geometry_version
map_id
split: train | dev | final_private
canonical_content_hash
delivered_file_sha256
units and coordinate convention
generator/build version
validator version and result
source/license record ID
```

For final maps, keep the clear name, seed, parameter tuple, preview, SVG, Unity prefab, derived centerline, and detailed difficulty summary out of the public manifest. A private hash commitment can show that a map was fixed, but it does not make a candidate route secret if participants already possess or can infer it.

### 4.4 Validation gates

A map is ready only after all five gates pass:

1. **Geometry:** finite values, valid schema/version, no duplicate or zero-length segments, valid closure, simple non-self-intersecting centerline, consistent arclength, positive widths, non-crossing nested boundaries/barriers, valid runoff, curvature continuity, and a feasible car footprint.
2. **Physics:** projection, raycasts, edge status, barrier/runoff behavior, and one complete lap work across the closure seam and representative positions.
3. **Determinism:** the same map/config/seed/action stream produces the same trajectory digest on the pinned evaluator.
4. **Difficulty:** a reference controller and fixed baseline are run on every candidate, using the same vehicle, action limits, starts, seeds, and step limit. Compare completion, lap time, crash/off-track/stuck rates, and route-feature distributions.
5. **Delivery and secrecy:** canonical and delivered hashes agree with the manifest; private geometry does not appear in public history, releases, packages, container layers, logs, replays, screenshots, Unity assets, or browser bundles.

As an initial engineering gate, investigate candidates that differ by roughly more than 10% in median reference performance or have materially different failure tails. This is a tuning threshold, not a universal research law. Freeze the final maps and manifest before participant training begins.

### 4.5 Secrecy red-team

Have a reviewer with only participant access inspect the final public release after tagging. Check:

- all branches, tags, deleted-file history, LFS objects, release archives, and CI logs/artifacts;
- issues, discussions, pull requests, caches, and notebooks;
- Docker layers and wheel/package contents;
- Unity scenes, prefabs, baked meshes, screenshots, thumbnails, and videos;
- debug messages, exception text, map IDs, and replay export paths; and
- public generator code and its seed space.

Keep private maps outside participant-visible repositories, history, release assets, public object storage, and public replay previews. Copy the private bundle to the evaluator through an organizer-controlled path immediately before evaluation and verify its hashes at startup. If production needs a preview, use a coarse difficulty card or cropped/abstract view that does not reveal route topology.

## 5. Minimal participant-facing training workflow

The workflow should have one supported path and a short success criterion: **a new participant can run the baseline, consume a known step budget, export a model, validate it locally, and submit it without network access after downloading the release.**

### Participant steps

1. **Download one release.** Use `event-2026.1` or its immutable commit/archive, not `main`. Download the matching native bundle or optional container image for the participant’s target.
2. **Verify the release.** Check `SHA256SUMS`; read `RELEASE.json`; confirm Python version, platform, event release, and model schema.
3. **Create a fresh environment.** Use an exact Python 3.11 patch release and a new virtual environment. Do not copy a `.venv` from another machine.
4. **Install offline.** Install the event wheel and dependencies from the local wheelhouse with no package index, no build step, and no network requirement. The release should provide the exact tested command.
5. **Run the smoke test.** Validate the package, load one public map, step a short deterministic episode, and run the baseline through the local validator. The smoke result should report the release ID, schema, map hash, and runtime.
6. **Read the contract, not internal state.** The policy receives only the versioned observation array. `info` is diagnostic unless a field is explicitly part of the observation contract. Reset, seed, `terminated`, `truncated`, action bounds, normalization, and step semantics are fixed.
7. **Start from the baseline.** Train the reference PPO/MLP on public maps with the fixed trainer. Use the declarative reward signal/weight configuration if reward design is the intended creative variable.
8. **Track one cumulative meter.** The launcher displays `steps_used`, `steps_remaining`, `run_id`, `release_id`, and reward-config ID. Checkpoint restores, reward edits, and process restarts continue the same counter. A crash recovery procedure must not reset it.
9. **Evaluate locally.** Use fixed public starts and deterministic mean actions. Treat local results as advisory; they are not the leaderboard score.
10. **Export and validate.** Export the approved model format, attach architecture/normalization/schema metadata, run malformed-input and shape checks, and confirm finite in-range actions on a fixed observation batch.
11. **Package one submission.** Submit only `model.onnx` and `manifest.json` unless a separate code division explicitly allows more. The manifest contains the event release and model hash.
12. **Receive an independent score.** The server validates the artifact, loads private tracks only in the evaluator, runs fixed starts and seeds, records per-track outcomes and resource use, and recomputes the rank.

The README should include a “first 15 minutes” path, a troubleshooting table for wrong Python/version/hash/network errors, and a clear explanation that local reward and replay results are not authoritative.

## 6. Fair environment-step budgets

### 6.1 Recommended accounting rule

For a cumulative budget `B`, charge as follows:

| Event | Charge |
|---|---:|
| One scalar `env.step(action)` | 1 transition |
| Vector call with `m` active lanes | `m` transitions |
| One repeated action for `r` simulator transitions | `r` per active lane |
| One skipped frame in frame-stack/action-repeat logic | 1 |
| Reset, including automatic post-terminal reset | 0 |
| Policy forward pass without stepping the environment | 0 |
| Optimizer/gradient update | 0 |
| Inactive padding lane in a final vector batch | 0 |

Reserve or debit the transitions before executing them. The terminal transition counts; the following reset does not. A failed setup before the first accepted transition should not consume budget, but a transition executed before a crash remains charged.

### 6.2 Cumulative policy

Use one persistent budget across the entire training window. Reward changes, checkpoint restores, process restarts, and hyperparameter changes cannot reset it. “Runs” may remain as reporting segments, but `cumulative_steps` is authoritative.

If training remains entirely local, this is a declared rule plus a useful tamper-evident log, not strong proof of provenance. A local participant can delete and regenerate an unsigned log. For stronger enforcement, choose one of these models explicitly:

- **Organizer retraining:** participant submits source/reward configuration, and the organizer trains under `B` in a fixed worker. Strongest budget enforcement, highest cost and queueing.
- **Organizer-controlled training launcher/service:** the participant trains through a signed or organizer-anchored runner that owns the counter. More accessible than retraining but requires service infrastructure.
- **Self-training honor system:** participant trains locally and submits a model; the cumulative budget is a rule, logs are retained for audit, and finalists may be audited or retrained. Lowest burden, weakest provenance.

The recommended default is model-only self-training with an explicit honor-system/audit policy for the first event, plus optional source retraining or code audit for finalists. Do not claim that a participant-owned hash chain proves training provenance.

### 6.3 Concrete budget proposal

The exact `B` must be selected by measurement; there is no universal fair number. The notes recommend testing equal-budget baselines and generalization rather than copying another event’s number. The following is a practical starting schedule for this small 2D environment:

| Use | Provisional budget | Purpose |
|---|---:|---|
| Smoke/onboarding | **10,000 transitions** | Validate installation, schema, export, and meter behavior; no ranking |
| Pilot/qualification | **250,000 transitions** | Compare setup, reward, map variety, and seed variance |
| Scored event candidate | **1,000,000 transitions per team** | Initial main-event candidate for a fixed two-layer MLP and 3–5 public maps |

The **1,000,000-transition value is a provisional implementation choice, not evidence that this is the universally correct budget**. Before freezing it, run at least five seeds for candidate values such as 100k, 250k, 500k, and 1M with equal optimizer settings, fixed architecture, public-map training, organizer-only development maps, and one untouched private split. Choose the smallest budget at which:

- the baseline reliably completes the public training set;
- held-out development performance is approaching its plateau rather than still increasing sharply;
- the final/private generalization gap is acceptable;
- seed variance and failure tails are manageable; and
- the event can process every submission within the organizer’s compute and memory envelope.

If 1M does not produce meaningful learning, increase by a documented factor rather than silently changing the budget. If 250k already saturates the baseline and generalizes, use the smaller budget to reduce compute and reward careful experimentation.

Do not rank by wall-clock training time, GPU hours, renderer FPS, optimizer updates, participant reward, or local replay checksum. Hardware speed may affect how quickly a participant uses its allowed transitions, but it must not buy more transitions.

## 7. Model-complexity constraints

### Recommended v1 constraint: fixed architecture

Use the current fixed policy architecture for the first scored division:

- 23-value `float32` observation;
- two 64-unit `tanh` hidden layers;
- two-dimensional bounded action output in `[-1, 1]`;
- fixed action mapping and preprocessing/normalization;
- deterministic policy mean during evaluation; and
- no recurrent state, frame stacking, external memory, test-time adaptation, or lookup table.

For the actor graph alone, this is approximately 5.8k trainable weights before any separate training-only value head or distribution parameters. The exact exported graph and metadata, not a participant-supplied architecture string, define the accepted artifact.

Fix the following in the event contract:

- input/output names, shape, dtype, and rank;
- action range, clipping, squashing, and longitudinal-control semantics;
- activation functions and distribution/mean-action behavior;
- parameter count, model byte size, and supported graph operators;
- inference device, thread count, and per-step timeout;
- ONNX IR/opset and runtime version;
- static versus dynamic dimensions;
- whether external tensor data is forbidden; and
- whether custom operators, random-generation operators, filesystem, network, subprocess, and dynamic loading are forbidden.

This keeps the competition focused on reward design, training behavior, and generalization rather than on memory capacity, unsupported runtime tricks, or hardware-specific latency.

### If architecture freedom is a required learning objective

Make it a separately scored division or a later event release. Prefer a small explicit architecture allowlist to arbitrary ONNX graphs. Validate graph topology, parameter count, tensor sizes, operator allowlist, artifact size, CPU memory, and worst-case inference latency. A parameter cap alone is insufficient because two graphs with the same parameter count can have different operators, temporary tensors, and latency.

The exact allowlist, parameter cap, byte cap, and timeout remain open decisions. A reasonable pilot-only starting point is a small set of static MLPs with no recurrence or custom operators and a conservative parameter/size cap, measured on the exact evaluator CPU. Do not let the allowed model vary the step-count fairness rule: one accepted environment transition remains one charged transition regardless of policy architecture.

## 8. Memory-conscious operational guidance

### Participant side

- Treat the wheelhouse and Docker image as **disk artifacts**, not objects to load into RAM. Provide one platform-specific bundle rather than asking participants to cache every OS/architecture wheel.
- Keep training lanes within a documented memory envelope. More vectorized lanes may improve wall-clock throughput but must not improve the step allowance because every active lane is charged.
- Stream trajectory digests and bounded diagnostics. Do not retain all observations, frames, replays, or episode objects in memory when a running hash and summary metrics are sufficient.
- Write checkpoints atomically to disk and keep a bounded number of recent checkpoints. The cumulative budget state must be durable before a transition batch is considered committed.
- Load one canonical map or a bounded map cache at a time when possible. Avoid materializing every procedural variant, every derived renderer mesh, and every replay in one process.
- Use the fixed CPU inference path for local validation so that memory and numerical behavior resemble the organizer evaluator.

### Organizer side

- Run one fresh evaluator worker/process per submission, or use strictly bounded concurrency based on measured peak RSS. Enforce memory, CPU, wall-time, process, file-size, and output limits.
- Stream per-track results and trajectory digests. Retain the compact result matrix, audit hashes, failure reason, and selected diagnostics; do not accumulate full trajectories for every submission unless needed for a dispute.
- Load private geometry only inside the worker that needs it and avoid copying all tracks, render meshes, public previews, and submission artifacts into every process.
- Reject ONNX external data and custom operators in v1. Bound graph dimensions and tensor allocations before inference to prevent memory-amplification inputs.
- Keep canonical JSON as the source of truth and derive caches; do not maintain separate hand-edited Python, Unity, raster, and binary versions that can drift or multiply storage.
- Use bounded queues and explicit back-pressure for uploads/evaluation. Preserve the original submission bytes on disk with a hash; do not keep unbounded copies in memory.
- Publish an offline bundle with hashes and a campus mirror/USB path so participants do not all repeatedly download large PyTorch/Docker artifacts from the same service.

## 9. Provisional v1 implementation choices

These are the recommended working defaults for a first release. They are not all settled event policy until the decision register is approved.

| Area | Provisional choice |
|---|---|
| Public delivery | Clean, tagged participant release derived from the current repository; never a moving branch |
| Runtime | Python 3.11, with the exact patch version recorded in the release |
| Native install | Versioned wheel plus platform-specific offline wheelhouse and hashes |
| Container | Optional participant Docker/Dev Container; organizer evaluator image pinned by digest |
| Public maps | Original procedural/hand-authored 2D JSON; 3–5 public training maps |
| Private maps | 2–3 route-family-balanced held-out maps in an organizer-only package |
| Map identity | Canonical JSON hash plus delivered-byte SHA-256 |
| Submission | ONNX model plus manifest; no participant code for the ordinary leaderboard |
| Evaluation | Organizer-owned Python simulator, fixed CPU/runtime, fixed starts/seeds, frozen normalization, deterministic mean actions |
| Observation | Current 23-value track-relative `float32` contract; no absolute position, track ID, or absolute progress |
| Action | Current two-value bounded action contract in `[-1, 1]`, with semantics frozen in the release |
| Architecture | Current two 64-unit `tanh` MLP for v1; architecture search deferred or separated |
| Reward freedom | Declarative signal/weight DSL only; no arbitrary raw Python reward in v1 |
| Budget | Cumulative accepted transitions; vector lanes and skipped/repeated frames charged individually |
| Budget candidate | 10k smoke, 250k pilot, 1M scored-event candidate, then freeze from measured results |
| Score | Python-authoritative completion/lap-time/DNF/physical outcomes, not training return |
| Logs | Append-only sequence/hash chain with organizer-held signature or anchor for strong claims |
| Licenses | Explicit starter-code license, third-party notices, map provenance ledger, and participant-artifact usage terms |

## 10. Major open decisions

These should remain visibly open in the event decision register rather than being inferred from local code or quietly changing during the event.

| Decision | Why it matters | Current direction |
|---|---|---|
| Self-training, organizer retraining, or signed launcher | Determines whether the step budget is enforceable or honor-system | Start with self-training plus finalist audit; decide whether stronger enforcement is affordable |
| Exact scored budget `B` | Controls opportunity, compute cost, and event difficulty | Pilot 100k–1M candidates; 1M is only a provisional starting point |
| Budget persistence/recovery | Determines how crashes, resumes, and reward edits are treated | One cumulative counter; define recovery and failed-setup rules explicitly |
| Exact ONNX support | ONNX is planned in the current project but not yet implemented | Freeze IR/opset/operators/runtime before public scoring; use no future contract as a current guarantee |
| Fixed architecture versus allowlist | Controls capacity, latency, and validation burden | Fixed current MLP for v1; architecture-search division later if desired |
| Model size/latency/memory caps | Needed for any architecture freedom or arbitrary code | Measure on evaluator CPU and publish before submissions |
| Exact public/private map counts | Affects learning coverage and secrecy | Start with 3–5 public and 2–3 private; select with map-count and variety experiments |
| Held-out starts and repeats | Predictable fixed starts can be overfit; hidden retries are unfair | Predeclare deterministic suite and retry/aggregation rule |
| Ranking and DNF formula | Can dominate model quality | Publish per-track outcomes, aggregate, tie-break, and DNF rule before training |
| Reward DSL and mutable weights | Raw Python expands attack surface and reproducibility burden | DSL-only v1; decide whether weights may change within the cumulative budget |
| Runoff/wall-contact policy | Penalties can create wall-grinding or stationary exploits | Measure physical diagnostics first; freeze any penalty or terminal rule before release |
| Supported native platforms | Each platform needs a tested wheelhouse and offline setup | Minimize the supported matrix or clearly label optional paths |
| Repository boundary | A public current-repo history may leak private assets | Clean public snapshot/repo; red-team after final tag |
| Map/IP policy | Exact real circuits and external assets may need permission or attribution | Prefer original fictional geometry; clear every external source individually |
| Log retention and appeals | Disputes require reproducible artifacts without exposing private maps | Define retention, participant-visible diagnostics, correction window, and rerun owner |
| Participant artifact rights | The organizer needs permission to store, execute, audit, and display results | Publish a limited event license while preserving participant ownership |

## 11. Concrete staged release plan

### Stage 0 — Freeze the event contract

**Outputs:** decision register, release ID scheme, public/private boundary, scoring draft, license plan, and supported-platform decision.

- Decide the ordinary leaderboard submission boundary and whether an optional code division exists.
- Resolve Python version/patch, observation/action schema, normalization, action distribution, fixed architecture, and reward surface.
- Define the transition counter, cumulative persistence, checkpoint/recovery rule, and candidate budgets.
- Define held-out-map access, evaluator ownership, log retention, appeals, and participant artifact rights.

**Gate:** no hidden policy or implementation detail may be required to train or submit locally.

### Stage 1 — Prepare maps and release artifacts

**Outputs:** public training maps, organizer-only development/final candidates, canonical/delivered hashes, wheelhouse, source archive, optional image, and release manifest.

- Author or generate original maps from a pinned, reviewable process.
- Cluster by route family; create train/development/final splits.
- Run geometry, physics, deterministic, reference-policy, and difficulty-parity checks.
- Build the wheel and platform wheelhouses from a clean environment; record hashes and notices.
- Build the organizer evaluator image from pinned inputs and record its digest.

**Gate:** every artifact identifies the same release ID and passes the public baseline compatibility test.

### Stage 2 — Private rehearsal and memory/security testing

**Outputs:** test report, peak-memory/resource envelope, evaluator audit procedure, and leakage-red-team report.

- Set up a fresh participant machine with network disabled for every supported native target.
- Load the optional Docker image from an offline tar and run the same smoke test.
- Submit valid, malformed, oversized, wrong-schema, custom-op, external-data, NaN/Infinity, archive-traversal, duplicate, and timeout cases.
- Repeat the same model evaluation and compare result/trajectory hashes.
- Verify worker death, queue recovery, duplicate submissions, appeals, and leaderboard publication.
- Red-team repository history, CI, release assets, wheel metadata, Docker layers, replays, thumbnails, and Unity/browser outputs for private-map leakage.

**Gate:** no private map is reachable from a participant artifact or public diagnostic; peak memory and timeout limits are measured rather than guessed.

### Stage 3 — Public alpha / no-stakes onboarding

**Outputs:** tagged participant release, first-15-minutes guide, setup support, and issue triage log.

- Publish only public training maps and the participant evaluator/validator.
- Give participants the 10k smoke budget and a baseline submission path.
- Collect failures by platform, Python patch, disk/RAM size, Docker availability, and offline setup.
- Fix only documented release defects; issue a new release ID for contract changes.

**Gate:** a new participant can install, run the baseline, export a model, and pass local validation without network access after acquiring the bundle.

### Stage 4 — Pilot and freeze

**Outputs:** selected budget, final map set, frozen scoring rules, evaluator digest, and participant rules.

- Compare 100k/250k/500k/1M equal-budget baselines over at least five seeds.
- Compare map counts and route-family variety with identical total transitions.
- Use organizer-only development maps for selection and keep the final private split untouched.
- Freeze held-out map files, private manifest, starts, retry count, DNF handling, aggregation, tie-break, model limits, and evaluator image.
- Publish the final public manifest and evaluation protocol without publishing private geometry or inferable previews.

**Gate:** no remaining open decision can change a participant’s training opportunity or score after the scored window opens.

### Stage 5 — Scored tournament

**Outputs:** immutable submissions, validation logs, per-track result matrix, rankings, and private audit record.

- Accept model artifacts through an authenticated intake and record original bytes, size, timestamp, team identity, release ID, and SHA-256.
- Validate the graph and manifest before loading private tracks.
- Run every accepted artifact in a fresh, resource-limited evaluator with the same CPU/runtime, starts, seeds, and track order.
- Repeat a selected run for deterministic integrity; treat a mismatch as an evaluator error requiring investigation.
- Recompute scores from physical outcomes and publish only permitted diagnostics.

### Stage 6 — Post-event publication and maintenance

**Outputs:** public results, release retrospective, and next-event change list.

- Publish the public release manifest, evaluator identity, aggregate/per-track results, and allowed audit summaries.
- Decide whether and when to release held-out maps and final geometry hashes.
- Document setup failures, memory peaks, scoring incidents, appeals, and budget observations.
- Treat security rebuilds or schema changes as new release IDs; never silently replace an artifact used by scored submissions.

## 12. Source ledger

The source notes state that the web sources below were accessed on **2026-10-17**. Publication/update dates are included where the notes recorded them. The report uses the notes’ source set; URLs should be rechecked before the event rules are frozen.

### Packaging, repositories, and installation

- Python 3.11 `venv` documentation (page updated 2026-10-01): https://docs.python.org/3.11/library/venv.html — accessed 2026-10-17.
- PyPA packaging projects tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/ — accessed 2026-10-17.
- PyPA `pyproject.toml` specification: https://packaging.python.org/en/latest/specifications/pyproject-toml/ — accessed 2026-10-17.
- PyPA binary distribution/wheel specification: https://packaging.python.org/en/latest/specifications/binary-distribution-format/ — accessed 2026-10-17.
- PyPA licensing guidance / PEP 639: https://packaging.python.org/en/latest/guides/licensing-examples-and-user-scenarios/ — accessed 2026-10-17.
- pip repeatable installs, hashes, and wheelhouses: https://pip.pypa.io/en/stable/topics/repeatable-installs/ — accessed 2026-10-17.
- pip requirements-file format: https://pip.pypa.io/en/stable/reference/requirements-file-format/ — accessed 2026-10-17.
- GitHub repository model and visibility: https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories — accessed 2026-10-17.
- GitHub source-code archives: https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives — accessed 2026-10-17.
- GitHub repository visibility caveats: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility — accessed 2026-10-17.
- GitHub licensing a repository: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository — accessed 2026-10-17.
- GitHub releases: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases — accessed 2026-10-17.
- GitHub managing releases: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository — accessed 2026-10-17.
- GitHub Large File Storage: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage — accessed 2026-10-17.
- GitHub repository duplication/mirroring: https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository — accessed 2026-10-17.
- PyInstaller operating modes and bytecode caveats: https://pyinstaller.org/en/stable/operating-mode.html — accessed 2026-10-17.
- PyInstaller usage: https://pyinstaller.org/en/stable/usage.html — accessed 2026-10-17.
- PyInstaller requirements/platform notes: https://pyinstaller.org/en/stable/requirements.html — accessed 2026-10-17.
- PyInstaller troubleshooting: https://pyinstaller.org/en/stable/when-things-go-wrong.html — accessed 2026-10-17.
- Docker build best practices and digest pinning: https://docs.docker.com/build/building/best-practices/ — accessed 2026-10-17.
- Docker Engine security: https://docs.docker.com/engine/security/ — accessed 2026-10-17.
- Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/ — accessed 2026-10-17.
- Docker Desktop installation/platform terms: https://docs.docker.com/desktop/setup/install/ — accessed 2026-10-17.
- Docker image save: https://docs.docker.com/reference/cli/docker/image/save/ — accessed 2026-10-17.
- Docker image load: https://docs.docker.com/reference/cli/docker/image/load/ — accessed 2026-10-17.
- Docker reproducible-build timestamps: https://docs.docker.com/build/ci/github-actions/reproducible-builds/ — accessed 2026-10-17.
- Development Container Specification: https://containers.dev/implementors/spec/ — accessed 2026-10-17.
- ONNX IR specification: https://github.com/onnx/onnx/blob/main/docs/IR.md — accessed 2026-10-17.
- ONNX Runtime documentation: https://onnxruntime.ai/docs/tutorials/ — accessed 2026-10-17.
- PyTorch serialization semantics (created/updated 2026-05-08): https://docs.pytorch.org/docs/2.14/notes/serialization.html — accessed 2026-10-17.
- PyTorch `torch.load` warning: https://docs.pytorch.org/docs/2.14/generated/torch.load.html — accessed 2026-10-17.

### Competition and evaluation practice

- Kaggle CLI repository: https://github.com/Kaggle/kaggle-cli — accessed 2026-10-17.
- Kaggle competition commands: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competitions.md — accessed 2026-10-17.
- Kaggle competition creation, public/private data, and frozen solution: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competition_creation.md — accessed 2026-10-17.
- Kaggle tutorials and versioned submissions: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/tutorials.md — accessed 2026-10-17.
- Codabench participant result/code submissions: https://docs.codabench.org/latest/Participants/User_Participating-in-a-Competition/ — accessed 2026-10-17.
- Codabench organizer competition image: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Competition-docker-image/ — accessed 2026-10-17.
- Codabench submission process and workers: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Process-Overview/ — accessed 2026-10-17.
- Codabench submission-container layout: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Docker-Container-Layout/ — accessed 2026-10-17.
- Codabench participant routing: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Participant-Routing/ — accessed 2026-10-17.
- Codabench source repository: https://github.com/codalab/codabench — accessed 2026-10-17.
- AWS DeepRacer on AWS implementation guide (publication date September 2026): https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/ — accessed 2026-10-17.
- AWS DeepRacer on AWS source repository: https://github.com/aws-solutions/deepracer-on-aws — accessed 2026-10-17.
- AWS DeepRacer README/architecture: https://github.com/aws-solutions/deepracer-on-aws/blob/main/README.md — accessed 2026-10-17.

### RL APIs, reproducibility, and fairness

- Gymnasium environment API: https://gymnasium.farama.org/api/env/ — accessed 2026-10-17.
- Gymnasium custom-environment guide: https://gymnasium.farama.org/introduction/create_custom_env/ — accessed 2026-10-17.
- Gymnasium time-limit handling: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ — accessed 2026-10-17.
- Gymnasium vectorized environments: https://gymnasium.farama.org/introduction/speed_up_env/ — accessed 2026-10-17.
- Stable-Baselines3 PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html — accessed 2026-10-17.
- Stable-Baselines3 RL tips: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html — accessed 2026-10-17.
- Stable-Baselines3 vector environments and `VecNormalize`: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize — accessed 2026-10-17.
- PyTorch reproducibility notes (created/updated 2026-05-14): https://docs.pytorch.org/docs/2.14/notes/randomness.html — accessed 2026-10-17.
- NIST FIPS 180-4, Secure Hash Standard (published 2015-08; update noted 2024-07-25): https://csrc.nist.gov/pubs/fips/180-4/upd1/final — accessed 2026-10-17.
- RFC 6962, Certificate Transparency (published 2013-06): https://www.rfc-editor.org/rfc/rfc6962 — accessed 2026-10-17.
- NIST SP 800-57 Part 1 Rev. 5 (published 2020-05): https://csrc.nist.gov/pubs/sp/800/57/pt1/r5/final — accessed 2026-10-17.
- MineRL 2021 challenge rules: https://www.aicrowd.com/challenges/neurips-2021-minerl-diamond-competition/challenge_rules — accessed 2026-10-17.
- MineRL competition starter template: https://github.com/minerllabs/competition_submission_template — accessed 2026-10-17.
- MineRL competition report, arXiv v3 (2021-01-19): https://arxiv.org/abs/1904.10079 — accessed 2026-10-17.
- MineRL 2019 competition page: https://www.aicrowd.com/challenges/neurips-2019-minerl-competition — accessed 2026-10-17.
- Patterson et al., *Empirical Design in Reinforcement Learning*, arXiv v2 (2024-10-29): https://arxiv.org/abs/2304.01315 — accessed 2026-10-17.
- Fuchs et al., *Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning*, arXiv v2 (2021-05-09): https://arxiv.org/html/2008.07971v2 — accessed 2026-10-17.
- Skalse et al., *Defining and Characterizing Reward Hacking*: https://arxiv.org/html/2209.13085v2 — accessed 2026-10-17.
- Cobbe et al., *Leveraging Procedural Generation to Benchmark Reinforcement Learning*: https://arxiv.org/html/1912.01588v2 — accessed 2026-10-17.
- Czechmanowski et al., *On learning racing policies with reinforcement learning*, arXiv v1 (2025-04-03): https://arxiv.org/html/2504.02420v1 — accessed 2026-10-17.

### Maps, formats, and rights

- OpenStreetMap Foundation copyright/license: https://www.openstreetmap.org/copyright — accessed 2026-10-17.
- Open Data Commons ODbL 1.0 legal code: https://opendatacommons.org/licenses/odbl/1-0/ — accessed 2026-10-17.
- ASAM OpenDRIVE (version 1.9.0 release noted as 2026-05-19): https://www.asam.net/standards/detail/opendrive/ — accessed 2026-10-17.
- ASAM Quality Checker Framework: https://www.asam.net/standards/asam-quality-checker/ — accessed 2026-10-17.
- F1TENTH custom map format: https://f1tenth-gym.readthedocs.io/en/stable/customized_usage.html#custom-map — accessed 2026-10-17.
- O’Kelly et al., F1TENTH, PMLR 123:77–89 (2020): https://proceedings.mlr.press/v123/o-kelly20a.html — accessed 2026-10-17.
- CARLA maps and navigation: https://carla.readthedocs.io/en/latest/core_map/ — accessed 2026-10-17.
- CommonRoad `commonroad-io` README: https://github.com/CommonRoad/commonroad-io/blob/master/README.md — accessed 2026-10-17.
- WIPO copyright: https://www.wipo.int/copyright/en/ — accessed 2026-10-17.
- WIPO copyright FAQ: https://www.wipo.int/en/web/copyright/faq-copyright — accessed 2026-10-17.
- U.S. Copyright Office general FAQ: https://www.copyright.gov/help/faq/faq-general.html — accessed 2026-10-17.
- Formula 1 guidelines: https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt — accessed 2026-10-17.
- Formula 1 legal notices: https://www.formula1.com/en/information/legal-notices.7egvZU48hzrypubGBNcQKt — accessed 2026-10-17.

### Local project documents noted in the research

The notes also consulted the current repository’s `README.md`, `RACING_EVENT_Agent_Handoff.md`, `docs/state_schema.md`, `docs/replay_contract.md`, and related research notes. These are repository-local evidence, not external source URLs. This report does not edit application code or those documents.
