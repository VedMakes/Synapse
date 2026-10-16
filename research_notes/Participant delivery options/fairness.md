# Participant-facing fairness controls for a fixed-budget RL competition

**Research date:** 2026-10-17
**Scope:** A participant-facing fairness contract for a fixed-budget reinforcement-learning racing competition. The focus is environment-step budgets, cumulative versus per-run accounting, tamper-evident logs, model architecture limits, observation/action contracts, dependency and runtime pinning, reproducible and deterministic evaluation, submission verification, and resistance to reward or environment exploits.

**Evidence convention:** Statements labelled **Evidence** summarize an external source or an observed project state. Statements labelled **Recommendation** are engineering or event-design proposals for this competition. Statements labelled **Open** require an organizer decision; they should not be silently converted into implementation defaults.

**Process note:** The requested `deep-research/references/researcher.md` path was attempted first, as requested, but it is not visible in the sandbox/container filesystem. The matching deep-research skill was loaded, and this note records the source URLs and access date so the work remains auditable.

## Executive synthesis

### Evidence

1. **An RL step is a contract boundary.** Gymnasium defines `step(action)` as one environment transition and returns the next observation, reward, separate `terminated` and `truncated` flags, and diagnostic `info`. It requires a reset after an episode ends. Its modern API separates task termination from external truncation, which matters for value bootstrapping.
2. **A sample budget can be made explicit.** The MineRL competition rules count interactions with the environment, count skipped frames when frame stacking or action repetition is used, and run submissions in controlled containers. The MineRL report describes training and evaluation from scratch on a prespecified hardware platform and held-out data/environment pairs.
3. **Training and evaluation must be separate.** Stable-Baselines3 recommends a separate test environment because training exploration is stochastic, recommends deterministic actions for inference, warns that wrappers can change evaluation results, and recommends multiple evaluation episodes for an estimate. Its PPO documentation describes `n_steps * n_envs` rollout accounting and notes that `total_timesteps` is a lower bound for PPO.
4. **Bitwise reproducibility has a scope.** PyTorch explicitly says complete reproducibility is not guaranteed across releases, commits, platforms, or CPU/GPU execution. It documents seeding Python, NumPy, and PyTorch, and deterministic algorithm controls for a fixed platform, device, and release.
5. **Hashes detect changes; they do not establish provenance by themselves.** NIST describes message digests as detecting whether messages changed. Certificate Transparency uses append-only Merkle trees, signed tree heads, and auditor consistency proofs to make log equivocation and history changes detectable. A local participant-owned checksum is therefore an integrity aid, not independent proof that training occurred under the declared budget.
6. **Reward is a proxy and can be exploited.** The Gran Turismo racing report describes wall-grinding under a progress proxy and reports that a fixed wall penalty could also produce stationary behavior. Reward-hacking research formalizes the general risk that an imperfect proxy can increase while the intended objective decreases. MineRL rules also prohibit extracting extra simulator information and restrict interaction to the provided `step` interface.

### Recommendation in one paragraph

Publish a versioned contract in which the budget is a **cumulative count of accepted environment transitions across all runs, reward edits, and restarts**, with one charge per active environment lane per transition. Count repeated/skipped simulator frames individually, do not charge resets, and stop exactly at the remaining budget. Enforce the hard limit in the frozen runner; make the log tamper-evident with an append-only sequence, hash chain, and organizer-held signature or external anchor. For v1, use a fixed architecture, fixed observation/action schema, fixed CPU evaluation container, frozen normalization statistics, deterministic mean actions, private held-out tracks, and an independent outcome scorer based on lap time/DNF rather than participant reward. Keep rewards editable through a constrained DSL, and audit reward terms and physical outcomes separately. The cumulative-budget policy, exact architecture, final submission format, ranking formula, and trust model for local training remain organizer decisions.

## 1. Fairness model and threat boundaries

### Evidence

- The project handoff states that participants train on their own machines while evaluation runs on shared hardware, and that environment steps rather than hours are intended to define fairness.
- MineRL provides a relevant competition precedent: it specifies an interaction limit, a hardware ceiling, containerized execution, a held-out round, and organizer-side retraining/evaluation. Its rules require interactions through the provided Gym `step` function and prohibit additional information extraction from the simulator.
- The current repository is Python-authoritative for physics, observations, rewards, termination, and evaluation. The renderer/replay layer is intended to consume recorded poses and events rather than recompute outcomes.

### Recommendation

Separate four properties that are often conflated:

| Property | What it protects | Suitable control |
|---|---|---|
| **Equal opportunity to collect experience** | Faster machines or unlimited restarts do not buy more simulator data | Cumulative environment-transition budget |
| **Equal task definition** | Participants cannot train against different physics or privileged data | Frozen release, contract version, environment/track/config hashes |
| **Equal scoring procedure** | Random action sampling, wrapper differences, or local hardware do not affect rank | Organizer-controlled deterministic evaluation runner |
| **Auditability** | Later disputes can be investigated | Append-only logs, signed release manifests, repeatable evaluation hashes |

Do not use wall-clock training duration, GPU hours, renderer FPS, participant-reported reward, or a participant-generated replay checksum as the ranking metric. Wall-clock time may be documented for usability, but it is a poor fairness unit when participants use different machines.

### Open

1. Is this event a **self-training submission** event, where a participant submits weights after local training, or an **organizer-retraining** event, where the participant submits source and the organizer trains it under a controlled budget? The second model gives stronger budget enforcement but increases organizer cost and runtime variability.
2. Which adversary is in scope: accidental mistakes, ordinary reward experimentation, intentional local budget bypass, malicious submission code, or all of these? The enforcement design depends on the answer.
3. How long should logs, artifacts, and evaluation trajectories be retained, and which parts will be shown to participants after ranking?

## 2. Environment-step budgets

### 2.1 Define the unit precisely

### Evidence

- Gymnasium defines a single call to `step(action)` as one environment transition.
- MineRL’s Research Track rules define the budget in interactions with the environment and explicitly state that skipped frames still count when stacking observations or repeating actions.
- SB3’s PPO documentation defines rollout size as `n_steps * n_envs` and says that `total_timesteps` is a lower bound. A library-level training parameter should therefore not be treated as the authoritative competition counter.
- Gymnasium’s vectorization guidance explains that several independent environments can be stepped together. A vectorized call is not automatically one sample: it can contain several independent transitions.

### Recommendation: transition accounting contract

Use this definition in the participant guide:

> One **environment transition** is one accepted action applied to one active environment lane, producing one next observation, reward, termination/truncation result, and info record. The budget counts environment transitions, not Python calls, policy batches, episodes, seconds, or optimizer updates.

For a budget `B`:

| Event | Budget charge |
|---|---:|
| One scalar `env.step(action)` | 1 |
| One vector call with `m` active lanes | `m` |
| One action repeated for `r` simulator transitions | `r` per active lane |
| One skipped frame in a frame-stack/action-repeat scheme | 1 |
| Reset, including automatic reset after termination | 0 |
| Policy forward pass without an environment transition | 0 |
| PPO gradient update | 0 |
| Inactive padding lanes in a final vector batch | 0 |

The runner should reserve or debit budget **before** executing each transition, with a final partial batch containing only the remaining active lanes. It should never execute a full vector batch and then report that the budget was exceeded. The terminal transition counts; the subsequent reset does not.

If the event instead wants to count vector calls or policy decisions, that must be stated explicitly. It would make larger `n_envs` or action repeat a potential advantage and would not match the usual sample-efficiency interpretation.

### 2.2 Cumulative versus per-run accounting

### Evidence

- MineRL demonstrates a cumulative interaction ceiling for a competition round, including interactions hidden by skipped frames. Its public rules do not support treating every local restart as a fresh allowance.
- The current project README describes a persistent `n` runs by `k` steps-per-run budget, says the budget state is hash-tamper-evident, and explicitly says it is **not yet** a cumulative-week budget. The handoff marks cumulative versus per-run accounting as open.

### Options

| Option | Fairness properties | Main failure mode |
|---|---|---|
| **Cumulative global budget** | Strongest control over total experience; reward edits and restarts consume the same pool; easy to state as `B` transitions | Requires durable resume state and a clear recovery process |
| **Fixed budget per run, unlimited runs** | Simple local UX; easy to retry after crashes | Participants can buy unlimited exploration through restarts; the budget is not meaningful |
| **Fixed number of runs, each with a cap** | Limits both total and run length if both values are frozen | More complex; gives a strategic advantage to how participants allocate runs unless the allocation is fixed |
| **Organizer retraining only** | Strongest enforcement if code runs in a controlled service | Less accessible; more queueing and infrastructure; source-code execution must be sandboxed |

### Recommendation

Use a cumulative budget across the full training window. Reward-file edits, checkpoint restores, process restarts, and changes to participant-selected hyperparameters must not reset the counter. A participant may stop early and resume, but the resume record must identify the same submission/training identity and continue from the persisted counter.

If the event wants “runs” for teaching or UX, treat them as reporting segments inside one global budget. Log `run_id` and per-run metrics, but keep `cumulative_steps` authoritative.

### Open

- Exact budget `B`.
- Whether budget counts all lanes in a vectorized trainer or only the lane transitions actually executed.
- Whether failed setup attempts consume budget. Recommendation: no charge before the first accepted transition, but a transition already executed before a process crash remains charged.
- Whether participants may resume from a checkpoint after a machine failure, and how the organizer distinguishes recovery from a deliberate reset.
- Whether reward edits are allowed without restarting the process. Recommendation: yes, but never with a counter reset.

## 3. Tamper-evident budget and training logs

### Evidence

- NIST FIPS 180-4 says hash algorithms generate digests used to detect whether messages have changed.
- RFC 6962 describes publicly auditable append-only logs, Merkle hash trees, signed tree heads, audit paths, consistency proofs, and detection of a log presenting inconsistent views.
- The current replay contract correctly states that its SHA-256 payload checksum detects accidental changes but is not an authenticated anti-cheat proof or proof of training provenance.

### Recommendation: minimum log record

Write an append-only record for every budget reservation or for bounded batches of reservations. Each record should contain:

```text
sequence_number
submission_or_training_id
run_id
timestamp_for_diagnostics_only
environment_release_id
environment_code_hash
track_set_hash_or_public_track_manifest_id
simulation_config_hash
reward_config_hash
architecture_id
runtime/container_digest
random_seed_or_seed_schedule_id
steps_before
steps_charged
steps_after
event_type
previous_record_hash
record_hash
```

Use canonical serialization with explicit field order or sorted keys, UTF-8, and rejection of NaN/Infinity. The hash should commit to the prior hash and the full current record. Periodically write a signed checkpoint/root containing the last sequence number and hash. Keep the signing key outside the participant workspace. If a public audit is important, publish a release manifest or signed checkpoint after the training window rather than exposing private track data.

The log verifier should reject:

- a missing or duplicated sequence number;
- a non-monotonic cumulative count;
- `steps_after != steps_before + steps_charged`;
- a changed release/config/reward identity without an explicit event;
- a batch that charges inactive lanes;
- a final count above `B`;
- an invalid previous-hash link or signature;
- records that claim a transition without a corresponding runner event.

### Important limitation

A participant-controlled hash chain is **tamper-evident after an external anchor**, not tamper-proof. Someone who controls the whole local workspace can delete and regenerate an unsigned chain. For strong enforcement, the counter and signed checkpoints must be produced by an organizer-controlled launcher/service or the final model must be retrained by the organizer in a controlled container. A participant log remains useful for transparency, debugging, and detecting accidental edits.

## 4. Model architecture constraints

### Evidence

- SB3 documents PPO support for common action and observation spaces and provides configurable policy network architecture. It also documents that recurrent PPO is a separate contrib component.
- SB3 does not establish a universal fair or optimal architecture for racing. Its documentation presents architecture and hyperparameters as experiment choices.
- MineRL’s competition model shows a different control: participants submitted training code, while the evaluator rebuilt and ran submissions in organizer infrastructure under a specified environment and hardware/time limit.

### Options

| Policy | Participant freedom | Verification burden | Fairness risk |
|---|---|---|---|
| **Fixed architecture** | Reward design and training behavior are the main creative variables | Low: exact input/output names and tensor shapes | Lowest capacity and latency variance |
| **Bounded architecture allowlist** | A small declared set, e.g. selected MLP depths/widths | Medium: graph, parameter, operator, memory, and runtime validation | Larger models may gain capacity or consume more wall-clock time |
| **Parameter/FLOP cap** | Broad architecture choices within a resource envelope | High: reliable FLOP, memory, operator, and worst-case latency measurement | Different architectures can still have different optimization and hardware behavior |
| **Free model plus standardized inference API** | Highest freedom | Very high: sandbox, timeout, memory, dependency, and network controls | Hardware and implementation tricks can dominate a step-only budget |

### Recommendation for v1

Use a fixed model architecture for the public event, such as the project’s current fixed two-layer MLP, unless the event’s purpose explicitly includes architecture search. If architecture choice is part of the lesson, publish a small allowlist and validate the serialized graph rather than trusting a participant-provided architecture string.

At minimum, fix:

- observation input shape and dtype;
- action output shape, dtype, and range;
- hidden-layer depth and width or an explicit allowlist;
- activation functions, policy distribution, and action-bound mapping;
- parameter count and artifact size;
- inference device and thread limits during evaluation;
- whether recurrent state, frame stacking, external memory, or test-time adaptation is allowed;
- forbidden operations such as filesystem, network, subprocess, environment imports, and dynamic code loading.

If ONNX becomes the submission format, also freeze the ONNX opset, execution provider, dynamic/static shape policy, supported operator allowlist, external-data behavior, and runtime version. An ONNX file by itself does not remove numerical or operator-version variability.

### Open

- Fixed architecture or bounded choice?
- Is inference latency a published rule, or is every policy run on a fixed CPU with a per-step timeout?
- Are recurrent policies, frame stacking, test-time adaptation, and external lookup tables allowed?
- Is participant reward design the only free variable, or may participants also choose model architecture/hyperparameters?

## 5. Observation and action contracts

### Evidence

- Gymnasium requires declaring `action_space` and `observation_space`, recommends a seeded `reset`, and exposes diagnostics through `info`. It says observations and info returned from calls should be fresh objects rather than mutated shared buffers.
- Gymnasium’s custom-environment guidance says observations should contain the information needed for decisions, and its API makes the reset seed the normal source of reproducible environment randomness.
- Gymnasium’s time-limit tutorial says `terminated` represents an MDP terminal state while `truncated` represents an external cutoff. Bootstrapping should use the distinction rather than a single `done` flag.
- The current project’s state schema fixes a 23-value float32 observation, a two-value action in `[-1, 1]`, finite-action validation, no absolute position/heading/track ID/absolute progress, and serialized normalization/evaluation metadata.
- The MineRL rules show why an observation/action contract must be enforced at the API boundary: they restrict interaction to `step`, prohibit extra simulator information, and obfuscate action/observation labels to prevent semantic shortcuts.

### Recommendation: public contract

Publish a versioned contract with:

1. Exact field order, shape, dtype, units, bounds, clipping behavior, and NaN/Infinity behavior.
2. Exact action semantics, including whether longitudinal control is signed throttle/brake or separate gas/brake channels.
3. Reset semantics, seed behavior, start distribution, episode step limit, and whether reset state itself earns reward.
4. `terminated` versus `truncated` semantics and the expected value-bootstrap behavior.
5. A statement that `info` is diagnostic and is not an allowed policy input unless a field is explicitly included in the observation contract.
6. Contract, environment, configuration, track-manifest, and normalization hashes in every training/evaluation artifact.
7. A schema-version rejection rule. Do not silently reinterpret an older field order or action mapping.

Keep global information that permits memorization out of the policy observation: absolute world coordinates, track identity, absolute lap progress, private geometry, future racing-line data, and held-out track identifiers. If a finite-horizon task requires remaining time to preserve the Markov property, include the predeclared remaining-time feature or define the cutoff as an external truncation and handle it consistently in training.

Keep training diagnostics and reward components available for participant learning, but prevent them from becoming an unreviewed side channel. A participant trainer should receive only the declared observation and permitted reward/diagnostic surface. Internal geometry projections, held-out geometry, collision structures, and organizer scoring data should not be available through Python object references or `info`.

## 6. Dependency and runtime pinning

### Evidence

- pip’s repeatable-install guidance says exact version pins protect against newly released versions and that hash-checking protects against package or index changes. It also describes wheelhouses for offline installation.
- Python’s `venv` documentation says virtual environments isolate project packages and are disposable/recreated from a requirements file; it warns that environments are not generally portable because scripts can contain absolute interpreter paths.
- Docker’s official best-practices documentation says image tags are mutable, recommends pinning base images by digest for exact image identity, and recommends rebuilding/testing images. The Dockerfile reference documents `RUN --network=none` for offline build steps.
- PyTorch’s reproducibility guidance says results can differ across releases, commits, platforms, and CPU/GPU, even with identical seeds.
- The current project has a `requirements.lock`, a Dockerfile, and a README claiming Python 3.11. However, `pyproject.toml` currently permits Python `>=3.10`; the Dockerfile uses mutable `python:3.11-slim`; the lock file pins versions but does not include per-wheel hashes; and no `.python-version` was found.

### Recommendation

For the public release, pin the complete evaluation identity:

- exact Python 3.11 patch release, not only `>=3.10` or `3.11`;
- exact OS/container image digest;
- exact PyTorch, NumPy, inference runtime, and transitive dependencies;
- hashes for every installable artifact, or a checked-in/offline wheelhouse with a manifest;
- CPU versus GPU execution provider;
- BLAS/OpenMP thread counts and environment variables;
- locale, timezone, and numeric serialization settings;
- seed policy for Python, NumPy, PyTorch, and every environment RNG;
- deterministic-algorithm flags and any performance trade-off they impose;
- package index/network policy during training and evaluation.

Use the same image for the reference baseline, participant smoke verification, final evaluation, and replay generation. A container that is reproducible to a tag but not a digest is not a complete release identity.

Do not describe deterministic evaluation as cross-platform bitwise identity unless it has been tested. A safer claim is: **same release image, same device/provider, same seeds, same initial state, and same action policy produce the same result and trajectory digest**.

## 7. Evaluation reproducibility and deterministic scoring

### Evidence

- SB3 recommends a separate evaluation environment, warns that reward and episode-length wrappers affect results, and recommends `deterministic=True` for inference because stochastic exploration can reduce or vary performance.
- SB3’s `VecNormalize` supports saving/loading running statistics and has a non-training mode in which statistics do not update. This makes normalization state part of the evaluation artifact.
- PyTorch documents deterministic algorithms as a way to reject known nondeterministic operations, while warning that deterministic operations can be slower and do not erase release/platform differences.
- The current project evaluator uses a one-lane fixed-start path, policy mean actions, frozen normalizer statistics, environment/config/track hashes, and a streaming trajectory digest. Its replay contract explicitly says checksums are integrity checks, not anti-cheat authentication.
- Empirical Design in Reinforcement Learning and the MineRL report both support controlled budgets, held-out evaluation, and attention to variance/implementation details. The MineRL competition used organizer-side re-training/evaluation against private held-out data in its research track.

### Recommendation: deterministic score run

Use this sequence for every submission:

1. Load only the approved artifact format and verify schema, architecture, finite tensors, input/output shapes, and metadata.
2. Verify the public environment release, configuration, and any visible track manifest hashes.
3. Load private held-out tracks only inside the organizer evaluation container.
4. Create a clean evaluation environment, distinct from any training wrapper.
5. Use fixed start states and a predeclared seed/start schedule. Do not allow the submission to choose them.
6. Freeze normalizer statistics; do not update observation or reward statistics during evaluation.
7. Use deterministic policy mean actions. Do not sample exploration noise for ranking unless a stochastic protocol is explicitly approved.
8. Record fixed-step outcome data: status, transition count, lap time if finished, DNF reason, crash/off-track/stuck metrics, action saturation, and trajectory digest.
9. Repeat a deterministic run as an integrity check. A digest mismatch under the same release image and inputs is an evaluation error, not a participant score.
10. Score from Python-authoritative physical outcomes. The participant’s training reward is not the competition score.

Keep the full per-track result matrix before calculating an aggregate. If multiple tracks are used, predeclare the set, order, start schedule, DNF handling, tie-break, and whether all tracks contribute equally. Report enough data to show whether a score comes from broad generalization or one unusually easy track.

### Deterministic does not mean overfit-proof

Fixed deterministic starts improve reproducibility but can make the protocol predictable. Use private held-out tracks and a predeclared suite of deterministic starts or start-line conditions. Do not add undisclosed retries or choose the best run after seeing outcomes. If multiple deterministic attempts are part of the rules, every submission receives the same count and the aggregation rule is published before training starts.

### Open

- Number and identity of held-out tracks.
- One fixed start per track, several predeclared starts, or a fixed deterministic start schedule.
- Deterministic mean action versus a fixed stochastic evaluation protocol.
- Number of attempts/retries and whether the best, median, mean, or all-run aggregate is used.
- Ranking metric, DNF penalty, normalization across tracks, and tie-break.
- Whether deterministic digest equality is required bit-for-bit or within an explicitly documented numerical tolerance.

## 8. Submission verification

### Evidence

- MineRL’s rules require open submissions, prohibit extra simulator information, restrict network access during evaluation, scrub large files to reduce hidden pretrained artifacts, and inspect/retrain top submissions for compliance.
- The MineRL starter kit requires participants to declare their environment, dependencies, and OS packages; it uses Docker for local/online parity and requires training code to save/load models through the specified interface.
- The current project intends safe state-dictionary loading with `weights_only=True`, architecture metadata validation, environment hashes, and future ONNX evaluation. Its replay contract says public checksums do not prove training provenance.

### Submission modes

| Mode | What is submitted | What it proves | What it cannot prove alone |
|---|---|---|---|
| **Weight-only** | Weights, architecture metadata, normalization state, reward config, hashes | The submitted artifact can be loaded and evaluated | The claimed local training budget or training provenance |
| **Source retraining** | Source, lockfile/container, reward config, and declared training entry point | The organizer can rerun training under a controlled budget | That the participant’s private local run used the same process unless local provenance is separately anchored |
| **Hybrid** | Weight artifact for ordinary entries; source/container for audit or finalists | Fast scoring plus stronger compliance checks for selected entries | Full audit of every participant unless every source is rerun |

### Recommendation: verification gates

Reject or return a package for correction when any gate fails:

1. **Artifact safety:** accepted format only; no arbitrary pickle deserialization; finite values; exact tensor keys/shapes/dtypes; declared architecture matches the allowlist.
2. **Contract match:** observation/action/schema versions, normalization state, environment release, and configuration hashes match the tagged event release.
3. **Budget record:** cumulative count is valid, does not exceed `B`, and is anchored to the approved runner/log identity if the event claims budget enforcement.
4. **Smoke test:** the artifact produces finite, in-range actions for a fixed observation batch and completes a short known-seed run.
5. **Sandbox:** no network, no hidden track access, no subprocess or filesystem access beyond the declared artifact directory, and a hard inference timeout/memory limit if participant code runs.
6. **Evaluation independence:** do not accept participant-provided final scores, trajectories, reward totals, or ranking calculations as authoritative. Recompute them.
7. **Source audit where applicable:** inspect for bundled weights, extra datasets, held-out geometry, environment imports, hard-coded action sequences, and unapproved external services. MineRL’s file-size scrubbing and network restrictions are useful precedents, but file-size limits alone are not a complete provenance check.

Give a deterministic failure reason and a fixed packaging-correction window. Define a separate rule for corruption, late submissions, missing dependencies, evaluator outages, and a mismatch discovered after scoring.

## 9. Reward and environment exploit resistance

### 9.1 Evidence

- Gymnasium’s `info` is intended for metrics/debugging and may contain variables hidden from observations; this makes a clear policy-input boundary essential.
- MineRL rules use API restrictions and observation/action obfuscation to prevent simulator-side shortcuts. The relevant transferable principle is to expose only the declared interface.
- Fuchs et al. report wall-grinding caused by a progress proxy and show that a naïve fixed wall penalty can produce a different degenerate behavior, such as stopping.
- Skalse et al. distinguish an optimized proxy from the intended objective and show why omitted distinctions do not automatically make a reward unhackable.
- The current project already records progress as a transition delta, treats runoff and barrier contact separately, keeps absolute track identity/progress out of the observation, and has diagnostic fields for off-track progress and boundary excess. Those are project features, not external evidence that the event is already exploit-proof.

### Recommendation: test exploits before adding penalties

Run an equal-budget reward gauntlet and report physical outcomes separately from training return. Include at least:

- progress-only reward;
- speed-only reward;
- time-only reward with both signs checked;
- no crash penalty;
- excessive smoothness penalty;
- high off-track tolerance;
- intentionally reversed progress sign;
- reward configs that encourage survival, oscillation, or boundary contact.

For every arm, collect:

- lap completion and fixed-step lap time;
- crash, stuck, backward, timeout, and DNF reason;
- off-track duration and maximum contiguous off-track run;
- positive progress while off track;
- boundary excess and barrier contacts;
- sector coverage as a diagnostic, not a reward by default;
- action saturation and action-change/smoothness metrics;
- training return and each raw reward signal separately;
- fixed-start performance on development and private held-out tracks.

Use these safeguards:

1. **Progress:** compute wrap-aware per-transition progress; never award absolute position on reset, and never let a teleport or respawn count as driven progress.
2. **Reset:** preserve the terminal transition; distinguish a new episode from an in-episode teleport; do not treat a reset as a lap completion.
3. **Termination:** publish the precedence among crash, finish, backward, stuck, and timeout; ensure simultaneous conditions have deterministic handling.
4. **Actions:** reject NaN/Infinity before clipping; document whether raw versus applied actions are logged; prevent an action outside bounds from becoming an unbounded physics input.
5. **Observation leakage:** keep private geometry, absolute track identity, hidden progress, collision internals, and organizer score fields out of the policy path.
6. **Reward surface:** prefer a declarative, versioned signal/weight DSL for v1. A raw Python reward escape hatch expands the attack surface, makes dependency pinning harder, and complicates independent verification.
7. **Independent score:** rank on physical evaluation outcomes, not the participant-selected reward. Reward design remains the training variable, while the event score measures the fixed task.
8. **No automatic wall-grind disqualification without a rule:** first measure boundary excess and off-track progress. If the organizer later adds a penalty, reset, or DNF rule, publish it and compare it under equal budgets before freezing it.

### Open

- DSL-only rewards or a restricted raw function mode.
- Whether reward weights can change during training.
- Whether runoff recovery remains soft or becomes a terminal condition.
- Whether wall contact is a penalty, a hard crash, or an audited diagnostic.
- Whether sector coverage is participant-visible, organizer-only, or included in replay.
- Whether any privileged curriculum or domain randomization is allowed.

## 10. Recommended v1 participant-facing rule set

This is a proposed contract, not an existing event decision.

1. **Frozen release:** Participants receive a tagged release containing the environment package, training tracks, observation/action schema, fixed trainer, reward DSL, dependency lock, and container recipe. The release has a manifest hash.
2. **Budget:** Each submission receives `B` cumulative environment transitions across the entire training window. A vectorized call consumes one unit per active lane. Repeated/skipped frames count individually. Resets and optimizer updates do not count.
3. **Persistence:** Reward edits, checkpoints, and process restarts continue from the same cumulative counter. The launcher stops exactly at `B`.
4. **Policy input:** The policy receives only the versioned observation array. `info` and internal simulator state are not policy inputs.
5. **Free variable:** Participants may edit the declarative reward weights within the published signal set. Reward changes cannot alter physics, observation, action, termination, tracks, PPO hyperparameters, or budget accounting.
6. **Architecture:** Use the fixed published policy architecture and safe state-dictionary artifact for v1. If a bounded architecture track is desired, make it a separately scored event or a separately frozen contract.
7. **Runtime:** Train using the pinned environment. Evaluation uses one organizer-pinned CPU container, inference runtime, thread setting, and deterministic seed policy for everyone.
8. **Submission:** Submit weights, architecture metadata, normalization state, final reward config, training log, release manifest hash, and a reproducibility manifest. Do not submit private held-out tracks or arbitrary evaluator code.
9. **Evaluation:** Run the deterministic mean policy from fixed predeclared start conditions on private held-out tracks. No hidden retry or best-of-N selection is permitted.
10. **Score:** Score Python-authoritative lap time and DNF outcomes using a published multi-track aggregation and DNF rule. Training reward and local replay checksums do not determine rank.
11. **Verification:** Validate the artifact, hashes, finite values, schema, architecture, smoke behavior, log chain, and package contents before evaluation. Recompute all scores.
12. **Transparency:** Publish the release manifest, public training-track hashes, scoring specification, evaluation image identity, and post-event per-track results. Keep private track geometry and private start schedules confidential until the event is complete.

## 11. Major decisions that must stay open

These decisions should be recorded in the event decision register before the participant release is tagged.

| Decision | Why it matters | Current recommendation |
|---|---|---|
| Cumulative global budget or per-run budget | Determines whether restarts and reward edits can buy more experience | Cumulative global budget |
| Exact transition unit | Prevents vectorization/action-repeat ambiguity | One active lane transition; skipped frames count |
| Exact `B` | Sets training opportunity and event difficulty | Measure from equal-budget baselines and generalization tests |
| Local honor system or organizer enforcement | Determines whether a participant-owned log can be trusted | Organizer runner or retraining for strong enforcement |
| Fixed versus bounded architecture | Controls capacity, speed, and validation complexity | Fixed architecture for v1 |
| Inference latency/resource cap | Needed if architecture or arbitrary code is allowed | Fixed evaluation CPU/container; add timeout if code runs |
| Reward DSL versus raw Python | Balances teaching flexibility against sandbox and reproducibility | DSL only for v1 |
| Exact observation fields and normalization | Affects learnability, leakage, and artifact compatibility | Freeze schema and serialize all preprocessing |
| Action distribution | Stochastic sampling, clipping, and tanh squashing can change behavior | Freeze one policy distribution and log it |
| Training spawn/reset distribution | Changes coverage, learning difficulty, and exploit surface | Publish and ablate before freezing |
| Held-out track count and secrecy | Determines generalization and leak risk | Private, difficulty-matched set; count still open |
| Evaluation starts and repeat count | Trades repeatability against fixed-start overfitting | Fixed predeclared deterministic suite; no hidden retries |
| Ranking and DNF rule | Can dominate the outcome more than model skill | Publish per-track results and aggregate before launch |
| CPU/GPU and numerical determinism | PyTorch warns that identical seeds do not ensure cross-device identity | One pinned evaluation image/device/provider |
| Submission format | Arbitrary code increases attack surface; weights alone weaken provenance | Safe artifact plus optional source audit/retraining |
| Log anchor and signer | Hash chains without an external anchor do not prove local history | Organizer-held signature or controlled service |
| Failure/appeal process | Packaging and infrastructure failures need consistent treatment | Fixed correction window and documented owner |

## 12. Current repository implications (no code changes made)

The following are observations for release planning, not edits made by this research task:

- `README.md` says Python 3.11, while `pyproject.toml` currently allows Python `>=3.10`. Align these before claiming a single runtime contract.
- `requirements.lock` pins package versions but does not use pip hash-checking. Decide whether the event needs hashes or an offline wheelhouse.
- `Dockerfile` uses the mutable `python:3.11-slim` tag. Pin the release image by digest for tournament evaluation.
- The current README says the budget is persistent across `n` runs by `k` steps but not yet cumulative across the week. This remains an open event policy, not a settled fairness guarantee.
- The current fixed PPO artifact, 23-value observation schema, two-action contract, deterministic mean-action evaluator, frozen normalizer, and trajectory digest are useful foundations. They should be treated as provisional until the event decision register and release manifest freeze them.
- The current replay contract explicitly distinguishes integrity checks from anti-cheat authentication. Preserve that distinction for budget and evaluation logs.
- The README states that ONNX evaluation is not implemented yet. If ONNX remains the public submission format, add a separate versioned runtime/operator contract before presenting it as a fairness control.
- The current repository contains unrelated working-tree modifications and untracked Unity files at the time of research. This notes task did not modify them.

## 13. Sources and access dates

All web sources below were accessed on **2026-10-17**. Existing project documents are listed separately. URLs are retained so organizers can re-check version changes before freezing the event rules.

### Primary and authoritative sources

1. **Gymnasium Env API.** `step`, `reset`, action/observation spaces, seeding, `terminated`, `truncated`, and diagnostic `info`.
   - https://gymnasium.farama.org/api/env/
2. **Gymnasium custom environment guide.** Environment design, public spaces, seeding, info, and environment checks.
   - https://gymnasium.farama.org/introduction/create_custom_env/
3. **Gymnasium time-limit tutorial.** Why termination and truncation must be distinct for learning targets.
   - https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
4. **Gymnasium speed-up guide.** Vectorized environments and their effect on sample throughput.
   - https://gymnasium.farama.org/introduction/speed_up_env/
5. **Stable-Baselines3 PPO documentation.** PPO rollout accounting, CPU/vectorization guidance, `total_timesteps`, policy options, and deterministic inference.
   - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
6. **Stable-Baselines3 RL tips.** Separate evaluation environments, deterministic actions, normalization, action bounds, wrapper cautions, multiple runs, and timeout handling.
   - https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
7. **Stable-Baselines3 vectorized environments and `VecNormalize`.** Vector reset behavior, terminal observations, saved normalization state, and frozen evaluation statistics.
   - https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
8. **PyTorch reproducibility notes.** Seed control, deterministic algorithms, and limits across releases/platforms/devices. Page metadata states created and updated 2026-05-14.
   - https://docs.pytorch.org/docs/2.14/notes/randomness.html
9. **pip repeatable installs.** Version pinning, hash-checking, and wheelhouses/offline installation.
   - https://pip.pypa.io/en/stable/topics/repeatable-installs/
10. **pip requirements-file format.** `--require-hashes`, local files, indexes, and package requirement semantics.
    - https://pip.pypa.io/en/stable/reference/requirements-file-format/
11. **Python 3.11 `venv` documentation.** Isolated project environments, recreation, and non-portability of moved virtual environments.
    - https://docs.python.org/3.11/library/venv.html
12. **Docker build best practices.** Trusted/minimal bases, rebuilding, network/build controls, and pinning image digests.
    - https://docs.docker.com/build/building/best-practices/
13. **Dockerfile reference.** Image references, `RUN --network=none`, exec form, and build instructions.
    - https://docs.docker.com/reference/dockerfile/
14. **NIST FIPS 180-4, Secure Hash Standard.** Hash digests for detecting message changes. Published 2015-08.
    - https://csrc.nist.gov/pubs/fips/180-4/upd1/final
15. **RFC 6962, Certificate Transparency.** Append-only logs, Merkle trees, signed tree heads, audit paths, and consistency proofs. Published 2013-06; RFC 9162 now obsoletes it, but the cited mechanisms remain relevant to the design pattern.
    - https://www.rfc-editor.org/rfc/rfc6962
16. **NIST SP 800-57 Part 1 Rev. 5.** General key-management guidance relevant to protecting any organizer signing key. Published 2020-05.
    - https://csrc.nist.gov/pubs/sp/800/57/pt1/r5/final

### Competition rules and reports

17. **AIcrowd MineRL 2021 challenge rules.** Interaction through `step`, no extra simulator information, eight-million interaction limit, skipped-frame accounting, no external network during evaluation, file scrubbing, container/retraining procedure, and private held-out round.
    - https://www.aicrowd.com/challenges/neurips-2021-minerl-diamond-competition/challenge_rules
18. **MineRL 2021 competition starter template.** Submission structure, Docker/runtime declaration, local versus evaluator execution, source submission, and four-day/eight-million-sample research-track constraints.
    - https://github.com/minerllabs/competition_submission_template
19. **Guss et al., “The MineRL 2019 Competition on Sample Efficient Reinforcement Learning using Human Priors.”** Competition report describing limited simulator samples, containerized algorithm submissions, prespecified hardware, and retraining/evaluation on held-out data/environment pairs. arXiv v3, 2021-01-19.
    - https://arxiv.org/abs/1904.10079
20. **AIcrowd MineRL 2019 competition page.** Historical participant-facing statement of an eight-million-sample/four-day limit, hardware ceiling, organizer verification, and held-out evaluation.
    - https://www.aicrowd.com/challenges/neurips-2019-minerl-competition

### RL reproducibility and exploit evidence

21. **Patterson et al., “Empirical Design in Reinforcement Learning.”** JMLR 2024; arXiv v2 2024-10-29. Statistical variation, seeds, budgets, environmental overfitting, and experiment design.
    - https://arxiv.org/abs/2304.01315
22. **Fuchs et al., “Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning.”** Reward/progress proxy and wall-grinding failure modes.
    - https://arxiv.org/html/2008.07971v2
23. **Skalse et al., “Defining and Characterizing Reward Hacking.”** Proxy objective versus intended objective and limits of reward safety.
    - https://arxiv.org/html/2209.13085v2
24. **Cobbe et al., “Leveraging Procedural Generation to Benchmark Reinforcement Learning.”** Training-set overfitting and fixed-level memorization evidence.
    - https://arxiv.org/html/1912.01588v2
25. **Czechmanowski et al., “On learning racing policies with reinforcement learning.”** Racing precedent for random track-position resets and exploration coverage; not a proof that random respawn improves this event’s unseen-track generalization.
    - https://arxiv.org/html/2504.02420v1

### Project documents consulted

26. `README.md` — current provisional decisions, runtime, budget wording, contracts, and evaluator behavior.
27. `RACING_EVENT_Agent_Handoff.md` — decision register, open budget/architecture/ranking questions, submission concept, and validation experiments.
28. `docs/state_schema.md` — current action/observation/training contract and deterministic evaluation artifact.
29. `docs/replay_contract.md` — replay integrity, hash limitations, renderer boundary, and private-track warning.
30. `research_notes/Racing QoL and gameplay ideas/anti_exploit_training.md` — current anti-grind diagnostics and reward-gauntlet recommendations.
31. `research_notes/Track sourcing and racing physics/racing_rl_controls.md` — current action/observation, PPO, normalization, and evaluation research.
32. `research_notes/Track sourcing and racing physics/2d_racing_physics.md` — fixed-step and deterministic-physics research.
