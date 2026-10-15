# RACING RL EVENT: AGENT HANDOFF SPEC

Audience: an implementing AI agent (Claude Opus 5.5 or similar) that will build this project from the ground up.
Status of this document: written at the end of a long design conversation. It records what the human decided, what the previous assistant recommended, and what is still open. Read Section 0 and Section 4 before doing anything.

---

## 0. HOW TO USE THIS DOCUMENT

1. Every decision in Section 4 carries one of three tags.
   - **LOCKED**: the human decided this. Do not change it. If you think it is wrong, say so and wait for the human to confirm.
   - **RECOMMENDED**: the previous assistant proposed it and the human did not object, or the human did not answer yet. Use it as the default. Tell the human when you rely on it.
   - **OPEN**: nobody decided. Do not pick silently. Propose an option with trade-offs, ask the human, and record the answer.
2. Numbers in this document are **planning estimates**, not measurements. Anything marked `[TUNE]` must be validated by experiment (Section 13) before it is frozen into the release.
3. Do not implement the fighter game (SYNAPSE ARENA). It is a separate, later event. See Section 17.
4. The human prefers direct answers, honest pushback, and a clear recommendation with trade-offs. Casual tone is fine. Do not pad.
5. Do not copy code from unlicensed repositories. See Section 15.
6. Verify any license, dataset, or library claim in this document with a fresh lookup before you depend on it. Some claims here come from memory of search snippets.

---

## 1. PROJECT SUMMARY

A college tech-fest event where participants train their own reinforcement learning (RL) racing agents, then watch them race on tracks they have never seen.

- Two student clubs run it together. **AIC** (AI Club) owns the Python simulation, the RL training pipeline, fairness rules, and tournament infrastructure. **VECTOR** (game development club) owns the Unity project: track and car visuals, ghost playback, camera, UI, audio, and show presentation. Both clubs sit under the college Technical Board.
- Format is **Trackmania-style ghosts**: cars never collide or interact. A "ghost" is a recorded run of one agent, drawn alongside the others on the same track. The race is a comparison of lap times with a head-to-head visual.
- Participants get about **one week** to train on a small set of **training tracks**. The final race uses **held-out tracks** that nobody has seen (a train-test split).
- Participants control the **reward function** (and possibly a bounded model architecture). Everything else is fixed and identical for everyone: algorithm, hyperparameters, environment, budget.
- Training is **headless Python** (fast, no renderer). The live show attaches a **Unity renderer** to the same Python simulation or to recorded trajectories. Unity never simulates.

Why this design: the event must feel fair, be cheap to run, teach reward design and generalization, and be fun to watch.

---

## 2. ROLES AND OWNERSHIP

| Area | Owner | Notes |
|---|---|---|
| Python simulation, physics, observations, rewards, termination | AIC | Single source of truth for all game logic |
| Training script, PPO, step-budget enforcement, logging | AIC | Frozen before the training window opens |
| Reward interface and architecture bounds | AIC | Participants edit only these |
| Track data pipeline (centerline, width, boundaries) | AIC | Exports one geometry file per track |
| Submission, hash check, evaluation runner | AIC | Runs on shared hardware at tournament time |
| Unity project, track mesh, car models, effects, camera, UI | VECTOR | Reads state or trajectory data only |
| Ghost presentation (simultaneous vs staggered reveal) | VECTOR | Presentation choice only |
| Held-out race tracks (design and secrecy) | AIC and VECTOR jointly | See Section 7.4 |

---

## 3. NON-NEGOTIABLE PRINCIPLES

**P1. Python is the only authority for game logic.** Unity is a renderer. Unity never simulates a car, never decides a crash, never computes a lap time. If Unity needs a value, Python sends it. This is the paramount rule. It removes the risk of two codebases drifting apart.

**P2. Unity must not use its own physics for cars.** Unity PhysX is off for car motion. Unity moves a car mesh to the pose Python reports and interpolates visually between ticks.

**P3. Everyone trains on the same environment.** The environment is frozen, versioned, and hash-checked. Participants cannot edit environment code.

**P4. Fairness is by environment steps, not hours.** The budget is a count of simulated steps, so hardware speed does not decide the result.

**P5. Reward function is the free variable.** Algorithm, hyperparameters, observation, action space, and tracks are fixed.

**P6. Let bad reward functions fail visibly.** Do not over-guard. Reward hacking is part of the lesson. Guard only against exploits that break the competition (editing the environment, exceeding the budget, leaking held-out tracks).

**P7. Track geometry is data, not code.** One geometry file per track feeds both Python and Unity. Unity builds its mesh from that file.

---

## 4. DECISION REGISTER

### 4.1 LOCKED (human decided)

| ID | Decision |
|---|---|
| L1 | Racing event is for the **technical fest**. Fighter game is for the annual fest and is out of scope now. |
| L2 | **Trackmania-style ghosts**: no car-to-car collision or interaction. |
| L3 | **Train-test split on tracks**: participants train on a few tracks, race on new unseen tracks. |
| L4 | Tracks are **inspired by real-world circuits**, mixing F1 and hypercar character and parameters. |
| L5 | **Python simulation, Unity rendering.** |
| L6 | Training window is about **one week**. |
| L7 | A **car-to-track-width ratio** must be pinned and fixed for all tracks. |
| L8 | Participants **can edit their reward function** during training. |
| L9 | Training happens on **participants' own machines**. Evaluation runs on **shared hardware**. |
| L10 | Compact **state-vector observations** (not pixels). |

### 4.2 RECOMMENDED (proposed, not contested)

| ID | Recommendation |
|---|---|
| R1 | Algorithm: **PPO**, fixed. |
| R2 | Observation: raycasts + lookahead curvature + self-state (Section 6.4). |
| R3 | Action: continuous steering and a single throttle/brake axis (Section 6.5). |
| R4 | Reward DSL with the signals in Section 6.6, including the speed, curvature-matched speed, and steering smoothness additions. |
| R5 | **Randomized spawn along the centerline** every episode. Progress reward is a per-tick delta, never absolute track position. |
| R6 | **Soft penalty** for leaving the drivable surface, **hard termination** on barrier contact. |
| R7 | Car-to-track-width starting point: track width = 4 to 5 times car width. `[TUNE]` |
| R8 | Custom vectorized numpy physics (kinematic bicycle model with a grip limit). Box2D only as an optional cross-check. |
| R9 | Step budget is enforced inside the training script and logged automatically. |
| R10 | Weights are saved in a safe format (safetensors or `torch.load(weights_only=True)`). |
| R11 | Tournament runs submitted policies headless on shared hardware, records trajectories, and Unity plays them back as ghosts. |
| R12 | Held-out tracks are never committed to any repository participants can reach. |
| R13 | Env is deterministic given a seed. Policy exploration noise is the only randomness at training time. Evaluation uses the deterministic (mean) action. |
| R14 | Reuse the process pattern from the fighter docs: frozen env, hash check, reward DSL, running step counter. |

### 4.3 OPEN (ask the human)

| ID | Question | Notes |
|---|---|---|
| O1 | Tech fest **date** and exact training window dates. | Drives everything else. |
| O2 | Number of **training tracks** N. | Hypothesis 3 to 5. Validate with Experiment E3. |
| O3 | Number and identity of **held-out race tracks** K. | Hypothesis K = 2 or 3. |
| O4 | Step budget: **cumulative across the week** or **fixed per run, unlimited runs**? | The previous assistant asked and got no answer. Default to cumulative if forced to choose, because it is the stronger fairness guarantee. |
| O5 | Exact **step budget number**. | Set from Experiment E1 and E2, not by feel. |
| O6 | **Model architecture** freedom: fixed, or bounded choice (width and depth)? | If bounded, state the allowed set. |
| O7 | Is a **per-decision latency limit** enforced at tournament time? | For a small MLP the latency is tiny. Recommend dropping it for racing unless the human wants the capacity-vs-latency lesson. |
| O8 | **Ranking rule**: total time across K tracks, best single track, or points per track? How are DNFs scored? | Recommendation in Section 10.4. |
| O9 | **Simulation tick rate.** | Recommend 30 Hz as default, 60 Hz if feel demands it. `[TUNE]` |
| O10 | Ghost presentation: **simultaneous** or **staggered reveal**. | VECTOR's call. No upstream effect. |
| O11 | **Team size**, prizes, budget, merch for this event. | The proposal doc for this event does not exist yet. |
| O12 | Is the **raw Python reward function** escape hatch allowed, or only the weighted-signal DSL? | Juniors are in the audience. Recommend DSL by default, raw function as an opt-in advanced mode. |
| O13 | Whether to make the **hard-crash vs soft-off-track** split exact as in Section 6.7. | Recommended, but confirm. |
| O14 | Which **real circuits** to draw from, and how closely to follow their layouts. | See Section 7 and Section 15. |
| O15 | Whether the final show runs agents **live** on stage or plays **recorded** runs. | Recommend recorded (R11). Live is possible with the same code. |

---

## 5. SYSTEM ARCHITECTURE

### 5.1 Two processes

- **Simulation process (Python).** Holds all state. Runs physics, observations, rewards, termination, scoring, and policy inference.
- **Render process (Unity).** Holds no rules. Receives state or trajectory data and draws it.

### 5.2 Three run modes

| Mode | Renderer attached | Where | Purpose |
|---|---|---|---|
| Training | No | Participant machine | Fast headless PPO. |
| Evaluation | No | Shared AIC machine | Run submitted policies on held-out tracks, produce scores and trajectories. |
| Presentation | Yes (Unity) | Show machine | Play back ghosts from recorded trajectories, or stream live state from Python. |

The same simulation code runs in all three. Only the attached consumer changes.

### 5.3 Transport (presentation mode)

- Recommended: local socket over localhost, Python to Unity, one message per tick, JSON first, binary later only if measured necessary.
- Direction is one way: Python to Unity. Unity may send technical acknowledgements only (ready, frame received). They never change game state.
- Messages carry a **schema version**. Both sides check it at connection time and refuse a mismatch.
- For recorded playback, skip the socket. Unity reads the trajectory file directly (Section 10.2).

### 5.4 Interpolation rule

Python ticks at a fixed rate. Unity may render faster and interpolate position and rotation between the last two ticks. Interpolation is visual smoothing only. It never changes an outcome.

---

## 6. ENVIRONMENT SPECIFICATION

### 6.1 Overview

- Gymnasium-style API (`reset`, `step`), batched over `n_envs` in numpy for throughput.
- Observation: float32 vector, fixed length and order.
- Action: 2 continuous values.
- Episode: one lap from a random spawn point, on a randomly chosen training track.
- Tick rate `dt = 1 / tick_hz` (default 30 Hz). `[TUNE]`

### 6.2 Track representation

Each track is one geometry file (JSON or NPZ) with:

- `centerline`: array of (x, y) points, **uniformly resampled** (for example 1 m spacing), closed loop.
- `s`: cumulative arclength per point. `L`: total lap length.
- `heading`, `curvature` (signed): per point, from a smoothed spline.
- `half_width_left`, `half_width_right`: per point (allows variable width).
- `left_boundary`, `right_boundary`: polylines of the drivable surface edge.
- `runoff_width`: width of the soft penalty band beyond the surface before the barrier.
- `barrier_left`, `barrier_right`: polylines. Contact ends the episode.
- `start_line_s`: arclength of the start line for evaluation runs.
- `meta`: id, display name, source note, geometry version, and a content hash.

Validation checks at build time (reject the track if any fail):

- Centerline is a closed, non-self-intersecting loop. **Tracks with crossings or overpasses (for example a figure-eight layout) break 2D projection and raycasts. Exclude them or redesign the crossing.**
- Minimum corner radius is physically drivable for the car.
- Boundaries do not cross or pinch.
- Width is consistent with the pinned car-to-track ratio.

### 6.3 Car and physics (Python-authoritative)

Recommended model: **kinematic or dynamic bicycle model** with a friction-circle grip limit. Pure numpy, vectorized over envs.

State per car: position (x, y), heading, speed (or velocity in car frame), yaw rate, current steering angle.

Required behaviors:

- Steering angle from steering command, with a **steering rate limit** and a **speed-dependent maximum angle** (so high-speed steering is not instant).
- Longitudinal acceleration from throttle with a **power limit** (acceleration falls at high speed), plus **drag**.
- Braking deceleration limit.
- **Lateral grip limit** (friction circle): combined lateral and longitudinal acceleration cannot exceed the grip budget. Exceeding it causes understeer or sliding in the model, not a hard cut.
- **Reduced grip** in the runoff band.
- Car footprint is a rectangle for track-edge and barrier tests.

Parameter ranges are `[TUNE]` placeholders. They are blended F1/hypercar-flavored, not authoritative. Verify against public data.

| Parameter | Starting range | Note |
|---|---|---|
| Car length / width | about 4.5 to 5.5 m / about 2.0 m | Generic, not a real model |
| Max speed | 70 to 90 m/s | Arcade-capped, below real F1 top speed |
| Peak lateral acceleration | 2.5 to 4 g | Between hypercar and F1 levels |
| Peak braking | 3 to 4.5 g | Same |
| Track width | 4 to 5 times car width (R7) | Narrower than real circuits for drama and signal |

Physics and fairness notes:

- All values live in **one config file**. Unity never holds a copy.
- Use fixed `dt`. No variable timestep.
- Use float64 inside the physics, cast observations to float32.
- Optional: build a Box2D version only to cross-check the bicycle model. Do not make Box2D the authority unless you are willing to pin its exact version.

### 6.4 Observation (fixed length, track-relative)

The observation must be **track-relative and local**. Absolute coordinates make unseen-track generalization impossible. Do not include absolute position, absolute heading, or a track id.

Proposed layout (about 22 to 26 floats). All normalized. `[TUNE]`

| Group | Fields |
|---|---|
| Self motion | speed, longitudinal velocity, lateral velocity (car frame), yaw rate |
| Track relation | signed lateral offset from centerline (divided by local half-width), heading error versus centerline tangent |
| Control state | previous steering command, previous throttle/brake command, current steering angle |
| Raycasts | 9 rays at about 0, ±15, ±30, ±60, ±90 degrees from heading, distance to track boundary, clipped at max range (about 60 to 100 m) |
| Lookahead | signed curvature (or heading change) at 4 lookahead distances, for example 15, 40, 80, 140 m ahead along the centerline. At 80 m/s, 50 m is under one second, so use longer distances than a first guess suggests. |
| Local width | track half-width at the nearest lookahead point (optional) |

Implementation notes:

- Vectorized raycasting is the main CPU cost. Intersect each ray only with boundary segments in a **local window** around the car's current centerline index (for example plus or minus 40 segments), not the whole polyline. Consider numba if numpy is too slow.
- Projection of the car onto the centerline should use the previous index as a hint (search a small window) for speed and to avoid jumping to a different part of a tight track.
- Do not expose the track id or any lap-progress value that lets the policy memorize a track.

### 6.5 Action space

Continuous, 2 values, both in [-1, 1]:

1. **Steering**: maps to steering angle through the rate and speed limits in Section 6.3.
2. **Throttle/brake**: positive is throttle, negative is brake.

Alternative if PPO training is unstable: discretize steering into 9 to 15 bins. Only change this after measuring.

### 6.6 Reward signals (the participant-facing DSL)

The environment computes **raw signals every tick** and returns them in `info["signals"]`. A participant's reward is a **weighted sum** of these signals. The training log must store each signal separately (this is the main debugging tool for juniors).

All signals are defined so that they are **independent of tick rate** (scaled by `dt` where needed) and **comparable across tracks** (progress normalized by lap length).

| Signal | Definition | Fires |
|---|---|---|
| `progress` | `(s_t - s_{t-1}) / L`, wrap-aware. Sums to about 1.0 per lap. Negative when going backward. | Every tick |
| `speed` | `clip((v - v_lo) / (v_hi - v_lo), 0, 1) * dt`. Rewards speed in a target band. | Every tick |
| `curvature_matched_speed` | `-abs(v - v_target) / v_max * dt`. `v_target = min(v_max, sqrt(a_lat_max / max(abs(kappa_ahead), eps)))`, `kappa_ahead` from a speed-scaled lookahead point. Encourages braking before corners. | Every tick |
| `off_track` | 1 while any car corner is outside the drivable surface (in the runoff band), else 0, times `dt`. | Every tick |
| `crash` | 1 on barrier contact, else 0. | Once, ends episode |
| `stuck` | 1 when the negligible-progress limit ends the episode, else 0. | Once, ends episode |
| `lap_complete` | 1 when the episode distance target (one full lap from the spawn point) is reached. | Once, ends episode |
| `time` | `dt`. Constant. Used with a negative weight as a per-tick cost. | Every tick |
| `steer_smoothness` | `abs(steer_t - steer_{t-1}) * (v / v_max)`. Used with a negative weight. | Every tick |

Interface for participants:

- Default: a config file such as `reward.yaml` with one weight per signal.
- Advanced (if O12 allows): a Python function that receives the signals dict, the observation vector, and a read-only info dict. It cannot import environment internals.
- Provide a **baseline reward config** that trains a mediocre but working driver, so nobody starts from zero.
- Weights can be edited at any time. Editing does **not** reset the step counter (Section 8.4).

Notes carried over from research (see Section 18):

- Speed-conditional reward and curvature-based target speed are the most consistent upgrades that move agents from "finishes the lap" to "races".
- Penalizing abrupt steering at high speed produces smoother, more car-like behavior and looks better on screen.
- A time penalty alone can create reckless cornering. A crash penalty and off-track penalty balance it. That tension is a teaching moment.
- Stretch goal only: a precomputed minimum-curvature racing line used as a reference signal. It is the strongest known upgrade in the literature but costs real engineering. Do not block v1 on it.

### 6.7 Episode structure, spawn, and termination

**Spawn (training):** at every reset, each env draws a track uniformly from the training set and a spawn arclength `s0` uniformly along the centerline. Add small noise: lateral offset within about ±30 percent of half-width, small heading error, and initial speed random from 0 to about half of max speed. Include a measured standing-start mixture so a policy sees the evaluation start distribution. `[TUNE]`

**Spawn (evaluation):** fixed start line, standing start, zero noise.

**Why random spawn matters:** with a fixed start, the agent over-practices the first corners and rarely reaches late corners. Random spawn gives roughly uniform coverage of the track.

**Distance target:** one full lap length `L` from `s0`. Because `progress` is a per-tick delta, a spawn near the finish gives no free reward.

**Termination:**

| Condition | Result |
|---|---|
| Barrier contact | Terminate. Large penalty comes from the participant's `crash` weight. |
| Distance target reached | Terminate with `lap_complete`. |
| Step limit (about 3 times a reference lap time) | Truncate. |
| Stuck: negligible progress for N ticks | Truncate. |
| Driving backward more than X meters | Terminate. |

**Soft vs hard:** leaving the drivable surface into the runoff band is a continuous penalty with reduced grip (episode continues). Barrier contact is final. This keeps episodes productive and gives `crash` real meaning. (R6)

**Not in v1:** domain randomization of physics, weather, tire wear, pit stops, opponent cars, collisions between cars.

---

## 7. TRACKS

### 7.1 Goals

- Real-world flavor and variety (hairpins, chicanes, long sweepers, mixed straights) for marketing and for teaching transferable driving.
- A training set that covers the corner types the race tracks use.
- Held-out tracks of **comparable difficulty distribution** to the training set. They must not be secretly harder or easier.

### 7.2 Sourcing (verify licenses first)

- Use **public geometric reference data** for real circuit layouts. Candidate sources to evaluate: public racetrack centerline and width datasets from academic motorsport groups, and OpenStreetMap-derived geometry.
  - A known academic dataset (the TUM racetrack-database on GitHub) was mentioned in discussion. **Verify its license and terms before use.**
  - If you use OpenStreetMap data, its license (ODbL) carries attribution and share-alike obligations. **Verify and comply.**
- Prefer **"inspired by"** geometry: resample, smooth, rescale, and lightly modify layouts rather than reproduce one circuit 1:1. Rename tracks with neutral names.
- Pipeline: raw geometry -> clean and close the loop -> uniform resampling -> spline smoothing -> compute heading and curvature -> assign width from the pinned ratio -> build boundaries, runoff, barriers -> validate -> export geometry file + content hash.

### 7.3 Sizing

- Real circuits are several kilometers long. At 30 Hz and 60 to 80 m/s a lap is on the order of thousands of ticks. That is fine, but check that step limits and the step budget make sense for lap length.
- Consider scaling all tracks by a common factor so the lap time is reasonable for the show and for training cost. Apply the same factor to every track.

### 7.4 Held-out track secrecy

- Held-out tracks are never committed to the participant-facing repository, its history, issues, or release assets.
- Store them offline or in a private AIC-only location. VECTOR gets them only when they need to build visuals, under the same rule.
- Evaluation code loads held-out tracks from a private path supplied at tournament time.

### 7.5 Number of training tracks

Too few tracks leads to memorization. Too many dilutes a fixed step budget. Start with N = 3 to 5 and decide with Experiment E3.

---

## 8. TRAINING PIPELINE

### 8.1 Fixed components (participants cannot change)

- Environment package (physics, observation, action, termination, training tracks).
- PPO implementation and **all hyperparameters**.
- Parallelism settings (`n_envs`, rollout length).
- Track sampling rule.
- Logging and budget enforcement.

Candidate PPO implementations: a single-file CleanRL-style PPO or Stable-Baselines3. Verify their licenses (both are believed to be permissive). Single-file is easier to audit and freeze.

### 8.2 Starting hyperparameters (`[TUNE]`, validate then freeze)

| Parameter | Starting value |
|---|---|
| `n_envs` | 32 to 64 |
| Rollout length per env | 128 to 256 |
| Discount (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Update epochs | 10 |
| Learning rate | 3e-4 with optional linear decay |
| Network | MLP, 2 hidden layers, 64 to 128 units, tanh |
| Entropy coefficient | 0.0 to 0.01 |
| Observation normalization | running mean/std (must be saved with the policy) |
| Reward normalization | optional, but if used it must be fixed for everyone |

Important: **observation normalization statistics are part of the policy.** Save and load them with the weights, or the evaluation run will behave differently from training.

### 8.3 Throughput and cost (planning estimate)

- Hot path is CPU-bound environment stepping, not GPU. A small MLP does not need a GPU.
- Planning estimate: about 3,000 to 20,000 env steps per second on a decent multi-core CPU with vectorized numpy. `[TUNE]`
- Planning estimate for "competent and generalizes to an unseen track": low hundreds of thousands to a few million steps. At the rates above that is roughly 10 to 30 minutes per full run on a mid laptop (an RTX 3050 class machine was the reference). `[TUNE]`
- These are guesses from the literature and general experience. **Experiment E1 and E2 replace them with measurements before any number goes into a public document.**

### 8.4 Step budget rules

- Budget is a **count of environment steps**. (L, P4)
- Counter is **cumulative across runs and reward edits** (recommended, pending O4).
- Training script **stops itself** at the budget. It logs the count automatically. Participants do not self-report.
- Editing the reward does not reset the counter.
- Hitting the budget is final even if calendar days remain.
- Store the counter in a signed or hash-chained log so tampering is detectable.

### 8.5 Participant workflow

1. Clone the repository and check out the tagged release.
2. Install pinned dependencies (lockfile). Offer a Colab or setup-script fallback.
3. Edit `reward.yaml` (or the advanced function if allowed).
4. Optionally choose an architecture from the allowed set (pending O6).
5. Run `train.py`. It trains headless until the budget or until the participant stops it.
6. View a **replay**: a lightweight viewer (matplotlib, pygame, or a simple HTML canvas) that renders a saved trajectory on the training tracks. This is required so participants can see why their car behaves as it does. It does not need Unity.
7. Edit the reward and continue, spending from the same budget.
8. Submit: policy weights, observation normalization stats, final reward file, architecture config, training log, and environment hash.

Recommend a short supervised kickoff session so setup problems do not eat the first days.

### 8.6 Logging requirements

Per run, log at minimum: step count, wall time, mean episode return, **each reward signal separately**, lap completion rate, mean speed, crash rate, off-track rate, per-track statistics. These logs feed the replay viewer and the diagnostics participants need.

---

## 9. SUBMISSION AND VERIFICATION

Submission package:

- Policy weights in a safe format (R10). **Never load untrusted pickle files.** Use safetensors or `torch.load(weights_only=True)`, and reject anything else.
- Observation normalization stats.
- Architecture config.
- Final reward config.
- Training log with step counts.
- Environment content hash.

AIC verifies:

1. Environment hash matches the tagged release (hash all environment files, the track data, and the config).
2. Step count in the log does not exceed the budget and the log is internally consistent.
3. Weights load and produce actions on a smoke test within the tournament runner.
4. Architecture is inside the allowed set.

A submission failing any check is rejected, and the participant gets a reason and a chance to fix a packaging problem before the deadline. Define who owns tournament-day edge cases (corrupt file, late submission). Placeholder owner: AIC.

---

## 10. TOURNAMENT, GHOSTS, AND THE STATE CONTRACT

### 10.1 Evaluation protocol (RECOMMENDED, pending O8)

For each held-out track:

1. Spawn at the start line with a standing start, no noise.
2. Run the policy with the **deterministic mean action**.
3. Record the full trajectory.
4. Record lap time, or DNF on barrier contact, step-limit truncation, or stuck.

Run on one shared machine with the same environment, the same package versions, and the same seeds for everyone.

### 10.2 Trajectory file format (one file per run)

This format is both the ghost replay format and the Unity input. It must be versioned.

```json
{
  "schema_version": "1.0",
  "env_hash": "<hash>",
  "track_id": "heldout_01",
  "track_hash": "<hash>",
  "tick_hz": 30,
  "participant_id": "<id>",
  "result": {"status": "finished|dnf_crash|dnf_timeout|dnf_stuck", "lap_time_s": 87.43},
  "frames": [
    {
      "tick": 0,
      "x": 0.0, "y": 0.0,
      "heading": 0.0,
      "speed": 0.0,
      "steer": 0.0,
      "throttle_brake": 0.0,
      "yaw_rate": 0.0,
      "off_track": false,
      "events": []
    }
  ]
}
```

Event strings (examples): `off_track_start`, `off_track_end`, `barrier_contact`, `lap_complete`, `checkpoint_k`.

Trajectories may be stored in a compact binary or NPZ form for size. JSON is the exchange format with VECTOR.

### 10.3 Live state message (if streaming)

Same fields as one trajectory frame, plus `participant_id` and a `schema_version` check at connect time. Python sends one message per tick. Unity never sends game-affecting data back.

### 10.4 Ranking rule (RECOMMENDED, pending O8)

- Score = **sum of lap times across all held-out tracks**.
- A DNF on a track scores a fixed penalty (for example 2 times a reference lap time for that track) or an explicit rule the human chooses.
- Tie-break: best single-track time.
- Show per-track times so the audience sees where each agent won or lost.

### 10.5 Unity responsibilities (VECTOR)

- Build the track mesh from the **same geometry file** AIC exports. Do not redraw the track by hand.
- Place and orient car meshes from `x, y, heading`. Interpolate between ticks.
- Drive animation and effects from `steer`, `throttle_brake`, `speed`, and `events` (tire smoke on `off_track`, impact on `barrier_contact`, confetti on `lap_complete`).
- Render N ghosts simultaneously or staggered (O10).
- Camera, UI, leaderboard, audio, bracket or results screen.
- Never compute collisions, lap times, or outcomes.

### 10.6 Joint tests before either side builds heavily

1. **Scripted-run parity test:** AIC records a trajectory from a fixed scripted action sequence. VECTOR plays it back. A reviewer confirms it looks like the Python run (position on track, timing, events).
2. **Geometry round-trip test:** the track mesh edges in Unity must match the Python boundary polylines within a small tolerance.
3. **Schema-version mismatch test:** a wrong version is rejected with a clear error.

---

## 11. REPOSITORY STRUCTURE (SUGGESTED)

```
racing-event/
  README.md
  LICENSE
  pyproject.toml / requirements.lock      # pinned dependencies
  env/                                    # FROZEN. hash-checked. AIC owned.
    car_physics.py
    track.py                              # loading, projection, raycast
    racing_env.py                         # vectorized env
    observation.py
    signals.py                            # raw reward signals
    config.yaml                           # all physics and balance values
    tracks/train/*.json                   # training track geometry
  trainer/                                # FROZEN.
    ppo.py
    train.py                              # budget enforcement, logging
    hashcheck.py
  tools/
    track_builder.py                      # raw geometry -> geometry file
    replay_viewer.py                      # lightweight viewer
    eval_runner.py                        # tournament runner (no held-out tracks here)
    trajectory_io.py
  submissions/                            # participant-editable surface
    reward.yaml
    arch.yaml
  examples/
    baseline_reward.yaml
  docs/
    architecture.md
    participant_guide.md
    state_schema.md
  tests/
    test_determinism.py
    test_budget_enforcement.py
    test_hashcheck.py
    test_signals.py
    test_track_validation.py
  private/                                # NOT IN THE PUBLIC REPO
    tracks/heldout/*.json
```

Tag a release when the environment freezes. Participants use the tag only, never `main`.

---

## 12. BUILD ORDER AND ACCEPTANCE TESTS

| Milestone | Deliverable | Acceptance test |
|---|---|---|
| M0 | Track pipeline: raw geometry to validated geometry file | 3 sample tracks pass validation. Plots look right. |
| M1 | Car physics and single-env step function | Scripted inputs give sane trajectories. Grip and runoff behavior verified by plots. |
| M2 | Vectorized env with observation, signals, termination, spawn | Determinism test passes. Throughput benchmark recorded (E1). |
| M3 | PPO trainer with budget enforcement and logging | Baseline reward trains a car that completes laps on a training track. Budget stops training exactly. |
| M4 | Replay viewer | A participant can see a saved run on a training track. |
| M5 | Validation experiments E2 to E7 (Section 13) | Results written up. Defaults chosen from data. |
| M6 | Freeze and package: tag, lockfile, hash tool, baseline, participant guide | A fresh machine installs and trains in under 30 minutes of human effort. |
| M7 | Trajectory export, schema v1, evaluation runner | A recorded run replays identically from file. |
| M8 | Joint test with VECTOR (Section 10.6) | All three joint tests pass. |
| M9 | Dry run: 3 to 5 volunteers play participant for a day | Collect setup friction and reward-design surprises. Fix before launch. |
| M10 | Launch, training window, submissions, evaluation, show | Tournament-day runbook exists and was rehearsed. |

Do M0 to M3 as thin vertical slices first (one track, one reward, one run), then widen.

---

## 13. VALIDATION EXPERIMENTS (RUN BEFORE FREEZING NUMBERS)

| ID | Experiment | Output |
|---|---|---|
| E1 | **Throughput benchmark**: steps per second versus `n_envs` on a mid laptop CPU, with and without raycast optimization | Realistic steps/sec. Decide if numba is needed. |
| E2 | **Convergence curve**: train the baseline reward on the training set, plot lap completion rate and mean speed against steps | Planning range for the step budget. |
| E3 | **Generalization**: train on N in {2, 3, 5, 8} training tracks, evaluate zero-shot on held-out tracks | Pick N. Confirm the held-out gap is acceptable. |
| E4 | **Width ratio sweep**: track width = 3, 4, 5, 6 times car width | Choose the ratio from training stability and visible skill difference. |
| E5 | **Reward gauntlet**: run deliberately bad reward configs (only speed, only progress, no crash penalty, huge smoothness penalty, only time penalty) | Document failure modes. Check the environment tolerates them without breaking. |
| E6 | **Spawn strategy ablation**: fixed start versus random spawn | Confirm random spawn improves late-track coverage and final lap time. |
| E7 | **Determinism and hash tests**: same seed gives same trajectory. A one-byte change in an environment file fails the hash check. | Fairness guarantee proven. |
| E8 | **Tick-rate check**: 20, 30, 60 Hz | Feel in Unity versus training cost. Choose the tick rate. |
| E9 | **Held-out difficulty parity**: run the same reference agent on all training and held-out tracks | Confirm held-out tracks are not secretly easier or harder. |

---

## 14. KNOWN PITFALLS

1. **Dual implementation drift.** Anything that makes Unity compute game logic breaks P1. Reject it.
2. **Track crossings and overpasses** break 2D projection and raycasts (Section 6.2).
3. **Observation leakage.** Absolute coordinates, track id, or absolute progress let policies memorize tracks.
4. **Missing normalization stats** at evaluation time cause different behavior than at training time.
5. **Pickle risk** in `.pt` files from participants (Section 9).
6. **Absolute progress reward** with random spawn gives free reward near the finish (use deltas).
7. **Reward exploits** you should expect: crawling for survival bonuses (no survival bonus exists, but a time-penalty sign error can create it), oscillating at a reward-optimal speed or offset, reversing for a mis-signed progress, hugging the barrier. Run E5 before launch.
8. **Held-out track leak** through repo history, screenshots, or VECTOR assets.
9. **Step budget ambiguity**: define cumulative versus per-run in writing before launch.
10. **Setup friction** for juniors can consume days. Provide a kickoff session and a fallback environment.
11. **Tick rate versus feel**: a low tick rate with fast cars can look floaty even with correct interpolation.
12. **Float nondeterminism** across machines: evaluation runs on one machine, so this is manageable, but keep the physics in float64 and avoid unseeded randomness.
13. **Throughput claims are unverified.** Do not publish them until E1 and E2 are done.

---

## 15. LICENSING AND IP

Treat the following as a starting point. Re-verify each item with a fresh search.

| Item | Believed license | Use |
|---|---|---|
| Gymnasium (including `CarRacing-v3`) | MIT | Safe to read and adapt. Use its reward shape as a reference. Do **not** adopt its 96x96 pixel observation (L10). |
| Box2D | zlib | Optional physics cross-check. Very permissive. |
| pygame | LGPL 2.1 | Fine as an unmodified dependency in AIC tooling. Unity does the real rendering. |
| PyGame Learning Environment | MIT | Architectural inspiration only. |
| CleanRL, Stable-Baselines3 | believed MIT | Verify before bundling. |
| Small GitHub hobby racing repos | mostly unlicensed | **Read for ideas only. Do not copy code.** Unlicensed means all rights reserved. |

IP guidance for tracks and cars (this is general guidance, not legal advice):

- The physical **layout** of a real circuit is a fact about a place. Building your own geometry from public reference data is standard practice.
- **Do not** use track files, 3D models, textures, or audio ripped from existing racing games.
- **Do not** use official F1, FIA, team, sponsor, or circuit trademarks or logos. Use neutral track names.
- **Do not** model a specific manufacturer's car with real badging or trade dress. Use generic original car designs. Real **performance numbers** may be used as physics reference.
- Datasets carry their own licenses (for example OpenStreetMap ODbL). Comply with attribution and share-alike terms.
- Cite external sources in the repository README.

---

## 16. ESCALATION: QUESTIONS TO ASK THE HUMAN EARLY

Ask these in one short batch before heavy work:

1. Fest date and the exact training window (O1).
2. Cumulative or per-run step budget (O4).
3. Architecture: fixed or bounded choice (O6). Keep or drop the latency rule (O7).
4. Reward interface: DSL only, or also a raw function (O12).
5. Held-out track count K and ranking rule (O3, O8).
6. Which real circuits are the style references (O14).
7. Team size, prizes, and budget for this event, so the proposal document can be written (O11).

---

## 17. RELATED EXISTING DOCUMENTS (FIGHTER GAME, PHASE 2)

Three documents already exist for the fighter event (SYNAPSE ARENA, annual fest). They are **not** the current build target. Reuse their patterns, not their content.

| Document | Reuse |
|---|---|
| Proposal (EdgeWatch-style format) | Template for the racing proposal. Same sections: Introduction, Why this event, The challenge, Event details, Tentative Budget. Title centered, blue section headings, zebra tables, green total row. |
| Training Sandbox Technical Reference | Template for the VECTOR-facing technical reference. Written in simplified technical English (ASD-STE100 style at about 70 to 80 percent, ML terms instead of aviation terms): short sentences, active voice, one idea per sentence, terms defined at first use. |
| System Architecture | Source of the "Python authoritative, Unity renderer" rule and the state-stream contract table. Adapt the fields for cars. |

Formatting rules the human asked for in the club documents: no em dashes, centered title, event name in capitals, first line of the description follows the pattern "<Club> under the Technical Board wants to organize <event>", and numbers left blank when the human has not given them.

Fighter game facts, for reference only: 1v1, diep.io-style movement, slash and shoot attacks, curriculum of scripted bot then self-play on own checkpoints, tournament of 10 hours on ground, budget 20k INR. Do not build any of it now.

Deliverables still missing for the racing event:

1. Lean proposal document (EdgeWatch format), blocked on O11.
2. VECTOR-facing technical reference for racing (STE style).
3. Racing version of the architecture and state-stream document.
4. Participant guide.

---

## 18. REFERENCES FOUND DURING DESIGN (VERIFY BEFORE CITING)

- Gymnasium `CarRacing-v3` documentation: reward shape (small per-tick cost plus progress reward plus off-track penalty), pixel observation (not used here).
- TraD-RL, "Expert Knowledge-driven Reinforcement Learning for Autonomous Racing via Trajectory Guidance and Dynamics Constraints" (arXiv 2603.05842): high-speed reward, low-speed penalty, curvature-based target speed, racing-line guidance, two-stage curriculum.
- "High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning" (arXiv 2306.07003): reward that follows a classical planner's action, base reward of lap completion and crash penalty plus a shaped term.
- "Reward design and hyperparameter tuning for generalizable deep reinforcement learning agents in autonomous racing" (Nature Scientific Reports, 2025): speed-steering penalty, off-track penalty, effect of reward structure on generalization.
- "End-to-End Race Driving with Deep Reinforcement Learning" (Inria, ICRA 2018): respawn strategy and reward shaping affect convergence.
- AWS DeepRacer community work on K1999 path optimization and line-of-sight approaches.
- "On learning racing policies with reinforcement learning" (arXiv 2504.02420) and related autonomous-racing work: large-scale parallel training with domain randomization (reported on the order of 100 million steps for competition-grade policies, far beyond this event's needs). Check the exact figures before quoting them.
- Self-play and curriculum literature (OpenAI Five, AlphaStar, Blade and Soul) applies to the fighter game, not this event. Racing needs no opponent modeling.
- Hobby repositories for raycast-sensor car RL (for design patterns only): `schmidtlennart/RL_Race_Game`, `elchuzade/racing-game-ai`.

---

## 19. DEFINITION OF DONE

The project is ready for launch when all of the following are true:

- Environment, trainer, and tracks are frozen, tagged, and hash-checkable.
- Step budget is enforced by the script and the counter is tamper-evident.
- Experiments E1 to E9 are done and the chosen values are written in the participant guide.
- A fresh machine goes from clone to first training run without help in under 30 minutes.
- Replay viewer works on every training track.
- Trajectory schema v1 is documented and the three joint tests with VECTOR pass.
- Held-out tracks exist, are validated for difficulty parity, and are stored privately.
- Evaluation runner produces scores and trajectories for a set of dry-run submissions.
- A dry run with volunteers was completed and its findings fixed.
- Tournament-day runbook (submission deadline, verification, run order, failure handling, show order) exists.
- Proposal and VECTOR technical reference documents are written and sent.

---

## 20. SESSION HANDOFF (2026-10-14)

### Goal

Continue the next engine slice from the README and Sections 12 to 13: add a
batched NumPy racing environment, freeze the versioned observation/action and
evaluation contract, then add the smallest fixed trainer interface and measure
E1 throughput before making any public step-budget claim.

The user has asked to resume from this handoff and to use the research notes in
`research_notes/Track sourcing and racing physics/`. The current session was
paused to prepare this handoff before the vector environment was integrated.

### State

- The existing scalar `RacingEnv` remains the authoritative behavior reference.
- No vector environment has been added yet.
- No PPO trainer has been added yet. E1 and E2 have not been run.
- The observation-contract module was started in
  `racing_event/observation.py`, but it is not wired into `RacingEnv` and has
  not been exercised with NumPy tests because the current shell lacks NumPy.
- The module defines observation schema version `1.0`, action schema version
  `1.0`, training contract version `1.0`, explicit field names/bounds, a
  serializable streaming running normalizer, and a serializable
  `TrainingContract` with deterministic evaluation metadata.
- The existing scalar code still clips non-finite actions incorrectly and still
  needs to be updated to reject NaN and infinity. The existing observation
  implementation still contains the duplicated speed slot at
  `racing_event/env.py:179-183`; decide and document the final v1 treatment
  before wiring the schema. The new schema currently preserves that slot under
  the explicit name `longitudinal_velocity` for API compatibility, but this is
  not yet validated against the research recommendation.

### Files changed in this session

- `racing_event/observation.py`: new, unintegrated versioned observation,
  action, normalization, and evaluation-contract definitions.
- `RACING_EVENT_Agent_Handoff.md`: this resumable session record.

There is no Git repository at `/home/redacted/Documents/RL-ENV`; use the file
paths and checks below rather than Git status/diff as the source of truth.

### Decisions and constraints

- Keep two actions: `[steering, throttle_brake]`, finite values in `[-1, 1]`.
  Positive longitudinal action is throttle, zero is coast, and negative action
  is brake. Reject NaN and infinity before clipping.
- Keep Python as the only authority for fixed-step physics, collision,
  termination, reward, and progress. Unity is only a later renderer.
- Use float64 internally and float32 observations. Preserve fixed `dt` and
  deterministic same-seed/same-action replay.
- Use a real batched NumPy state path for physics. It may loop over geometry
  queries initially, but must return batched observations, rewards, signals,
  termination/truncation bits, events, and relevant state.
- Preserve random training spawns and provide an explicit evaluation mode with
  `start_line_s`, zero lateral/heading noise, and standing start.
- Serialize normalizer statistics and evaluation metadata. Evaluation must use
  the saved statistics without updating them and must use the deterministic
  policy mean-action path.
- Do not freeze the final cumulative/per-run budget or tick rate. The config is
  still provisionally 30 Hz; the physics notes recommend measuring 20/30/60 Hz.
- Preserve synthetic/original track geometry. Do not add copied real F1 assets
  or unverified third-party geometry.

### Research sources used

- `research_notes/Track sourcing and racing physics/racing_rl_controls.md`
  Sections “Recommended next contract for this project” and the cited
  normalization, deterministic-evaluation, batching, and logging findings.
- `research_notes/Track sourcing and racing physics/2d_racing_physics.md`
  Sections “Recommended model for this RL project” and “Determinism and
  verification checklist”.
- `README.md:43-45` and `RACING_EVENT_Agent_Handoff.md:548-580` for the build
  order and E1/E2 acceptance tests.

### Checks

- Syntax check: **passed** — `python3 -m py_compile racing_event/observation.py`
  from `/home/redacted/Documents/RL-ENV`.
- Test suite: **blocked/missing dependency** —
  `python3 -m unittest discover -s tests -v` from the project root failed during
  collection because the active `python3` has no `numpy` (`ModuleNotFoundError`).
  Do not describe the suite as passed. Install the declared project dependency
  or activate the intended project environment in the next session, then rerun
  the full suite.
- Benchmark: **not run** — no vector environment exists yet.
- Trainer/convergence: **not run** — no trainer exists yet.

### Exact next action

1. Ensure the intended Python environment has the declared `numpy>=1.24,<3`
   dependency, then rerun the existing suite.
2. Add `racing_event/vector_env.py` with a `VectorizedRacingEnv` class. Use
   vectorized NumPy car state/physics, support one or more `TrackGeometry`
   instances, and retain scalar parity for reset, step, rewards, signals,
   termination, truncation, events, and spawn behavior. Add parity,
   determinism, shape/dtype, evaluation-start, and non-finite-action tests.
3. Wire `RacingEnv` to `build_observation_schema`, resolve the
   `longitudinal_velocity`/duplicate-speed decision, and enforce the fixed
   observation bounds and finite-action rule.
4. Add a small E1 benchmark tool and write measured environment-only,
   policy-plus-environment, sensor/projection, and `n_envs` scaling results to
   `reports/e1_vectorized_benchmark.md`. Do not copy the planning estimate into
   public docs as a measured result.
5. Only after the environment contract is tested, add the fixed trainer slice
   and E2 convergence logging. Keep the saved normalizer and evaluation
   metadata from `racing_event/observation.py` as part of its artifact.

---

## 21. SESSION HANDOFF (2026-10-14, ENGINE SLICE COMPLETE)

### Goal and result

The engine slice requested from Section 20 is implemented: the scalar engine
now exposes the versioned contract, a batched NumPy environment is available,
and E1 has a measured local baseline. Work is stopping now to honor the
user's instruction to stop and update this handoff before the remaining token
pool threshold is reached.

### Implemented files

- `racing_event/observation.py`
  - Defines observation schema `1.0`, action schema `1.0`, and training
    contract `1.0`.
  - Defines the default 23-field order, field names, units, and bounds.
  - Provides `RunningObservationNormalizer` with streaming statistics,
    freeze mode, and JSON round-trip support.
  - Provides `TrainingContract` with action semantics, observation schema,
    track/config hashes, deterministic evaluation metadata, and optional
    normalizer state.
- `racing_event/physics.py`
  - Rejects malformed/non-finite scalar actions.
  - Adds `CarStateBatch` and vectorized `step_car_batch` using the same fixed
    timestep equations as scalar physics.
- `racing_event/env.py`
  - Wires in the observation schema and training contract.
  - Rejects NaN/infinite actions before clipping.
  - Adds `reset(evaluation=True)` for fixed start-line, standing, zero-noise
    evaluation starts.
  - Makes the explicit `longitudinal_velocity` field clear in the v1 model.
- `racing_event/vector_env.py`
  - Adds `VectorizedRacingEnv` and `RacingVectorEnv` alias.
  - Supports one or more training tracks, random training spawn, fixed
    evaluation spawn, batched actions, observations, rewards, signals,
    events, state, and termination/truncation arrays.
  - Preserves scalar behavior for a one-environment fixed-start sequence.
- `racing_event/track.py`
  - Caches static segment lengths, boundary/barrier nesting, and ray segment
    arrays. This reduces repeated geometry setup while preserving the scalar
    geometry authority.
- `racing_event/__init__.py`
  - Exports the vector environment and contract types.
- `tests/test_vector_env.py`
  - Covers scalar parity, batch determinism, evaluation reset, and non-finite
    action rejection.
- `tests/test_observation_contract.py`
  - Covers schema shape/names/bounds and contract/normalizer persistence.
- `tests/test_env.py`
  - Covers scalar fixed evaluation reset.
- `tools/benchmark_vector_env.py`
  - Reproducible E1 benchmark for full environment, policy plus environment,
    batched physics, and geometry/reward/observation overhead.
- `docs/state_schema.md`
  - Human-readable v1 action and observation contract.
- `README.md`
  - Documents the batched engine, contract, and benchmark.
- `reports/e1_vectorized_benchmark.md`
  - Measured local benchmark report.

### Measured E1 result

Command, from `/home/redacted/Documents/RL-ENV`:

```bash
python3 tools/benchmark_vector_env.py --steps 10 --n-envs 1 2 4 8 16 --output reports/e1_vectorized_benchmark.md
```

The current local report measures approximately 101 environment steps/s for
each tested batch size, approximately 97 to 101 policy-plus-environment
steps/s, and approximately 9.8 ms of geometry/reward/observation overhead per
environment step. Physics-only throughput scales much better, so geometry and
sensing remain the hot path. These are local measurements, not a public
throughput guarantee.

### Checks

- Dependency restored: `python3 -m pip install --user 'numpy>=1.24,<3'`;
  installed NumPy `2.4.6` for the active Python 3.11 environment.
- Full test suite: **passed** —
  `python3 -m unittest discover -s tests -v`; 28 tests passed.
- Scripted smoke run: **passed** —
  `OUTPUT="$(mktemp)"; python3 tools/run_scripted.py --steps 10 --output "$OUTPUT"`;
  wrote 10 frames and completed with the expected timeout DNF for this short
  smoke horizon.
- E1 benchmark: **passed/measured** — report saved at
  `reports/e1_vectorized_benchmark.md`.
- PPO trainer: **not implemented**.
- E2 baseline convergence: **not run**.
- ONNX evaluation: **not implemented**.

### Next session action

Start with the fixed trainer slice, not another environment rewrite:

1. Review `racing_event/observation.py` and `docs/state_schema.md` as the
   frozen input/output boundary.
2. Add a trainer-facing rollout/metrics interface that consumes
   `VectorizedRacingEnv`, updates a saved normalizer during training only,
   enforces the existing tamper-evident budget, and logs step count, returns,
   every reward signal, lap completion, speed, crash/off-track rates, action
   saturation/smoothness, and per-track statistics.
3. Keep PPO hyperparameters fixed and explicit. Do not claim convergence until
   E2 is run and its report records lap completion and speed against steps.
4. Re-run the full suite and add trainer tests before changing the public
    budget or tick-rate decisions.

## 22. SESSION HANDOFF (2026-10-15, BUDGETED ROLLOUT SLICE)

### Goal and result

The next trainer boundary is implemented. The vector environment now supports
selective lane reset and masked stepping, so a final partial batch can consume
an exact number of environment transitions without advancing inactive lanes.
`RolloutMetrics` and `RolloutCollector` now provide the fixed, testable input to
the future PPO implementation. PPO updates and E2 convergence are still not
implemented and must not be described as complete.

### Implemented files

- `racing_event/vector_env.py`
  - Adds `reset(indices=...)` for unique, validated lane indices.
  - Adds `step(..., active_mask=...)` that advances only selected lanes.
  - Inactive lanes retain physics state, counters, track assignment, and done
    state. Their returned rewards, signals, events, terminated flags, and
    truncated flags are zero/empty/false.
  - Existing full-batch reset/step behavior and scalar parity remain intact.
- `racing_event/metrics.py`
  - Adds batch metrics accumulation across calls.
  - Records raw signal totals, weighted signal totals, episode outcomes, speed,
    off-track rate, action saturation, action deltas, and per-track summaries.
  - Rejects malformed or non-finite input before mutation and emits JSON-safe
    snapshots with `None` for unavailable denominators.
- `racing_event/rollout.py`
  - Adds `PPOHyperparameters` with explicit provisional settings.
  - Adds `RolloutCollector`, which accepts a callable policy, reserves the
    persistent `RunBudget` in transition units immediately before stepping,
    auto-resets completed lanes only when more transitions are needed, updates
    a running normalizer only in training mode, and returns a full-lane
    `RolloutBatch` plus an active-transition mask.
  - Adds `RolloutResult` and serializable metric output. This is a rollout
    interface, not a PPO trainer.
- `racing_event/__init__.py`
  - Exports the rollout and metrics-facing trainer types.
- `tests/test_vector_env.py`
  - Adds selective-reset, masked-step, done-lane, and validation coverage.
- `tests/test_rollout_metrics.py`
  - Adds denominator, cross-call episode, masked-lane, weighted-signal, empty
    snapshot, and atomic-validation coverage.
- `tests/test_rollout.py`
  - Adds exact partial-budget, auto-reset, normalizer/contract persistence,
    bad-policy, and hyperparameter coverage.
- `README.md`
  - Documents the trainer boundary and warns that PPO/E2 values remain
    provisional.

### API details for the next session

- `VectorizedRacingEnv.reset(indices=None)` keeps the old full reset. A
  selective reset accepts unique integer lane indices. Explicit spawn arrays
  remain scalar or full `(n_envs,)` arrays; selected lanes use corresponding
  positions.
- `VectorizedRacingEnv.step(actions, active_mask=None)` validates all actions,
  then advances only `True` lanes. A done lane is legal when inactive and is
  rejected when active.
- `RolloutCollector.collect(steps)` requires a positive integer transition
  count and rejects a request larger than the current per-run budget before
  stepping. The returned `RolloutBatch` has shape `(step_calls, n_envs, ...)`;
  use `batch.active_mask` to flatten valid PPO transitions.
- `RolloutCollector` expects a policy callable from float32 observations to
  finite `(n_envs, 2)` raw actions. Actions outside `[-1, 1]` are allowed for
  saturation logging; the environment remains authoritative for clipping.
- Save a frozen normalizer with
  `env.training_contract.save(path, normalizer=collector.normalizer)` after
  calling `collector.normalizer.freeze()`. Evaluation must call the normalizer
  with `training=False` and use deterministic mean actions.
- The implemented budget remains the current README rule: `n` logical runs by
  `k` transitions per run, with persistent hash-chained reservations. This is
  still a provisional event choice even though the older design register marks
  cumulative-versus-per-run as open.

### Checks

- Focused rollout tests: **passed** —
  `python3 -m unittest discover -s tests -p 'test_rollout*.py' -v` from
  `/home/redacted/Documents/RL-ENV`; 11 tests passed.
- Full test suite: **passed** —
  `python3 -m unittest discover -s tests -v` from the project root; 43 tests
  passed.
- Syntax/compile check: **passed** —
  `python3 -m py_compile racing_event/*.py tools/*.py tests/*.py`.
- E1 benchmark: **not rerun in this slice**. The prior measured local report
  remains `reports/e1_vectorized_benchmark.md` at about 101 environment
  steps/s for the tested local batch sizes. It is not a public guarantee.
- PPO trainer/update: **not implemented**.
- E2 baseline convergence: **not run**.
- ONNX evaluation and tournament runner: **not implemented**.

### Next session action

Implement the smallest fixed PPO update loop on top of `RolloutBatch`: choose
and document the policy/value serialization format, compute masked GAE with
termination versus truncation handled explicitly, perform one fixed PPO update,
and log approximate KL, clip fraction, value loss, explained variance, and
environment/update steps per second. Then run E2 with the baseline reward and
write a measured convergence report. Do not change the public budget, tick
rate, track count, or architecture policy from measured results in that slice.

## 23. SESSION HANDOFF (2026-10-16, FIXED PPO AND E2 SLICE)

### Goal and result

The fixed PPO update slice from Section 22 is implemented and measured. The
environment, rollout collector, PPO model, optimizer, artifact checks, and
baseline E2 experiment now form one tested training path. This session did not
freeze the event step budget, tick rate, track count, architecture freedom, or
ONNX/tournament contract.

### Implemented files

- `racing_event/ppo.py`
  - Adds `compute_masked_gae` with explicit termination versus truncation
    handling. Terminations do not bootstrap. Truncations bootstrap but do not
    carry GAE across the reset boundary. Inactive lanes are ignored.
  - Adds `FixedMLPActorCritic`: two 64-unit tanh layers, two-action
    tanh-squashed Gaussian policy, scalar value head, and learned two-entry
    global log standard deviation.
  - Adds `PPOTrainer.update` with full-batch clipped PPO epochs, normalized
    active-transition advantages, gradient clipping, and diagnostics for policy
    loss, value loss, entropy, approximate KL, clip fraction, explained
    variance, and update throughput.
  - Adds state-dictionary-only artifact save/load with separate JSON metadata.
    Loading uses `weights_only=True` and validates the fixed schema,
    distribution, architecture, tensor names, shapes, and finite values.
- `racing_event/rollout.py`
  - Adds explicit `value_loss_coefficient` to `PPOHyperparameters` and tightens
    learning-rate, gradient-limit, and clip-range validation.
- `racing_event/observation.py`
  - Changes the trainer action-distribution contract to `tanh_normal`.
- `examples/baseline_reward.json`
  - Adds the documented baseline reward weights.
- `tools/run_e2_baseline.py`
  - Runs a reproducible local E2 loop over the synthetic fixture and writes
    per-update lap completion, speed, crash/off-track, return, PPO diagnostic,
    and environment/update throughput measurements.
- `reports/e2_baseline_convergence.md`
  - Contains the measured 32-update, 8,192-transition local run with seed 17,
    four environments, and 64 steps per environment per rollout. The run did
    not establish lap completion; this is recorded as evidence that the
    baseline still needs tuning, not as a public performance claim.
- `tests/test_ppo.py`
  - Covers masked GAE boundaries, inactive lanes, action bounds, weight change,
    inactive-transition exclusion, constant-return explained variance, and
    safe artifact round-trip/rejection.
- `docs/state_schema.md`, `README.md`
  - Document the fixed PPO model and safe artifact boundary.
- `.python-version`, `requirements.lock`, `Dockerfile`, `Makefile`
  - Pin Python/dependencies and provide `make test`, E1, E2, and a container
    test entry point. The lock uses the CPU PyTorch wheel index.

### Measured E2 result

Command:

```bash
.venv/bin/python tools/run_e2_baseline.py --updates 32 --n-envs 4 --rollout-length 64 --output reports/e2_baseline_convergence.md
```

The report records 8,192 environment transitions. The local environment
throughput was roughly 59 to 67 transitions/s in the report's per-rollout
windows; the PPO optimizer throughput was roughly 52,000 to 81,000 update
transitions/s.
Lap completion remained zero in the measured windows. Speed, crash, and
off-track metrics vary by rollout, so the report must not be read as a claim of
convergence. Repeat E2 after adding the intended multi-track training set and
before selecting a public step budget.

### Checks

- Dependency setup: **passed** — created `.venv`, installed the editable
  package, NumPy `2.4.6`, and CPU PyTorch `2.7.1+cpu`; `pip check` reported no
  broken requirements.
- Focused PPO tests: **passed** —
  `.venv/bin/python -m unittest discover -s tests -p 'test_ppo.py' -v`; 8 tests
  passed.
- Full test suite: **passed** —
  `.venv/bin/python -m unittest discover -s tests -v`; 51 tests passed.
- Syntax check: **passed** —
  `.venv/bin/python -m py_compile racing_event/*.py tools/*.py tests/*.py`.
- E2: **measured** — report saved at
  `reports/e2_baseline_convergence.md`; no convergence claim is made.
- E1 rerun: **measured** — `make e1` refreshed
  `reports/e1_vectorized_benchmark.md`; the current local environment column is
  about 69 to 72 transitions/s across `n_envs` 1 to 16, with approximately
  13.9 to 14.5 ms of geometry/reward/observation overhead per transition.
- ONNX evaluation, trajectory export, held-out-track runner, ranking, and
  Unity playback: **not implemented**.

### Next session action

Investigate the poor E2 baseline before changing public settings. First add a
small deterministic evaluation helper that runs the saved deterministic policy
from the fixed start line and reports lap/DNF outcomes separately from rollout
metrics. Then diagnose whether the no-lap result comes from reward scale,
episode spawn/termination behavior, track difficulty, or PPO rollout/update
length. Use an ablation or a longer measured run, and record the result before
freezing any budget or architecture decision. Keep the ONNX/tournament work
after the training behavior is credible.

## 24. SESSION HANDOFF (2026-10-16, DETERMINISTIC EVALUATION SLICE)

### Goal and result

The next engine slice is implemented: saved PPO artifacts can now be checked
against the exact environment contract and evaluated from the fixed start line
with deterministic mean actions. Evaluation is deliberately one-lane and
streaming so a long run does not accumulate trajectory frames in memory.

### Implemented files

- `racing_event/evaluation.py`
  - Adds `EvaluationResult`, `evaluate_policy`, and strict contract/hash
    validation.
  - Uses `reset(evaluation=True)`, a frozen normalizer path, fixed `dt` lap
    timing, physical outcome metrics, and a streaming SHA-256 transition hash.
  - Distinguishes `finished`, `dnf_crash`, `dnf_timeout`, `dnf_stuck`, and
    `dnf_backward` outcomes.
- `tools/evaluate_policy.py`
  - Loads the safe state-dictionary PPO artifact plus `TrainingContract`.
  - Evaluates sequentially and rejects a repeated deterministic run if its
    trajectory digest changes.
- `tools/run_e2_baseline.py`
  - Adds optional `--artifact-dir` output for policy weights, metadata, and a
    frozen training contract/normalizer.
- `racing_event/env.py` and `racing_event/vector_env.py`
  - Reject custom spawn, speed, lateral, or heading overrides in evaluation
    mode, and validate scalar reset inputs before state mutation.
- `tests/test_evaluation.py` and `tests/test_vector_env.py`
  - Cover deterministic hashes, normalizer immutability, artifact contract
    mismatches, DNF classification, and canonical evaluation starts.
- `README.md` and `docs/state_schema.md`
  - Document the low-memory evaluation workflow and artifact boundary.

### Checks

- Focused evaluation tests: **passed** — 4 tests with
  `.venv/bin/python -m unittest discover -s tests -p 'test_evaluation.py' -v`.
- Full test suite before this handoff update: **passed** — 55 tests with
  `.venv/bin/python -m unittest discover -s tests -v`.
- Syntax check: **passed** — `.venv/bin/python -m py_compile racing_event/*.py tools/*.py tests/*.py`.
- Artifact/evaluation smoke: **passed** — a 2-update, 2-lane E2 run saved a
  policy and frozen contract under a temporary directory; two
  deterministic evaluations at a 16-step horizon both returned
  `dnf_timeout` with the identical trajectory hash
  `d292786217a1658deb70150a095d5ab56f600e8b44c4ab9c2b67842931754166`.
  This temporary path is not a repository deliverable.
- E2 diagnosis: **not complete** — the smoke run only proves the artifact and
  evaluator path. It does not explain the earlier zero-lap result.

### Next session action

Run the evaluator against a longer, measured E2 artifact and record the
deterministic fixed-start outcome beside the stochastic rollout metrics. Then
use a bounded ablation to identify whether the zero-lap result comes from
reward scale, random-spawn coverage, track geometry/difficulty, or PPO update
settings. Keep the run small enough to avoid memory pressure, and do not freeze
the public step budget, tick rate, or track count until the evidence is written.

## 25. SESSION HANDOFF (2026-10-16, E2 DIAGNOSIS AND ENGINE HARDENING)

### Goal and result

The longer E2 artifact was run and evaluated. The first 32-update artifact
showed that the throttle prior removed the zero-speed local failure, but a
64-update run with the old PPO settings later collapsed. A bounded PPO
stability ablation found a better provisional update configuration. The current
engine passes the physics, scalar/vector parity, reset, reward, rollout, GAE,
artifact, and deterministic-evaluation checks.

### Measured diagnosis

- Old 32-update PPO settings (`10` epochs, `3e-4` learning rate,
  `0.01` entropy) produced a finished fixed-start lap at `50.7333 s` after
  `8,192` transitions, but the corresponding 64-update run later ended
  `dnf_stuck` after `186` steps. Its final approximate KL was `0.0267` and clip
  fraction was `0.3164`.
- Stabilized settings (`4` epochs, `1e-4` learning rate, `0.005` entropy)
  remained competent after `16,384` transitions. The deterministic artifact
  finished in `923` steps (`30.7667 s`) on both repeated evaluations. The final
  approximate KL was `0.000062` and clip fraction was `0.0`.
- The stabilized evaluation used `204` off-track steps. A stronger baseline
  off-track weight of `-4.0` was tested and rejected because its policy crashed
  after `272` steps with `146` off-track steps.
- The synthetic fixture is physically completable: the scripted controller
  finished in `863` steps (`28.7667 s`). The remaining gap is policy quality,
  not a proven impossible track.

### Implemented changes

- `racing_event/config.py` and `config/simulation.json`
  - Add `standing_start_probability=0.25` to expose evaluation-like starts
    during training while retaining random spawns.
- `racing_event/reward.py`, `racing_event/signals.py`,
  `examples/baseline_reward.json`
  - Add a one-shot `stuck` signal and baseline weight `-8.0`, preventing an
    early negligible-progress truncation from being a cheap reward escape.
- `racing_event/physics.py`
  - Couple longitudinal acceleration and lateral grip with a friction-circle
    budget.
  - Reject non-positive or non-finite `dt` and grip factors.
  - Preserve scalar/batched parity.
- `racing_event/ppo.py`
  - Model artifact format is now version `2`.
  - Fixed policy mean bias is `[0.0, 0.5]` for steering and throttle/brake.
    The throttle prior helps standing starts leave the zero-speed local basin.
  - The bias is stored and validated in artifact metadata.
- `racing_event/rollout.py` and `tools/run_e2_baseline.py`
  - Adopt the measured provisional PPO settings: four update epochs,
    learning rate `1e-4`, and entropy coefficient `0.005`.
- Tests now cover 61 cases, including random scalar/vector spawn parity,
  one-shot stuck rewards, friction-circle behavior, invalid physics inputs,
  and policy-bias artifact metadata.
- `reports/e2_deterministic_evaluation.md` records the full comparison and
  reproduction commands.

### Checks

- Full suite: **passed** — 61 tests with `make test`.
- Physics-focused suite: **passed** — scalar/batched parity, deterministic
  repeatability, grip coupling, acceleration/braking, and invalid input tests.
- PPO/rollout suite: **passed** — masked GAE, safe artifact loading, inactive
  lanes, exact budget behavior, normalizer persistence, and provisional
  hyperparameter serialization.
- E2 artifact: **passed/measured** — 64 updates, 4 environments, 64-step
  rollout, 16,384 transitions, frozen normalizer, repeated deterministic
  fixed-start evaluation, identical trajectory hash.
- UI/Unity: **not implemented**. Python remains the only physics authority.

### Next session action: UI boundary

Begin the renderer-facing slice without moving any game logic into Unity:

1. Freeze the Python trajectory/state message fields and schema version.
2. Add a Python trajectory exporter that records the state and event stream
   needed by a renderer without changing simulation outcomes.
3. Add a lightweight local replay preview or HTML canvas first, using only
   exported Python frames.
4. Then build the Unity playback adapter against the same versioned data.

Do not add Unity-side collision, lap timing, reward, progress, or physics.

## 26. SESSION HANDOFF (2026-10-16, RESPAWN REVIEW AND REPLAY UI)

### Goal and current state

The user asked to investigate random respawns, defer questions until after the
session, and begin the UI boundary in checked layers. Research and the first
read-only replay UI are complete. Training/physics rules were not changed.
Source changes are on `master`: research/reset regression in `0023a8a`, replay
export/viewer in `65f505e`.

### Respawn findings and deferred decisions

- The primary text of Czechmanowski et al. (2025), arXiv `2504.02420v1`,
  Section III-E, explicitly uses random track-position resets after exceeding
  limits to improve exploration. It does not provide an isolated respawn
  ablation proving unseen-track generalization or prevention of memorization.
- Our collector already makes a new random episode after a crash/end. A new
  regression verifies that the crash transition is retained, the spawn changes,
  no free lap is awarded, and the persistent budget continues counting.
- Keep soft runoff recovery and the 25% standing-start mixture unchanged.
  Random respawn is not a replacement for local-only observations, varied
  training geometry, and held-out evaluation.
- Sources, limits, and E6/E3 experiment recommendations are saved in
  `research_notes/Track sourcing and racing physics/racing_rl_controls.md`.

**Questions saved for discussion after the session, not blockers here:**

1. Can the user share the paper title/link they recall? We found a direct
   precedent, but have not identified their exact paper.
2. Keep soft runoff recovery, or run an equal-budget training-only ablation of
   immediate reset on any off-track entry? Recommendation: keep recovery for
   now; a reset must remain an episode boundary with failure/budget accounting.
   Evaluation crashes remain final with no respawn.
3. Before freezing spawn rules, approve an equal-budget multi-seed E6/E3 test
   measuring section coverage and unseen public-development-track performance.
   Do not use private tournament tracks for a public experiment report.

### Implemented layers

- `racing_event/trajectory.py`: replay schema **1.1**, initial tick-zero pose,
  exact embedded geometry, car dimensions, engine/config/track identities,
  Python-authored result, independent frame snapshots, streaming/atomic writer,
  bounded loader, strict validation and payload checksum.
- `racing_event/evaluation.py`: optional one-way `frame_sink`; recording does
  not change the existing trajectory hash or evaluation result. Frame consumers
  cannot mutate state through the copied snapshot or return an action.
- `tools/export_replay.py`: export smoke-controller or saved-policy evaluation
  using the same Python transition path. No optimizer runs during export.
- `tools/build_replay.py`: validate then build one offline, self-contained HTML.
  Reject source overwrites and escape embedded text against script injection.
- `racing_event/assets/replay.html`, `replay.css`, `replay.js`: responsive
  canvas track/car view, play/pause/restart/next tick, seeking, speed, recorded
  telemetry/events, Python results, and provenance. Only pose is interpolated.
- `docs/replay_contract.md`: time-zero/tick conventions, SI units, checksum
  algorithm, limits, legacy 1.0 rejection, and future Unity axis/yaw mapping.
- `README.md`: export/build/serve instructions, optional JS test explanation,
  and container preview commands. Package data includes viewer assets;
  `.dockerignore` excludes local artifacts.

No Unity adapter is implemented yet. The old `run_scripted.py` output remains
a legacy 1.0 smoke record; it is deliberately rejected by the new 1.1 viewer.
Replay checksums are integrity checks, not anti-cheat or authenticated training
provenance. The viewer is a local tool, not a hardened public upload service.

### Checks and real-run evidence

All commands ran from the project root.

- **Tests passed:** `make test` ran 77 unittest cases, including telemetry,
  scalar/vector parity, metrics, and the JS-runner case, with no skips here.
- **JS checks passed:** `node --test tests/replay_frontend.test.cjs` ran 59
  playback/validation/DOM-control tests. They use a DOM/canvas double, not a
  real browser. On machines without Node, `make test` explicitly skips this
  optional runner; Python/export checks still run offline.
- **Syntax passed:** `node --check racing_event/assets/replay.js`,
  `.venv/bin/python -m py_compile racing_event/*.py tools/*.py tests/*.py`,
  and `git diff --check`.
- **End-to-end saved-policy export passed:** `artifacts/replay/e2-stable.json`
  contains 924 frames (initial + 923 steps), a `30.7666666667 s` finish, and the
  unchanged E2 trajectory hash
  `eac69718eeed370c7d23a393a6d4ce65f838704b8665408d634e160be909ee13`.
- **Scripted export passed:** `artifacts/replay/scripted.json` contains 864
  frames, an `863`-step / `28.7666666667 s` finish with zero off-track steps.
- Both JSONs passed strict Python read-back. The generated E2 HTML also passed
  the JS validator, exact final-pose checks, and embedded JS syntax compilation.
- Preview HTML is `artifacts/replay/index.html` (508,279 bytes); the scripted
  comparison is `artifacts/replay/scripted.html`. Artifacts are Git-ignored and
  reproducible with the documented commands. No training sweep, browser, or
  additional package installation was started for this session.
- **HTTP smoke passed:** local preview returned HTTP 200 with embedded data at
  `http://localhost:8765/`. Server PID `1176` in this session serves only
  `artifacts/replay`, bound to loopback. It may not survive session/container
  restarts; restart with the README's `http.server` command if needed.
- **Telemetry smoke passed:** a scripted export/build/read-back produced 864
  frames and 863 steps with schema `1.1`, tick `0`, and final tick/result
  consistency.
- **Coverage semantics:** a reset reports sector `0` but starts with zero driven
  coverage; the first transition marks the reached sector. This prevents random
  respawns or selective resets from being counted as policy achievement.
- **Not run:** real-browser screenshot/visual layout verification, Unity mesh
  parity, and Docker image build. No browser or Unity tool is available in this
  session. No dedicated lint/typecheck configuration exists.

### Exact next action

Open the local E2 and scripted previews in a real browser and review layout,
playback controls, tick-zero pose, final result/event timing, and mobile sizing;
this remains blocked because no browser binary is installed in the current
container. Next, implement the lowest-risk viewer-only QoL slice: an event-marker
rail with exact-tick jumps, then keyboard help/reduced-motion behavior. Keep
Unity read-only and defer camera modes, multi-ghost comparison, and multi-episode
training replays until the browser layer is stable.
