# Racing gameplay and engine quality of life

**Research date:** 2026-10-16
**Scope:** low-risk gameplay and engine improvements for the Python-authoritative
RL racing simulator. Sources were freshly checked on 2026-10-16. No code,
assets, track files, or UI materials are copied from the references.

## Executive recommendation

The simulator already has the right fairness boundary for a first release:
Python owns geometry, physics, progress, rewards, termination, evaluation, and
trajectory hashes; the future renderer receives recorded results and does not
re-simulate. The safest next step is **diagnostic telemetry that does not enter
the observation, reward, termination, or ranking contract**.

| Recommendation | Decision | Why |
|---|---|---|
| Add sector/checkpoint counters and event timestamps to `info` and rollout metrics | **Safe to implement now** | Uses existing authoritative `s`, `distance_since_spawn`, tick count, and event boundary. It helps debugging without changing policy behavior. |
| Add progress coverage, no-progress, runoff, boundary-excess, and off-track-progress metrics | **Safe to implement now** | Exposes reward-hacking and wall-grinding behavior without guessing a new penalty. |
| Keep 30 Hz fixed-step physics, float64 internal state, deterministic mean-action evaluation, and trajectory hashes | **Safe to implement now / preserve** | This is already the strongest fairness feature in the repository. Add more regression checks rather than replacing it. |
| Log raw policy command, bounded applied command, steering angle, and command deltas separately | **Safe to implement now** | The engine already rate-limits steering and clips commands; the distinction should be visible. |
| Improve replay/presentation for sector splits, recovery, warnings, and exact events | **Safe to implement now** | Renderer-only presentation can consume recorded telemetry and must not infer outcomes. |
| Add longitudinal slew-rate filtering or a new low-pass actuator model | **Experiment first** | Changes the transition function, action latency, PPO distribution, and likely lap-time trade-offs. |
| Penalize wall grinding, reset immediately on off-track, or auto-recover/teleport | **Experiment and user approval** | These change the task and can reward or punish policies for reasons not currently in the contract. |
| Freeze a multi-track leaderboard formula | **User approval after a scoring experiment** | The repository evaluates one track at a time but does not yet freeze `O8` ranking or DNF policy. |
| Add checkpoint events to replay schema 1.1 | **Defer or version first** | Current replay events are a closed set. A schema 1.2 extension is reasonable, but it is not a silent v1.1 addition. |

## Repository fit and current behavior

### Authority and state flow

- [`README.md`](../../README.md) states that Python is authoritative and Unity is
  a future consumer of recorded state. The renderer must not compute contacts,
  resets, rewards, progress, finishes, or lap times.
- [`racing_event/env.py`](../../racing_event/env.py) and
  [`racing_event/vector_env.py`](../../racing_event/vector_env.py) use the same
  fixed-step model at `dt = 1 / tick_hz`. The scalar and batched paths project
  the car onto the shared `TrackGeometry`, compute wrap-aware `delta_s`, and
  accumulate `distance_since_spawn`.
- [`config/simulation.json`](../../config/simulation.json) currently pins 30 Hz,
  a 2-D action, 2,400 maximum steps, a 90-tick stuck limit, a 25 m backward
  limit, a steering-rate limit, and reduced runoff grip. These are part of the
  hashed training/evaluation contract.
- [`racing_event/trajectory.py`](../../racing_event/trajectory.py) writes replay
  frames atomically, validates tick order and terminal events, and records
  fixed-step elapsed time. `engine.code_hash`, `config_hash`, track hashes, and
  a streaming trajectory hash already support auditability.

### Reward, termination, and existing metrics

- [`racing_event/signals.py`](../../racing_event/signals.py) currently exposes
  `progress`, `speed`, `curvature_matched_speed`, `off_track`, `crash`, `stuck`,
  `lap_complete`, `time`, and `steer_smoothness`.
- Progress is a **wrap-aware per-transition delta**, normalized by lap length. It
  is not absolute lap position, so a random spawn does not grant free progress.
- Runoff is recoverable: the engine reduces grip by `runoff_grip_factor` and
  emits an off-track penalty. Barrier contact is terminal. This is a good
  gameplay split for a learning environment because a mistake can produce a
  recovery example rather than immediately discarding the episode.
- Stuck is currently defined by `abs(delta_s) < 0.01` for 90 consecutive ticks;
  backward termination uses accumulated distance. This catches stationary
  policies but does not by itself identify a car that advances while scraping a
  wall.
- [`racing_event/metrics.py`](../../racing_event/metrics.py) already records
  per-track signal sums, lap completions, crashes, off-track rate, mean speed,
  action saturation, and mean action delta. The additions below should extend
  this diagnostic layer before changing reward weights.
- [`racing_event/evaluation.py`](../../racing_event/evaluation.py) starts at the
  fixed start line with zero noise, uses the deterministic policy mean, freezes
  normalization, and reports fixed-step lap time or a DNF reason. This is the
  correct authority for event scoring; participant reward is a training aid,
  not a leaderboard score.

### Important contract boundary

The current observation intentionally contains no absolute position, absolute
heading, track id, or absolute lap progress; see
[`docs/state_schema.md`](../../docs/state_schema.md). Sector/checkpoint data
should therefore begin as hidden diagnostic `info`, replay telemetry, and
evaluation output. Adding it to observations or reward signals would change
the policy contract and requires an experiment plus a version bump.

## 1. Checkpoint and sector telemetry

### What public references suggest

- The [F1TENTH Gym documentation](https://f1tenth-gym.readthedocs.io/en/latest/)
  describes an explicitly stepped environment designed for reproducibility and
  exposes lap timing and checkpoint-completion state in its environment API.
  Its [GitHub repository](https://github.com/f1tenth/f1tenth_gym) is MIT licensed
  according to its metadata and license file. This is a useful design reference
  only; the project has a different geometry and authority boundary.
- The [HighwayEnv racetrack documentation](https://highway-env.farama.org/environments/racetrack/)
  makes versioned racetrack configurations and termination choices visible,
  including an explicit off-road termination option. Its
  [GitHub repository](https://github.com/Farama-Foundation/HighwayEnv) is MIT
  licensed. The useful lesson is to make diagnostics/configuration explicit and
  versioned, not to copy its reward or lane model.
- The 2026 paper [Expert Knowledge-driven Reinforcement Learning for Autonomous
  Racing via Trajectory Guidance and Dynamics Constraints](https://arxiv.org/abs/2603.05842)
  uses geometry-aware references, curvature-based guidance, dynamics constraints,
  and staged training. This supports collecting section-level curvature and
  stability telemetry; it does not justify adding privileged global progress to
  this project’s observation.

### Safe design now: derive telemetry from the existing `s` authority

Start with deterministic, equal-arclength sectors. A first diagnostic layout
could use 8 or 16 sectors at fixed fractions of `track.length`; the exact count
is a display choice and should not affect physics. At every step, retain:

```text
tick
s_wrapped
distance_since_spawn_m
lap_fraction_from_spawn
evaluation_sector_index
sector_elapsed_ticks
last_checkpoint_tick
checkpoints_crossed_this_step
```

Recommended semantics:

1. Use the already computed unwrapped transition distance, not wall-clock time.
2. For training episodes, make the primary sector index spawn-relative so a
   random spawn does not expose absolute track identity to a policy. For fixed
   evaluation, also report the absolute `start_line_s`-relative sector index.
3. Detect crossings from the directed interval between the previous and current
   unwrapped progress values. If a large step crosses more than one boundary,
   report all crossed sector IDs in deterministic order.
4. Keep sector split times as integer ticks and convert to seconds only as
   `ticks / tick_hz`. Never use `time.perf_counter()` for a score.
5. Make the first implementation diagnostic only: no reward weight, no
   termination, no observation slot, and no change to the existing progress
   signal.

This can be implemented as a small pure helper used by scalar and vector
environments, with scalar/vector parity tests. It should return new immutable
data for each `step`, matching the environment API principle documented by
[Gymnasium’s `Env` API](https://gymnasium.farama.org/api/env/).

### Replay and schema choice

The existing replay contract supports only these events:
`off_track_start`, `off_track_end`, `barrier_contact`, and `lap_complete`.
[`docs/replay_contract.md`](../../docs/replay_contract.md) also requires exact
integer ticks and says that the renderer must not infer outcomes. Therefore:

- **Now:** put sector/checkpoint data in Python `info`, evaluation summaries,
  and a separate diagnostic export or viewer-derived index. Keep replay schema
  1.1 valid.
- **Later:** if Unity needs sector markers, add a versioned `telemetry` object
  or new checkpoint event vocabulary in replay schema 1.2. Validate crossing
  order, tick timing, and terminal-event precedence before exposing it.
- Do not add a `checkpoint_k` string to the existing v1.1 event set without a
  schema/version decision; the current validator intentionally fails closed.

### Trade-offs

| Choice | Benefit | Cost/risk |
|---|---|---|
| Equal-arclength sectors | Track-comparable and easy to hash | A visually important corner may fall across a boundary |
| Curvature/corner-defined sectors | Better coaching and presentation | Track metadata grows; boundaries need review and private-track handling |
| Diagnostic-only sectors | No policy/reward shift | Does not directly solve sparse-reward learning |
| Sector reward bonus | Can improve credit assignment | Can be farmed by oscillation or wrong-way crossing; changes participant incentives |

Recommendation: use equal-arclength diagnostic sectors now, and only add
corner-aware sectors after the track set is stable. Do not turn either into a
reward until an ablation shows improved unseen-track lap time without making
wrong-way or wall-hugging behavior more attractive.

## 2. Progress and event instrumentation

### Safe-now telemetry packet

Extend `info` and `RolloutMetrics` with quantities that are already available or
cheaply derived from the authoritative transition:

- `delta_s_m` and `positive_progress_m` after wrap handling;
- `distance_since_spawn_m` and `lap_fraction_from_spawn`;
- `progress_direction` (`forward`, `stationary`, `backward`), for diagnostics;
- `off_track_transition` (`start`, `continue`, `end`, `none`);
- `stuck_ticks` and a non-terminal `stuck_warning` threshold, for example at
  50% and 80% of the configured limit;
- `maximum_left_excess_m` and `maximum_right_excess_m`, already returned by
  `TrackGeometry.edge_status`;
- action command versus applied command, steering angle, and per-step deltas;
- `event_tick`, `event_order`, and a compact reason/status field for terminal
  transitions.

The warning is presentation/diagnostic state only. It must not truncate an
episode early. Store it in metrics and replay diagnostics rather than adding it
to `SIGNAL_NAMES` by default.

### Event invariants to test

1. `off_track_start` occurs once on the false-to-true transition and carries the
   same tick as the first off-track frame.
2. `off_track_end` occurs once on the true-to-false transition.
3. `barrier_contact` is terminal and remains the terminal reason if a lap
   threshold is reached on the same transition; this matches the current replay
   validator’s crash precedence.
4. `lap_complete` is emitted only after one full `distance_since_spawn` lap.
5. A terminal event is not emitted for an inactive vector lane.
6. A reset starts a new episode and never creates a fake progress or checkpoint
   crossing at the teleport.
7. Events are ordered by simulation tick and never reconstructed by Unity.

The [Gymnasium step API](https://gymnasium.farama.org/api/env/) and its
[time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
support keeping terminal state separate from external truncation. The current
PPO path already distinguishes `terminated` from `truncated`; instrumentation
must preserve that distinction.

## 3. Anti-stall and anti-wall-grind metrics

### What is already covered

The engine already has four useful defenses/observations:

- low-progress consecutive ticks lead to `stuck` truncation;
- large accumulated backward motion terminates the episode;
- runoff changes grip and applies an off-track signal;
- barrier contact uses the car footprint and terminates the episode.

The remaining blind spot is a trajectory that makes positive progress while
spending a large fraction of time off the drivable surface or repeatedly
approaching the boundary. A hard-coded “wall grind” penalty could punish a
legitimate recovery or a narrow but valid racing line, so measure it first.

### Safe-now metrics, not new rules

Add these to per-episode and per-track summaries:

```text
off_track_steps
off_track_duration_s
max_contiguous_off_track_steps
positive_progress_m_while_off_track
positive_progress_fraction_while_off_track
mean_abs_lateral_offset_while_off_track
max_boundary_excess_m
stalled_steps
max_stalled_run_steps
backward_progress_m
barrier_contact_count
```

Definitions should be explicit:

- `positive_progress_m_while_off_track` sums `max(delta_s_m, 0)` only on
  off-track transitions. It is a diagnostic, not a reward.
- `positive_progress_fraction_while_off_track` divides that value by total
  positive progress, with a zero denominator producing `null` rather than NaN.
- `max_contiguous_off_track_steps` measures persistence, not just entry count.
- `max_boundary_excess_m` is the maximum of the existing left/right excess
  fields. It distinguishes a shallow runoff excursion from severe edge abuse.
- `barrier_contact_count` remains separate from a near-wall warning; contact is
  the authoritative crash event.

Use a `wall_grind_candidate` label only in analysis if a run exceeds a
documented combination such as high off-track-progress fraction plus persistent
boundary excess. Do not turn that label into automatic DNF or a score until the
team has reviewed false positives.

### Experiment-gated changes

Run equal-budget, equal-seed comparisons on all public development tracks for:

1. Current soft runoff plus current crash rule.
2. Current rule plus a small continuous edge-excess penalty.
3. Current rule plus an off-track-progress cap or terminal wall-grind rule.

Measure finished-lap rate, normalized lap time, crash rate, off-track duration,
off-track progress fraction, stuck rate, and unseen-track performance. A rule
that reduces off-track time but increases crashes or reduces recovery quality is
not an automatic improvement.

The [HighwayEnv racetrack configuration](https://highway-env.farama.org/environments/racetrack/)
is a useful reminder that `terminate_off_road` is a task configuration choice,
not a universal racing rule. The 2026 [TraD-RL paper](https://arxiv.org/abs/2603.05842)
also treats dynamics constraints and stability costs as explicit design choices
requiring validation. Neither source supports silently changing this project’s
soft-runoff contract.

## 4. Fixed-step determinism and replay authority

### Preserve the current approach

The current fixed-step path is appropriate:

- `tick_hz` is configuration, and physics receives `dt` explicitly;
- scalar and batch physics use float64 state and the same equations;
- evaluation uses a fixed start, deterministic mean action, frozen normalizer,
  config/track hashes, and a streamed trajectory digest;
- replay time is `steps / tick_hz`, not wall-clock duration.

The [F1TENTH Gym documentation](https://f1tenth-gym.readthedocs.io/en/latest/)
explicitly emphasizes simultaneous stepping, seeded randomness, and reproducible
experiments. Glenn Fiedler’s [Fix Your Timestep!](https://gafferongames.com/post/fix_your_timestep/)
explains why variable `dt` makes physics behavior depend on render rate and why
fixed simulation steps can be decoupled from rendering. These references support
the repository’s design; they are not a reason to copy an integration loop.

### Low-risk determinism checks

Add or retain tests for:

1. Same seed, same action sequence, same scalar observations/rewards/info.
2. Scalar and one-lane vector parity over reset, normal transitions, wraparound,
   off-track recovery, crash, stuck, and lap completion.
3. Repeated fixed-start evaluation produces the same result and trajectory hash.
4. Changing only a policy action after a checkpoint changes only the expected
   suffix of the trajectory.
5. Replay frame zero is the reset pose; final tick equals `result.steps`.
6. The outcome does not depend on wall-clock load or renderer availability.
7. `terminated` and `truncated` remain separate for value bootstrapping.

Do not replace NumPy float64 physics with `decimal` or fixed-point arithmetic as
a first step. Python’s [decimal documentation](https://docs.python.org/3/library/decimal.html)
describes correctly rounded decimal arithmetic, but a numeric representation
rewrite would be a new physics contract with a throughput and compatibility
cost. Shared-machine evaluation plus hashes is the practical guarantee for this
project; bit-identical results across every BLAS/CPU platform should be treated
as a separate portability goal.

### Tick-rate experiment

The handoff already calls for a 20/30/60 Hz check. Keep 30 Hz provisionally and
compare fixed action sequences at each candidate rate. Report:

- finish/DNF status;
- fixed-step lap time;
- off-track duration and crash count;
- action saturation and smoothness;
- trajectory digest within the rate-specific contract;
- visual interpolation quality.

Do not expect hashes or exact lap times to match across different tick rates.
Choose one rate, then include it in the config hash and freeze it for an event.

## 5. Actuator smoothing and command authority

### Existing smoothing

`racing_event/physics.py` already limits steering-angle change by
`steering_rate_rad_s * dt`. The car state retains the previous command, and the
reward has a steering-smoothness signal. `RolloutMetrics` records action deltas.
This is already a low-risk actuator-quality feature because it is explicit,
fixed-step, and shared by scalar and batch physics.

### Safe-now instrumentation

Expose three layers distinctly in diagnostics and replay exports:

1. `policy_action_raw`: finite policy output before authoritative clipping;
2. `command_applied`: bounded `[-1, 1]` command sent to physics;
3. `actuator_state`: actual steering angle and resulting speed/acceleration.

The current replay contract already records the bounded applied commands, not the
unbounded policy request. Preserve that contract and add raw-action fields only
to a diagnostic artifact if needed. Never let Unity smooth a pose and then use
that pose as a new simulation input.

Use separate metrics for steering delta and longitudinal-command delta. A single
sum can hide a throttle/brake chatter problem behind a calm steering signal.

### What needs an experiment

Possible future models include:

- a longitudinal slew-rate limit;
- a first-order command low-pass filter;
- explicit throttle/brake actuator delay;
- a speed-dependent steering-rate or steering-angle limit;
- a higher-fidelity tire/ABS/traction model.

These can make the car more physical, but they add latency and can prevent an
agent from making a necessary last-moment correction. They also change the
Markov transition and the saved policy/environment contract. If tried, compare
no filter, steering-only filter, and both-axis filter with the same seeds and
budget. Store the actual applied command and filter parameters in the config
hash.

The official [Isaac Lab actuator documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html)
describes explicit actuator models, delays/limits, command computation, and
clipping as mechanisms that can differ from ideal physics. It is good evidence
that actuator behavior deserves an explicit model and artifact identity; it is
not a reason to import Isaac Lab into this small Python-native simulator.

## 6. Recovery and off-track presentation

### Keep physics recovery and presentation separate

The current behavior is suitable for a first event:

- entering runoff emits `off_track_start`;
- the car continues with reduced grip;
- returning emits `off_track_end`;
- barrier contact remains terminal;
- stuck and backward outcomes remain visible DNF reasons.

Safe presentation additions can show:

- an off-track banner and exact start/end tick;
- a recovery timer based on simulation ticks;
- runoff grip state and boundary-excess telemetry;
- a non-terminal “stuck risk” warning;
- a barrier-contact marker and terminal DNF reason;
- sector split degradation after an off-track interval.

The replay viewer or future Unity adapter should consume recorded values only.
This follows the current [replay contract](../../docs/replay_contract.md) and
the same principle used by Gymnasium’s separation of `terminated`, `truncated`,
and `info` diagnostics.

### Defer automatic recovery rules

Do not teleport the car back to the centerline, grant a reset bonus, or silently
continue after barrier contact in the scored environment. A training-only
respawn can improve state coverage, but it changes episode boundaries and must
not be confused with a completed lap. The existing
[`racing_rl_controls.md`](../Track%20sourcing%20and%20racing%20physics/racing_rl_controls.md)
follow-up recommends retaining soft runoff recovery and testing immediate
off-track reset only as an explicit ablation.

If an optional “assist mode” is desired for a demo, mark the run as assisted,
keep it out of the leaderboard, and record the assist event in Python. The
presentation layer must not inject it into the authoritative trajectory.

## 7. Fair race scoring

### Separate training reward from competition score

Participant reward weights are intentionally editable. They must never directly
determine leaderboard ranking. The event runner should score Python-authoritative
outcomes using a frozen scoring config, held-out tracks, fixed start, deterministic
mean action, shared engine/config hashes, and one published retry policy.

This separation is supported by:

- [Stable-Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html),
  which recommends separate evaluation, deterministic actions, and tracking
  wrapper effects. Its evaluator statically distributes episodes across vector
  lanes to avoid bias.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560),
  which documents high variance and non-determinism as threats to meaningful
  comparisons and calls for standardized reporting.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
  and its [Apache-2.0 `rliable` implementation](https://github.com/google-research/rliable),
  which recommend interval estimates, performance profiles, and robust aggregate
  metrics rather than trusting a single mean. The repository was archived in
  2025, so use the methodology as a reference rather than adding it as a new
  dependency now.
- The official [AWS DeepRacer on AWS guide](https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/),
  which documents multi-track evaluation, timekeeping, lap validation, and
  combined leaderboards as separate event infrastructure concerns.

### Recommended scoring shape, pending approval

For `K` held-out tracks, use lower-is-better normalized time:

```text
track_score = lap_time_s / frozen_reference_par_s       if finished
track_score = dnf_penalty_multiplier                    if DNF
total_score = sum(track_score for each held-out track)
```

Freeze one reference par per track before submissions. It can come from a
published deterministic baseline/reference controller evaluated with the same
engine. A DNF penalty should be at least worse than a normal completed lap and
must be published before the event; using a normalized fixed penalty avoids
making a long track dominate solely because it is long. The exact multiplier is
an open event decision, not a code default.

Suggested secondary reporting:

- finish count;
- per-track normalized time and raw fixed-step time;
- DNF reason (`crash`, `stuck`, `backward`, `timeout`);
- off-track duration and positive progress while off-track;
- action saturation and smoothness;
- trajectory/config/track hashes.

Suggested tie-break order, subject to approval: more finished tracks, then lower
total normalized time among finished tracks, then best single-track normalized
time, then a deterministic published ID order. Do not use wall-clock runtime,
training reward, number of training attempts, or renderer FPS.

### Scoring trade-offs

| Formula | Advantage | Risk |
|---|---|---|
| Sum of raw lap times | Easy to explain | Long tracks dominate; track mix changes the contest |
| Mean normalized lap time | Comparable across track lengths | Needs frozen, defensible reference pars |
| Best single track | Rewards peak performance | Encourages specialization and ignores generalization |
| Finish count first | Rewards robust completion | Two safe but slow finishes may beat one excellent racer unless time is secondary |
| DNF as a large fixed penalty | Deterministic and easy to audit | The penalty magnitude can dominate the entire ranking |
| DNF as missing/omitted | Simple arithmetic | Rewards failing hard tracks and is unfair for generalization |

Recommendation: rank by mean or sum of normalized per-track time with an
explicit DNF penalty, while publishing raw per-track results. This matches the
train/test/generalization goal better than one best lap. The formula should be
implemented as a pure versioned scorer after the human resolves the existing
handoff question `O8`.

## Safe-now backlog

1. Add a pure sector/checkpoint accumulator based on directed unwrapped `s`.
2. Add diagnostic `delta_s`, sector, progress-direction, off-track-duration,
   boundary-excess, and off-track-progress fields to scalar/vector `info`.
3. Extend `RolloutMetrics` with max contiguous runoff, stuck-warning, backward,
   and off-track-progress summaries while leaving `SIGNAL_NAMES` unchanged.
4. Add exact crossing, wraparound, reset, inactive-lane, scalar/vector-parity,
   and deterministic-evaluation tests.
5. Add a diagnostic report command that prints per-sector splits and physical
   metrics without changing the replay v1.1 payload.
6. Expose recorded recovery/event state in the existing replay viewer; keep
   renderer interpolation visual-only.
7. Publish a scoring-protocol document and a pure scorer only after `O8` is
   approved; keep it separate from participant reward configuration.

## Major changes requiring experiments or approval

- Add sector bonuses, progress gates, or new reward signals.
- Put absolute sector/track progress into observations.
- Add a new low-pass/slew-rate actuator model or change tick rate.
- Change soft runoff into immediate reset or DNF.
- Add a wall-grind penalty or automatic near-wall disqualification.
- Add checkpoint events/telemetry fields to replay schema 1.1 instead of a new
  version.
- Use training reward, wall-clock training speed, or renderer timing in ranking.
- Add dynamic opponents, car-to-car collision, multi-agent race rules, or a
  second physics authority.
- Replace the current fixed-start deterministic evaluation with randomized
  starts in the leaderboard.

## Focused experiment plan

| Experiment | Controlled comparison | Acceptance evidence |
|---|---|---|
| Sector telemetry | Same policy/action stream with telemetry on/off | Same observations, rewards, terminations, and trajectory hash; correct wrap/crossing ticks |
| Anti-grind diagnostics | Current metrics on scripted, baseline, and deliberately wall-hugging policies | Metrics separate legitimate runoff recovery from persistent off-track progress |
| Wall-grind rule | No new rule vs continuous excess penalty vs terminal threshold | Equal-seed completion, normalized lap time, crash/off-track time, and unseen-track results |
| Actuator smoothing | Existing steering rate vs steering-only and two-axis filtering | Lap time, crash rate, recovery, action saturation, and command-lag diagnostics |
| Tick rate | 20/30/60 Hz with rate-specific hashes | Choose one rate from stability, cost, and visual quality; do not compare hashes across rates |
| Scoring | Raw time vs normalized time and several published DNF penalties | Ranking stability across held-out track mixes and clear per-track explanations |
| Evaluation uncertainty | Repeated deterministic runs plus multiple training seeds | Report deterministic repeatability separately from training-seed variability |

## Source register

### Official documentation and standards

- [Gymnasium `Env` API](https://gymnasium.farama.org/api/env/) — `step` returns
  observations, rewards, termination/truncation, and diagnostics; seeding and
  fresh return objects are documented. Gymnasium is MIT licensed:
  [license](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE).
- [Gymnasium handling time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
  — distinguishes terminal states from external truncation and explains the
  bootstrapping consequence.
- [Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/) — documents
  separate action, observation, and reward transformations, useful for keeping
  optional preprocessing outside the authoritative environment.
- [Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
  and [evaluation helper](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)
  — normalization, separate deterministic evaluation, repeated runs, wrapper
  effects, and evaluation bias.
- [F1TENTH Gym documentation](https://f1tenth-gym.readthedocs.io/en/latest/)
  and [MIT GitHub repository](https://github.com/f1tenth/f1tenth_gym) — explicit
  stepping, seeded reproducibility, lap/checkpoint state, and an autonomous
  racing reference with a permissive license.
- [HighwayEnv racetrack docs](https://highway-env.farama.org/environments/racetrack/)
  and [MIT GitHub repository](https://github.com/Farama-Foundation/HighwayEnv) —
  versioned racetrack configuration, off-road policy choice, and fixed-shape
  observation patterns.
- [Isaac Lab actuators](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html)
  — official actuator-modeling guidance on command delays/limits, clipping, and
  the difference between ideal and explicit actuators.
- [Python `decimal` documentation](https://docs.python.org/3/library/decimal.html)
  — correctly rounded decimal arithmetic; cited to explain why a numeric rewrite
  is a separate determinism project, not a free improvement.

### Papers and engineering references

- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
  — reproducibility, variance, and standardized reporting.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
  — uncertainty-aware aggregate evaluation; the paper is available under
  CC-BY-4.0 according to arXiv metadata.
- [Policy Invariance under Reward Transformations for General-Sum Stochastic Games](https://arxiv.org/abs/1401.3907)
  — potential-based shaping can preserve equilibria under stated assumptions;
  use as theory guidance, not as proof that arbitrary racing reward edits are
  harmless.
- [Expert Knowledge-driven Reinforcement Learning for Autonomous Racing via Trajectory Guidance and Dynamics Constraints](https://arxiv.org/abs/2603.05842)
  — fresh 2026 autonomous-racing work on curvature/racing-line guidance and
  explicit dynamic stability constraints; major-change evidence only.
- [Fix Your Timestep!](https://gafferongames.com/post/fix_your_timestep/) — fixed
  simulation time, render decoupling, and the reproducibility implications of
  variable `dt`.

### Public repositories, used as references only

- [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) — MIT.
- [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) — MIT.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — MIT.
- [`rliable`](https://github.com/google-research/rliable) — Apache-2.0; archived
  by its owner in 2025, so do not add it as a dependency for this small slice.
- [AWS DeepRacer on AWS](https://github.com/aws-solutions/deepracer-on-aws) —
  Apache-2.0 repository and official multi-track/lap-validation reference;
  do not copy assets or vendor its infrastructure.

All repository references above were read for behavior, API, methodology, or
license facts. No implementation or asset is copied into this project.
