# Live policy inference and racing loops

**Research date:** 2026-10-23  
**Scope:** Open-source racing, robotics, and game-AI systems that run policy or controller actions against a live/fixed-step simulator. The emphasis is on fixed-step clocks, the policy-to-simulation boundary, simultaneous agents and ghost evaluation, action deadlines and late actions, and streaming state to a renderer.

## Executive findings

1. **The best common shape is a three-stage loop:** capture an authoritative state snapshot, obtain/validate an action for that snapshot, then advance the simulation by an integer number of fixed physics steps. Rendering observes a published state packet; it is not part of the physics decision boundary.
2. **Control frequency and physics frequency should be separate concepts.** F1TENTH, ML Racing, dm_control, IsaacGymEnvs, Unity ML-Agents, and RLGym all show variants of action repeat/decimation: a policy action is held for a known number of physics ticks.
3. **Simultaneous-agent semantics need an explicit barrier.** PettingZoo's `ParallelEnv`, F1TENTH's all-agent `Simulator.step`, and RLGym's action dictionary all gather actions before the transition. This prevents agent order from allowing one policy to observe a partially updated world.
4. **A simulation timeout is not an inference deadline.** ML Racing explicitly has no hard wall-clock inference deadline. Synapse's `IsolatedPolicy` is stronger because a parent process can kill a stuck worker, but its current two-second action timeout is an outer operational default, not a 30 Hz control deadline.
5. **Late-action behavior must be a versioned rule.** RLGym's `rlbot_delay` models a deliberate one-tick delay; Unity repeats the stored action between decisions; Donkeycar logs loop jitter and continues; ML Racing DNFs an inference error but does not interrupt a hung call. These are different semantics and should not be left implicit.
6. **State streaming should be keyed by simulation frame/tick, not renderer time.** CARLA's synchronous mode waits for world and sensor data with the same frame ID, while RLGym's RLViser renderer receives the current state and an incrementing tick count. A browser or Pygame viewer can interpolate between packets, but should never invent or feed back simulation state.
7. **For Synapse, keep the current ONNX batch-one CPU contract and Python-authoritative engine.** The lowest-risk live path is operational: warm one session, use a monotonic deadline, make action freshness explicit, publish immutable state snapshots, and retain the current replay exporter as an offline/read-only presentation path.

## Repository comparison at a glance

| Repository | Domain | Concrete loop pattern | Late/deadline behavior | Renderer/state pattern |
|---|---|---|---|---|
| [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing) | ONNX racing ghosts | 25 Hz policy, 100 Hz physics, action held for four physics ticks | Invalid inference DNF; explicitly no hard per-inference wall deadline | Independent worlds; viewer combines positions only |
| [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) | Racing simulator | Explicit `timestep`; all cars receive controls before collision/scan updates | No policy deadline in the Gym API | `render_obs` copied after each simulator step; human rendering adds sleep |
| [RLGym/rlgym](https://github.com/RLGym/rlgym) | Rocket League game AI | Action parser -> transition engine; `RepeatAction` expands actions over ticks | Optional RLBot-like one-tick delay; repeated actions are explicit | Renderer consumes `GameState` and tick metadata |
| [Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | Multi-agent game AI API | `ParallelEnv.step()` receives an action dict for all live agents | AEC API has explicit dead-agent final step; parallel wrapper asserts cycle order | `render()` is an environment output, separate from action collection |
| [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) | Game/robotics engine | Academy `FixedUpdate` orders state send, decision, and action events | Stored action can repeat between decisions; no hard deadline in the loop | Agent observations/actions are synchronized through Academy events |
| [carla-simulator/carla](https://github.com/carla-simulator/carla) | Robotics/autonomy simulator | Synchronous mode + fixed delta; `world.tick()` gates sensors | Sensor queue timeout is the operational failure point | Every world/sensor sample is checked against a common frame ID |
| [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | GPU robotics/RL | `controlFrequencyInv` repeats physics after `pre_physics_step` | No policy deadline; vectorized environments return reset masks | Viewer/rendering can be forced per physics substep or decoupled |
| [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control) | Robotics control | `control_timestep` must be an integer multiple of physics timestep | No wall deadline; exact control-step contract | Observation is produced after `before_step`/physics/`after_step` |
| [microsoft/AirSim](https://github.com/microsoft/AirSim) | Robotics/vehicle simulator | `simPause`, `simContinueForTime`, and `simContinueForFrames` | RPC timeout/cancellation is external to the sim clock | Images, kinematics, and traces are explicit API streams |
| [autorope/donkeycar](https://github.com/autorope/donkeycar) | Live vehicle loop | Parts pipeline at a target rate, with optional threaded sensors | Blocking parts lower actual rate; jitter is logged and loop continues | Shared memory/pipeline connects sensors, pilot, and actuators |

The source links and observations below are the useful implementation details, not just project descriptions.

## Racing repositories

### 1. B3rK-3/ml-racing: closest ONNX ghost-race analogue

**Repository:** [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing)  
**Files:** [README.md](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/README.md), [agents.md](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/agents.md), [examples/ghost_policy.py](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_policy.py), [examples/ghost_race.py](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_race.py)

#### Concrete patterns

- The policy contract is static batch-one ONNX: `obs` `[1,156]` to `action` `[1,2]`.
- Four 10 ms physics ticks make one 40 ms policy interval: 100 Hz physics, 25 Hz policy.
- `GhostTrial.step()` validates the action once, converts it to the simulator control format, and holds the converted action through `CONTROL_STEPS` calls to `env.step()`.
- The two entrants are independent single-car worlds. They share the track, start pose, and sensor seed, but one ghost is never inserted into the other ghost's observations.
- Scoring is based on simulated elapsed time, not time consumed by ONNX inference or rendering. The render path only displays snapshots and sleeps toward real time.
- The main loop checks each trial independently. If one trial finishes or DNFs, the other continues.
- The runner intentionally says it has **no hard per-inference wall-clock deadline**, memory quota, or sandbox. Its simulated timeout cannot interrupt a hung inference call.
- Invalid/nonfinite inference output, collision, off-track footprint, and simulated timeout all DNF only that entrant. There is no fallback controller.

#### Why it matters for Synapse

This is the closest public comparison for Synapse's current `run_onnx_race.py`: fixed action hold, deterministic ghost worlds, no interaction, and simulation-time results. It also documents the exact gap a live evaluator must close: a simulated episode timeout does not provide a wall-clock action deadline. Keep the independent-ghost architecture for comparisons, but add a control-cycle deadline and a recorded miss policy if the run is intended to be genuinely live.

### 2. f1tenth/f1tenth_gym: fixed physics and all-agent transition ordering

**Repository:** [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)  
**Files:** [f110_env.py](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/f110_env.py), [base_classes.py](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/base_classes.py)

#### Concrete patterns

- `F110Env` has an explicit `timestep` defaulting to `0.01`. `step()` calls the simulator, then increments `current_time` by exactly that timestep.
- `Simulator.step(control_inputs)` first calls `RaceCar.update_pose()` for every car, storing all new poses. It then checks collisions, updates each car's opponent poses, updates scans, and constructs the observations dictionary.
- This ordering is important: all agents' actions affect the transition before any agent's next observation is finalized.
- The car model contains an explicit steering delay buffer, which is a simulator-level action-age effect rather than a policy-loop accident.
- `F110Env` keeps a small `render_obs` structure with poses and lap counters. `render()` pushes that snapshot to `EnvRenderer`, processes events, draws, flips, and only then sleeps 5 ms in human mode. `human_fast` omits the sleep.

#### Why it matters for Synapse

Use the `Simulator.step` ordering as the reference if Synapse ever adds true interacting cars: collect all actions from the same pre-step state, apply them together, advance physics, then calculate collision/sensor/termination results. Use the `render_obs` idea for a renderer packet, but keep rendering and any pacing sleep out of the authoritative state transition.

## Game-AI and multi-agent repositories

### 3. RLGym/RLGym: explicit policy boundary, action repeat, and intentional one-tick delay

**Repository:** [RLGym/rlgym](https://github.com/RLGym/rlgym)  
**Files:** [rlgym/api/rlgym.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/api/rlgym.py), [transition_engine.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/api/config/transition_engine.py), [action_parser.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/api/config/action_parser.py), [repeat_action.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/action_parsers/repeat_action.py), [rocketsim_engine.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/sim/rocketsim_engine.py), [rlviser_renderer.py](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/rlviser/rlviser_renderer.py)

#### Concrete patterns

- `RLGym.step(actions)` is a clear policy-to-simulation boundary: action dict -> `ActionParser.parse_actions()` -> `TransitionEngine.step()` -> shared-info update -> observation/reward/termination calculation.
- `TransitionEngine` owns mutable state and the actual simulation. The policy-facing layer does not implement physics.
- `RepeatAction` turns one policy action into an `(repeats, action_dim)` array. The repeated action is therefore part of the action parser contract, not a hidden loop detail.
- `RocketSimEngine.step()` validates that it received an action for every car, then loops over the action rows. Each row sets controls for all cars before the arena advances one tick.
- `RocketSimEngine` has an explicit `rlbot_delay` option. When enabled, the arena advances one tick before applying the first action, producing a deliberate one-tick no-op/delay. This is a useful model for action freshness and should not be confused with a random scheduler delay.
- `RLGym.render()` calls a renderer with the current state and shared info. `RLViserRenderer` increments a `packet_id` and sends `tick_count` and `tick_rate` along with ball/car state.

#### Why it matters for Synapse

RLGym provides the cleanest vocabulary for a future live loop: **state**, **action parser**, **transition engine**, **repeat count**, and **renderer packet**. If Synapse chooses a one-tick action delay for realism, expose it as a fixed contract setting and apply it to every entrant; never let it emerge from a slow inference call. The `packet_id`/tick-rate pattern is also a good model for state streaming to a browser renderer.

### 4. Farama-Foundation/PettingZoo: simultaneous actions versus sequential turns

**Repository:** [Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo)  
**Files:** [env.py](https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/main/pettingzoo/utils/env.py), [conversions.py](https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/main/pettingzoo/utils/conversions.py)

#### Concrete patterns

- `ParallelEnv.step(actions)` accepts a dictionary keyed by agent ID and returns next observations, rewards, terminations, truncations, and infos keyed by agent.
- The class documentation explicitly defines this as stepping every live agent at once.
- `AECEnv` intentionally has a different contract: one `agent_selection` acts at a time, and a terminated agent must take one final vacuous `step(None)` so its final transition is observable before removal.
- `aec_to_parallel_wrapper` asserts that agents are selected in a complete cycle and rejects environments where termination happens part-way through a cycle. It gathers all actions, lets the AEC environment execute its full cycle, then returns synchronized dictionaries.
- Both APIs expose a `state()` hook for a global state appropriate to centralized training, separate from each agent's observation.

#### Why it matters for Synapse

For simultaneous racing agents, prefer a parallel action barrier with stable agent IDs. If an agent terminates early, define whether it is removed, held at a terminal pose, or receives a neutral action; do not allow a missing dictionary entry to shift the timing of other agents. For replay/live packets, preserve both per-agent observation/action status and a global frame/tick.

### 5. Unity-Technologies/ml-agents: fixed-update event ordering and action holding

**Repository:** [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)  
**Files:** [DecisionRequester.cs](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/DecisionRequester.cs), [Agent.cs](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Agent.cs), [Academy.cs](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Academy.cs), [ActuatorManager.cs](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Actuators/ActuatorManager.cs)

#### Concrete patterns

- `AcademyFixedUpdateStepper.FixedUpdate()` calls `Academy.EnvironmentStep()`, making the environment step live in Unity's fixed-update phase.
- `Academy.EnvironmentStep()` has a stable event sequence: `AgentPreStep`, increment counters, `AgentIncrementStep`, `AgentSendState`, `DecideAction`, then `AgentAct`.
- `DecisionRequester.DecisionPeriod` requests a new decision only on selected Academy steps. `TakeActionsBetweenDecisions` controls whether the agent reapplies its stored action on the intervening steps.
- `Agent.RequestDecision()` also requests an action; `RequestAction()` explicitly replays the most recent action without asking for a new policy decision.
- The Agent documentation says observations/actions normally occur during `FixedUpdate`; `Update` is for cosmetic purposes.
- `ActuatorManager` preallocates stored action buffers, updates them from policy output, and executes them later through `OnActionReceived`. This separates action delivery from action execution.
- Sensors are updated and observations collected before the policy request. Actions execute only after the decision phase.

#### Why it matters for Synapse

Unity is a strong reference for an event-ordered loop with lower-rate decisions and held actions. It is not evidence of a hard wall-clock deadline: the fixed update is a synchronization boundary, not a policy timeout. Synapse can borrow the event ordering—snapshot sensors, request decision, validate/store action, execute action—without introducing Unity or an engine-side renderer authority.

## Robotics and simulator repositories

### 6. carla-simulator/carla: synchronous world ticks and frame-matched sensor queues

**Repository:** [carla-simulator/carla](https://github.com/carla-simulator/carla)  
**Files:** [PythonAPI/examples/synchronous_mode.py](https://raw.githubusercontent.com/carla-simulator/carla/master/PythonAPI/examples/synchronous_mode.py), [PythonAPI/examples/manual_control.py](https://raw.githubusercontent.com/carla-simulator/carla/master/PythonAPI/examples/manual_control.py)

#### Concrete patterns

- `CarlaSyncMode` applies `synchronous_mode=True` and `fixed_delta_seconds=1/fps`.
- Each loop calls `world.tick()`, then retrieves world and sensor queue entries. `_retrieve_data()` discards entries until the requested frame arrives, and `tick()` asserts that every returned object has the same frame ID.
- The manual-control example separates server simulation FPS from client display FPS. HUD state is updated from `on_world_tick(timestamp)`, which records `timestamp.frame` and `timestamp.elapsed_seconds`.
- Camera and sensor callbacks are asynchronous producers, but the synchronous control loop consumes only the frame that corresponds to the authoritative world tick.

#### Why it matters for Synapse

Use a frame ID in every live state packet. If a renderer or sensor stream is delayed, it can be labeled stale rather than silently treated as current. A live browser viewer can consume the newest complete packet while the simulation retains its own tick. For camera-based policies, a frame-matched sensor barrier is more important than raw render FPS.

### 7. isaac-sim/IsaacGymEnvs: action decimation and vectorized physics

**Repository:** [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)  
**Files:** [tasks/base/vec_task.py](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/tasks/base/vec_task.py), [docs/framework.md](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/docs/framework.md)

#### Concrete patterns

- `VecTask` exposes `controlFrequencyInv`, the number of simulator steps between RL actions.
- `step(actions)` clips actions, calls the task's `pre_physics_step`, executes `gym.simulate(self.sim)` `control_freq_inv` times, fetches CPU results when needed, then calls `post_physics_step` to compute observations/rewards/resets.
- The returned observation dictionary, reward tensor, reset tensor, and extras are produced only after the complete control interval.
- Rendering may be forced at each physics substep, but the viewer has its own synchronization toggle and `sync_frame_time`; render pacing is not the policy/physics contract.
- The design is vectorized: one call advances many independent environments with the same control boundary.

#### Why it matters for Synapse

The `pre_physics_step -> repeat fixed simulation -> post_physics_step` split is directly reusable for a higher-throughput training path. Keep it distinct from live one-entrant latency: vectorized throughput and a renderer's frame rate should not determine the official policy deadline.

### 8. google-deepmind/dm_control: exact control timestep as an integer multiple

**Repository:** [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control)  
**File:** [dm_control/rl/control.py](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/rl/control.py)

#### Concrete patterns

- The environment accepts either `n_sub_steps` or `control_timestep`, never both.
- `compute_n_steps()` rejects a control timestep smaller than the physics timestep or not an integer multiple of it.
- `step(action)` calls `task.before_step(action)`, advances physics by exactly `_n_sub_steps`, then calls `task.after_step`, reward, observation, and termination functions.
- `control_timestep()` reports the resulting interval explicitly.

#### Why it matters for Synapse

Make the relationship between `tick_hz`, policy/control cadence, and action hold count mechanically validated. Do not let a floating-point control interval silently produce a variable number of physics steps.

### 9. microsoft/AirSim: externally clocked simulation and explicit sensor APIs

**Repository:** [microsoft/AirSim](https://github.com/microsoft/AirSim)  
**File:** [PythonClient/airsim/client.py](https://raw.githubusercontent.com/microsoft/AirSim/master/PythonClient/airsim/client.py)

#### Concrete patterns

- `VehicleClient` exposes `simPause`, `simContinueForTime(seconds)`, and `simContinueForFrames(frames)`. These APIs make the external controller's simulation-clock boundary explicit.
- State and sensor reads are explicit RPC calls: ground-truth kinematics, collision info, camera images, image batches, IMU, GPS, and Lidar.
- `simGetImages()` returns a batch of image responses, while `simGetGroundTruthKinematics()` provides current state for control/telemetry.
- Trace-line and plot APIs can annotate the simulator without changing the policy observation.

#### Why it matters for Synapse

AirSim is a useful model when the policy process is outside the simulator: pause, request a bounded state/sensor snapshot, return an action, then continue a known number of frames. The RPC boundary must carry tick/frame identity and timeout behavior; otherwise an external renderer or stale sensor response can be mistaken for the current state.

## Live vehicle loop

### 10. autorope/donkeycar: practical rate loop and explicit jitter profiling

**Repository:** [autorope/donkeycar](https://github.com/autorope/donkeycar)  
**Files:** [donkeycar/vehicle.py](https://raw.githubusercontent.com/autorope/donkeycar/master/donkeycar/vehicle.py), [donkeycar/parts/keras.py](https://raw.githubusercontent.com/autorope/donkeycar/master/donkeycar/parts/keras.py)

#### Concrete patterns

- `Vehicle.start(rate_hz=10)` runs a sequence of parts, then sleeps for the remaining period.
- Parts can be threaded, allowing camera/sensor acquisition to run independently while the main loop invokes the pilot and actuator parts.
- If parts take longer than the target period, the loop logs a jitter warning and continues; the docstring explicitly says actual frequency may be lower.
- `PartProfiler` reports min/max/average plus p50, p90, p99, and p99.9 after discarding warm-up/incomplete samples.
- `KerasPilot.run()` normalizes the sensor input and delegates to `inference()`; the output is a steering/throttle pair consumed as another pipeline part.

#### Why it matters for Synapse

Borrow the part-level timing breakdown and percentile reporting. Do not borrow the “warn and continue” late-action rule for a strict scored race unless the event explicitly defines it. A competition loop should turn an over-budget policy call into a recorded policy outcome, not merely a log message.

## ONNX Runtime evidence for live policy calls

These are official runtime references rather than racing repositories:

- [Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) distinguishes intra-op threads, inter-op graph parallelism, sequential execution, and thread spinning. For a small batch-one policy, explicit one-thread/sequential settings reduce contention and scheduling variability, but should still be benchmarked.
- [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html) documents `InferenceSession.run`, `run_async`, `run_with_iobinding`, `RunOptions.terminate`, `SessionOptions.use_deterministic_compute`, and explicit execution providers.
- [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) is primarily valuable when inputs/outputs are already on an accelerator and CPU-device copies dominate. It is not automatically a win for a CPU `[1,N] -> [1,2]` policy.
- [Profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) produce operator/thread traces for diagnosis. Profiling should be disabled during the acceptance race because it changes the timed path.

### Runtime implications

- Create one `InferenceSession` per artifact before tick zero; never load a session in the control tick.
- Warm up with representative inputs before starting the measured race clock. Record model hash, runtime version, provider, graph optimization level, thread settings, and relevant math-library thread variables.
- Measure input preparation, ORT call, output validation, physics/sensor work, complete cycle time, and tick-release jitter separately.
- Record p50/p95/p99/p99.9/max rather than only average latency.
- A Python thread/future timeout only stops the caller from waiting; it does not necessarily stop native inference. `RunOptions.terminate` is a cooperative runtime feature that needs a pinned-version experiment.
- A process/container boundary is the enforceable option for an untrusted or strict deadline: after timeout, kill the worker and do not send another request to it.

## What Synapse already has

The following was inspected locally to make the recommendations specific. No source files were changed.

### Policy and simulation boundary

- [`racing_event/evaluation.py`](../../racing_event/evaluation.py) calls the policy synchronously on a copied normalized observation, then calls `env.step(raw_action)`. `max_steps` is a fixed simulated-step limit. The optional `frame_sink` receives tick-zero and post-step snapshots, and its return value cannot send an action back into the engine.
- [`racing_event/onnx_artifact.py`](../../racing_event/onnx_artifact.py) validates a restricted static graph and uses a long-lived CPU session with one intra-op thread, one inter-op thread, `ORT_SEQUENTIAL`, disabled graph optimization/memory arena/pattern, and deterministic-compute configuration. The boundary is fixed batch-one `observations -> actions`.
- [`racing_event/isolated_policy.py`](../../racing_event/isolated_policy.py) already has the strongest deadline mechanism in the project: frozen input snapshot, separate worker/container, monotonic startup/action timeout, bounded JSONL, sequence IDs, validation, and worker teardown after any error/timeout.

### Ghost evaluation and rendering boundary

- [`racing_event/race.py`](../../racing_event/race.py) evaluates entrants sequentially and independently on the same explicit tracks. It records `independent_ghosts`; entrants do not collide or appear in each other's observations. The isolated mode stores startup/action timeout configuration in the report.
- [`racing_event/trajectory.py`](../../racing_event/trajectory.py) streams an atomic schema-1.1 replay writer, with tick-zero reset state followed by one frame per executed transition. The writer publishes only a complete recording.
- [`racing_event/replay_group.py`](../../racing_event/replay_group.py) validates common track/config/engine/tick/car/tick-zero identity for 1–4 recordings and defines `hold_final_pose` for shorter ghosts.
- [`racing_event/pygame_renderer.py`](../../racing_event/pygame_renderer.py) and [`racing_event/replay.py`](../../racing_event/replay.py) use a visual replay clock. They interpolate only recorded pose/heading and never step physics, calculate contacts, or send actions.
- [`renderer/app.js`](../../renderer/app.js) is also replay-driven: it fetches two complete JSON recordings, advances a browser playback clock, samples frames, and renders Three.js state. It is not currently a live state subscriber.

This means Synapse already separates authoritative Python simulation from presentation. The missing layer for a live run is a freshness/deadline-aware action scheduler and a state-stream transport, not a second simulator.

## Implementation recommendations for Synapse

### A. Define three clocks explicitly

Use distinct names and values in the runtime/report:

```text
sim_tick       integer physics transition index; authoritative
sim_time       sim_tick * fixed_dt; scoring/result time
control_tick   policy release index; action hold interval
wall_deadline  monotonic deadline for the action response
render_time    viewer-only clock or latest received packet time
```

Recommended loop shape:

```text
load/validate artifact and create session
warm session; start the measured wall clock
for control_tick:
    snapshot = immutable state at the control boundary
    observation = build observation from snapshot
    deadline = scheduled_release + policy_budget
    action = infer(snapshot, deadline)
    validate action and action freshness
    apply action for exactly N physics ticks
    publish state packet(s) after the transition
    if live pacing is enabled, pace wall time only
```

Do not change `dt` based on time spent in inference, skip physics to catch up, or charge renderer frame time as simulated lap time.

### B. Make the policy-to-simulation packet explicit

Every request/response should carry at least:

```text
run_id, entrant_id, track_hash, engine/config hash
control_tick, state_tick, sim_time
observation schema/version and observation bytes or digest
request monotonic timestamp and deadline
action, action validation result
completed timestamp, action_age_ticks, late_by_ns
```

The engine should apply an action only to the state tick for which it was requested. A response for an older tick is stale, even if it arrives before the next loop iteration. This prevents a slow response from being applied to a newer state accidentally.

### C. Choose late-action behavior before calling the loop live

Use separate named profiles rather than one implicit behavior:

| Profile | Late/invalid response | Intended use |
|---|---|---|
| `strict_scored` | Kill/reject the worker and mark entrant failed/DNF; no fallback action | Untrusted or official competition |
| `trusted_live` | Hold the last action for at most one configured interval, mark `stale_action=true`, and record age | Local live demo where continuity matters |
| `neutral_stop` | Apply a fixed neutral/safe action and end or quarantine the entrant | Safety-oriented integration testing |

The recommended official rule is `strict_scored`: deadline miss, invalid output, EOF, or sequence mismatch is an entrant failure, with no replacement controller. This matches Synapse's existing isolated-worker failure posture and ML Racing's no-fallback rule, while adding the missing hard wall cutoff.

If a hold-last-action mode is offered for demos, bound the hold duration and expose it in the report. RLGym's one-tick delay is a useful precedent only when it is intentional and applied consistently to every entrant.

### D. Separate two ghost modes

1. **Independent ghost evaluation:** retain Synapse's current model. One fresh environment per entrant, same explicit track/start/seed, no opponent features, sequential or equally provisioned isolated workers, simulation-time scoring.
2. **Interacting multi-agent race:** add a separate parallel action barrier. Build every observation from the same pre-step snapshot, gather all actions keyed by stable entrant/agent ID, validate the complete set, apply all actions, then advance physics and collision/sensor processing together.

Do not reuse the current independent ghost report to imply car-to-car interaction. Keep `race_type` explicit (`independent_ghosts` versus a future `interactive_agents`).

### E. Add a live state stream without weakening the replay contract

Keep schema 1.1 replay files for complete offline artifacts. Add a separate live packet shape, for example:

```json
{
  "kind": "synapse_live_state",
  "schema_version": "1.0",
  "run_id": "...",
  "tick": 123,
  "sim_time_s": 4.1,
  "tick_hz": 30,
  "track_hash": "...",
  "entrants": [{
    "participant_id": "team-a",
    "status": "running",
    "x": 1.0, "y": 2.0, "heading": 0.1,
    "speed": 3.2, "steering_command": 0.0,
    "throttle_brake": 0.4,
    "events": [],
    "action_tick": 123,
    "action_age_ticks": 0,
    "late": false
  }]
}
```

Recommended transport behavior:

- Publish only immutable post-transition packets from the simulation thread.
- Use a bounded queue or latest-state slot so a slow renderer cannot back-pressure physics.
- Include monotonically increasing `tick`/`sequence`; reject or ignore out-of-order packets.
- Publish after all entrant states for a shared tick are complete.
- A renderer may interpolate between adjacent received states, display a gap/stale indicator, or hold the last packet. It must not extrapolate a result, generate an event, or send an action.
- Preserve the current replay writer as an asynchronous evidence sink if recording is needed; avoid synchronous JSON serialization in the timed policy path.

CARLA's frame-matched queues and RLGym's `packet_id`/`tick_rate` are concrete precedents for this packet identity.

### F. Use a control cadence that is mechanically tied to physics

Synapse currently documents a provisional 30 Hz simulation and the ONNX profile is batch-one. If the live policy is called less often than physics:

- declare `physics_hz`, `control_hz`, and `action_hold_ticks` in the runtime profile;
- require `physics_hz % control_hz == 0`, or use dm_control's exact-multiple validation;
- store the action's source `control_tick` in each applied-action record;
- record any steering actuator rate limit separately from policy action age;
- do not let a late policy call silently change the hold count.

For an initial live implementation, a one-policy-call-per-engine-step path is easiest to audit. Add decimation only as an explicit contract setting after testing action age and controller stability.

### G. Strengthen ONNX runtime observability before optimizing it

- Keep the current explicit CPU provider, one intra/inter thread, sequential execution, and static batch-one shape as the baseline.
- Add a real warm-up phase before tick zero; the existing deterministic probe is useful validation but should not be confused with a measured warm-up policy.
- Add monotonic histograms for input preparation, `session.run`, output validation, complete control cycle, and release jitter.
- Record p50/p95/p99/p99.9/max, deadline misses, invalid actions, worker restarts, and the first differing tick in any repeated trajectory hash.
- Keep ORT profiling and per-tick logging out of the normal timed path. Run them in a diagnostic benchmark.
- Do not add batching or I/O binding to the official path without a contract/performance experiment. The current output is only two CPU floats; the main risk is scheduling/tail latency, not tensor-copy throughput.
- Tie the action deadline to the configured control interval. The existing two-second isolated timeout is suitable as a bounded worker default but cannot represent a live 30 Hz deadline.

### H. Tests that would distinguish a live loop from a replay loop

Use deterministic fake policies/workers and test at least:

1. policy returns before the deadline: action is applied to the requested tick;
2. policy returns exactly at, just before, and just after the deadline;
3. policy hangs: the parent returns a bounded failure and no subsequent request is sent to the same worker;
4. stale response from an older `state_tick`: rejected, never applied to the newer state;
5. invalid shape/nonfinite/out-of-range action: entrant failure with an explicit reason;
6. two simultaneous agents: both observe the same pre-step state and both actions affect one transition;
7. one agent terminates: remaining agents retain the same global tick and documented terminal-pose/removed-agent semantics;
8. renderer queue is slow or drops packets: physics continues, packet sequence remains monotonic, renderer never controls the run;
9. repeated run under the same artifact/config/runtime: identical simulated trajectory hash and tick count;
10. live run with rendering enabled versus disabled: identical authoritative results.

## Bottom line

Synapse's current independent ghost evaluator and read-only replay boundary are already aligned with the strongest racing precedent. The next live-inference slice should add a deadline-aware scheduler and immutable state stream around that boundary, rather than moving physics into a renderer or changing the ONNX graph contract. Use F1TENTH/RLGym/PettingZoo for simultaneous transition semantics, CARLA for frame-matched streaming, Unity/dm_control/IsaacGymEnvs for fixed-step and action-repeat structure, and the existing `IsolatedPolicy` for enforceable failure handling.

## Source index

- [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing)
- [B3rK-3/ml-racing README](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/README.md)
- [B3rK-3/ml-racing agents.md](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/agents.md)
- [B3rK-3/ml-racing ghost_policy.py](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_policy.py)
- [B3rK-3/ml-racing ghost_race.py](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_race.py)
- [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)
- [f1tenth_gym f110_env.py](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/f110_env.py)
- [f1tenth_gym base_classes.py](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/base_classes.py)
- [RLGym/rlgym](https://github.com/RLGym/rlgym)
- [RLGym main API](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/api/rlgym.py)
- [RLGym transition engine interface](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/api/config/transition_engine.py)
- [RLGym repeat action parser](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/action_parsers/repeat_action.py)
- [RLGym RocketSim engine](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/sim/rocketsim_engine.py)
- [RLGym RLViser renderer](https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/rlviser/rlviser_renderer.py)
- [Farama PettingZoo env API](https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/main/pettingzoo/utils/env.py)
- [Farama PettingZoo conversions](https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/main/pettingzoo/utils/conversions.py)
- [Unity ML-Agents DecisionRequester](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/DecisionRequester.cs)
- [Unity ML-Agents Agent](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Agent.cs)
- [Unity ML-Agents Academy](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Academy.cs)
- [Unity ML-Agents ActuatorManager](https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Actuators/ActuatorManager.cs)
- [CARLA synchronous mode example](https://raw.githubusercontent.com/carla-simulator/carla/master/PythonAPI/examples/synchronous_mode.py)
- [CARLA manual control example](https://raw.githubusercontent.com/carla-simulator/carla/master/PythonAPI/examples/manual_control.py)
- [IsaacGymEnvs VecTask](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/tasks/base/vec_task.py)
- [IsaacGymEnvs framework docs](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/docs/framework.md)
- [DeepMind dm_control control.py](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/rl/control.py)
- [Microsoft AirSim client.py](https://raw.githubusercontent.com/microsoft/AirSim/master/PythonClient/airsim/client.py)
- [Donkeycar vehicle.py](https://raw.githubusercontent.com/autorope/donkeycar/master/donkeycar/vehicle.py)
- [Donkeycar keras.py](https://raw.githubusercontent.com/autorope/donkeycar/master/donkeycar/parts/keras.py)
- [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ONNX Runtime I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [ONNX Runtime profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
