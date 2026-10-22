# Live ONNX race inference

**Date:** 2026-10-23  
**Scope:** Move Synapse from offline replay-based ONNX comparison to live, fixed-rate policy actions while keeping Python as the only simulation authority and keeping `renderer/` changes small.

## Decision

Add a live execution path beside the existing replay path:

```text
ONNX artifact/worker
        │  action request/response (internal JSONL for isolated workers)
        ▼
Python fixed-step race scheduler ──► Python physics/termination/scoring
        │                                  │
        │                                  └─ immutable post-transition snapshot
        ▼                                             │
loopback WebSocket, state only ◄─────────────────────┘
        │
        ▼
browser renderer (read-only presentation)
```

The existing `race.py` and schema-1.1 replay files should remain the deterministic offline/evidence path. Add a `live_race`/scheduler module rather than changing replay semantics or making the browser a second simulator. This matches the current seams: `evaluation.py` already obtains an action before `env.step()`, `onnx_artifact.py` already defines a restricted batch-one CPU contract, and `isolated_policy.py` already supplies a killable, sequence-checked JSONL worker.

The first live competition profile should remain **independent ghosts**: each entrant has the same explicit track, fixed start, configuration, and simulated clock, and cars do not collide or appear in one another’s observations. If a real interacting race is later required, add a separate simultaneous-action mode; do not silently change the meaning of the current comparison.

## Fixed-rate contract

Use three clocks with different authority:

| Clock | Recommended meaning |
|---|---|
| `tick` | Integer Python physics/published-state index; authoritative. |
| `sim_time_s` | `tick / physics_hz`; used for lap time, timeout, and result. |
| monotonic wall time | Pacing, action deadlines, liveness, and diagnostics only. |

### Initial event profile

- Freeze the current provisional **30 Hz** physics rate for the first live slice.
- Start with `control_hz = 30` and `action_hold_ticks = 1`; this is the smallest change to the current scalar `env.step()`/policy loop.
- If a lower policy rate is later needed, require `physics_hz % control_hz == 0` and validate the integer `action_hold_ticks`, as `dm_control` does for control and physics timesteps.
- Create and validate every ONNX Runtime session, then warm it with representative inputs **before tick zero** and before starting the measured race clock.

One control cycle is:

1. At authoritative state `tick = k`, copy an immutable snapshot and build the observation.
2. Call ONNX with a request tagged with `run_id`, `entrant_id`, `state_tick = k`, `control_tick = k`, observation/schema identity, and deadline.
3. Accept only a finite, bounded `(1, 2)` action whose response identifies exactly the current state tick.
4. Apply the validated action and advance exactly one fixed physics step to `tick = k + 1`.
5. Compute all Python-owned events, progress, termination, and score; then publish a complete post-transition snapshot.

Tick zero remains the true reset pose, consistent with [`docs/replay_contract.md`](../docs/replay_contract.md); publish it as the initial state packet after `hello`. A live state packet for tick `n > 0` is the state after the transition to `n`; it is never reconstructed from packet arrival time. Do not change `dt`, skip physics, or run extra steps to catch up with a slow inference call.

### Deadline and pacing semantics

For a 30 Hz control period, `T = 33.333 ms`. Use a scheduled release and a monotonic deadline:

```text
release(k) = live_start + k * T
deadline(k) = release(k) + policy_budget
```

Use **20 ms as the initial candidate policy budget**, leaving approximately 13 ms for observation preparation, validation, Python physics, publication, and scheduler margin. Freeze that value only after measuring end-to-end p99/p99.9 latency on the actual organizer hardware. The budget must include parent/worker IPC for isolated runs; it is not just `session.run()` time. A response at or before the deadline is on time; a response after it is late, with `late_by_ns` recorded using `time.monotonic_ns()`/`perf_counter_ns()`.

Warm-up, model loading, and artifact validation are outside the measured race. In a paced live run, sleep only toward the next scheduled release. Rendering and WebSocket sends must never determine the simulation tick or score. The official lap time remains fixed-step time (`steps * dt`), as in the existing evaluator and the closest ONNX race comparator, [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing).

### Late, invalid, or missing actions

Make the rule a named profile in the race report. Recommended profiles:

| Profile | Rule | Use |
|---|---|---|
| `strict_scored` **(default)** | Reject a late, stale, malformed, non-finite, out-of-range, EOF, or sequence-mismatched response; kill the isolated worker; mark that entrant DNF/failure. Do not apply a fallback action. | Official competition. |
| `trusted_live` | Hold the last validated action for at most one configured control interval, set `stale_action = true`, and record `action_age_ticks`. | Local visual demo only. |
| `neutral_stop` | Apply a fixed neutral/stop command and quarantine or end the entrant. | Safety/integration testing. |

The recommendation for scored races is strict failure. It is auditable and prevents a slow model from receiving an implicit controller. The existing two-second `IsolatedPolicy` timeout is an operational bound, not a 30 Hz deadline; the live path needs its own millisecond budget. A Python thread/future timeout is not sufficient because native ORT work may continue after the caller stops waiting. Use the existing process/container boundary for enforceable deadlines; experimentally tested `RunOptions.terminate` may be a trusted-local optimization, not the default guarantee.

For a future interacting multi-agent race, collect every action from the same pre-step snapshot and apply them together. A missing action must not let agent order change the world. For v1 independent ghosts, a missed entrant can DNF independently. If an interacting heat continues after a DNF, explicitly choose “terminal pose held” or “agent removed”; otherwise abort the heat. Do not let this policy emerge from scheduler timing.

## ONNX Runtime implementation

Preserve the current `synapse_onnx_policy` model contract and runtime profile:

- one long-lived `InferenceSession` per artifact;
- CPU execution provider, batch one, `session.run()` with contiguous float32 input;
- one intra-op and one inter-op thread, sequential execution, deterministic-compute setting;
- pinned ONNX Runtime version and recorded model/runtime/provider/session hashes and settings;
- warm-up before tick zero and no session creation or per-tick model loading;
- latency histograms for input preparation, ORT call, output validation, complete control cycle, deadline misses, and tick-release jitter;
- acceptance based on p50/p95/p99/p99.9/max **and** repeated trajectory hashes, not average latency alone.

Do not batch entrants in the official path: the current ONNX boundary is batch one, and batching introduces a wait-for-the-slowest-input barrier and changes fairness. Do not add I/O binding initially. ONNX Runtime’s API documents I/O binding primarily for avoiding CPU/device copies when tensors live on an accelerator; for Synapse’s tiny CPU `[1, N] -> [1, 2]` graph, ordinary `run()` is lower risk. Benchmark it later only if measured input/output conversion is material. Keep profiling disabled during acceptance races because it changes the timed path. References: [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), [thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), and [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).

For multiple independent entrants, serial evaluation is simplest and most repeatable but is not a simultaneous wall-clock race. Parallel workers give synchronized release times but introduce CPU contention and require equal CPU/memory reservations. Either run independent entrants as separately provisioned fixed-rate lanes or clearly label serial execution as a comparison of ghost trials, not a simultaneous heat.

## Live state transport

Use a **one-way application contract over WebSocket text JSON**:

```text
Python: ws://127.0.0.1:<port>/live/v1  ──► browser
```

WebSocket is the smallest transport compatible with the current browser renderer: browsers have a native client, while they cannot open arbitrary TCP/UDP sockets. Raw TCP + JSONL remains appropriate for the existing Python worker boundary; UDP would require a relay and would add loss/order/discovery semantics. The [`websockets` broadcast guidance](https://websockets.readthedocs.io/en/stable/topics/broadcast.html) specifically warns that a slow client can block serialized sends and recommends disconnecting or using per-client queues. The browser API also has no incoming backpressure mechanism ([MDN WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)), so the server must drop presentation states before they accumulate.

### Protocol v1

Every message is one bounded JSON object with unique keys, finite numbers, and:

```json
{
  "kind": "synapse.live.state",
  "schema_version": 1,
  "run_id": "race-20261023-abc123",
  "seq": 17
}
```

Define three message kinds:

- `synapse.live.hello`: `run_id`, protocol/schema, `physics_hz`, `control_hz`, track ID/hash and the validated track payload, car metadata, and entrant IDs. Sending the existing track shape once avoids a new renderer asset protocol.
- `synapse.live.state`: one complete **post-transition** snapshot for the current tick, including `tick`, `sim_time_s`, all displayed cars, authoritative pose/heading/speed/commands/events/status, `action_tick`, `action_age_ticks`, `stale_action`, and terminal flags.
- `synapse.live.end`: final tick/status/result/trajectory hashes. Treat this as control data, not a replaceable state.

Use `seq` for all published messages and `tick` for the simulation timeline. Gaps in `seq` are expected presentation loss, not engine failure. On reconnect, send a fresh `hello` and the newest complete snapshot for the current `run_id`; do not replay a backlog. Generate a new `run_id` on reset and reject old-run packets. Bind loopback only, cap message size (for example 256 KiB), validate schema/kinds/types, and collect publish/drop/send/reconnect diagnostics.

The publisher path must be:

```text
fixed-step transition
  -> immutable compact snapshot
  -> replace each client’s latest-state slot without awaiting
  -> independent sender task
```

Keep at most one pending state, or two if the renderer needs a previous/current interpolation pair. Replace older states. Preserve `hello` and `end` in a separate reliable/control path. Disconnect a client that cannot make progress within a bounded liveness interval. The simulation must produce the same trajectory hash with the renderer connected, disconnected, or deliberately slow.

The renderer sends no steering, throttle, brake, reset, pause, scoring, or track-selection commands in v1. Optional `subscribe` and application-level liveness messages are not actions. A future human-control feature needs a separate endpoint with command sequence, target tick, acknowledgement, and failure semantics.

## Minimal `renderer/` change

Do not rewrite the scene, car model, physics, replay validation, or current replay controls. Add a live source adapter alongside the replay source:

1. Add `connectLive()`/`parseLiveMessage()` (ideally a small `renderer/live_source.js`) that validates `schema_version`, `run_id`, `seq`, `tick`, finite fields, and message size assumptions.
2. Keep `previousState` and `latestState`; interpolate only between received authoritative snapshots using `tick`/`sim_time_s`. Never extrapolate an unreceived physics result.
3. Normalize live state to the fields already consumed by `updateCars()`, HUD, camera, and minimap. Reuse the existing track payload and two car models; v1 should expose the renderer’s current two-entrant limit rather than expanding the scene to four cars at the same time.
4. In live mode, replace the replay clock with connection states `CONNECTING`, `LIVE`, `STALE`, and `ENDED`. Hide or disable scrub/restart/next-tick controls; camera and other presentation controls remain local.
5. On a new `run_id`, clear interpolation history. On a stale stream (for example no new packet for `max(250 ms, 3 / tick_hz)`), hold the last valid pose and show `STALE`. Treat Python’s events, terminal status, and final result as authoritative.

This is an adapter, not a renderer-side simulator. The existing replay path stays unchanged and remains the fallback when no live server is available. Pygame can keep consuming completed replays; a live Pygame adapter is a later optional client, not part of the first slice.

## Suggested implementation sequence

1. Add a shared live protocol validator and immutable snapshot type; reuse existing track, state, action, and result field names.
2. Add `live_race.py` and `tools/run_live_onnx.py`. Reuse entrant/artifact/contract validation from `race.py`; do not alter `race.py`’s replay output. Add `run_id`, tick/deadline configuration, action outcome, latency summaries, and live diagnostics to a separate report.
3. Refactor only enough of evaluation orchestration to expose “snapshot → action → fixed transition”; keep `VectorizedRacingEnv` and its Python physics unchanged.
4. Add the pinned `websockets` dependency under a live/server extra. Keep JSONL and `IsolatedPolicy` for policy actions; WebSocket is presentation-only.
5. Add the renderer live adapter and a two-entrant smoke command. Make track/config/engine hashes and `race_type` explicit in the hello/report.
6. Add tests for deadline boundaries, hung worker teardown, stale responses, fixed ticks despite sleep/rendering, dropped WebSocket states, reconnect/run reset, and identical trajectory hashes with live rendering enabled/disabled.

## Tradeoffs and release gates

| Choice | Benefit | Cost/explicit tradeoff |
|---|---|---|
| WebSocket JSON snapshots | Browser-native, inspectable, framed, ordered; smallest renderer integration. | TCP head-of-line behavior and JSON overhead; latest-only dropping is required. |
| Raw TCP/JSONL | Simple and efficient for Python/native workers. | Browser cannot consume it directly; framing/reconnect code becomes application-owned. |
| Strict DNF on missed action | Fair, deterministic, auditable; no hidden controller. | One tail-latency spike ends an entrant. Hold-last is smoother but changes competition rules and can reward slow inference. |
| 30 Hz control = 30 Hz physics initially | Lowest code and contract risk; matches current provisional rate. | Less actuator fidelity than a faster physics loop; later decimation must be explicit and integer-validated. |
| Independent ghosts | Preserves current comparison, avoids opponent leakage and multi-agent physics changes. | Not a physical wheel-to-wheel race. Interacting mode needs a shared action barrier and new termination semantics. |
| CPU, one-thread ORT | Better contention control and repeatability for small models. | May reduce mean throughput for larger models; optimize only after p99 evidence. |
| Process/container deadline | Only reliable way to stop a stuck native inference call. | IPC/startup and operational complexity must be included in the deadline budget. |
| Latest state slot | Renderer cannot stall physics and stale frames are harmless. | Presentation can skip ticks; `seq` gaps and stale indicators must be visible diagnostics. |

Release only when: repeated fixed-step trajectory hashes match; p99.9 end-to-end action latency fits the frozen budget with margin; late/invalid actions produce the documented result; slow or disconnected renderers cannot alter physics; reconnects cannot show an old run; `end` is delivered; and the live path preserves the existing replay contract.

## Sources and similar implementations

**Synapse sources:** [`README.md`](../README.md), [`docs/state_schema.md`](../docs/state_schema.md), [`docs/replay_contract.md`](../docs/replay_contract.md), [`racing_event/evaluation.py`](../racing_event/evaluation.py), [`racing_event/race.py`](../racing_event/race.py), [`racing_event/onnx_artifact.py`](../racing_event/onnx_artifact.py), and [`racing_event/isolated_policy.py`](../racing_event/isolated_policy.py). The full research notes are [`live_racing_loops.md`](../research_notes/Live%20ONNX%20race%20inference/live_racing_loops.md), [`onnx_realtime.md`](../research_notes/Live%20ONNX%20race%20inference/onnx_realtime.md), and [`live_state_transport.md`](../research_notes/Live%20ONNX%20race%20inference/live_state_transport.md).

**Comparable repositories and documentation:**

- [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing) — closest ONNX ghost-race analogue: 25 Hz policy, 100 Hz physics, held actions, independent worlds, and simulated-time scoring; it explicitly lacks a hard inference deadline.
- [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) — explicit physics timestep and all-agent transition ordering; rendering is separate from the simulator step.
- [RLGym](https://github.com/RLGym/rlgym) — explicit action parser/repeat semantics and renderer `packet_id`/tick-rate metadata.
- [CARLA synchronous-mode example](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py) — fixed delta, world tick, and frame-matched sensor data; a precedent for rejecting stale frame IDs.
- [PettingZoo `ParallelEnv`](https://pettingzoo.farama.org/api/parallel/) — simultaneous action dictionaries and a clear contrast with sequential turn semantics.
- [Unity ML-Agents `DecisionRequester`](https://github.com/Unity-Technologies/ml-agents/blob/develop/com.unity.ml-agents/Runtime/DecisionRequester.cs) — fixed-update decision periods and explicit held-action behavior.
- [Python `time.monotonic_ns()`](https://docs.python.org/3/library/time.html#time.monotonic_ns) — monotonic deadline clock unaffected by system-clock updates.
- [`websockets` broadcasting guide](https://websockets.readthedocs.io/en/stable/topics/broadcast.html) and [MDN WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket) — slow-client/backpressure behavior and the browser’s lack of incoming backpressure.
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html) — `InferenceSession.run`, explicit providers, I/O binding, `RunOptions.terminate`, and deterministic/session options.
