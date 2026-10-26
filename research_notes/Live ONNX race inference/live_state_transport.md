# Live state transport for the Synapse ONNX race

**Research date:** 2026-10-23  
**Scope:** localhost transport patterns for streaming fixed-rate, Python-authoritative race state to a separate renderer/UI. The comparison covers WebSocket, raw TCP, UDP, and JSON Lines (JSONL), with emphasis on sequencing, backpressure, stale-frame behavior, clock semantics, reconnects, and whether the renderer sends actions.

## Executive recommendation

Use a **one-way WebSocket state stream**:

```text
Python authoritative process  -- ws://127.0.0.1:<port>/live --> browser renderer
```

Each WebSocket text message is one UTF-8 JSON object. The Python process remains the only simulation authority. The renderer subscribes, validates, drops old state, interpolates only for presentation, and never sends vehicle actions. The initial protocol needs three server-to-client message kinds:

1. `synapse.live.hello`: run identity, schema, tick rate, car metadata, and the track payload needed by the current renderer.
2. `synapse.live.state`: one complete post-transition snapshot for all cars at one authoritative tick.
3. `synapse.live.end`: terminal status and final tick/result.

The server should maintain a **latest-state slot per client**, or a bounded queue whose state entry is replaced by a newer one. It must never await a slow renderer from the simulation loop. Gaps in `seq` or `tick` are expected when presentation frames are dropped; a state packet is a snapshot, not an accounting record. Do not build a replay-history or reliable-frame protocol for live presentation.

This is the smallest protocol that fits the workspace and the renderer ownership boundary because:

- browsers provide a native WebSocket client, while they cannot open arbitrary TCP or UDP sockets;
- WebSocket supplies ordered, reliable message delivery and message boundaries over TCP;
- JSON keeps the contract inspectable by Python and JavaScript and matches existing Synapse schemas;
- the authoritative engine can remain independent of renderer timing;
- reconnecting clients can receive a fresh `hello` and current snapshot without replaying an unbounded backlog.

The current `pyproject.toml` does not declare a WebSocket server dependency. That is an integration task for the owning team; this note does not add one or edit application code.

## What Synapse already establishes

The local code makes the transport boundary unusually clear:

- [`README.md`](../../README.md) says Python is authoritative and the browser/Pygame renderers are read-only. The provisional simulation rate is 30 Hz.
- [`docs/state_schema.md`](../../docs/state_schema.md) defines versioned action/observation contracts and says renderers do not recompute physics, progress, rewards, or termination.
- [`renderer/app.js`](../../renderer/app.js) currently loads complete schema-1.1 replay JSON and advances a browser playback clock. It is not currently a live subscriber.
- [`racing_event/trajectory.py`](../../racing_event/trajectory.py) already defines the useful dynamic fields: `tick`, position, heading, speed, steering, throttle/brake, off-track state, events, and terminal flags.
- [`racing_event/isolated_policy.py`](../../racing_event/isolated_policy.py) is a strong existing **JSONL action-worker** example: newline framing, sequence numbers, a 16 KiB line cap, finite/bounded action validation, monotonic timeouts, and worker teardown on failure. It should remain an internal model-worker boundary, not become the browser transport.
- [`racing_event/race.py`](../../racing_event/race.py) currently evaluates entrants sequentially and independently and emits complete replays. A future live packet should state explicitly whether entrants are lockstep/interacting or independent ghosts.

The live layer should be additive: keep schema-1.1 replay files for offline evidence and add a separate live packet schema. Do not make the renderer a second physics engine and do not route renderer messages into the ONNX action boundary.

## Transport comparison

| Transport | Browser renderer | Ordering/reliability | Framing | Backpressure | Reconnect | Assessment for Synapse |
|---|---|---|---|---|---|---|
| **WebSocket text JSON** | Native `WebSocket` API | Ordered/reliable because WebSocket is layered on TCP | Native message boundaries | TCP/library flow control exists, but browser API has no automatic receive backpressure; application must bound/drop | Application `onclose` loop; server can send a fresh snapshot | **Recommended** |
| **Raw TCP + JSONL** | Not available to a normal browser page; works for Node/Electron/native UI | Ordered/reliable byte stream | Newline delimiter, with line-size limit | `write` + `drain`; blocking can stall the producer | Reconnect manually; frame protocol must be recovered | Good Python/native IPC, poor fit for current renderer |
| **Raw TCP + length-prefixed JSON/binary** | Not available to a normal browser page | Ordered/reliable byte stream | Application length header | Explicit flow control and framing | Manual | Robust but more protocol than the renderer needs; SUMO is a useful precedent |
| **UDP JSON/datagrams** | Not available to a normal browser page | No delivery, duplicate, or order guarantee | Datagram boundary | No sender-side connection backpressure; kernel buffers can overflow | No connection to reconnect; application heartbeat/sequence required | Reject for the first UI path |
| **JSONL over stdin/stdout** | Only through a native child-process bridge | Pipe ordering; EOF is the failure signal | Newline delimiter | Pipe buffers can block a child; use bounded messages and a reader | Restart child | Keep for policy workers and test fixtures, not renderer delivery |

### WebSocket

WebSocket is explicitly a two-way protocol, but that does not require a two-way application contract. It provides a browser-compatible TCP connection, message framing, close semantics, and optional subprotocol negotiation. The application still needs to add `schema_version`, `run_id`, and `seq`; the WebSocket protocol does not know what a simulation tick means.

The main trap is backpressure. The `websockets` documentation explains that normal `send()` applies flow control, which is useful for a single stream but can make a broadcast loop wait for a slow client. Its broadcast guide recommends either disconnecting clients that cannot keep up or using per-client queues where the application can skip or batch messages. For Synapse, the appropriate policy is a per-client latest-state slot: skip old state packets, preserve `hello` and `end`, and disconnect only if a client cannot receive control/heartbeat messages within a bounded time.

The browser API has no automatic receive backpressure. MDN documents `WebSocket.bufferedAmount` for client-to-server writes and warns that the browser `WebSocket` API has no way to apply backpressure to incoming messages. That is another reason to drop before sending and to keep state packets compact.

### Raw TCP and JSONL

TCP provides a reliable, in-order **byte stream**, not application messages. A TCP implementation therefore needs an application framing rule. JSONL is the smallest human-readable rule: one complete JSON value per UTF-8 line. A length prefix is safer when payloads may contain arbitrary bytes or when binary compression is planned.

Python `asyncio` streams expose exactly the primitives needed for a small TCP protocol: `readline()` for bounded JSONL, `readexactly()` for a length prefix, `write()`, and `drain()` for write-buffer flow control. `StreamReader` has a configurable input limit. A raw TCP/JSONL design would be reasonable if the renderer were a native process, an Electron main process, or a Node client. It is not directly consumable by the current browser page.

SUMO's open-source TraCI is a concrete, mature TCP design. It uses a length-prefixed message container, command identifiers, response validation, and an explicit simulation-step barrier. That is a good reference for a future control protocol, but it is more machinery than a read-only renderer stream needs.

### UDP

UDP preserves datagram boundaries and avoids TCP head-of-line blocking, which can be attractive for “newest pose wins” telemetry. However, RFC 768 explicitly says delivery and duplicate protection are not guaranteed and recommends TCP for ordered reliable streams. A usable UDP state protocol would still need a `run_id`, sequence/tick numbers, stale rejection, possibly checksums, and a discovery/heartbeat story. A browser page cannot receive raw localhost UDP, so UDP would also require a native relay. It is not the smallest compatible choice.

### JSONL

JSON Lines defines UTF-8 values separated by `\n` and is intended for records processed one at a time and for cooperating processes. It is an excellent fit for Synapse's existing Python worker boundary. It is not itself a network transport and does not solve reconnects, backpressure, or browser access. Treat it as a framing/serialization convention that can sit on stdio or TCP, not as a replacement for WebSocket for the browser renderer.

## Proposed protocol v1

### Connection and ownership

- Bind only to `127.0.0.1`; do not expose the live endpoint on `0.0.0.0`.
- The Python process listens and publishes. The renderer connects as a read-only subscriber.
- The renderer may optionally send a `subscribe` or application-level `ping`, but it does not send steering, throttle, brake, reset, pause, track selection, or scoring commands.
- A separate control channel would be required if human actions are ever introduced. It should have a distinct endpoint and explicit command/acknowledgement schema rather than overloading state packets.
- Use a WebSocket subprotocol name or path that can be versioned, for example `synapse.live.v1` or `/live/v1`. A breaking wire change should change the version.

### Message envelope

Every message is one JSON object in one WebSocket text message. Require UTF-8 and finite JSON numbers. Use exact/unique object keys. Suggested common fields:

```json
{
  "kind": "synapse.live.state",
  "schema_version": 1,
  "run_id": "race-20261023-abc123",
  "seq": 17
}
```

`seq` is a publisher sequence for all messages in one run. It is not required to be gap-free at the renderer because state packets may be intentionally replaced or skipped. The receiver should reject `seq <= last_seq` and count `seq > last_seq + 1` as a dropped presentation packet.

### `hello`

Send `hello` immediately when a connection opens and again on every reconnect. It is a static/session snapshot, not a per-tick message:

```json
{
  "kind": "synapse.live.hello",
  "schema_version": 1,
  "run_id": "race-20261023-abc123",
  "seq": 0,
  "tick_hz": 30,
  "physics_hz": 30,
  "control_hz": 30,
  "track": {
    "id": "synthetic_oval",
    "content_hash": "<lowercase sha256>",
    "version": "1.0",
    "L": 100.0,
    "runoff_width": 8.0,
    "start_line_s": 0.0,
    "centerline": [[0.0, 0.0]],
    "s": [0.0],
    "heading": [0.0],
    "curvature": [0.0],
    "half_width_left": [5.0],
    "half_width_right": [5.0],
    "left_boundary": [[0.0, 5.0]],
    "right_boundary": [[0.0, -5.0]],
    "barrier_left": [[0.0, 7.0]],
    "barrier_right": [[0.0, -7.0]],
    "meta": {"id": "synthetic_oval", "display_name": "Oval"}
  },
  "car": {"length_m": 4.8, "width_m": 2.0},
  "entrants": [
    {"participant_id": "team-a", "label": "Racer Alpha"},
    {"participant_id": "team-b", "label": "Racer Beta"}
  ]
}
```

The track example is abbreviated; the actual payload should use the same validated track shape already embedded in Synapse replay headers. Sending it once avoids making the renderer team implement a new asset lookup protocol. If a future renderer asset registry exists, `track` can be replaced by `track_id` plus `track_hash`, but that is a later compatible optimization only if both sides agree.

### `state`

Publish a complete post-transition state after the authoritative engine has finished one shared tick for all displayed entrants:

```json
{
  "kind": "synapse.live.state",
  "schema_version": 1,
  "run_id": "race-20261023-abc123",
  "seq": 17,
  "tick": 16,
  "sim_time_s": 0.5333333333333333,
  "cars": [
    {
      "participant_id": "team-a",
      "status": "running",
      "x": 12.3,
      "y": -4.1,
      "heading": 1.57,
      "speed": 8.2,
      "lateral_velocity": 0.1,
      "yaw_rate": 0.02,
      "steer": 0.03,
      "steering_command": 0.04,
      "throttle_brake": 0.55,
      "off_track": false,
      "events": [],
      "action_tick": 16,
      "action_age_ticks": 0,
      "stale_action": false,
      "terminated": false,
      "truncated": false
    }
  ]
}
```

Use the existing replay field names wherever possible. `action_tick`, `action_age_ticks`, and `stale_action` are telemetry about the authoritative action scheduler; they do not grant the renderer any control authority. If the race is independent-ghost evaluation rather than an interacting race, state that in the run metadata and do not imply that cars observed or collided with one another.

### `end`

The terminal message is control-like and must not be discarded by a latest-only state queue:

```json
{
  "kind": "synapse.live.end",
  "schema_version": 1,
  "run_id": "race-20261023-abc123",
  "seq": 421,
  "tick": 420,
  "sim_time_s": 14.0,
  "status": "finished",
  "result": {
    "trajectory_hash": "<lowercase sha256>",
    "cars": [
      {"participant_id": "team-a", "status": "finished"},
      {"participant_id": "team-b", "status": "dnf_crash"}
    ]
  }
}
```

### Bounds and validation

The first implementation should enforce simple bounded limits:

- maximum message size, for example 256 KiB for JSON text;
- maximum entrants and event strings consistent with current race limits;
- `schema_version == 1` and known `kind` values;
- `run_id` is nonempty and changes when a run is restarted;
- `seq` and `tick` are nonnegative integers;
- `tick` is nondecreasing within a run;
- all floating-point values are finite;
- actions/telemetry are within the published bounds where applicable;
- unknown optional fields may be ignored, but missing required fields or wrong types close/reject the subscription.

RFC 8259 disallows NaN and Infinity in JSON numbers and recommends unique object member names. This matches Synapse's existing `allow_nan=False` and strict validation style.

## Sequencing and fixed-rate semantics

Use three distinct notions of time:

```text
tick          authoritative integer physics/published-state index
sim_time_s    tick / tick_hz; part of the simulation contract
wall time     only for pacing, timeouts, diagnostics, and connection liveness
```

Rules:

1. Advance physics by the configured fixed `dt`, never by elapsed socket or rendering time.
2. Snapshot and publish only after the state transition is complete. If multiple displayed entrants share a live race tick, publish one packet after all entrant states for that tick are assembled.
3. `sim_time_s` is derived from the integer tick; it is not reconstructed from packet arrival times.
4. The renderer accepts only the current `run_id` and strictly newer `seq`/`tick` values. An old packet received after a reconnect or delayed event is ignored.
5. A sequence gap is a presentation loss, not an engine error. The renderer displays the newest complete snapshot and may increment a diagnostic `dropped_packets` counter.
6. The renderer may interpolate between two received authoritative snapshots for smooth motion. It must not extrapolate an unreceived physics result, invent collisions/overtakes/finish events, or feed a predicted state back to Python.

CARLA's synchronous-mode example is the clearest frame-matching precedent: it calls `world.tick()`, retrieves data from queues, discards entries until the requested frame arrives, and asserts that all returned data have the same frame ID. RLGym's RLViser adapter uses an incrementing `packet_id` and a declared `tick_rate`. Synapse should use the same idea with `seq`/`tick` while keeping the tick authoritative in Python.

## Backpressure and stale-frame policy

### Producer side

The simulation loop must not perform a blocking `await websocket.send(...)`, wait for a renderer acknowledgement, or wait for a full per-client queue. Use this shape:

```text
authoritative step
  -> immutable compact snapshot
  -> nonblocking publish to each client's latest-state slot
  -> independent sender task drains the slot
```

For each client:

- keep at most one pending `state` packet, or a queue of at most two if interpolation needs a previous/current pair;
- replacing a pending state with a newer state is allowed and expected;
- keep `hello` and `end` outside the replaceable state slot;
- if the sender cannot make progress for a bounded liveness interval, close the connection and let the browser reconnect;
- count replacements, queue overflows, send failures, and disconnects;
- serialize JSON off the critical physics path when possible. If serialization is done in the publisher task, copy the state first so the renderer cannot mutate authoritative objects.

The `websockets` project explicitly warns that naive serialized broadcast can block on a slow client's write buffer. Its documented alternatives are disconnecting slow clients, per-client queues, or a publish/subscribe stream. The per-client queue pattern is the best fit here because the application knows that intermediate state packets are disposable.

### Renderer side

The browser should maintain `latestState`, `previousState`, `lastSeq`, `lastTick`, and `lastReceivePerformanceMs`:

- reject wrong `schema_version`, wrong `run_id`, old `seq`, and non-finite fields;
- if a new state is newer, replace the pending state rather than enqueueing every packet;
- render from the latest two snapshots if both are close enough in tick space;
- if no newer packet arrives for roughly `max(250 ms, 3 * 1/tick_hz)`, label the stream stale and hold the last valid pose;
- after a reconnect, clear interpolation history when `run_id` changes or when the server sends a new `hello` for a reset run;
- treat `end` as authoritative for finish status and show the last received pose, even if some final presentation packets were dropped.

The WebSocket API's `bufferedAmount` is useful only for renderer-to-server writes and is therefore not a substitute for server-side inbound-flow control. Since the renderer sends no state-producing commands, the server-side latest-only policy is the important one.

## Clock synchronization

No synchronized wall clock is needed for rendering or scoring. `tick` and `sim_time_s` are sufficient to place a state on the simulation timeline. The renderer's `performance.now()` can drive a local animation between received packets; it should not be compared directly with Python's `time.monotonic_ns()` as if they shared an epoch.

Recommended fields/uses:

- required: `tick`, `sim_time_s`, `tick_hz` in `hello`;
- optional diagnostic: `emitted_at_unix_ms` or `emitted_at_unix_ns`, clearly labeled as wall-clock telemetry and never used as simulation time;
- server-side scheduling/deadlines: Python `time.monotonic_ns()`;
- renderer-side receive age: browser `performance.now()`;
- optional future latency diagnostic: application `ping`/`pong` carrying a client timestamp and a server timestamp. Estimate round-trip/offset only for display diagnostics, not for authoritative tick advancement.

WebSocket protocol Ping/Pong frames are useful for server-side liveness. The `websockets` documentation notes that its Python implementation sends periodic Ping frames and closes connections that fail to answer. Browser JavaScript cannot directly create or observe protocol Ping/Pong frames, so a paused simulation should either keep the server's protocol keepalive enabled or send a small application-level heartbeat if the UI needs a visible “server alive” indicator during long idle periods.

## Reconnect and run identity

### Server behavior

- On connection: send `hello`, then the newest complete `state` if a run is active, or an `idle`/`hello` status if no run is active.
- On reconnect: send the same current `run_id` and current snapshot; do not replay every missed state.
- On a new race/reset: generate a new `run_id` and reset `seq`/`tick` for that run.
- Keep the final `end` snapshot available to newly connected clients for a short grace period if the UI may open after the checkered flag.
- If no current run exists, return a bounded, explicit idle state rather than leaving the browser waiting indefinitely.
- Reject unsupported protocol versions at handshake or with a bounded error message; do not stream an ambiguous mixture of schemas.

### Renderer behavior

- Connect to a fixed loopback URL and use `onopen`, `onmessage`, `onerror`, and `onclose`.
- Retry with bounded exponential backoff and reset the delay after a successful connection. On localhost, a short cap such as 2–5 seconds is adequate.
- Never apply a state from an old `run_id` after a new run has been observed.
- Show `CONNECTING`, `LIVE`, `STALE`, and `ENDED` as presentation states. These labels do not change Python simulation status.
- A reconnect is a new subscription, not a request to resume control. No action or acknowledgement needs replaying.

The Python `websockets` client documentation provides an asynchronous-iterator reconnect pattern with exponential backoff. Browser JavaScript needs its own small `onclose` retry loop; MDN documents the close event but does not supply automatic reconnect semantics.

## Does the renderer send actions?

**No for protocol v1.** The renderer is a read-only presentation client.

- ONNX actions remain inside the authoritative Python evaluation path.
- The existing JSONL worker protocol is request/response and sequence-checked because it carries actions; it should not be mixed with a lossy/latest-only presentation stream.
- Renderer keyboard controls such as play/pause, camera selection, speed, and track selection should remain local presentation controls for replay/live display. They must not pause or mutate the Python race unless a future control contract explicitly says so.
- If a future demo needs a human steering source, use a separate `control` WebSocket/TCP endpoint with `command_seq`, `target_tick`, explicit acknowledgements, and an authority/failure policy. Do not add `action` fields to ordinary state packets and do not let a renderer reconnect replay a stale action.

The only reasonable client-to-server messages for v1 are optional subscription metadata and liveness diagnostics, for example:

```json
{"kind":"synapse.live.subscribe","schema_version":1,"client":"renderer"}
```

This is a subscription request, not an action channel.

## Concrete repository and official-document patterns

### WebSocket library and browser behavior

- **python-websockets broadcast experiment** — the repository includes naive, concurrent, queue, and pub/sub broadcast variants. It shows why a slow client can block serialized sends, why the built-in broadcast intentionally does not apply backpressure, and how per-client queues can support custom skip/drop behavior.  
  Repository: <https://github.com/python-websockets/websockets/tree/main/experiments/broadcast>  
  Source: <https://raw.githubusercontent.com/python-websockets/websockets/main/experiments/broadcast/server.py>
- **websockets broadcasting guide** — recommends disconnecting clients that fall too far behind or using per-client queues when the application needs to skip/batch messages.  
  <https://websockets.readthedocs.io/en/stable/topics/broadcast.html>
- **websockets memory and buffers** — explains TCP buffers, write high-water marks, `max_size`, and `max_queue`; larger buffers can increase latency through bufferbloat.  
  <https://websockets.readthedocs.io/en/stable/topics/memory.html>
- **websockets keepalive** — documents protocol Ping/Pong, failure detection, and why browsers need application-level handling if they need their own visible heartbeat.  
  <https://websockets.readthedocs.io/en/stable/topics/keepalive.html>
- **websockets client reconnect** — documents `async for websocket in connect(...)` and exponential backoff.  
  <https://websockets.readthedocs.io/en/stable/faq/client.html#how-do-i-reconnect-when-the-connection-drops>  
  API: <https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html>
- **MDN WebSocket** — confirms the browser-native client and explicitly warns that the browser WebSocket API has no incoming backpressure mechanism.  
  <https://developer.mozilla.org/en-US/docs/Web/API/WebSocket>
- **MDN `bufferedAmount`** — reports bytes queued by browser calls to `send()`; this concerns client-to-server writes and does not solve server-to-browser receive pressure.  
  <https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/bufferedAmount>

### Fixed-step and frame identity

- **CARLA synchronous mode** — enables synchronous mode and fixed delta, calls `world.tick()`, pulls queue entries until the requested frame arrives, and asserts that all sensor/world objects have the same frame ID. This is the model for `tick`/`seq` validation and stale-frame rejection.  
  Repository: <https://github.com/carla-simulator/carla>  
  Example: <https://raw.githubusercontent.com/carla-simulator/carla/master/PythonAPI/examples/synchronous_mode.py>
- **RLGym RLViser renderer** — receives a Python `GameState`, increments `packet_id`, and sends `tick_count` and `tick_rate` to the renderer. It is a concrete example of explicit packet identity and renderer-only consumption.  
  Repository: <https://github.com/RLGym/rlgym>  
  Source: <https://raw.githubusercontent.com/RLGym/rlgym/main/rlgym/rocket_league/rlviser/rlviser_renderer.py>
- **Unity ML-Agents Academy** — its source documents a versioned communication protocol and a fixed-update event order: pre-step, send state, decide action, then act. It is useful for event ordering and versioning, but it is a training/control protocol rather than a read-only renderer stream.  
  Source: <https://raw.githubusercontent.com/Unity-Technologies/ml-agents/develop/com.unity.ml-agents/Runtime/Academy.cs>

### TCP, JSONL, and explicit framing

- **Python `asyncio` streams** — `readline()` supports newline framing, `readexactly()` supports length framing, and `drain()` is the write-buffer flow-control point. The stream reader/server also expose a configurable buffer limit.  
  <https://docs.python.org/3/library/asyncio-stream.html>
- **JSON Lines specification** — requires UTF-8, one valid JSON value per line, and `\n` line termination; it explicitly calls out cooperating-process message passing.  
  <https://jsonlines.org/>
- **Synapse isolated policy** — local example of bounded JSONL request/response with strict sequences, 16 KiB lines, finite action checks, and monotonic timeouts.  
  [`racing_event/isolated_policy.py`](../../racing_event/isolated_policy.py)
- **SUMO TraCI** — an open-source Python/C++ TCP client/server protocol with length-prefixed messages, command identifiers, response validation, subscriptions, and an explicit simulation-step barrier. It also documents that subscriptions return current results rather than an unlimited history, which is analogous to latest-state presentation.  
  Overview: <https://sumo.dlr.de/docs/TraCI/index.html>  
  Protocol: <https://sumo.dlr.de/docs/TraCI/Protocol.html>  
  Python client: <https://raw.githubusercontent.com/eclipse-sumo/sumo/main/tools/traci/connection.py>

### Transport specifications

- **TCP RFC 9293** — TCP provides a reliable, in-order byte-stream service; application framing is still required.  
  <https://datatracker.ietf.org/doc/html/rfc9293#section-2.2>
- **UDP RFC 768** — UDP is datagram-oriented and does not guarantee delivery or duplicate protection; ordered reliable streams should use TCP.  
  <https://datatracker.ietf.org/doc/html/rfc768>
- **WebSocket RFC 6455** — WebSocket provides an opening handshake and message framing over TCP, with close and Ping/Pong control frames. Application metadata belongs in the subprotocol/application layer.  
  <https://datatracker.ietf.org/doc/html/rfc6455>
- **JSON RFC 8259** — JSON is text-based and language-independent; network JSON should be UTF-8, numbers cannot be NaN/Infinity, and object names should be unique.  
  <https://datatracker.ietf.org/doc/html/rfc8259>
- **Python monotonic clock** — `time.monotonic()` cannot go backwards and is not affected by system-clock updates; use monotonic time for pacing/deadlines, not wall time or packet arrival time.  
  <https://docs.python.org/3/library/time.html#time.monotonic>

## Recommended implementation split

### Python authoritative team

1. Own the live schema and `run_id`/`seq`/`tick` rules.
2. Create immutable snapshots after fixed-step transitions.
3. Publish snapshots to a nonblocking latest-only fan-out layer.
4. Keep WebSocket connection management and JSON serialization out of the physics critical path.
5. Preserve complete replay writing separately for evidence and offline playback.
6. Expose only loopback by default and bound message sizes, clients, and queues.
7. Record stream diagnostics: published count, replaced/dropped count per client, send failures, reconnects, max queue age, and last published tick.

### Renderer team

1. Add a live source beside the existing replay source; do not rewrite replay validation or physics assumptions.
2. Connect to the documented loopback URL and accept only schema v1.
3. Keep the newest two complete state packets for interpolation; never queue unlimited packets.
4. Reject old `seq`/`tick`, reset on a new `run_id`, and show stale/disconnected status from local receive age.
5. Treat events, status, and final results as authoritative fields from Python.
6. Do not send action-like messages.

## Acceptance checklist

Before calling the stream live-ready, test with a deterministic fake publisher and a deliberately slow renderer:

- state tick 0 is accepted and later ticks are strictly newer;
- a duplicate, old, or wrong-run packet is ignored;
- a deliberate sequence/tick gap is counted but does not stop rendering;
- a slow client causes state replacement or disconnect, never a simulation stall;
- `end` is delivered even when intermediate state packets were dropped;
- closing/restarting the server causes bounded reconnects and a fresh `hello`;
- reconnecting after a run reset cannot display packets from the prior `run_id`;
- no NaN, Infinity, oversized message, unknown required field, or malformed JSON is accepted;
- renderer enabled versus renderer disconnected produces identical authoritative trajectory/result hashes;
- the renderer never changes actions, physics, scoring, or termination;
- packet arrival time and renderer FPS do not alter `tick` or `sim_time_s`.

## Bottom line

For this workspace, choose **WebSocket + text JSON snapshots + latest-only per-client buffering + tick/sequence IDs + read-only renderer semantics**. Use the existing JSONL pattern only for Python-to-worker action exchange. Use TCP framing and SUMO's step/response discipline as design references, CARLA's frame identity for stale handling, RLGym's packet/tick metadata for renderer packets, and the `websockets` queue guidance for backpressure. Keep fixed-step simulation time authoritative and treat reconnects, rendering delay, and wall-clock timestamps as presentation/diagnostic concerns.
