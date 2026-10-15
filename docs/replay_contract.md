# Python to renderer replay contract

## Authority and scope

Replay schema **1.1** (`kind: racing_event_replay`) is the first validated UI
boundary. Python owns simulation, collisions, progress, rewards, termination,
and results. A browser or Unity reads poses and events. Playback controls
change only a visual clock; there is no renderer-to-simulation control channel.

The old `tools/run_scripted.py` output is a **legacy 1.0 smoke recording**, not
this contract. It lacks a true initial pose and used tick 0 for the first
post-step pose. New readers reject it rather than silently shifting timing.
Use `tools/export_replay.py` for schema 1.1.

## Document fields

| Field | Meaning |
|---|---|
| `schema_version` | Exactly `1.1`. Unknown versions fail closed. |
| `kind` | Exactly `racing_event_replay`. |
| `engine.code_hash` | SHA-256 of the named Python simulation source files in `trajectory.ENGINE_FILES`, at recording time. |
| `engine.config_hash` | Canonical parsed simulation config hash from `TrainingContract`. |
| `source.mode` | `scripted` smoke controller or `evaluation` saved-policy mean actions. |
| `source.participant_id` | Display label, not an authenticated identity. |
| `tick_hz` | Positive integer fixed simulation frequency, from Python. |
| `car` | `length_m`, `width_m` for drawing the same rectangle. |
| `track` | Exact geometry 1.0 document, including `meta.content_hash`. No redrawing or inferred widths. |
| `frames` | Initial pose plus one independent snapshot per executed transition. |
| `result` | Python result: `status`, `steps`, `elapsed_s`, `lap_time_s`, `trajectory_hash`. |
| `payload_hash` | Integrity checksum of header, frames, and result (algorithm below). |

The embedded track is the selected original geometry. Loading a replay does
not rerun topological physics validation; the exporter loads a validated track
and readers check the geometry schema, finite arrays, and content hash. Never
export a private held-out track to a participant-facing file, web directory, or
repository. The current preview is public synthetic training geometry only.

These hashes detect accidental changes. They are not signatures, anti-cheat
authentication, or proof of training provenance. `engine.code_hash` describes
the recording engine; an older training artifact does not thereby become
attested to that source revision. Tournament release/submission verification
remains separate work.

## Frame fields and time

Every frame contains exactly:

```text
tick, x, y, heading, speed, lateral_velocity, yaw_rate, steer,
steering_command, throttle_brake, off_track, events, terminated, truncated
```

- Tick `0` is the actual reset pose at time zero, zero controls and no events.
- Tick `n` is the post-step pose after transition `n`; its simulation time is
  `n / tick_hz`. Ticks must be consecutive integers. The last tick equals
  `result.steps`, and there are `steps + 1` frames. No duplicate end frame.
- Coordinates and dimensions are metres, speeds m/s, angles radians, yaw rate
  rad/s. Heading zero points along world +X; heading increases counterclockwise
  toward world +Y. Heading is not forcibly wrapped.
- `steer` is the actual steering angle in radians. `steering_command` and
  `throttle_brake` are the applied bounded commands in `[-1, 1]`, not raw
  unclipped policy requests. Positive longitudinal command means throttle;
  zero means coast; negative means brake.
- `off_track` and the two done flags are recorded booleans. They must not be
  computed by the renderer. At most one done flag can be true. A done frame
  must be last.
- Supported events are `off_track_start`, `off_track_end`, `barrier_contact`,
  and `lap_complete`. A renderer may trigger effects at those recorded ticks;
  it must not invent outcomes from the visible mesh.
- `finished` requires the recorded finish event and has a Python fixed-step
  lap time. DNF statuses are `dnf_crash`, `dnf_stuck`, `dnf_backward`, and
  `dnf_timeout`, with `lap_time_s: null`. Crash wins over simultaneous finish.
  A preview cut short by `--steps` is a timeout, not an engine-completed lap.

No reset occurs within an evaluation replay. Future multi-episode training
replays need explicit episode boundaries and must not interpolate teleports.

## Unity adapter recipe (no Unity implementation yet)

1. Validate the schema and track hash before displaying the replay. Build the
   track mesh from the embedded matching boundary points. Polylines close from
   their last unique point to the first; do not add a physical seam.
2. Map Python `(x, y)` to Unity `(X=x, Y=visual height, Z=y)`, one unit per
   metre. Car pivot is its geometric centre.
3. For a car mesh with local +Z forward and +Y up, yaw degrees are
   `90 - heading * 180/pi`. Sanity check: Python heading 0 faces Unity +X;
   heading `pi/2` faces Unity +Z. Adapt the fixed mesh-axis offset if a model
   has a different forward axis.
4. Disable vehicle PhysX motion/contact outcome logic. Never simulate a rigid
   body to "match" the replay. Set transforms directly from Python poses.
5. Between ticks, linearly interpolate position and use shortest-angle heading
   interpolation. Do not extrapolate after the last frame. Hold events and
   telemetry at the last completed tick. At an exact tick use the exact stored
   values, including the final pose.
6. Display the recorded Python result and lap time, not a stopwatch derived
   from Unity frames or a visual start-line crossing.

Joint acceptance tests remain: compare known poses and forward axes, compare
mesh boundary coordinates, check exact end time/events, and reject a mismatched
version. A real Unity round-trip has not yet been run.

## Storage, integrity, and memory

`ReplayWriter` writes frames incrementally to a temporary sibling file and
replaces the destination only after `finish`. A crash/error before finish
leaves no partial published recording and preserves any previous file.

The payload SHA-256 is updated in order with canonical JSON header, each frame,
then result, each followed by one LF byte. Canonical JSON uses sorted keys,
UTF-8, compact separators, ASCII escaping (Python default `ensure_ascii=True`),
and rejects NaN/Infinity. The header is every top-level field except `frames`,
`result`, and `payload_hash`. It is not the hash of pretty-printed file bytes.

Preview limits: 16 MiB input, 20,001 frames, and 20,000 track samples. Duplicate
JSON keys, non-finite states, broken timing, invalid result/terminal pairs,
unexpected keys, and changed hashes are rejected. This is a bounded local
developer tool, not a hardened public upload service. The writer uses constant
frame-history memory; the HTML bundler and browser load the bounded recording.
No renderer process is started during training.

`tools/build_replay.py` validates first and embeds the recording and viewer
assets in one HTML with no network dependencies. Participant text is escaped
as inert JSON and inserted with `textContent`, not HTML.
