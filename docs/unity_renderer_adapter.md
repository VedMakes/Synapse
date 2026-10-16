# Unity renderer adapter foundation

## Target

The first adapter target is **Unity 2022.3.22f1 LTS with URP 14.0.8**. The
project and package versions are pinned under `unity/`. The adapter uses the
Unity Newtonsoft JSON package only to parse the versioned replay document.

This is a presentation target, not a second racing engine. The Python replay
reader remains the authoritative validation reference.

## Input and authority

The adapter accepts one `racing_event_replay` schema `1.1` document. It
validates the schema, fixed-step frame order, finite pose/control values, event
vocabulary, terminal-event rules, result status, and the embedded track content
hash before displaying the replay.

The following values are read-only inputs:

- car pose, heading, speed, and controls;
- `off_track`, `terminated`, and `truncated` flags;
- recorded event names and their exact ticks;
- Python-authored result and lap time;
- embedded track boundaries and metadata.

Unity must not compute or replace any of these values. In particular, do not
add a Rigidbody, vehicle collider, finish-line trigger, lap stopwatch, reward
calculation, progress calculation, or collision outcome logic to the adapter.

## Coordinate and timing mapping

The Python world is `(x, y)`. Unity uses `(X, Y, Z) = (x, visual_height, y)`.
For a car mesh with local `+Z` forward, use:

```text
Unity yaw degrees = 90 - Python heading * 180 / pi
```

The `ReplayClock` advances a visual clock only. At a non-terminal time between
two Python ticks, it linearly interpolates `x` and `y` and uses shortest-angle
heading interpolation. It does not extrapolate after the final frame. Exact
tick sampling returns the stored pose and frame telemetry without blending.

## Presentation configuration

`GhostPresentationConfig` contains renderer-only options:

- visual height and scale;
- simultaneous or staggered reveal mode and delay;
- trail and label visibility;
- event-effect toggles for off-track smoke, barrier impact, and lap confetti.

These settings do not enter the replay payload, hashes, result, timing, or
event decisions. Schema `1.1` still represents one replay stream. A show with
multiple ghosts needs a future versioned group manifest with explicit
alignment, missing-frame, and terminal policies.

## Joint acceptance tests

Before VECTOR builds a full show scene, run these checks against a known Python
replay:

1. A Python pose at `(0, 0)` with heading `0` appears at Unity `(0, h, 0)`
   facing `+X` with yaw `90` degrees.
2. A Python pose at heading `pi/2` faces Unity `+Z` with yaw `0` degrees.
3. The visual track boundary vertices match the embedded Python arrays within
   the agreed mesh tolerance. The Unity object has no MeshCollider.
4. An event effect fires only on its recorded integer tick.
5. The final frame and Python result remain visible after the visual clock
   reaches the end. Unity does not derive a new time or status.
6. A schema-version, terminal-state, or track-hash mismatch is rejected before
   any car or mesh is displayed.
7. Changing reveal mode or visual material does not alter replay JSON, hashes,
   frame count, or recorded result.

The repository currently has Unity EditMode tests for clock mapping and basic
reader rejection. A real Unity import, mesh round-trip, and scene screenshot
remain required on a machine with Unity installed.
