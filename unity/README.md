# SYNAPSE Unity renderer foundation

This is a read-only renderer foundation for **Unity 2022.3.22f1 LTS with
URP 14.0.8**. It consumes one validated Python replay schema `1.1` document at
a time. It does not contain a racing simulation.

## Authority boundary

- Python owns physics, collision, progress, rewards, termination, lap time, and
  result status.
- Unity reads recorded frames and recorded events only.
- The ghost controller sets the car transform directly. It does not use a
  Rigidbody, Collider, vehicle package, or Unity physics callback.
- The track renderer builds visual geometry from the embedded Python boundary
  arrays. It does not add a MeshCollider or infer a finish line.
- Presentation choices are renderer-only. `GhostPresentationConfig` can later
  select simultaneous or staggered reveal without changing the replay payload.

## First import path

1. Open this `unity/` directory in Unity `2022.3.22f1`.
2. Install or keep the locked packages from `Packages/manifest.json`.
3. Create a scene with a `TrackRenderer` and a `ReplayGhostController`.
4. Assign a `TextAsset` produced by `tools/export_replay.py` and a car visual
   transform. Use `artifacts/replay/scripted.json` only as a local development
   fixture; it is not a tournament asset.
5. Build visual materials and camera presentation in VECTOR-owned scenes. Keep
   the runtime adapter scripts below the same schema and authority rules.

The adapter currently targets one replay document per ghost controller. A
future multi-ghost show should add a versioned replay-group manifest instead of
silently changing schema `1.1`.

## Playback rules

- Python `(x, y)` maps to Unity `(X, Y, Z) = (x, visual height, y)`.
- For a car mesh whose local `+Z` points forward, yaw is
  `90 - heading * 180 / pi` degrees.
- Position is linearly interpolated between recorded ticks. Heading uses
  shortest-angle interpolation. There is no extrapolation after the final tick.
- Effects may be triggered only by recorded event strings.
- The displayed result and lap time come from Python. Unity never starts a
  stopwatch or detects a visual finish-line crossing.

## Unity-only checks

When Unity is available, run the EditMode tests with the Unity Test Runner or:

```bash
Unity -batchmode -nographics -projectPath unity \
  -runTests -testPlatform editmode -testResults unity/test-results.xml -quit
```

The repository's offline `make test` remains the Python and Node check. It does
not require Unity, and no Unity installation is present in the current build
container.
