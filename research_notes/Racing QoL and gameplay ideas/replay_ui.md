# Replay and UI quality-of-life ideas

**Research date:** 2026-10-16
**Scope:** low-risk, dependency-free improvements for a Python-authoritative racing
simulator whose recorded output will eventually be displayed by Unity. The current
repository is the first Python-only vertical slice; Unity is not implemented here.

## Bottom line

The current replay boundary is already a good low-risk foundation: Python records
the authoritative pose, telemetry, events, and result; the viewer only advances a
visual clock and interpolates position/heading; the writer is atomic and streams
frames; the offline HTML has no network or frontend dependency. The safest next
steps are therefore viewer-only additions that derive from existing frames:

1. Add an event-marker rail with labeled, keyboard-reachable jumps.
2. Add a small, documented keyboard command layer and a reduced-motion mode.
3. Add optional full-track/follow-car/heading-up presentation modes without
   changing coordinates or simulation results.
4. Add a compact telemetry trend strip using a dependency-free min/max envelope,
   while preserving exact-tick readouts for values and events.
5. Keep the single-replay schema as-is, but reserve a future replay-group
   manifest for multiple ghosts rather than smuggling several cars into v1.1.
6. Keep JSON/HTML bounded for now. Treat indexed chunked or columnar replay as a
   separate large-recording milestone, not a prerequisite for these QoL features.

These recommendations do not copy code, UI, or assets from the references below;
the references are evidence and design constraints only.

## Current repository evidence

### Existing boundary and fit

**Evidence — already implemented:**

- [README](../../README.md#replay-ui-boundary-python-simulation-only) describes a
  dependency-free, read-only offline preview with play/pause, restart, next tick,
  scrubbing, playback speed, telemetry, events, track geometry, car pose, and the
  Python result. The README also states that Unity and the renderer must not
  compute contacts, resets, rewards, progress, finishes, or lap times.
- [Replay contract](../../docs/replay_contract.md#frame-fields-and-time) defines
  tick 0 as the reset pose, one frame per executed transition, fixed units, exact
  event vocabulary, and an explicit no-reset boundary. This supports UI-derived
  markers and cameras without adding simulation authority.
- [Replay contract storage section](../../docs/replay_contract.md#storage-integrity-and-memory)
  records a 16 MiB input cap, 20,001-frame cap, 20,000-track-sample cap, atomic
  publishing, constant frame-history memory in the writer, and a bounded offline
  reader. This is a strong guardrail for memory-conscious work.
- [ReplayWriter](../../racing_event/trajectory.py#L224-L299) validates and writes
  frames incrementally to a temporary sibling file, then publishes only after a
  valid result/footer. It does not retain the frame list while recording.
- [Evaluation](../../racing_event/evaluation.py#L146-L267) already supports a
  `frame_sink` while otherwise retaining only counters and a streaming trajectory
  digest. Summary-only evaluation is therefore available without adding a viewer.
- [Viewer playback](../../racing_event/assets/replay.js#L85-L170) uses a visual
  time clock, exact tick selection, shortest-angle heading interpolation, and
  no writes to the recording. [Viewer rendering](../../racing_event/assets/replay.js#L248-L335)
  caches the static background/path and redraws the moving car separately.
- [Viewer telemetry](../../racing_event/assets/replay.js#L366-L405) deliberately
  updates telemetry only when the completed tick changes. It displays the
  recorded state and events rather than blending or inferring them.
- [Accessibility surface](../../racing_event/assets/replay.html#L72-L87) already
  uses native buttons, a native range input, visible labels, status text, and
  `aria-valuetext`; [CSS](../../racing_event/assets/replay.css#L15-L23) has visible
  `:focus-visible` styling and 44px controls.
- [Frontend tests](../../tests/replay_frontend.test.cjs#L336-L545) run without a
  browser or npm package through a DOM/canvas double. New UI work can preserve
  this low-cost test seam.

**Implication:** event indexing, keyboard commands, marker drawing, telemetry
plots, camera transforms, and reduced-motion behavior can be implemented in
`racing_event/assets/` plus dependency-free tests. They do not require stepping
Python physics or changing the replay payload.

## Evidence from public sources

All public sources below were fetched on 2026-10-16. “Evidence” records what a
source says. The following “Ideas” sections are proposals for this repository,
not claims that the sources prescribe a racing UI.

### Playback timing and offline rendering

- **Evidence — browser animation timing:** MDN’s
  [`requestAnimationFrame()` documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)
  says callbacks run before repaint, are one-shot, may be paused in background
  tabs, and should use the supplied timestamp (or another real clock) rather
  than assuming a fixed display refresh rate. The current viewer’s anchored
  `performance.now()` clock is consistent with this principle.
- **Evidence — static canvas layers:** MDN’s
  [canvas optimization guidance](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas)
  recommends pre-rendering repeated/static primitives to an offscreen canvas,
  layering static and dynamic content, batching path calls, and using
  `requestAnimationFrame()`. The current viewer already caches one background
  canvas; this supports adding marker/camera overlays as separate redraw layers
  rather than rebuilding the track every animation frame.
- **Evidence — standard-library file mapping:** Python’s
  [`mmap` documentation](https://docs.python.org/3/library/mmap.html) describes
  file-backed byte mappings that can be sought and read like file/byte-array
  objects. It is a possible building block for a future indexed binary/chunked
  viewer, but it does not make the current JSON parser lazy by itself.

**Ideas:**

- Keep the current clock model. Do not derive simulation time from Unity frames,
  browser refresh count, or a newly introduced wall-clock stopwatch.
- Add a cached marker/background layer only when markers are visible. A redraw on
  resize, camera-mode change, or marker-toggle is enough; the moving car/ghost
  layer can remain the only per-RAF dynamic layer.
- If a future viewer is moved into a worker, keep the same recorded-clock API and
  event semantics. Worker/offscreen rendering is an optimization, not a reason
  to alter the Python contract.

### Keyboard and accessibility

- **Evidence — slider keys:** The W3C ARIA Authoring Practices
  [Slider Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/slider/) specifies
  Right/Up to increase, Left/Down to decrease, Home/End for range endpoints, and
  optional Page Up/Page Down for larger changes. It also calls out
  `aria-valuetext` when the raw value is not user-friendly; its media-seek
  example formats seconds as time. The current native range already implements
  arrows, Home/End, Page Up/Down, and formatted `aria-valuetext`.
- **Evidence — keyboard operability:** MDN’s
  [Keyboard accessible guide](https://developer.mozilla.org/en-US/docs/Web/Accessibility/Guides/Understanding_WCAG/Keyboard)
  says pointer/touch interactions need equivalent keyboard interactions, native
  interactive semantics are preferable, focus order should remain logical, and
  focusable elements need visible focus styling. These rules favor keeping
  buttons/range/select elements native and avoiding a canvas-only control layer.
- **Evidence — reduced motion:** MDN’s
  [`prefers-reduced-motion`](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
  describes a user preference to remove, reduce, or replace nonessential motion,
  including large-object panning/scaling that can trigger vestibular discomfort.

**Ideas:**

- Add a short “Keyboard” disclosure next to playback controls: `Space` play/pause
  (when focus is not in a text/select control), `Left/Right` one tick, `Shift +
  Left/Right` one second, `PageUp/PageDown` ten seconds, `Home/End` endpoints,
  `[`/`]` slower/faster, and `R` restart. Keep the range’s native behavior when
  it has focus; never hijack text-entry keys.
- Add `aria-keyshortcuts` to buttons or an accessible help region only after the
  shortcuts are stable. A visible help label is more useful than a hidden
  shortcut claim.
- Under `@media (prefers-reduced-motion: reduce)`, keep the visual clock and
  controls but snap the car to the selected recorded tick instead of smoothly
  interpolating between ticks. Still expose the exact current telemetry/event
  tick. This is a reversible viewer preference and does not change replay data.
- Retain text labels and state words next to color: `OFF TRACK`, `ON TRACK`,
  `barrier_contact`, and the result status should remain understandable in
  forced-colors or color-vision conditions.

### Event markers and replay navigation

- **Evidence — replay annotations:** The MIT-licensed
  [rrweb project](https://github.com/rrweb-io/rrweb) separates a serializable
  initial snapshot from timestamped/incremental replay events and has a replay
  plugin/event model. Its MIT license is listed by the GitHub repository API and
  in its [license file](https://github.com/rrweb-io/rrweb/blob/main/LICENSE).
- **Evidence — marker-oriented player UI:** rrweb-player’s MIT-licensed
  [README](https://github.com/rrweb-io/rrweb/blob/main/packages/rrweb-player/README.md)
  documents play/pause, speed, `goto`, range playback, a playback-position
  slider, captions, timeline markers, custom marker colors, and keyboard seeking.
  This is a session-replay reference, not racing evidence, but it demonstrates a
  useful separation between a replay clock and navigable annotations.
- **Repository evidence:** replay v1.1 already stores a list of allowed event
  names on each frame and tests that events remain at their exact recorded tick
  ([contract](../../docs/replay_contract.md#frame-fields-and-time),
  [validation](../../racing_event/trajectory.py#L166-L191)).

**Ideas:**

- Build an in-memory `markers` array at viewer boot by scanning `frames[i].events`:
  `{tick, time_s, event, label, severity}`. This is derived data, not a new
  authority or payload field. Add one marker per event, or coalesce only for
  display while retaining the full event list in the tick readout.
- Draw a thin marker rail aligned with the scrubber. Use text/icon/pattern
  differences in addition to color; `barrier_contact` and `lap_complete` should
  be visually prominent, while off-track start/end should be less prominent.
- Make markers keyboard reachable as native buttons or links. Activating one
  should seek and pause at the exact event tick, update the live status text, and
  preserve the existing “telemetry from a completed tick” rule.
- Add “previous/next event” controls. They should navigate the derived index and
  never search geometry or infer an event from a visible collision.

### Telemetry presentation

- **Evidence — lightweight time-series reference:** The MIT-licensed
  [uPlot repository](https://github.com/leeoniya/uPlot) describes a small Canvas
  2D time-series chart with multiple series, toggles, cursor synchronization,
  and a focus on memory/performance. Its license is MIT in the GitHub API and
  [license file](https://github.com/leeoniya/uPlot/blob/master/LICENSE). It is a
  reference for product shape/performance tradeoffs, not a dependency to add.
- **Evidence — decimation semantics:** Chart.js’s MIT-licensed
  [data-decimation documentation](https://www.chartjs.org/docs/latest/configuration/decimation.html)
  describes min/max decimation for retaining peaks and LTTB for reducing points
  while preserving trends. The repository’s [MIT license](https://github.com/chartjs/Chart.js/blob/master/LICENSE.md)
  is visible in GitHub metadata and the linked license file. Again, use the
  algorithmic idea only; adding Chart.js would violate the current dependency-free
  viewer constraint.
- **Repository evidence:** The current frame schema already has speed, lateral
  velocity, yaw rate, steer, steering command, throttle/brake, off-track, flags,
  and events ([contract](../../docs/replay_contract.md#frame-fields-and-time)).
  The viewer formats these values and intentionally does not blend telemetry.

**Ideas:**

- Add a compact “Telemetry trend” canvas beneath the current-value grid. Start
  with speed, throttle/brake, and steer as toggleable series; draw a vertical
  cursor at the current tick and marker ticks in the same coordinate system.
- For a dependency-free implementation, map the visible tick interval to pixel
  columns and reduce each column to min/max (plus the first/last sample when
  needed). This preserves spikes without allocating a second full series. If
  the source frames are already resident because of the current JSON viewer,
  this improves draw cost and clarity more than process memory.
- Keep exact readouts as the primary diagnostic surface. A plotted envelope is a
  presentation summary and must not replace the recorded value at the selected
  tick or alter hashes/results.
- Add a “copy current tick as text” button only if an explicit export policy is
  wanted; it should copy formatted recorded values, not a re-simulated result.

### Camera and track presentation

- **Evidence — future Unity pose injection:** Unity 6’s official
  [`Transform.SetPositionAndRotation`](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Transform.SetPositionAndRotation.html)
  API sets world position and rotation together and says doing so is more
  efficient than assigning each property separately. This aligns with the
  current contract’s direct-transform recipe and reinforces that a replay car
  should not be driven by Unity physics.
- **Evidence — future camera composition:** Unity’s official
  [Cinemachine Camera component documentation](https://docs.unity3d.com/Packages/com.unity.cinemachine@3.1/manual/CinemachineCamera.html)
  describes separate tracking/look-at targets, priority/channel selection,
  blending hints, follow/orbital/third-person/spline-dolly position modes, and
  performance-tunable standby updates. These are renderer-side camera concepts,
  not Python simulation state.
- **Evidence — future path/trail drawing:** Unity’s official
  [Line Renderer component reference](https://docs.unity3d.com/6000.0/Documentation/Manual/class-LineRenderer.html)
  documents rendering a line between points in 3D space. It is a plausible
  renderer primitive for a recorded path or ghost trail, but its visual output
  must be fed by recorded data.
- **Repository evidence:** The browser viewer currently uses equal-axis full-track
  fitting, `+Y` up, a recorded path, and a car rectangle; its track fit includes
  the entire path and car footprint ([viewer](../../racing_event/assets/replay.js#L172-L205)).
  The Unity mapping and interpolation rules are already written down
  ([contract](../../docs/replay_contract.md#unity-adapter-recipe)).

**Ideas:**

- Add three browser view modes: **Full track** (current), **Follow car** (camera
  center follows the interpolated pose), and **Heading up** (follow plus rotate
  the view so the car’s forward direction is up). Keep a clear “North up”/full
  track fallback and an always-available text description for accessibility.
- Implement each mode as a pure coordinate transform around the existing
  `projectPoint`; do not change `frame.x`, `frame.y`, `heading`, track geometry,
  hashes, or interpolation semantics.
- Add optional start/finish and event labels from `track.start_line_s` and
  recorded events only. Do not draw a guessed finish line from a mesh crossing.
- Treat cinematic shots, spline dolly paths, camera blends, 3D car meshes,
  shadows, and Unity input as future renderer work. They are not low-risk
  changes to this Python repository.

### Multiple ghosts and comparison

- **Evidence — multiple synchronized series:** uPlot documents multiple series,
  toggles, and cursor synchronization in its [feature list](https://github.com/leeoniya/uPlot#features).
  This supports the general UI idea of a shared time cursor and independently
  toggled comparison series, while not proving a racing-specific file design.
- **Evidence — replay groups need identity:** rrweb’s README describes an
  initial snapshot plus incremental events and a player that controls a shared
  replay timeline. This is analogous evidence for “one clock, many recorded
  streams,” not evidence that the current racing schema already supports it.
- **Repository evidence:** v1.1 has one `source`, one `car`, one embedded `track`,
  and one `frames` array ([contract](../../docs/replay_contract.md#document-fields)).
  The contract explicitly says future multi-episode replays need boundaries and
  must not interpolate teleports. It has no participant array, per-ghost
  identity, missing-tick policy, or group-level result.

**Ideas:**

- **Future format proposal:** a replay-group manifest should contain one shared
  track/config identity and a list of participants. Each participant needs an
  immutable id/label/color, source mode, hashes, tick rate, frame range, result,
  and event index. Prefer separate per-participant recordings referenced by a
  manifest at first; this lets a viewer load/unload ghosts and avoids copying
  every frame into one JSON object.
- Define alignment before UI work: same `tick_hz` and tick zero, or explicit
  time mapping; explicit termination/missing-frame behavior; no interpolation
  across a reset, gap, or end-of-recording. A ghost that finishes early should
  disappear or hold its final pose according to a declared policy, not silently
  teleport.
- Use one visual clock and one selected participant for telemetry; other ghosts
  can show pose/path/result only until comparison semantics are specified.
- In the current viewer, a low-risk preparatory change is only a UI abstraction
  around “one active recording.” Do not add fake ghost fields to schema 1.1 or
  reuse `participant_id` as a list.

### Memory-conscious offline viewers

- **Evidence — current bounded model:** The repository caps replay bytes/frames,
  rejects malformed input, and keeps the Python writer constant-memory. This is
  sufficient for the present local developer preview and should be preserved.
- **Evidence — columnar future option:** The Apache Arrow
  [Columnar Format specification](https://arrow.apache.org/docs/format/Columnar.html)
  describes language-agnostic columnar buffers, sequential access, O(1) random
  access, SIMD/vectorization-friendly layout, zero-copy relocation, chunked large
  arrays, and serialized record batches. The Apache Arrow GitHub repository is
  [Apache-2.0 licensed](https://github.com/apache/arrow/blob/main/LICENSE.txt),
  with additional vendored notices; it is suitable evidence for a future format,
  not a reason to add a large dependency now.
- **Evidence — decoupled replay serialization:** rrweb’s MIT-licensed
  [serialization notes](https://github.com/rrweb-io/rrweb/blob/main/docs/serialization.md)
  explain why replay data can use a compact initial snapshot plus incremental
  changes instead of repeatedly serializing a full state. The racing analogue
  would be a static header/track plus compact frame/event chunks, but the exact
  representation remains a design task.

**Ideas and fit:**

| Option | Memory behavior | Fit now | Change size |
|---|---|---|---|
| Keep bounded JSON + atomic streaming writer | Writer is constant-history; viewer has bounded full-load cap | Yes | None |
| Derived marker index and on-demand telemetry reduction | Small derived arrays; no payload change | Yes | Low |
| Separate lightweight preview export with fewer rendered samples | Lower viewer data, but must preserve event ticks and label it as a preview | Maybe, after measurement | Low/medium |
| JSON Lines or chunked JSON with byte offsets | Can seek by chunk, but standalone browser bundling and validation become more complex | Later | Medium/high |
| Binary columnar chunks / Arrow IPC + mmap/worker reader | Better column access, random access, and multi-ghost loading | Only for large recordings | Major |

Do not claim that `mmap`, Arrow, or decimation make a current embedded JSON file
memory-free. The current `load_replay()` and browser JSON parse necessarily hold
the bounded recording. First measure real replay sizes and profile the viewer;
then choose a format that preserves hashes, schema validation, exact events, and
offline operation.

## Recommended backlog, ordered by risk

### P0 — low-risk, schema-preserving

1. **Event marker rail and jump controls.** Derive markers from `frames[*].events`;
   add clickable/keyboard-reachable markers, previous/next event, marker labels,
   and an accessible status announcement. Add JS tests for exact tick seeks and
   terminal events.
2. **Keyboard help and reduced motion.** Keep native controls; add non-conflicting
   global shortcuts, visible help, and `prefers-reduced-motion` snapping. Add
   tests that typing/select controls are not hijacked and that reduced motion
   never changes the selected telemetry tick.
3. **View modes without new data.** Add full-track/follow-car/heading-up modes as
   pure transforms. Keep equal axis scale where appropriate, clamp to track/path
   bounds, and test known orientation/center cases.
4. **Small dependency-free telemetry trend.** Add one canvas plot with toggleable
   channels and a min/max-per-pixel envelope. The current exact-value panel and
   event readout remain authoritative.
5. **Non-color event/result cues.** Use labels, symbols, dash/pattern differences,
   and forced-colors-friendly styles for off-track, contact, finish, and DNF.

### P1 — useful after measurement or a small contract extension

- Add a preview/export mode that intentionally down-samples pose/telemetry but
  always retains tick 0, terminal tick, every event tick, and enough samples for
  a faithful path. Give it a distinct preview schema/flag and test that it never
  gets mistaken for a scoring replay.
- Add a local “replay manifest” for a folder of independent single-run files,
  without changing schema 1.1. The first viewer can compare metadata/results and
  only load one run at a time; this validates naming and identity before a
  multi-ghost renderer exists.
- Add a dedicated diagnostic export (CSV/JSONL) only if users need external
  analysis. Keep it derived from the validated replay and never use it as the
  authority for event results.

### P2 — major changes; defer until requirements are real

- Native multi-ghost playback with shared clocks, alignment, visibility, per-ghost
  color/identity, missing-frame rules, and combined camera framing.
- A replay-group schema or binary columnar/chunked container with random access,
  lazy browser loading, worker decoding, and compatibility/version negotiation.
- Unity Cinemachine shot authoring, 3D meshes/materials, visual effects, camera
  blends, or renderer-side recording/export.
- Physics-assisted visual replay, collision reconstruction, finish-line
  detection, or any renderer-to-simulation control channel. These violate the
  current authoritative-boundary intent unless explicitly redesigned.

## Acceptance criteria for the P0 ideas

- The same replay payload, hashes, results, units, event names, and exact tick
  semantics remain valid.
- No browser/Unity code calls Python physics or infers an event from geometry.
- Every marker action resolves to a recorded integer tick; telemetry is read from
  that frame, while only pose/heading may be interpolated between frames.
- Keyboard-only use can reach playback, scrubber, marker, and view-mode controls;
  focus is visible; shortcut help is discoverable; text remains meaningful without
  color.
- Reduced-motion mode remains usable and does not disable seeking or exact-tick
  inspection.
- Static track/path work is cached; per-frame work is bounded by visible layers,
  not by a newly allocated copy of every frame.
- Tests remain runnable by `make test` and, when Node is present, the existing
  dependency-free frontend test command. A visual Unity parity pass is still a
  separate milestone.

## Source register

### Repository sources

- [Project README](../../README.md) — current scope, commands, replay viewer,
  dependency-free/offline constraints, and Unity boundary.
- [Replay contract](../../docs/replay_contract.md) — schema 1.1, authority,
  frame/event semantics, Unity mapping, integrity, and limits.
- [Replay writer/reader](../../racing_event/trajectory.py) — atomic streaming
  writer, validation, hashes, and bounded JSON loading.
- [Policy evaluation](../../racing_event/evaluation.py) — deterministic fixed-start
  evaluation, low-memory counters/digest, and optional frame sink.
- [Replay viewer JS/HTML/CSS](../../racing_event/assets/replay.js),
  [HTML](../../racing_event/assets/replay.html), [CSS](../../racing_event/assets/replay.css)
  — current playback clock, rendering, telemetry, controls, semantics, and styles.
- [Frontend tests](../../tests/replay_frontend.test.cjs) — dependency-free playback,
  validation, DOM/canvas, keyboard, and error-path coverage.

### Public standards and official documentation

- [W3C ARIA APG Slider Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/slider/)
  — keyboard increments, endpoints, optional page steps, and `aria-valuetext`.
- [MDN Keyboard accessible](https://developer.mozilla.org/en-US/docs/Web/Accessibility/Guides/Understanding_WCAG/Keyboard)
  — keyboard-equivalent interactions, native semantics, logical focus, and focus styling.
- [MDN `prefers-reduced-motion`](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
  — user motion preference and replacing/reducing nonessential animation.
- [MDN `requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)
  — repaint scheduling, timestamps, refresh-rate independence, and background behavior.
- [MDN Optimizing canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas)
  — offscreen/static layers, batching, reduced redraws, and canvas animation guidance.
- [Python `mmap`](https://docs.python.org/3/library/mmap.html) — standard-library
  memory-mapped file behavior for a possible future indexed reader.
- [Unity `Transform.SetPositionAndRotation`](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Transform.SetPositionAndRotation.html)
  — direct world-pose application in the future renderer.
- [Unity Cinemachine Camera](https://docs.unity3d.com/Packages/com.unity.cinemachine@3.1/manual/CinemachineCamera.html)
  — future tracking, camera modes, priorities, blends, and update policy.
- [Unity Line Renderer](https://docs.unity3d.com/6000.0/Documentation/Manual/class-LineRenderer.html)
  — future renderer primitive for recorded paths/trails.

### Public GitHub projects, used as references only

- [rrweb](https://github.com/rrweb-io/rrweb) — MIT; replay snapshots/incremental
  events and plugin-oriented replay architecture. [MIT license](https://github.com/rrweb-io/rrweb/blob/main/LICENSE).
- [rrweb-player README](https://github.com/rrweb-io/rrweb/blob/main/packages/rrweb-player/README.md)
  — MIT project’s documented timeline markers, captions, speeds, range playback,
  and keyboard seeking.
- [uPlot](https://github.com/leeoniya/uPlot) — MIT; small Canvas time-series
  reference with multiple series/toggles/cursor sync. [MIT license](https://github.com/leeoniya/uPlot/blob/master/LICENSE).
- [Chart.js decimation documentation](https://www.chartjs.org/docs/latest/configuration/decimation.html)
  and [repository](https://github.com/chartjs/Chart.js) — MIT; min/max/LTTB
  decimation reference. [MIT license](https://github.com/chartjs/Chart.js/blob/master/LICENSE.md).
- [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html)
  and [repository](https://github.com/apache/arrow) — Apache-2.0; columnar,
  chunked, random-access/zero-copy format evidence for a future large-replay
  milestone. [License and notices](https://github.com/apache/arrow/blob/main/LICENSE.txt).

## 2026-10-19 implementation update

The active target is now Pygame (the Unity references above are historical).
Single-replay Pygame has cached geometry/path drawing, visible mouse controls,
scrubbing, exact-tick keys, recorded-event markers/jumps, keyboard help,
resizable reserved regions and clear current/final outcome text. The dashboard
can export either a reference or safe participant artifact and exposes both
browser fallback and JSON-download links. Actual Chromium flow/layout QA now
exists in `tools/check_ui_browser.py`; screenshots caught and corrected clipped
axial SVG rays and an incorrectly oriented preview range arc. Read the latest
handoff for evidence. Replay schema 1.1 is unchanged; follow-car cameras,
telemetry plots, versioned replay groups and multi-ghost work remain deferred.
