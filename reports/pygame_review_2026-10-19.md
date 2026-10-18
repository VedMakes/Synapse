# Pygame replay presentation and comparison review

**Date:** 2026-10-19

**Result:** bounded replay-group creation/playback implemented; screenshot review
and deterministic SDL checks passed at **640×480** and **1280×800**.

Read the final section of `handoffs/2026-10-17-next-session/HANDOFF.md` and
`research_notes/Racing QoL and gameplay ideas/replay_ui.md` before implementation.
This report records the presentation work and publishes its initial group limits.

## Files changed by this work

| File | Change |
|---|---|
| `racing_event/replay_group.py` | Separate v1.0 group manifest, bounded no-symlink file reader, identity/hash validation, atomic exclusive builder. |
| `racing_event/pygame_renderer.py` | Shared-clock ghost rendering, cached distinguishable paths, selected telemetry/events, controls/help, responsive group layout and explicit final holds. |
| `tools/build_replay_group.py` | Standalone create CLI with published limits and useful validation errors. |
| `tools/run_pygame.py` | Optional `--group`, selected ghost, exact start tick, recorded event seeks and screenshot-able help. |
| `tools/check_pygame_review.py` | Reproducible CLI PNG capture pass using retained full laps and real current-engine constant-action crashes; writes a hash/command inventory. |
| `tests/test_replay_group.py` | Nine format/reader/builder tests, including valid unequal recordings and invalid identities, paths, hashes and limits. |
| `tests/test_pygame_group.py` | Nine real dummy-SDL tests for shared playback, selection/toggles, exact telemetry, holds, pixel cleanup, layout and deterministic CLI captures. |
| `reports/pygame_review_2026-10-19.md` | This review, contract, commands, evidence and gaps. |

`replay.py` and the single-car schema 1.1 writer/validator did not need changes.
No dashboard, engine/training/ONNX, README, Dockerfile or lock changes were made
by this work. Separate reference-training/checkpoint edits appeared in the shared
workspace during verification; they were left alone. The pre-existing staged
`.gitignore`, `participant_run/` and `ui-run/` artifacts were preserved. No staging,
committing, delegation or dependency installation was performed.

## Published replay-group v1.0 contract

The manifest has exactly these fields:

- `kind: "racing_event_replay_group"`, `schema_version: "1.0"`.
- `alignment: "standing_start_tick_zero"`.
- `end_behavior: "hold_final_pose"`.
- `identity`: recorded `track_id`, `track_hash`, complete `engine` identity
  (`code_hash` and `config_hash`), `tick_hz`, metric `car` dimensions and complete
  recorded `tick_zero` frame.
- `participants`: entries containing exactly `participant_id`, `label`, `color`,
  `file`, `file_sha256` and `payload_hash`.

Each participant still references an ordinary, independently validated schema-1.1
single-car replay. Source mode, contiguous frame range, recorded events and final
result are read from that recording, rather than duplicated or invented in the
manifest. The manifest never rewrites participant IDs or replay data.

| Limit | Published value |
|---|---:|
| Ghosts / referenced replay files | 1–4 distinct participants/files |
| Manifest | 16 KiB |
| Each replay | 16 MiB, 20,001 frames, 20,000 track samples (existing replay limits) |
| Total input | 32 MiB **including the manifest** |
| Participant display label | 1–48 printable characters |
| Reference | At most 512 characters; relative POSIX path below the manifest directory |
| Palette | Four distinct published colors: cyan, gold, violet, green |

Absolute manifest *CLI locations* are permitted; absolute references *inside the
manifest* are rejected. Empty components, `.`, `..`, backslashes, drive/URI colons,
symlink files/directories/manifest ancestors, non-regular inputs, duplicate keys,
non-finite JSON constants, unsupported versions/policies and extra fields fail.
Directory-FD traversal with `O_NOFOLLOW` prevents a symlink swap redirecting a
subsequent read. Per-file and remaining-total limits are checked before parsing.

All recordings must agree with the manifest's recorded track/config/engine/tick/
car identity and the exact same reset frame. Tick zero must be stationary, with
zero speed/lateral velocity/yaw/steer/applied controls and on-track state. This is
an equality check against recorded standing-start data, not a re-simulation of
the reset or a new engine authority. Tick zero already cannot have events or
terminal flags under schema 1.1. Participant IDs must be unique and equal their
recorded source IDs. File SHA-256, recorded payload hash, embedded track hash and
the single-replay payload validation must all pass.

Matching is against **recorded** identities, not today's source hash: compatible
historical recordings remain viewable together. Integrity hashes do not establish
tournament eligibility or authenticate a claimed engine provenance.

The shared duration is the longest recorded frame range. Each shorter ghost
holds its final recorded pose, telemetry, events and result; its sample has the
same first/next final frame and interpolation alpha zero. No extrapolation,
velocity-to-position integration, restart, disappearance, missing-tick filling or
interpolation past an individual recording's end occurs. Only x/y and shortest-
angle heading interpolate before the end. Current telemetry comes from a completed
recorded integer tick; displayed decimal formatting does not blend values.

Builder publication is atomic and exclusive: it refuses an existing destination,
including a source recording or symlink, and cleans temporary files on failure.
PNG capture also refuses replay/group-input targets and existing hard-link aliases.

## Real CLI flow and controls

From the project root with the existing `.venv`:

```bash
# Creation command used for the retained oval comparison. It creates a NEW file;
# choose a new output name if this manifest already exists.
.venv/bin/python tools/build_replay_group.py \
  --output artifacts/replay/pygame-review-oval.group.json \
  --replays artifacts/replay/pygame-imitation-synthetic_oval.json artifacts/replay/pygame-serious.json \
  --labels Imitation PPO

.venv/bin/python tools/run_pygame.py --group artifacts/replay/pygame-review-oval.group.json

# Existing single-replay entry point remains valid.
.venv/bin/python tools/run_pygame.py --replay artifacts/replay/pygame-imitation-synthetic_oval.json

# Reproduce all final captures, real crashes, manifests and inventory.
.venv/bin/python tools/check_pygame_review.py

# Exact shared tick, showing a shorter ghost held at its final frame.
.venv/bin/python tools/run_pygame.py \
  --group artifacts/replay/pygame-review-oval.group.json --start-tick 880 \
  --size 640x480 --screenshot artifacts/pygame-review-2026-10-19/group-hold-640x480.png

# Select PPO and seek its first recorded event, not imitation's event rail.
.venv/bin/python tools/run_pygame.py \
  --group artifacts/replay/pygame-review-oval.group.json --select-ghost 2 --seek-event next \
  --size 1280x800 --screenshot artifacts/pygame-review-2026-10-19/group-event-seek-1280x800.png

.venv/bin/python tools/run_pygame.py \
  --group artifacts/pygame-review-2026-10-19/crashes.group.json --show-help \
  --size 640x480 --screenshot artifacts/pygame-review-2026-10-19/four-help-640x480.png
```

Group keys are visible in the footer and help sheet:

- **1–4**, **Tab / Shift+Tab**: select telemetry/event ghost without changing time.
- **V**: show/hide selected ghost and its path; hidden ghost telemetry remains
  selectable. **T**: toggle the selected ghost's path preference independently.
- **P / N**, marker clicks: previous/next recorded event of the selected ghost;
  seeking pauses at its exact tick on the shared timeline.
- **Space**, **R**, **arrows**, **Shift+arrows**, **Home/End**, **[ / ] / − / +**:
  existing playback, restart, tick/second seeks, endpoints and speed controls.
- **H / ?**: help; **Esc / Q**: exit.

The selected ghost has a `>` in the numbered legend and an explicit panel number.
Colors, number labels, different cached dash cadences and independent visibility
make comparison possible without displacing or enlarging cars. Results stay in
manifest order, explicitly unranked. No leaderboard/scoring formula is introduced.

## Recordings and actual screenshot observations

The capture tool generated **48 final PNGs**, 24 at each requested native size,
using actual `tools/run_pygame.py` subprocesses and Pygame surfaces/fonts/drawing.
Screenshots were opened as images for visual inspection, including both sizes
for start/mid/end, help, crash and event seeking. The complete list, exact command
arguments and SHA-256 digests are in:

`artifacts/pygame-review-2026-10-19/inventory.json`

That inventory also confirms identical before/after SHA-256 values for every
retained full-lap input. Unit tests separately compare source bytes and parsed
documents before/after playback, selection, toggles, navigation and CLI captures.

### Full laps and real unequal-length comparison

| Retained recording | Final recorded tick | Recorded lap |
|---|---:|---:|
| `artifacts/replay/pygame-imitation-synthetic_flowing.json` | 834 | 27.800 s |
| `artifacts/replay/pygame-imitation-synthetic_oval.json` | 863 | 28.766666666666666 s |
| `artifacts/replay/pygame-imitation-synthetic_tight_loop.json` | 727 | 24.233333333333334 s |
| `artifacts/replay/pygame-serious.json` (oval PPO) | 930 | 31.000 s |

The imitation recordings contain clean on-track laps and recorded terminal
`lap_complete`. Their start/mid/end captures show complete track/barrier rings,
contained paths, readable telemetry and controls at both sizes. Current status
remains `IN PROGRESS` at mid-lap although the separately labeled recorded final
outcome is already available. End captures say `FINISHED`, show the recorded final
tick/event and disable the next-event action.

The oval imitation and serious PPO runs have compatible recorded identities and
form `artifacts/replay/pygame-review-oval.group.json`. The two mid-lap poses and
paths differ at one shared clock position. At shared tick **880**, imitation
stays at **863**, labeled **HOLD FINAL: FINISHED**, while PPO is still in progress.
At shared tick **930**, both final poses are shown, with imitation held and PPO
at its own end. The retained imitation final speed remains **21.77 m/s** in the
readout while its pose is held: that is its recorded telemetry, not a synthesized
zero-speed frame. The hold wording makes the distinction explicit.

Selected-ghost event seeking at the start of the oval group reaches PPO's
recorded `off_track_start` at **tick 132**, not imitation's lap event at 863.
The selected panel/rail/footer all identify ghost 2. Cars can coincide at the
standing start; numbered labels and the legend show identities without inventing
start-grid offsets or changing the metric footprint.

### Actual crash recordings

The QA tool records constant actions through the current authoritative
`evaluate_policy` / `ReplayWriter` path on the default oval/config. It validates
every output and asserts the result is `dnf_crash` before capturing it.

| File under `artifacts/pygame-review-2026-10-19/` | Constant `[steering, drive]` | Off-track tick | Contact/final tick | Elapsed |
|---|---|---:|---:|---:|
| `crash-straight.json` | `[0.0, 1.0]` | 63 | 82 | 2.7333333333333334 s |
| `crash-left.json` | `[0.15, 1.0]` | 101 | 119 | 3.966666666666667 s |
| `crash-right.json` | `[-0.15, 1.0]` | 50 | 66 | 2.200 s |
| `crash-gentle.json` | `[0.3, 0.8]` | 65 | 83 | 2.7666666666666666 s |

The straight crash trajectory digest is
`45b07617e730af25ea8a5430b65a733ecbb8bbeb9602b79e7f77fa74e47f71b3`;
payload hash is
`49d7d743f5e8a03ef177390e5b4a8ce3f3cfb6bff4d4b4ca3751110de97dd4e0`.
Repeated QA capture passes produced the same crashes and validated the existing
four-ghost manifest against their hashes.

The straight replay's event seek displays `OFF TRACK` at **63 / 2.100 s**. At
**81 / 2.700 s**, it still says `IN PROGRESS`, with no contact event leaked from
tick 82. At **82 / 2.733 s**, it displays `DNF - barrier contact`, the exact contact
event and the recorded final controls. These observations come from recorded
events, not sprite/wall intersection tests.

All four crashes share today's recorded identity and form `crashes.group.json`.
At group end **119**, selecting ghost 3 shows held **tick 66**, yaw **−0.429 rad/s**
and command **−0.150**, while the footer remains at shared 119. Earlier ghost
terminations do not stop the group's clock. Today's engine hash differs from
the historical retained full laps, so the crashes are deliberately in their own
compatible group; no historical hash was edited to force a match.

### Presentation issues found and fixed

1. **Compact crash hold text clipped.** The first four-ghost end capture at
   640×480 truncated `HOLD FINAL: DNF - barrier contact`. The final panel uses
   `HOLD FINAL: DNF crash`, with the exact barrier event still in the footer.
   Compact telemetry spacing was increased. Compare initial
   `four-end-640x480.png` with final `four-end-reviewed-640x480.png` in the evidence
   directory; both were opened during review.
2. **Names could consume final-result space.** Legend identity and result rows
   now have independent width budgets. Long names are ellipsized within their
   region; status, recorded time and `HOLD FINAL` retain their own space. A
   four-ghost/48-character-name regression checks the actual font widths.
3. **Endpoint marker remnants on ghost switch.** Boundary review found the
   rail-clearing rectangle did not include the few pixels by which an endpoint
   diamond extends past the scrubber. The complete marker footprint is now
   cleared. A real surface-byte comparison proves switching to the longest
   ghost and back restores the original event rail exactly.
4. **Ambiguous “running” word in a paused group.** Per-ghost state now says
   `IN PROGRESS`, `AT END` or `HOLD FINAL`, distinct from shared Playing/Paused.

At 640×480, single-replay help fits in its reserved side panel while preserving
the map. Four-ghost help uses a clearly headed two-column body sheet so all keys
fit; controls/legend remain visible. At 1280×800 the same layout has ample room.
The cars are necessarily small at minimum full-track size, especially with four
ghosts. Their rectangles were not inflated: the metric-footprint, heading and
Y-up tests still pass. External ordinal labels are annotations, not car geometry.

### Key PNG paths

All names below are relative to **`artifacts/pygame-review-2026-10-19/`**. Each
listed stem has both **`-640x480.png`** and **`-1280x800.png`** variants.

| Stems | What they show |
|---|---|
| `synthetic_oval-start`, `synthetic_oval-mid`, `synthetic_oval-end` | Tick 0, 431, 863. |
| `synthetic_flowing-start`, `synthetic_flowing-mid`, `synthetic_flowing-end` | Tick 0, 417, 834. |
| `synthetic_tight_loop-start`, `synthetic_tight_loop-mid`, `synthetic_tight_loop-end` | Tick 0, 363, 727. |
| `single-help`, `single-event-seek` | Visible help; exact terminal lap-event seek. |
| `crash-before-contact`, `crash-contact`, `crash-event-seek` | Tick 81, 82, 63 respectively. |
| `group-start`, `group-mid`, `group-hold`, `group-end` | Shared tick 0, 432, 880, 930; two real full laps. |
| `group-help`, `group-event-seek` | Group keys and selected PPO event tick 132. |
| `four-start`, `four-end-reviewed`, `four-help`, `four-selected-hold` | Four-crash comparison, all controls/help, selected held ghost 3. |

Initial before-change single-replay images (`before-start-640x480.png`,
`before-mid-1280x800.png`) and the initial pre-fix four-end captures are retained
as additional review evidence, outside the 48-image final inventory.

## Exact verification results

Commands ran from the project root using the existing pinned environment. The
SDL tests use real Pygame initialization, fonts, windows/surfaces and event
dispatch with `SDL_VIDEODRIVER=dummy` / `SDL_AUDIODRIVER=dummy`, not renderer
stand-ins. New group-key tests post to and consume the actual Pygame event queue.

| Command | Actual result |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -p 'test_pygame_renderer.py' -v` | Baseline **15 passed**, 2.647 s. |
| `.venv/bin/python -m unittest discover -s tests -p 'test_replay_group.py' -v` | Final focused rerun: **9 passed**, **3.875 s**, after correcting one initially invalid test fixture. |
| `.venv/bin/python -m unittest discover -s tests -p 'test_replay*.py' -v` | **16 passed**, 7.665 s; includes clock and existing frontend-wrapper checks. |
| `.venv/bin/python -m unittest discover -s tests -p 'test_pygame*.py' -v` | Final focused rerun: **24 passed**, **9.541 s**, including the strengthened selected-command assertion. |
| `make test` (tool timeout **360 s**) | Initial full pass: **189 passed**, 303.286 s. Final full pass after endpoint fix and strengthened telemetry assertion: **192 passed**, **210.939 s**, no skips/failures. The shared suite also includes separate reference-checkpoint work. |
| `.venv/bin/python tools/check_pygame_review.py` | Repeated successful real CLI passes; **48 final PNGs**, four actual validated crashes, two validated groups, unchanged retained-input hashes. |
| `.venv/bin/python -m py_compile racing_event/pygame_renderer.py racing_event/replay_group.py tools/run_pygame.py tools/build_replay_group.py tools/check_pygame_review.py tests/test_pygame_group.py tests/test_replay_group.py` | Passed. |
| `git diff --check -- racing_event/pygame_renderer.py tools/run_pygame.py` | Passed; no staging changes. |

The three-frame interactive-loop entry point also exited successfully for both
groups under real dummy SDL (these are not physical-display checks):

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python tools/run_pygame.py --group artifacts/replay/pygame-review-oval.group.json --size 640x480 --max-frames 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python tools/run_pygame.py --group artifacts/pygame-review-2026-10-19/crashes.group.json --size 1280x800 --select-ghost 3 --max-frames 3
```

`git diff --no-index --check /dev/null FILE` also passed for each new module,
tool, test and this report. The final staged diff still shows exactly the
original **100 files, 5,856 insertions, 1 deletion**.

One initial group-test run had **one fixture error**: changing a track ID also
invalidated its embedded track hash, so it failed single-replay validation before
reaching the intended group-identity check. The fixture now recomputes the valid
track hash, and both different-ID and different-content cases reach the group
identity rejection. No test was disabled or weakened.

Tests cover byte limits at the exact total boundary and one byte over; per-file
and manifest limits; maximum ghost count; path traversal, Windows/absolute refs,
symlink ancestors/files/manifests and FIFO rejection; hash tampering; distinct
IDs/colors/files; common recorded track/config/engine/tick/car/start identity;
exclusive publication and unchanged sources. Playback checks include unequal
durations, final-frame identity/alpha zero, hidden selected telemetry, different
recorded applied commands at one shared tick, pose-only interpolation, event
selection, full-range restart, cached layers, long-label layout and byte-identical
repeated CLI screenshots at both sizes.

## True remaining gaps

- This is screenshot and real **dummy-SDL** evidence. A human-operated physical
  desktop/window-manager/DPI/display review and dashboard-watch manual session
  were not run. Automated resize/input tests are not that evidence.
- Group creation/playback is a CLI/Pygame flow. There is no dashboard group picker
  or browser-group viewer in this slice.
- Input is bounded full-load JSON; 32 MiB on disk is not a promise of 32 MiB
  process memory. Lazy/chunked storage, larger grids and streaming comparisons
  remain separate work.
- The research note's optional follow/heading-up cameras, trend strips and an
  explicit reduced-motion preference are not part of this completed comparison
  flow. Exact-tick paused inspection remains available.
- Scoring/ranking/held-out eligibility and organizer provenance enforcement are
  still separate decisions. This viewer only presents recorded per-run outcomes.
- Docker/clean-machine release rehearsal and a configured lint/typecheck were
  not run; compilation is syntax checking, not type checking.
