# Next-session handoff / updated after Pygame pivot

**Latest continuation: 2026-10-19.** Read the final section below first; it
supersedes the old browser-availability and reference-only watch limitations.

## User priority

1. Move reward editing from participant Python/script code into the UI. The
   reward is a fixed set of environment signals; UI edits weights and writes
   the reward configuration. Do not silently remove participant control of
   those weights.
2. Let participants choose the number and angles of forward-facing car rays.
   Keep the observation contract/hash honest and memory bounded. Decide and
   document safe count/angle limits before implementation; do not hide limits.

## Current state

The two requested participant controls are implemented and committed in
`0666c32 feat: add UI reward and ray controls`:

- UI edits all nine static reward signal weights, validates them to `[-100,100]`,
  writes `reward_config.json`, and records a canonical reward hash.
- UI edits ray count plus ordered angles. Limits: 1–16 unique finite angles,
  `[-90,90]` degrees, and required center ray `0`. Ray order is preserved and
  is part of the observation/config identity.
- UI writes `run_config.json` and `run_identity.json` beside the run.
- Observation contracts and public training manifest now expose ray metadata.
- Scalar/vector custom-ray parity and UI/reward regression tests exist.
- The participant UI now has a cyberpunk/touge-night visual treatment, inline
  signal definitions, and a live SVG dummy car diagram that renders ordered ray
  indices/angles without changing the backend contract.
- The first diagram pass was revised after visual feedback: rays now use a
  clean front-bumper origin, positive angles render on the visual left, labels
  use fixed non-overlapping side rails with leader lines, invalid angles are not
  drawn, and the car has explicit wheels, windows, lights, bumpers, and shadow.
  The page still needs real browser screenshot review when a browser is
  available; this container has no browser binary.
- Exact `-90`, `0`, and `+90` UI ray endpoints are now browserless-tested at
  `(514,270)`, `(320,76)`, and `(126,270)` respectively; angle placement is
  based on the angle rather than input order. UI input validation and
  `SensorConfig` both reject values outside `[-90,90]`.
- `TrainingContract` now carries reference-training reward provenance hashes,
  and reference resume checks compare the full saved contract. The UI refuses
  to overwrite an existing run with different config/reward/ray identity.
- Participant-style dummy cases completed successfully under
  `participant_run/participant-case-*`: a linear policy with 3 rays, a tiny
  8-unit MLP with 5 rays, and a second linear policy with the same rays but a
  different reward. All wrote generic artifacts, contracts, and evaluation
  reports; repeated artifact evaluation was deterministic. Same-architecture
  reward cases produced different output bias parameters.
- `docs/participant_guide.md` now documents the UI flow, all nine signal
  definitions, reward-sign guidance, ray semantics, UI-vs-custom reward
  boundaries, and dummy confirmation commands.

- The preceding generic artifact/evaluation slice is `c82c3a2 feat: start generic participant evaluation boundary`.
- Preserve unrelated dirty `.gitignore`, `participant_run/`, and `ui-run/`
  changes; they are not part of the committed feature work. Unity is
  intentionally retired and was removed in commit `0b40fc8`.
- New/changed major-gap files include:
  - `racing_event/model_artifact.py`: safe declarative NumPy MLP artifact loader.
  - `tools/evaluate_participant.py`: generic artifact evaluator CLI.
  - `tools/export_replay.py`: `--artifact-dir` replay path.
  - `racing_event/evaluation.py`: holdout-compatible validation and generic
    artifact evaluation.
  - `tests/test_model_artifact.py` and generic replay tests.
  - `tools/participant_ui.py`: process-group stop supervision and unbuffered
    child output.
  - `racing_event/ppo.py`, `racing_event/observation.py`,
    `tools/run_reference_baseline.py`: atomic writes and bounded reference
    checkpoint recovery work.

## Verification truth

- Before the latest uncommitted changes: `make test` passed 94 Python tests;
  Node replay suite passed 65 tests.
- Current verification: `make test` passed **117 Python tests**; the direct
  Node replay suite passed **65 tests**; Python compilation, JS syntax, and
  the longer UI/API smoke passed. `make render` and the Pygame headless smoke
  passed. The participant UI watch path also built a validated synthetic-oval
  replay and completed its headless Pygame process with return code 0.
  Docker was not run because Docker is unavailable.

## First next steps

1. Manually review the Pygame window and participant-dashboard watch action on a
   machine with a display; the container has no browser or display.
2. Keep ray order participant-specified (the current documented behavior); changing
   it would be a contract decision.
3. Add PPO fine-tuning/checkpoint selection around the multi-track imitation
   baseline, then add tournament replay groups and multi-model comparison.
4. Address final ONNX/private evaluator and organizer-grade resource enforcement
   only after decisions/assets exist.

## Important design constraints

- Participant model architecture remains participant-owned. The NumPy MLP is
  only the first safe generic artifact format, not the final ONNX contract.
- Participant reward code must not run in organizer evaluation. The requested
  UI reward editor should produce static weights over the published signal set.
- Ray angles are forward-facing, deterministic, finite, unique, and bounded;
  current implementation preserves participant order because order is part of
  the observation contract. Never allow an unbounded list from the UI.
- Keep `racing_event` internal compatibility identifiers unless a contract
  migration is explicit.
- Local custom-trainer budget enforcement remains cooperative; organizer-grade
  process/resource enforcement is still a separate gap.
- Real browser screenshot validation is unavailable in the container.

## Pygame pivot and serious racer (latest)

- Unity is abandoned as an active target. The Unity tree and
  `docs/unity_renderer_adapter.md` were removed. Pygame is now the active
  read-only replay renderer; the HTML viewer remains a fallback.
- New renderer path: `racing_event/replay.py`,
  `racing_event/pygame_renderer.py`, and `tools/run_pygame.py`.
- This pivot and the Unity removal are committed as
  `0b40fc8 feat: pivot replay rendering to pygame`.
- The participant UI reference watch now exports validated replay JSON and
  launches Pygame instead of building/serving HTML by default.
- Pygame is pinned to `2.6.1`; setup checks, Docker SDL runtime packages, and
  headless dummy-driver tests were added.
- `artifacts/pygame-serious/` is a 64-update fixed PPO oval run. It finished
  in 930 steps / 31.0 seconds but used 153 off-track steps, and crashed on
  flowing and tight-loop diagnostic evaluation. This identified oval overfit
  and runoff exploitation.
- `tools/train_scripted_imitation.py` is the improvement experiment.
  `artifacts/pygame-imitation/` was trained from 2,700 observations/actions
  generated by the authoritative scripted teacher across all three tracks.
  It finished all three deterministically with zero off-track steps:
  flowing 834 steps / 27.800 s, oval 863 / 28.767 s, tight loop 727 / 24.233 s.
- Pygame replay JSON QA outputs are under `artifacts/replay/pygame-imitation-*.json`.
- The UI watch smoke used `build_best_run("synthetic_oval", "participant_run/ui-rays-smoke")`;
  replay export returned `dnf_crash` for that existing UI artifact, while the
  Pygame renderer itself completed headlessly with return code 0. This is a
  renderer/integration pass, not evidence that the UI artifact is a good racer.

Next session should manually review the Pygame window, then add tournament
replay groups/multi-model comparison. Do not reintroduce Unity. Keep the
imitation racer as a strong baseline, then add PPO fine-tuning and checkpoint
selection before tournament work.

## 2026-10-19 continuation: participant UI and engine polish

### Result and boundaries

Completed the local dashboard/replay flow and research-backed evaluation audit
slice. This is not a declaration that the event release is frozen. Pygame stays
the active renderer; HTML is an explicit working fallback. No physics tuning,
signal weights, action/observation schema, scoring formula, budget policy,
private maps, or replay schema 1.1 were changed.

Code commits on `master`:

- `99a7287 fix: audit evaluation and seed reference initialization`
- `aed14b2 feat: finish participant replay UI and browser checks`
- `63e9b46 fix: bootstrap standalone participant dashboard`

The pre-existing staged `.gitignore`, `participant_run/`, and `ui-run/` changes
remain untouched and are not included in these commits. A whole-index whitespace
check finds a pre-existing final blank line in `.gitignore`; scoped checks of
this session's changes passed. Do not "fix" or commit the user's staged files.

### Implemented files and behavior

- `tools/participant_ui.py`, `tests/test_participant_ui.py`, and
  `tests/participant_dashboard.test.cjs`: explicit reference/generic-artifact
  source and Pygame/browser selection; validated browser and JSON-download
  links; actual SDL readiness and honest headless one-frame wording. Generic
  contracts/weights are never rewritten or replaced by a reference fallback.
  The selected run's saved `run_config.json` is now required for watch.
- Dashboard inputs freeze during work; invalid submissions are disabled and
  errors are inline/focusable. Duplicate actions cannot overwrite snapshots.
  Trailing process output preserves stop state, and stale readers cannot clear
  a newer process. HTTP bodies are bounded and negative/ambiguous lengths fail.
  Host/origin checks protect local code-launch actions; the CLI binds only to
  `127.0.0.1` or `localhost`. This is still not a hardened public service.
- Real screenshots found an SVG bug: object-bounding-box glow hid exact axial
  rays. Removing the line filter makes `-90/0/+90` visible; the range arc now
  curves forward rather than behind the car. Ordered ray semantics are unchanged.
- `racing_event/pygame_renderer.py`, `racing_event/replay.py`, and
  `tools/run_pygame.py`: clickable playback/speed/help, drag seeking, exact-tick
  arrows, event markers/jumps, clear current versus final outcome, cached static
  drawing, resizable 640x480 minimum, and deterministic `--screenshot` PNG.
- `racing_event/evaluation_audit.py`, `racing_event/evaluation.py`, and tests:
  bounded version-1 diagnostics for runoff/stalls/progress/boundary excess,
  reached-sector coverage, first forward crossing ticks, and per-channel action
  saturation/deltas. These consume existing authoritative info with no new
  geometry queries and do not enter reward, observations, ranking, or replay.
  Evaluation rejects random track pools and checks schema semantics/bounds.
- `tools/run_reference_baseline.py` and tests: seed Torch before constructing
  initial weights; two fresh identical-seed runs now produce identical saved
  policy/normalizer data. Also fixes the missing `StepLimitExceeded` import.
- `tools/check_ui_browser.py`, `requirements-ui-qa.lock`, `Dockerfile`,
  `Makefile`, README, and participant guide: optional pinned Chromium QA with
  an isolated `ui-qa` image target. The default offline test path needs no browser.

### Verification evidence

All commands run from the repository root:

- **Passed:** `make test`, **159 Python tests**, including the optional Node
  wrappers (no skips in this environment).
- **Passed:** `node --test tests/participant_dashboard.test.cjs tests/replay_frontend.test.cjs`,
  **72 frontend tests** (7 dashboard + 65 replay).
- **Passed:** `.venv/bin/python -m py_compile racing_event/*.py tools/*.py tests/*.py`,
  `node --check racing_event/assets/replay.js`, and session-scoped diff checks.
- **Passed:** `make render`; Pygame dummy-SDL controls/screenshots tests;
  1280x800 and 640x480 screenshots inspected. Real desktop input/display review
  is still not run.
- **Passed:** real Chromium 141 smoke through `tools/check_ui_browser.py`,
  with custom three-ray layout/weights, 8 reference-training transitions,
  browser replay tick navigation, 390/800/1440px widths and no JS errors.
  Images were read/reviewed under `artifacts/ui-polish/`: `rays-desktop.png`,
  `rays-mobile.png`, `dashboard-desktop.png`, `dashboard-390.png`,
  `dashboard-800.png`, `browser-replay.png`, and `pygame-small.png`.
- **Passed:** direct standalone dashboard launch and HTTP status. A late smoke
  caught a `tools.build_replay` import failure masked by module-import tests;
  `63e9b46` adds root bootstrapping and an actual CLI subprocess regression from
  another working directory with `PYTHONPATH` removed.
- **Passed:** retained imitation policy evaluated twice per public track, with
  identical full reports/hashes: flowing 834 steps / 27.800s, oval 863 /
  28.767s, tight loop 727 / 24.233s. All have zero off-track steps, zero maximum
  boundary excess, and 16/16 coverage/splits. This is development-track
  repeatability evidence, not an unseen-track or new-training claim.
- An initial full suite timed out at 180 seconds while six full evaluations
  ran concurrently; a standalone rerun passed, then the final 159-test run
  passed in 141 seconds with a 360-second tool timeout.
- **Not run:** Docker build (Docker absent), clean offline release rehearsal,
  private evaluator, tournament scoring, or PPO fine-tuning. No configured
  lint/typecheck exists; Python compilation is syntax checking only.

### Preview and browser QA setup

The standalone dashboard was started at **http://127.0.0.1:8766/**, with its
generated log in `artifacts/ui-polish/dashboard-server.log`; `/api/status`
returned HTTP 200/idle. It may not survive container/session restarts. Restart
with `.venv/bin/python tools/participant_ui.py`.

Browser QA is optional. Normal desktop/Linux setup is documented in README.
This minimal container has no root package-install permission, so Chromium
libraries/fonts were downloaded and extracted under `.venv/qa-apt/root/`
without sudo, and the pinned browser lives in `.venv/qa-browsers/`. Its local
rerun is:

```bash
LD_LIBRARY_PATH="$PWD/.venv/qa-apt/root/usr/lib/x86_64-linux-gnu:$PWD/.venv/qa-apt/root/lib/x86_64-linux-gnu" \
FONTCONFIG_FILE="$PWD/.venv/qa-fontconfig.xml" \
PLAYWRIGHT_BROWSERS_PATH=.venv/qa-browsers \
  .venv/bin/python tools/check_ui_browser.py
```

These extracted QA libraries/config are disposable, ignored local tooling, not
runtime inputs. The committed Docker `ui-qa` target and optional lockfile are
the portable setup. Missing fonts originally caused Chromium layout stalls;
the local font configuration resolved that before the successful browser pass.

### Remaining work / exact next action

1. On a real desktop, run Pygame on a validated replay and exercise window
   resize, pause/scrub, terminal events, and the dashboard watch action. Automated
   SDL events and screenshot review passed, but are not real-display evidence.
2. Implement bounded **optional reference-only** PPO fine-tuning of the
   imitation baseline, with per-track deterministic diagnostics and declared
   checkpoint selection. Watch now deliberately says "latest saved", not best.
   Persist optimizer/RNG/normalizer coherently before claiming exact resume;
   the current reference multi-file checkpoint is not a transactional snapshot.
3. Only then add versioned replay groups/multi-model comparison. Do not silently
   extend single-car schema 1.1. Final ONNX/operator/runtime isolation, resource
   enforcement, scored budget/ranking, map experiments and clean participant
   release rehearsal remain explicit release gaps.

Research used: `research_notes/Racing QoL and gameplay ideas/replay_ui.md`,
`gameplay_engine.md`, `anti_exploit_training.md`, and
`research_notes/Track sourcing and racing physics/racing_rl_controls.md`.
Followed the notes' safe-now presentation/audit recommendations; did not adopt
experiment-gated wall penalties, respawn changes, or new actuator/tick settings.

## 2026-10-19 continuation: resume, comparison, and submission candidate

This section supersedes the preceding remaining-engineering list. User asked
to close the open items, verify Docker after host installation, and review
Pygame screenshots. A parallel launch was interrupted; the user requested a
retry. The retry completed with two bounded agents (worker tests and
training/docs verification); both results were integrated, with no agents left
running. Preserve unrelated staged `.gitignore`, `participant_run/`, `ui-run/`.

### Implemented and committed

- `1a584de feat: persist exact reference resume and fine-tune selection`:
  `racing_event/reference_checkpoint.py`, runner and tests implement immutable
  generations/atomic pointer, optimizer/normalizer/RNG/collector/environment
  persistence, writer locking and fail-closed source/runtime identity. External
  budget is never restored. Spent-but-uncommitted work refuses exact recovery.
  Legacy artifacts require explicit non-exact warm-start in a new directory.
- Optional reference `--fine-tune --warm-start <trusted source>` preserves
  source preprocessing; no normalizer is fitted around imitation. Public-track
  best selection uses finish count, then less off-track duration, then less
  finished-lap time, strictly improving only. This is NOT event ranking.
- `91d985f feat: compare bounded replay groups in pygame`: separate group v1.0,
  1–4 ghosts, matching engine/config/track/standing tick zero, safe relative
  files/hash/size checks, exact selected telemetry/events, visibility/path
  toggles and explicit final holds. Single-car 1.1 is unchanged. See
  `reports/pygame_review_2026-10-19.md` (48 reviewed images).
- `a0b233a feat: add bounded onnx submission and isolated worker candidate`:
  `onnx_artifact.py`, `submission.py`, `isolated_policy.py`, exporter, JSONL
  worker, optional lock, Docker evaluator target and tests. Profile supports
  static dense/residual float32/float64 DAGs, opset 17, CPU ORT 1.20.1, embedded
  preprocessing and bounded resources. Full contract: `docs/submission_runtime.md`.
  The evaluator image starts separately, never from the full source-copy runtime
  stage. Only frozen model files mount into it; private track data stays host-side.
- `d1aa762 fix: preserve reference provenance through dashboard replay`:
  direct multi-track reference export, freeze only the loaded evaluation
  normalizer copy, no temporary track-contract rewriting. Browser QA reports
  inline watch failure immediately instead of hanging for a missing link.
- `docs/participant_guide.md`, README/state/first-slice docs and the delivery
  audit now distinguish implemented candidates, actual evidence and final gates.
  `reports/release_readiness_2026-10-19.md` is the consolidated readiness report.

### Actual checks

- **Tests passed:** standalone `make test`, 246 run / **245 passed / 1 skipped**
  (real Docker opt-in), 179.936 s. Optional ONNX tests ran. Log:
  `artifacts/release-checks-2026-10-19/final-tests.log`.
- **JS passed:** `node --test tests/participant_dashboard.test.cjs
  tests/replay_frontend.test.cjs`, 72 passed.
- **Syntax passed:** `.venv/bin/python -m compileall -q racing_event tools tests`
  and `node --check racing_event/assets/replay.js`. **Lint/typecheck:** not
  separately configured/run. `.venv/bin/python -m pip check` passed.
- **Smoke passed:** `make render`; real Chromium flow using the local QA library
  environment above, eight transitions, replay and 390/800/1440px, no JS errors.
- **Real training evidence:** 35 transitions versus split 16 + 19 were exact
  for policy/optimizer/RNG/collector/environment/contract/non-timing metrics;
  best update 1 retained through later regressions. Genuine interruption left
  32 spent / 16 committed and refused recovery without refund. Evidence:
  `artifacts/participant-guide-checkpoint-smoke-20261019/results.json`.
  Smoke source identities predate the final evaluator integration; use a new
  run for more training, not a bypass of deliberate runtime/source checks.
- **Replay fix confirmed:** that retained `best/` now exports a complete oval,
  862 ticks / 28.733 s, zero off-track steps. Reviewed replay/screenshot:
  `artifacts/release-checks-2026-10-19/best-oval.json` / `best-oval.png`.
- **ONNX passed locally:** retained imitation export evaluated twice on every
  development map: flowing 834 / 27.800 s, oval 863 / 28.767 s, tight loop
  727 / 24.233 s, zero off-track steps and matching repeated trajectory hashes.
  Report: `artifacts/release-checks-2026-10-19/onnx-evaluation.json`.
- **Maps passed:** `tools/build_training_manifest.py --output
  artifacts/release-checks-2026-10-19/train_manifest.json` validated three original
  synthetic fixtures. Default manifest location is `tracks/train_manifest.json`.
- **Useful failed attempt:** parallel heavy checks exceeded HTTP-test timeouts;
  fresh-reference replay also exposed the normalizer freeze regression and an
  error-label mismatch. Both code regressions were fixed and focused cases,
  then standalone Chromium/full-suite reruns passed. Failed first log retained
  as `full-tests.log`; do not claim it passed.

### Blockers and next exact action

**Docker is still absent in this session:** `docker version` returns command
not found; no accessible socket was observed. User chose **Expose host Docker**,
not a nested daemon. Once CLI and daemon are visible with the checkout shared
at the same path, run:

```bash
docker build --target participant -t synapse-participant:v1 .
docker run --rm synapse-participant:v1
docker build --target ui-qa -t synapse-ui-qa .
docker run --rm synapse-ui-qa
docker build --target evaluator -t synapse-evaluator:v1 .
SYNAPSE_TEST_DOCKER=1 SYNAPSE_TEST_DOCKER_IMAGE=synapse-evaluator:v1 \
  .venv/bin/python -m unittest tests.test_isolated_policy -v
```

Those builds, real cgroup/security restriction checks, layer inspection and
actual container inference are **not verified**. Protocol tests use real local
subprocesses with the Docker launch boundary patched; they are not sandbox proof.
Do not accept untrusted submissions yet. Add/rehearse an outer organizer job
supervisor and freeze image/base digests before scoring use.

OPEN policy remains: final budget/enforcement, ranking/DNF/retry/appeal rules,
accepted runtime/resource profile, maps/counts/family split/private assets and
parity/generalization experiments, platform matrix, exact dated window, licenses
and artifact rights. Clean public snapshot, hashed offline wheelhouse and fresh
machine rehearsal still need doing after those choices. See the readiness
report. Never put private geometry/seeds/previews in public files/history, never
reintroduce Unity, and never silently change physics or single-car replay 1.1.

## 2026-10-23 continuation: live ONNX inference and actions

### User goal

Move the competition from replay-driven model comparison to live model
inference and live actions. ONNX policies must be called during the race;
Python must remain authoritative for observations, actions, fixed-step physics,
termination, timing, and results. Renderer changes must stay minimal because
`renderer/` is owned by another team.

### Implemented state

- Added `racing_event/live_race.py` with a fixed-rate live heat scheduler.
  It supports 1 to 4 independent ghost entrants, shared fixed-start track
  state, concurrent batch-one ONNX calls, a 30 Hz physics/control default,
  configurable integer action holding, monotonic release/deadline timing, and
  `strict_scored`/`trusted_live` late-action profiles.
- The strict profile marks an entrant `dnf_inference` on a late, missing,
  malformed, non-finite, out-of-range, or failed action. Trusted-live may hold
  the last action for one interval for local demos only.
- Sessions are loaded, contract-validated, and warmed before tick zero. Live
  reports include artifact/model/contract hashes, fixed tick rate, action
  budget, latency percentiles, release jitter, deadline misses, inference
  failures, and per-entrant trajectory hashes. Output is `live-race.json`,
  not a replay.
- Added `racing_event/live_protocol.py`: loopback WebSocket text JSON protocol
  with `hello`, latest-only `state`, and reliable `end` messages. State
  publishing never blocks the physics loop; slow clients lose intermediate
  presentation packets, not simulation transitions. The renderer sends no
  actions or control commands.
- Added `tools/run_live_onnx.py` and `make live-onnx-race`. The CLI prints the
  actual WebSocket endpoint, supports `--isolate`, `--policy-budget-ms`,
  `--control-hz`, `--late-policy`, `--no-pace`, and a new output directory.
- Added `requirements-live.lock` and the `live` optional dependency in
  `pyproject.toml`, pinned to `websockets==15.0.1` with the existing ONNX
  runtime dependencies.
- Added minimal browser live integration only: `renderer/live_source.js`, a
  Connect Live URL control in `renderer/index.html`, live handling in
  `renderer/app.js`, and small styles/docs. Replay controls remain unchanged
  for replay mode and are disabled in live mode. The browser validates packets,
  displays stale status, interpolates only received poses, and never simulates.
- The previous non-renderer ONNX multi-model replay orchestration remains in
  `racing_event/race.py` and `tools/run_onnx_race.py`; live mode is a separate
  path and does not alter replay schema 1.1.

### Research

Research notes and the implementation report are saved at:

- `research_notes/Live ONNX race inference/onnx_realtime.md`
- `research_notes/Live ONNX race inference/live_racing_loops.md`
- `research_notes/Live ONNX race inference/live_state_transport.md`
- `reports/Live ONNX race inference.md`

The chosen design follows fixed-step/action-barrier patterns from racing and
robotics projects, ONNX Runtime session/thread guidance, and latest-only
WebSocket backpressure handling. The initial live event profile is 30 Hz with
a 20 ms candidate action budget; this budget still requires organizer-hardware
p99.9 measurement before it becomes a final scored rule.

### Checks from this session

- **Passed:** `.venv/bin/python -m unittest tests.test_live_race
  tests.test_onnx_race tests.test_onnx_artifact -v` — 17 tests.
- **Passed:** `node --test tests/live_source.test.cjs
  tests/participant_dashboard.test.cjs tests/replay_frontend.test.cjs` — 74
  tests.
- **Passed:** `.venv/bin/python -m py_compile racing_event/*.py tools/*.py
  tests/*.py`, `node --check renderer/app.js`, `node --check
  renderer/live_source.js`, `.venv/bin/python -m pip check`, and `git diff
  --check`.
- **Passed:** `make render` after live changes.
- **Passed:** live CLI smoke with two ONNX entrants, `--max-steps 10`,
  `--port 0`, and `--no-pace`; report showed 30/30 Hz, 10 steps per entrant,
  zero deadline misses, and an actual dynamically assigned WebSocket endpoint.
- **Passed:** real local WebSocket smoke; client received hello, ordered live
  states, and end, with final state delivery preserved before end.
- **Passed earlier in this session:** full `make test` — 254 tests, 253
  passed, 1 Docker integration skipped, before the final small callback/jitter
  and JavaScript live-source test additions.
- **Not currently claimable:** a fresh full `make test` after the final small
  live changes. Two later attempts exceeded the 360 s and 600 s tool limits
  while running the long reference-checkpoint portion. Re-run it if a full
  suite result is required; focused current checks above pass.
- **Not run:** Docker build/real isolated live heat, browser screenshot/manual
  display review, or organizer-grade p99.9 deadline measurement. Docker was
  already absent in the prior handoff and remains a release gate.

### Usage

Install live dependencies:

```bash
.venv/bin/python -m pip install -r requirements-live.lock
```

Serve the owned renderer separately:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory renderer
```

Run a live heat:

```bash
.venv/bin/python tools/run_live_onnx.py \
  --entrant team-a submissions/team-a \
  --entrant team-b submissions/team-b \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --output-dir artifacts/live/oval
```

Open the printed endpoint through the browser, for example:
`http://127.0.0.1:8765/?live=ws://127.0.0.1:8766/live/v1`.
Omit `--emit-replays`; live reports do not embed track geometry.

### Next exact action

Review/stage only the intended live-race, prior ONNX race, research, docs,
tests, and minimal renderer files; preserve the unrelated dirty `.gitignore`.
The implementation and handoff are committed in `3e2d063 feat: run live ONNX
race heats`. The unrelated `.gitignore` change remains unstaged. Then, if
event-grade live scoring is desired,
re-run the full Python suite with a longer tool allowance, build/verify the
Docker evaluator, and measure p99.9 end-to-end isolated action latency before
freezing the 20 ms budget or any final DNF/ranking policy.

## 2026-10-24 continuation: participant live-heat dashboard

### User goal

Add an easy participant-facing side-by-side live race flow to the existing
training UI. The requested presentation is motorsport/broadcast inspired
without directly naming the protected F1 brand. The UI must start and stop a
live ONNX heat, show model telemetry and resource information in the terminal,
and keep Python/ONNX/physics authoritative. `renderer/` remains owned by the
other team; only the existing live renderer contract is consumed.

### Implemented state

- `tools/participant_ui.py` now has a separate `RACE_PROCESS` supervisor and
  `/api/race/start` / `/api/race/stop` endpoints. Training, replay watch, and
  live heat processes remain separate and cannot be started over one another.
- The participant dashboard contains a full-width “Grid control / live heat”
  panel with two racer IDs, two ONNX artifact paths, track/config/output,
  WebSocket port, action budget, tick count, strict/trusted late policy, and
  optional organizer isolation. Controls freeze while the heat is active.
- The embedded dashboard renders two side-by-side cyan/amber racer tiles with
  status, latest inference latency, deadline misses, inference failures, and
  model-file size. A terminal panel keeps the bounded child output, including
  parsed `LIVE_TELEMETRY` lines and the live WebSocket endpoint.
- Resource sampling uses the supervised Linux process tree without adding a
  dependency: RSS, peak RSS, PID count, and thread count are exposed through
  `/api/status`. Runtime RSS is intentionally labeled shared across the race
  process; model-file sizes are shown per racer rather than misrepresented as
  isolated per-model RSS.
- `tools/run_live_onnx.py` emits optional `LIVE_TELEMETRY {...}` lines with
  per-entrant status, latency, deadline misses, inference failures, and model
  file size. The live runner remains authoritative and still writes only
  `live-race.json`, not a replay.
- `racing_event/live_race.py` includes `artifact_bytes` and `model_bytes` in
  live entrant summaries and sends `model_mb` in per-tick telemetry.
- Added focused tests for telemetry parsing, live command construction,
  process-tree resource reporting, race API supervision, embedded dashboard
  race controls, and live telemetry/resource rendering.

### Actual checks

- **Passed:** `.venv/bin/python -m unittest discover -s tests -v` via `make
  test`: **258 tests passed, 1 Docker integration skipped**. This is the
  project’s configured full Python suite.
- **Passed:** `node --test tests/*.test.cjs`: **75 tests passed**.
- **Passed:** `node --test tests/participant_dashboard.test.cjs`: 8 tests
  passed, including live heat single-flight/resource rendering.
- **Passed:** `.venv/bin/python -m unittest tests.test_participant_ui.ParticipantUiTests
  tests.test_participant_ui.ParticipantUiApiTests.test_live_race_api_supervises_command_and_exposes_telemetry
  -v`: 25 tests passed.
- **Passed:** Python compilation for modified Python/test files and `git diff
  --check`.
- **Passed:** `make render`.
- **Passed:** live ONNX CLI smoke with two real `artifacts/onnx-imitation`
  entrants, 8 ticks, telemetry stdout, dynamically assigned port, and a
  generated `live-race.json`; both entrants reached `dnf_timeout` at the
  configured step limit and emitted 8 telemetry lines. Model files were
  reported as 23,727 bytes each.
- **Not runnable as requested:** `python3 -m pytest -q` failed immediately
  because the system interpreter has no `pytest` module and no project
  dependencies. The repository’s actual configured suite is `make test`, which
  passed above using `.venv/bin/python` and unittest.
- **Not run:** Docker build/real isolated live heat, real-browser screenshot
  review, and organizer-grade p99.9 memory/latency measurement. Docker remains
  unavailable and the browser is not required by the default test suite.

### Current working tree and commit state

- Current intended edits are still uncommitted in:
  - `tools/participant_ui.py`
  - `tools/run_live_onnx.py`
  - `racing_event/live_race.py`
  - `tests/test_participant_ui.py`
  - `tests/participant_dashboard.test.cjs`
- `.gitignore` has an unrelated pre-existing dirty change and must remain
  unstaged.
- The preceding live implementation remains committed in `3e2d063`; the
  participant dashboard/resource-monitoring slice still needs its own focused
  commit.

### Next exact action

Inspect the final diff once more, stage only the five intended implementation
and test files plus this handoff, leave `.gitignore` untouched, create a
conventional feature commit, and report the commit hash. Do not claim Docker,
browser display, or separate per-model runtime RSS verification beyond the
evidence above.

## 2026-10-26 continuation: Monaco, Spa, and live-heat smoke

### Implemented

- Added `tracks/train/monaco.json` and `tracks/train/spa.json` as public
  development maps. Their centerlines derive from the MIT-licensed
  `bacinger/f1-circuits` GeoJSON references, are projected/resampled/scaled to
  3,337 m and 7,004 m, and receive SYNAPSE-owned 10 m road, 5 m runoff, and
  barrier geometry.
- Added `tools/build_reference_tracks.py` for deterministic regeneration and
  `docs/track_sources.md` for source, license, dimensions, and transformation
  notes.
- Refreshed `tracks/train_manifest.json`, the participant map count/docs, and
  map/manifest tests. `tools/build_training_manifest.py` now works when run
  directly from the repository root.

### Checks

- Focused map/manifest/UI tests: **50 passed**.
- Full Python suite: **258 passed, 1 Docker integration skipped** via `make test`.
- Frontend suite: **75 passed** via `node --test tests/*.test.cjs`.
- `make render`, JSON parsing, Python compilation, and `git diff --check`:
  **passed**.
- Scripted Monaco and Spa exports rendered headlessly for 120 ticks without
  crashes or off-track steps.
- Real ONNX live CLI smoke ran two entrants on each new map for 12 ticks with
  30/30 Hz, zero inference failures, and zero deadline misses. A real loopback
  WebSocket smoke received Monaco hello, ticks 0–4, and end in order.

### Renderer branch status

- Local `origin/feat/renderer` is commit `86fe080`, already an ancestor of
  `origin/master` through merge commit `c2c05ed`; `git merge-tree` reports no
  conflicts in the available refs.
- The GitHub CLI is not installed and the repository API is not accessible from
  this session, so remote PR status/push was not verified. Confirm the PR URL or
  provide authenticated GitHub CLI access before changing the remote branch.

### Next exact action

Review the generated map previews on a display, then commit the map/source/docs
slice. Keep final held-out tracks private and do not claim event-grade live
scoring, Docker isolation, or p99.9 latency from these smoke runs.

## 2026-10-26 continuation: one-click live model packaging and PR resolution

### Implemented

- Added `tools/package_live_submission.py`. It exports a validated ONNX
  submission from a reference or safe NumPy run, copies the matching simulation
  config and selected public track, includes bounded provenance snapshots, and
  writes `package_manifest.json` hashes plus a copy-ready live command. The
  output is always a new directory and never includes participant training code,
  Torch checkpoints, or reward callbacks.
- Added a **Package for live race** panel and `/api/package` endpoint to
  `tools/participant_ui.py`. The button validates the input, assigns the new
  `submission/` directory to racer A or B, and fills the live race track/config
  fields automatically.
- Updated `README.md`, `docs/submission_runtime.md`, and packaging/dashboard
  tests. The browser dashboard test now covers package assignment and the
  optional ONNX packaging test builds its own temporary artifact.
- Resolved PR #2's renderer conflicts locally. The merged renderer keeps the
  PR's night-stadium/HUD/audio/assets work and the Python live WebSocket
  integration. Resolution commit: `6682768 merge: resolve renderer PR with live
  race support`; packaging commit: `013a635 feat: add dashboard live model
  packaging`.

### Checks

- Full configured Python suite passed: **259 tests, 1 Docker integration
  skipped** via direct `unittest discover` with a 360-second allowance.
- Frontend dashboard/live-source tests passed: **11 Node tests**; `node --check
  renderer/app.js` passed.
- Real package smoke passed using the saved reference run and synthetic oval;
  the generated package ran a two-entrant four-tick live ONNX heat with a
  written `live-race.json`.
- PR API status at check time: open, `mergeable: false`,
  `mergeable_state: dirty`, base `c2c05ed`, head
  `76dd9a241ab74e...`. The resolved branch was not pushed because GitHub
  returned HTTP 403 for the unauthenticated local Git remote.

### Next exact action

Push `6682768` to the authenticated PR head branch (`feat/renderer`) or apply
the same merge resolution through GitHub. Then re-check PR #2's mergeability.
Manual renderer/browser screenshot review and Docker isolation remain separate
checks.
