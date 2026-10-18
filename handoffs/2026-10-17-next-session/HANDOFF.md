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
