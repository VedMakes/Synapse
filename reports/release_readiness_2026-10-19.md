# Participant release readiness — 2026-10-19

## Verdict

The local development workflow is working and regression-tested. **Do not call
this a frozen event release or an organizer-secure deployment.** Python remains
authoritative; Pygame is a read-only renderer, Unity stays retired, single-car
replay schema 1.1 is unchanged, and no physics/reward/scoring settings changed.
No held-out inputs were added or read for this work.

## Completed engineering

| Area | Result |
|---|---|
| Reference checkpoint | Immutable generations + authoritative atomic pointer; optimizer, normalizer, RNG, collector and environment restore coherently; writer lock; source/runtime identity; no budget refund |
| Optional fine-tuning | Warm-start imitation/reference weights, preserve preprocessing, bounded public-track diagnostics, retain strictly better `best/` without using training reward |
| Replay comparison | Separate bounded group v1.0; 1–4 recorded ghosts, shared standing-start clock, exact telemetry/events and explicit shorter-run final hold |
| Submission candidate | Static-vector ONNX opset-17 validator, pinned CPU runtime, embedded preprocessing, custom supported dense/residual architectures, trusted-local evaluator/export |
| Worker candidate | Model-only Docker launch/controller, bounded snapshot/protocol, memory/CPU/process/network/filesystem restrictions and deadlines; private geometry never mounted |
| End-to-end fixes | Multi-track reference `best/` exports one track directly; dashboard no longer rewrites track provenance; loaded reference normalizer freezes only in memory |

Participant architecture, algorithm, reward code, and preprocessing remain
participant-owned. Reference selection is a development heuristic, not event
ranking. Local custom-trainer budget enforcement remains cooperative.

## Verification performed

- `make test`: **246 tests run, 245 passed, 1 explicitly skipped** (real Docker
  opt-in) in **179.936 s**. Optional ONNX checks ran. Log:
  `artifacts/release-checks-2026-10-19/final-tests.log`.
- `node --test tests/participant_dashboard.test.cjs tests/replay_frontend.test.cjs`:
  **72 passed**, no skips.
- `.venv/bin/python -m compileall -q racing_event tools tests` and
  `node --check racing_event/assets/replay.js`: passed (syntax, not typechecking).
- `.venv/bin/python -m pip check`: passed. No configured separate linter or
  typechecker was run.
- `make render`: passed through real dummy-SDL rendering. Its intentionally
  120-tick scripted preview ends `dnf_timeout`, not a claimed complete lap.
- Real Chromium `tools/check_ui_browser.py`: passed custom rays/weights,
  eight-transition training, browser replay and 390/800/1440px layouts; no JS
  errors. Local library/font environment is in the session handoff. Screenshots:
  `artifacts/ui-polish/`.
- Pygame screenshot and SDL-event review: 48 images at 640×480/1280×800, group
  selection/visibility/path/help/event seeking, crashes and final holds. Details:
  `reports/pygame_review_2026-10-19.md`. Real-display interaction remains unrun.
- Real fine-tuning smoke: uninterrupted 35 transitions exactly matched split
  16 + 19 for policy/optimizer/RNG/environment/collector/contract/non-timing
  metrics. Final partial batch used exactly 3 transitions. Best stayed at update
  1 when later updates regressed. A genuinely interrupted update left 32 spent /
  16 committed and refused inexact recovery without refund. Raw development
  evidence: `artifacts/participant-guide-checkpoint-smoke-20261019/results.json`.
- After fixing replay export, the retained best drove oval in **862 ticks /
  28.733 s**, zero off-track steps. Replay and reviewed tick-200 screenshot:
  `artifacts/release-checks-2026-10-19/best-oval.json` and `best-oval.png`.

The first parallel full-suite/browser/evaluation attempt failed: loaded fresh
reference normalization was not frozen for replay, and competing heavy work
also exceeded HTTP-test timeouts. The normalizer and error-label regressions
were fixed, the focused cases passed, and standalone browser/full-suite reruns
passed. The first failed log is retained as `full-tests.log`, not presented as a
pass. Browser QA now fails promptly on an inline watch error rather than waiting
two minutes for a link that cannot appear.

### ONNX development-track evidence

Exported trusted imitation weights and ran:

```bash
.venv/bin/python tools/evaluate_participant.py \
  --artifact-dir artifacts/onnx-imitation --config config/simulation.json \
  --track tracks/train/synthetic_flowing.json \
  --track tracks/train/synthetic_oval.json \
  --track tracks/train/synthetic_tight_loop.json --repeats 2 --max-steps 1200 \
  --output artifacts/release-checks-2026-10-19/onnx-evaluation.json
```

| Development track | Ticks | Lap time | Off-track ticks |
|---|---:|---:|---:|
| Flowing | 834 | 27.800 s | 0 |
| Oval | 863 | 28.767 s | 0 |
| Tight loop | 727 | 24.233 s | 0 |

Every repeat matched its trajectory hash. This is **trusted-local development
repeatability**, not real container evidence or unseen-track generalization.
Tests also cover non-reference residual architecture and scalar/per-field
normalization, malformed/external/dynamic graphs, resource limits, protocol
faults, strict sequences/actions, snapshot integrity and cleanup.

### Maps and packaging audit

`.venv/bin/python tools/build_training_manifest.py --output
artifacts/release-checks-2026-10-19/train_manifest.json` validated all **three
original synthetic development tracks**, recording canonical and delivered-byte
hashes and current config/rays. They remain development fixtures, not a chosen
final map set. The tool's default output is `tracks/train_manifest.json` (not
inside `tracks/train/`). This work did not generate or publish private holdouts,
nor claim a repository-history secrecy or license clearance review.

## Remaining gates, in order

1. **Docker unavailable in this session.** `docker version` returned command not
   found; no accessible daemon socket was observed. The user chose to expose
   host Docker. Reconnect/expose CLI + daemon with the workspace shared at the
   same path, then build participant, UI-QA and evaluator targets and run their
   real checks. Do not install a nested daemon as a silent substitute.
2. **Runtime isolation/release rehearsal.** Run the opt-in worker test, verify
   actual cgroup/no-network/read-only/non-root behavior and image layers, freeze
   image/base digests, add the outer job supervisor/cancellation/runbook and
   rehearse failed-worker handling. Docker commands alone are not proof.
3. **Event policy approval.** Confirm budget model/number/enforcement, ranking
   and DNF aggregation, retry/appeal handling, supported architecture/runtime
   profile and scoring resource rules, platform matrix, exact dated window,
   participant artifact rights and distribution boundary. Existing worker
   defaults and public-development selection do not decide these.
4. **Maps and experiments.** Select original public route families and private
   organizer-held development/final sets; verify provenance/contributor rights,
   split by route family, measure generalization and difficulty parity, then
   freeze manifests privately/publicly as appropriate. No private coordinates,
   seeds, previews, or raw payloads belong in this repo or its history.
5. **Clean delivery.** Create the chosen immutable participant snapshot, explicit
   project license/notices, platform-matched hashed wheelhouse/image bundle, and
   fresh-machine offline setup/test rehearsal. Existing lockfiles do not mean
   the first install works without network. Review the pre-existing staged run
   artifacts separately; do not accidentally include them in a public release.

Recommended sequence is to prove Docker on the current engineering candidate
before freezing policy/maps or packaging a public release. No OPEN decision was
silently selected by this implementation.
