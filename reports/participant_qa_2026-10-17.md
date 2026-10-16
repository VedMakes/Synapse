# SYNAPSE participant QA notes

**Date:** 2026-10-17
**Perspective:** participant using the checked-out module, not the organizer
**Memory posture:** tiny vector batches, zero-action/light policies, no long training run

## Journey exercised

### Setup and environment import

```bash
python3 tools/setup_participant.py --check
.venv/bin/python tools/setup_participant.py --check
```

The project virtual environment reported NumPy `2.4.6` and optional Torch
available. The system interpreter has NumPy but no Torch. Before this QA pass,
importing `racing_event` with that system interpreter failed because the package
eagerly imported the optional PPO module. After the fix, environment-only
imports work without Torch; requesting the PPO reference still correctly
requires Torch.

### Reference starter, custom weights, and resume

The reference starter was run with three lanes, rollout length two, and a
seven-transition budget:

```bash
.venv/bin/python tools/train_participant.py \
  --trainer examples/participant_train.py \
  --reward examples/baseline_reward.json \
  --steps 5 --budget-steps 7 --n-envs 3 --rollout-length 2 \
  --output-dir artifacts/participant-qa/reference-run
```

Observed first run:

- exactly 5 transitions charged;
- one metrics row;
- reference policy, normalizer, contract, state, report, and budget artifacts;
- 2 transitions remaining.

Repeating the same command charged exactly 2 more transitions, reached 7/7,
and did not overrun the budget.

Two lightweight reward-weight files were exercised:

- `progress_only.json`: progress weight `25.0`, all other weights zero;
- `crash_averse.json`: progress `4.0`, crash `-40.0`, off-track `-8.0`, time
  `-0.05`.

Both runs completed with separate reward hashes and four-transition budgets.
Changing the reward file while resuming an existing run was rejected with a
specific reward-hash mismatch.

### Participant-owned model and reward function

A disposable light trainer used:

- `VectorizedRacingEnv(..., reward=None)`;
- a zero-action policy;
- a custom reward callback using `progress - 2 * crash + 0.01 * throttle`;
- three lanes and five total transitions.

The callback ran exactly five times. The partial final batch had active-mask
shape `[[true, true, true], [true, true, false]]`; the inactive lane was not
passed to the callback and received zero reward.

Additional checks covered:

- callback mutation of observation/action inputs does not mutate environment state;
- NaN reward output is rejected after the reserved transition;
- the collector refuses reuse after a reward callback failure instead of retrying
  with a stale observation;
- out-of-range policy requests are retained in the rollout batch for saturation
  diagnostics, while the reward callback sees the actually applied clipped action;
- unbounded `collect(10_000_000)`-style requests are rejected by the default
  `max_batch_steps=8192` cap.

### UI and failure cases

Before fixes, the participant UI probes found:

- `steps=1.5` silently became `1`;
- `steps=Infinity` raised an uncaught `OverflowError` through the HTTP path;
- missing config paths reached the child trainer and produced a traceback;
- the Stop button stayed disabled in the page;
- a stopped process remained in the `stopping` state indefinitely;
- the system interpreter could import NumPy but not the reference Torch trainer.

The corrected UI now:

- rejects fractional, boolean, NaN, Infinity, zero, and negative numeric values;
- returns structured HTTP 400 errors for malformed start requests;
- validates trainer/config/track/reward/output paths before spawning;
- enables Stop while a process is running and reports a terminal `stopped` state;
- escalates to kill if a trainer ignores termination;
- checks Torch only when the bundled reference trainer is selected;
- allows a custom trainer to omit `--reward`.

## Fixes made from the participant experience

- Added strict input/path validation to `tools/train_participant.py` and
  `tools/participant_ui.py`.
- Made the reference reward input optional for custom trainers while retaining
  the reference starter default.
- Made `racing_event` lazily import the optional PyTorch PPO module.
- Added bounded rollout collection through `RolloutCollector.max_batch_steps`.
- Made reward-callback failure state explicit and non-reusable.
- Passed clipped/applied actions to participant reward callbacks.
- Fixed UI process lifecycle state, Stop-button behavior, and termination escalation.
- Enforced Python 3.11 when creating/checking the project environment.
- Added regression tests for the above behaviors.
- Updated the participant guide and README with the trainer CLI contract,
  cooperative budget limitation, clipping semantics, and memory guidance.

## Remaining gaps

These are intentionally not claimed as complete:

- A custom trainer must accept the documented forwarded CLI options. The
  launcher is now explicit about that contract, but it does not adapt arbitrary
  no-argument scripts.
- The local launcher is cooperative. A custom trainer can ignore budget values
  unless it uses `RunBudget`/`RolloutCollector`; independent event-grade budget
  and resource enforcement belongs in the organizer evaluator.
- The reference replay button still supports only the reference `.pt` artifact.
  Generic participant artifact export/replay and the final ONNX/runtime
  validator remain future work.
- Crash recovery around the reference trainer's multi-file checkpoint update is
  still conservative rather than transactional. The persisted budget correctly
  charges reservations, but final event-grade checkpoint journaling is pending.
- No real browser screenshot was run because the container has no browser binary.

## Verification after fixes

- `make test` — **94 Python tests passed**.
- `node --check racing_event/assets/replay.js` — **passed**.
- `node --test tests/replay_frontend.test.cjs` — **65 tests passed**.
- `python3 -m unittest tests.test_rollout tests.test_participant_ui` — **17
  environment-only tests passed without Torch**.
- Python compilation for package, tools, examples, and tests — **passed**.
- `git diff --check` — **passed**.
- Live HTTP probes for fractional, infinite, and missing-trainer start requests
  returned structured HTTP 400 responses and spawned no trainer.
