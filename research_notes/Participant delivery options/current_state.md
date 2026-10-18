# Current SYNAPSE participant-delivery audit

**Audit date:** 2026-10-17

**Implementation update, 2026-10-19:** this original audit is historical. Local
participant UI, exact reference-generation resume/fine-tuning/best retention,
bounded replay groups, and the candidate ONNX/model-only worker now exist.
See `reports/release_readiness_2026-10-19.md` for actual checks and remaining
gates, and `docs/submission_runtime.md` for the proposed engineering profile.
This does not freeze maps, ranking, budget, submission rules, or release policy;
Docker build/isolation and clean offline delivery remain unverified.

## Verdict

The repository is a strong Python engine prototype, but it is not yet a final
participant release. The simulator, reward signals, persistent budget primitive,
PPO update, deterministic evaluation, replay tooling, public maps, and pinned
runtime foundations exist. The participant-facing orchestration, final map set,
submission validator, ONNX runtime contract, private evaluator, and release
boundary remain unfinished.

## What participants can do in the current prototype

- Install the Python 3.11 environment from the lockfile.
- Run the 77+ test suite and the deterministic scripted smoke test.
- Inspect three original synthetic public training maps.
- Use `RacingEnv` or `VectorizedRacingEnv`, the raw transition/signal boundary,
  participant-owned reward callback, `RolloutCollector`, and deterministic
  evaluator APIs. The fixed PPO path is only an optional reference starter.
- Build the public map manifest with `tools/build_training_manifest.py`.
- Run the provisional participant workflow with `tools/train_participant.py`.
- Inspect the bounded terminal progress output and
  `participant_run/training_report.md`.

## Implemented in this slice

- The participant launcher forwards public maps and environment inputs to an
  arbitrary participant-owned trainer instead of constructing a model itself.
- A local one-run persistent budget is resumed across invocations and charges
  active vector lanes through the existing `RunBudget`/`RolloutCollector`
  boundary.
- The optional reference starter saves policy, training normalizer, contract,
  state, JSONL metrics, and a small Markdown report after each bounded PPO
  update; participant trainers may use any artifact and algorithm.
- The runner refuses a changed reward/config/map identity or a missing policy
  after budget usage. It does not accumulate the full budget in memory.
- Raw participant reward callbacks receive copied observations/actions/signals
  and reject non-finite scalar outputs before a training batch is created.
- `tracks/train/manifest.json` records public map IDs, canonical content hashes,
  delivered-byte hashes, units, and validator identity.

## Still missing before delivery-ready release

1. A clean public participant snapshot/repository and immutable release ID.
2. A tested offline wheelhouse with hashes and a supported platform matrix.
3. A final public map set plus organizer-only development/final holdouts.
4. Route-family split, difficulty parity, map-count, and generalization tests.
5. A frozen ONNX export/operator/runtime contract and safe submission validator.
6. A private evaluator, ranking/DNF policy, resource limits, and appeal runbook.
7. A complete optimizer/checkpoint persistence policy for resumable training.
8. License, third-party notice, contributor-rights, and map-provenance records.
9. A fresh-machine offline setup rehearsal and participant support loop.

## Major decisions deliberately not frozen

- Full public repository versus curated participant release boundary.
- Exact cumulative scored transition budget and enforcement strength.
- Participant artifact format/operator validator and resource limits.
- Reward-code packaging/checkpoint rules and whether local training is honor-system,
  launcher-anchored, or organizer-retrained.
- ONNX submission details, held-out track count, deterministic starts/retries,
  ranking, DNF penalties, supported platforms, and artifact rights.

## Evidence and memory notes

- The local E1 report measured roughly 69–72 environment transitions per second
  and about 14 ms of geometry/reward/observation overhead per transition. This
  makes wall-clock duration a bigger participant risk than policy memory.
- Rollouts are retained only for one bounded update. The participant runner's
  smoke path uses four lanes and 64 steps per lane, not a whole budget in RAM.
- Evaluation already streams a trajectory digest one lane at a time.
- The local hash chain detects accidental edits but is not independent proof of
  participant training provenance.

## Source documents

- `reports/Participant delivery options.md`
- `research_notes/Participant delivery options/packaging.md`
- `research_notes/Participant delivery options/fairness.md`
- `research_notes/Participant delivery options/maps.md`
- `RACING_EVENT_Agent_Handoff.md` Sections 4, 7, 8, 9, 12, 13, and 27
