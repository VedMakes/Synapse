# Participant guide draft

This is the first participant-facing training path for the local prototype. It
is intentionally marked as a draft. The exact event budget, final map set,
submission format, ranking rule, and architecture policy are still open in the
event decision register.

## Recommended delivery

The event should publish one immutable participant release, not a moving branch.
The release should contain:

- the inspectable Python simulator and fixed trainer;
- public training maps and their manifest;
- the observation/action contract;
- the reward configuration template;
- a local smoke test and deterministic evaluation helper;
- a versioned wheelhouse for the supported Python 3.11 platform; and
- an optional Docker path for participants who want it.

The ordinary leaderboard should later accept a model artifact and manifest. It
should not execute an arbitrary participant checkout as its scoring authority.
Held-out maps and final scoring must stay in an organizer-owned evaluator.

## First local run

From the repository root, with the project virtual environment installed:

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --steps 2048 \
  --budget-steps 10000 \
  --output-dir participant_run
```

The command loads all public JSON maps in `tracks/train/`, trains the fixed
two-layer PPO policy in bounded rollout chunks, and prints a small progress
line after each update. It writes:

```text
participant_run/
  budget.json              # persistent local transition counter
  training_state.json      # release/config/reward/map identity
  training_metrics.jsonl   # one bounded-size record per update
  training_report.md       # human-readable local dashboard
  policy.pt                # safe state dictionary for the prototype
  policy.json              # architecture metadata
  training_contract.json   # schema, map/config hashes, normalizer
```

Run the same command again with the same `--budget-steps` and output directory
to continue the local counter. The budget is charged per active environment
lane, not per vector call. The final partial batch is supported. Reset and PPO
optimizer updates are not environment transitions.

This prototype persists the policy and normalizer but does not persist the
PyTorch optimizer state. A process restart therefore resumes from the saved
policy with a fresh optimizer. This is acceptable for the onboarding slice but
must be decided before the scored event.

## Participant contract

The current provisional policy receives 23 `float32` track-relative values and
returns two finite actions in `[-1, 1]`:

1. steering;
2. throttle/brake, where positive is throttle and negative is braking.

The fixed prototype model is two hidden `tanh` layers of 64 units each. The
participant-facing free variable is the declarative reward weights in the
published signal set. Physics, observations, action semantics, PPO settings,
termination, tracks, and budget accounting are not participant-editable.

Reward configuration is JSON in this prototype. Non-finite values such as NaN
and Infinity are rejected before training.

## Budget and fairness draft

The recommended event rule is one cumulative transition budget across the full
training window. A transition is one accepted action applied to one active
environment lane. Vector lanes and repeated simulator transitions count
individually. Resets and optimizer updates do not count. The local budget file
is tamper-evident but is not independent proof of training provenance because a
participant controls the workspace.

The `10,000` value in the example is an onboarding smoke budget only. It is not
the proposed final scored budget. Research recommends measuring candidate
budgets such as 100k, 250k, 500k, and 1M with equal settings and multiple seeds
before freezing one.

## Maps

The current three maps are original synthetic development fixtures. The
manifest records both the canonical geometry hash and the delivered-file hash.
Before public release, the organizers must still:

- select the final public route families;
- prepare organizer-only development and final holdouts;
- run difficulty-parity and generalization experiments;
- red-team repository history, artifacts, previews, and Unity assets for leaks;
- record provenance, licenses, and contributor rights; and
- freeze the public release ID before the participant training window.

Do not place held-out geometry, seeds, previews, or derived Unity assets in the
participant repository, its history, release assets, or public replays.

## Open event decisions

The following remain discussion items rather than hidden implementation choices:

- cumulative budget enforcement model: local honor system, signed launcher, or
  organizer retraining;
- exact scored transition budget;
- final public/private map counts and identities;
- fixed architecture versus a bounded architecture division;
- ONNX operator/runtime contract and submission validator;
- reward-edit and checkpoint rules;
- deterministic evaluation starts, retries, DNF scoring, and ranking; and
- supported native platforms and offline wheelhouse contents.
