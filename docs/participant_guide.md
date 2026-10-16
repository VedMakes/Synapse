# SYNAPSE participant guide draft

This is the first participant-facing training path for the local prototype. It
is intentionally marked as a draft. The exact event budget, final map set,
submission format, ranking rule, and artifact validator are still open in the
event decision register.

## Recommended delivery

The event should publish one immutable participant release, not a moving branch.
The release should contain:

- the inspectable Python simulator and an optional reference trainer;
- public training maps and their manifest;
- the observation/action contract;
- raw transition signals and a participant-owned reward/training boundary;
- a local smoke test and deterministic evaluation helper;
- a versioned wheelhouse for the supported Python 3.11 platform; and
- an optional Docker path for participants who want it.

The ordinary leaderboard should later accept a model artifact and manifest. It
should not execute an arbitrary participant checkout as its scoring authority.
Held-out maps and final scoring must stay in an organizer-owned evaluator.

## First local run

From the repository root, prepare the project environment first:

```bash
python3 tools/setup_participant.py
```

Then launch the trainer you own:

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python tools/train_participant.py \
  --trainer examples/participant_train.py \
  --reward examples/baseline_reward.json \
  --steps 2048 \
  --budget-steps 10000 \
  --output-dir participant_run
```

The launcher validates workspace paths and forwards the published inputs to
your trainer. The bundled `examples/participant_train.py` delegates to the
optional reference PPO only so the checkout has a runnable smoke test. Replace
that file with your own training program to own the model architecture,
algorithm, preprocessing, checkpoints, and reward function. A trainer that
supports the dashboard should print the documented `update ...` progress line.
The reference starter writes:

```text
participant_run/
  budget.json              # persistent local transition counter
  training_state.json      # release/config/reward/map identity
  training_metrics.jsonl   # one bounded-size record per update
  training_report.md       # human-readable local dashboard
  policy.pt                # reference starter artifact only
  policy.json              # reference architecture metadata
  training_contract.json   # environment schema and normalizer for the starter
```

Run the same command again with the same `--budget-steps` and output directory
to continue the local counter. The budget is charged per active environment
lane, not per vector call. The final partial batch is supported. Reset and PPO
optimizer updates are not environment transitions.

This prototype persists the policy and normalizer but does not persist the
PyTorch optimizer state. A process restart therefore resumes from the saved
policy with a fresh optimizer. This is acceptable for the onboarding slice but
must be decided before the scored event.

## Optional local dashboard

The same workflow has a small dependency-free Python web UI:

```bash
.venv/bin/python tools/participant_ui.py
```

Open `http://127.0.0.1:8766/`. The dashboard provides editable trainer/reward
paths and bounded-run settings, start/resume and stop controls, budget
progress, update metrics, a live log, and the SYNAPSE environment contract. It
launches the same `tools/train_participant.py` command. It binds to loopback
only and is not a network service or a second model, reward function, or
simulator.

The dashboard also has a **Set up / repair Python** button. If the UI detects
that NumPy is missing, it refuses to launch a broken environment and shows the
setup command instead. The bundled reference starter has a **Watch reference
artifact for map** action; a generic participant artifact replay/export
contract is not implemented yet.

## Participant-owned training boundary

The SYNAPSE environment publishes 23 `float32` track-relative values and
accepts two finite actions in `[-1, 1]`:

1. steering;
2. throttle/brake, where positive is throttle and negative is braking.

Participants own the policy/model architecture, inference distribution,
optimizer, training algorithm, preprocessing, checkpoint format, weights, and
reward function. SYNAPSE owns the physics transition, observation/action
semantics, termination, public tracks, raw signal names, evaluation starts,
and scoring environment.

`VectorizedRacingEnv.step_raw` returns the next observation, termination flags,
and raw signals. A participant reward callback may use those signals, the
observation, action, and transition outcome to return one finite scalar.
`examples/participant_reward.py` shows the callable shape only; copy and
replace it in your own trainer workspace.
`RewardConfig` and `examples/baseline_reward.json` are only a reference
weighted-signal adapter. Non-finite reward values are rejected before the
training batch is created.

A participant-owned trainer can use the reusable transition adapter without
accepting its reference PPO implementation:

```python
env = VectorizedRacingEnv(tracks, config, reward=None, n_envs=your_batch_size)
collector = RolloutCollector(
    env,
    your_policy,
    budget,
    run_id,
    reward_function=your_reward,
)
batch = collector.collect(your_transition_count).batch
# Update your own model/optimizer/algorithm from `batch`.
```

The callback receives copies of the observation, action, raw signals, and
terminal flags. It cannot replace or mutate the environment transition.

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
- model artifact format, operator/runtime contract, and submission validator;
- reward-code packaging, checkpoint, and budget-enforcement rules;
- deterministic evaluation starts, retries, DNF scoring, and ranking; and
- supported native platforms and offline wheelhouse contents.
