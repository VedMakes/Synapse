# E2 deterministic evaluation

This report records the local E2 diagnosis. It is evidence for the current
synthetic fixture, not a public step-budget or event-performance claim.

## Workload

- Track: `synthetic_oval`
- Seed: `17`
- Environments: `4`
- Rollout length: `64` steps per environment
- Updates: `64`
- Transitions: `16,384`
- Tick rate: `30 Hz`
- PPO update: `4` epochs, learning rate `1e-4`, entropy coefficient `0.005`
- Evaluation: one lane, fixed start line, standing start, deterministic policy
  mean action, frozen normalizer
- Repeated evaluation: two runs, same saved artifact

The run was kept headless and used one evaluation lane. The evaluator streamed
the trajectory digest and did not retain trajectory frames.

## Diagnosis

The pre-fix policy learned a low-speed escape strategy. At the fixed standing
start it applied little or no throttle, reached the negligible-progress limit,
and avoided the larger crash penalty. The synthetic track is completable: the
existing scripted controller completed a lap in `863` steps (`28.7667 s`).

The first engine fix made the failure visible by adding a one-shot `stuck`
signal with a baseline weight of `-8.0`. A standing-start spawn mixture was also
added so training includes the evaluation start distribution. The policy still
collapsed without an initial action prior, so the fixed PPO model now starts
with a small longitudinal-action mean bias of `0.5`. This is part of the saved
model metadata and is not a participant-editable setting.

## Results

| Artifact | Engine/trainer state | Status | Steps | Lap time (s) | Mean speed (m/s) | Final distance (m) | Crash | Trajectory hash |
|---|---|---:|---:|---:|---:|---:|---|---|
| Previous E2 artifact | Before stuck signal, spawn mixture, and throttle prior | `dnf_stuck` | 90 | n/a | 0.000 | 0.000 | false | `bebb222f89a1968085b8eafb588d2b69e2778dd19f47ead66dc635a9f5c49d14` |
| Fixed engine, no throttle prior | Stuck signal, standing-start mixture, combined grip | `dnf_stuck` | 90 | n/a | 0.000 | 0.000 | false | `473308e656ca3959d4c63324f33ddd01ca7ba0296e5ca84cd93a33173ee160dc` |
| Fixed engine and PPO prior | All fixes, 32 updates, 10 PPO epochs | `finished` | 1,522 | 50.7333 | 11.078 | 558.301 | false | `2625d58438cc97a506ca1838ccde4f6f6464c11dea8125e08aadd8cf6020ff5d` |
| Unstable PPO ablation | All fixes, 64 updates, 10 PPO epochs | `dnf_stuck` | 186 | n/a | 0.651 | 3.697 | false | `595ecc8fdfb7ed3bc0252ea5dd9bde1c86231f106ba598a7f3e9aaf2f5f75d0e` |
| Stabilized PPO artifact | All fixes, 64 updates, 4 epochs, `lr=1e-4`, `entropy=0.005` | `finished` | 923 | 30.7667 | 18.177 | 558.400 | false | `eac69718eeed370c7d23a393a6d4ce65f838704b8665408d634e160be909ee13` |

The final row was produced by two deterministic evaluations. Both runs returned
the same status, measurements, and trajectory hash. It recorded `204` off-track
steps, so the policy is not yet a clean racing line even though it completes.

The unstable 64-update run ended with approximate KL `0.0267` and clip fraction
`0.3164`. The stabilized run ended with approximate KL `0.000062` and clip
fraction `0.0`. The lower update size and learning rate are therefore adopted
as the new provisional trainer defaults. They are not final event settings.

## Interpretation boundary

The throttle prior removes the immediate zero-speed local failure and produces a
valid fixed-start lap on the fixture. The stabilized PPO settings remain
competent after twice the training transitions and reduce update drift. The lap
is only slightly slower than the scripted controller, but it uses the runoff
often, so this is not convergence or racing-line quality. Stochastic rollout
metrics in the 16,384-transition window still showed frequent crashes and no
completed episode in the final rollout window. More training and multi-track
evaluation are still required.

The physics change is intentionally small: longitudinal acceleration now shares
a friction-circle budget with lateral grip, and scalar and batched transitions
are regression-tested for parity. No renderer or Unity logic was added.

An ablation that increased the baseline `off_track` weight from `-1.5` to
`-4.0` was not adopted: its fixed-start policy crashed after `272` steps with
`146` off-track steps. The reward change was too aggressive for this short
training window.

## Reproduction

```bash
ARTIFACT_DIR=/tmp/racing-e2-artifact

.venv/bin/python tools/run_e2_baseline.py \
  --updates 64 --n-envs 4 --rollout-length 64 --seed 17 \
  --artifact-dir "$ARTIFACT_DIR" \
  --output /tmp/e2-artifact.md

.venv/bin/python tools/evaluate_policy.py \
  --weights "$ARTIFACT_DIR/policy.pt" \
  --metadata "$ARTIFACT_DIR/policy.json" \
  --contract "$ARTIFACT_DIR/training_contract.json" \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --repeats 2 \
  --output /tmp/e2-deterministic-evaluation.json
```
