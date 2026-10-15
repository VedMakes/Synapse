# Versioned training state schema

The Python engine is authoritative. Unity and a future trainer consume this
schema; they do not recompute physics, progress, rewards, or termination.

## Contract versions

- Training contract: `1.0`
- Action schema: `1.0`
- Observation schema: `1.0`
- Observation dtype: `float32`
- Observation normalization: static, environment-side bounds. A trainer may
  add a running mean/std transform only when it saves and restores the state
  in the training artifact.

## Action

The action has shape `(2,)` for the scalar environment and `(n_envs, 2)` for
the batched environment:

| Index | Name | Bounds | Meaning |
|---:|---|---|---|
| 0 | `steering` | `[-1, 1]` | Normalized steering command, subject to the configured steering-rate limit. |
| 1 | `throttle_brake` | `[-1, 1]` | Positive throttle, zero coast, negative brake. |

Finite values are required. NaN and infinity are rejected before clipping to
the interval.

## Observation

The default configuration emits 23 values in this exact order:

| Index | Name | Bounds | Normalization |
|---:|---|---|---|
| 0 | `speed` | `[0, 1]` | Forward speed divided by maximum speed. |
| 1 | `longitudinal_velocity` | `[-1, 1]` | Signed forward velocity divided by maximum speed. Reverse is disabled in v1, so it equals `speed` in the current model and is explicitly named rather than left as an unnamed duplicate. |
| 2 | `lateral_velocity` | `[-1, 1]` | Car-frame lateral velocity divided by maximum speed. |
| 3 | `yaw_rate` | `[-1, 1]` | Yaw rate clipped to five radians per second. |
| 4 | `lateral_offset` | `[-1, 1]` | Signed centerline offset divided by twice the local maximum half-width. |
| 5 | `heading_error` | `[-1, 1]` | Wrapped vehicle/track heading error divided by pi. |
| 6 | `previous_steering` | `[-1, 1]` | Previous steering action. |
| 7 | `previous_longitudinal` | `[-1, 1]` | Previous signed throttle/brake action. |
| 8 | `steering_angle` | `[-1, 1]` | Current steering angle divided by configured maximum steering angle. |
| 9-17 | `ray_0` ... `ray_8` | `[0, 1]` | Boundary distance divided by ray maximum range. The configured ray order is `0, -15, 15, -30, 30, -60, 60, -90, 90` degrees. |
| 18-21 | `curvature_lookahead_0` ... `curvature_lookahead_3` | `[-1, 1]` | Signed curvature multiplied by 100 and clipped, at 15, 40, 80, and 140 metres. |
| 22 | `local_width` | `[0, 1]` | Minimum local half-width divided by five times car width. |

The observation contains no absolute position, absolute heading, track id, or
absolute lap progress.

## Evaluation artifact

`racing_event.observation.TrainingContract` serializes the versions, action
schema, observation fields, track ids/hashes, simulation-config hash,
deterministic evaluation policy mode, fixed-start spawn rule, and optional
`RunningObservationNormalizer` statistics. Evaluation loads this artifact,
uses the policy mean action, and does not update normalization statistics.

## Fixed PPO model artifact

The first trainer slice uses `racing_event.ppo.FixedMLPActorCritic`:

- two hidden layers of 64 units with `tanh` activation;
- a two-action tanh-squashed Gaussian policy;
- a scalar value head; and
- a learned global log standard deviation with two entries.
- The initial policy-mean bias is `[0.0, 0.5]` for steering and
  throttle/brake. This gives a new policy a small throttle prior so standing
  starts do not collapse into the zero-speed stuck termination. The value is
  stored in the model metadata.

`FixedMLPActorCritic.save_artifact` writes a state dictionary to `policy.pt`
and architecture metadata to a separate JSON file. The loader uses
`torch.load(..., weights_only=True)` and rejects schema, distribution,
architecture, tensor-key, tensor-shape, and non-finite-value mismatches. The
model artifact is separate from the `TrainingContract`, which stores the
frozen observation normalizer and environment hashes.

`PPOTrainer.update` consumes only active rollout transitions. It distinguishes
terminated transitions from truncated transitions for value bootstrapping and
never carries GAE across a lane reset. Its output records policy loss, value
loss, entropy, approximate KL, clip fraction, explained variance, and update
throughput.

## Deterministic evaluation

`racing_event.evaluation.evaluate_policy` requires a one-lane vector
environment and calls `reset(evaluation=True)`. Evaluation owns the start line,
standing start, and zero spawn noise. Supplying custom spawn parameters in
evaluation mode is rejected. A loaded normalizer is called with
`training=False`, so its statistics cannot change.

The evaluator reports a fixed-step lap time (`steps * dt`) only for a finished
lap. For a DNF it reports the reason separately. It keeps running counters and
updates a SHA-256 trajectory digest per transition instead of storing frames.
The CLI in `tools/evaluate_policy.py` also checks the saved observation,
configuration, and track hashes before stepping the engine.
