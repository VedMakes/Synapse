# Racing RL controls and training contract

Research scope: state-vector racing environments, steering/longitudinal action representations, PPO/SAC training practice, deterministic evaluation, normalization, action bounds, throughput, and reuse terms. Sources were checked on 2026-10-14. No source code is copied here; action shapes and design facts are paraphrased.

## Existing environments, trainers, and licensed repositories

### Takeaway

The strongest directly relevant pattern is a compact, track-relative vector observation with continuous control. Public examples cover both a two-output steering plus throttle representation and a three-output steering plus separate gas/brake representation. The best evidence for this project’s fairness requirements is a single fixed-step simulator used for human control, PPO training, replay, and evaluation, with deterministic evaluation separated from stochastic training.

### Cited Findings

- **Gymnasium CarRacing-v3:** the default continuous action space is a 3-vector with bounds `[-1, 0, 0]` to `[1, 1, 1]`, representing steering, gas, and braking; the discrete alternative has five actions: no-op, steer right, steer left, gas, and brake. Its observation is a generated 96x96x3 RGB image, not a compact state vector. The reward is a per-frame cost plus progress for visited track tiles, with a large off-playfield failure penalty. The car starts at rest in the road center, and a generated track is random each episode. [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) [CarRacing source](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/box2d/car_racing.py)

- **Gymnasium license:** Gymnasium is MIT-licensed. The license permits reuse, modification, and distribution if the copyright notice and license are retained. [Gymnasium LICENSE](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)

- **Track Racer:** this current public PPO example uses a custom Gymnasium environment with 9 ray sensors, bicycle-style car physics, and an 11-value observation consisting of normalized distance-to-center, normalized heading error, and ray distances. Its action has two continuous values, steering and throttle, both in `[-1, 1]`. The README describes progress, speed, heading-alignment, off-track, lap, and time reward terms, and gives a PPO setup using `n_steps=2048`, `gamma=0.99`, `gae_lambda=0.95`, 10 update epochs, clip `0.2`, and entropy coefficient `0.01`. [Track Racer repository](https://github.com/nikhil-ghind/track_racer)

- **Track Racer license status:** GitHub’s repository API reports `license: null`; no repository license was verified. Treat it as a design reference only and do not copy its code. [Track Racer API metadata](https://api.github.com/repos/nikhil-ghind/track_racer)

- **TMRL / TrackMania:** TMRL exposes an analog action convention written as `[gas, brake, steer]`, with values described as analog from `-1.0` to `+1.0`. Its LIDAR environment includes speed, a history of four 19-beam LIDAR observations, and previous-action history; the project says an MLP is used for LIDAR observations. The example pipeline trains SAC by default and also supports REDQ-SAC, while the real-time environment runs at 20 Hz by default. TMRL reports that a single LIDAR observation led to constant-speed behavior on a sharp turn, while a four-frame LIDAR history enabled braking, apexing, and acceleration afterward. [TMRL README](https://github.com/trackmania-rl/tmrl) [TMRL documentation](https://tmrl.readthedocs.io/en/latest/)

- **TMRL training architecture:** TMRL is built for real-time applications and can collect samples through multiple rollout workers connected to a central server and trainer. This is a throughput pattern for slow, externally controlled simulators; it is not needed for the current headless NumPy simulator until local vectorization is measured. [TMRL README, remote training architecture](https://github.com/trackmania-rl/tmrl#remote-training-architecture)

- **TMRL license:** the repository is MIT-licensed. [TMRL LICENSE](https://github.com/trackmania-rl/tmrl/blob/master/LICENSE)

- **Ghostline2DRacerRL:** this current public project explicitly uses one deterministic 2D physics core for keyboard play, PPO training, exact replay, and evaluation. Its README specifies a fixed 120 Hz simulator, 60 Hz control updates, and rendering interpolation between states; given the same initial state and action stream, it says the run is exactly reproducible. The project reports lookahead and wall-sensor observations as the change that closed its final corner-speed gap, and it records failed reward designs and regressions in an experiment log. [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL) [Ghostline experiment log](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL/blob/main/docs/RL_EXPERIMENT_LOG.md)

- **Ghostline license:** the repository is MIT-licensed. [Ghostline LICENSE](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL/blob/main/LICENSE)

- **QDrive:** this current MIT-licensed project provides SB3 and SB3-Contrib training/evaluation orchestration, ordinary vectorized Gymnasium environments, checkpointing, deterministic evaluation, and saved `VecNormalize` statistics. Its TrackMania adapter is single-instance and real-time, so it forces one environment and `DummyVecEnv` rather than ordinary parallel vectorization. This is useful evidence that normalization state and environment-specific vectorization settings belong in the saved experiment contract. [QDrive README](https://github.com/Salzaufsauge/QDrive) [QDrive LICENSE](https://github.com/Salzaufsauge/QDrive/blob/main/LICENSE)

- **Box2D Racing:** this public custom environment uses raycast sensing, Gymnasium, Stable-Baselines3, and SAC/PPO evaluation scripts. Its README explicitly lists multiprocessing vectorization as unfinished and time/step count in observations as unfinished. That is a useful warning against assuming that a visually working racing environment is already ready for throughput or time-limit-correct PPO training. [Box2D Racing README](https://github.com/roboserg/box2d-racing)

- **Box2D Racing license status:** GitHub’s repository API reports `license: null`; no repository license was verified. Use for ideas only. [Box2D Racing API metadata](https://api.github.com/repos/roboserg/box2d-racing)

- **FSAE RL Racing:** this public Isaac Sim thesis repository includes a single-car PPO baseline, a two-car self-play lineage, a pure-pursuit baseline, curvature-aware and racing-line variants, and an unseen-track evaluation. Its README emphasizes that training reward was not a reliable proxy: some high rewards were simulator artifacts, and results were re-measured with lap time, smoothness, crashes, and other physical metrics. This supports logging outcome metrics separately from reward. [FSAE RL Racing README](https://github.com/mk217alt/fsae_rl_racing)

- **FSAE RL Racing license status:** GitHub’s repository API reports `license: null`; no repository license was verified. Do not copy code or assets. [FSAE RL Racing API metadata](https://api.github.com/repos/mk217alt/fsae_rl_racing)

- **AWS DeepRacer on AWS:** the current public AWS solution describes training and evaluation in a 3D simulated autonomous-driving environment, customizable reward functions, virtual racing and leaderboards, model export to physical cars, multi-track events, and SageMaker-based training workflows. The public implementation guide does not expose a compact state-vector action contract; it is useful here as evidence for reward customization, separate evaluation, artifact handling, and multi-simulator orchestration rather than as a direct control-space precedent. [AWS DeepRacer on AWS implementation guide](https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/) [AWS DeepRacer product page](https://aws.amazon.com/deepracer/)

- **AWS DeepRacer source terms:** the current `aws-solutions/deepracer-on-aws` repository is Apache-2.0 and includes a `NOTICE.txt`; Apache-2.0 requires retaining the license and relevant notices and does not grant trademark rights. The community `deepracer-simapp` repository, which says its bundle closely matches the AWS simulator/training code, is also Apache-2.0. [DeepRacer on AWS repository](https://github.com/aws-solutions/deepracer-on-aws) [DeepRacer on AWS LICENSE.txt](https://github.com/aws-solutions/deepracer-on-aws/blob/main/LICENSE.txt) [DeepRacer Simapp repository](https://github.com/aws-deepracer-community/deepracer-simapp) [DeepRacer Simapp LICENSE](https://github.com/aws-deepracer-community/deepracer-simapp/blob/master/LICENSE)

- **DeepRacer-for-Cloud terms and throughput:** the community `deepracer-for-cloud` repository is labeled MIT-0 and describes multiple simulator instances per training job, parallel sessions, multi-node swarm mode, independent evaluation, trace logging, and CPU/GPU container options. This is relevant as an orchestration precedent, but its Docker/Gazebo/RL Coach stack is much heavier than the current headless state-vector project. [DeepRacer-for-Cloud repository](https://github.com/aws-deepracer-community/deepracer-for-cloud) [DeepRacer-for-Cloud LICENSE](https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/master/LICENSE)

- **Trainer licenses:** Stable-Baselines3 is MIT-licensed, and CleanRL is MIT-licensed for its main project with additional notices for adapted files. These are viable trainer references, but a copied implementation still requires preserving the applicable notices and checking each file’s provenance. [Stable-Baselines3 LICENSE](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) [CleanRL LICENSE](https://github.com/vwxyzjn/cleanrl/blob/master/LICENSE)

### Inferences

- For this project, the most reusable pattern is the Ghostline-style single authoritative fixed-step simulator combined with Track Racer’s compact ray/state vector. This fits the existing Python-authoritative/Unity-renderer split and avoids training on pixels when the event explicitly requires state vectors. [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL) [Track Racer repository](https://github.com/nikhil-ghind/track_racer)

- The project should keep a source-and-license ledger and should not use unlicensed hobby repositories as code dependencies. MIT and Apache-2.0 sources are referenceable with their notices; repositories whose GitHub metadata reports no license remain read-only references. [QDrive LICENSE](https://github.com/Salzaufsauge/QDrive/blob/main/LICENSE) [DeepRacer on AWS LICENSE.txt](https://github.com/aws-solutions/deepracer-on-aws/blob/main/LICENSE.txt) [Track Racer API metadata](https://api.github.com/repos/nikhil-ghind/track_racer)

### Gaps

- I did not find a current official AWS page that states the original managed DeepRacer simulator’s exact internal action-vector bounds and policy distribution. The current AWS public material documents the product and self-hosted architecture, while the public community simulator exposes implementation material under Apache-2.0. Do not claim that the AWS action space is identical to Gymnasium CarRacing or TMRL. [AWS DeepRacer on AWS implementation guide](https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/) [DeepRacer Simapp repository](https://github.com/aws-deepracer-community/deepracer-simapp)

## Control, normalization, PPO, and throughput comparison

### Takeaway

For continuous PPO, a symmetric normalized action space and bounded, consistently scaled observations remove avoidable optimization problems. PPO evaluation should use a separate deterministic environment path and the policy mean action. Running observation statistics can work, but they are part of the model artifact and must be frozen at evaluation. Action clipping and action squashing are different: stock Gaussian PPO commonly samples unbounded actions and clips them at the environment boundary, whereas a tanh policy bounds actions during sampling and must apply the corresponding log-probability correction.

### Cited Findings

- **Two outputs: steering plus signed longitudinal control.** The current project already implements two values in `RacingEnv.step`: steering and throttle/brake, each clipped to `[-1, 1]`; positive longitudinal control is throttle and negative longitudinal control is braking. The physics then derives throttle and brake from the sign, so one action cannot request both at once. [Current `RacingEnv`](../../racing_event/env.py) [Current physics](../../racing_event/physics.py)

- **Three outputs: steering plus separate gas and brake.** Gymnasium’s CarRacing continuous API exposes independent steering, gas, and braking components, with one-sided gas/brake bounds. This increases direct control but leaves the environment responsible for defining what simultaneous gas and brake mean. [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) [CarRacing source](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/box2d/car_racing.py)

- **Three outputs with all-signed analog components:** TMRL documents `[gas, brake, steer]` with each value described as analog in `[-1, 1]`. Its environment-specific convention should not be copied blindly into a physical model: a brake channel with negative values needs a documented mapping, unlike Gymnasium’s nonnegative gas/brake channels. [TMRL README, TrackMania Gymnasium environment](https://github.com/trackmania-rl/tmrl#trackmania-gymnasium-environment)

- **Discrete alternatives:** Gymnasium provides a 5-action discrete representation. A categorical PPO policy can use this representation, but it cannot express proportional steering, throttle, or braking at the API boundary without increasing the number of discrete actions. [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)

- **State-vector design:** Track Racer uses a small vector containing lateral/heading relation and ray distances, while TMRL uses LIDAR plus speed and temporal history. The current project uses self motion, track relation, previous controls, raycasts, curvature lookahead, and local width, all normalized or clipped. [Track Racer repository](https://github.com/nikhil-ghind/track_racer) [TMRL README](https://github.com/trackmania-rl/tmrl) [Current `RacingEnv` observation code](../../racing_event/env.py)

- **Normalization guidance:** SB3’s current guidance says to normalize inputs when possible, normalize continuous actions to a symmetric `[-1, 1]` interval, and rescale inside the environment. It calls normalization critical for continuous-action algorithms and recommends evaluating PPO with `deterministic=True`. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

- **Running statistics:** SB3’s `VecNormalize` is a moving-average wrapper for observations and rewards, supports saving/loading its statistics, and has an explicit `training` mode that controls whether statistics update. QDrive’s README confirms that its training/evaluation artifacts include `VecNormalize` statistics. [SB3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize) [QDrive README](https://github.com/Salzaufsauge/QDrive)

- **PPO stability mechanics:** SB3 documents PPO as a clipped policy-update method, with normalized advantages by default and optional value-function clipping. Its PPO API exposes rollout length, batch size, update epochs, GAE, clip range, entropy coefficient, gradient clipping, target KL, and gSDE settings. [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

- **Action clipping versus squashing:** SB3’s `ActorCriticPolicy` defaults `squash_output=False`; for a non-squashed continuous action it clips predicted actions to the Box bounds at `predict` time. The same source restricts `squash_output=True` to gSDE in this policy class. SB3’s distributions module includes a tanh-squashed Gaussian with a log-probability correction, but that is distinct from simply clipping an action after sampling. [SB3 policy source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/policies.py) [SB3 distribution source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/distributions.py) [SB3 RL tips on Gaussian action bounds](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#why-should-i-normalize-the-action-space)

- **CleanRL PPO reference:** the current continuous-action PPO example uses `ClipAction`, running observation normalization, clipped observations, running reward normalization, clipped rewards, normalized advantages, a 2048-step rollout, 10 update epochs, clip coefficient `0.2`, gradient norm clipping `0.5`, and a deterministic Torch setting. Its action distribution is Gaussian, so `ClipAction` is an explicit wrapper rather than a tanh-squashed policy. [CleanRL continuous PPO source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)

- **Deterministic evaluation:** SB3 recommends a separate test environment because training exploration is stochastic and says `deterministic=True` frequently improves evaluation performance. Ghostline supplies a stronger simulator-level precedent: fixed-step physics, fixed control cadence, identical action-stream replay, and evaluation through the same physics core. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm) [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL)

- **Vectorization and throughput:** SB3 says PPO is primarily CPU-oriented when using MLPs, recommends multiple environments, and documents `SubprocVecEnv` for computationally complex environments while warning that the number of environments should not exceed logical CPU cores for non-IO-bound work. QDrive uses vectorized Gymnasium environments for ordinary experiments but explicitly falls back to one environment for real-time TrackMania. [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) [SB3 vectorized environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) [QDrive README](https://github.com/Salzaufsauge/QDrive)

- **Trainer throughput patterns outside PPO:** TMRL uses distributed rollout workers because external real-time simulators are slow, while DeepRacer-for-Cloud uses multiple simulator instances, parallel sessions, and optional multi-node swarm mode. For this project the analogous first step is in-process batched NumPy physics, then multiprocessing only if profiling proves it necessary. [TMRL README](https://github.com/trackmania-rl/tmrl#remote-training-architecture) [DeepRacer-for-Cloud README](https://github.com/aws-deepracer-community/deepracer-for-cloud)

### Inferences

- A pre-normalized state vector is preferable as the public environment contract because this project already knows the physical bounds for speed, steering, ray distances, heading error, and width. A learned running normalizer may still be used by the fixed trainer, but its mean, variance, clip setting, and `training=False` evaluation mode must be serialized with the policy. This follows SB3’s normalization guidance and QDrive’s artifact practice. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) [SB3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize) [QDrive README](https://github.com/Salzaufsauge/QDrive)

- A tanh-squashed Gaussian is mathematically cleaner than post-sample clipping for PPO because the policy’s stored log probability corresponds to its bounded action. If the first trainer uses stock SB3 PPO, the project should explicitly document that it uses the default Gaussian-plus-clipping path, log action saturation, and run an ablation against a corrected tanh implementation before freezing stability claims. [SB3 policy source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/policies.py) [SB3 distribution source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/distributions.py) [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#why-should-i-normalize-the-action-space)

- For a kinematic model with one net longitudinal acceleration, separate gas and brake create redundant action combinations and a larger exploration volume. A signed longitudinal axis removes that redundancy, makes coasting explicit at zero, and gives PPO a symmetric action dimension. This is a design inference from the current sign-based physics and SB3’s symmetric-action guidance, not a claim that three pedals are wrong for a higher-fidelity simulator. [Current physics](../../racing_event/physics.py) [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment)

### Gaps

- The public repositories do not provide a controlled, same-track, same-seed ablation of two actions versus three actions. Any claim that one representation converges faster must be measured in this project with equal PPO seeds, step budget, rollout layout, reward, and physics.

- Existing public throughput numbers are not comparable: TMRL is real-time at 20 Hz, Ghostline reports fixed simulation/control rates, DeepRacer-for-Cloud uses containers and multiple simulator instances, and SB3 documents scaling patterns rather than a racing-specific steps-per-second number. The current project’s handoff also labels its throughput numbers as planning estimates, so E1 must replace them with a local benchmark. [TMRL README](https://github.com/trackmania-rl/tmrl) [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL) [DeepRacer-for-Cloud README](https://github.com/aws-deepracer-community/deepracer-for-cloud) [Current project handoff](../../RACING_EVENT_Agent_Handoff.md)

## Recommended next contract for this project

### Takeaway

Keep **two total continuous outputs**: steering plus one signed longitudinal control. Do not switch to three outputs for the next slice. The current action choice is aligned with PPO’s normalized symmetric-action guidance, avoids simultaneous gas/brake ambiguity in the present physics, and is sufficient for the project’s stated kinematic model. Freeze it only after the observation field names, normalization policy, action-distribution behavior, and deterministic evaluation path are written as a versioned contract and tested.

### Cited Findings

- **Public action contract:** `action = [steering, longitudinal]`, both `float32` values in `[-1, 1]`. Steering maps to the existing steering angle/rate limits. Longitudinal maps as `+1 = full throttle`, `0 = coast`, `-1 = full brake`; the environment may clip finite inputs to the interval, but it must reject NaN and infinite values. This preserves the project’s current semantics and follows SB3’s recommendation for symmetric continuous action spaces. [Current `RacingEnv`](../../racing_event/env.py) [Current physics](../../racing_event/physics.py) [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment)

- **Three-output change is deferred:** only introduce `[steering, throttle, brake]` if the physics gains an independently meaningful pedal model, traction/ABS behavior, reverse, clutch, or deliberate pedal overlap. If that happens, use explicit channel bounds and semantics, define the result of both pedals being nonzero, and benchmark it as a new action-contract version. Gymnasium demonstrates the separate-pedal precedent; TMRL demonstrates that another simulator may choose a different signed convention. [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) [TMRL README](https://github.com/trackmania-rl/tmrl#trackmania-gymnasium-environment)

- **Observation contract:** retain a fixed `float32` track-relative vector with no absolute position, absolute heading, track ID, or absolute lap progress. Keep self motion, signed lateral offset, heading error, previous controls, steering state, local ray distances, curvature lookahead, and local width. Each field needs a name, unit, bound, clipping rule, and index in a versioned schema. The current project already implements this shape conceptually and returns a default 23-value vector. [Current `RacingEnv` observation code](../../racing_event/env.py) [Project first-slice contract](../../docs/first_slice.md) [Track Racer repository](https://github.com/nikhil-ghind/track_racer)

- **Observation cleanup before freezing:** the current observation list contains the speed value twice. Keep the slot only if the second field is renamed and changed to the intended longitudinal velocity; otherwise remove it. Do not freeze a duplicated field under two different names. [Current `RacingEnv` observation code](../../racing_event/env.py)

- **Normalization contract:** make environment-level physical normalization authoritative and deterministic: bounded quantities such as speed, steering command, steering angle, heading error, ray distances, curvature features, and width should be emitted in documented bounded ranges. If the fixed trainer adds running mean/std normalization, save the statistics and clip settings in the submission, load them for evaluation, and disable updates during evaluation. Avoid stacking an undocumented running normalizer on top of a second normalization layer. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) [SB3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize) [QDrive README](https://github.com/Salzaufsauge/QDrive)

- **PPO action distribution:** define one trainer behavior and log it. Preferred implementation: a tanh-bounded Gaussian policy whose log probability includes the tanh correction. Acceptable first baseline: stock PPO’s Gaussian policy with explicit action clipping, with action saturation rate logged and treated as a known baseline behavior. Do not compare these two approaches without an ablation because clipping after sampling and tanh squashing have different policy distributions. [SB3 policy source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/policies.py) [SB3 distribution source](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/distributions.py) [CleanRL continuous PPO source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)

- **Evaluation protocol:** use a separate evaluation environment with fixed start line, standing start, zero spawn noise, fixed physics/config/track hashes, and deterministic policy mean action. Store the observation-normalization artifact and trainer version with the weights. For each evaluation run, record the seed, action distribution mode, trajectory hash, lap time, completion/DNF status, crash, off-track, and step count. This follows SB3’s separate deterministic evaluation guidance and Ghostline’s same-simulator replay model. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm) [SB3 PPO `predict` documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL)

- **Training throughput:** implement batched NumPy stepping before adding multiprocessing. Benchmark environment-only steps/s, policy-plus-environment steps/s, sensor/projection cost, and PPO update time across `n_envs` values. Use headless execution and no Unity/Pygame in the hot path. Use logical-core-aware parallelism only after profiling; SB3’s guidance supports CPU PPO and vectorized environments, while real-time projects show that one environment may be correct when the simulator itself is externally rate-limited. [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) [SB3 vectorized environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) [TMRL README](https://github.com/trackmania-rl/tmrl) [Project README](../../README.md)

- **Trainer metrics required for stability:** log deterministic evaluation return/lap time separately from training return; per-signal reward terms; lap completion rate; crash and off-track rates; mean speed; action saturation; action delta or smoothness; approximate KL; PPO clip fraction; value loss/explained variance; and environment/trainer steps per second. FSAE’s public results show why physical metrics must be checked against training reward, and CleanRL/SB3 expose the PPO diagnostics needed to identify update instability. [FSAE RL Racing README](https://github.com/mk217alt/fsae_rl_racing) [CleanRL continuous PPO source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

- **Ablation before freezing:** run equal-budget, equal-seed comparisons for (a) two-action signed longitudinal control versus (b) three-action independent gas/brake, and separately compare (c) environment-only bounded observations versus (d) saved running normalization and (e) clipped Gaussian versus corrected tanh policy. Compare completion, lap time, crash/off-track rate, action saturation, action smoothness, approximate KL/clip fraction, and steps/s. [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) [Current project handoff experiments](../../RACING_EVENT_Agent_Handoff.md)

### Inferences

- The recommended public interface is therefore **two outputs, not three**: `steering` and `throttle_brake`. Three independent outputs add degrees of freedom without adding physical meaning to the current net-acceleration model, while the signed axis makes the action space symmetric and removes simultaneous-pedal ambiguity. This is the smallest contract consistent with the current simulator and the PPO guidance. [Current physics](../../racing_event/physics.py) [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment)

- The next implementation slice should freeze the schema and evaluation artifact format before optimizing the trainer. A deterministic replay hash and serialized normalization state will protect the competition more than changing from two to three outputs. [Ghostline repository](https://github.com/mreinhofferxd-pixel/Ghostline2DRacerRL) [QDrive README](https://github.com/Salzaufsauge/QDrive) [SB3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)

- If the action ablation later shows that three outputs materially improve lap time without increasing saturation, instability, or unfair tuning burden, introduce them as `action_schema_version=2`; do not make the trainer infer or silently translate between action conventions. Gymnasium’s and TMRL’s different pedal bounds show why the mapping must be explicit. [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) [TMRL README](https://github.com/trackmania-rl/tmrl#trackmania-gymnasium-environment)

### Gaps

- No public source found here establishes a universal best action dimension for racing PPO. The recommendation is a project-specific engineering choice grounded in the current signed-control physics and must be validated by the proposed ablation.

- No local throughput benchmark has been run as part of this research. The existing project README correctly treats vectorization and fixed training throughput as the next implementation milestone; publish no steps-per-second target until that benchmark is complete. [Project README](../../README.md) [Project first-slice document](../../docs/first_slice.md)

- The observation schema still needs a final field table and a decision on whether normalization is purely static or includes a saved running normalizer. That decision should be made together with the fixed PPO trainer, because action distribution, observation preprocessing, and evaluation loading are one reproducibility contract. [Current `RacingEnv` observation code](../../racing_event/env.py) [SB3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)

## Random respawns: verified follow-up (2026-10-16)

### Evidence and its limits

- Czechmanowski et al., **On learning racing policies with reinforcement
  learning** (2025), Section III-E, explicitly resets a rollout to a random
  position on the track when track limits are exceeded, "to ensure full track
  exploration." Their setup uses PPO, 400 concurrent instances and 120 million
  environment steps. This is a precedent for coverage, not a measured result
  for our small NumPy environment.
  [Primary full text, v1](https://arxiv.org/html/2504.02420v1#S3.SS5)
- That paper does **not** isolate random respawn versus fixed respawn as an
  unseen-track generalization ablation. Random respawn should reduce repeated
  practice of early corners, but does not prove that a policy cannot identify
  familiar sections from local geometry.
- Fuchs et al., **Super-Human Performance in Gran Turismo Sport Using Deep
  Reinforcement Learning**, Section V-C and conclusion, report failure to
  finish an unseen track despite local geometry/range observations. They also
  describe full-braking/stationary behavior with fixed wall penalties in
  Section III-A. These are useful warnings against equating good training
  reward or one-track performance with generalization.
  [Primary full text, v2](https://arxiv.org/html/2008.07971v2)
- No source code, geometry, dataset, or assets were copied from either paper.
  The exact paper the user recalled has not been identified; the 2025 paper is
  a directly verified match to the reset practice, not a claim to be that paper.

### Current project behavior

- `RolloutCollector` already resets ended lanes to a random track from the
  public training pool and a uniform centreline arclength. This includes barrier
  crashes, lap endings, backward endings, and stuck/time truncations.
- A reset is a **new episode**, not a teleport transition inside an existing
  lap. The terminal transition/reward is preserved, GAE does not cross the
  reset, distance starts at zero, and the persistent transition budget does not
  reset. Sampling a new spawn does not earn progress.
- Ordinary runoff entry currently continues with lower grip and an off-track
  signal. It does not reset the episode. Evaluation always has a fixed standing
  start and ends on a crash, with no respawn or second chance.

### Recommendation and questions saved for after the session

1. **Keep the present crash -> new random episode behavior.** Keep the 25%
   standing-start mixture pending a multi-seed experiment; random moving spawns
   alone can undertrain standing starts.
2. **Decision needed: retain soft runoff recovery, or test immediate off-track
   reset as a training-only ablation?** Recommend retaining recovery now. An
   immediate reset improves coverage but removes recovery examples, changes the
   task, and must not skip failure costs or be presented as a completed lap.
3. **Experiment E6/E3:** compare fixed-start and random-respawn training at equal
   budgets and several seeds, then evaluate fixed-start completion, lap time,
   crash/off-track rates, section coverage, and unseen public development tracks.
   Do not expose private tournament tracks or claim anti-memorization without
   that measurement.
4. **Question:** can the user supply the paper title/link later? It may contain
   a more specific reset ablation than these verified sources.

No training or physics rule is changed by this research/UI session.
