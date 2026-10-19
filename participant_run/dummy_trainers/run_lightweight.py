"""Ephemeral participant smoke trainer used to exercise two tiny model shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from racing_event.budget import RunBudget, RunLimitExceeded
from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.reward import RewardConfig
from racing_event.rollout import RolloutCollector
from racing_event.track import TrackGeometry
from racing_event.trajectory import engine_code_hash
from racing_event.vector_env import VectorizedRacingEnv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--budget-steps", type=int, required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--rollout-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


class TinyPolicy:
    def __init__(self, observation_size: int, architecture: str, seed: int) -> None:
        if architecture not in {"linear", "tiny_mlp"}:
            raise ValueError(f"unsupported smoke architecture: {architecture}")
        rng = np.random.default_rng(seed)
        self.architecture = architecture
        self.hidden_size = 8 if architecture == "tiny_mlp" else None
        if self.hidden_size is None:
            self.weights = [rng.normal(0.0, 0.02, (2, observation_size)).astype(np.float32)]
            self.biases = [np.zeros(2, dtype=np.float32)]
        else:
            self.weights = [
                rng.normal(0.0, 0.02, (self.hidden_size, observation_size)).astype(np.float32),
                rng.normal(0.0, 0.02, (2, self.hidden_size)).astype(np.float32),
            ]
            self.biases = [np.zeros(self.hidden_size, dtype=np.float32), np.zeros(2, dtype=np.float32)]

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        values = values @ self.weights[0].T + self.biases[0]
        if self.hidden_size is not None:
            values = np.maximum(values, 0.0)
            values = values @ self.weights[1].T + self.biases[1]
        return np.tanh(values).astype(np.float32)

    def update_from_reward(self, reward: float) -> None:
        # This deliberately tiny participant algorithm proves that reward
        # configuration reaches model updates without adding a framework.
        adjustment = np.float32(np.tanh(reward) * 0.01)
        self.biases[-1][0] += adjustment
        self.biases[-1][1] -= adjustment

    def artifact_layers(self) -> list[dict[str, str]]:
        layers: list[dict[str, str]] = []
        for index, activation in enumerate(((["relu", "tanh"] if self.hidden_size is not None else ["tanh"]))):
            layers.append({
                "name": f"layer_{index}",
                "weights": f"layer_{index}_weights",
                "bias": f"layer_{index}_bias",
                "activation": activation,
            })
        return layers

    def arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"layer_{index}_weights"] = weights
            arrays[f"layer_{index}_bias"] = bias
        return arrays


def open_budget(path: Path, maximum: int) -> tuple[RunBudget, str]:
    if maximum <= 0:
        raise ValueError("--budget-steps must be positive")
    if path.exists():
        budget = RunBudget.open(path, max_runs=1, max_steps_per_run=maximum)
        status = budget.status()
        if status["runs"]:
            return budget, status["runs"][0]["run_id"]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        budget = RunBudget.create(path, max_runs=1, max_steps_per_run=maximum)
    try:
        return budget, budget.start_run()
    except RunLimitExceeded as exc:
        raise ValueError("smoke budget cannot start") from exc


def write_artifact(
    output_dir: Path,
    env: VectorizedRacingEnv,
    policy: TinyPolicy,
    reward: RewardConfig,
    reward_path: Path,
) -> dict[str, object]:
    artifact_dir = output_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    contract = TrainingContract(
        observation=env.training_contract.observation,
        track_ids=env.training_contract.track_ids,
        track_hashes=env.training_contract.track_hashes,
        config_hash=env.training_contract.config_hash,
        action_distribution="participant_defined",
        evaluation_spawn=env.training_contract.evaluation_spawn,
        deterministic_evaluation=True,
        reward_config_hash=reward.content_hash,
        reward_sha256=sha256(reward_path),
    )
    contract_path = artifact_dir / "training_contract.json"
    contract.save(contract_path)
    weights_path = artifact_dir / "weights.npz"
    np.savez(weights_path, **policy.arrays())
    manifest = {
        "format": "synapse_numpy_mlp",
        "format_version": 1,
        "environment_code_hash": engine_code_hash(),
        "observation": {
            "schema_version": "1.0",
            "size": env.observation_size,
            "dtype": "float32",
            "ray_angles_deg": list(env.config.sensors.ray_angles_deg),
            "ray_count": env.config.sensors.ray_count,
        },
        "action": {
            "schema_version": "1.0",
            "size": 2,
            "dtype": "float32",
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]],
            "output_activation": "tanh",
        },
        "preprocessing": {"kind": "identity"},
        "layers": policy.artifact_layers(),
        "weights_sha256": sha256(weights_path),
        "training_contract_sha256": sha256(contract_path),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "artifact_sha256": sha256(artifact_dir / "manifest.json"),
        "contract_config_hash": contract.config_hash,
        "reward_config_hash": contract.reward_config_hash,
        "observation_size": env.observation_size,
        "ray_angles_deg": list(env.config.sensors.ray_angles_deg),
        "architecture": policy.architecture,
    }


def run(architecture: str) -> None:
    args = parse_args()
    if min(args.steps, args.n_envs, args.rollout_length) <= 0:
        raise ValueError("training inputs must be positive")
    if args.n_envs * args.rollout_length > 8192:
        raise ValueError("smoke rollout is too large")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.from_file(args.config)
    reward = RewardConfig.from_file(args.reward)
    tracks = tuple(
        TrackGeometry.from_file(path)
        for path in sorted(args.track_dir.glob("*.json"))
        if path.name != "manifest.json"
    )
    if not tracks:
        raise ValueError("no tracks found")
    env = VectorizedRacingEnv(tracks, config, reward, n_envs=args.n_envs)
    policy = TinyPolicy(env.observation_size, architecture, args.seed)
    budget, run_id = open_budget(output_dir / "budget.json", args.budget_steps)
    collector = RolloutCollector(env, policy, budget, run_id, seed=args.seed)
    target = min(args.steps, budget.status(run_id)["remaining_steps"])
    rows: list[dict[str, object]] = []
    update = 0
    while target > 0:
        batch_steps = min(target, args.n_envs * args.rollout_length)
        result = collector.collect(batch_steps)
        active = result.batch.active_mask
        mean_reward = float(result.batch.rewards[active].mean())
        policy.update_from_reward(mean_reward)
        update += 1
        status = budget.status(run_id)
        rows.append({"update": update, "budget_used": status["used_steps"], "mean_reward": mean_reward})
        print(f"update {update}: steps {status['used_steps']}/{args.budget_steps} return={mean_reward:.6f} lap=n/a speed=n/a m/s crash=n/a off_track=n/a")
        target -= batch_steps
    artifact = write_artifact(output_dir, env, policy, reward, args.reward)
    status = budget.status(run_id)
    state = {
        "architecture": architecture,
        "budget_steps": args.budget_steps,
        "budget_used": status["used_steps"],
        "budget_remaining": status["remaining_steps"],
        "updates": update,
        "config_hash": env.training_contract.config_hash,
        "reward_config_hash": reward.content_hash,
        "ray_angles_deg": list(config.sensors.ray_angles_deg),
        "observation_size": env.observation_size,
        "artifact": artifact,
    }
    (output_dir / "training_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "training_metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "training_contract.json").write_bytes((output_dir / "artifact" / "training_contract.json").read_bytes())
    (output_dir / "training_report.md").write_text(
        f"# Dummy participant run\n\n- Architecture: `{architecture}`\n- Observation size: `{env.observation_size}`\n- Transitions: `{status['used_steps']}`\n",
        encoding="utf-8",
    )
    print(f"saved dummy participant run in {output_dir}")


if __name__ == "__main__":
    run("linear")
