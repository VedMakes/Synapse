"""Run the optional local PPO reference baseline.

This command is not the participant contract. It is an inspectable example
trainer used for smoke tests and local comparison. Participants may replace
the model, algorithm, reward function, and training loop with their own code.
It uses the existing Python-authoritative environment, a fixed reference PPO
model, a declarative reference reward config, and a persistent local budget.
Each rollout remains bounded so a multi-million-step budget is never retained
in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from racing_event.budget import RunBudget, RunLimitExceeded
from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer, TrainingContract
from racing_event.ppo import FixedMLPActorCritic, PPOTrainer
from racing_event.reward import RewardConfig
from racing_event.rollout import PPOHyperparameters, RolloutCollector
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK_DIR = ROOT / "tracks" / "train"
DEFAULT_CONFIG = ROOT / "config" / "simulation.json"
DEFAULT_REWARD = ROOT / "examples" / "baseline_reward.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_tracks(track_dir: Path, explicit: list[Path] | None) -> list[Path]:
    paths = explicit or sorted(track_dir.glob("*.json"))
    paths = [Path(path) for path in paths if Path(path).name != "manifest.json"]
    if not paths:
        raise ValueError("no training track JSON files were found")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--tracks", type=Path, nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("participant_run"))
    parser.add_argument("--budget-steps", type=int, default=10_000, help="provisional local budget; used only when creating a new state")
    parser.add_argument("--steps", type=int, default=2_048, help="maximum transitions to collect in this invocation")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _open_budget(path: Path, budget_steps: int) -> tuple[RunBudget, str]:
    if budget_steps <= 0:
        raise ValueError("--budget-steps must be positive")
    if path.exists():
        budget = RunBudget.open(path, max_runs=1, max_steps_per_run=budget_steps)
        status = budget.status()
        if len(status["runs"]) > 1:
            raise ValueError("participant prototype allows exactly one cumulative budget run")
        if status["runs"]:
            return budget, status["runs"][0]["run_id"]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        budget = RunBudget.create(path, max_runs=1, max_steps_per_run=budget_steps)
    try:
        return budget, budget.start_run()
    except RunLimitExceeded as exc:  # pragma: no cover - guarded by the status check
        raise ValueError("budget has no resumable run") from exc


def _load_or_create_policy(
    output_dir: Path,
    observation_size: int,
    hyperparameters: PPOHyperparameters,
    *,
    has_used_steps: bool,
) -> tuple[FixedMLPActorCritic, RunningObservationNormalizer, bool]:
    weights = output_dir / "policy.pt"
    metadata = output_dir / "policy.json"
    contract_path = output_dir / "training_contract.json"
    if weights.exists() != metadata.exists():
        raise ValueError("policy.pt and policy.json must be present together")
    if weights.exists() and not contract_path.exists():
        raise ValueError("training_contract.json is required beside a saved policy")
    if weights.exists():
        policy, saved_metadata = FixedMLPActorCritic.load_artifact(weights, metadata, expected_observation_size=observation_size)
        saved_ppo = saved_metadata.get("ppo")
        if saved_ppo is not None and saved_ppo != hyperparameters.to_dict():
            raise ValueError("saved PPO settings do not match this invocation")
        _, normalizer = TrainingContract.load(contract_path)
        if normalizer is None or not normalizer.training:
            raise ValueError("saved training contract must contain an unfrozen training normalizer")
        return policy, normalizer, True
    if has_used_steps:
        raise ValueError("budget is already used but no resumable policy artifact exists")
    return FixedMLPActorCritic(observation_size), RunningObservationNormalizer(observation_size), False


def _write_report(output_dir: Path, state: dict[str, Any]) -> None:
    metrics_path = output_dir / "training_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    lines = [
        "# Reference baseline training run",
        "",
        "This is an optional reference baseline. It is not the participant model or event contract.",
        "",
        "## Current state",
        "",
        f"- Release: `{state['release_id']}`",
        f"- Tracks: `{', '.join(state['track_ids'])}`",
        f"- Budget used: `{state['budget_used']}` / `{state['budget_steps']}` transitions",
        f"- Budget remaining: `{state['budget_remaining']}`",
        f"- Updates: `{state['updates']}`",
         f"- Reference model: fixed two-layer MLP `{state['hidden_sizes']}`",
        f"- Resume note: `{state['resume_note']}`",
        "",
        "## Update metrics",
        "",
        "| Update | Steps used | Mean return | Lap completion | Mean speed | Crash rate | Off-track rate | Approx. KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(name: str) -> str:
            item = row.get(name)
            return "n/a" if item is None else f"{item:.6f}" if isinstance(item, float) else str(item)

        lines.append("| " + " | ".join(value(name) for name in (
            "update", "budget_used", "mean_episode_return", "lap_completion_rate",
            "mean_speed_mps", "crash_rate", "off_track_rate", "approx_kl",
        )) + " |")
    (output_dir / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.n_envs <= 0 or args.rollout_length <= 0:
        raise ValueError("--steps, --n-envs, and --rollout-length must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    track_paths = _resolve_tracks(args.track_dir, args.tracks)
    config = SimulationConfig.from_file(args.config)
    reward = RewardConfig.from_file(args.reward)
    tracks = tuple(TrackGeometry.from_file(path) for path in track_paths)
    env = VectorizedRacingEnv(tracks, config, reward, n_envs=args.n_envs)
    hyperparameters = PPOHyperparameters(rollout_length=args.rollout_length)
    budget, run_id = _open_budget(output_dir / "budget.json", args.budget_steps)
    budget_status = budget.status(run_id)
    policy, normalizer, resumed = _load_or_create_policy(
        output_dir,
        env.observation_size,
        hyperparameters,
        has_used_steps=budget_status["used_steps"] > 0,
    )
    if resumed:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    trainer = PPOTrainer(policy, hyperparameters)
    collector = RolloutCollector(
        env,
        policy.as_policy(),
        budget,
        run_id,
        normalizer=normalizer,
        seed=args.seed,
    )
    state_path = output_dir / "training_state.json"
    state = {
        "release_id": "synapse-local-participant-prototype",
        "run_id": run_id,
        "budget_steps": args.budget_steps,
        "budget_used": budget_status["used_steps"],
        "budget_remaining": budget_status["remaining_steps"],
        "updates": 0,
        "track_ids": [track.track_id for track in tracks],
        "track_hashes": [track.content_hash for track in tracks],
        "config_hash": TrainingContract.config_hash_for(config),
        "reward_sha256": _sha256(args.reward),
        "hidden_sizes": [64, 64],
        "n_envs": args.n_envs,
        "rollout_length": args.rollout_length,
        "seed": args.seed,
        "resume_note": "optimizer state is reset on process restart in this prototype",
    }
    if state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        for name in ("track_ids", "track_hashes", "config_hash", "reward_sha256", "budget_steps", "n_envs", "rollout_length"):
            if saved_state.get(name) != state[name]:
                raise ValueError(f"saved training state does not match {name}")
        state["updates"] = int(saved_state.get("updates", 0))

    remaining = budget_status["remaining_steps"]
    target = min(args.steps, remaining)
    if target == 0:
        _write_report(output_dir, state)
        print(f"budget exhausted: {budget_status['used_steps']} / {args.budget_steps} transitions")
        return state

    metrics_path = output_dir / "training_metrics.jsonl"
    update_number = state["updates"]
    transitions_per_rollout = args.n_envs * args.rollout_length
    while target > 0:
        batch_steps = min(transitions_per_rollout, target)
        rollout = collector.collect(batch_steps)
        ppo_metrics = trainer.update(rollout.batch)
        update_number += 1
        budget_status = budget.status(run_id)
        row = {
            "update": update_number,
            "budget_used": budget_status["used_steps"],
            "budget_remaining": budget_status["remaining_steps"],
            "transitions": rollout.batch.transitions,
            "mean_episode_return": rollout.metrics["mean_episode_return"],
            "lap_completion_rate": rollout.metrics["lap_completion_rate"],
            "mean_speed_mps": rollout.metrics["mean_speed_mps"],
            "crash_rate": rollout.metrics["crash_rate"],
            "off_track_rate": rollout.metrics["off_track_rate"],
            **ppo_metrics.to_dict(),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        normalizer_state = normalizer
        policy.save_artifact(output_dir / "policy.pt", output_dir / "policy.json", hyperparameters=hyperparameters)
        env.training_contract.save(output_dir / "training_contract.json", normalizer=normalizer_state)
        state["updates"] = update_number
        state["budget_used"] = budget_status["used_steps"]
        state["budget_remaining"] = budget_status["remaining_steps"]
        _atomic_json(state_path, state)
        _write_report(output_dir, state)
        def display(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.3f}"

        print(
            f"update {update_number}: steps {budget_status['used_steps']}/{args.budget_steps} "
            f"return={display(row['mean_episode_return'])} "
            f"lap={display(row['lap_completion_rate'])} "
            f"speed={display(row['mean_speed_mps'])} m/s "
            f"crash={display(row['crash_rate'])} "
            f"off_track={display(row['off_track_rate'])}"
        )
        target -= batch_steps
    return state


def main() -> None:
    args = _parse_args()
    state = run(args)
    print(f"saved participant run in {args.output_dir}")
    print(f"budget remaining: {state.get('budget_remaining', 'see training_state.json')}")


if __name__ == "__main__":
    main()
