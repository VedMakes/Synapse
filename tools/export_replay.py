"""Export a Python-authored scripted or saved-policy run for a renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_policy, validate_evaluation_contract
from racing_event.observation import TrainingContract
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, replay_header, replay_result
from racing_event.vector_env import VectorizedRacingEnv
from tools.run_scripted import scripted_action


def export(args: argparse.Namespace) -> dict:
    sources = (args.config, args.track, args.weights, args.metadata, args.contract)
    if any(p is not None and Path(p).resolve() == args.output.resolve() for p in sources):
        raise ValueError("replay output must not overwrite a source artifact")
    config = SimulationConfig.from_file(args.config)
    track = TrackGeometry.from_file(args.track)
    env = VectorizedRacingEnv(track, config, n_envs=1)
    normalizer = None
    mode = "scripted" if args.weights is None else "evaluation"
    if args.weights is None:
        if args.contract is not None or args.metadata is not None:
            raise ValueError("--contract and --metadata require --weights")

        def policy(_observations: np.ndarray) -> np.ndarray:
            # The smoke controller reads a scalar snapshot, never steps a
            # second simulator. All transitions still go through evaluation.
            snapshot = SimpleNamespace(track=track, config=config, state=env.state.item(0))
            return scripted_action(snapshot)[None, :]
    else:
        from racing_event.ppo import FixedMLPActorCritic

        if args.contract is None:
            raise ValueError("--weights requires --contract with saved normalizer")
        contract, normalizer = TrainingContract.load(args.contract)
        validate_evaluation_contract(env, contract, normalizer)
        model, _ = FixedMLPActorCritic.load_artifact(
            args.weights, args.metadata, expected_observation_size=env.observation_size,
        )
        policy = model.as_policy(deterministic=True)

    header = replay_header(track, config, mode=mode, participant_id=args.participant_id)
    with ReplayWriter(args.output, header) as writer:
        result = evaluate_policy(
            env, policy, normalizer=normalizer, max_steps=args.steps, frame_sink=writer.append,
        )
        writer.finish(replay_result(result, config.tick_hz))
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--scripted", action="store_true", help="smoke controller, not a trained policy")
    policy_group.add_argument("--weights", type=Path, help="saved reference-PPO state dictionary")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--track", type=Path, default=Path("tracks/train/synthetic_oval.json"))
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--steps", type=int, default=None, help="optional preview horizon; never extends engine limits")
    parser.add_argument("--participant-id", default="local_driver")
    parser.add_argument("--output", type=Path, required=True, help="JSON output; parent directory must exist")
    args = parser.parse_args()
    try:
        result = export(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Replay export failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
