"""Launch a participant-owned training program.

SYNAPSE does not construct a participant model, choose an algorithm, or
calculate participant reward. This launcher only validates workspace paths and
forwards the published environment/training inputs to the participant's own
script. The bundled ``examples/participant_train.py`` is an optional reference
starter and may be replaced with any participant implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _workspace_path(value: str, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project workspace") from exc
    return resolved


def _positive(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("examples/participant_train.py"),
        help="participant-owned Python training entrypoint",
    )
    parser.add_argument("--reward", type=Path, default=Path("examples/baseline_reward.json"))
    parser.add_argument("--track-dir", type=Path, default=Path("tracks/train"))
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("participant_run"))
    parser.add_argument("--steps", type=lambda value: _positive(value, "--steps"), default=2048)
    parser.add_argument("--budget-steps", type=lambda value: _positive(value, "--budget-steps"), default=10000)
    parser.add_argument("--n-envs", type=lambda value: _positive(value, "--n-envs"), default=4)
    parser.add_argument("--rollout-length", type=lambda value: _positive(value, "--rollout-length"), default=64)
    parser.add_argument("--seed", type=lambda value: _positive(value, "--seed"), default=17)
    return parser.parse_args()


def build_training_command(args: argparse.Namespace) -> list[str]:
    trainer = _workspace_path(str(args.trainer), label="trainer")
    reward = _workspace_path(str(args.reward), label="reward")
    track_dir = _workspace_path(str(args.track_dir), label="track_dir")
    config = _workspace_path(str(args.config), label="config")
    output_dir = _workspace_path(str(args.output_dir), label="output_dir")
    if not trainer.is_file():
        raise ValueError(
            f"participant trainer does not exist: {trainer}. Copy the starter or set --trainer to your own script."
        )
    return [
        sys.executable,
        str(trainer),
        "--reward", str(reward),
        "--track-dir", str(track_dir),
        "--config", str(config),
        "--output-dir", str(output_dir),
        "--steps", str(args.steps),
        "--budget-steps", str(args.budget_steps),
        "--n-envs", str(args.n_envs),
        "--rollout-length", str(args.rollout_length),
        "--seed", str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    try:
        command = build_training_command(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
