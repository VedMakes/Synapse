"""Launch a participant-owned training program.

SYNAPSE does not construct a participant model, choose an algorithm, or
calculate participant reward. This launcher only validates workspace paths and
forwards the published environment/training inputs to the participant's own
script. The bundled ``examples/participant_train.py`` is an optional reference
starter and may be replaced with any participant implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.client_budget import (
    ClientTrainingBudgetError,
    DEFAULT_MAX_RETRAINS,
    DEFAULT_MAX_STEPS_PER_RETRAIN,
    reserve_training_allocation,
)


ROOT = Path(__file__).resolve().parents[1]
CLIENT_BUDGET_PATH = ROOT / ".synapse" / "client_training_budget.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_public_training_tracks(track_dir: Path) -> tuple[tuple[str, str], ...]:
    """Require the participant launcher to train on the complete public set."""

    candidates = sorted(path for path in track_dir.glob("*.json") if path.name != "manifest.json")
    if not candidates:
        raise ValueError(f"track_dir contains no JSON tracks: {track_dir}")
    try:
        manifest = json.loads((ROOT / "tracks" / "train_manifest.json").read_text(encoding="utf-8"))
        expected = {
            (str(item["map_id"]), str(item["canonical_content_hash"]), str(item["delivered_file_sha256"]))
            for item in manifest["maps"]
        }
        selected = set()
        for path in candidates:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data["meta"]
            selected.add((str(meta["id"]), str(meta["content_hash"]), _sha256(path)))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"could not validate public training maps in {track_dir}") from exc
    if selected != expected:
        expected_ids = ", ".join(sorted(item[0] for item in expected))
        received = ", ".join(sorted(item[0] for item in selected)) or "none"
        raise ValueError(
            "participant training must use all public training maps "
            f"({expected_ids}); received {received}"
        )
    return tuple(sorted((item[0], item[1]) for item in selected))


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


def _positive_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("examples/participant_train.py"),
        help="participant-owned Python training entrypoint",
    )
    parser.add_argument("--reward", type=Path, default=None, help="optional participant reward input")
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
    track_dir = _workspace_path(str(args.track_dir), label="track_dir")
    config = _workspace_path(str(args.config), label="config")
    output_dir = _workspace_path(str(args.output_dir), label="output_dir")
    if not trainer.is_file():
        raise ValueError(
            f"participant trainer does not exist: {trainer}. Copy the starter or set --trainer to your own script."
        )
    if not config.is_file():
        raise ValueError(f"config does not exist: {config}")
    if not track_dir.is_dir():
        raise ValueError(f"track_dir is not a directory: {track_dir}")
    validate_public_training_tracks(track_dir)
    reward_value = getattr(args, "reward", None)
    reward = None if reward_value is None else _workspace_path(str(reward_value), label="reward")
    if reward is not None and not reward.is_file():
        raise ValueError(f"reward does not exist: {reward}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir must be a directory path: {output_dir}")
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise ValueError(f"output_dir parent must be a directory: {output_dir.parent}")
    steps = _positive_value(args.steps, "--steps")
    budget_steps = _positive_value(args.budget_steps, "--budget-steps")
    if budget_steps > DEFAULT_MAX_STEPS_PER_RETRAIN:
        raise ValueError(
            f"--budget-steps cannot exceed the client cap of {DEFAULT_MAX_STEPS_PER_RETRAIN:,} transitions per retrain"
        )
    n_envs = _positive_value(args.n_envs, "--n-envs")
    rollout_length = _positive_value(args.rollout_length, "--rollout-length")
    seed = _positive_value(args.seed, "--seed")
    command = [
        sys.executable,
        str(trainer),
        "--track-dir", str(track_dir),
        "--config", str(config),
        "--output-dir", str(output_dir),
        "--steps", str(steps),
        "--budget-steps", str(budget_steps),
        "--n-envs", str(n_envs),
        "--rollout-length", str(rollout_length),
        "--seed", str(seed),
    ]
    if reward is not None:
        command[2:2] = ["--reward", str(reward)]
    return command


def main() -> None:
    args = parse_args()
    try:
        command = build_training_command(args)
        allocation = reserve_training_allocation(
            CLIENT_BUDGET_PATH,
            _workspace_path(str(args.output_dir), label="output_dir"),
            _positive_value(args.budget_steps, "--budget-steps"),
            max_retrains=DEFAULT_MAX_RETRAINS,
            max_steps_per_retrain=DEFAULT_MAX_STEPS_PER_RETRAIN,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"client training allocation {allocation.slot}/{allocation.max_retrains} "
        f"({allocation.remaining_retrains} new retrains remaining)"
        + ("; resuming existing allocation" if allocation.reused else ""),
        flush=True,
    )
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
