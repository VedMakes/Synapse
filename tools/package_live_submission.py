"""Create a self-contained participant package for a local live heat.

The package contains the validated ONNX submission, the exact saved config used
by the run, one selected public track, bounded provenance snapshots, and a
manifest with hashes plus a ready-to-copy live runner command. It never copies
participant Python, Torch checkpoints, or reward callbacks into the live
submission directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.onnx_artifact import load_onnx_artifact
from racing_event.track import TrackGeometry
from tools.export_onnx import export


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "simulation.json"
DEFAULT_TRACK = ROOT / "tracks" / "train" / "monaco.json"
PACKAGE_VERSION = "1"
_PROVENANCE_FILES = ("run_config.json", "reward_config.json", "run_identity.json", "training_report.md")
_MAX_PROVENANCE_BYTES = 4 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_regular(source: Path, destination: Path, *, max_bytes: int | None = None) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"package input must be a regular file: {source}")
    if max_bytes is not None and source.stat().st_size > max_bytes:
        raise ValueError(f"package input exceeds its byte limit: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def _source_kind(source: Path) -> tuple[Path, str, Path]:
    """Return (artifact source, kind, run directory)."""

    if source.is_symlink() or not source.is_dir():
        raise ValueError("model source must be an existing regular directory")
    nested = source / "artifact"
    if nested.is_dir() and (nested / "manifest.json").is_file():
        return nested, "numpy_artifact", source
    if (source / "manifest.json").is_file() and (source / "weights.npz").is_file():
        return source, "numpy_artifact", source.parent
    if all((source / name).is_file() for name in ("policy.pt", "policy.json", "training_contract.json")):
        return source, "reference", source
    raise ValueError(
        "model source must contain artifact/manifest.json or the reference "
        "policy.pt, policy.json, and training_contract.json"
    )


def _relative_file_list(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "package_manifest.json":
            continue
        files[str(path.relative_to(root))] = _sha256(path)
    return files


def _read_config(source_run_dir: Path, config_path: Path | None) -> Path:
    candidate = config_path or (source_run_dir / "run_config.json")
    if not candidate.is_file():
        candidate = DEFAULT_CONFIG
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("live package config must be a regular JSON file")
    SimulationConfig.from_file(candidate)
    return candidate


def package_live_submission(
    source: str | Path,
    output: str | Path,
    *,
    track: str | Path = DEFAULT_TRACK,
    config: str | Path | None = None,
    participant_id: str = "participant-model",
) -> Path:
    """Build a new live package directory and return its path."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    track_path = Path(track).resolve()
    config_path_arg = None if config is None else Path(config).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("live package output must be a new directory")
    if output_path == source_path or source_path in output_path.parents:
        raise ValueError("live package output cannot be inside the model source")
    if not participant_id or any(character in participant_id for character in "/\\\x00"):
        raise ValueError("participant_id must be a simple non-empty label")

    artifact_source, source_kind, source_run_dir = _source_kind(source_path)
    config_path = _read_config(source_run_dir, config_path_arg)
    config_value = SimulationConfig.from_file(config_path)
    track_value = TrackGeometry.from_file(track_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        submission = temporary / "submission"
        export(artifact_source, submission, reference=source_kind == "reference")
        onnx_artifact = load_onnx_artifact(submission)
        contract, _ = TrainingContract.load(submission / "training_contract.json")
        expected_config_hash = TrainingContract.config_hash_for(config_value)
        if contract.config_hash != expected_config_hash:
            raise ValueError(
                "saved live config does not match the model contract; use the run's run_config.json"
            )

        _copy_regular(config_path, temporary / "config" / "simulation.json")
        bundled_track = temporary / "tracks" / f"{track_value.track_id}.json"
        _copy_regular(track_path, bundled_track)

        provenance: list[str] = []
        for name in _PROVENANCE_FILES:
            candidate = source_run_dir / name
            if candidate.is_file() and not candidate.is_symlink():
                _copy_regular(candidate, temporary / "provenance" / name, max_bytes=_MAX_PROVENANCE_BYTES)
                provenance.append(f"provenance/{name}")

        # The runner remains the repository's authoritative runtime. Use paths
        # relative to the repository root so this command is copy-ready from
        # the same checkout that built the package.
        package_prefix = Path(os.path.relpath(output_path, ROOT)).as_posix()
        command = [
            "python", "tools/run_live_onnx.py",
            "--entrant", participant_id, f"{package_prefix}/submission",
            "--track", f"{package_prefix}/tracks/{track_value.track_id}.json",
            "--config", f"{package_prefix}/config/simulation.json",
            "--output-dir", f"{package_prefix}/live-results",
        ]
        readme = (
            "# SYNAPSE live model package\n\n"
            f"Participant: `{participant_id}`\n"
            f"Track: `{track_value.track_id}` ({track_value.length:.1f} m)\n\n"
            "This package contains a validated ONNX submission and the matching "
            "public track/config snapshot. Run from the repository root:\n\n"
            "```bash\n"
            + " ".join(command)
            + "\n```\n\n"
            "Python remains authoritative for observations, actions, physics, "
            "deadlines, and results. The package does not contain participant "
            "training code or reward callbacks.\n"
        )
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        manifest = {
            "kind": "synapse_live_package",
            "package_version": PACKAGE_VERSION,
            "source_kind": source_kind,
            "participant_id": participant_id,
            "submission_dir": "submission",
            "config_path": "config/simulation.json",
            "track_path": f"tracks/{track_value.track_id}.json",
            "track_id": track_value.track_id,
            "track_hash": track_value.content_hash,
            "config_hash": expected_config_hash,
            "observation_size": onnx_artifact.observation_size,
            "provenance": provenance,
            "recommended_command": command,
            "files": _relative_file_list(temporary),
        }
        (temporary / "package_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        # The manifest intentionally excludes itself because its file map is
        # self-referential; all other regular package files are hashed.
        os.rename(temporary, output_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="run directory or NumPy artifact directory")
    parser.add_argument("--output", type=Path, required=True, help="new package directory")
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--participant-id", default="participant-model")
    args = parser.parse_args()
    try:
        print(package_live_submission(args.source, args.output, track=args.track, config=args.config, participant_id=args.participant_id))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Live package failed: {exc}\n")


if __name__ == "__main__":
    main()
