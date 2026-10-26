"""Serve a dependency-free local web UI for a participant-owned trainer.

The UI is a thin Python controller around the participant launcher. It does
not implement a second trainer, model, environment, reward calculation, or
budget. Bind it to loopback only; the browser is a local presentation layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.reward import RewardConfig
from racing_event.trajectory import MAX_REPLAY_BYTES, load_replay
from tools.build_replay import build as build_browser_replay
from tools.package_live_submission import package_live_submission


ROOT = Path(__file__).resolve().parents[1]
UPDATE_PATTERN = re.compile(
    r"update\s+(?P<update>\d+):\s+steps\s+(?P<used>\d+)/(?P<budget>\d+)\s+"
    r"return=(?P<return>\S+)\s+lap=(?P<lap>\S+)\s+speed=(?P<speed>\S+)\s+m/s\s+"
    r"crash=(?P<crash>\S+)\s+off_track=(?P<off_track>\S+)"
)
LIVE_TELEMETRY_PREFIX = "LIVE_TELEMETRY "


def _workspace_path(value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty workspace path")
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project workspace") from exc
    return resolved


def _renderer_headless() -> bool:
    return os.environ.get("SDL_VIDEODRIVER") in ("dummy", "offscreen") or (
        sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    )


def _ready_python(*, require_torch: bool = False, require_pygame: bool = False, require_live: bool = False, headless: bool = False) -> Path:
    candidates = [ROOT / ".venv" / "bin" / "python", Path(sys.executable)]
    checked: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            probe = "import numpy"
            if require_torch:
                probe += "; import torch"
            if require_live:
                probe += "; import onnxruntime; import websockets"
            if require_pygame:
                # Import alone does not prove SDL can actually open a surface.
                probe += "; import pygame; pygame.display.init(); pygame.display.set_mode((320, 240))"
                if not headless:
                    probe += "; assert pygame.display.get_driver() not in ('dummy', 'offscreen'), 'SDL selected a headless driver; use Browser or an explicit dummy-driver check'"
                probe += "; pygame.quit()"
            result = subprocess.run(
                [str(candidate), "-c", probe],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
                env={**os.environ, "PYGAME_HIDE_SUPPORT_PROMPT": "1", **(
                    {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"} if headless else {}
                )},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checked.append(f"{candidate}: {exc}")
            continue
        if result.returncode == 0:
            return candidate
        detail = (result.stderr or result.stdout).strip().splitlines()
        checked.append(
            f"{candidate}: {detail[-1] if detail else ('numpy/torch import failed' if require_torch else 'numpy import failed')}"
        )
    raise ValueError(
        "SYNAPSE Python/SDL readiness check failed. Run "
        f"'{sys.executable} tools/setup_participant.py', then start the UI again. "
        + " | ".join(checked)
    )


def parse_update_line(line: str) -> dict[str, object] | None:
    match = UPDATE_PATTERN.search(line)
    if match is None:
        return None
    values: dict[str, object] = {
        "update": int(match.group("update")),
        "budget_used": int(match.group("used")),
        "budget_steps": int(match.group("budget")),
    }
    try:
        for name in ("return", "lap", "speed", "crash", "off_track"):
            raw = match.group(name)
            values[name] = None if raw == "n/a" else _finite_float(raw, name)
    except ValueError:
        return None  # Malformed trainer telemetry is still logged, not authoritative.
    return values


def parse_live_telemetry_line(line: str) -> dict[str, object] | None:
    """Parse one bounded live-race telemetry line emitted by the runner."""
    if not line.startswith(LIVE_TELEMETRY_PREFIX):
        return None
    try:
        value = json.loads(line[len(LIVE_TELEMETRY_PREFIX):])
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("kind") != "live_tick":
        return None
    if not isinstance(value.get("entrants"), list):
        return None
    return value


def _process_resource_snapshot(pid: int) -> dict[str, object]:
    """Return Linux process-tree resource telemetry without a new dependency."""
    if not sys.platform.startswith("linux"):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {"scope": "child_processes", "rss_mb": None, "peak_rss_mb": usage.ru_maxrss / 1024.0, "pids": 1}
    proc = Path("/proc")
    if not proc.is_dir():
        return {"scope": "process_tree", "rss_mb": None, "peak_rss_mb": None, "pids": 0}
    pids = {str(pid)}
    changed = True
    while changed:
        changed = False
        for entry in proc.glob("[0-9]*"):
            status = entry / "status"
            try:
                data = status.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parent = next((line.split(":", 1)[1].strip() for line in data.splitlines() if line.startswith("PPid:")), None)
            if parent in pids and entry.name not in pids:
                pids.add(entry.name)
                changed = True
    rss_kb = 0
    peak_kb = 0
    threads = 0
    for value in pids:
        try:
            data = (proc / value / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fields = {}
        for line in data.splitlines():
            if ":" in line:
                key, raw = line.split(":", 1)
                fields[key] = raw.strip()
        try:
            rss_kb += int(fields.get("VmRSS", "0 kB").split()[0])
            peak_kb += int(fields.get("VmHWM", "0 kB").split()[0])
            threads += int(fields.get("Threads", "0"))
        except (ValueError, IndexError):
            continue
    return {
        "scope": "process_tree",
        "rss_mb": round(rss_kb / 1024.0, 2),
        "peak_rss_mb": round(peak_kb / 1024.0, 2),
        "pids": len(pids),
        "threads": threads,
    }


def _atomic_json(path: Path, payload: object) -> None:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_incompatible_existing_run(output_dir: Path, identity: dict[str, object]) -> None:
    identity_path = output_dir / "run_identity.json"
    markers = (
        "budget.json",
        "training_state.json",
        "policy.pt",
        "policy.json",
        "training_contract.json",
        "training_metrics.jsonl",
    )
    if identity_path.exists():
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing run_identity.json is invalid; choose a new output directory") from exc
        if not isinstance(existing, dict):
            raise ValueError("existing run_identity.json must be an object; choose a new output directory")
        for name in ("config_hash", "reward_config_hash", "reward_sha256", "ray_angles_deg", "ray_count", "observation_size"):
            if existing.get(name) != identity.get(name):
                raise ValueError(f"output directory is already bound to a different {name}; choose a new output directory")
        return
    if any((output_dir / marker).exists() for marker in markers):
        raise ValueError("existing training artifacts have no run identity; choose a new output directory")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _parse_ray_angles(value: object, *, default: tuple[float, ...]) -> list[float]:
    raw_values: object = value
    if raw_values is None or raw_values == "":
        raw_values = list(default)
    if isinstance(raw_values, str):
        raw_values = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not isinstance(raw_values, (list, tuple)):
        raise ValueError("ray_angles_deg must be a comma-separated list or array")
    angles = [_finite_float(item, "ray angle") for item in raw_values]
    if not 1 <= len(angles) <= 16:
        raise ValueError("ray_angles_deg must contain 1..16 angles")
    if any(angle < -90.0 or angle > 90.0 for angle in angles):
        raise ValueError("ray_angles_deg must stay within [-90, 90] degrees")
    if len(set(angles)) != len(angles):
        raise ValueError("ray_angles_deg must not contain duplicates")
    if 0.0 not in angles:
        raise ValueError("ray_angles_deg must include one forward center ray at 0 degrees")
    return angles


def _parse_reward_weights(form: dict[str, object], reward_path: Path | None) -> RewardConfig | None:
    submitted = form.get("reward_weights")
    if submitted is not None:
        if not isinstance(submitted, dict):
            raise ValueError("reward_weights must be an object")
        return RewardConfig(weights=submitted)
    if reward_path is None:
        return None
    return RewardConfig.from_file(reward_path)


def _prepare_run_inputs(
    form: dict[str, object], *, config: Path, output_dir: Path, reward_path: Path | None,
) -> tuple[Path, Path | None, SimulationConfig, RewardConfig | None]:
    base_data = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(base_data, dict):
        raise ValueError("config must contain a JSON object")
    base_config = SimulationConfig.from_file(config)
    angles = _parse_ray_angles(form.get("ray_angles_deg"), default=base_config.sensors.ray_angles_deg)
    ray_count_value = form.get("ray_count")
    if ray_count_value not in (None, ""):
        if isinstance(ray_count_value, bool):
            raise ValueError("ray_count must be a positive integer")
        if isinstance(ray_count_value, int):
            ray_count = ray_count_value
        elif isinstance(ray_count_value, str) and ray_count_value.strip().isdigit():
            ray_count = int(ray_count_value.strip(), 10)
        else:
            raise ValueError("ray_count must be a positive integer")
        if isinstance(ray_count_value, bool) or ray_count != len(angles):
            raise ValueError("ray_count must equal the number of ray angles")
    sensors = dict(base_data.get("sensors", {}))
    sensors["ray_angles_deg"] = angles
    base_data["sensors"] = sensors
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run_config.json", "reward_config.json", "run_identity.json"):
        _workspace_path(str(output_dir / name), label=name)
    effective_config_path = output_dir / "run_config.json"
    candidate_config_path = output_dir / f".run_config.{os.getpid()}.candidate"
    candidate_reward_path = output_dir / f".reward_config.{os.getpid()}.candidate"
    try:
        _atomic_json(candidate_config_path, base_data)
        effective_config = SimulationConfig.from_file(candidate_config_path)
        reward = _parse_reward_weights(form, reward_path)
        effective_reward_path: Path | None = None
        reward_sha256: str | None = None
        if reward is not None:
            _atomic_json(candidate_reward_path, reward.to_dict())
            effective_reward_path = output_dir / "reward_config.json"
            reward_sha256 = _sha256_file(candidate_reward_path)
        identity = {
            "identity_version": 1,
            "config_hash": TrainingContract.config_hash_for(effective_config),
            "reward_config_hash": None if reward is None else reward.content_hash,
            "reward_sha256": reward_sha256,
            "ray_angles_deg": list(effective_config.sensors.ray_angles_deg),
            "ray_count": effective_config.sensors.ray_count,
            "observation_size": 10 + effective_config.sensors.ray_count + len(effective_config.sensors.lookahead_distances_m),
        }
        _reject_incompatible_existing_run(output_dir, identity)
        os.replace(candidate_config_path, effective_config_path)
        if reward is not None and effective_reward_path is not None:
            os.replace(candidate_reward_path, effective_reward_path)
        elif (output_dir / "reward_config.json").exists() and not identity["reward_config_hash"]:
            raise ValueError("existing reward_config.json conflicts with a reward-free run")
        _atomic_json(output_dir / "run_identity.json", identity)
    finally:
        candidate_config_path.unlink(missing_ok=True)
        candidate_reward_path.unlink(missing_ok=True)
    return effective_config_path, effective_reward_path, effective_config, reward


def build_training_command(form: dict[str, object]) -> list[str]:
    def text(name: str, default: str) -> str:
        value = form.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def positive_int(name: str, default: int) -> int:
        value = form.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = int(value.strip(), 10)
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
        else:
            raise ValueError(f"{name} must be a positive integer")
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    trainer = _workspace_path(text("trainer", "examples/participant_train.py"), label="trainer")
    reward_value = form.get("reward", "examples/baseline_reward.json")
    if reward_value is None or (isinstance(reward_value, str) and not reward_value.strip()):
        reward = None
    else:
        reward = _workspace_path(text("reward", "examples/baseline_reward.json"), label="reward")
    config = _workspace_path(text("config", "config/simulation.json"), label="config")
    track_dir = _workspace_path(text("track_dir", "tracks/train"), label="track_dir")
    output_dir = _workspace_path(text("output_dir", "participant_run"), label="output_dir")
    if not trainer.is_file():
        raise ValueError(f"trainer does not exist: {trainer}")
    if not config.is_file():
        raise ValueError(f"config does not exist: {config}")
    if not track_dir.is_dir():
        raise ValueError(f"track_dir is not a directory: {track_dir}")
    if not any(path.name != "manifest.json" for path in track_dir.glob("*.json")):
        raise ValueError(f"track_dir contains no JSON tracks: {track_dir}")
    if reward is not None and not reward.is_file():
        raise ValueError(f"reward does not exist: {reward}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir must be a directory path: {output_dir}")
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise ValueError(f"output_dir parent must be a directory: {output_dir.parent}")
    steps = positive_int("steps", 2048)
    budget_steps = positive_int("budget_steps", 10000)
    n_envs = positive_int("n_envs", 4)
    rollout_length = positive_int("rollout_length", 64)
    seed = positive_int("seed", 17)
    reference_trainer = trainer == (ROOT / "examples" / "participant_train.py").resolve()
    python = _ready_python(require_torch=reference_trainer)
    effective_config, effective_reward, _, _ = _prepare_run_inputs(
        form, config=config, output_dir=output_dir, reward_path=reward,
    )
    command = [
        str(python),
        str(ROOT / "tools" / "train_participant.py"),
        "--trainer", str(trainer),
        "--config", str(effective_config),
        "--track-dir", str(track_dir),
        "--output-dir", str(output_dir),
        "--steps", str(steps),
        "--budget-steps", str(budget_steps),
        "--n-envs", str(n_envs),
        "--rollout-length", str(rollout_length),
        "--seed", str(seed),
    ]
    if effective_reward is not None:
        command[2:2] = ["--reward", str(effective_reward)]
    return command


def build_live_race_command(form: dict[str, object]) -> list[str]:
    """Build the participant-facing two-model live heat command."""
    def text(name: str, default: str = "") -> str:
        value = form.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def positive_number(name: str, default: float) -> float:
        value = form.get(name, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return parsed

    def positive_int(name: str, default: int) -> int:
        value = form.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if str(value).strip() not in (str(parsed),) and not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer")
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    participant_a = text("participant_a", "Racer Alpha")
    participant_b = text("participant_b", "Racer Beta")
    if participant_a == participant_b:
        raise ValueError("participant IDs must be different")
    artifact_a = _workspace_path(text("artifact_a"), label="model A artifact")
    artifact_b = _workspace_path(text("artifact_b"), label="model B artifact")
    if not artifact_a.is_dir() or not artifact_b.is_dir():
        raise ValueError("both model artifacts must be existing directories")
    track = _workspace_path(text("race_track", "tracks/train/synthetic_oval.json"), label="race track")
    config = _workspace_path(text("race_config", "config/simulation.json"), label="race config")
    output_dir = _workspace_path(text("race_output", "artifacts/live/ui-heat"), label="race output")
    if not track.is_file():
        raise ValueError(f"race track does not exist: {track}")
    if not config.is_file():
        raise ValueError(f"race config does not exist: {config}")
    if output_dir.exists():
        raise ValueError("race output already exists; choose a new output directory")
    python = _ready_python(require_live=True)
    budget_ms = positive_number("race_budget_ms", 20.0)
    max_steps = positive_int("race_max_steps", 2400)
    port = positive_int("race_port", 8767)
    late_policy = form.get("race_late_policy", "strict_scored")
    if late_policy not in ("strict_scored", "trusted_live"):
        raise ValueError("race_late_policy must be strict_scored or trusted_live")
    command = [
        str(python), str(ROOT / "tools" / "run_live_onnx.py"),
        "--entrant", participant_a, str(artifact_a),
        "--entrant", participant_b, str(artifact_b),
        "--track", str(track), "--config", str(config),
        "--output-dir", str(output_dir), "--port", str(port),
        "--policy-budget-ms", str(budget_ms), "--max-steps", str(max_steps),
        "--late-policy", str(late_policy), "--telemetry-stdout",
    ]
    if form.get("race_isolate") in (True, "true", "1", 1):
        command.append("--isolate")
    return command


def build_live_package(form: dict[str, object]) -> dict[str, object]:
    """Export a trained run into a self-contained live-race package."""
    def text(name: str, default: str = "") -> str:
        value = form.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    source = _workspace_path(text("package_source", "participant_run"), label="package source")
    output = _workspace_path(text("package_output", "artifacts/live-packages/participant-run"), label="package output")
    track = _workspace_path(text("package_track", "tracks/train/synthetic_oval.json"), label="package track")
    participant_id = text("package_participant", "participant-model")
    if not source.is_dir():
        raise ValueError(f"package source does not exist: {source}")
    if not track.is_file():
        raise ValueError(f"package track does not exist: {track}")
    if output.exists():
        raise ValueError("package output already exists; choose a new directory")
    package = package_live_submission(source, output, track=track, participant_id=participant_id)
    manifest = package / "package_manifest.json"
    package_data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "package_dir": str(package),
        "submission_dir": str(package / "submission"),
        "manifest_path": str(manifest),
        "config_path": str(package / "config" / "simulation.json"),
        "track_path": str(package / str(package_data["track_path"])),
        "track_id": package_data["track_id"],
        "participant_id": participant_id,
        "source_kind": package_data["source_kind"],
    }


class TrainingProcess:
    def __init__(self) -> None:
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: Thread | None = None
        self._resource_thread: Thread | None = None
        self._resource_stop = Event()
        self._status: dict[str, object] = {
            "state": "idle",
            "lines": [],
            "metrics": {},
            "telemetry": {},
            "resources": {},
            "error": None,
            "started_at": None,
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            current = dict(self._status)
            current["lines"] = list(self._status["lines"])
            current["metrics"] = dict(self._status["metrics"])
            current["telemetry"] = dict(self._status["telemetry"])
            current["resources"] = dict(self._status["resources"])
            # Keep inputs frozen until the output reader has finalized this run.
            current["running"] = self._process is not None
            return current

    def start(self, form: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._ensure_idle()
            command = build_training_command(form)
            self._start_locked(command, kind="training")
        return self.snapshot()

    def _ensure_idle(self) -> None:
        if self._process is not None:
            raise RuntimeError("a process is already running or finishing")

    def start_command(self, command: list[str], *, kind: str, env: dict[str, str] | None = None) -> dict[str, object]:
        with self._lock:
            self._ensure_idle()
            self._start_locked(command, kind=kind, env=env)
        return self.snapshot()

    def _start_locked(self, command: list[str], *, kind: str, env: dict[str, str] | None = None) -> None:
        self._status = {
            "state": "starting",
            "kind": kind,
            "lines": [],
            "metrics": {},
            "telemetry": {},
            "resources": {},
            "error": None,
            "started_at": time.time(),
            "command": command,
        }
        try:
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._resource_stop.clear()
            self._resource_thread = Thread(target=self._sample_resources, args=(self._process,), daemon=True)
            self._resource_thread.start()
            self._thread = Thread(target=self._read_output, args=(self._process,), daemon=True)
            self._thread.start()
        except OSError as exc:
            self._status.update(state="failed", error=str(exc))
            raise

    def stop(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._status["state"] = "stopping"
                Thread(target=self._escalate_stop, args=(process,), daemon=True).start()
        return self.snapshot()

    @staticmethod
    def _escalate_stop(process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                with self._lock:
                    if self._process is not process:
                        continue
                    lines = self._status["lines"]
                    assert isinstance(lines, list)
                    lines.append(line)
                    del lines[:-100]
                    parsed = parse_update_line(line)
                    if parsed is not None:
                        self._status["metrics"] = parsed
                    live = parse_live_telemetry_line(line)
                    if live is not None:
                        self._status["telemetry"] = live
                    if self._status["state"] != "stopping":
                        self._status["state"] = "running"
            return_code = process.wait()
            with self._lock:
                if self._process is not process:
                    return
                self._status["returncode"] = return_code
                if self._status["state"] == "stopping":
                    self._status["state"] = "stopped"
                elif return_code == 0:
                    self._status["state"] = "completed"
                else:
                    self._status["state"] = "failed"
                    self._status["error"] = f"{self._status['kind']} exited with status {return_code}"
        except Exception as exc:  # pragma: no cover - defensive process boundary
            with self._lock:
                if self._process is process:
                    self._status["state"] = "failed"
                    self._status["error"] = str(exc)
        finally:
            self._resource_stop.set()
            if process.stdout is not None:
                process.stdout.close()
            with self._lock:
                if self._process is process:
                    self._process = None

    def _sample_resources(self, process: subprocess.Popen[str]) -> None:
        while not self._resource_stop.wait(0.25):
            snapshot = _process_resource_snapshot(process.pid)
            with self._lock:
                if self._process is not process:
                    return
                self._status["resources"] = snapshot
        snapshot = _process_resource_snapshot(process.pid)
        with self._lock:
            if self._process is process:
                self._status["resources"] = snapshot


PROCESS = TrainingProcess()
RACE_PROCESS = TrainingProcess()
RENDER_PROCESS = TrainingProcess()
REPLAY_OUTPUTS: dict[str, Path] = {}
REPLAY_LOCK = Lock()
# Serializes export with start/setup preparation; duplicate requests fail promptly.
ACTION_LOCK = Lock()


def public_maps() -> list[dict[str, str]]:
    maps = []
    for path in sorted((ROOT / "tracks" / "train").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            map_id = str(data["meta"]["id"])
            display_name = str(data["meta"].get("display_name", map_id))
            content_hash = str(data["meta"]["content_hash"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            continue
        maps.append({"id": map_id, "display_name": display_name, "path": str(path), "content_hash": content_hash})
    return maps


def build_watch_replay(form: dict[str, object]) -> dict[str, object]:
    """Export the explicitly selected artifact; renderers only read the result."""
    source, renderer = form.get("source"), form.get("renderer")
    if source not in ("reference", "artifact"):
        raise ValueError("source must explicitly be reference or artifact")
    if renderer not in ("pygame", "browser"):
        raise ValueError("renderer must explicitly be pygame or browser")
    map_id = form.get("map_id")
    if not isinstance(map_id, str) or not map_id:
        raise ValueError("map_id must be selected from the public SYNAPSE maps")
    selected = next((item for item in public_maps() if item["id"] == map_id), None)
    if selected is None:
        raise ValueError("unknown public SYNAPSE map")
    output_dir = _workspace_path(form.get("output_dir", "participant_run"), label="output_dir")
    config_path = _workspace_path(str(output_dir / "run_config.json"), label="saved run config")
    if not config_path.is_file():
        raise ValueError("saved run_config.json is required; choose the run directory matching this artifact's rays/config")
    artifact_dir = None
    if source == "artifact":
        artifact_dir = _workspace_path(form.get("artifact_dir"), label="artifact_dir")
        if not artifact_dir.is_dir():
            raise ValueError("artifact_dir must be an existing safe NumPy artifact directory")
    reference_files = [
        _workspace_path(str(output_dir / name), label=name)
        for name in ("policy.pt", "policy.json", "training_contract.json")
    ] if source == "reference" else []
    if source == "reference" and not all(path.is_file() for path in reference_files):
        raise ValueError(
            "the bundled reference starter must produce policy.pt, policy.json, and "
            "training_contract.json before its replay can be built"
        )
    python = _ready_python(require_torch=source == "reference")
    replay_dir = _workspace_path(str(output_dir / "replays"), label="replay directory")
    if artifact_dir is not None and (replay_dir == artifact_dir or artifact_dir in replay_dir.parents):
        raise ValueError("replay directory must stay outside the participant artifact directory")
    replay_dir.mkdir(parents=True, exist_ok=True)
    # Opaque per-build URLs prevent one source/run from replacing another tab's replay.
    replay_id = uuid4().hex
    replay_path = replay_dir / f"{replay_id}.json"
    html_path = replay_path.with_suffix(".html")
    try:
        export_command = [
            str(python), str(ROOT / "tools" / "export_replay.py"),
            "--track", selected["path"], "--config", str(config_path),
            "--participant-id", f"SYNAPSE-{source}-{map_id}", "--output", str(replay_path),
        ]
        if artifact_dir is not None:
            # The generic evaluator validates the original, hashed training contract.
            export_command += ["--artifact-dir", str(artifact_dir)]
        else:
            weights, metadata, contract = reference_files
            # The exporter supports multi-track training provenance directly;
            # preserve it and freeze only its in-memory evaluation normalizer.
            export_command += ["--weights", str(weights), "--metadata", str(metadata), "--contract", str(contract)]
        subprocess.run(export_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "replay command failed").strip()
        raise RuntimeError(f"could not build SYNAPSE replay: {detail}") from exc
    replay = load_replay(replay_path)
    build_browser_replay(replay_path, html_path)
    with REPLAY_LOCK:
        REPLAY_OUTPUTS[replay_id] = replay_path
        while len(REPLAY_OUTPUTS) > 32:
            del REPLAY_OUTPUTS[next(iter(REPLAY_OUTPUTS))]
    result = {
        "map_id": map_id,
        "source": source,
        "source_label": "Latest saved reference artifact" if source == "reference" else "Safe NumPy participant artifact",
        "display_name": selected["display_name"],
        "status": replay["result"]["status"],
        "lap_time_s": replay["result"]["lap_time_s"],
        "renderer": renderer,
        "headless": _renderer_headless(),
        "replay_path": str(replay_path),
        "browser_url": f"/replays/{replay_id}",
        "download_url": f"/replays/{replay_id}/download",
        "renderer_state": "ready",
    }
    if renderer == "pygame":
        try:
            headless = result["headless"]
            renderer_python = _ready_python(require_pygame=True, headless=headless)
            command = [str(renderer_python), str(ROOT / "tools" / "run_pygame.py"), "--replay", str(replay_path)]
            if headless:
                command.append("--headless")
            snapshot = RENDER_PROCESS.start_command(command, kind="pygame-renderer", env=(
                {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"} if headless else None
            ))
            result["renderer_state"] = snapshot["state"]
        except (ValueError, RuntimeError, OSError) as exc:
            # Keep the selected renderer honest and the validated fallback reachable.
            result.update(renderer_state="unavailable", renderer_error=str(exc))
    return result


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SYNAPSE · Training Console</title>
  <style>
     :root { color-scheme: dark; --bg: #05070d; --panel: #0b111b; --panel2: #101b29; --line: #20354a; --grid: #153044; --muted: #91a7bb; --text: #eef7ff; --cyan: #45f3ff; --pink: #ff3fa4; --amber: #ffd166; --acid: #b7f23a; --danger: #ff657c; --ink: #061018; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
      * { box-sizing: border-box; }
      [hidden] { display: none !important; }
     body { position: relative; min-height: 100vh; margin: 0; overflow-x: hidden; color: var(--text); background: radial-gradient(circle at 78% -5%, #183d49 0, transparent 34%), radial-gradient(circle at 8% 45%, #251b39 0, transparent 30%), var(--bg); line-height: 1.45; }
     body::before { position: fixed; inset: 0; z-index: 0; pointer-events: none; content: ""; opacity: .3; background-image: linear-gradient(115deg, transparent 0 46%, #45f3ff12 47%, transparent 48% 100%), repeating-linear-gradient(0deg, transparent 0 31px, #45f3ff0a 32px, transparent 33px), repeating-linear-gradient(90deg, transparent 0 31px, #ff3fa40b 32px, transparent 33px); mask-image: linear-gradient(to bottom, #000, transparent 86%); }
     body::after { position: fixed; inset: 0; z-index: 0; pointer-events: none; content: ""; opacity: .1; background: repeating-linear-gradient(180deg, transparent 0 3px, #fff 4px, transparent 5px); mix-blend-mode: screen; }
     button, input, select { font: inherit; }
     button { cursor: pointer; border: 1px solid #2e6074; border-radius: 4px; padding: 11px 16px; color: var(--text); background: linear-gradient(135deg, #14283a, #0d1825); font-weight: 800; letter-spacing: .03em; text-transform: uppercase; box-shadow: inset 0 0 0 1px #ffffff08, 0 0 0 transparent; transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease, background .18s ease; }
     button:hover:enabled { background: linear-gradient(135deg, #1e3b4b, #132236); border-color: var(--cyan); box-shadow: 0 0 18px #45f3ff26; transform: translateY(-1px); }
     button:disabled { cursor: not-allowed; opacity: .45; }
     .primary { color: #061118; background: linear-gradient(100deg, var(--cyan), #91ffe7); border-color: var(--cyan); box-shadow: 0 0 24px #45f3ff38; }
     .primary:hover:enabled { color: #061118; background: linear-gradient(100deg, #9affee, var(--cyan)); }
     input { width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 3px; color: var(--text); background: #060c14d9; padding: 9px 11px; font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; box-shadow: inset 0 0 18px #00000042; }
     input:focus-visible, button:focus-visible, select:focus-visible { outline: 2px solid var(--amber); outline-offset: 3px; }
     .shell { position: relative; z-index: 1; width: min(1180px, calc(100% - 32px)); margin: 0 auto; }
     header { display: flex; align-items: center; justify-content: space-between; min-height: 86px; border-bottom: 1px solid var(--line); gap: 20px; }
     .brand { font-weight: 900; letter-spacing: .13em; font-size: 13px; text-shadow: 0 0 14px #45f3ff55; }
     .brand-mark { color: var(--cyan); font-size: 28px; font-style: italic; margin-right: 9px; }
     .read-only { color: var(--muted); font: 10px ui-monospace, monospace; letter-spacing: .08em; }
     .read-only span { color: var(--acid); text-shadow: 0 0 9px var(--acid); }
     .hero { position: relative; padding: 31px 0 24px; }
     .hero::after { position: absolute; right: 0; bottom: 27px; width: 160px; height: 2px; content: ""; background: linear-gradient(90deg, transparent, var(--pink), var(--cyan)); box-shadow: 0 0 16px #ff3fa466; }
     .eyebrow { color: var(--pink); font: 800 10px ui-monospace, monospace; letter-spacing: .16em; text-transform: uppercase; }
     h1, h2, h3, p { margin: 0; }
     h1 { margin-top: 8px; color: var(--text); font-size: clamp(28px, 4vw, 48px); letter-spacing: -.045em; line-height: 1.05; text-shadow: 3px 3px 0 #ff3fa41c, 0 0 28px #45f3ff22; }
     .hero p { max-width: 680px; margin-top: 12px; color: var(--muted); font-size: 14px; }
     .hero-stamp { display: inline-flex; gap: 10px; margin-top: 16px; padding: 6px 9px; border: 1px solid #6a3e68; color: var(--amber); background: #1b1020b8; font: 10px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; }
     .grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, .65fr); gap: 16px; padding-bottom: 32px; }
     .panel { position: relative; border: 1px solid var(--line); border-radius: 5px; background: linear-gradient(150deg, #111f2eeb, var(--panel)); box-shadow: 0 18px 40px #00000045, inset 0 0 35px #45f3ff05; overflow: hidden; }
     .panel::before { position: absolute; top: 0; left: 0; width: 42px; height: 2px; content: ""; background: var(--cyan); box-shadow: 0 0 16px var(--cyan); }
     .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 16px 19px; border-bottom: 1px solid var(--line); background: linear-gradient(90deg, #45f3ff09, transparent 45%, #ff3fa408); }
     h2 { font: 800 13px ui-monospace, monospace; letter-spacing: .07em; text-transform: uppercase; }
     h3 { color: var(--cyan); font: 800 11px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; }
     .section { padding: 18px 19px; }
     .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px 13px; }
     label { display: grid; gap: 6px; color: var(--muted); font-size: 11px; }
     label.full { grid-column: 1 / -1; }
     .actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 19px; }
     .hint { margin-top: 10px; color: var(--muted); font-size: 11px; }
     .subcopy { margin: 7px 0 13px; color: var(--muted); font-size: 11px; }
     .signal-label { display: flex; align-items: center; justify-content: space-between; gap: 7px; color: var(--text); font: 700 10px ui-monospace, monospace; letter-spacing: .04em; text-transform: uppercase; }
     .signal-tag { color: var(--pink); font-size: 9px; }
     .signal-help { min-height: 30px; color: var(--muted); font-size: 10px; line-height: 1.35; }
     .signal-help code { color: var(--amber); font: inherit; }
     .status-main { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    .status-label { display: flex; align-items: center; gap: 9px; font-weight: 750; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #7890a5; }
     .dot[data-state="running"], .dot[data-state="starting"] { background: var(--cyan); box-shadow: 0 0 12px var(--cyan); }
     .dot[data-state="failed"] { background: var(--danger); box-shadow: 0 0 12px var(--danger); }
    .dot[data-state="completed"] { background: var(--amber); }
    .status-chip { padding: 5px 9px; border: 1px solid var(--line); border-radius: 6px; color: var(--muted); font: 10px ui-monospace, monospace; text-transform: uppercase; }
    .progress-wrap { margin-top: 21px; }
    .progress-meta { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font: 11px ui-monospace, monospace; }
    .progress { height: 9px; margin-top: 8px; overflow: hidden; border-radius: 5px; background: #0b141e; }
     .progress span { display: block; width: 0; height: 100%; background: linear-gradient(90deg, var(--pink), var(--cyan)); box-shadow: 0 0 16px var(--cyan); transition: width .25s ease; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin-top: 18px; }
     .metric { padding: 11px; border: 1px solid var(--line); border-radius: 3px; background: #0a131f; }
    .metric small { display: block; color: var(--muted); font-size: 10px; }
     .metric strong { display: block; margin-top: 5px; color: var(--cyan); font-size: 18px; font-variant-numeric: tabular-nums; text-shadow: 0 0 10px #45f3ff55; }
    .log { min-height: 166px; max-height: 245px; margin-top: 17px; padding: 12px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #081019; color: #b9d2dc; font: 11px/1.6 ui-monospace, monospace; white-space: pre-wrap; }
    .contract { display: grid; gap: 12px; }
     .contract-item { padding: 12px; border-left: 2px solid var(--pink); background: #0b1521; }
    .contract-item strong { display: block; font-size: 13px; }
     .contract-item span { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
     .callout { margin-top: 16px; padding: 13px; border: 1px solid #6f593d; border-radius: 3px; color: var(--amber); background: #2c251b; font-size: 11px; }
    .watch { display: grid; gap: 9px; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line); }
    .watch label { color: var(--muted); font-size: 11px; }
    .watch select { width: 100%; min-height: 42px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; color: var(--text); background: #0b141e; }
     .watch-result { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
      .watch-result a { color: var(--cyan); }
      .race-panel { grid-column: 1 / -1; }
      .race-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 17px; }
      .race-tile { padding: 14px; border: 1px solid var(--line); border-radius: 4px; background: #08121e; }
      .race-tile.cyan-tile { border-color: #45f3ff66; box-shadow: inset 0 0 25px #45f3ff0d; }
      .race-tile.amber-tile { border-color: #ffd16666; box-shadow: inset 0 0 25px #ffd1660d; }
      .race-tile-head { display: flex; justify-content: space-between; gap: 10px; color: var(--text); font: 800 12px ui-monospace, monospace; letter-spacing: .08em; }
      .race-tile-head strong { color: var(--acid); font-size: 10px; }
       .race-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; color: var(--muted); font: 10px ui-monospace, monospace; }
      .race-stats b { display: block; margin-top: 3px; color: var(--cyan); font-size: 14px; }
      .amber-tile .race-stats b { color: var(--amber); }
      .race-resource { margin-top: 13px; padding: 9px 11px; border: 1px solid #29465a; color: var(--muted); background: #07101a; font: 11px ui-monospace, monospace; }
      .race-check { display: flex; align-items: center; grid-template-columns: auto 1fr; gap: 9px; min-height: 42px; padding: 9px 11px; border: 1px solid var(--line); background: #060c14d9; }
      .race-check input { width: auto; min-height: auto; }
      .button-link { display: inline-flex; align-items: center; border: 1px solid #2e6074; border-radius: 4px; padding: 11px 16px; color: var(--cyan); background: linear-gradient(135deg, #14283a, #0d1825); font-weight: 800; letter-spacing: .03em; text-transform: uppercase; text-decoration: none; }
      .button-link:hover { border-color: var(--cyan); box-shadow: 0 0 18px #45f3ff26; }
      .race-terminal { min-height: 150px; max-height: 250px; }
      .subpanel { grid-column: 1 / -1; padding: 12px; border: 1px solid var(--line); border-radius: 3px; background: #0a131f; box-shadow: inset 0 0 20px #0000002b; }
      .subpanel h3 { margin-bottom: 10px; }
      .weight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
      .weight-grid label { gap: 5px; padding: 9px; border: 1px solid #1c3043; background: #07101a; }
      .weight-grid input { min-height: 36px; }
      .diagram-shell { margin-top: 14px; padding: 10px; border: 1px solid #21485a; background: linear-gradient(145deg, #07131d, #0d0b1b); }
      .diagram-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; color: var(--muted); font: 10px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; }
      .diagram-count { color: var(--acid); text-shadow: 0 0 9px #b7f23a66; }
      #ray-diagram { display: block; width: 100%; height: auto; aspect-ratio: 640 / 430; border: 1px solid #17354a; background: radial-gradient(circle at 50% 72%, #142f3c 0, transparent 27%), #060b12; }
      #ray-diagram .diagram-grid { fill: url(#diagram-grid); opacity: .52; }
      #ray-diagram .range-arc { fill: none; stroke: #45f3ff38; stroke-dasharray: 5 8; stroke-width: 2; }
      #ray-diagram .range-label { fill: #91a7bb; font: 10px ui-monospace, monospace; }
       #ray-diagram .ray-line { stroke: var(--pink); stroke-width: 2.5; opacity: .86; stroke-linecap: round; }
      #ray-diagram .ray-line.center { stroke: var(--cyan); stroke-width: 3.5; }
      #ray-diagram .ray-endpoint { fill: var(--pink); stroke: #fef6ff; stroke-width: 1.5; }
      #ray-diagram .ray-endpoint.center { fill: var(--cyan); }
      #ray-diagram .ray-leader { fill: none; stroke: #ff3fa466; stroke-width: 1.25; stroke-dasharray: 3 4; }
      #ray-diagram .ray-leader.center { stroke: #45f3ff99; }
      #ray-diagram .ray-label-box { fill: #07131ee8; stroke: #2b5367; stroke-width: 1; rx: 3; }
      #ray-diagram .ray-label-box.center { stroke: #45f3ffb8; }
      #ray-diagram .ray-label { fill: var(--text); font: 11px ui-monospace, monospace; }
      #ray-diagram .ray-label.center { fill: var(--cyan); }
      #ray-diagram .car-shadow { fill: #00000080; filter: url(#car-shadow); }
      #ray-diagram .car-body { fill: #15283a; stroke: var(--amber); stroke-width: 3; filter: url(#car-glow); }
      #ray-diagram .car-bumper { fill: #263e50; stroke: #8ea6b8; stroke-width: 1.5; }
      #ray-diagram .car-window { fill: #164b5d; stroke: var(--cyan); stroke-width: 2; }
      #ray-diagram .car-roof { fill: #102233; stroke: #55788a; stroke-width: 1; }
      #ray-diagram .car-wheel { fill: #050a10; stroke: #9bb4c4; stroke-width: 2; }
      #ray-diagram .car-rim { fill: #263c4a; stroke: var(--cyan); stroke-width: 1; }
      #ray-diagram .car-lamp { fill: var(--amber); filter: url(#ray-glow); }
      #ray-diagram .car-tail { fill: var(--pink); filter: url(#ray-glow); }
      #ray-diagram .car-centerline { stroke: #45f3ff77; stroke-width: 1.5; stroke-dasharray: 5 6; }
      #ray-diagram .car-caption { fill: #91a7bb; font: 10px ui-monospace, monospace; letter-spacing: .08em; }
      .ray-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 6px; margin-top: 9px; color: var(--muted); font: 11px ui-monospace, monospace; }
      .ray-summary span { min-width: 0; padding: 5px 7px; overflow: hidden; border: 1px solid #28445a; background: #091522; white-space: nowrap; text-overflow: ellipsis; }
      .ray-summary span[data-center="true"] { color: var(--cyan); border-color: #45f3ff7a; }
      .ray-summary span[data-invalid="true"] { color: var(--danger); border-color: #ff657c88; }
      .validation { min-height: 16px; margin-top: 8px; color: var(--acid); font-size: 10px; }
      .validation[data-kind="error"] { color: var(--danger); }
      .validation[data-kind="warn"] { color: var(--amber); }
      footer { padding: 0 0 30px; color: var(--muted); font-size: 11px; }
      footer code { color: var(--cyan); }
     @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } .hero::after { display: none; } }
      @media (max-width: 700px) { #ray-diagram .ray-callout, #ray-diagram .ray-leader, #ray-diagram .range-label { display: none; } }
       @media (max-width: 520px) { .shell { width: min(100% - 22px, 1180px); } .form-grid, .metrics, .weight-grid, .race-grid { grid-template-columns: 1fr; } label.full { grid-column: auto; } header { min-height: 70px; } .read-only { max-width: 120px; text-align: right; line-height: 1.5; } }
     @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .001ms !important; scroll-behavior: auto !important; } body::before, body::after { opacity: .12; } }
  </style>
</head>
<body>
  <main class="shell">
    <header><div class="brand"><span class="brand-mark">S/</span> SYNAPSE <span style="color:var(--muted);font-size:10px;letter-spacing:.04em">NIGHT SHIFT / TRAINING CONSOLE</span></div><div class="read-only"><span>●</span> TOUGE NIGHT // LOOPBACK</div></header>
     <section class="hero"><div class="eyebrow">PROJECT SYNAPSE / GRID CONTROL / PROVISIONAL RELEASE</div><h1>Own the line. Take the flag.</h1><p>Prepare your driver, then put two validated ONNX models on the same live circuit. Python owns the road, the clock, and every action; this console gives you the pit wall view.</p><div class="hero-stamp"><span>RUN MODE: LOCAL</span><span>TELEMETRY: LIVE</span><span>GRID: SIDE BY SIDE</span></div></section>
    <section class="grid">
      <section class="panel">
        <div class="panel-head"><h2>01 · Run setup</h2><span class="status-chip">participant code</span></div>
        <div class="section">
          <div class="form-grid">
            <label class="full">Your trainer script<input id="trainer" value="examples/participant_train.py" spellcheck="false"></label>
            <div class="subpanel"><h3>Reward function · signal weights</h3><p class="subcopy">Tune the nine published transition signals. Positive rewards attract behavior; negative weights penalize it. Zero disables a signal. Limits: -100 to +100.</p><div class="weight-grid">
              <label><span class="signal-label">Progress <span class="signal-tag">DENSE +</span></span><span class="signal-help"><code>progress</code> Signed forward track advance.</span><input id="reward-progress" type="number" step="0.01" value="12" aria-label="Progress weight"></label>
              <label><span class="signal-label">Speed <span class="signal-tag">DENSE +</span></span><span class="signal-help"><code>speed</code> Encourages useful speed above the floor.</span><input id="reward-speed" type="number" step="0.01" value="0.5" aria-label="Speed weight"></label>
              <label><span class="signal-label">Curvature speed <span class="signal-tag">MATCH</span></span><span class="signal-help"><code>curvature_matched_speed</code> Is zero when speed matches the curve target.</span><input id="reward-curvature_matched_speed" type="number" step="0.01" value="0.25" aria-label="Curvature matched speed weight"></label>
              <label><span class="signal-label">Off track <span class="signal-tag">COST -</span></span><span class="signal-help"><code>off_track</code> Time spent outside the drivable surface.</span><input id="reward-off_track" type="number" step="0.01" value="-1.5" aria-label="Off track weight"></label>
              <label><span class="signal-label">Crash <span class="signal-tag">EVENT -</span></span><span class="signal-help"><code>crash</code> One-shot impulse when barrier contact ends the run.</span><input id="reward-crash" type="number" step="0.01" value="-10" aria-label="Crash weight"></label>
              <label><span class="signal-label">Stuck <span class="signal-tag">EVENT -</span></span><span class="signal-help"><code>stuck</code> One-shot impulse when the vehicle stalls.</span><input id="reward-stuck" type="number" step="0.01" value="-8" aria-label="Stuck weight"></label>
              <label><span class="signal-label">Lap complete <span class="signal-tag">EVENT +</span></span><span class="signal-help"><code>lap_complete</code> One-shot impulse on a completed lap.</span><input id="reward-lap_complete" type="number" step="0.01" value="20" aria-label="Lap complete weight"></label>
              <label><span class="signal-label">Time <span class="signal-tag">DENSE -</span></span><span class="signal-help"><code>time</code> Charges every transition to discourage stalling.</span><input id="reward-time" type="number" step="0.01" value="-0.15" aria-label="Time weight"></label>
              <label><span class="signal-label">Steer smoothness <span class="signal-tag">ROUGH -</span></span><span class="signal-help"><code>steer_smoothness</code> Measures abrupt steering changes; larger means rougher.</span><input id="reward-steer_smoothness" type="number" step="0.01" value="-0.05" aria-label="Steer smoothness weight"></label>
            </div></div>
            <div class="subpanel"><h3>Forward-facing ray sensors</h3><div class="form-grid">
              <label>Ray count<input id="ray-count" type="number" min="1" max="16" step="1" value="9" aria-describedby="ray-hint ray-validation"></label>
              <label class="full">Angles in degrees, ordered as observation fields<input id="ray-angles" value="0,-15,15,-30,30,-60,60,-90,90" spellcheck="false" aria-describedby="ray-hint ray-validation"></label>
            </div><p id="ray-hint" class="hint">Use 1–16 unique finite angles in [-90, 90], including 0. Positive angles sweep left; negative angles sweep right. Order changes the observation contract.</p><div class="diagram-shell"><div class="diagram-head"><span>Forward sensor layout / dummy preview</span><span id="diagram-count" class="diagram-count">RAYS 9 / 9</span></div><svg id="ray-diagram" viewBox="0 0 640 430" role="img" aria-labelledby="ray-diagram-title ray-diagram-desc">
                <title id="ray-diagram-title">Forward-facing sensor layout</title>
                 <desc id="ray-diagram-desc">A clean top-down dummy sports car with one colored sensor line for each configured forward ray. The diagram represents angles only; it does not perform physics or raycasts.</desc>
                <defs>
                  <pattern id="diagram-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="#45f3ff18" stroke-width="1"></path></pattern>
                  <filter id="ray-glow"><feGaussianBlur stdDeviation="2.6" result="blur"></feGaussianBlur><feMerge><feMergeNode in="blur"></feMergeNode><feMergeNode in="SourceGraphic"></feMergeNode></feMerge></filter>
                  <filter id="car-glow"><feGaussianBlur stdDeviation="1.2" result="blur"></feGaussianBlur><feMerge><feMergeNode in="blur"></feMergeNode><feMergeNode in="SourceGraphic"></feMerge></feMerge></filter>
                   <filter id="car-shadow"><feGaussianBlur stdDeviation="5"></feGaussianBlur></filter>
                 </defs>
                 <rect class="diagram-grid" x="0" y="0" width="640" height="430"></rect>
                  <path class="range-arc" d="M126 270 A194 194 0 0 1 514 270"></path>
                 <text class="range-label" x="320" y="54" text-anchor="middle">MAX RANGE / 100 M</text>
                 <g id="ray-lines" aria-hidden="true"></g>
                 <g id="ray-callouts" aria-hidden="true"></g>
                 <g aria-hidden="true">
                   <path class="car-shadow" d="M274 304 Q274 286 292 279 Q320 268 348 279 Q366 286 366 304 L371 370 Q370 405 320 412 Q270 405 269 370 Z"></path>
                   <rect class="car-wheel" x="258" y="304" width="18" height="43" rx="8"></rect><circle class="car-rim" cx="267" cy="325" r="5"></circle>
                   <rect class="car-wheel" x="364" y="304" width="18" height="43" rx="8"></rect><circle class="car-rim" cx="373" cy="325" r="5"></circle>
                   <rect class="car-wheel" x="258" y="360" width="18" height="43" rx="8"></rect><circle class="car-rim" cx="267" cy="381" r="5"></circle>
                   <rect class="car-wheel" x="364" y="360" width="18" height="43" rx="8"></rect><circle class="car-rim" cx="373" cy="381" r="5"></circle>
                   <path class="car-body" d="M274 304 Q274 286 292 279 Q320 268 348 279 Q366 286 366 304 L371 370 Q370 405 320 412 Q270 405 269 370 Z"></path>
                   <path class="car-bumper" d="M289 286 Q320 274 351 286 L347 298 Q320 289 293 298 Z"></path>
                   <path class="car-window" d="M294 310 Q320 288 346 310 L341 340 Q320 348 299 340 Z"></path>
                   <path class="car-roof" d="M301 349 Q320 357 339 349 L345 377 Q320 391 295 377 Z"></path>
                   <path class="car-centerline" d="M320 301 L320 396"></path>
                   <rect class="car-lamp" x="293" y="286" width="14" height="6" rx="2"></rect><rect class="car-lamp" x="333" y="286" width="14" height="6" rx="2"></rect>
                   <rect class="car-tail" x="291" y="395" width="18" height="5" rx="2"></rect><rect class="car-tail" x="331" y="395" width="18" height="5" rx="2"></rect>
                 </g>
                 <text class="car-caption" x="320" y="426" text-anchor="middle">CAR HEADING ↑ · 0° = FORWARD</text>
              </svg><div id="ray-summary" class="ray-summary" aria-label="Ordered ray summary"></div><p id="ray-validation" class="validation" role="status" aria-live="polite">Layout ready. Edit the fields to preview the next pass.</p></div></div>
            <label>Training maps<input id="track-dir" value="tracks/train" spellcheck="false"></label>
            <label>Output directory<input id="output-dir" value="participant_run" spellcheck="false"></label>
            <label>Steps this run<input id="steps" type="number" min="1" step="1" value="2048"></label>
            <label>Cumulative budget<input id="budget-steps" type="number" min="1" step="1" value="10000"></label>
            <label>Parallel envs<input id="n-envs" type="number" min="1" step="1" value="4"></label>
            <label>Rollout length<input id="rollout-length" type="number" min="1" step="1" value="64"></label>
            <label>Seed<input id="seed" type="number" min="1" step="1" value="17"></label>
          </div>
          <div class="actions"><button id="start" class="primary" aria-describedby="run-error">Start / resume participant training</button><button id="stop" disabled>Stop after current process boundary</button><button id="setup">Set up / repair Python</button></div>
          <p id="run-error" class="validation" data-kind="error" role="alert" tabindex="-1"></p>
          <p class="hint">The trainer script is yours to replace. If NumPy is missing, use the setup button first. The first run is intentionally small. Repeat with the same output directory to continue the local counter.</p>
        </div>
      </section>
      <aside class="panel">
        <div class="panel-head"><h2>02 · SYNAPSE contract</h2></div>
        <div class="section contract">
          <div class="contract-item"><strong>Configurable observation</strong><span>10 self-state values + your ordered rays + four curvature lookaheads.</span></div>
          <div class="contract-item"><strong>2 bounded actions</strong><span>Steering and signed throttle / brake in [-1, 1].</span></div>
          <div class="contract-item"><strong>UI reward function</strong><span>All nine published signals are editable as static weights; the generated file is saved with the run.</span></div>
          <div class="contract-item"><strong>UI sensor layout</strong><span>Ray count and angles are part of the observation/config identity; changing them requires a new compatible run.</span></div>
          <div class="contract-item"><strong>SYNAPSE public maps: 5</strong><span>Three synthetic development maps plus Monaco and Spa reference layouts. Final event maps are not here.</span></div>
          <div class="watch">
            <label for="watch-source">Replay source</label><select id="watch-source"><option value="reference">Reference · latest saved artifact</option><option value="artifact">Participant · safe NumPy artifact</option></select>
            <label id="artifact-label" for="artifact-dir" hidden>Workspace-local artifact directory<input id="artifact-dir" value="participant_run/artifact" spellcheck="false" aria-describedby="watch-hint watch-error"></label>
            <label for="watch-map">Watch artifact for map:</label><select id="watch-map"><option value="">Loading SYNAPSE maps…</option></select>
            <label for="watch-renderer">Read-only renderer</label><select id="watch-renderer"><option value="pygame">Pygame · local display / headless check</option><option value="browser">Browser · explicit headless / remote fallback</option></select>
            <p id="watch-hint" class="hint">Uses the selected run's saved run_config.json, including ordered rays. Reference means latest saved, not best. Pygame opens on the server machine; headless mode only checks one frame. Browser and JSON download links are built for either choice.</p>
            <button id="watch" disabled aria-describedby="watch-hint watch-error">Build / watch replay</button>
            <p id="watch-error" class="validation" data-kind="error" role="alert" tabindex="-1"></p>
            <div id="watch-result" class="watch-result" role="status" aria-live="polite">Select the artifact source and renderer, then build a deterministic replay.</div>
            <p id="renderer-status" class="watch-result" role="status" aria-live="polite"></p>
          </div>
          <div class="callout">10,000 transitions is an onboarding example, not the final scored budget.</div>
        </div>
      </aside>
        <section class="panel race-panel">
          <div class="panel-head"><h2>03 · Package for live race</h2><span id="package-state-chip" class="status-chip">ready</span></div>
          <div class="section">
            <p class="subcopy">After training, build one clean ONNX bundle. It includes the validated model, matching config, selected public circuit, hashes, and a copy-ready live command.</p>
            <div class="form-grid">
              <label>Trained run / model source<input id="package-source" value="participant_run" spellcheck="false"></label>
              <label>New package directory<input id="package-output" value="artifacts/live-packages/participant-run" spellcheck="false"></label>
              <label>Live circuit<select id="package-track"></select></label>
              <label>Package participant ID<input id="package-participant" value="racer-alpha" spellcheck="false"></label>
              <label>Assign package to<select id="package-entrant"><option value="a">Racer Alpha</option><option value="b">Racer Beta</option></select></label>
            </div>
            <div class="actions"><button id="package" data-api-endpoint="/api/package" class="primary">Build ONNX live package</button></div>
            <p id="package-error" class="validation" data-kind="error" role="alert" tabindex="-1"></p>
            <div id="package-result" class="watch-result" role="status" aria-live="polite">No live package built yet. The package output must be a new directory; package_manifest.json will record its hashes.</div>
          </div>
        </section>
        <section class="panel race-panel">
          <div class="panel-head"><h2>04 · Grid control / live heat</h2><span id="race-state-chip" class="status-chip">idle</span></div>
         <div class="section">
           <div class="form-grid">
             <label>Racer Alpha ID<input id="participant-a" value="racer-alpha" spellcheck="false"></label>
             <label>Racer Beta ID<input id="participant-b" value="racer-beta" spellcheck="false"></label>
             <label>Model A artifact<input id="artifact-a" value="artifacts/onnx-imitation" spellcheck="false"></label>
             <label>Model B artifact<input id="artifact-b" value="artifacts/onnx-imitation" spellcheck="false"></label>
             <label>Race circuit<input id="race-track" value="tracks/train/synthetic_oval.json" spellcheck="false"></label>
             <label>Race config<input id="race-config" value="config/simulation.json" spellcheck="false"></label>
             <label>Output directory<input id="race-output" value="artifacts/live/ui-heat" spellcheck="false"></label>
             <label>WebSocket port<input id="race-port" type="number" min="1" max="65535" step="1" value="8767"></label>
             <label>Action budget (ms)<input id="race-budget-ms" type="number" min="1" step="0.1" value="20"></label>
             <label>Live ticks<input id="race-max-steps" type="number" min="1" step="1" value="2400"></label>
             <label>Late action policy<select id="race-late-policy"><option value="strict_scored">Strict scored / DNF late action</option><option value="trusted_live">Demo hold-last action</option></select></label>
             <label class="race-check"><input id="race-isolate" type="checkbox"> Run model workers in organizer isolation</label>
           </div>
           <div class="actions"><button id="race-start" class="primary">Start live heat</button><button id="race-stop" disabled>Red flag / stop heat</button><a id="race-open" class="button-link" href="#" target="_blank" rel="noopener" hidden>Open broadcast view</a></div>
           <p id="race-error" class="validation" data-kind="error" role="alert" tabindex="-1"></p>
           <div class="race-grid">
              <div class="race-tile cyan-tile"><div class="race-tile-head"><span id="race-name-a">RACER ALPHA</span><strong id="race-status-a">READY</strong></div><div class="race-stats"><span>LATENCY <b id="race-latency-a">—</b></span><span>MODEL FILE <b id="race-model-a">—</b></span><span>DEADLINES <b id="race-deadlines-a">0</b></span><span>FAILURES <b id="race-failures-a">0</b></span></div></div>
              <div class="race-tile amber-tile"><div class="race-tile-head"><span id="race-name-b">RACER BETA</span><strong id="race-status-b">READY</strong></div><div class="race-stats"><span>LATENCY <b id="race-latency-b">—</b></span><span>MODEL FILE <b id="race-model-b">—</b></span><span>DEADLINES <b id="race-deadlines-b">0</b></span><span>FAILURES <b id="race-failures-b">0</b></span></div></div>
           </div>
            <div class="race-resource" id="race-resource">Shared race process: waiting for telemetry. Runtime RSS is shared; model-file sizes are shown per racer.</div>
           <div id="race-terminal" class="log race-terminal">Grid control idle. Choose two model artifacts and start a live heat.</div>
         </div>
       </section>
       <section class="panel">
         <div class="panel-head"><h2>04 · Training status</h2><div id="state-chip" class="status-chip">idle</div></div>
        <div class="section">
          <div class="status-main"><div class="status-label"><span id="state-dot" class="dot" data-state="idle"></span><span id="state-label">Ready to train</span></div><span id="runtime" class="hint" style="margin:0">No process running</span></div>
          <div class="progress-wrap"><div class="progress-meta"><span id="budget-label">0 / 10,000 transitions</span><span id="remaining-label">10,000 remaining</span></div><div class="progress"><span id="progress-fill"></span></div></div>
          <div class="metrics"><div class="metric"><small>Current update</small><strong id="metric-update">—</strong></div><div class="metric"><small>Mean return</small><strong id="metric-return">—</strong></div><div class="metric"><small>Lap completion</small><strong id="metric-lap">—</strong></div><div class="metric"><small>Mean speed</small><strong id="metric-speed">—</strong></div><div class="metric"><small>Crash rate</small><strong id="metric-crash">—</strong></div><div class="metric"><small>Off-track rate</small><strong id="metric-offtrack">—</strong></div></div>
          <div id="log" class="log">Waiting for a local trainer process…</div>
        </div>
      </section>
    </section>
    <footer>Local SYNAPSE UI only. Start with <code>python tools/participant_ui.py</code>. The server binds to loopback and launches the existing CLI; it never becomes a second environment authority.</footer>
  </main>
  <script>
     const $ = id => document.getElementById(id);
     const signalNames = ["progress", "speed", "curvature_matched_speed", "off_track", "crash", "stuck", "lap_complete", "time", "steer_smoothness"];
        const fields = ["trainer", "track-dir", "output-dir", "steps", "budget-steps", "n-envs", "rollout-length", "seed", "ray-count", "ray-angles", ...signalNames.map(name => `reward-${name}`)];
        const packageFields = ["package-source", "package-output", "package-track", "package-participant", "package-entrant"];
       const raceFields = ["participant-a", "participant-b", "artifact-a", "artifact-b", "race-track", "race-config", "race-output", "race-port", "race-budget-ms", "race-max-steps", "race-late-policy", "race-isolate"];
     const defaultRayAngles = [0, -15, 15, -30, 30, -60, 60, -90, 90];
     const stateText = { idle: "Ready to train", starting: "Starting trainer…", running: "Training in progress", stopping: "Stopping trainer…", stopped: "Training stopped", completed: "Training completed", failed: "Training failed" };
      const svgNamespace = "http://www.w3.org/2000/svg";
       let pending = null, latestStatus = {}, latestRace = {}, actionVersion = 0, lastWatch = null;
      function formData() { return { trainer: $("trainer").value, track_dir: $("track-dir").value, output_dir: $("output-dir").value, steps: $("steps").value, budget_steps: $("budget-steps").value, n_envs: $("n-envs").value, rollout_length: $("rollout-length").value, seed: $("seed").value, ray_count: $("ray-count").value, ray_angles_deg: $("ray-angles").value, reward_weights: Object.fromEntries(signalNames.map(name => [name, $("reward-" + name).value])) }; }
      function raceFormData() { return { participant_a: $("participant-a").value, participant_b: $("participant-b").value, artifact_a: $("artifact-a").value, artifact_b: $("artifact-b").value, race_track: $("race-track").value, race_config: $("race-config").value, race_output: $("race-output").value, race_port: $("race-port").value, race_budget_ms: $("race-budget-ms").value, race_max_steps: $("race-max-steps").value, race_late_policy: $("race-late-policy").value, race_isolate: $("race-isolate").checked }; }
     function formatAngle(value) { const rounded = Number(Number(value).toFixed(2)); return `${rounded > 0 ? "+" : ""}${rounded}°`; }
     function rayPreview() {
       const raw = $("ray-angles").value.trim();
       const tokens = (raw === "" ? defaultRayAngles.map(String) : raw.split(",")).map(token => token.trim()).filter(token => token !== "");
       const entries = tokens.map((token, index) => { const value = Number(token); return { index, token, value, valid: token !== "" && Number.isFinite(value) }; });
       const values = entries.filter(entry => entry.valid).map(entry => entry.value);
       const rayCount = Number($("ray-count").value);
       let error = null;
        if (!/^\d+$/.test($("ray-count").value.trim()) || !Number.isInteger(rayCount) || rayCount < 1 || rayCount > 16) error = "Ray count must be an integer from 1 to 16.";
       else if (entries.some(entry => !entry.valid)) error = "Every ray angle must be a finite number.";
       else if (values.length !== rayCount) error = `Count mismatch: ${rayCount} rays requested, ${values.length} angles supplied.`;
       else if (values.some(value => value < -90 || value > 90)) error = "Angles must stay within -90° to +90°.";
       else if (new Set(values).size !== values.length) error = "Angles must be unique; duplicate rays would share an observation slot.";
       else if (!values.includes(0)) error = "Include a 0° center ray so the car has a straight-ahead sensor.";
       return { entries, values, rayCount, error };
     }
      function svgNode(name, attributes = {}) {
        const node = document.createElementNS(svgNamespace, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
        return node;
      }
      function renderRayDiagram() {
        const preview = rayPreview(), group = $("ray-lines"), callouts = $("ray-callouts"), summary = $("ray-summary"), validation = $("ray-validation");
        const originX = 320, originY = 270, radius = 194, labelTop = 64, centerLabelTop = 98, rowHeight = 24;
        let leftRow = 0, rightRow = 0;
        group.replaceChildren(); callouts.replaceChildren(); summary.replaceChildren();
        const drawable = preview.entries.filter(entry => entry.valid && entry.value >= -90 && entry.value <= 90);
        drawable.forEach(entry => {
          const radians = entry.value * Math.PI / 180;
          const endX = originX - Math.sin(radians) * radius, endY = originY - Math.cos(radians) * radius;
          const center = entry.value === 0, side = center ? "center" : entry.value > 0 ? "left" : "right";
          const row = side === "left" ? leftRow++ : side === "right" ? rightRow++ : 0;
          const boxX = side === "left" ? 8 : side === "right" ? 510 : 259;
          const boxY = side === "center" ? centerLabelTop : labelTop + row * rowHeight;
          const boxEdgeX = side === "left" ? 130 : side === "right" ? 510 : originX;
          const elbowX = side === "left" ? 144 : side === "right" ? 496 : originX;
          const boxCenterY = boxY + 10;
          const labelAnchor = side === "left" ? "start" : side === "right" ? "end" : "middle";
          const labelX = side === "left" ? boxX + 8 : side === "right" ? boxX + 114 : originX;
          const line = svgNode("line", { class: `ray-line${center ? " center" : ""}`, x1: originX, y1: originY, x2: endX.toFixed(2), y2: endY.toFixed(2), "aria-label": `Ray ${entry.index}, ${formatAngle(entry.value)}` });
          const endpoint = svgNode("circle", { class: `ray-endpoint${center ? " center" : ""}`, cx: endX.toFixed(2), cy: endY.toFixed(2), r: center ? 4 : 3 });
          group.appendChild(line); group.appendChild(endpoint);
          const leaderPoints = center
            ? `${endX.toFixed(2)},${endY.toFixed(2)} ${originX},${boxY + 20}`
            : `${endX.toFixed(2)},${endY.toFixed(2)} ${elbowX},${boxCenterY} ${boxEdgeX},${boxCenterY}`;
          const leader = svgNode("polyline", { class: `ray-leader${center ? " center" : ""}`, points: leaderPoints });
          const box = svgNode("rect", { class: `ray-label-box ray-callout${center ? " center" : ""}`, x: boxX, y: boxY, width: 122, height: 20 });
          const label = svgNode("text", { class: `ray-label ray-callout${center ? " center" : ""}`, x: labelX, y: boxY + 14, "text-anchor": labelAnchor });
          label.textContent = `ray_${entry.index} · ${formatAngle(entry.value)}`;
          callouts.appendChild(leader); callouts.appendChild(box); callouts.appendChild(label);
        });
        preview.entries.forEach(entry => {
          const chip = document.createElement("span");
          const drawableEntry = entry.valid && entry.value >= -90 && entry.value <= 90;
          chip.dataset.center = String(entry.value === 0); chip.dataset.invalid = String(!drawableEntry);
          chip.textContent = entry.valid ? `ray_${entry.index} · ${formatAngle(entry.value)}` : `ray_${entry.index} · invalid`;
          summary.appendChild(chip);
        });
       $("diagram-count").textContent = `RAYS ${preview.values.length} / ${Number.isInteger(preview.rayCount) ? preview.rayCount : "?"}`;
       validation.dataset.kind = preview.error ? "error" : "ok";
       validation.textContent = preview.error || `Layout valid: ${preview.values.length} ordered rays. The cyan line is 0° forward.`;
       $("ray-count").setAttribute("aria-invalid", preview.error ? "true" : "false"); $("ray-angles").setAttribute("aria-invalid", preview.error ? "true" : "false");
     }
      function format(value, suffix = "") { return value === null || value === undefined ? "—" : `${Number(value).toFixed(3)}${suffix}`; }
       function renderRace(data) {
        latestRace = data || {};
        const telemetry = latestRace.telemetry || {}, entrants = telemetry.entrants || [];
        $("race-state-chip").textContent = latestRace.state || "idle";
        $("race-terminal").textContent = (latestRace.lines || []).join("\n") || "Grid control idle. Choose two model artifacts and start a live heat.";
        $("race-terminal").scrollTop = $("race-terminal").scrollHeight;
        const resource = latestRace.resources || {};
        $("race-resource").textContent = resource.rss_mb == null ? `Shared race process: ${resource.scope || "waiting"}. Runtime RSS is shared; model-file sizes are shown per racer.` : `Shared race process RSS ${Number(resource.rss_mb).toFixed(1)} MB · peak ${Number(resource.peak_rss_mb || 0).toFixed(1)} MB · ${resource.pids || 0} PIDs · ${resource.threads || 0} threads · per-racer model file sizes below`;
        [0, 1].forEach(index => {
          const entrant = entrants[index] || {}, suffix = index === 0 ? "a" : "b";
          $(`race-name-${suffix}`).textContent = String(entrant.label || (index === 0 ? $("participant-a").value : $("participant-b").value) || "RACER").toUpperCase();
          $(`race-status-${suffix}`).textContent = String(entrant.status || "READY").toUpperCase();
          $(`race-latency-${suffix}`).textContent = entrant.latency_ms == null ? "—" : `${Number(entrant.latency_ms).toFixed(2)} ms`;
          $(`race-model-${suffix}`).textContent = entrant.model_mb == null ? "—" : `${Number(entrant.model_mb).toFixed(3)} MB`;
          $(`race-deadlines-${suffix}`).textContent = entrant.deadline_misses ?? 0;
          $(`race-failures-${suffix}`).textContent = entrant.inference_failures ?? 0;
        });
        const endpointLine = (latestRace.lines || []).find(line => line.startsWith("Live WebSocket endpoint:"));
         if (endpointLine) { const endpoint = endpointLine.split(": ").slice(1).join(": "); const link = $("race-open"); link.href = `http://127.0.0.1:8765/?live=${encodeURIComponent(endpoint)}`; link.hidden = false; }
       }
       function renderPackage(result) {
         const target = $("package-result");
         target.textContent = `ONNX live package ready · ${result.source_kind} · ${result.track_id} · package: ${result.package_dir} · submission: ${result.submission_dir} · manifest: ${result.manifest_path}`;
         $("package-state-chip").textContent = "ready";
         const entrant = $("package-entrant").value === "b" ? "b" : "a";
         $(entrant === "a" ? "artifact-a" : "artifact-b").value = result.submission_dir;
         $(entrant === "a" ? "participant-a" : "participant-b").value = result.participant_id;
         $("race-track").value = result.track_path;
         $("race-config").value = result.config_path;
       }
       function render(data) {
        latestStatus = data;
      const state = data.state || "idle", metrics = data.metrics || {}, used = Number(metrics.budget_used || 0), budget = Number(metrics.budget_steps || $("budget-steps").value || 1), remaining = Math.max(0, budget - used);
      $("state-chip").textContent = state; $("state-label").textContent = stateText[state] || state; $("state-dot").dataset.state = state; $("runtime").textContent = data.running ? "Process active" : (data.returncode === undefined ? "No process running" : `Exit ${data.returncode}`);
      $("budget-label").textContent = `${used.toLocaleString()} / ${budget.toLocaleString()} transitions`; $("remaining-label").textContent = `${remaining.toLocaleString()} remaining`; $("progress-fill").style.width = `${Math.min(100, used / Math.max(1, budget) * 100)}%`;
      $("metric-update").textContent = metrics.update ?? "—"; $("metric-return").textContent = format(metrics.return); $("metric-lap").textContent = format(metrics.lap); $("metric-speed").textContent = format(metrics.speed, " m/s");
      $("metric-crash").textContent = format(metrics.crash); $("metric-offtrack").textContent = format(metrics.off_track);
       $("log").textContent = (data.lines || []).join("\n") || "Waiting for a local trainer process…"; $("log").scrollTop = $("log").scrollHeight;
        if (lastWatch && lastWatch.renderer === "pygame" && data.renderer && lastWatch.renderer_state !== "unavailable") {
         $("renderer-status").textContent = `${lastWatch.headless ? "Headless Pygame · one-frame check only" : "Local Pygame"}: ${data.renderer.state}${data.renderer.error ? ` · ${data.renderer.error}` : ""}`;
        }
        renderRace(data.race || {});
        syncControls();
     }
     function trainingError() {
       for (const id of ["trainer", "track-dir", "output-dir"]) if (!$(id).value.trim()) return `${id} is required.`;
       for (const id of ["steps", "budget-steps", "n-envs", "rollout-length", "seed"]) {
         const value = Number($(id).value);
         if (!/^\d+$/.test($(id).value.trim()) || !Number.isSafeInteger(value) || value <= 0) return `${id} must be a positive integer.`;
       }
       for (const name of signalNames) {
         const raw = $("reward-" + name).value, value = Number(raw);
         if (!raw.trim() || !Number.isFinite(value) || value < -100 || value > 100) return `${name} weight must be finite and within [-100, 100].`;
       }
       return rayPreview().error;
     }
      function syncControls() {
        const active = Boolean(latestStatus.running) || latestStatus.state === "stopping";
        const raceActive = Boolean(latestRace.running) || latestRace.state === "stopping";
        const busy = Boolean(pending) || active || Boolean(latestStatus.action_busy);
         const raceBusy = Boolean(pending) || raceActive || active || Boolean(latestStatus.action_busy);
         fields.forEach(id => { $(id).disabled = busy; });
         packageFields.forEach(id => { $(id).disabled = busy || raceActive || active; });
         raceFields.forEach(id => { $(id).disabled = raceBusy; });
       $("start").disabled = busy || Boolean(trainingError());
       $("setup").disabled = busy;
       $("stop").disabled = Boolean(pending) || !active || latestStatus.state === "stopping";
       const watchBusy = busy || Boolean(latestStatus.renderer && latestStatus.renderer.running);
       ["watch-source", "watch-map", "watch-renderer"].forEach(id => { $(id).disabled = watchBusy; });
       $("artifact-label").hidden = $("watch-source").value !== "artifact";
       $("artifact-dir").disabled = watchBusy || $("artifact-label").hidden;
        $("watch").disabled = watchBusy || !$("watch-map").value || !$("output-dir").value.trim() || ($("watch-source").value === "artifact" && !$("artifact-dir").value.trim());
         $("race-start").disabled = raceBusy || Boolean(raceValidation());
         $("race-stop").disabled = Boolean(pending) || !raceActive || latestRace.state === "stopping";
         $("package").disabled = busy || raceActive || active || Boolean(packageValidation());
         $("package").setAttribute("aria-busy", String(pending === "package"));
         $("race-start").setAttribute("aria-busy", String(pending === "race/start"));
       $("watch").textContent = pending === "watch" ? "Building replay…" : "Build / watch replay";
        $("watch").setAttribute("aria-busy", String(pending === "watch"));
       }
       function packageValidation() {
         for (const id of ["package-source", "package-output", "package-track", "package-participant"]) if (!$(id) || !String($(id).value || "").trim()) return `${id} is required.`;
         return "";
       }
       function raceValidation() {
        const ids = [$("participant-a").value.trim(), $("participant-b").value.trim()];
        if (!ids[0] || !ids[1] || ids[0] === ids[1]) return "Use two different racer IDs.";
        for (const id of ["artifact-a", "artifact-b", "race-track", "race-config", "race-output"]) if (!$(id).value.trim()) return `${id} is required.`;
        for (const id of ["race-port", "race-max-steps"]) if (!/^\d+$/.test($(id).value.trim()) || Number($(id).value) <= 0) return `${id} must be a positive integer.`;
        if (!Number.isFinite(Number($("race-budget-ms").value)) || Number($("race-budget-ms").value) <= 0) return "Action budget must be positive.";
        return "";
      }
     async function request(path, options) { const response = await fetch(path, options); const body = await response.json(); if (!response.ok) throw new Error(body.error || "Request failed"); return body; }
     async function refresh() { const version = actionVersion; if (pending) return; try { const data = await request("/api/status"); if (!pending && version === actionVersion) render(data); } catch (error) { if (!pending && version === actionVersion) $("state-label").textContent = error.message; } }
      async function refreshMaps() { try { const maps = await request("/api/maps"); const select = $("watch-map"), packageSelect = $("package-track"); select.replaceChildren(); packageSelect.replaceChildren(); maps.forEach(map => { const option = document.createElement("option"); option.value = map.id; option.textContent = map.display_name; select.appendChild(option); const packageOption = document.createElement("option"); packageOption.value = map.path; packageOption.textContent = map.display_name; packageSelect.appendChild(packageOption); }); syncControls(); } catch (error) { $("watch-error").textContent = error.message; } }
     function renderWatch(result) {
       lastWatch = result;
       const target = $("watch-result");
       target.textContent = `${result.source_label} · ${result.display_name} · ${result.status} · ${result.lap_time_s == null ? "no lap time" : `${Number(result.lap_time_s).toFixed(3)} s`} · `;
       for (const [label, url, download] of [["Open browser fallback", result.browser_url, false], ["Download replay JSON", result.download_url, true]]) {
         const link = document.createElement("a"); link.textContent = label; link.href = url;
         if (download) link.download = "replay.json"; else { link.target = "_blank"; link.rel = "noopener"; }
         target.appendChild(link); target.appendChild(document.createTextNode(" · "));
       }
       $("renderer-status").textContent = result.renderer === "browser" ? "Browser fallback ready · open the link above on this device." : `${result.headless ? "Headless Pygame · one-frame check only; use the browser link for playback" : "Pygame on the server's local display"} · ${result.renderer_state}`;
       if (result.renderer_error) $("watch-error").textContent = result.renderer_error;
     }
       async function action(name, payload = {}) {
         const actionButton = name === "race/start" ? $("race-start") : name === "race/stop" ? $("race-stop") : $(name);
         if (pending || actionButton.disabled) return;
        if (name === "start" && trainingError()) return;
        if (name === "package" && packageValidation()) return;
        pending = name; actionVersion++;
         const errorTarget = $(name === "watch" ? "watch-error" : name === "package" ? "package-error" : name.startsWith("race/") ? "race-error" : "run-error"); errorTarget.textContent = "";
       syncControls();
       try {
         const result = await request(`/api/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
           if (name === "watch") renderWatch(result); else if (name === "package") renderPackage(result); else if (name.startsWith("race/")) render({ ...latestStatus, race: result }); else render(result);
       } catch (error) { errorTarget.textContent = error.message; errorTarget.focus(); }
       finally { pending = null; actionVersion++; syncControls(); refresh(); }
     }
      $("start").addEventListener("click", () => action("start", formData()));
       $("stop").addEventListener("click", () => action("stop"));
       $("setup").addEventListener("click", () => action("setup"));
       $("package").addEventListener("click", () => action("package", { package_source: $("package-source").value, package_output: $("package-output").value, package_track: $("package-track").value, package_participant: $("package-participant").value }));
       $("race-start").addEventListener("click", () => { const error = raceValidation(); $("race-error").textContent = error; if (!error) action("race/start", raceFormData()); });
      $("race-stop").addEventListener("click", () => action("race/stop"));
     $("watch").addEventListener("click", () => action("watch", { map_id: $("watch-map").value, output_dir: $("output-dir").value, source: $("watch-source").value, artifact_dir: $("artifact-dir").value, renderer: $("watch-renderer").value }));
      ["ray-count", "ray-angles"].forEach(id => $(id).addEventListener("input", renderRayDiagram));
      fields.forEach(id => { $(id).setAttribute("aria-describedby", `${$(id).getAttribute("aria-describedby") || ""} run-error`.trim()); $(id).addEventListener("input", () => { $("run-error").textContent = trainingError() || ""; syncControls(); }); });
        ["watch-source", "watch-map", "watch-renderer", "artifact-dir"].forEach(id => $(id).addEventListener("input", syncControls));
        packageFields.forEach(id => $(id).addEventListener("input", syncControls));
       raceFields.forEach(id => $(id).addEventListener("input", () => { $("race-error").textContent = raceValidation(); syncControls(); }));
      renderRayDiagram(); render({}); refreshMaps(); refresh(); setInterval(refresh, 700);
  </script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _local_request(self, *, mutation: bool = False) -> bool:
        # The dashboard can execute local code. Do not let a remote web page
        # trigger setup/start/stop, or use DNS rebinding to read local results.
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        invalid_origin = origin is not None and origin.lower() != f"http://{host}"
        cross_site = mutation and self.headers.get("Sec-Fetch-Site") == "cross-site"
        if host not in allowed or invalid_origin or cross_site:
            self.close_connection = True
            self._json(403, {"error": "dashboard requests must come from its local loopback origin"})
            return False
        return True

    def _send(self, status: int, content_type: str, body: bytes, *, download: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="replay.json"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, allow_nan=False).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if not self._local_request():
            return
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/status":
            self._json(200, {
                **PROCESS.snapshot(),
                "action_busy": ACTION_LOCK.locked(),
                "renderer": RENDER_PROCESS.snapshot(),
                "race": RACE_PROCESS.snapshot(),
            })
        elif path == "/api/maps":
            self._json(200, public_maps())
        elif path == "/api/report":
            report = ROOT / "participant_run" / "training_report.md"
            if not report.exists():
                self._json(404, {"error": "no participant report exists yet"})
            else:
                self._send(200, "text/markdown; charset=utf-8", report.read_bytes())
        elif path.startswith("/replays/"):
            replay_id, _, action = unquote(path.removeprefix("/replays/")).partition("/")
            with REPLAY_LOCK:
                replay_path = REPLAY_OUTPUTS.get(replay_id)
            if replay_path is None or action not in ("", "download"):
                self._json(404, {"error": "replay link is unknown or expired; build it again"})
                return
            try:
                replay_path = _workspace_path(str(replay_path), label="replay")
                load_replay(replay_path)
                download = action == "download"
                target = replay_path if download else replay_path.with_suffix(".html")
                target = _workspace_path(str(target), label="browser replay")
                limit = MAX_REPLAY_BYTES if download else 2 * MAX_REPLAY_BYTES + 1024 * 1024
                with target.open("rb") as stream:
                    body = stream.read(limit + 1)
                if len(body) > limit:
                    raise ValueError("replay exceeds byte limit")
                self._send(200, "application/json" if download else "text/html; charset=utf-8", body, download=download)
            except (ValueError, OSError) as exc:
                self._json(400, {"error": str(exc)})
        else:
            self._json(404, {"error": "not found"})

    def _read_form(self) -> dict[str, object]:
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            raise ValueError("use one bounded Content-Length; transfer encoding is unsupported")
        raw = lengths[0] if lengths else "0"
        if not re.fullmatch(r"[0-9]{1,8}", raw.strip()):
            raise ValueError("Content-Length must be a non-negative bounded integer")
        length = int(raw)
        if length > 16 * 1024:
            raise ValueError("request is too large")
        self.connection.settimeout(10)
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("incomplete request body")
        form = json.loads(payload.decode("utf-8")) if length else {}
        if not isinstance(form, dict):
            raise ValueError("request must be a JSON object")
        return form

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        # Close even on rejected lengths, so unread bytes cannot become a request.
        self.close_connection = True
        if not self._local_request(mutation=True):
            return
        acquired = False
        try:
            form = self._read_form()
            path = urlparse(self.path).path
            if path == "/api/stop":
                self._json(200, PROCESS.stop())
                return
            if path == "/api/race/stop":
                self._json(200, RACE_PROCESS.stop())
                return
            if path not in ("/api/start", "/api/setup", "/api/watch", "/api/package", "/api/race/start"):
                self._json(404, {"error": "not found"})
                return
            acquired = ACTION_LOCK.acquire(blocking=False)
            if not acquired:
                raise RuntimeError("a dashboard action is already in progress")
            if path == "/api/watch":
                if PROCESS.snapshot()["running"]:
                    raise RuntimeError("stop training before building a replay")
                if RACE_PROCESS.snapshot()["running"]:
                    raise RuntimeError("stop the live heat before building a replay")
                if RENDER_PROCESS.snapshot()["running"]:
                    raise RuntimeError("close the current Pygame replay before building another")
                result = build_watch_replay(form)
            elif path == "/api/package":
                if PROCESS.snapshot()["running"]:
                    raise RuntimeError("stop training before building a live package")
                if RACE_PROCESS.snapshot()["running"]:
                    raise RuntimeError("stop the live heat before building a live package")
                result = build_live_package(form)
            elif path == "/api/race/start":
                if PROCESS.snapshot()["running"]:
                    raise RuntimeError("stop training before starting a live heat")
                result = RACE_PROCESS.start_command(build_live_race_command(form), kind="live-race")
            elif path == "/api/setup":
                result = PROCESS.start_command([sys.executable, str(ROOT / "tools" / "setup_participant.py")], kind="setup")
            else:
                result = PROCESS.start(form)
            self._json(200, result)
        except (ValueError, RuntimeError, OSError, OverflowError, UnicodeError, RecursionError, subprocess.SubprocessError) as exc:
            self._json(400, {"error": str(exc)})
        finally:
            if acquired:
                ACTION_LOCK.release()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[participant-ui] {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address; keep loopback for local use")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("participant UI executes local code; --host must be 127.0.0.1 or localhost")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SYNAPSE participant UI: http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping participant UI")
    finally:
        PROCESS.stop()
        RACE_PROCESS.stop()
        RENDER_PROCESS.stop()
        server.server_close()


if __name__ == "__main__":
    main()
