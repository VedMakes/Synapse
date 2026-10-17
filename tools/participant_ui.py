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
import signal
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.reward import RewardConfig, SIGNAL_NAMES


ROOT = Path(__file__).resolve().parents[1]
UPDATE_PATTERN = re.compile(
    r"update\s+(?P<update>\d+):\s+steps\s+(?P<used>\d+)/(?P<budget>\d+)\s+"
    r"return=(?P<return>\S+)\s+lap=(?P<lap>\S+)\s+speed=(?P<speed>\S+)\s+m/s\s+"
    r"crash=(?P<crash>\S+)\s+off_track=(?P<off_track>\S+)"
)


def _workspace_path(value: str, *, label: str) -> Path:
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


def _ready_python(*, require_torch: bool = False) -> Path:
    candidates = [ROOT / ".venv" / "bin" / "python", Path(sys.executable)]
    checked: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            probe = "import numpy"
            if require_torch:
                probe += "; import torch"
            result = subprocess.run(
                [str(candidate), "-c", probe],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
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
        "SYNAPSE environment dependencies are not ready. Run "
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
    for name in ("return", "lap", "speed", "crash", "off_track"):
        raw = match.group(name)
        values[name] = None if raw == "n/a" else float(raw)
    return values


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
    return [_finite_float(item, "ray angle") for item in raw_values]


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
    effective_config_path = output_dir / "run_config.json"
    _atomic_json(effective_config_path, base_data)
    effective_config = SimulationConfig.from_file(effective_config_path)
    reward = _parse_reward_weights(form, reward_path)
    effective_reward_path: Path | None = None
    if reward is not None:
        effective_reward_path = output_dir / "reward_config.json"
        _atomic_json(effective_reward_path, reward.to_dict())
    identity = {
        "identity_version": 1,
        "config_hash": TrainingContract.config_hash_for(effective_config),
        "reward_config_hash": None if reward is None else reward.content_hash,
        "ray_angles_deg": list(effective_config.sensors.ray_angles_deg),
        "ray_count": effective_config.sensors.ray_count,
        "observation_size": 10 + effective_config.sensors.ray_count + len(effective_config.sensors.lookahead_distances_m),
    }
    _atomic_json(output_dir / "run_identity.json", identity)
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
    effective_config, effective_reward, _, _ = _prepare_run_inputs(
        form, config=config, output_dir=output_dir, reward_path=reward,
    )
    reference_trainer = trainer == (ROOT / "examples" / "participant_train.py").resolve()
    python = _ready_python(require_torch=reference_trainer)
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


class TrainingProcess:
    def __init__(self) -> None:
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: Thread | None = None
        self._status: dict[str, object] = {
            "state": "idle",
            "lines": [],
            "metrics": {},
            "error": None,
            "started_at": None,
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            current = dict(self._status)
            current["lines"] = list(self._status["lines"])
            current["metrics"] = dict(self._status["metrics"])
            current["running"] = self._process is not None and self._process.poll() is None
            if not current["running"] and current.get("state") == "stopping":
                current["state"] = "stopped"
            return current

    def start(self, form: dict[str, object]) -> dict[str, object]:
        command = build_training_command(form)
        return self.start_command(command, kind="training")

    def start_command(self, command: list[str], *, kind: str) -> dict[str, object]:
        with self._lock:
            if self._process is not None and (self._process.poll() is None or self._status.get("state") == "stopping"):
                raise RuntimeError("a training process is already running")
            self._status = {
                "state": "starting",
                "kind": kind,
                "lines": [],
                "metrics": {},
                "error": None,
                "started_at": time.time(),
                "command": command,
            }
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._thread = Thread(target=self._read_output, args=(self._process,), daemon=True)
            self._thread.start()
        return self.snapshot()

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
                    lines = self._status["lines"]
                    assert isinstance(lines, list)
                    lines.append(line)
                    del lines[:-100]
                    parsed = parse_update_line(line)
                    if parsed is not None:
                        self._status["metrics"] = parsed
                    self._status["state"] = "running"
            return_code = process.wait()
            with self._lock:
                self._status["returncode"] = return_code
                if return_code == 0:
                    self._status["state"] = "completed"
                elif self._status["state"] == "stopping":
                    self._status["state"] = "stopped"
                else:
                    self._status["state"] = "failed"
                    self._status["error"] = f"trainer exited with status {return_code}"
        except Exception as exc:  # pragma: no cover - defensive process boundary
            with self._lock:
                self._status["state"] = "failed"
                self._status["error"] = str(exc)
        finally:
            if process.stdout is not None:
                process.stdout.close()
            with self._lock:
                self._process = None


PROCESS = TrainingProcess()
REPLAY_OUTPUTS: dict[str, Path] = {}
REPLAY_LOCK = Lock()


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


def build_best_run(map_id: str, output_dir_value: str) -> dict[str, object]:
    if not isinstance(map_id, str) or not map_id:
        raise ValueError("map_id must be selected from the public SYNAPSE maps")
    selected = next((item for item in public_maps() if item["id"] == map_id), None)
    if selected is None:
        raise ValueError("unknown public SYNAPSE map")
    output_dir = _workspace_path(output_dir_value, label="output_dir")
    weights = output_dir / "policy.pt"
    metadata = output_dir / "policy.json"
    contract = output_dir / "training_contract.json"
    if not all(path.exists() for path in (weights, metadata, contract)):
        raise ValueError(
            "the bundled reference starter must produce policy.pt, policy.json, and "
            "training_contract.json before its replay can be built"
        )
    python = _ready_python(require_torch=True)
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / f"{map_id}.json"
    html_path = replay_dir / f"{map_id}.html"
    participant_id = f"SYNAPSE-reference-{map_id}"
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        config_path = ROOT / "config" / "simulation.json"
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_payload["tracks"] = {"ids": [map_id], "hashes": [selected["content_hash"]]}
    temporary_contract: Path | None = None
    fd, temporary_name = tempfile.mkstemp(prefix=f".{map_id}.contract.", suffix=".json", dir=replay_dir)
    temporary_contract = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(contract_payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        export_command = [
            str(python), str(ROOT / "tools" / "export_replay.py"),
            "--weights", str(weights), "--metadata", str(metadata), "--contract", str(temporary_contract),
            "--track", selected["path"], "--config", str(config_path),
            "--participant-id", participant_id, "--output", str(replay_path),
        ]
        build_command = [str(python), str(ROOT / "tools" / "build_replay.py"), str(replay_path), "--output", str(html_path)]
        subprocess.run(export_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=180)
        subprocess.run(build_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "replay command failed").strip()
        raise RuntimeError(f"could not build SYNAPSE replay: {detail}") from exc
    finally:
        if temporary_contract is not None:
            temporary_contract.unlink(missing_ok=True)
    with replay_path.open(encoding="utf-8") as stream:
        replay = json.load(stream)
    with REPLAY_LOCK:
        REPLAY_OUTPUTS[map_id] = html_path
    return {
        "map_id": map_id,
        "display_name": selected["display_name"],
        "status": replay.get("result", {}).get("status"),
        "lap_time_s": replay.get("result", {}).get("lap_time_s"),
        "url": f"/replays/{quote(map_id, safe='')}",
    }


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SYNAPSE · Training Console</title>
  <style>
    :root { color-scheme: dark; --bg: #091018; --panel: #121d29; --panel2: #172535; --line: #2b3d51; --muted: #9aacbf; --text: #eef7ff; --accent: #62ead1; --amber: #ffd28a; --danger: #ff9d91; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: radial-gradient(circle at 80% 0, #1d3a46 0, transparent 38%), var(--bg); line-height: 1.45; }
    button, input { font: inherit; }
    button { cursor: pointer; border: 1px solid #4d657c; border-radius: 8px; padding: 11px 16px; color: var(--text); background: #203143; font-weight: 700; }
    button:hover:enabled { background: #2d465b; border-color: #7591aa; }
    button:disabled { cursor: not-allowed; opacity: .45; }
    .primary { color: #08251f; background: var(--accent); border-color: var(--accent); }
    .primary:hover:enabled { color: #08251f; background: #9af8e5; }
    input { width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 7px; color: var(--text); background: #0b141e; padding: 9px 11px; }
    input:focus-visible, button:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; }
    header { display: flex; align-items: center; justify-content: space-between; min-height: 86px; border-bottom: 1px solid var(--line); gap: 20px; }
    .brand { font-weight: 850; letter-spacing: .11em; font-size: 13px; }
    .brand-mark { color: var(--accent); font-size: 28px; font-style: italic; margin-right: 9px; }
    .read-only { color: var(--muted); font: 10px ui-monospace, monospace; letter-spacing: .08em; }
    .read-only span { color: var(--accent); }
    .hero { padding: 31px 0 24px; }
    .eyebrow { color: var(--muted); font-size: 10px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1, h2, h3, p { margin: 0; }
    h1 { margin-top: 8px; font-size: clamp(28px, 4vw, 48px); letter-spacing: -.045em; line-height: 1.05; }
    .hero p { max-width: 680px; margin-top: 12px; color: var(--muted); font-size: 14px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, .65fr); gap: 16px; padding-bottom: 32px; }
    .panel { border: 1px solid var(--line); border-radius: 13px; background: linear-gradient(150deg, #142333, var(--panel)); box-shadow: 0 12px 30px #00000026; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 16px 19px; border-bottom: 1px solid var(--line); }
    h2 { font-size: 14px; }
    h3 { color: var(--muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
    .section { padding: 18px 19px; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px 13px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 11px; }
    label.full { grid-column: 1 / -1; }
    .actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 19px; }
    .hint { margin-top: 10px; color: var(--muted); font-size: 11px; }
    .status-main { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    .status-label { display: flex; align-items: center; gap: 9px; font-weight: 750; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #7890a5; }
    .dot[data-state="running"], .dot[data-state="starting"] { background: var(--accent); box-shadow: 0 0 12px var(--accent); }
    .dot[data-state="failed"] { background: var(--danger); }
    .dot[data-state="completed"] { background: var(--amber); }
    .status-chip { padding: 5px 9px; border: 1px solid var(--line); border-radius: 6px; color: var(--muted); font: 10px ui-monospace, monospace; text-transform: uppercase; }
    .progress-wrap { margin-top: 21px; }
    .progress-meta { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font: 11px ui-monospace, monospace; }
    .progress { height: 9px; margin-top: 8px; overflow: hidden; border-radius: 5px; background: #0b141e; }
    .progress span { display: block; width: 0; height: 100%; background: linear-gradient(90deg, #62ead1, #9af8e5); transition: width .25s ease; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin-top: 18px; }
    .metric { padding: 11px; border: 1px solid var(--line); border-radius: 8px; background: #0e1924; }
    .metric small { display: block; color: var(--muted); font-size: 10px; }
    .metric strong { display: block; margin-top: 5px; font-size: 18px; font-variant-numeric: tabular-nums; }
    .log { min-height: 166px; max-height: 245px; margin-top: 17px; padding: 12px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #081019; color: #b9d2dc; font: 11px/1.6 ui-monospace, monospace; white-space: pre-wrap; }
    .contract { display: grid; gap: 12px; }
    .contract-item { padding: 12px; border-left: 2px solid #3d6570; background: #0f1a25; }
    .contract-item strong { display: block; font-size: 13px; }
    .contract-item span { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
    .callout { margin-top: 16px; padding: 13px; border: 1px solid #6f593d; border-radius: 8px; color: var(--amber); background: #2c251b; font-size: 11px; }
    .watch { display: grid; gap: 9px; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line); }
    .watch label { color: var(--muted); font-size: 11px; }
    .watch select { width: 100%; min-height: 42px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; color: var(--text); background: #0b141e; }
     .watch-result { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
     .watch-result a { color: var(--accent); }
     .subpanel { grid-column: 1 / -1; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #0e1924; }
     .subpanel h3 { margin-bottom: 10px; }
     .weight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
     .weight-grid label { font-size: 10px; }
    footer { padding: 0 0 30px; color: var(--muted); font-size: 11px; }
    footer code { color: var(--text); }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    @media (max-width: 520px) { .shell { width: min(100% - 22px, 1180px); } .form-grid, .metrics { grid-template-columns: 1fr; } label.full { grid-column: auto; } header { min-height: 70px; } .read-only { max-width: 120px; text-align: right; line-height: 1.5; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .001ms !important; scroll-behavior: auto !important; } }
  </style>
</head>
<body>
  <main class="shell">
    <header><div class="brand"><span class="brand-mark">S/</span> SYNAPSE <span style="color:var(--muted);font-size:10px;letter-spacing:.04em">TRAINING CONSOLE</span></div><div class="read-only"><span>●</span> LOCAL / PYTHON AUTHORITATIVE</div></header>
    <section class="hero"><div class="eyebrow">SYNAPSE participant workspace / provisional release</div><h1>Train your driver.</h1><p>Choose reward weights and forward-facing rays in the UI, then run your own trainer. SYNAPSE owns the environment boundary; your code owns the model, algorithm, and weights.</p></section>
    <section class="grid">
      <section class="panel">
        <div class="panel-head"><h2>01 · Run setup</h2><span class="status-chip">participant code</span></div>
        <div class="section">
          <div class="form-grid">
            <label class="full">Your trainer script<input id="trainer" value="examples/participant_train.py" spellcheck="false"></label>
            <div class="subpanel"><h3>Reward function · signal weights</h3><div class="weight-grid">
              <label>Progress<input id="reward-progress" type="number" step="0.01" value="12"></label>
              <label>Speed<input id="reward-speed" type="number" step="0.01" value="0.5"></label>
              <label>Curvature speed<input id="reward-curvature_matched_speed" type="number" step="0.01" value="0.25"></label>
              <label>Off track<input id="reward-off_track" type="number" step="0.01" value="-1.5"></label>
              <label>Crash<input id="reward-crash" type="number" step="0.01" value="-10"></label>
              <label>Stuck<input id="reward-stuck" type="number" step="0.01" value="-8"></label>
              <label>Lap complete<input id="reward-lap_complete" type="number" step="0.01" value="20"></label>
              <label>Time<input id="reward-time" type="number" step="0.01" value="-0.15"></label>
              <label>Steer smoothness<input id="reward-steer_smoothness" type="number" step="0.01" value="-0.05"></label>
            </div></div>
            <div class="subpanel"><h3>Forward-facing ray sensors</h3><div class="form-grid">
              <label>Ray count<input id="ray-count" type="number" min="1" max="16" step="1" value="9"></label>
              <label class="full">Angles in degrees, ordered as observation fields<input id="ray-angles" value="0,-15,15,-30,30,-60,60,-90,90" spellcheck="false"></label>
            </div><p class="hint">Use 1–16 unique finite angles in [-90, 90], including 0. Order changes the observation contract.</p></div>
            <label>Training maps<input id="track-dir" value="tracks/train" spellcheck="false"></label>
            <label>Output directory<input id="output-dir" value="participant_run" spellcheck="false"></label>
            <label>Steps this run<input id="steps" type="number" min="1" step="1" value="2048"></label>
            <label>Cumulative budget<input id="budget-steps" type="number" min="1" step="1" value="10000"></label>
            <label>Parallel envs<input id="n-envs" type="number" min="1" step="1" value="4"></label>
            <label>Rollout length<input id="rollout-length" type="number" min="1" step="1" value="64"></label>
            <label>Seed<input id="seed" type="number" min="1" step="1" value="17"></label>
          </div>
          <div class="actions"><button id="start" class="primary">Start / resume participant training</button><button id="stop" disabled>Stop after current process boundary</button><button id="setup">Set up / repair Python</button></div>
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
          <div class="contract-item"><strong>SYNAPSE public maps: 3</strong><span>Original synthetic development maps. Final event maps are not here.</span></div>
          <div class="watch"><label for="watch-map">Watch reference artifact for map:</label><select id="watch-map"><option>Loading SYNAPSE maps…</option></select><button id="watch">Build / watch reference replay</button><div id="watch-result" class="watch-result">The bundled starter can build a deterministic replay from its reference artifact. A generic participant artifact exporter is a separate event boundary.</div></div>
          <div class="callout">10,000 transitions is an onboarding example, not the final scored budget.</div>
        </div>
      </aside>
      <section class="panel">
        <div class="panel-head"><h2>03 · Training status</h2><div id="state-chip" class="status-chip">idle</div></div>
        <div class="section">
          <div class="status-main"><div class="status-label"><span id="state-dot" class="dot" data-state="idle"></span><span id="state-label">Ready to train</span></div><span id="runtime" class="hint" style="margin:0">No process running</span></div>
          <div class="progress-wrap"><div class="progress-meta"><span id="budget-label">0 / 10,000 transitions</span><span id="remaining-label">9,? remaining</span></div><div class="progress"><span id="progress-fill"></span></div></div>
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
    const stateText = { idle: "Ready to train", starting: "Starting trainer…", running: "Training in progress", stopping: "Stopping trainer…", stopped: "Training stopped", completed: "Training completed", failed: "Training failed" };
    function formData() { return { trainer: $("trainer").value, track_dir: $("track-dir").value, output_dir: $("output-dir").value, steps: $("steps").value, budget_steps: $("budget-steps").value, n_envs: $("n-envs").value, rollout_length: $("rollout-length").value, seed: $("seed").value, ray_count: $("ray-count").value, ray_angles_deg: $("ray-angles").value, reward_weights: Object.fromEntries(signalNames.map(name => [name, $("reward-" + name).value])) }; }
    function format(value, suffix = "") { return value === null || value === undefined ? "—" : `${Number(value).toFixed(3)}${suffix}`; }
    function render(data) {
      const state = data.state || "idle", metrics = data.metrics || {}, used = Number(metrics.budget_used || 0), budget = Number(metrics.budget_steps || $("budget-steps").value || 1), remaining = Math.max(0, budget - used);
      $("state-chip").textContent = state; $("state-label").textContent = stateText[state] || state; $("state-dot").dataset.state = state; $("runtime").textContent = data.running ? "Process active" : (data.returncode === undefined ? "No process running" : `Exit ${data.returncode}`);
      $("budget-label").textContent = `${used.toLocaleString()} / ${budget.toLocaleString()} transitions`; $("remaining-label").textContent = `${remaining.toLocaleString()} remaining`; $("progress-fill").style.width = `${Math.min(100, used / Math.max(1, budget) * 100)}%`;
      $("metric-update").textContent = metrics.update ?? "—"; $("metric-return").textContent = format(metrics.return); $("metric-lap").textContent = format(metrics.lap); $("metric-speed").textContent = format(metrics.speed, " m/s");
      $("metric-crash").textContent = format(metrics.crash); $("metric-offtrack").textContent = format(metrics.off_track);
      $("start").disabled = Boolean(data.running) || state === "stopping"; $("stop").disabled = !Boolean(data.running); $("setup").disabled = Boolean(data.running) || state === "stopping"; $("watch").disabled = Boolean(data.running);
      $("log").textContent = (data.lines || []).join("\n") || "Waiting for a local trainer process…"; $("log").scrollTop = $("log").scrollHeight;
    }
    async function request(path, options) { const response = await fetch(path, options); const body = await response.json(); if (!response.ok) throw new Error(body.error || "Request failed"); return body; }
    async function refresh() { try { render(await request("/api/status")); } catch (error) { $("state-label").textContent = error.message; } }
    async function refreshMaps() { try { const maps = await request("/api/maps"); const select = $("watch-map"); select.innerHTML = ""; maps.forEach(map => { const option = document.createElement("option"); option.value = map.id; option.textContent = map.display_name; select.appendChild(option); }); } catch (error) { $("watch-result").textContent = error.message; } }
    $("start").addEventListener("click", async () => { try { render(await request("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formData()) })); } catch (error) { alert(error.message); } });
    $("stop").addEventListener("click", async () => { try { render(await request("/api/stop", { method: "POST" })); } catch (error) { alert(error.message); } });
    $("setup").addEventListener("click", async () => { try { render(await request("/api/setup", { method: "POST" })); } catch (error) { alert(error.message); } });
    $("watch").addEventListener("click", async () => { try { const result = await request("/api/watch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ map_id: $("watch-map").value, output_dir: $("output-dir").value }) }); const target = $("watch-result"); target.textContent = `${result.display_name} · ${result.status || "no result"} · ${result.lap_time_s === null || result.lap_time_s === undefined ? "no lap time" : `${Number(result.lap_time_s).toFixed(3)} s`} · `; const link = document.createElement("a"); link.href = result.url; link.target = "_blank"; link.rel = "noopener"; link.textContent = "Open replay"; target.appendChild(link); } catch (error) { $("watch-result").textContent = error.message; } });
    fields.forEach(id => $(id).addEventListener("change", refresh));
    refreshMaps(); refresh(); setInterval(refresh, 700);
  </script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, allow_nan=False).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/status":
            self._json(200, PROCESS.snapshot())
        elif path == "/api/maps":
            self._json(200, public_maps())
        elif path == "/api/report":
            report = ROOT / "participant_run" / "training_report.md"
            if not report.exists():
                self._json(404, {"error": "no participant report exists yet"})
            else:
                self._send(200, "text/markdown; charset=utf-8", report.read_bytes())
        elif path.startswith("/replays/"):
            map_id = unquote(path.removeprefix("/replays/")).strip("/")
            with REPLAY_LOCK:
                replay_path = REPLAY_OUTPUTS.get(map_id)
            if replay_path is None or not replay_path.exists():
                self._json(404, {"error": "replay has not been built for this map yet"})
            else:
                self._send(200, "text/html; charset=utf-8", replay_path.read_bytes())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/stop":
            self._json(200, PROCESS.stop())
            return
        if path == "/api/setup":
            try:
                command = [sys.executable, str(ROOT / "tools" / "setup_participant.py")]
                self._json(200, PROCESS.start_command(command, kind="setup"))
            except (RuntimeError, OSError, ValueError, OverflowError) as exc:
                self._json(400, {"error": str(exc)})
            return
        if path == "/api/watch":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16 * 1024:
                    raise ValueError("request is too large")
                form = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(form, dict):
                    raise ValueError("request must be a JSON object")
                if PROCESS.snapshot().get("running"):
                    raise RuntimeError("stop training before building a best-run replay")
                result = build_best_run(form.get("map_id", ""), str(form.get("output_dir", "participant_run")))
                self._json(200, result)
            except (ValueError, RuntimeError, OSError, OverflowError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
                self._json(400, {"error": str(exc)})
            return
        if path != "/api/start":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16 * 1024:
                raise ValueError("request is too large")
            form = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(form, dict):
                raise ValueError("request must be a JSON object")
            self._json(200, PROCESS.start(form))
        except (ValueError, RuntimeError, OSError, OverflowError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

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
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SYNAPSE participant UI: http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping participant UI")
    finally:
        PROCESS.stop()
        server.server_close()


if __name__ == "__main__":
    main()
