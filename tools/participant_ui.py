"""Serve a dependency-free local web UI for the participant trainer.

The UI is a thin Python controller around ``tools/train_participant.py``. It
does not implement a second trainer, environment, reward calculation, or
budget. Bind it to loopback only; the browser is a local presentation layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock, Thread
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


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
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    reward = _workspace_path(text("reward", "examples/baseline_reward.json"), label="reward")
    config = _workspace_path(text("config", "config/simulation.json"), label="config")
    track_dir = _workspace_path(text("track_dir", "tracks/train"), label="track_dir")
    output_dir = _workspace_path(text("output_dir", "participant_run"), label="output_dir")
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    return [
        str(python),
        str(ROOT / "tools" / "train_participant.py"),
        "--reward", str(reward),
        "--config", str(config),
        "--track-dir", str(track_dir),
        "--output-dir", str(output_dir),
        "--steps", str(positive_int("steps", 2048)),
        "--budget-steps", str(positive_int("budget_steps", 10000)),
        "--n-envs", str(positive_int("n_envs", 4)),
        "--rollout-length", str(positive_int("rollout_length", 64)),
        "--seed", str(positive_int("seed", 17)),
    ]


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
            return current

    def start(self, form: dict[str, object]) -> dict[str, object]:
        command = build_training_command(form)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("a training process is already running")
            self._status = {
                "state": "starting",
                "lines": [],
                "metrics": {},
                "error": None,
                "started_at": time.time(),
                "command": command,
            }
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._thread = Thread(target=self._read_output, args=(self._process,), daemon=True)
            self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                self._status["state"] = "stopping"
        return self.snapshot()

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
                elif self._status["state"] != "stopping":
                    self._status["state"] = "failed"
                    self._status["error"] = f"trainer exited with status {return_code}"
        except Exception as exc:  # pragma: no cover - defensive process boundary
            with self._lock:
                self._status["state"] = "failed"
                self._status["error"] = str(exc)
        finally:
            with self._lock:
                self._process = None


PROCESS = TrainingProcess()


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Racing Event · Training Console</title>
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
    footer { padding: 0 0 30px; color: var(--muted); font-size: 11px; }
    footer code { color: var(--text); }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    @media (max-width: 520px) { .shell { width: min(100% - 22px, 1180px); } .form-grid, .metrics { grid-template-columns: 1fr; } label.full { grid-column: auto; } header { min-height: 70px; } .read-only { max-width: 120px; text-align: right; line-height: 1.5; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .001ms !important; scroll-behavior: auto !important; } }
  </style>
</head>
<body>
  <main class="shell">
    <header><div class="brand"><span class="brand-mark">R/</span> RACING EVENT <span style="color:var(--muted);font-size:10px;letter-spacing:.04em">TRAINING CONSOLE</span></div><div class="read-only"><span>●</span> LOCAL / PYTHON AUTHORITATIVE</div></header>
    <section class="hero"><div class="eyebrow">Participant workspace / provisional release</div><h1>Train your driver.</h1><p>Choose a reward configuration, start or resume a bounded local run, and watch the same metrics that will help you understand the agent. The browser is only a control surface. Python owns the environment, transitions, and budget.</p></section>
    <section class="grid">
      <section class="panel">
        <div class="panel-head"><h2>01 · Run setup</h2><span class="status-chip">release prototype</span></div>
        <div class="section">
          <div class="form-grid">
            <label class="full">Reward configuration<input id="reward" value="examples/baseline_reward.json" spellcheck="false"></label>
            <label>Training maps<input id="track-dir" value="tracks/train" spellcheck="false"></label>
            <label>Output directory<input id="output-dir" value="participant_run" spellcheck="false"></label>
            <label>Steps this run<input id="steps" type="number" min="1" step="1" value="2048"></label>
            <label>Cumulative budget<input id="budget-steps" type="number" min="1" step="1" value="10000"></label>
            <label>Parallel envs<input id="n-envs" type="number" min="1" step="1" value="4"></label>
            <label>Rollout length<input id="rollout-length" type="number" min="1" step="1" value="64"></label>
            <label>Seed<input id="seed" type="number" min="1" step="1" value="17"></label>
          </div>
          <div class="actions"><button id="start" class="primary">Start / resume training</button><button id="stop" disabled>Stop after current process boundary</button></div>
          <p class="hint">The first run is intentionally small. Repeat with the same output directory to continue the local counter.</p>
        </div>
      </section>
      <aside class="panel">
        <div class="panel-head"><h2>02 · Fixed contract</h2></div>
        <div class="section contract">
          <div class="contract-item"><strong>23-value observation</strong><span>Track-relative float32 state. No track ID or absolute progress.</span></div>
          <div class="contract-item"><strong>2 bounded actions</strong><span>Steering and signed throttle / brake in [-1, 1].</span></div>
          <div class="contract-item"><strong>Fixed PPO policy</strong><span>Two tanh layers, 64 units each. Reward weights are the experiment surface.</span></div>
          <div class="contract-item"><strong>Public maps: 3</strong><span>Original synthetic development maps. Final event maps are not here.</span></div>
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
    <footer>Local UI only. Start with <code>python tools/participant_ui.py</code>. The server binds to loopback and launches the existing CLI; it never becomes a second environment authority.</footer>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    const fields = ["reward", "track-dir", "output-dir", "steps", "budget-steps", "n-envs", "rollout-length", "seed"];
    const stateText = { idle: "Ready to train", starting: "Starting trainer…", running: "Training in progress", stopping: "Stopping trainer…", completed: "Training completed", failed: "Training failed" };
    function formData() { return { reward: $("reward").value, track_dir: $("track-dir").value, output_dir: $("output-dir").value, steps: $("steps").value, budget_steps: $("budget-steps").value, n_envs: $("n-envs").value, rollout_length: $("rollout-length").value, seed: $("seed").value }; }
    function format(value, suffix = "") { return value === null || value === undefined ? "—" : `${Number(value).toFixed(3)}${suffix}`; }
    function render(data) {
      const state = data.state || "idle", metrics = data.metrics || {}, used = Number(metrics.budget_used || 0), budget = Number(metrics.budget_steps || $("budget-steps").value || 1), remaining = Math.max(0, budget - used);
      $("state-chip").textContent = state; $("state-label").textContent = stateText[state] || state; $("state-dot").dataset.state = state; $("runtime").textContent = data.running ? "Process active" : (data.returncode === undefined ? "No process running" : `Exit ${data.returncode}`);
      $("budget-label").textContent = `${used.toLocaleString()} / ${budget.toLocaleString()} transitions`; $("remaining-label").textContent = `${remaining.toLocaleString()} remaining`; $("progress-fill").style.width = `${Math.min(100, used / Math.max(1, budget) * 100)}%`;
      $("metric-update").textContent = metrics.update ?? "—"; $("metric-return").textContent = format(metrics.return); $("metric-lap").textContent = format(metrics.lap); $("metric-speed").textContent = format(metrics.speed, " m/s");
      $("metric-crash").textContent = format(metrics.crash); $("metric-offtrack").textContent = format(metrics.off_track);
      $("start").disabled = Boolean(data.running); $("stop").disabled = !data.running;
      $("log").textContent = (data.lines || []).join("\n") || "Waiting for a local trainer process…"; $("log").scrollTop = $("log").scrollHeight;
    }
    async function request(path, options) { const response = await fetch(path, options); const body = await response.json(); if (!response.ok) throw new Error(body.error || "Request failed"); return body; }
    async function refresh() { try { render(await request("/api/status")); } catch (error) { $("state-label").textContent = error.message; } }
    $("start").addEventListener("click", async () => { try { render(await request("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formData()) })); } catch (error) { alert(error.message); } });
    $("stop").addEventListener("click", async () => { try { render(await request("/api/stop", { method: "POST" })); } catch (error) { alert(error.message); } });
    fields.forEach(id => $(id).addEventListener("change", refresh));
    refresh(); setInterval(refresh, 700);
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
        elif path == "/api/report":
            report = ROOT / "participant_run" / "training_report.md"
            if not report.exists():
                self._json(404, {"error": "no participant report exists yet"})
            else:
                self._send(200, "text/markdown; charset=utf-8", report.read_bytes())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/stop":
            self._json(200, PROCESS.stop())
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
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
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
    print(f"participant UI: http://{args.host}:{args.port}/")
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
