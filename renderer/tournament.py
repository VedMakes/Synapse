"""Synapse F1 Grand Prix Tournament Replay Engine and Bracket Broadcaster."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import webbrowser

import numpy as np

try:
    import onnxruntime as ort
    ort.__version__ = "1.20.1"
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from racing_event.config import SimulationConfig
from racing_event.track import TrackGeometry
from racing_event.trajectory import replay_header
from racing_event.vector_env import VectorizedRacingEnv

REPLAY_DIR = Path(__file__).resolve().parent / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def resolve_circuit(track_arg: str) -> Path:
    p = Path(track_arg)
    if p.exists():
        return p.resolve()
    candidate = ROOT_DIR / "tracks" / "train" / f"{track_arg}.json"
    if candidate.exists():
        return candidate.resolve()
    candidate_raw = ROOT_DIR / track_arg
    if candidate_raw.exists():
        return candidate_raw.resolve()
    raise FileNotFoundError(f"Circuit file not found: {track_arg}")


def find_model_file(entrant_path: Path) -> Path:
    if entrant_path.is_file() and entrant_path.suffix.lower() == ".onnx":
        return entrant_path.resolve()
    if entrant_path.is_dir():
        direct = entrant_path / "model.onnx"
        if direct.is_file():
            return direct.resolve()
        sub = entrant_path / "submission" / "model.onnx"
        if sub.is_file():
            return sub.resolve()
        found = list(entrant_path.glob("*.onnx"))
        if found:
            return found[0].resolve()
        found_nested = list(entrant_path.glob("**/*.onnx"))
        if found_nested:
            return found_nested[0].resolve()
    raise FileNotFoundError(f"Could not locate model.onnx in: {entrant_path}")


def simulate_single_entrant(
    model_path: Path,
    entrant_id: str,
    track: TrackGeometry,
    config: SimulationConfig,
    max_steps: int = 25000,
) -> dict:
    session = ort.InferenceSession(str(model_path))

    effective_config = config
    if max_steps != config.episode.max_steps:
        effective_config = dataclasses.replace(
            config,
            episode=dataclasses.replace(config.episode, max_steps=max_steps),
        )

    env = VectorizedRacingEnv(track, effective_config, reward=None, n_envs=1)
    obs, _ = env.reset(evaluation=True)

    header = replay_header(track, effective_config, participant_id=entrant_id, mode="evaluation")
    frames = []

    status = "running"
    finish_step = max_steps
    info = {}

    for step in range(max_steps):
        action = session.run(None, {"observations": obs[0:1]})[0]
        st = env.state_to_dict()

        steer_cmd = float(action[0, 0])
        thr_cmd = float(action[0, 1])

        frames.append({
            "tick": step,
            "x": float(st["x"][0]),
            "y": float(st["y"][0]),
            "heading": float(st["heading"][0]),
            "speed": float(st["speed"][0]),
            "lateral_velocity": float(st["lateral_velocity"][0]),
            "yaw_rate": float(st["yaw_rate"][0]),
            "steer": float(st["steer"][0]),
            "steering_command": steer_cmd,
            "throttle_brake": thr_cmd,
            "off_track": bool(info["off_track"][0]) if "off_track" in info else False,
            "events": list(info["events"][0]) if "events" in info and info["events"] else [],
            "terminated": False,
            "truncated": False,
        })

        obs, rewards, term, trunc, info = env.step(action)

        if bool(info["crash"][0]):
            status = "dnf_crash"
            finish_step = step + 1
            break
        if bool(info["lap_complete"][0]):
            status = "finished"
            finish_step = step + 1
            break
        if bool(info.get("stuck", [False])[0]) or env.stuck_ticks[0] >= effective_config.episode.stuck_steps:
            status = "dnf_stuck"
            finish_step = step + 1
            break
        if term[0]:
            status = "dnf_reverse"
            finish_step = step + 1
            break
        if trunc[0]:
            status = "dnf_timeout"
            finish_step = step + 1
            break

    if status == "running":
        status = "dnf_timeout"
        finish_step = len(frames)

    if frames:
        frames[-1]["terminated"] = status in ("finished", "dnf_crash", "dnf_reverse")
        frames[-1]["truncated"] = status in ("dnf_timeout", "dnf_stuck")

    elapsed_s = finish_step / 30.0
    distance_covered = float(env.distance_since_spawn[0])
    mean_speed_mps = distance_covered / max(0.1, elapsed_s)
    speed_kmh = mean_speed_mps * 3.6

    result_payload = {
        "status": status,
        "steps": finish_step,
        "elapsed_s": elapsed_s,
        "distance_m": distance_covered,
        "mean_speed_mps": mean_speed_mps,
        "lap_time_s": elapsed_s if status == "finished" else None,
    }

    payload = dict(header)
    payload["frames"] = frames
    payload["result"] = result_payload
    payload["summary"] = {
        "entrant_id": entrant_id,
        "status": status,
        "lap_time_s": elapsed_s if status == "finished" else None,
        "elapsed_s": elapsed_s,
        "distance_m": distance_covered,
        "avg_speed_kmh": speed_kmh,
        "steps": finish_step,
    }
    return payload


def cmd_simulate(args: argparse.Namespace) -> None:
    track_path = resolve_circuit(args.track)
    config_path = ROOT_DIR / args.config if not Path(args.config).is_absolute() else Path(args.config)

    track = TrackGeometry.from_file(track_path)
    config = SimulationConfig.from_file(config_path)
    circuit_name = track_path.stem

    print("====================================================================", flush=True)
    print("  SYNAPSE TOURNAMENT REPLAY GENERATOR", flush=True)
    print("====================================================================", flush=True)
    print(f"  Circuit:        {circuit_name} ({track.length:.1f} meters)")
    print(f"  Physics Rate:   30 Hz authoritative simulation")
    print(f"  Step Ceiling:   {args.max_steps} ticks (runs until lap finish or incident)")
    print("====================================================================\n", flush=True)

    targets: list[tuple[str, Path]] = []

    if args.entrant:
        e_path = Path(args.entrant)
        if not e_path.is_absolute():
            e_path = ROOT_DIR / e_path
        m_path = find_model_file(e_path)
        label = args.label or (e_path.name if e_path.is_dir() else e_path.stem)
        targets.append((label, m_path))

    if args.all:
        root_entrants = Path(args.all)
        if not root_entrants.is_absolute():
            root_entrants = ROOT_DIR / root_entrants
        if not root_entrants.exists():
            print(f"Error: entrants folder not found: {root_entrants}")
            return
        for item in sorted(root_entrants.iterdir()):
            if item.is_dir():
                try:
                    m = find_model_file(item)
                    targets.append((item.name, m))
                except FileNotFoundError:
                    pass
            elif item.is_file() and item.suffix.lower() == ".onnx":
                targets.append((item.stem, item.resolve()))

    if not targets:
        print("No entrant packages found. Pass --entrant <path> or --all <folder>")
        return

    results = []

    for index, (entrant_id, model_path) in enumerate(targets, 1):
        print(f"[{index}/{len(targets)}] Evaluating {entrant_id}...", flush=True)
        replay_payload = simulate_single_entrant(
            model_path=model_path,
            entrant_id=entrant_id,
            track=track,
            config=config,
            max_steps=args.max_steps,
        )

        out_name = f"{entrant_id}_{circuit_name}.json"
        out_path = REPLAY_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(replay_payload, f)

        res = replay_payload["summary"]
        results.append(res)

        lap_display = f"{res['lap_time_s']:.2f}s" if res["lap_time_s"] else "NO TIME"
        print(f"    Status:   {res['status'].upper()}")
        print(f"    Lap Time: {lap_display} ({res['elapsed_s']:.2f}s elapsed)")
        print(f"    Distance: {res['distance_m']:.1f}m / {track.length:.1f}m")
        print(f"    AvgSpeed: {res['avg_speed_kmh']:.1f} km/h")
        print(f"    Saved:    renderer/replays/{out_name}\n", flush=True)

    results.sort(key=lambda x: (0 if x["status"] == "finished" else 1, x["lap_time_s"] or 999999, -x["distance_m"]))

    board_file = REPLAY_DIR / f"leaderboard_{circuit_name}.json"
    with open(board_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_leaderboard(results, circuit_name)


def print_leaderboard(results: list[dict], circuit: str) -> None:
    print("====================================================================", flush=True)
    print(f"  SYNAPSE GP OFFICIAL LEADERBOARD: {circuit.upper()}", flush=True)
    print("====================================================================", flush=True)
    print(f"{'POS':<4} {'ENTRANT':<24} {'STATUS':<14} {'LAP TIME':<12} {'DISTANCE':<12} {'AVG SPEED'}", flush=True)
    print("--------------------------------------------------------------------", flush=True)
    for pos, r in enumerate(results, 1):
        lap_str = f"{r['lap_time_s']:.2f}s" if r["lap_time_s"] else "-"
        dist_str = f"{r['distance_m']:.1f}m"
        spd_str = f"{r['avg_speed_kmh']:.1f} km/h"
        print(f"{pos:<4} {r['entrant_id']:<24} {r['status']:<14} {lap_str:<12} {dist_str:<12} {spd_str}", flush=True)
    print("====================================================================\n", flush=True)


def cmd_match(args: argparse.Namespace) -> None:
    circuit_name = args.track
    team_a = args.team_a
    team_b = args.team_b

    def resolve_replay(team_query: str) -> Path:
        direct = Path(team_query)
        if direct.is_file():
            return direct.resolve()
        cand1 = REPLAY_DIR / f"{team_query}_{circuit_name}.json"
        if cand1.is_file():
            return cand1.resolve()
        cand2 = REPLAY_DIR / f"{team_query}.json"
        if cand2.is_file():
            return cand2.resolve()
        matches = list(REPLAY_DIR.glob(f"*{team_query}*.json"))
        if matches:
            return matches[0].resolve()
        raise FileNotFoundError(f"Could not find replay for '{team_query}'. Run simulate first!")

    replay_a_path = resolve_replay(team_a)
    replay_b_path = resolve_replay(team_b)

    with open(replay_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(replay_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    target_a = Path(__file__).resolve().parent / "match_a.json"
    target_b = Path(__file__).resolve().parent / "match_b.json"

    shutil.copyfile(replay_a_path, target_a)
    shutil.copyfile(replay_b_path, target_b)

    name_a = data_a.get("source", {}).get("participant_id", team_a).upper()
    name_b = data_b.get("source", {}).get("participant_id", team_b).upper()

    res_a = data_a.get("result", {})
    res_b = data_b.get("result", {})

    status_a = res_a.get("status", "unknown")
    status_b = res_b.get("status", "unknown")

    time_a = res_a.get("lap_time_s") or res_a.get("elapsed_s", 999.0)
    time_b = res_b.get("lap_time_s") or res_b.get("elapsed_s", 999.0)

    dist_a = res_a.get("distance_m", 0.0)
    dist_b = res_b.get("distance_m", 0.0)

    fin_a = status_a == "finished"
    fin_b = status_b == "finished"

    if fin_a and not fin_b:
        winner = name_a
        interval = f"{name_a} Won (Competitor DNF)"
    elif fin_b and not fin_a:
        winner = name_b
        interval = f"{name_b} Won (Competitor DNF)"
    elif fin_a and fin_b:
        gap = abs(time_a - time_b)
        winner = name_a if time_a <= time_b else name_b
        interval = f"+{gap:.3f}s"
    else:
        winner = name_a if dist_a >= dist_b else name_b
        interval = f"+{abs(dist_a - dist_b):.1f}m distance margin"

    print("====================================================================", flush=True)
    print("  SYNAPSE KNOCKOUT MATCH STAGED SUCCESSFULLY", flush=True)
    print("====================================================================", flush=True)
    print(f"  Circuit:       {circuit_name.upper()}")
    print(f"  Entrant 1:     {name_a} ({status_a.upper()}, {time_a:.2f}s)")
    print(f"  Entrant 2:     {name_b} ({status_b.upper()}, {time_b:.2f}s)")
    print("--------------------------------------------------------------------", flush=True)
    print(f"  PROVISIONAL WINNER: {winner}")
    print(f"  INTERVAL MARGIN:    {interval}")
    print("====================================================================", flush=True)
    print("\n  Staged replay A: renderer/match_a.json")
    print("  Staged replay B: renderer/match_b.json")
    print("  Broadcast URL:   http://127.0.0.1:8765/?match=true\n", flush=True)

    if args.open_browser:
        webbrowser.open("http://127.0.0.1:8765/?match=true")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    board_file = REPLAY_DIR / f"leaderboard_{args.track}.json"
    if not board_file.exists():
        replays = list(REPLAY_DIR.glob(f"*_{args.track}.json"))
        if not replays:
            print(f"No replays recorded yet for circuit: {args.track}")
            return
        summaries = []
        for r in replays:
            try:
                with open(r, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if "summary" in d:
                        summaries.append(d["summary"])
            except Exception:
                pass
        summaries.sort(key=lambda x: (0 if x["status"] == "finished" else 1, x["lap_time_s"] or 999999, -x["distance_m"]))
        print_leaderboard(summaries, args.track)
    else:
        with open(board_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print_leaderboard(data, args.track)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synapse Tournament Replay Pre-Generator and Knockout Stager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_p = subparsers.add_parser("simulate", help="Simulate models to generate full race replays")
    sim_p.add_argument("--entrant", help="Path to single participant submission folder or model.onnx")
    sim_p.add_argument("--label", help="Optional custom display label for entrant")
    sim_p.add_argument("--all", help="Path to directory containing multiple participant submission folders")
    sim_p.add_argument("--track", default="monaco", help="Circuit name (monaco, spa) or path to track JSON")
    sim_p.add_argument("--config", default="config/simulation.json", help="Simulation configuration file")
    sim_p.add_argument("--max-steps", type=int, default=25000, help="Generous step ceiling allowing full completion")

    match_p = subparsers.add_parser("match", help="Pair any 2 teams together for an instant knockout match")
    match_p.add_argument("team_a", help="Name or replay path of Entrant A")
    match_p.add_argument("team_b", help="Name or replay path of Entrant B")
    match_p.add_argument("--track", default="monaco", help="Circuit name to match on")
    match_p.add_argument("--open-browser", action="store_true", help="Automatically open browser")

    board_p = subparsers.add_parser("leaderboard", help="View current tournament leaderboard")
    board_p.add_argument("--track", default="monaco", help="Circuit name to show leaderboard for")

    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "match":
        cmd_match(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)


if __name__ == "__main__":
    main()

