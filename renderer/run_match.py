"""Synapse F1 Grand Prix Knockout Match Evaluator and 3D Broadcaster."""

import argparse
import dataclasses
import datetime
import json
from pathlib import Path
import sys
import webbrowser
import numpy as np

try:
    import onnxruntime
    onnxruntime.__version__ = "1.20.1"
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from racing_event.config import SimulationConfig
from racing_event.track import TrackGeometry
from racing_event.trajectory import replay_header
from racing_event.vector_env import VectorizedRacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate knockout head to head race and broadcast 3D replay")
    parser.add_argument(
        "--track",
        default="monaco",
        help="Circuit name (monaco, spa) or path to track JSON",
    )
    parser.add_argument(
        "--alpha",
        default="artifacts/onnx-packages/team_alpha/submission",
        help="Path to team alpha ONNX submission folder",
    )
    parser.add_argument(
        "--beta",
        default="artifacts/onnx-packages/team_beta/submission",
        help="Path to team beta ONNX submission folder",
    )
    parser.add_argument(
        "--label-a",
        default="TEAM ALPHA",
        help="Display label for entrant A",
    )
    parser.add_argument(
        "--label-b",
        default="TEAM BETA",
        help="Display label for entrant B",
    )
    parser.add_argument(
        "--config",
        default="config/simulation.json",
        help="Simulation configuration file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6500,
        help="Maximum simulation ticks (default: 6500 for full lap)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open renderer browser window automatically on completion",
    )
    return parser.parse_args()


def resolve_track(track_arg: str) -> Path:
    p = Path(track_arg)
    if p.exists():
        return p.resolve()
    candidate = ROOT_DIR / "tracks" / "train" / f"{track_arg}.json"
    if candidate.exists():
        return candidate.resolve()
    candidate_raw = ROOT_DIR / track_arg
    if candidate_raw.exists():
        return candidate_raw.resolve()
    raise FileNotFoundError(f"Track geometry not found: {track_arg}")


def build_replay_payload(header: dict, frames: list, result: dict) -> dict:
    payload = dict(header)
    payload["frames"] = frames
    payload["result"] = result
    return payload


def main() -> None:
    args = parse_args()

    track_path = resolve_track(args.track)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    alpha_dir = Path(args.alpha)
    if not alpha_dir.is_absolute():
        alpha_dir = ROOT_DIR / alpha_dir
    model_a_path = alpha_dir / "model.onnx" if alpha_dir.is_dir() else alpha_dir

    beta_dir = Path(args.beta)
    if not beta_dir.is_absolute():
        beta_dir = ROOT_DIR / beta_dir
    model_b_path = beta_dir / "model.onnx" if beta_dir.is_dir() else beta_dir

    print("====================================================================", flush=True)
    print("  SYNAPSE GP KNOCKOUT MATCH EVALUATOR", flush=True)
    print("====================================================================", flush=True)
    print(f"  Circuit:    {track_path.name}", flush=True)
    print(f"  Entrant 1:  {args.label_a} ({model_a_path.parent.name})", flush=True)
    print(f"  Entrant 2:  {args.label_b} ({model_b_path.parent.name})", flush=True)
    print("  Simulating authoritative physics...", flush=True)

    track = TrackGeometry.from_file(track_path)
    config = SimulationConfig.from_file(config_path)
    if args.max_steps != config.episode.max_steps:
        config = dataclasses.replace(
            config,
            episode=dataclasses.replace(config.episode, max_steps=args.max_steps),
        )

    import onnxruntime as ort
    sess_a = ort.InferenceSession(str(model_a_path))
    sess_b = ort.InferenceSession(str(model_b_path))

    env = VectorizedRacingEnv(track, config, reward=None, n_envs=2)
    obs, _ = env.reset(evaluation=True)

    header_a = replay_header(track, config, participant_id=args.label_a, mode="evaluation")
    header_b = replay_header(track, config, participant_id=args.label_b, mode="evaluation")

    frames_a = []
    frames_b = []
    active = [True, True]
    finish_step = [None, None]
    status_result = ["running", "running"]
    info = {}

    for step in range(args.max_steps):
        act_a = sess_a.run(None, {"observations": obs[0:1]})[0] if active[0] else np.zeros((1, 2), dtype=np.float32)
        act_b = sess_b.run(None, {"observations": obs[1:2]})[0] if active[1] else np.zeros((1, 2), dtype=np.float32)
        actions = np.vstack([act_a, act_b])

        st = env.state_to_dict()

        if active[0]:
            frames_a.append({
                "tick": step,
                "x": float(st["x"][0]),
                "y": float(st["y"][0]),
                "heading": float(st["heading"][0]),
                "speed": float(st["speed"][0]),
                "lateral_velocity": float(st["lateral_velocity"][0]),
                "yaw_rate": float(st["yaw_rate"][0]),
                "steer": float(st["steer"][0]),
                "steering_command": float(act_a[0, 0]),
                "throttle_brake": float(act_a[0, 1]),
                "off_track": bool(info["off_track"][0]) if "off_track" in info else False,
                "events": [],
                "terminated": False,
                "truncated": False,
            })
        if active[1]:
            frames_b.append({
                "tick": step,
                "x": float(st["x"][1]),
                "y": float(st["y"][1]),
                "heading": float(st["heading"][1]),
                "speed": float(st["speed"][1]),
                "lateral_velocity": float(st["lateral_velocity"][1]),
                "yaw_rate": float(st["yaw_rate"][1]),
                "steer": float(st["steer"][1]),
                "steering_command": float(act_b[0, 0]),
                "throttle_brake": float(act_b[0, 1]),
                "off_track": bool(info["off_track"][1]) if "off_track" in info else False,
                "events": [],
                "terminated": False,
                "truncated": False,
            })

        obs, rewards, term, trunc, info = env.step(actions, active_mask=np.array(active, dtype=bool))

        for idx in range(2):
            if active[idx]:
                if bool(info["crash"][idx]):
                    active[idx] = False
                    status_result[idx] = "dnf_crash"
                    finish_step[idx] = step
                elif bool(info["lap_complete"][idx]):
                    active[idx] = False
                    status_result[idx] = "finished"
                    finish_step[idx] = step
                elif term[idx] or trunc[idx]:
                    active[idx] = False
                    status_result[idx] = "dnf_timeout"
                    finish_step[idx] = step

        if not any(active):
            break

    for idx in range(2):
        if finish_step[idx] is None:
            finish_step[idx] = len(frames_a if idx == 0 else frames_b)
            if status_result[idx] == "running":
                status_result[idx] = "dnf_timeout"

    time_a = finish_step[0] / 30.0
    time_b = finish_step[1] / 30.0

    dist_a = env.distance_since_spawn[0]
    dist_b = env.distance_since_spawn[1]

    speed_a = (dist_a / max(0.1, time_a)) * 3.6
    speed_b = (dist_b / max(0.1, time_b)) * 3.6

    fin_a = status_result[0] == "finished"
    fin_b = status_result[1] == "finished"

    if fin_a and not fin_b:
        winner = args.label_a
        margin_text = f"{args.label_a} Won (Competitor DNF)"
    elif fin_b and not fin_a:
        winner = args.label_b
        margin_text = f"{args.label_b} Won (Competitor DNF)"
    elif fin_a and fin_b:
        gap = abs(time_a - time_b)
        winner = args.label_a if time_a <= time_b else args.label_b
        margin_text = f"+{gap:.3f}s"
    else:
        winner = args.label_a if dist_a >= dist_b else args.label_b
        margin_text = f"+{abs(dist_a - dist_b):.1f}m distance margin"

    print("\n====================================================================", flush=True)
    print("  FIA FORMULA 1 SYNAPSE GP PROVISIONAL CLASSIFICATION", flush=True)
    print("====================================================================", flush=True)
    print(f"  MATCH WINNER:  {winner}", flush=True)
    print(f"  INTERVAL:      {margin_text}", flush=True)
    print(f"  P1 {args.label_a}: Status={status_result[0]}, Steps={finish_step[0]}, Time={time_a:.2f}s, AvgSpd={speed_a:.1f} km/h", flush=True)
    print(f"  P2 {args.label_b}: Status={status_result[1]}, Steps={finish_step[1]}, Time={time_b:.2f}s, AvgSpd={speed_b:.1f} km/h", flush=True)
    print("====================================================================", flush=True)

    result_payload_a = {
        "status": status_result[0],
        "steps": finish_step[0],
        "elapsed_s": time_a,
        "mean_speed_mps": speed_a / 3.6,
        "lap_time_s": time_a if fin_a else None,
    }
    result_payload_b = {
        "status": status_result[1],
        "steps": finish_step[1],
        "elapsed_s": time_b,
        "mean_speed_mps": speed_b / 3.6,
        "lap_time_s": time_b if fin_b else None,
    }

    replay_obj_a = build_replay_payload(header_a, frames_a, result_payload_a)
    replay_obj_b = build_replay_payload(header_b, frames_b, result_payload_b)

    renderer_dir = ROOT_DIR / "renderer"
    match_a_path = renderer_dir / "match_a.json"
    match_b_path = renderer_dir / "match_b.json"

    match_a_path.write_text(json.dumps(replay_obj_a), encoding="utf-8")
    match_b_path.write_text(json.dumps(replay_obj_b), encoding="utf-8")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = ROOT_DIR / "artifacts" / "match-results" / f"match_{track_path.stem}_{timestamp}"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "match_a.json").write_text(json.dumps(replay_obj_a), encoding="utf-8")
    (record_dir / "match_b.json").write_text(json.dumps(replay_obj_b), encoding="utf-8")

    match_url = "http://127.0.0.1:8765/?match=true"
    print(f"\n  3D Replay Broadcast Ready: {match_url}", flush=True)
    print(f"  Match records saved to:     {record_dir}\n", flush=True)

    if args.open_browser:
        webbrowser.open(match_url)


if __name__ == "__main__":
    main()

