"""Fixed-rate live ONNX broadcast launcher for Synapse 2.5D renderer."""

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import webbrowser

# Windows high resolution timer precision
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

# Patch onnxruntime version string on Windows environments where host has newer runtime
try:
    import onnxruntime
    onnxruntime.__version__ = "1.20.1"
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from racing_event.config import SimulationConfig
from racing_event.live_protocol import LiveStatePublisher
from racing_event.live_race import LiveRaceOptions, run_live_onnx_race
from racing_event.race import RaceEntrant
from racing_event.track import TrackGeometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch live ONNX head-to-head race broadcast")
    parser.add_argument(
        "--track",
        default="tracks/train/monaco.json",
        help="Circuit JSON path (default: tracks/train/monaco.json or tracks/train/spa.json)",
    )
    parser.add_argument(
        "--alpha",
        default="artifacts/onnx-packages/team_alpha/submission",
        help="Path to team alpha submission folder",
    )
    parser.add_argument(
        "--beta",
        default="artifacts/onnx-packages/team_beta/submission",
        help="Path to team beta submission folder",
    )
    parser.add_argument(
        "--config",
        default="config/simulation.json",
        help="Simulation configuration file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8767,
        help="WebSocket telemetry broadcast port (default: 8767)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="WebSocket host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open renderer browser window automatically on start",
    )
    parser.add_argument(
        "--policy-budget-ms",
        type=float,
        default=250.0,
        help="Inference deadline budget in milliseconds (default: 250.0)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=60,
        help="Keep WebSocket server open after race completion to view final podium",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step limit (e.g. 300 for a 10s sprint, or leave blank for full race)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_path = Path(args.track)
    if not track_path.is_absolute():
        track_path = ROOT_DIR / track_path
    if not track_path.exists():
        if (ROOT_DIR / "tracks" / "train" / f"{args.track}.json").exists():
            track_path = ROOT_DIR / "tracks" / "train" / f"{args.track}.json"
        else:
            raise FileNotFoundError(f"Track file not found: {args.track}")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    alpha_path = Path(args.alpha)
    if not alpha_path.is_absolute():
        alpha_path = ROOT_DIR / alpha_path

    beta_path = Path(args.beta)
    if not beta_path.is_absolute():
        beta_path = ROOT_DIR / beta_path

    # Unique output directory using timestamp to prevent directory collision errors
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    track_stem = track_path.stem
    output_dir = ROOT_DIR / "artifacts" / "live-results" / f"live_{track_stem}_{timestamp}"

    print("====================================================================")
    print("  SYNAPSE GP 2.5D LIVE ONNX BROADCAST")
    print("====================================================================")
    print(f"  Circuit:       {track_path.name}")
    print(f"  Team Alpha:    {alpha_path}")
    print(f"  Team Beta:     {beta_path}")
    print(f"  Output Record: {output_dir}")
    print("====================================================================")

    entrants = (
        RaceEntrant("team_alpha", alpha_path, "TEAM ALPHA"),
        RaceEntrant("team_beta", beta_path, "TEAM BETA"),
    )
    config = SimulationConfig.from_file(config_path)
    track = TrackGeometry.from_file(track_path)
    publisher = LiveStatePublisher(host=args.host, port=args.port)

    options = LiveRaceOptions(
        control_hz=30,
        policy_budget_ms=args.policy_budget_ms,
        late_policy="trusted_live",
        pace=True,
        max_steps=args.max_steps,
    )

    ws_url = f"ws://{args.host}:{args.port}/live/v1"
    renderer_url = f"http://127.0.0.1:8765/?live={ws_url}"

    def announce(endpoint: str) -> None:
        print(f"\n[LIVE TELECAST READY]", flush=True)
        print(f"  WebSocket Endpoint: {endpoint}", flush=True)
        print(f"  Renderer 3D URL:    {renderer_url}", flush=True)
        print("  Press Ctrl+C to abort at any time.\n", flush=True)
        if args.open_browser:
            webbrowser.open(renderer_url)
        print("  Awaiting 3D viewer connection before dropping green flag...", flush=True)
        for _ in range(80):
            if publisher.connected_clients > 0:
                print("  3D viewer connected! Green flag sequence in 1.5s...", flush=True)
                time.sleep(1.5)
                return
            time.sleep(0.1)
        print("  Starting race clock...", flush=True)

    try:
        report = run_live_onnx_race(
            entrants,
            track,
            config,
            publisher=publisher,
            options=options,
            output_dir=output_dir,
            on_started=announce,
        )

        print("\n====================================================================")
        print("  RACE COMPLETED OFFICIAL PROVISIONAL RESULTS")
        print("====================================================================")
        report_dict = report.to_dict()
        for car in report_dict.get("results", []):
            status = car.get("status")
            steps = car.get("steps")
            spd = car.get("mean_speed_mps", 0) * 3.6
            print(f"  {car.get('label')}: Status={status}, Steps={steps}, AvgSpeed={spd:.1f} km/h")
        print(f"  Recorded Heat Log: {output_dir / 'live-race.json'}")

        if args.cooldown_seconds > 0:
            print(f"\n  Keeping WebSocket telecast active for {args.cooldown_seconds}s for podium celebration...")
            time.sleep(args.cooldown_seconds)

    except KeyboardInterrupt:
        print("\n  Broadcast stopped by user.")
    finally:
        publisher.close(grace_seconds=1.0)
        print("  Live telecast closed.\n")


if __name__ == "__main__":
    main()
